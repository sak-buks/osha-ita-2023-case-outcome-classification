"""Lanza la corrida anidada completa (o un subconjunto) con reanudación.

Cada tarea es (modelo, balanceo, optimizador, pliegue externo). Las tareas ya terminadas
(JSON presente) se saltan, así que basta con volver a lanzar el mismo comando tras una interrupción.

Ejemplos
--------
python run_nested.py --workers 7
python run_nested.py --models logreg tree --folds 1 --n-evals 5
python run_nested.py --references
"""
import argparse
import csv
import ctypes
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
from joblib import Parallel, delayed

from osha3.models import REGISTRY, REFERENCES
from osha3.nested import LOCK_DIR, RUN_DIR, run_combination
from osha3.optimizers import OPTIMIZERS
from osha3.pipeline import BALANCES

LOG = Path(__file__).resolve().parent / 'results' / 'run_log.csv'
MEM_LOG = Path(__file__).resolve().parent / 'results' / 'memory_log.csv'
# Costo medido en el pliegue 3 (timing.py): s/evaluación sin balanceo, con SMOTE, y reentrenamiento.
COST = {'knn': (32, 51, 160), 'naive_bayes': (1, 1, 2), 'logreg': (18, 14, 64), 'tree': (2, 2, 4),
        'random_forest': (22, 33, 64), 'xgboost': (17, 47, 25), 'svm': (81, 117, 271)}
FOLD_SIZE = {1: 0.4, 2: 0.7, 3: 1.0}


def estimated_cost(task, n_evals):
    model, balance, _, fold, _ = task
    en, es, refit = COST.get(model, (1, 1, 1))
    per_eval = es if balance in ('smote', 'adasyn') else en
    return FOLD_SIZE[fold] * (n_evals * per_eval + refit)


def keep_awake():
    """Evita la suspensión de Windows mientras este proceso viva (se revierte sola al salir)."""
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def memory_logger(stop, every=60):
    new = not MEM_LOG.exists()
    with MEM_LOG.open('a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['time', 'ram_used_gb', 'ram_percent', 'cpu_percent'])
        while not stop.wait(every):
            vm = psutil.virtual_memory()
            w.writerow([datetime.now().isoformat(timespec='seconds'), round(vm.used / 2**30, 2), vm.percent,
                        psutil.cpu_percent()])
            fh.flush()


def build_tasks(models, balances, optimizers, folds, references):
    if references:
        return [('dummy', 'none', 'grid', f, 1) for f in folds] + \
               [('logreg_e1', 'class_weight', 'grid', f, 1) for f in folds]
    tasks = []
    for model in models:
        for balance in balances:
            if balance == 'class_weight' and REGISTRY[model].weight_mode is None:
                continue   # KNN × class_weight: no aplica
            for opt in optimizers:
                for fold in folds:
                    tasks.append((model, balance, opt, fold, None))
    return tasks


def _task(model, balance, opt, fold, n_evals, device, run_dir):
    start = time.perf_counter()
    status = run_combination(model, balance, opt, fold, n_evals=n_evals, n_jobs=1,
                             device=device if model == 'xgboost' else 'cpu', run_dir=run_dir)
    return model, balance, opt, fold, status, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(REGISTRY))
    ap.add_argument('--balances', nargs='+', default=list(BALANCES))
    ap.add_argument('--optimizers', nargs='+', default=list(OPTIMIZERS))
    ap.add_argument('--folds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--n-evals', type=int, default=20)   # opción A del presupuesto (D5)
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--xgb-device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--references', action='store_true', help='Solo Dummy y la logística del entregable 1')
    ap.add_argument('--run-dir', default=str(RUN_DIR))
    args = ap.parse_args()

    tasks = build_tasks(args.models, args.balances, args.optimizers, args.folds, args.references)
    # Las más largas primero (programación LPT): evita que una tarea larga quede sola al final.
    tasks = sorted(tasks, key=lambda t: -estimated_cost(t, args.n_evals))
    shutil.rmtree(LOCK_DIR, ignore_errors=True)   # bloqueos de una corrida interrumpida
    keep_awake()
    stop = threading.Event()
    threading.Thread(target=memory_logger, args=(stop,), daemon=True).start()
    print(f'{len(tasks)} tareas | workers={args.workers} | n_evals={args.n_evals} | xgb={args.xgb_device}', flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open('a', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(['finished_at', 'model', 'balance', 'optimizer', 'fold', 'status', 'seconds'])
        jobs = (delayed(_task)(m, b, o, f, n if n is not None else args.n_evals, args.xgb_device, args.run_dir)
                for m, b, o, f, n in tasks)
        done = 0
        for m, b, o, f, status, sec in Parallel(n_jobs=args.workers, return_as='generator_unordered')(jobs):
            done += 1
            writer.writerow([datetime.now().isoformat(timespec='seconds'), m, b, o, f, status, round(sec, 1)])
            fh.flush()
            print(f'[{done}/{len(tasks)}] {datetime.now():%H:%M} {m} {b} {o} fold{f}: {status} ({sec:.0f}s)', flush=True)
    stop.set()


if __name__ == '__main__':
    main()
