"""Validación anidada (D2): búsqueda en el bucle interno, medición en el bucle externo.

Para cada (modelo, balanceo, optimizador, pliegue externo):
1. Bucle interno: el optimizador evalúa configuraciones con StratifiedGroupKFold(3) por
   establecimiento sobre la muestra de búsqueda (25 % del entrenamiento externo, todas las muertes).
2. Se reentrena el pipeline con la mejor configuración sobre el 100 % del entrenamiento externo.
3. Se mide una sola vez en la validación externa y se guardan métricas y probabilidades.
El resultado se escribe al terminar; si ya existe, la combinación se salta (reanudación).
"""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import json
import os
import random
import time
import traceback
import warnings

import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from .data import LABELS, RAW_FEATURES, SEED, INNER_SEED, SEARCH_FRACTION, read_fold, inner_splits
from .metrics import classification_metrics
from .models import ALL_SPECS
from .optimizers import optimize
from .pipeline import build_pipeline

RUN_DIR = Path(__file__).resolve().parents[1] / 'results' / 'runs'
LOCK_DIR = Path(__file__).resolve().parents[1] / 'results' / 'locks'
TRAIN_METRIC_ROWS = 10000   # filas del entrenamiento externo usadas para medir la brecha entrenamiento-validación
BIG_SLOTS = 1               # reentrenamientos de mucha memoria que pueden correr a la vez


def needs_big_memory(model, params, n_rows):
    """Reentrenamientos que pueden pasar de ~4 GB (medido: SVM Nystroem m=1000 en el pliegue 3)."""
    if model == 'svm' and params.get('kernel') == 'nystroem':
        return params['n_components'] * n_rows >= 50e6
    if model == 'random_forest':
        return params['n_estimators'] >= 300 and n_rows >= 90000
    return False


@contextmanager
def memory_slot(needed):
    """Semáforo entre procesos con archivos de bloqueo: limita los reentrenamientos grandes simultáneos."""
    path = None
    if needed:
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        while path is None:
            for i in range(BIG_SLOTS):
                candidate = LOCK_DIR / f'slot{i}.lock'
                try:
                    fd = os.open(str(candidate), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                path = candidate
                break
            else:
                time.sleep(15)
    try:
        yield
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _labels(frame):
    return frame.incident_outcome.astype(int).to_numpy()


def _fit(pipe, X, y, fit_params, caught):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        pipe.fit(X, y, **fit_params)
    caught.extend(f'{x.category.__name__}: {str(x.message)[:160]}' for x in w)
    return pipe


def make_objective(spec, balance, sample, splits, n_jobs, device):
    """Función objetivo común a los cuatro optimizadores (pipeline completo por pliegue interno)."""
    X, y = sample[RAW_FEATURES], _labels(sample)

    def objective(params):
        scores, caught, notes = [], [], []
        for a, b in splits:
            pipe, fit_params, note = build_pipeline(spec, params, balance, y[a], n_jobs, device)
            _fit(pipe, X.iloc[a], y[a], fit_params, caught)
            scores.append(float(balanced_accuracy_score(y[b], pipe.predict(X.iloc[b]))))
            notes.append(note)
        return float(np.mean(scores)), {'inner_scores': scores, 'n_warnings': len(caught),
                                        'warnings': sorted(set(caught))[:5], 'balance_notes': notes}
    return objective


def combo_dir(model, balance, optimizer, run_dir=RUN_DIR):
    return Path(run_dir) / f'{model}__{balance}__{optimizer}'


def run_combination(model, balance, optimizer, fold, n_evals=30, n_jobs=1, device='cpu', run_dir=RUN_DIR):
    """Ejecuta una combinación en un pliegue externo y guarda JSON + predicciones.

    Returns
    -------
    str
        'skip' si ya existía, 'ok' o 'error'.
    """
    out = combo_dir(model, balance, optimizer, run_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path, error_path = out / f'fold{fold}.json', out / f'fold{fold}.error.json'
    if result_path.exists():
        return 'skip'
    np.random.seed(SEED)
    random.seed(SEED)
    sklearn.set_config(working_memory=256)   # KNN calcula distancias en bloques de 256 MB
    spec = ALL_SPECS[model]
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=max(1, n_jobs)):
            sample, train, valid = read_fold(fold, 'search'), read_fold(fold, 'train'), read_fold(fold, 'valid')
            # Controles de fuga: establecimientos disjuntos y muestra contenida en el entrenamiento.
            assert not set(train.establishment_id) & set(valid.establishment_id)
            assert set(sample.id) <= set(train.id) and not set(sample.id) & set(valid.id)
            splits = inner_splits(sample)
            for a, b in splits:
                assert not set(sample.establishment_id.iloc[a]) & set(sample.establishment_id.iloc[b])
            objective = make_objective(spec, balance, sample, splits, n_jobs, device)
            search_start = time.perf_counter()
            history, best, details = optimize(optimizer, spec, objective, n_evals)
            search_seconds = time.perf_counter() - search_start

            y_train, y_valid = _labels(train), _labels(valid)
            caught = []
            pipe, fit_params, note = build_pipeline(spec, best['params'], balance, y_train, n_jobs, device)
            big = needs_big_memory(model, best['params'], len(train))
            wait_start = time.perf_counter()
            with memory_slot(big):
                wait_seconds = time.perf_counter() - wait_start
                refit_start = time.perf_counter()
                _fit(pipe, train[RAW_FEATURES], y_train, fit_params, caught)
                refit_seconds = time.perf_counter() - refit_start
                assert list(pipe.classes_) == LABELS
                predict_start = time.perf_counter()
                proba = pipe.predict_proba(valid[RAW_FEATURES])
                predict_seconds = time.perf_counter() - predict_start
                assert proba.shape == (len(valid), 4) and np.isfinite(proba).all()
                rows = np.random.default_rng(SEED).choice(len(train), size=min(TRAIN_METRIC_ROWS, len(train)), replace=False)
                train_metrics = classification_metrics(y_train[rows], pipe.predict_proba(train[RAW_FEATURES].iloc[rows]))
                n_features = int(len(pipe.named_steps['onehot'].get_feature_names_out()))
            del pipe

        np.savez_compressed(out / f'fold{fold}_pred.npz', proba=proba.astype(np.float32), y=y_valid.astype(np.int8),
                            establishment_id=valid.establishment_id.to_numpy(dtype=str),
                            case_id=valid.id.to_numpy(dtype=str), fold=np.full(len(valid), fold, dtype=np.int8))
        record = {
            'model': model, 'balance': balance, 'optimizer': optimizer, 'outer_fold': fold,
            'seed': SEED, 'inner_seed': INNER_SEED, 'search_fraction': SEARCH_FRACTION, 'device': device,
            'n_evals': len(history), 'best_params': best['params'], 'best_inner_score': best['score'],
            'best_eval': best['eval'], 'metrics': classification_metrics(y_valid, proba),
            'train_metrics_subsample': train_metrics, 'n_encoded_features': n_features,
            'seconds_total': time.perf_counter() - start, 'seconds_search': search_seconds,
            'seconds_per_eval': search_seconds / max(1, len(history)), 'seconds_refit': refit_seconds,
            'seconds_predict': predict_seconds, 'seconds_wait_memory_slot': wait_seconds,
            'big_memory_refit': big, 'search_rows': len(sample), 'train_rows': len(train),
            'valid_rows': len(valid), 'train_deaths': int((y_train == 1).sum()),
            'valid_deaths': int((y_valid == 1).sum()), 'refit_balance_note': note,
            'refit_warnings': dict(Counter(caught)), 'history': history, 'optimizer_details': details,
            'pid': os.getpid(),
        }
        tmp = result_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        tmp.replace(result_path)
        if error_path.exists():
            error_path.unlink()
        return 'ok'
    except Exception as exc:  # se registra y la corrida sigue con la siguiente combinación
        error_path.write_text(json.dumps({'model': model, 'balance': balance, 'optimizer': optimizer,
                                          'outer_fold': fold, 'error': repr(exc),
                                          'traceback': traceback.format_exc(),
                                          'seconds': time.perf_counter() - start}, indent=1), encoding='utf-8')
        return 'error'
