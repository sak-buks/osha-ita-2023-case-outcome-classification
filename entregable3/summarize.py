"""Resume el avance y construye el registro maestro y las curvas de desempeño de los optimizadores.

Salidas (en results/):
- master_registry.csv / .parquet: una fila por (modelo, balanceo, optimizador, pliegue externo).
- convergence_curves.csv: mejor valor interno hasta la evaluación t, por optimizador.
- errors.csv: tareas con error y la última línea del traceback.
"""
import json
from pathlib import Path

import pandas as pd

from osha3.models import REGISTRY
from osha3.optimizers import OPTIMIZERS
from osha3.pipeline import BALANCES

ROOT = Path(__file__).resolve().parent / 'results'
RUNS = ROOT / 'runs'


def main():
    expected = sum(1 for m in REGISTRY for b in BALANCES if not (b == 'class_weight' and REGISTRY[m].weight_mode is None)) \
        * len(OPTIMIZERS) * 3
    rows, curves, errors = [], [], []
    for path in sorted(RUNS.glob('*/fold*.json')):
        if path.name.endswith('.error.json'):
            e = json.loads(path.read_text(encoding='utf-8'))
            errors.append({**{k: e[k] for k in ['model', 'balance', 'optimizer', 'outer_fold']},
                           'error': e['error'], 'last_line': e['traceback'].strip().splitlines()[-1]})
            continue
        r = json.loads(path.read_text(encoding='utf-8'))
        n_warn = sum(h.get('n_warnings', 0) for h in r['history'])
        rows.append({
            **{k: r[k] for k in ['model', 'balance', 'optimizer', 'outer_fold', 'seed', 'inner_seed', 'device',
                                 'n_evals', 'best_eval', 'best_inner_score', 'seconds_total', 'seconds_search',
                                 'seconds_per_eval', 'seconds_refit', 'seconds_predict', 'train_rows', 'valid_rows',
                                 'train_deaths', 'valid_deaths', 'n_encoded_features', 'refit_balance_note']},
            'best_params': json.dumps(r['best_params'], default=str),
            **{f'outer_{k}': v for k, v in r['metrics'].items()},
            **{f'train_{k}': r['train_metrics_subsample'][k] for k in ['balanced_accuracy', 'f1_macro', 'recall_death']},
            'search_warnings': n_warn, 'refit_warnings': sum(r['refit_warnings'].values()),
        })
        best = -1.0
        for h in r['history']:
            best = max(best, h['score'])
            curves.append({'model': r['model'], 'balance': r['balance'], 'optimizer': r['optimizer'],
                           'outer_fold': r['outer_fold'], 'eval': h['eval'], 'score': h['score'],
                           'best_so_far': best, 'seconds': h['seconds']})
    real = [r for r in rows if r['model'] in REGISTRY]
    print(f'Terminadas: {len(real)}/{expected} (más {len(rows) - len(real)} de referencia) | errores: {len(errors)}')
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(ROOT / 'master_registry.csv', index=False)
        table.to_parquet(ROOT / 'master_registry.parquet', index=False)
        pd.DataFrame(curves).to_csv(ROOT / 'convergence_curves.csv', index=False)
        hours = table.groupby('model').seconds_total.sum() / 3600
        print('Horas de cómputo acumuladas por modelo:\n', hours.round(2).to_string())
        cols = ['outer_balanced_accuracy', 'outer_f1_macro', 'outer_roc_auc_macro_ovr', 'outer_recall_death']
        print('Media sobre pliegues externos (combinaciones con 3 pliegues):')
        g = table.groupby(['model', 'balance', 'optimizer'])
        summary = g[cols].mean()[g.size() == 3].sort_values('outer_balanced_accuracy', ascending=False)
        print(summary.round(3).head(20).to_string())
        print('Avisos durante la búsqueda / reentrenamiento:', int(table.search_warnings.sum()),
              '/', int(table.refit_warnings.sum()))
    if errors:
        pd.DataFrame(errors).to_csv(ROOT / 'errors.csv', index=False)
        print(pd.DataFrame(errors).to_string(index=False))


if __name__ == '__main__':
    main()
