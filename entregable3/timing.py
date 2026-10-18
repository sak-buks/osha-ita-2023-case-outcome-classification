"""Mide el tiempo real por modelo en el pliegue externo 3 (el más grande) para proyectar el presupuesto.

Usa configuraciones de rango medio de cada espacio (no los valores por defecto) y un núcleo por ajuste.
"""
import sys
import time
import warnings

import pandas as pd
from threadpoolctl import threadpool_limits

from osha3.data import RAW_FEATURES, read_fold, inner_splits
from osha3.metrics import classification_metrics
from osha3.models import REGISTRY
from osha3.nested import make_objective, _labels
from osha3.pipeline import build_pipeline

warnings.filterwarnings('ignore')

CASES = [
    ('knn', {'n_neighbors': 25, 'weights': 'distance'}, 'cpu'),
    ('naive_bayes', {'alpha': 1.0}, 'cpu'),
    ('logreg', {'C': 1.0, 'penalty': 'l1'}, 'cpu'),
    ('logreg', {'C': 100.0, 'penalty': 'l1'}, 'cpu'),
    ('tree', {'max_depth': 20, 'min_samples_leaf': 10, 'ccp_alpha': 1e-4}, 'cpu'),
    ('random_forest', {'n_estimators': 300, 'max_depth': 30, 'min_samples_leaf': 3}, 'cpu'),
    ('xgboost', {'n_estimators': 350, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.75,
                 'colsample_bytree': 0.65}, 'cpu'),
    ('xgboost', {'n_estimators': 350, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.75,
                 'colsample_bytree': 0.65}, 'cuda'),
    ('svm', {'kernel': 'linear', 'C': 1.0}, 'cpu'),
    ('svm', {'kernel': 'nystroem', 'n_components': 300, 'gamma': 0.05, 'C': 1.0}, 'cpu'),
    ('svm', {'kernel': 'nystroem', 'n_components': 1000, 'gamma': 0.05, 'C': 1.0}, 'cpu'),
]


def main(fold=3, only=None):
    sample, train, valid = read_fold(fold, 'search'), read_fold(fold, 'train'), read_fold(fold, 'valid')
    splits = inner_splits(sample)
    rows = []
    with threadpool_limits(limits=1):
        for model, params, device in [c for c in CASES if not only or c[0] in only]:
            spec = REGISTRY[model]
            row = {'model': model, 'device': device, 'params': params}
            for balance in ['none', 'smote']:
                t = time.perf_counter()
                score, _ = make_objective(spec, balance, sample, splits, 1, device)(params)
                row[f'eval_{balance}_s'] = time.perf_counter() - t
                row[f'inner_ba_{balance}'] = score
            y = _labels(train)
            pipe, fp, _ = build_pipeline(spec, params, 'none', y, 1, device)
            t = time.perf_counter()
            pipe.fit(train[RAW_FEATURES], y, **fp)
            row['refit_s'] = time.perf_counter() - t
            t = time.perf_counter()
            proba = pipe.predict_proba(valid[RAW_FEATURES])
            pipe.predict_proba(train[RAW_FEATURES].iloc[:10000])
            row['predict_s'] = time.perf_counter() - t
            row['outer_ba'] = classification_metrics(_labels(valid), proba)['balanced_accuracy']
            rows.append(row)
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(f'results/timing_fold{fold}{"_" + "_".join(only) if only else ""}.csv', index=False)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, sys.argv[2:] or None)
