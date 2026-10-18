"""Cuenta casos y muertes en cada entrenamiento externo e interno.

Sirve para fijar k_neighbors de SMOTE / n_neighbors de ADASYN:
k = min(5, muertes_en_entrenamiento_interno - 1); si hay menos de 2 muertes,
esa celda usa RandomOverSampler.
"""
import json
from pathlib import Path

import pandas as pd

from osha3.data import load_data, outer_folds, search_sample, inner_splits


def deaths(frame):
    return int(frame.incident_outcome.eq('1').sum())


def main():
    d_dev, dev, test = load_data()
    rows = []
    for fold, train, valid in outer_folds(d_dev):
        sample = search_sample(train)
        base = {'outer_fold': fold, 'outer_train_rows': len(train), 'outer_train_deaths': deaths(train),
                'outer_train_groups': train.establishment_id.nunique(),
                'outer_valid_rows': len(valid), 'outer_valid_deaths': deaths(valid),
                'train_date_max': str(train.date.max().date()), 'valid_dates': f'{valid.date.min().date()} a {valid.date.max().date()}',
                'search_rows': len(sample), 'search_deaths': deaths(sample)}
        for inner, (a, b) in enumerate(inner_splits(sample), 1):
            fit, val = sample.iloc[a], sample.iloc[b]
            assert not set(fit.establishment_id) & set(val.establishment_id)
            n = deaths(fit)
            rows.append(dict(base, inner_fold=inner, inner_train_rows=len(fit), inner_train_deaths=n,
                             inner_valid_rows=len(val), inner_valid_deaths=deaths(val),
                             k_neighbors=min(5, n - 1) if n >= 2 else 'RandomOverSampler'))
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 250, 'display.max_columns', 30)
    print(table.to_string(index=False))
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    table.to_csv(out / 'death_counts_by_fold.csv', index=False)
    print(json.dumps({'dev_rows': len(dev), 'test_rows': len(test)}))


if __name__ == '__main__':
    main()
