"""Group-disjoint, forward-in-time baseline for the OSHA project."""
from pathlib import Path
import hashlib
import json
import warnings
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from osha_deliverable_utils import (
    RAW_FEATURES, FEATURES, LABELS, PARAMS, SEED, fit_cached, predict,
    metrics, digest, bootstrap_groups, cm_metrics,
)

CUTOFF = pd.Timestamp('2023-10-01')
GAP_DAYS = 7
HOLDOUT_GROUP_FRACTION = 0.60
WINDOWS = [
    (pd.Timestamp('2023-04-01'), pd.Timestamp('2023-06-01')),
    (pd.Timestamp('2023-06-01'), pd.Timestamp('2023-08-01')),
    (pd.Timestamp('2023-08-01'), CUTOFF),
]
DESIGN = 'group-time-forward-1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def assign_test_groups(groups):
    """Randomly reserve whole establishments, independent of their outcomes."""
    ordered = np.asarray(sorted(pd.unique(groups)), dtype=object)
    selected = np.random.default_rng(SEED).permutation(ordered)
    n_test = int(np.floor(len(ordered) * HOLDOUT_GROUP_FRACTION))
    return set(selected[:n_test]), len(ordered)


def sector_codes(values):
    return values.str[:2].replace({
        '31': '31-33', '32': '31-33', '33': '31-33',
        '44': '44-45', '45': '44-45', '48': '48-49', '49': '48-49',
    })


def subgroup_scores(y, p):
    truth = np.asarray(y)
    pred = np.asarray(LABELS)[p.argmax(axis=1)]
    cm = confusion_matrix(truth, pred, labels=LABELS)
    report = classification_report(
        truth, pred, labels=LABELS, output_dict=True, zero_division=0
    )
    present = cm.sum(axis=1) > 0
    recalls = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(4), where=present)
    return {
        'accuracy': float(np.trace(cm) / cm.sum()),
        'balanced_accuracy_present_classes': float(recalls[present].mean()),
        'macro_f1_four_classes': float(np.mean([report[str(k)]['f1-score'] for k in LABELS])),
        'class_support': {str(k): int(report[str(k)]['support']) for k in LABELS},
        'classification_report': report,
        'auc_macro_ovr': metrics(truth, p)['roc_auc_macro_ovr'] if present.all() else None,
    }


def wilson_interval(successes, n, z=1.959963984540054):
    if n <= 0:
        return [None, None]
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]


def sector_intervals(y, groups, p, repeats=500):
    codes, unique = pd.factorize(groups, sort=True)
    cells = (np.asarray(y) - 1) * 4 + p.argmax(axis=1)
    matrices = np.bincount(codes * 16 + cells, minlength=len(unique) * 16).reshape(len(unique), 4, 4)
    rng = np.random.default_rng(SEED + 200)
    values = []
    for _ in range(repeats):
        weights = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        cm = np.einsum('g,gij->ij', weights, matrices)
        score = cm_metrics(cm)
        values.append([score['accuracy'], score['macro_f1']])
    bounds = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        'accuracy_ci95': bounds[:, 0].tolist(),
        'macro_f1_ci95': bounds[:, 1].tolist(),
        'bootstrap_replicates': repeats,
    }


def _class_counts(y):
    counts = pd.Series(y).value_counts().reindex(LABELS, fill_value=0)
    return {str(k): int(counts.loc[k]) for k in LABELS}


def _fold_data(d, fold, start, end):
    # Fold assignment uses only eligible development establishments; the date
    # filter then enforces forward evaluation and the seven-day purge gap.
    available = d.loc[d.date.lt(end)]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    train_idx, valid_idx = list(splitter.split(
        np.zeros(len(available)), available.incident_outcome.astype(int), available.establishment_id
    ))[fold - 1]
    train = available.iloc[train_idx]
    valid = available.iloc[valid_idx]
    train = train.loc[train.date.lt(start - pd.Timedelta(days=GAP_DAYS))]
    valid = valid.loc[valid.date.ge(start) & valid.date.lt(end)]
    assert not set(train.establishment_id).intersection(valid.establishment_id)
    assert set(train.incident_outcome.astype(int)) == set(LABELS)
    assert set(valid.incident_outcome.astype(int)) == set(LABELS)
    return train, valid


def run_study(data_path, cache_dir):
    source, cache = Path(data_path), Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    signature = {
        'raw_sha256': source_hash,
        'raw_size': source.stat().st_size,
        'raw_mtime_ns': source.stat().st_mtime_ns,
        'implementation_sha256': sha256(__file__),
        'utility_sha256': sha256(Path(__file__).with_name('osha_deliverable_utils.py')),
        'features': RAW_FEATURES,
        'parameters': PARAMS,
        'sklearn': sklearn.__version__,
        'seed': SEED,
        'holdout_group_fraction': HOLDOUT_GROUP_FRACTION,
        'cutoff': str(CUTOFF.date()),
        'gap_days': GAP_DAYS,
        'windows': [[str(a.date()), str(b.date())] for a, b in WINDOWS],
        'learning_fractions': [0.25, 0.5, 1.0],
        'bootstrap_cm': 1000,
        'bootstrap_auc': 200,
        'design': DESIGN,
    }
    report_path = cache / 'evaluation.json'
    if report_path.exists():
        saved = json.loads(report_path.read_text(encoding='utf-8'))
        if saved.get('signature') == json.loads(json.dumps(signature)):
            print('Reusing completed group-time study.', flush=True)
            return saved

    columns = RAW_FEATURES + ['id', 'establishment_id', 'incident_outcome', 'date_of_incident', 'naics_code']
    d = pd.read_csv(source, usecols=columns, dtype=str, encoding='latin1')
    d['date'] = pd.to_datetime(d.date_of_incident, format='%m/%d/%Y', errors='coerce')
    d = d.loc[d.date.dt.year.eq(2023)].copy()
    assert d.establishment_id.notna().all(), 'Cannot group cases with missing establishment identifiers.'
    assert d.id.is_unique and d.incident_outcome.isin(['1', '2', '3', '4']).all()
    held_groups, total_groups = assign_test_groups(d.establishment_id)
    is_held = d.establishment_id.isin(held_groups)
    cutoff_train = CUTOFF - pd.Timedelta(days=GAP_DAYS)
    tr = d.loc[~is_held & d.date.lt(cutoff_train)].copy()
    te = d.loc[is_held & d.date.ge(CUTOFF)].copy()
    assert tr.date.max() < cutoff_train and te.date.min() >= CUTOFF
    assert not set(tr.establishment_id).intersection(te.establishment_id)
    y_train, y_test = tr.incident_outcome.astype(int), te.incident_outcome.astype(int)
    assert set(y_train) == set(y_test) == set(LABELS)
    assert int(y_test.eq(1).sum()) >= 30, 'Insufficient death support for the prespecified group holdout.'

    split = pd.Series('excluded_by_group_time_design', index=d.index, dtype=object)
    split.loc[tr.index] = 'development'
    split.loc[te.index] = 'test'
    pd.DataFrame({
        'id': d.id, 'establishment_id': d.establishment_id, 'date': d.date,
        'split': split.to_numpy(),
    }).to_csv(cache / 'partition.csv', index=False)

    cv_rows, learning = [], []
    cv_path = cache / 'group_time_validation.json'
    for fold, (start, end) in enumerate(WINDOWS, 1):
        a, b = _fold_data(d.loc[~is_held], fold, start, end)
        groups = np.random.default_rng(SEED + fold).permutation(np.sort(a.establishment_id.unique()))
        for fraction in [0.25, 0.5, 1.0]:
            chosen = groups[:max(1, int(np.ceil(len(groups) * fraction)))]
            z = a.loc[a.establishment_id.isin(chosen)]
            y, yv = z.incident_outcome.astype(int), b.incident_outcome.astype(int)
            assert set(y) == set(yv) == set(LABELS)
            fitted = fit_cached(
                f'cv{fold}_fraction{int(fraction * 100)}', z[RAW_FEATURES], y, cache, signature
            )
            row = {
                'fold': fold, 'fraction': fraction, 'train_rows': len(z), 'validation_rows': len(b),
                'train_groups': int(z.establishment_id.nunique()),
                'validation_groups': int(b.establishment_id.nunique()), 'group_overlap': 0,
                'train_deaths': int(y.eq(1).sum()), 'validation_deaths': int(yv.eq(1).sum()),
                'train_date_max': str(z.date.max().date()), 'validation_date_min': str(b.date.min().date()),
                'validation_date_max': str(b.date.max().date()), 'gap_days': GAP_DAYS,
                'iterations': fitted['iterations'], 'converged': fitted['converged'],
                'train_metrics': metrics(y, predict(fitted['model'], z[RAW_FEATURES])),
                'validation_metrics': metrics(yv, predict(fitted['model'], b[RAW_FEATURES])),
            }
            learning.append(row)
            if fraction == 1.0:
                cv_rows.append(dict(row, model='LogisticRegression', metrics=row['validation_metrics']))
                dummy = fit_cached(f'cv{fold}_dummy', z[RAW_FEATURES], y, cache, signature, dummy=True)
                cv_rows.append(dict(
                    row, model='DummyClassifier',
                    metrics=metrics(yv, predict(dummy['model'], b[RAW_FEATURES])),
                ))
        cv_path.write_text(json.dumps({'signature': signature, 'cv': cv_rows, 'learning_curve': learning}, indent=2), encoding='utf-8')

    final = fit_cached('final_lr', tr[RAW_FEATURES], y_train, cache, signature)
    dummy = fit_cached('final_dummy', tr[RAW_FEATURES], y_train, cache, signature, dummy=True)
    probabilities = {
        'LogisticRegression': predict(final['model'], te[RAW_FEATURES]),
        'DummyClassifier': predict(dummy['model'], te[RAW_FEATURES]),
    }
    np.savez_compressed(
        cache / 'holdout_predictions.npz', y=y_test.to_numpy(), groups=te.establishment_id.to_numpy(dtype=str),
        dates=te.date.to_numpy(dtype='datetime64[D]'), case_ids=te.id.to_numpy(dtype=str),
        lr=probabilities['LogisticRegression'], dummy=probabilities['DummyClassifier'],
    )

    sectors = sector_codes(te.naics_code)
    selected = sector_codes(tr.naics_code).value_counts()
    selected = selected[selected.ge(1000)].index.tolist()
    sector_results = []
    for sector in selected:
        mask = sectors.eq(sector).to_numpy()
        if not mask.any():
            continue
        for name, p in probabilities.items():
            sector_results.append({
                'sector': sector, 'model': name, 'rows': int(mask.sum()),
                'establishments': int(te.loc[mask, 'establishment_id'].nunique()),
                **subgroup_scores(y_test.to_numpy()[mask], p[mask]),
                **sector_intervals(y_test.to_numpy()[mask], te.loc[mask, 'establishment_id'], p[mask]),
            })

    boot = bootstrap_groups(y_test, te.establishment_id, probabilities)
    residuals = pd.DataFrame({
        'date': te.date,
        'error': (np.asarray(LABELS)[probabilities['LogisticRegression'].argmax(axis=1)] != y_test).astype(float),
    })
    daily = residuals.groupby('date').error.agg(['mean', 'count']).reindex(pd.date_range(CUTOFF, '2023-12-31'))
    daily.to_csv(cache / 'daily_test_errors.csv', index_label='date')
    error_acf = {str(lag): float(daily['mean'].autocorr(lag=lag)) for lag in [1, 2, 7, 14]}
    names = final['model'].named_steps['onehot'].get_feature_names_out(FEATURES)
    pd.DataFrame(final['model'].named_steps['classifier'].coef_.T, index=names, columns=LABELS).to_csv(cache / 'coefficients.csv')
    reports = {
        name: classification_report(y_test, np.asarray(LABELS)[p.argmax(axis=1)], labels=LABELS,
                                    output_dict=True, zero_division=0)
        for name, p in probabilities.items()
    }
    cms = {
        name: confusion_matrix(y_test, np.asarray(LABELS)[p.argmax(axis=1)], labels=LABELS).tolist()
        for name, p in probabilities.items()
    }
    death_cm = np.asarray(cms['LogisticRegression'])
    death_n = int(y_test.eq(1).sum())
    death_hits = int(death_cm[0, 0])
    class_names = ['Death', 'Days away', 'Transfer/restriction', 'Other recordable']
    result = {
        'signature': signature,
        'design': {
            'group_assignment': 'Random 60% establishment holdout; seed 42; assignment independent of outcome.',
            'development_period': f'2023-01-01 through {str((CUTOFF - pd.Timedelta(days=GAP_DAYS) - pd.Timedelta(days=1)).date())}; remaining 40% of establishments.',
            'test_period': '2023-10-01 through 2023-12-31; held-out establishments only.',
            'purged_gap_days': GAP_DAYS,
            'classes': class_names,
            'prior_test_exposure': 'Q4 outcomes and aggregate results were examined in earlier project analyses; the new group-time evaluation is not an untouched test.',
        },
        'class_counts': {'train': _class_counts(y_train), 'test': _class_counts(y_test)},
        'holdout_metrics': {name: metrics(y_test, p) for name, p in probabilities.items()},
        'classification_reports': reports,
        'confusion_matrices': cms,
        'death_sensitivity_wilson_ci95': wilson_interval(death_hits, death_n),
        'death_correctly_identified': death_hits,
        'final_fit': {k: v for k, v in final.items() if k not in ['model', 'identity']},
        'cv': cv_rows,
        'learning_curve': learning,
        'bootstrap': boot,
        'train_rows': len(tr), 'test_rows': len(te),
        'train_groups': int(tr.establishment_id.nunique()), 'test_groups': int(te.establishment_id.nunique()),
        'all_2023_groups': total_groups,
        'test_death_groups': int(te.loc[y_test.eq(1), 'establishment_id'].nunique()),
        'encoded_features': len(names), 'features': FEATURES, 'raw_features': RAW_FEATURES,
        'sector_analysis': sector_results,
        'daily_error_rate_acf': error_acf,
        'group_overlap': len(set(tr.establishment_id) & set(te.establishment_id)),
        'train_case_hash': digest(tr.id), 'test_case_hash': digest(te.id),
        'excluded_2023_rows': int(len(d) - len(tr) - len(te)),
        'uncertainty_limit': 'Establishment bootstrap is conditional on the fixed quarter and fitted models; it does not cover common calendar shocks or training uncertainty.',
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Group-time study completed.', flush=True)
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    run_study(root / 'ITA Case Detail Data 2023 through 12-31-2023OIICS.csv', root / 'osha_group_time_results')
