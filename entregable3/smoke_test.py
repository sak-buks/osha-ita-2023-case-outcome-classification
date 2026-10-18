"""Prueba de humo: fidelidad de los datos, ausencia de fugas y una combinación de punta a punta."""
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone

from osha3.data import RAW_FEATURES, load_data, outer_folds, prepare_features, read_fold, inner_splits
from osha3.models import REGISTRY
from osha3.nested import run_combination
from osha3.pipeline import build_pipeline

warnings.filterwarnings('ignore', category=FutureWarning)
SMOKE = Path(__file__).resolve().parent / 'results' / 'smoke_runs'


def check_parquet_matches_csv():
    d_dev, _, _ = load_data()
    _, train_csv, _ = next(outer_folds(d_dev))
    train_pq = read_fold(1, 'train')
    a = prepare_features(train_csv[RAW_FEATURES].reset_index(drop=True)).fillna('<NA>')
    b = prepare_features(train_pq[RAW_FEATURES]).fillna('<NA>')
    assert a.equals(b), 'El parquet no reproduce el CSV'
    print('OK parquet = CSV tras prepare_features:', a.shape)


def check_no_leakage():
    sample = read_fold(1, 'search')
    a, b = inner_splits(sample)[0]
    y = sample.incident_outcome.astype(int).to_numpy()
    pipe, fp, note = build_pipeline(REGISTRY['logreg'], {'C': 1.0, 'min_frequency': 5e-4}, 'smote', y[a])
    pipe.fit(sample[RAW_FEATURES].iloc[a], y[a], **fp)
    enc = pipe.named_steps['onehot']
    fit_values = prepare_features(sample[RAW_FEATURES].iloc[a]).fillna('Unknown')
    for col, cats in zip(fit_values.columns, enc.categories_):
        assert set(cats) <= set(fit_values[col]), f'categoría fuera del entrenamiento en {col}'
    assert len(pipe.predict(sample[RAW_FEATURES].iloc[b])) == len(b), 'predict no debe remuestrear'
    Xt = enc.transform(pipe.named_steps['imputer'].transform(prepare_features(sample[RAW_FEATURES].iloc[a])))
    Xr, yr = pipe.named_steps['sampler'].fit_resample(Xt, y[a])
    print('OK one-hot aprendido solo del entrenamiento interno;', note,
          '| muertes antes/después de SMOTE:', int((y[a] == 1).sum()), int((yr == 1).sum()),
          '| columnas:', Xt.shape[1])


def check_xgb_clone():
    pipe, _, _ = build_pipeline(REGISTRY['xgboost'], {}, 'none', np.array([1, 2, 3, 4]))
    clone(pipe.named_steps['clf'])
    print('OK XGBoost con clases 1..4 se puede clonar')


def main():
    check_parquet_matches_csv()
    check_no_leakage()
    check_xgb_clone()
    shutil.rmtree(SMOKE, ignore_errors=True)
    jobs = [('logreg', 'smote', 'random', 1)] + [('naive_bayes', bal, opt, 1)
            for bal, opt in [('none', 'grid'), ('class_weight', 'random'), ('adasyn', 'bayes'), ('smote', 'genetic')]]
    for model, bal, opt, fold in jobs:
        start = time.perf_counter()
        status = run_combination(model, bal, opt, fold, n_evals=3 if model == 'logreg' else 12, run_dir=SMOKE)
        name = f'{model}__{bal}__{opt}'
        path = SMOKE / name / f'fold{fold}.json'
        if status != 'ok':
            print('ERROR', name, (SMOKE / name / f'fold{fold}.error.json').read_text()[-1500:])
            continue
        r = json.loads(path.read_text(encoding='utf-8'))
        m = r['metrics']
        print(f"{name}: {time.perf_counter() - start:.1f}s | evals={r['n_evals']} | s/eval={r['seconds_per_eval']:.2f} "
              f"| refit={r['seconds_refit']:.1f}s | interno={r['best_inner_score']:.3f} | externo BA={m['balanced_accuracy']:.3f} "
              f"F1m={m['f1_macro']:.3f} AUC={m['roc_auc_macro_ovr']:.3f} recall_death={m['recall_death']:.2f} "
              f"| avisos refit={sum(r['refit_warnings'].values())} | {r['refit_balance_note']}")
        if opt == 'genetic':
            print('   generaciones:', [(g['generation'], round(g['diversity'], 3), round(g['best'], 3))
                                     for g in r['optimizer_details']['generations']])


if __name__ == '__main__':
    main()
