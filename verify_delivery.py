"""Verify the executed group-time notebook and saved study artifacts."""
from pathlib import Path
import json
import os
import numpy as np
import nbformat

from osha_group_time_study import run_study
from osha_deliverable_utils import metrics

root = Path(__file__).resolve().parent
results = root / 'osha_group_time_results'
n = nbformat.read(root / 'osha_2023_eda_revisado.ipynb', as_version=4)
code = [c for c in n.cells if c.cell_type == 'code']
errors = [o for c in code for o in c.get('outputs', []) if o.output_type == 'error']
assert not errors, errors
assert all(c.execution_count is not None for c in code), 'Unexecuted code cells'
data_root = Path(os.environ.get('OSHA_DATA_DIR', root))
raw = data_root / 'ITA Case Detail Data 2023 through 12-31-2023OIICS.csv'
report = run_study(raw, results)
assert report['design']['test_period'].startswith('2023-10-01')
assert report['train_rows'] == 269516 and report['test_rows'] == 127391
assert report['train_groups'] > 0 and report['test_groups'] > 0
assert report['group_overlap'] == 0
assert report['class_counts']['train'] == {'1': 69, '2': 99697, '3': 79848, '4': 89902}
assert report['class_counts']['test'] == {'1': 39, '2': 46169, '3': 37231, '4': 43952}
assert report['test_death_groups'] == 39
assert report['final_fit']['converged']
assert all(r['converged'] and r['group_overlap'] == 0 for r in report['learning_curve'])
assert len(report['learning_curve']) == 9 and len(report['cv']) == 6
assert all(
    np.datetime64(r['train_date_max']) < np.datetime64(r['validation_date_min']) - np.timedelta64(7, 'D')
    for r in report['cv'] if r['model'] == 'LogisticRegression'
)
pred = np.load(results / 'holdout_predictions.npz', allow_pickle=False)
assert len(np.unique(pred['case_ids'])) == len(pred['case_ids']) == report['test_rows']
assert pred['dates'].min() >= np.datetime64('2023-10-01')
assert pred['dates'].max() <= np.datetime64('2023-12-31')
assert np.allclose(pred['lr'].sum(axis=1), 1) and np.isfinite(pred['lr']).all()
assert np.allclose(pred['dummy'].sum(axis=1), 1) and np.isfinite(pred['dummy']).all()
assert len(n.cells) == 65 and len(code) == 28
assert raw.stat().st_size == report['signature']['raw_size']
assert (root / '_build/html/index.html').exists()
assert (root / '_build/html/osha_2023_eda_revisado.html').exists()
build_log = (results / 'book_build.log').read_text(encoding='utf-8')
assert 'build succeeded' in build_log.lower()
for name, key in [('LogisticRegression', 'lr'), ('DummyClassifier', 'dummy')]:
    actual = metrics(pred['y'], pred[key])
    assert all(np.isclose(actual[k], v, rtol=0, atol=1e-12) for k, v in report['holdout_metrics'][name].items())
summary = {
    'status': 'passed', 'executed_code_cells': len(code), 'notebook_cells': len(n.cells),
    'models_converged': True, 'train_rows': report['train_rows'], 'test_rows': report['test_rows'],
    'train_establishments': report['train_groups'], 'test_establishments': report['test_groups'],
    'establishment_overlap': report['group_overlap'], 'train_class_counts': report['class_counts']['train'],
    'test_class_counts': report['class_counts']['test'], 'test_death_support': report['class_counts']['test']['1'],
    'death_sensitivity_wilson_ci95': report['death_sensitivity_wilson_ci95'],
    'historical_test_exposure': 'disclosed; test is not wholly untouched',
    'raw_content_sha256_verified': True,
    'primary_metrics_recomputed': True,
    'jupyter_book_build': 'passed',
    'excluded_2023_rows': report['excluded_2023_rows'],
}
(results / 'delivery_verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
