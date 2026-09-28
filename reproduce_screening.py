"""Reproduce only the read-only inventory; preserve historical baseline outputs."""
from pathlib import Path
import importlib.util
import json
import os
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_screening',root/'provenance/osha_screening.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.DATA=Path(os.environ.get('OSHA_DATA_DIR',root))/'ITA Case Detail Data 2023 through 12-31-2023OIICS.csv'
module.REPORT=root/'osha_temporal_results/screening_inventory_reproduced.json'
module.inspect()
original=json.loads((root/'osha_screening_results.json').read_text(encoding='utf-8'))
new=json.loads(module.REPORT.read_text(encoding='utf-8'))
assert all(original[k]==v for k,v in new.items()), 'Inventory differs from historical screening'
print('Inventory reproduced; historical baseline unchanged.')
