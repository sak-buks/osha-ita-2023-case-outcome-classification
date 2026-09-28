"""Rapid OSHA screening; raw CSV is read-only. Run with ml_env Python."""
from pathlib import Path
from collections import Counter
import json
import sys
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'osha' / 'ITA Case Detail Data 2023 through 12-31-2023OIICS.csv'
REPORT = ROOT / 'osha_screening_results.json'

def inspect():
    missing = Counter()
    counts = {c: Counter() for c in ['incident_outcome', 'type_of_incident', 'naics_code', 'soc_description', 'soc_code', 'year_filing_for']}
    sectors = Counter()
    establishments, ids, keys = [], [], []
    hashes, content_hashes = [], []
    n = 0
    earliest, latest, invalid_dates, outside_2023 = None, None, 0, 0
    # Byte-preserving decoding: file fails both strict UTF-8 and CP1252.
    # Non-ASCII narrative text is inspected only for missingness/duplicates, never modeled.
    for d in pd.read_csv(DATA, dtype=str, encoding='latin1', chunksize=50000):
        d = d.replace(r'^\s*$', np.nan, regex=True)
        n += len(d)
        missing.update(d.isna().sum().to_dict())
        for c in counts:
            counts[c].update(d[c].fillna('<missing>').value_counts().to_dict())
        sectors.update(d.naics_code.str[:2].replace({'32':'31-33','33':'31-33','31':'31-33','45':'44-45','44':'44-45','49':'48-49','48':'48-49'}).fillna('<missing>').value_counts().to_dict())
        establishments.append(d.establishment_id)
        ids.append(d.id)
        keys.append(d[['establishment_id', 'case_number']])
        hashes.append(pd.util.hash_pandas_object(d, index=False).to_numpy())
        content_hashes.append(pd.util.hash_pandas_object(d.drop(columns=['id','case_number','created_timestamp']), index=False).to_numpy())
        dates = pd.to_datetime(d.date_of_incident, format='%m/%d/%Y', errors='coerce')
        invalid_dates += int((dates.isna() & d.date_of_incident.notna()).sum())
        outside_2023 += int((dates.notna() & dates.dt.year.ne(2023)).sum())
        if dates.notna().any():
            earliest = dates.min() if earliest is None else min(earliest, dates.min())
            latest = dates.max() if latest is None else max(latest, dates.max())
        print('Inspected', n, 'rows', flush=True)
    establishment = pd.concat(establishments, ignore_index=True)
    ids = pd.concat(ids, ignore_index=True)
    keys = pd.concat(keys, ignore_index=True)
    result = {'rows':n, 'columns':list(d.columns), 'column_count':len(d.columns),
              'unique_establishments':int(establishment.nunique()),
              'missing_counts':dict(missing), 'distributions':{c:dict(v.most_common()) for c,v in counts.items()},
              'industry_sectors':dict(sectors.most_common()),
              'duplicate_id_extra_rows':int(ids.duplicated().sum()),
              'duplicate_establishment_case_extra_rows':int(keys.duplicated().sum()),
              'duplicate_establishment_case_all_rows':int(keys.duplicated(keep=False).sum()),
              'exact_duplicate_extra_rows_hash_check':int(pd.Series(np.concatenate(hashes)).duplicated().sum()),
              'duplicate_content_excluding_id_case_submission_time_extra_rows':int(pd.Series(np.concatenate(content_hashes)).duplicated().sum()),
              'date_min':str(earliest), 'date_max':str(latest), 'invalid_dates':invalid_dates, 'outside_2023':outside_2023}
    REPORT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    concise = {k:v for k,v in result.items() if k != 'distributions'}
    concise['distributions'] = {k:dict(list(v.items())[:12]) for k,v in result['distributions'].items()}
    print(json.dumps(concise, indent=2), flush=True)

def baseline():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
    from threadpoolctl import threadpool_limits
    import sklearn
    import time

    result = json.loads(REPORT.read_text(encoding='utf-8'))
    categorical = ['state', 'naics_code', 'establishment_type', 'soc_code', 'type_of_incident', 'time_unknown']
    time_fields = ['time_started_work', 'time_of_incident']
    columns = categorical + time_fields + ['id', 'establishment_id', 'case_number', 'incident_outcome', 'date_of_incident']
    d = pd.read_csv(DATA, usecols=columns, dtype=str, encoding='latin1')
    d = d.replace(r'^\s*$', np.nan, regex=True)
    # Reused employer case numbers often have different dates/outcomes. They are
    # not a verified duplicate-case key; unique ITA IDs remain separate records.
    incident_dates = pd.to_datetime(d.date_of_incident, format='%m/%d/%Y', errors='coerce')
    valid = d.incident_outcome.isin(['1','2','3','4']) & d.establishment_id.notna() & incident_dates.dt.year.eq(2023)
    assert d.id.is_unique
    excluded_rows = int((~valid).sum())
    d = d.loc[valid].copy()
    X = d[categorical].copy()
    X['soc_code'] = X.soc_code.replace({'9999':np.nan, '0000':np.nan})
    for c in time_fields:
        hour = pd.to_numeric(d[c].str.extract(r'^(\d{1,2}):', expand=False), errors='coerce')
        X[c + '_hour'] = hour.where(hour.between(0,23)).map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    X.loc[d.time_unknown.eq('1'), 'time_of_incident_hour'] = np.nan
    y = d.incident_outcome.astype(int)
    groups = d.establishment_id
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X,y,groups))
    assert not set(groups.iloc[train]) & set(groups.iloc[test])
    assert set(y.iloc[train]) == set(y.iloc[test]) == {1,2,3,4}
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    print('Fitting one grouped multiclass logistic-regression baseline:', len(train), 'train rows;', len(test), 'test rows.', flush=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=2):
            pipeline.fit(X.iloc[train], y.iloc[train])
            prediction = pipeline.predict(X.iloc[test])
    majority = int(y.iloc[train].mode().iloc[0])
    metrics = {
        'features':list(X.columns), 'excluded_rows':excluded_rows,
        'raw_feature_columns':categorical + time_fields,
        'excluded_raw_columns': [c for c in result['columns'] if c not in categorical + time_fields + ['incident_outcome']],
        'preprocessing':'SOC 9999/0000 treated as missing; hours parsed per record; unknown incident times masked. Training-only constant imputation and sparse one-hot encoding with unseen categories ignored.',
        'exclusion_rule':'Exclude missing/invalid target or establishment ID, and incident dates outside 2023. Retain unique ITA IDs with reused employer case numbers; no exact full-row duplicates were found.',
        'train_rows':len(train), 'test_rows':len(test),
        'train_establishments':int(groups.iloc[train].nunique()), 'test_establishments':int(groups.iloc[test].nunique()),
        'group_overlap':0, 'random_state':42, 'test_group_fraction':0.2,
        'train_distribution':y.iloc[train].value_counts().sort_index().to_dict(),
        'test_distribution':y.iloc[test].value_counts().sort_index().to_dict(),
        'model':'LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42); other defaults',
        'balanced_accuracy':balanced_accuracy_score(y.iloc[test],prediction),
        'accuracy':accuracy_score(y.iloc[test],prediction),
        'macro_f1':f1_score(y.iloc[test],prediction,average='macro',zero_division=0),
        'confusion_matrix_labels':[1,2,3,4],
        'confusion_matrix':confusion_matrix(y.iloc[test],prediction,labels=[1,2,3,4]).tolist(),
        'classification_report':classification_report(y.iloc[test],prediction,zero_division=0,output_dict=True),
        'majority_class':majority, 'majority_accuracy_reference':float(y.iloc[test].eq(majority).mean()),
        'constant_classifier_balanced_accuracy_reference':0.25,
        'iterations':pipeline.named_steps['classifier'].n_iter_.tolist(),
        'warnings':[str(w.message) for w in caught],
        'elapsed_seconds':time.perf_counter()-started, 'sklearn_version':sklearn.__version__,
        'test_establishment_ids':sorted(groups.iloc[test].unique().tolist()),
    }
    result['baseline'] = metrics
    REPORT.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k != 'test_establishment_ids'},indent=2),flush=True)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--baseline':
        baseline()
    else:
        inspect()
