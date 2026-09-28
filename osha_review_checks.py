"""Supplementary diagnostics; never changes the primary fit, split, or raw data."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def single_feature_checks(features, target, dates, cache):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from threadpoolctl import threadpool_limits
    import sklearn
    cache = Path(cache)
    identity = hashlib.sha256(pd.util.hash_pandas_object(
        features.assign(_target=target, _date=dates), index=True).values.tobytes()).hexdigest()
    signature = {'data': identity, 'implementation': sha256(__file__), 'sklearn': sklearn.__version__}
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    if saved.get('signature') != signature:
        saved = {'signature': signature, 'checks': []}
    before = dates.lt('2023-08-01')
    for field in features:
        if any(r['Predictor'] == field for r in saved['checks']):
            continue
        z = features[[field]].astype(object).applymap(lambda x: str(x) if pd.notna(x) else np.nan)
        attempts = []
        # Same objective, C, weights and data. Change solver only after a numerical warning.
        for solver in ['newton-cg', 'lbfgs']:
            model = Pipeline([
                ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encode', OneHotEncoder(handle_unknown='ignore')),
                ('model', LogisticRegression(solver=solver, max_iter=1000,
                    class_weight='balanced', random_state=42))])
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                warnings.simplefilter('always')
                model.fit(z.loc[before], target.loc[before])
            messages = [f'{w.category.__name__}: {w.message}' for w in caught]
            iterations = int(model.named_steps['model'].n_iter_.max())
            valid = not messages and iterations < 1000
            attempts.append({'solver': solver, 'iterations': iterations, 'warnings': messages,
                             'numerically_verified': valid})
            if valid:
                break
        score = float(roc_auc_score(target.loc[~before], model.predict_proba(z.loc[~before]),
                      multi_class='ovr', average='macro')) if valid else None
        saved['checks'].append({'Predictor': field, 'AUC macro OvR en agosto–septiembre': score,
            'Solver final': solver, 'Iteraciones': iterations, 'Verificado': valid,
            'Intentos': attempts})
        cache.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Diagnóstico univariable:', field, solver, 'verificado:', valid, flush=True)
    return saved['checks']


def cross_partition_audit(data_path, cache):
    """Inspect repeated keys without exporting narrative text or recoding incidents."""
    cache = Path(cache)
    identity = {'raw_sha256': sha256(data_path), 'implementation_sha256': sha256(__file__)}
    if cache.exists():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        if saved['signature'] == identity:
            return saved
    keys = ['establishment_id', 'case_number']
    d = pd.read_csv(data_path, usecols=keys + ['id', 'date_of_incident'], dtype=str, encoding='latin1')
    date = pd.to_datetime(d.date_of_incident, format='%m/%d/%Y', errors='coerce')
    a, b = d.loc[date.dt.year.eq(2023) & date.lt('2023-10-01')], d.loc[date.dt.year.eq(2023) & date.ge('2023-10-01')]
    ka = pd.MultiIndex.from_frame(a[keys].dropna())
    kb = pd.MultiIndex.from_frame(b[keys].dropna())
    shared = ka.unique().intersection(kb.unique())
    parts = []
    for chunk in pd.read_csv(data_path, dtype=str, encoding='latin1', chunksize=50000):
        mask = pd.MultiIndex.from_frame(chunk[keys]).isin(shared)
        if mask.any():
            parts.append(chunk.loc[mask])
    z = pd.concat(parts, ignore_index=True).replace(r'^\s*$', np.nan, regex=True)
    date = pd.to_datetime(z.date_of_incident, format='%m/%d/%Y', errors='coerce')
    z = z.loc[date.dt.year.eq(2023)].copy()
    split = date.loc[z.index].ge('2023-10-01')
    omitted = ['id', 'case_number', 'created_timestamp', 'date_of_incident']
    content_columns = [c for c in z if c not in omitted]
    content = pd.util.hash_pandas_object(z[content_columns], index=False)
    comparison = z[keys].copy()
    comparison['_content'] = content
    ca = pd.MultiIndex.from_frame(comparison.loc[~split])
    cb = pd.MultiIndex.from_frame(comparison.loc[split])
    matches = ca.unique().intersection(cb.unique())
    result = {'signature': identity, 'shared_case_keys': len(shared),
        'shared_ita_ids': len(set(a.id) & set(b.id)),
        'development_rows_with_shared_key': int(ka.isin(shared).sum()),
        'test_rows_with_shared_key': int(kb.isin(shared).sum()),
        'shared_keys_with_matching_content_except_administrative_fields_and_incident_date':
            len(set((x[0], x[1]) for x in matches)),
        'matched_content_patterns': len(matches), 'excluded_from_content_comparison': omitted,
        'interpretation': 'Different incident dates by construction; matching content is a flag, not proof of a duplicate. No rows or labels changed.'}
    cache.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def temporal_supplement(train, display, plt, Markdown):
    daily = train.groupby(train.incident_date.dt.normalize()).size().reindex(
        pd.date_range('2023-01-01', '2023-09-30'), fill_value=0)
    frame = daily.rename('Casos').to_frame()
    frame['Día de semana'] = frame.index.dayofweek + 1
    frame['Mes'] = frame.index.month
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    daily.rolling(7, min_periods=7).var().plot(ax=axes[0])
    axes[0].set(title='Varianza móvil de 7 días', ylabel='Varianza de conteos')
    frame.boxplot(column='Casos', by='Día de semana', ax=axes[1])
    frame.boxplot(column='Casos', by='Mes', ax=axes[2])
    fig.suptitle('Distribuciones de conteos diarios: desarrollo')
    plt.tight_layout(); plt.show()
    aggregates = train.assign(day=train.incident_date.dt.normalize()).groupby('day').agg(
        n=('incident_outcome', 'size'),
        away=('incident_outcome', lambda x: x.eq(2).mean()),
        manufacturing=('industry_sector_code', lambda x: x.eq('31-33').mean()))
    aggregates = aggregates.reindex(daily.index)
    rows = []
    for lag in [0, 1, 7, 14]:
        pair = pd.concat([aggregates.manufacturing.shift(lag), aggregates.away], axis=1).dropna()
        # Remove each series' weekday average as a descriptive sensitivity.
        centered = pair - pair.groupby(pair.index.dayofweek).transform('mean')
        rows.append({'Rezago (días)': lag, 'Días utilizables': len(pair),
            'Correlación: proporción manufactura t-k / ausencia t': pair.iloc[:, 0].corr(pair.iloc[:, 1]),
            'Correlación tras centrar por día semanal': centered.iloc[:, 0].corr(centered.iloc[:, 1])})
    display(pd.DataFrame(rows))
    adjusted = daily - daily.groupby(daily.index.dayofweek).transform('mean')
    calendar = pd.DataFrame({'Casos diarios': daily, 'Desviación respecto al día semanal': adjusted})
    display(calendar.groupby(calendar.index.month).agg(['mean', 'std']))
    # Predefined descriptive screen: 28 days before/after, no optimized breakpoint or split.
    differences = adjusted.rolling(28).mean().shift(-28) - adjusted.rolling(28).mean()
    extreme = differences.abs().idxmax()
    display(Markdown(f'El mayor contraste descriptivo entre ventanas consecutivas de 28 días aparece en **{extreme.date()}**, con diferencia **{differences.loc[extreme]:.1f} casos/día** tras centrar por día semanal. Es un máximo exploratorio, no una prueba de cambio estructural; no se usa para cambiar el corte.'))
    quarter = train.incident_date.dt.quarter
    counts = pd.crosstab(train.establishment_id, quarter).reindex(columns=[1, 2, 3], fill_value=0)
    selected = counts.sum(axis=1).ge(30) & counts.gt(0).all(axis=1)
    volume = counts.loc[selected].div(counts.loc[selected].sum(axis=1), axis=0)
    away = pd.crosstab(train.loc[train.incident_outcome.eq(2), 'establishment_id'], quarter).reindex(
        index=counts.index, columns=[1, 2, 3], fill_value=0).div(counts)
    summary = pd.DataFrame({'Rango entre trimestres de proporción del volumen propio': volume.max(axis=1)-volume.min(axis=1),
        'Rango entre trimestres de proporción de ausencia': away.loc[selected].max(axis=1)-away.loc[selected].min(axis=1)})
    display(summary.describe(percentiles=[.25, .5, .75, .9]))
    display(Markdown(f'Heterogeneidad: **{selected.sum():,} establecimientos** con ≥30 casos de desarrollo y presencia en los tres trimestres, criterio descriptivo fijado antes del cálculo. No se publican sus identidades. Las correlaciones agregadas no prueban relaciones individuales ni capacidad predictiva. El calendario se describe por semana y mes; no se atribuyen cambios a festivos, políticas o exposición sin datos externos.'))
