"""Carga de datos y particiones del entregable 3 (validación anidada).

Reutiliza sin modificar el código del entregable 1 (``osha_deliverable_utils`` y
``osha_group_time_study``), que vive en la carpeta indicada por ``OSHA_CODE_DIR``.
El CSV original se lee desde ``OSHA_DATA_DIR`` y nunca se escribe.
"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def _find_code_dir():
    """Carpeta con el código del entregable 1: ``OSHA_CODE_DIR``, la raíz del repositorio
    (donde están ``osha_deliverable_utils.py`` y ``osha_group_time_study.py``) o la ruta local original."""
    here = Path(__file__).resolve()
    candidates = [Path(os.environ['OSHA_CODE_DIR'])] if 'OSHA_CODE_DIR' in os.environ else []
    candidates += [here.parents[2], here.parents[3] / 'Tareas' / 'tarea' / 'osha']
    for candidate in candidates:
        if (candidate / 'osha_deliverable_utils.py').exists():
            return candidate
    raise FileNotFoundError('No se encontró osha_deliverable_utils.py; defina OSHA_CODE_DIR. Buscado en: '
                            + ', '.join(str(c) for c in candidates))


CODE_DIR = _find_code_dir()
DATA_DIR = Path(os.environ.get('OSHA_DATA_DIR', CODE_DIR))   # el CSV va junto al código si no se define
DATA_FILE = DATA_DIR / 'ITA Case Detail Data 2023 through 12-31-2023OIICS.csv'

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from osha_deliverable_utils import RAW_FEATURES, LABELS, SEED, prepare_features  # noqa: E402
from osha_group_time_study import (  # noqa: E402
    assign_test_groups, _fold_data, WINDOWS, GAP_DAYS, CUTOFF,
)

INNER_SEED = SEED + 1000   # semilla del bucle interno, distinta de la del externo (42)
N_INNER = 3
SEARCH_FRACTION = 0.25

# Conteos del entregable 1; se verifican al cargar.
EXPECTED = {'dev_rows': 269516, 'dev_deaths': 69, 'test_rows': 127391, 'test_deaths': 39}


def load_data(path=DATA_FILE):
    """Lee el CSV y separa desarrollo y test igual que el entregable 1.

    Parameters
    ----------
    path : str or Path
        Ruta al CSV original de OSHA ITA 2023.

    Returns
    -------
    d_dev : pandas.DataFrame
        Todos los casos 2023 de establecimientos no reservados (todas las fechas);
        es la entrada que ``_fold_data`` espera para construir los pliegues externos.
    dev : pandas.DataFrame
        Desarrollo: establecimientos no reservados, fecha < 2023-09-24.
    test : pandas.DataFrame
        Test: establecimientos reservados, oct a dic. No se usa hasta el final.
    """
    columns = RAW_FEATURES + ['id', 'establishment_id', 'incident_outcome', 'date_of_incident']
    d = pd.read_csv(path, usecols=columns, dtype=str, encoding='latin1')
    d['date'] = pd.to_datetime(d.date_of_incident, format='%m/%d/%Y', errors='coerce')
    d = d.loc[d.date.dt.year.eq(2023)].copy()
    held, _ = assign_test_groups(d.establishment_id)
    is_held = d.establishment_id.isin(held)
    cutoff_train = CUTOFF - pd.Timedelta(days=GAP_DAYS)
    dev = d.loc[~is_held & d.date.lt(cutoff_train)]
    test = d.loc[is_held & d.date.ge(CUTOFF)]
    found = {'dev_rows': len(dev), 'dev_deaths': int(dev.incident_outcome.eq('1').sum()),
             'test_rows': len(test), 'test_deaths': int(test.incident_outcome.eq('1').sum())}
    assert found == EXPECTED, f'La partición no coincide con el entregable 1: {found}'
    return d.loc[~is_held], dev, test


def outer_folds(d_dev):
    """Genera los 3 pliegues externos hacia adelante del entregable 1.

    Yields
    ------
    fold : int
    train, valid : pandas.DataFrame
        Establecimientos disjuntos y brecha de 7 días entre ambos.
    """
    for fold, (start, end) in enumerate(WINDOWS, 1):
        train, valid = _fold_data(d_dev, fold, start, end)
        yield fold, train, valid


def search_sample(train, fraction=SEARCH_FRACTION, seed=INNER_SEED):
    """Muestra estratificada por clase para la búsqueda de hiperparámetros.

    Conserva todas las muertes (clase 1) y toma ``fraction`` de cada una de las
    demás clases. Las filas se eligen al azar con semilla fija.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for label, part in train.groupby('incident_outcome'):
        if label == '1':
            keep.append(part.index.to_numpy())
        else:
            n = int(np.ceil(len(part) * fraction))
            keep.append(rng.choice(part.index.to_numpy(), size=n, replace=False))
    return train.loc[np.sort(np.concatenate(keep))]


FOLD_DIR = Path(__file__).resolve().parents[1] / 'results' / 'folds'
KEEP = RAW_FEATURES + ['id', 'establishment_id', 'incident_outcome', 'date']


def materialize_folds(out_dir=FOLD_DIR):
    """Guarda en parquet el entrenamiento, la validación y la muestra de búsqueda de cada pliegue externo.

    El conjunto de test no se guarda: no se usa hasta el final.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d_dev, _, _ = load_data()
    for fold, train, valid in outer_folds(d_dev):
        sample = search_sample(train)
        assert not set(train.establishment_id) & set(valid.establishment_id)
        assert set(sample.index) <= set(train.index)
        for name, part in [('train', train), ('valid', valid), ('search', sample)]:
            part[KEEP].reset_index(drop=True).to_parquet(out_dir / f'fold{fold}_{name}.parquet', index=False)


def read_fold(fold, part, fold_dir=FOLD_DIR):
    """Lee un pliegue guardado: ``part`` es 'train', 'valid' o 'search'.

    Parquet devuelve los vacíos como ``None``; se convierten a ``NaN`` como en la lectura del CSV.
    """
    frame = pd.read_parquet(Path(fold_dir) / f'fold{fold}_{part}.parquet')
    text = frame.columns.drop('date')
    frame[text] = frame[text].astype(object).where(frame[text].notna(), np.nan)
    return frame


def inner_splits(train, seed=INNER_SEED):
    """Pliegues internos agrupados por establecimiento y estratificados por clase.

    Returns
    -------
    list of (ndarray, ndarray)
        Índices posicionales (entrenamiento, validación) dentro de ``train``.
    """
    cv = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    y = train.incident_outcome.astype(int).to_numpy()
    return list(cv.split(np.zeros(len(train)), y, train.establishment_id.to_numpy()))
