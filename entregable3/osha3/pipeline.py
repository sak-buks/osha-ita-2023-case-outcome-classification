"""Construcción del pipeline completo de un pliegue (sin fugas).

Orden: prepare_features (sin estadísticas aprendidas, igual que el entregable 1) →
imputación → one-hot (con ``min_frequency``) → balanceo → [Nystroem] → clasificador.
Todo lo que aprende de los datos se ajusta solo con el entrenamiento que recibe ``fit``.
Los muestreadores de imblearn solo actúan en ``fit``; ``predict`` no remuestrea.
"""
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .data import SEED, prepare_features

BALANCES = ('none', 'smote', 'adasyn', 'class_weight')


def make_sampler(balance, y_fit):
    """Crea SMOTE/ADASYN con k = min(5, muertes - 1); si hay menos de 2 muertes, RandomOverSampler.

    Ambos sobremuestrean solo la clase minoritaria (Death), como los define el PDF; con la
    estrategia por defecto ADASYN falla porque las clases 2-4 ya están casi equilibradas.

    Returns
    -------
    sampler or None, str
        Muestreador y nota para el registro.
    """
    if balance not in ('smote', 'adasyn'):
        return None, ''
    deaths = int(np.sum(np.asarray(y_fit) == 1))
    if deaths < 2:
        return RandomOverSampler(sampling_strategy='minority', random_state=SEED), f'RandomOverSampler (muertes={deaths})'
    k = min(5, deaths - 1)
    if balance == 'smote':
        return SMOTE(sampling_strategy='minority', k_neighbors=k, random_state=SEED), f'k={k} (muertes={deaths})'
    return ADASYN(sampling_strategy='minority', n_neighbors=k, random_state=SEED), f'k={k} (muertes={deaths})'


def complete_params(spec, params):
    """Completa con los valores por defecto y elimina parámetros condicionales inactivos."""
    full = dict(spec.defaults)
    full.update(params)
    for name, dist in spec.space.items():
        if dist.when is not None and full.get(dist.when[0]) != dist.when[1]:
            full.pop(name, None)
    return full


def build_pipeline(spec, params, balance, y_fit, n_jobs=1, device='cpu'):
    """Devuelve (pipeline, fit_params, nota) para un modelo, un balanceo y unos hiperparámetros.

    Parameters
    ----------
    spec : ModelSpec
    params : dict
        Hiperparámetros (se completan con ``spec.defaults``).
    balance : {'none', 'smote', 'adasyn', 'class_weight'}
    y_fit : array-like
        Clases del conjunto con el que se ajustará el pipeline (para fijar k y los pesos).
    """
    if balance == 'class_weight' and spec.weight_mode is None:
        raise ValueError(f'{spec.name} no admite class_weight (combinación "no aplica").')
    p = complete_params(spec, params)
    mf = p.get('min_frequency')
    steps = [
        ('features', FunctionTransformer(prepare_features, validate=False)),
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=mf)),
    ]
    sampler, note = make_sampler(balance, y_fit)
    if sampler is not None:
        steps.append(('sampler', sampler))
    use_cw = balance == 'class_weight' and spec.weight_mode == 'class_weight'
    steps += spec.build(p, 'balanced' if use_cw else None, n_jobs, device)
    fit_params = {}
    if balance == 'class_weight' and spec.weight_mode == 'sample_weight':
        fit_params['clf__sample_weight'] = compute_sample_weight('balanced', np.asarray(y_fit))
        note = 'compute_sample_weight(balanced)'
    elif use_cw:
        note = "class_weight='balanced'"
    return Pipeline(steps), fit_params, note
