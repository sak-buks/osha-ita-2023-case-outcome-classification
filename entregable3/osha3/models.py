"""Registro de modelos: estimador, espacio de búsqueda, rejilla y soporte de pesos (D6).

Añadir un modelo (p. ej., el modelo original, D8) es añadir una entrada a ``REGISTRY``.

Rangos, según el material de clase:
- KNN (cap. 2): n_neighbors entre 1 y 50; weights uniform/distance. No se ajusta la métrica:
  con one-hot cada fila tiene exactamente 8 unos, y las distancias euclídea, Manhattan y
  coseno ordenan igual a los vecinos (todas dependen solo de cuántas variables difieren).
- BernoulliNB (cap. 4): el one-hot es binario. ``binarize=0.5`` para que los casos
  sintéticos de SMOTE/ADASYN (valores fraccionarios) se redondeen a la categoría más cercana.
- Regresión logística L1/L2: C en escala logarítmica. Solver ``liblinear``, el que usa el
  cap. 3 para L1 (admite L1 y L2; en multiclase ajusta uno contra el resto). ``saga`` se
  descartó: no convergía en 1.000 iteraciones y tardaba ~40 s por evaluación (prueba de humo).
- Árbol (cap. 5): max_depth, min_samples_leaf, ccp_alpha (poda), criterion.
- Random Forest y XGBoost: tabla de hiperparámetros del cap. 5.
  XGBoost sin parada temprana: el conjunto de parada saldría del entrenamiento del pliegue y,
  con SMOTE/ADASYN, contendría casos sintéticos; ``n_estimators`` se busca como hiperparámetro.
- SVM (cap. 6): el kernel exacto es cuadrático en n. Se usa LinearSVC solo o Nystroem
  (m ∈ {100, 300, 1000}) + LinearSVC, calibrado con CalibratedClassifierCV (sigmoide o
  isotónica, cv=5 como en el material) para obtener probabilidades.
- ``min_frequency`` del one-hot (D4) es una fracción del número de filas de entrenamiento,
  para que signifique lo mismo en la muestra de búsqueda (25 %) y en el reentrenamiento (100 %).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .data import LABELS, SEED


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    log: bool = False
    when: Optional[tuple] = None   # (parámetro, valor): solo existe si se cumple


@dataclass(frozen=True)
class Int:
    low: int
    high: int
    log: bool = False
    when: Optional[tuple] = None


@dataclass(frozen=True)
class Cat:
    choices: tuple
    when: Optional[tuple] = None


MIN_FREQUENCY = Cat((None, 1e-4, 5e-4, 2e-3))


@dataclass
class ModelSpec:
    """Entrada del registro.

    Attributes
    ----------
    name : str
    build : callable
        ``build(params, class_weight, n_jobs, device)`` devuelve la lista de pasos finales
        del pipeline; el último se llama ``'clf'``.
    space : dict
        Espacio de búsqueda (Random, Bayesiano y Genético).
    grid : list of dict
        Rejilla (``ParameterGrid``) de como máximo 30 puntos.
    defaults : dict
        Valores para los parámetros que la rejilla no fija.
    weight_mode : {'class_weight', 'sample_weight', None}
        Cómo se aplica ``class_weight='balanced'``; None = no aplica (KNN).
    """
    name: str
    build: Callable
    space: dict
    grid: list
    defaults: dict
    weight_mode: Optional[str]
    notes: str = ''
    fixed: dict = field(default_factory=dict)


class XGBShift(XGBClassifier):
    """XGBClassifier con clases 1..4 (XGBoost exige 0..K-1)."""

    def fit(self, X, y, **kwargs):
        self._shifted = False    # durante el ajuste XGBoost exige ver las clases 0..3
        super().fit(X, np.asarray(y) - 1, **kwargs)
        self._shifted = True
        return self

    @property
    def classes_(self):
        return np.asarray(LABELS) if getattr(self, '_shifted', False) else np.arange(self.n_classes_)

    def predict(self, X, **kwargs):
        return np.asarray(LABELS)[self.predict_proba(X).argmax(axis=1)]


def _knn(p, class_weight, n_jobs, device):
    return [('clf', KNeighborsClassifier(n_neighbors=p['n_neighbors'], weights=p['weights'],
                                         algorithm='brute', n_jobs=n_jobs))]


def _bnb(p, class_weight, n_jobs, device):
    return [('clf', BernoulliNB(alpha=p['alpha'], fit_prior=p['fit_prior'], binarize=0.5))]


def _logreg(p, class_weight, n_jobs, device):
    return [('clf', LogisticRegression(C=p['C'], penalty=p['penalty'], solver='liblinear', max_iter=1000,
                                       class_weight=class_weight, random_state=SEED))]


def _tree(p, class_weight, n_jobs, device):
    return [('clf', DecisionTreeClassifier(max_depth=p['max_depth'], min_samples_leaf=p['min_samples_leaf'],
                                           ccp_alpha=p['ccp_alpha'], criterion=p['criterion'],
                                           class_weight=class_weight, random_state=SEED))]


def _rf(p, class_weight, n_jobs, device):
    return [('clf', RandomForestClassifier(
        n_estimators=p['n_estimators'], max_depth=p['max_depth'], min_samples_split=p['min_samples_split'],
        min_samples_leaf=p['min_samples_leaf'], max_features=p['max_features'], bootstrap=p['bootstrap'],
        criterion=p['criterion'], class_weight=class_weight, n_jobs=n_jobs, random_state=SEED))]


def _xgb(p, class_weight, n_jobs, device):
    return [('clf', XGBShift(
        n_estimators=p['n_estimators'], learning_rate=p['learning_rate'], max_depth=p['max_depth'],
        subsample=p['subsample'], colsample_bytree=p['colsample_bytree'], gamma=p['gamma'],
        min_child_weight=p['min_child_weight'], reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
        tree_method='hist', device=device, n_jobs=n_jobs, random_state=SEED, eval_metric='mlogloss'))]


def _svm(p, class_weight, n_jobs, device):
    base = LinearSVC(C=p['C'], dual=False, class_weight=class_weight, max_iter=5000, random_state=SEED)
    steps = []
    if p['kernel'] == 'nystroem':
        steps.append(('nystroem', Nystroem(kernel='rbf', gamma=p['gamma'], n_components=p['n_components'],
                                           random_state=SEED)))
    steps.append(('clf', CalibratedClassifierCV(base, method=p['calibration'], cv=5)))
    return steps


def _dummy(p, class_weight, n_jobs, device):
    return [('clf', DummyClassifier(strategy='prior', random_state=SEED))]


def _logreg_e1(p, class_weight, n_jobs, device):
    return [('clf', LogisticRegression(solver='newton-cg', max_iter=1000, class_weight=class_weight,
                                       random_state=SEED))]


# Referencias (no cuentan en las 108 combinaciones): se evalúan con la misma validación anidada.
REFERENCES = {
    'dummy': ModelSpec('dummy', _dummy, space={}, grid=[{}], defaults={'min_frequency': None}, weight_mode=None,
                       notes='DummyClassifier(prior): referencia mínima.'),
    'logreg_e1': ModelSpec('logreg_e1', _logreg_e1, space={}, grid=[{}], defaults={'min_frequency': None},
                           weight_mode='class_weight',
                           notes="Modelo del entregable 1: newton-cg, L2, C=1, class_weight='balanced'."),
}

REGISTRY = {
    'knn': ModelSpec(
        'knn', _knn,
        space={'n_neighbors': Int(1, 50), 'weights': Cat(('uniform', 'distance')), 'min_frequency': MIN_FREQUENCY},
        grid=[{'n_neighbors': [1, 5, 15, 30, 50], 'weights': ['uniform', 'distance'],
               'min_frequency': [None, 2e-3]}],
        defaults={'n_neighbors': 5, 'weights': 'uniform', 'min_frequency': None},
        weight_mode=None),
    'naive_bayes': ModelSpec(
        'naive_bayes', _bnb,
        space={'alpha': Float(1e-3, 10, log=True), 'fit_prior': Cat((True, False)), 'min_frequency': MIN_FREQUENCY},
        grid=[{'alpha': [1e-3, 1e-2, 1e-1, 1, 10], 'fit_prior': [True, False], 'min_frequency': [None, 2e-3]}],
        defaults={'alpha': 1.0, 'fit_prior': True, 'min_frequency': None},
        weight_mode='sample_weight'),
    'logreg': ModelSpec(
        'logreg', _logreg,
        space={'C': Float(1e-3, 1e2, log=True), 'penalty': Cat(('l1', 'l2')), 'min_frequency': MIN_FREQUENCY},
        grid=[{'C': [1e-3, 1e-2, 1e-1, 1, 10], 'penalty': ['l1', 'l2'], 'min_frequency': [None, 2e-3]}],
        defaults={'C': 1.0, 'penalty': 'l2', 'min_frequency': None},
        weight_mode='class_weight'),
    'tree': ModelSpec(
        'tree', _tree,
        space={'max_depth': Int(3, 40), 'min_samples_leaf': Int(1, 200, log=True),
               'ccp_alpha': Float(1e-6, 1e-2, log=True), 'criterion': Cat(('gini', 'entropy')),
               'min_frequency': MIN_FREQUENCY},
        grid=[{'max_depth': [5, 10, 20, 40], 'min_samples_leaf': [1, 20], 'ccp_alpha': [1e-5, 1e-3]}],
        defaults={'max_depth': 15, 'min_samples_leaf': 1, 'ccp_alpha': 1e-6, 'criterion': 'gini', 'min_frequency': None},
        weight_mode='class_weight'),
    'random_forest': ModelSpec(
        'random_forest', _rf,
        space={'n_estimators': Int(100, 500), 'max_depth': Int(5, 60), 'min_samples_split': Int(2, 50, log=True),
               'min_samples_leaf': Int(1, 50, log=True), 'max_features': Cat(('sqrt', 'log2', 0.1)),
               'bootstrap': Cat((True, False)), 'criterion': Cat(('gini', 'entropy')), 'min_frequency': MIN_FREQUENCY},
        grid=[{'n_estimators': [100, 300], 'max_depth': [10, 40], 'min_samples_leaf': [1, 5],
               'max_features': ['sqrt', 'log2']}],
        defaults={'n_estimators': 100, 'max_depth': 40, 'min_samples_split': 2, 'min_samples_leaf': 1,
                  'max_features': 'sqrt', 'bootstrap': True, 'criterion': 'gini', 'min_frequency': None},
        weight_mode='class_weight'),
    'xgboost': ModelSpec(
        'xgboost', _xgb,
        space={'n_estimators': Int(100, 600), 'learning_rate': Float(0.01, 0.3, log=True), 'max_depth': Int(2, 10),
               'subsample': Float(0.5, 1.0), 'colsample_bytree': Float(0.3, 1.0), 'gamma': Float(0.0, 5.0),
               'min_child_weight': Float(0.5, 20, log=True), 'reg_alpha': Float(1e-4, 10, log=True),
               'reg_lambda': Float(1e-3, 10, log=True), 'min_frequency': MIN_FREQUENCY},
        grid=[{'learning_rate': [0.05, 0.1, 0.3], 'max_depth': [3, 6, 9], 'n_estimators': [100, 300]}],
        defaults={'n_estimators': 100, 'learning_rate': 0.3, 'max_depth': 6, 'subsample': 1.0,
                  'colsample_bytree': 1.0, 'gamma': 0.0, 'min_child_weight': 1.0, 'reg_alpha': 0.0,
                  'reg_lambda': 1.0, 'min_frequency': None},
        weight_mode='sample_weight'),
    'svm': ModelSpec(
        'svm', _svm,
        space={'kernel': Cat(('linear', 'nystroem')), 'C': Float(1e-3, 1e2, log=True),
               'n_components': Cat((100, 300, 1000), when=('kernel', 'nystroem')),
               'gamma': Float(1e-3, 1.0, log=True, when=('kernel', 'nystroem')),
               'calibration': Cat(('sigmoid', 'isotonic')), 'min_frequency': MIN_FREQUENCY},
        grid=[{'kernel': ['linear'], 'C': [1e-2, 1e-1, 1, 10], 'calibration': ['sigmoid', 'isotonic']},
              {'kernel': ['nystroem'], 'n_components': [100, 300, 1000], 'gamma': [0.01, 0.1], 'C': [0.1, 10]}],
        defaults={'kernel': 'linear', 'C': 1.0, 'calibration': 'sigmoid', 'min_frequency': None},
        weight_mode='class_weight'),
}

# Modelos del curso + referencias, para buscar por nombre.
ALL_SPECS = {**REGISTRY, **REFERENCES}
