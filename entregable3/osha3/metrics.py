"""Métricas del bucle externo (D3) e intervalo de Wilson."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)

from .data import LABELS

CLASS_NAMES = {1: 'death', 2: 'days_away', 3: 'transfer', 4: 'other'}


def wilson_interval(successes, n, z=1.959963984540054):
    """Intervalo de Wilson al 95 % para una proporción (p. ej., recall de Death)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(center - half), float(center + half))


def classification_metrics(y, proba):
    """Calcula las métricas principales y secundarias a partir de probabilidades.

    Parameters
    ----------
    y : array-like of int
        Clases verdaderas en {1, 2, 3, 4}.
    proba : ndarray of shape (n, 4)
        Probabilidades por clase en el orden de ``LABELS``.

    Returns
    -------
    dict
        Balanced accuracy (primaria), F1 macro, AUC macro uno contra el resto, MCC,
        log-loss, Brier, accuracy, precisión macro, y recall/precisión/F1 por clase.
    """
    y = np.asarray(y)
    pred = np.asarray(LABELS)[proba.argmax(axis=1)]
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    y_bin = (y[:, None] == np.asarray(LABELS)).astype(float)
    out = {
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0)),
        'mcc': float(matthews_corrcoef(y, pred)),
        'log_loss': float(log_loss(y, np.clip(proba, 1e-15, 1), labels=LABELS)),
        'brier': float(np.mean(np.sum((proba - y_bin) ** 2, axis=1))),
        'accuracy': float(accuracy_score(y, pred)),
        'precision_macro': float(prec.mean()),
        'recall_macro': float(rec.mean()),
    }
    try:
        out['roc_auc_macro_ovr'] = float(roc_auc_score(y, proba, labels=LABELS, multi_class='ovr', average='macro'))
    except ValueError:
        out['roc_auc_macro_ovr'] = np.nan
    for i, k in enumerate(LABELS):
        name = CLASS_NAMES[k]
        out[f'recall_{name}'] = float(rec[i])
        out[f'precision_{name}'] = float(prec[i])
        out[f'f1_{name}'] = float(f1[i])
        out[f'support_{name}'] = int(support[i])
    hits = int(((y == 1) & (pred == 1)).sum())
    out['death_hits'] = hits
    out['recall_death_wilson_low'], out['recall_death_wilson_high'] = wilson_interval(hits, int((y == 1).sum()))
    return out
