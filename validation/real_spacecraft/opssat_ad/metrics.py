"""
metrics.py — Evaluation metric utilities for OPS-SAT-AD.

All metrics are computed from actual predictions — never imputed.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_all_metrics(y_true, y_pred, y_prob=None, label: str = "") -> dict:
    """
    Compute full metric suite for binary classification.

    Parameters
    ----------
    y_true : array-like of int (0/1)
    y_pred : array-like of int (0/1)
    y_prob : array-like of float (probability of class 1), optional
    label  : identifier string for this result set

    Returns
    -------
    dict with all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        'label'             : label,
        'n_samples'         : int(len(y_true)),
        'n_positive'        : int(y_true.sum()),
        'n_negative'        : int((y_true == 0).sum()),
        'accuracy'          : float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy' : float(balanced_accuracy_score(y_true, y_pred)),
        'precision'         : float(precision_score(y_true, y_pred, zero_division=0)),
        'recall'            : float(recall_score(y_true, y_pred, zero_division=0)),
        'f1'                : float(f1_score(y_true, y_pred, zero_division=0)),
        'mcc'               : float(matthews_corrcoef(y_true, y_pred)),
        'tp'                : int(tp),
        'tn'                : int(tn),
        'fp'                : int(fp),
        'fn'                : int(fn),
        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else None,
    }

    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            result['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            result['roc_auc'] = None
        try:
            result['pr_auc'] = float(average_precision_score(y_true, y_prob))
        except Exception:
            result['pr_auc'] = None
    else:
        result['roc_auc'] = None
        result['pr_auc']  = None

    return result


def majority_baseline_metrics(y_true) -> dict:
    """Metrics for a majority-class (always-predict-0) baseline."""
    y_true = np.array(y_true)
    majority = int(np.bincount(y_true).argmax())
    y_pred = np.full_like(y_true, majority)
    return compute_all_metrics(y_true, y_pred, label='majority_baseline')


def positive_rate_baseline_metrics(y_true) -> dict:
    """
    Metrics for a random baseline that predicts positive at the observed rate.
    Uses a fixed seed for reproducibility.
    """
    y_true = np.array(y_true)
    rng = np.random.default_rng(42)
    rate = y_true.mean()
    y_pred = (rng.random(len(y_true)) < rate).astype(int)
    return compute_all_metrics(y_true, y_pred, label='random_rate_baseline')
