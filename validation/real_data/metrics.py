"""
HelioMesh â€” Validation Metrics
================================
Computes classification metrics for snapshot and temporal validation runs.
"""

from collections import Counter


def classification_report(y_true: list, y_pred: list,
                           labels: list | None = None) -> dict:
    """
    Compute per-class precision, recall, F1 and overall accuracy / macro F1.
    Works for any string labels.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    results = {}
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        results[lbl] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   tp + fn,
        }

    accuracy  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1  = sum(v["f1"] for v in results.values()) / len(results) if results else 0.0

    return {
        "accuracy":  round(accuracy,  4),
        "macro_f1":  round(macro_f1,  4),
        "n_samples": len(y_true),
        "per_class": results,
    }


def confusion_matrix(y_true: list, y_pred: list,
                     labels: list | None = None) -> dict:
    """Returns confusion matrix as {true_label: {pred_label: count}}."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in labels:
            matrix[t][p] += 1
    return matrix


def label_distribution(labels: list) -> dict:
    """Return {label: count, ...} sorted by label name."""
    c = Counter(labels)
    return {k: c[k] for k in sorted(c)}

