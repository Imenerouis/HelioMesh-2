"""
evaluate.py — OPS-SAT-AD evaluation runner.

Evaluates all trained baselines on the official TEST partition.
Saves results to validation/results/opssat_ad_real_metrics.json.

Usage:
  python -m validation.real_spacecraft.opssat_ad.evaluate
"""

import json
import os
import sys
import numpy as np

from .train import train_all_models
from .metrics import compute_all_metrics

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'results',
)


def run_evaluation(verbose: bool = True) -> dict:
    """
    Train all models, evaluate on test set, return full results dict.
    """
    if verbose:
        print("Training models on official TRAIN partition...")
    models = train_all_models()

    X_test  = models['X_test']
    y_test  = models['y_test']

    results = {}

    # ---- Majority baseline ----
    maj_class = models['majority_baseline']['predict_class']
    y_pred_maj = np.full(len(y_test), maj_class, dtype=int)
    results['majority_baseline'] = compute_all_metrics(
        y_test, y_pred_maj, y_prob=None, label='majority_baseline'
    )
    if verbose:
        print(f"  Majority baseline (predict={maj_class}): "
              f"acc={results['majority_baseline']['accuracy']:.4f}, "
              f"f1={results['majority_baseline']['f1']:.4f}")

    # ---- Random Forest ----
    rf = models['random_forest']
    y_pred_rf  = rf.predict(X_test)
    y_prob_rf  = rf.predict_proba(X_test)[:, 1]
    results['random_forest'] = compute_all_metrics(
        y_test, y_pred_rf, y_prob=y_prob_rf, label='random_forest'
    )
    if verbose:
        r = results['random_forest']
        print(f"  Random Forest: acc={r['accuracy']:.4f}, "
              f"bal_acc={r['balanced_accuracy']:.4f}, "
              f"f1={r['f1']:.4f}, "
              f"roc_auc={r['roc_auc']:.4f}, "
              f"pr_auc={r['pr_auc']:.4f}")

    # ---- Isolation Forest ----
    iso = models['isolation_forest']
    # IsolationForest: predict returns 1 (inlier) or -1 (outlier)
    iso_raw   = iso.predict(X_test)          # 1 or -1
    y_pred_iso = (iso_raw == -1).astype(int)  # 1=anomaly, 0=nominal
    iso_scores = iso.score_samples(X_test)    # lower = more anomalous
    y_prob_iso = 1.0 - (iso_scores - iso_scores.min()) / \
                       (iso_scores.max() - iso_scores.min() + 1e-12)
    results['isolation_forest'] = compute_all_metrics(
        y_test, y_pred_iso, y_prob=y_prob_iso, label='isolation_forest'
    )
    if verbose:
        r = results['isolation_forest']
        print(f"  Isolation Forest: acc={r['accuracy']:.4f}, "
              f"bal_acc={r['balanced_accuracy']:.4f}, "
              f"f1={r['f1']:.4f}, "
              f"roc_auc={r['roc_auc']:.4f}, "
              f"pr_auc={r['pr_auc']:.4f}")

    # ---- Feature importances (RF) ----
    importances = {
        name: float(imp)
        for name, imp in zip(X_test.columns, rf.feature_importances_)
    }
    top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    results['rf_feature_importances_top10'] = dict(top10)

    results['meta'] = {
        'n_test_samples'    : int(len(y_test)),
        'n_test_positive'   : int(y_test.sum()),
        'n_test_negative'   : int((y_test == 0).sum()),
        'contamination_used': float(models['contamination']),
        'rf_n_estimators'   : 200,
        'iso_n_estimators'  : 200,
    }

    return results


def save_results(results: dict) -> str:
    """Save evaluation results to JSON and return the output path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'opssat_ad_real_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    return out_path


if __name__ == '__main__':
    results = run_evaluation(verbose=True)
    path = save_results(results)
    print(f"\nResults saved to: {path}")
