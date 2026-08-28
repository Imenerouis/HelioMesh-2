"""
channel_loo.py — Cross-channel leave-one-out generalization experiment.

Scientific purpose
------------------
Tests whether the anomaly-detection *training procedure* (Random Forest on
segment-level statistical features) generalizes to OPS-SAT telemetry channels
NOT seen during training.

This is distinct from the frozen Phase 2 benchmark:
  - Phase 2:  RF trained on official train split (mixed channels) → evaluated
              on official test split (mixed channels).
  - This LOO: RF trained on N-1 channels → evaluated on the held-out channel.

The frozen ml/ artifacts are NEVER touched.  This script creates new models
inside each LOO fold and discards them after evaluation; nothing is persisted.

Outputs (validation/results/)
------------------------------
  channel_loo_results.json  — per-fold and aggregate metrics
  CHANNEL_LOO_REPORT.md     — human-readable report

Reproducibility
---------------
  random_state=42 throughout.
  StandardScaler fitted on training channels only in each fold.
  Channel identity is NOT used as a feature in any LOO fold.

Usage
-----
  python -m validation.real_spacecraft.opssat_ad.channel_loo
"""

import json
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    matthews_corrcoef, average_precision_score, balanced_accuracy_score,
)

from .loader import load_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'results',
)

# ---------------------------------------------------------------------------
# Feature definition
# NOTE: channel identity is deliberately excluded — this tests cross-channel
# generalization without the model knowing which channel it is looking at.
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    'duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew',
    'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks',
    'diff_peaks', 'diff2_peaks', 'diff_var', 'diff2_var',
    'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len',
    # sampling_5s included — it is a segment-level feature, not a channel id
    'sampling_5s',
]

# Feature scale regime classification (based on median var, audited)
# CADC0872/0873/0874: var ~ 1e-10  (scale regime A)
# CADC0884/0886/0888/0890/0892/0894: var ~ 1e-2 to 1e-1  (scale regime B)
REGIME_A = {'CADC0872', 'CADC0873', 'CADC0874'}
REGIME_B = {'CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894'}

# Minimum anomalous samples in test fold to include in primary aggregate
MIN_ANOMALIES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_X_y(df: pd.DataFrame) -> tuple:
    """Build feature matrix and label vector from a channel-subset DataFrame."""
    df = df.copy()
    df['sampling_5s'] = (df['sampling'] == 5).astype(int)
    X = df[FEATURE_COLS].copy()
    y = df['anomaly'].copy()
    return X, y


def _loo_fold(df: pd.DataFrame, held_out_channel: str) -> dict:
    """
    Run one LOO fold: train on all channels except held_out_channel,
    evaluate on held_out_channel.  All preprocessing fitted on train only.

    Returns a dict of evaluation metrics plus fold metadata.
    """
    train_df = df[df['channel'] != held_out_channel].copy()
    test_df  = df[df['channel'] == held_out_channel].copy()

    X_train, y_train = _build_X_y(train_df)
    X_test,  y_test  = _build_X_y(test_df)

    # Scale: fit on training channels only
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index,
    )

    # contamination = training anomaly rate (observed, not tuned on test)
    contamination = float(y_train.mean())

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    y_pred  = rf.predict(X_test_s)
    y_prob  = rf.predict_proba(X_test_s)[:, 1]
    y_true  = y_test.values

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    metrics = {
        'held_out_channel' : held_out_channel,
        'scale_regime'     : 'A' if held_out_channel in REGIME_A else 'B',
        'n_train'          : int(len(y_train)),
        'n_train_anomaly'  : int(y_train.sum()),
        'n_train_nominal'  : int((y_train == 0).sum()),
        'n_test'           : int(len(y_true)),
        'n_test_anomaly'   : n_pos,
        'n_test_nominal'   : n_neg,
        'anomaly_base_rate': float(y_true.mean()),
        'precision'        : float(precision_score(y_true, y_pred, zero_division=0)),
        'recall'           : float(recall_score(y_true, y_pred, zero_division=0)),
        'f1'               : float(f1_score(y_true, y_pred, zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'mcc'              : float(matthews_corrcoef(y_true, y_pred)),
    }

    # AUC metrics — only valid when both classes present
    if n_pos > 0 and n_neg > 0:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        metrics['pr_auc']  = float(average_precision_score(y_true, y_prob))
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc']  = None

    # Primary aggregate eligibility
    metrics['in_primary_aggregate'] = (n_pos >= MIN_ANOMALIES)
    if not metrics['in_primary_aggregate']:
        metrics['exclusion_reason'] = (
            f"n_test_anomaly={n_pos} < MIN_ANOMALIES={MIN_ANOMALIES}"
        )

    return metrics


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_channel_loo(verbose: bool = True) -> dict:
    """
    Run all 9 LOO folds. Returns full results dict.
    """
    df = load_dataset()
    channels = sorted(df['channel'].unique())

    if verbose:
        print(f"\n{'='*60}")
        print("Channel LOO Generalization Experiment")
        print("Anomaly-detection APPROACH (not frozen model) evaluated")
        print(f"{'='*60}")
        print(f"Total channels: {len(channels)}")
        print(f"Total segments: {len(df)}  |  Anomalies: {df.anomaly.sum()}")
        print(f"Features: {len(FEATURE_COLS)} (no channel identity)")
        print(f"Min anomalies for primary aggregate: {MIN_ANOMALIES}")
        print()

    fold_results = []
    for ch in channels:
        fold = _loo_fold(df, ch)
        fold_results.append(fold)
        if verbose:
            flag = "" if fold['in_primary_aggregate'] else "  [EXCLUDED from aggregate]"
            auc_str = f"{fold['roc_auc']:.4f}" if fold['roc_auc'] is not None else 'N/A'
            print(
                f"  {ch} (regime {fold['scale_regime']}): "
                f"n_test={fold['n_test']} anom={fold['n_test_anomaly']} "
                f"F1={fold['f1']:.4f}  AUC={auc_str}"
                f"{flag}"
            )

    # ---- Primary aggregate (sufficient anomalies) ----
    primary = [f for f in fold_results if f['in_primary_aggregate']]
    excluded = [f for f in fold_results if not f['in_primary_aggregate']]

    def _mean(key):
        vals = [f[key] for f in primary if f[key] is not None]
        return float(np.mean(vals)) if vals else None

    regime_a = [f for f in primary if f['scale_regime'] == 'A']
    regime_b = [f for f in primary if f['scale_regime'] == 'B']

    aggregate = {
        'n_folds_primary'    : len(primary),
        'n_folds_excluded'   : len(excluded),
        'excluded_channels'  : [f['held_out_channel'] for f in excluded],
        'mean_f1'            : _mean('f1'),
        'mean_roc_auc'       : _mean('roc_auc'),
        'mean_pr_auc'        : _mean('pr_auc'),
        'mean_precision'     : _mean('precision'),
        'mean_recall'        : _mean('recall'),
        'mean_mcc'           : _mean('mcc'),
        'mean_balanced_acc'  : _mean('balanced_accuracy'),
        'regime_A': {
            'channels'       : [f['held_out_channel'] for f in regime_a],
            'n_folds'        : len(regime_a),
            'mean_f1'        : float(np.mean([f['f1'] for f in regime_a])) if regime_a else None,
            'mean_roc_auc'   : float(np.mean([f['roc_auc'] for f in regime_a if f['roc_auc'] is not None])) if regime_a else None,
        },
        'regime_B': {
            'channels'       : [f['held_out_channel'] for f in regime_b],
            'n_folds'        : len(regime_b),
            'mean_f1'        : float(np.mean([f['f1'] for f in regime_b])) if regime_b else None,
            'mean_roc_auc'   : float(np.mean([f['roc_auc'] for f in regime_b if f['roc_auc'] is not None])) if regime_b else None,
        },
    }

    if verbose:
        print()
        print(f"  Primary aggregate ({len(primary)} folds):")
        print(f"    Mean F1      = {aggregate['mean_f1']:.4f}")
        print(f"    Mean ROC-AUC = {aggregate['mean_roc_auc']:.4f}")
        ra_auc = f"{aggregate['regime_A']['mean_roc_auc']:.4f}" if aggregate['regime_A']['mean_roc_auc'] is not None else 'N/A'
        rb_auc = f"{aggregate['regime_B']['mean_roc_auc']:.4f}" if aggregate['regime_B']['mean_roc_auc'] is not None else 'N/A'
        ra_f1 = f"{aggregate['regime_A']['mean_f1']:.4f}" if aggregate['regime_A']['mean_f1'] is not None else 'N/A'
        rb_f1 = f"{aggregate['regime_B']['mean_f1']:.4f}" if aggregate['regime_B']['mean_f1'] is not None else 'N/A'
        print(f"    Regime A (var~1e-10): F1={ra_f1}  AUC={ra_auc}")
        print(f"    Regime B (var~1e-1):  F1={rb_f1}  AUC={rb_auc}")
        print()
        if excluded:
            print(f"  Excluded (< {MIN_ANOMALIES} test anomalies): {[f['held_out_channel'] for f in excluded]}")

    results = {
        'experiment'  : 'channel_leave_one_out',
        'description' : (
            'Cross-channel LOO generalization of the anomaly-detection training '
            'procedure. Each fold trains a new RF on N-1 channels and evaluates '
            'on the held-out channel. The frozen Phase 2 benchmark is unaffected.'
        ),
        'design_notes': {
            'frozen_artifacts_touched': False,
            'channel_identity_as_feature': False,
            'scaler_fit_on_train_only': True,
            'hyperparameters_tuned_on_test': False,
            'min_anomalies_for_primary_agg': MIN_ANOMALIES,
            'random_state': 42,
            'n_estimators': 200,
            'class_weight': 'balanced',
        },
        'per_channel' : fold_results,
        'aggregate'   : aggregate,
    }

    return results


def save_results(results: dict) -> str:
    """Save LOO results to JSON. Does NOT overwrite any existing Phase 2 artifact."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'channel_loo_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    return out_path


if __name__ == '__main__':
    results = run_channel_loo(verbose=True)
    path = save_results(results)
    print(f"Results saved to: {path}")
