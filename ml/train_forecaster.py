"""
HelioMesh — Supervised Temporal Predictor Training
===================================================
Trains a Gradient Boosting classifier to answer:

    "Given the last 30 minutes of simulated telemetry (6 snapshots,
     5 min apart), will the satellite be in a critical operational
     state (SAFE_MODE or STANDBY) 30 minutes from now?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Supervised classification over a fixed look-back window.
This is NOT classical time-series forecasting (no RNN, no ARIMA).

Gradient Boosting operates on a flattened 42-feature vector:
  - 36 raw values (6 steps × 6 features per step)
  - 6 delta values (change from t0 to t5 per feature)

The delta features give the model directional awareness (trend)
without any recurrent structure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMING CLARIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each sequence spans 60 minutes total from first observation to target:

  t=0 ── t=25     INPUT window  (6 steps × 5 min = 30 min look-back)
  t=25 ── t=55    Horizon gap   (6 steps × 5 min = 30 min ahead)
  t=55            TARGET label  (derived from simulated state at t+30
                                 beyond the end of the input window)

The model is NOT predicting the next step after the window.
It is predicting the operational state 30 full minutes after the
last observed snapshot. This is the meaningful forecasting horizon.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEQUENCE INDEPENDENCE (no overlap / no leakage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each sequence is generated independently with its own random initial
conditions (KP, wind speed, density, Bz, sail angle). There are NO
sliding-window overlaps between sequences — this dataset does NOT
extract sequence[0] = t0…t5 and sequence[1] = t1…t6 from the same
continuous trajectory. Every sequence starts fresh.

Consequence: the chronological split is conservative but there is no
within-split contamination from overlapping windows. The test set is
guaranteed to contain sequences generated with different initial
conditions from all training sequences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPLIT STRATEGY — CHRONOLOGICAL 70 / 15 / 15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Train      : first 70%  — model fits on these
  Validation : next  15%  — used for early stopping / hyperparameter
               checks; strictly after all training sequences
  Test       : final 15%  — reported metrics; strictly after both
               train and validation; never seen during development

WHY NOT RANDOM SPLIT:
  Even though sequences are independent, a random split would mix
  sequences generated under the same random seed neighbourhood,
  which could share regime patterns. Chronological splitting is the
  more rigorous and defensible choice for any temporal dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASELINE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A "last-known-state" baseline is computed alongside the model:
  Prediction = whatever label the LAST step in the input window
               would produce under the HelioMesh snapshot rules.

If the model does not clearly beat this baseline, the temporal
information in the window is not providing real value. This is an
honest diagnostic, not a benchmark to game.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Binary, derived from HelioMesh safety rules at the FUTURE step:
  0 → NOMINAL_AHEAD   — no critical state expected at t+30
  1 → CRITICAL_AHEAD  — STANDBY or SAFE_MODE expected at t+30

IMPORTANT: Labels represent HelioMesh simulation-defined operational
states. This model does NOT predict real satellite failures.
All thresholds are HelioMesh prototype rules, not NASA/NOAA standards.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    average_precision_score,
)

# ── Config ────────────────────────────────────────────────────
DATASET_PATH = "ml/sequences.csv"
MODEL_PATH   = "ml/forecaster_model.pkl"
METRICS_PATH = "ml/forecaster_metrics.json"
FEATURE_PATH = "ml/forecaster_feature_importance.json"

TARGET    = "label"
DROP_COLS = ["timestamp", "regime", "label"]

# Chronological split — strictly ordered, no shuffling ever
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (implicit: remainder)

# ── HelioMesh snapshot label rule (for baseline) ──────────────
STEP_FEATURES = ["kp_index", "solar_wind_speed", "orbit_deviation",
                 "power_output", "solar_wind_density", "b_field"]


def _snapshot_label(kp: float, orbit_dev: float, power_out: float) -> int:
    """
    HelioMesh safety rules applied to a single snapshot.
    Used to compute the last-known-state baseline.
    Returns 1 (CRITICAL) if SAFE_MODE or STANDBY would be triggered.
    """
    if kp > 6.0 or orbit_dev > 1.5 or power_out < 10.0:
        return 1
    if kp > 4.0 or orbit_dev > 0.8 or power_out < 40.0:
        return 1
    return 0


def _compute_baseline(X: np.ndarray, feature_cols: list) -> np.ndarray:
    """
    Last-known-state baseline:
      Predict the label that the LAST window step (t5) would produce
      under the snapshot safety rules.
    Feature layout: 6 steps × 6 features, step 5 is at indices [30:36].
    Features per step: kp, wind, orbit_dev, power_out, density, bz.
    """
    # Step 5 raw feature indices (0-indexed within the flattened vector)
    # Layout: [step0_kp, step0_wind, step0_orb, step0_pwr, step0_dens, step0_bz,
    #          step1_kp, ... step5_kp, step5_wind, step5_orb, step5_pwr, ...]
    N_STEP_FEATS = len(STEP_FEATURES)   # 6
    LAST_STEP_START = 5 * N_STEP_FEATS  # index 30

    kp_idx    = LAST_STEP_START + 0   # 30
    orb_idx   = LAST_STEP_START + 2   # 32
    power_idx = LAST_STEP_START + 3   # 33

    predictions = np.array([
        _snapshot_label(row[kp_idx], row[orb_idx], row[power_idx])
        for row in X
    ])
    return predictions


def train():
    print("=" * 62)
    print("  HelioMesh — Supervised Temporal Predictor Training")
    print("=" * 62)

    # ── Load dataset (preserve row order = temporal order) ──
    print("\nLoading sequence dataset...")
    df = pd.read_csv(DATASET_PATH)
    # CRITICAL: do NOT shuffle. Row order == generation order == temporal order.
    print(f"  Rows:     {len(df)}")
    print(f"  Features: {len(df.columns) - len(DROP_COLS)}")
    vc = df[TARGET].value_counts().sort_index()
    print(f"  Label distribution:\n"
          f"    NOMINAL_AHEAD  (0): {vc.get(0, 0)}\n"
          f"    CRITICAL_AHEAD (1): {vc.get(1, 0)}\n")

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].values
    y = df[TARGET].values
    n = len(X)

    # ── Chronological 70 / 15 / 15 split ─────────────────────
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end],      y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],         y[val_end:]

    def _dist(y_arr):
        u, c = np.unique(y_arr, return_counts=True)
        d = dict(zip(u.tolist(), c.tolist()))
        return d

    print("Chronological split  (70 / 15 / 15):")
    print(f"  Train : {len(X_train):5d} sequences  {_dist(y_train)}")
    print(f"  Val   : {len(X_val):5d} sequences  {_dist(y_val)}")
    print(f"  Test  : {len(X_test):5d} sequences  {_dist(y_test)}\n")
    print("  Train covers earliest sequences; Test covers latest — no future")
    print("  information is visible during training or validation.\n")

    # ── Train Gradient Boosting ────────────────────────────────
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        min_samples_leaf=8,
        subsample=0.85,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Validation performance (for transparency — not used to retrain)
    val_pred = model.predict(X_val)
    val_f1   = f1_score(y_val, val_pred, average="macro")
    val_prob = model.predict_proba(X_val)[:, 1]
    val_auc  = roc_auc_score(y_val, val_prob)
    print(f"  Validation Macro F1 : {val_f1:.4f}")
    print(f"  Validation ROC-AUC  : {val_auc:.4f}")
    print("  (Validation set used for diagnostic only — model not retrained.)\n")

    # ── Evaluate on chronological TEST set ────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report   = classification_report(
        y_test, y_pred,
        target_names=["NOMINAL_AHEAD", "CRITICAL_AHEAD"],
        output_dict=True,
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    roc_auc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    cm       = confusion_matrix(y_test, y_pred).tolist()

    # ── Last-known-state baseline on TEST set ─────────────────
    baseline_pred = _compute_baseline(X_test, feature_cols)
    baseline_f1   = f1_score(y_test, baseline_pred, average="macro")
    baseline_auc  = roc_auc_score(y_test, baseline_pred)
    baseline_report = classification_report(
        y_test, baseline_pred,
        target_names=["NOMINAL_AHEAD", "CRITICAL_AHEAD"],
        output_dict=True,
    )
    baseline_cm = confusion_matrix(y_test, baseline_pred).tolist()

    # ── Print results ──────────────────────────────────────────
    print("=" * 62)
    print("  TEST SET RESULTS  (final 15% — never seen during training)")
    print("=" * 62)
    print(classification_report(
        y_test, y_pred,
        target_names=["NOMINAL_AHEAD", "CRITICAL_AHEAD"],
    ))
    print(f"Macro F1              : {macro_f1:.4f}")
    print(f"ROC-AUC               : {roc_auc:.4f}")
    print(f"Average Precision (AP): {avg_prec:.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"  Classes: [NOMINAL_AHEAD, CRITICAL_AHEAD]")
    for row in cm:
        print(f"  {row}")

    # ── Baseline comparison ────────────────────────────────────
    print("\n" + "=" * 62)
    print("  BASELINE COMPARISON  (last-known-state rule, same test set)")
    print("  Baseline predicts: 'whatever the last window step implies'")
    print("=" * 62)
    print(classification_report(
        y_test, baseline_pred,
        target_names=["NOMINAL_AHEAD", "CRITICAL_AHEAD"],
    ))
    print(f"Baseline Macro F1     : {baseline_f1:.4f}")
    print(f"Baseline ROC-AUC      : {baseline_auc:.4f}")

    f1_gain  = macro_f1 - baseline_f1
    auc_gain = roc_auc  - baseline_auc
    print(f"\n  Model vs Baseline:")
    print(f"  Macro F1  gain : {f1_gain:+.4f}   "
          f"({'✅ model adds value' if f1_gain > 0.02 else '⚠ marginal gain' if f1_gain > 0 else '❌ no gain'})")
    print(f"  ROC-AUC   gain : {auc_gain:+.4f}   "
          f"({'✅ model adds value' if auc_gain > 0.02 else '⚠ marginal gain' if auc_gain > 0 else '❌ no gain'})")

    # ── Feature importance (top 10) ───────────────────────────
    fi = dict(zip(feature_cols, model.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    print(f"\nTop 10 Feature Importances:")
    for feat, imp in list(fi_sorted.items())[:10]:
        print(f"  {feat:35s}: {imp:.4f}")

    # ── Save model ────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved   → {MODEL_PATH}")

    # ── Save metrics ──────────────────────────────────────────
    metrics = {
        "trained_at":   datetime.now().isoformat(),
        "model_type":   "GradientBoostingClassifier",

        # ── What this model does ──
        "task": (
            "Supervised classification: given 6 simulated telemetry snapshots "
            "(5-min intervals, 30-min look-back window), predict whether the "
            "satellite's simulated operational state will be CRITICAL (STANDBY "
            "or SAFE_MODE) 30 minutes after the last observed snapshot."
        ),
        "honest_description": (
            "HelioMesh predicts the satellite's future operational safety state "
            "from a sequence of simulated telemetry observations. "
            "This is NOT a prediction of real satellite failures."
        ),
        "model_type_clarification": (
            "Supervised classification over a fixed look-back window. "
            "Not classical ARIMA or RNN time-series forecasting."
        ),

        # ── Timing ──
        "window_steps":          6,
        "horizon_steps":         6,
        "step_minutes":          5,
        "look_back_minutes":     30,
        "look_ahead_minutes":    30,
        "total_span_minutes":    60,
        "timing_note": (
            "Input window: t0…t25 (6 snapshots). "
            "Horizon gap: t25…t55 (6 more simulated steps, not in features). "
            "Target: operational state at t55, i.e. 30 min beyond the last input."
        ),

        # ── Split ──
        "split_strategy": "chronological 70 / 15 / 15",
        "split_rationale": (
            "Train = first 70% (oldest). "
            "Validation = next 15% (strictly after train). "
            "Test = final 15% (strictly after both — reported metrics). "
            "No shuffling. No future information visible during training."
        ),

        # ── Sequence independence ──
        "sequence_overlap": "none",
        "sequence_overlap_note": (
            "Each sequence is generated with independent random initial conditions. "
            "This dataset does NOT use a sliding window over a continuous "
            "trajectory — there are no overlapping sequences."
        ),

        # ── Sizes ──
        "n_train":  int(len(X_train)),
        "n_val":    int(len(X_val)),
        "n_test":   int(len(X_test)),
        "n_features": len(feature_cols),

        # ── Model metrics (test set) ──
        "test_macro_f1":        round(macro_f1, 4),
        "test_roc_auc":         round(roc_auc, 4),
        "test_average_precision": round(avg_prec, 4),
        "val_macro_f1":         round(val_f1, 4),
        "val_roc_auc":          round(val_auc, 4),

        "test_per_class": {
            "NOMINAL_AHEAD": {
                "precision": round(report["NOMINAL_AHEAD"]["precision"], 4),
                "recall":    round(report["NOMINAL_AHEAD"]["recall"],    4),
                "f1_score":  round(report["NOMINAL_AHEAD"]["f1-score"],  4),
                "support":   int(report["NOMINAL_AHEAD"]["support"]),
            },
            "CRITICAL_AHEAD": {
                "precision": round(report["CRITICAL_AHEAD"]["precision"], 4),
                "recall":    round(report["CRITICAL_AHEAD"]["recall"],    4),
                "f1_score":  round(report["CRITICAL_AHEAD"]["f1-score"],  4),
                "support":   int(report["CRITICAL_AHEAD"]["support"]),
            },
        },
        "test_confusion_matrix": cm,

        # ── Baseline comparison ──
        "baseline": {
            "name":        "last-known-state",
            "description": (
                "Predict the label that the LAST window snapshot (t5) would "
                "produce under HelioMesh snapshot safety rules. "
                "Represents 'no temporal learning — just use current state'."
            ),
            "macro_f1":    round(baseline_f1, 4),
            "roc_auc":     round(baseline_auc, 4),
            "per_class": {
                "NOMINAL_AHEAD": {
                    "precision": round(baseline_report["NOMINAL_AHEAD"]["precision"], 4),
                    "recall":    round(baseline_report["NOMINAL_AHEAD"]["recall"],    4),
                    "f1_score":  round(baseline_report["NOMINAL_AHEAD"]["f1-score"],  4),
                },
                "CRITICAL_AHEAD": {
                    "precision": round(baseline_report["CRITICAL_AHEAD"]["precision"], 4),
                    "recall":    round(baseline_report["CRITICAL_AHEAD"]["recall"],    4),
                    "f1_score":  round(baseline_report["CRITICAL_AHEAD"]["f1-score"],  4),
                },
            },
            "confusion_matrix": baseline_cm,
        },
        "model_vs_baseline": {
            "macro_f1_gain":  round(f1_gain, 4),
            "roc_auc_gain":   round(auc_gain, 4),
            "interpretation": (
                "positive gain = temporal window adds predictive value beyond "
                "simply using the current snapshot state"
            ),
        },

        "train_label_dist": _dist(y_train),
        "val_label_dist":   _dist(y_val),
        "test_label_dist":  _dist(y_test),

        "model_params": model.get_params(),
        "disclaimer": (
            "This model predicts future HelioMesh simulation-defined operational "
            "states. It does NOT predict real satellite failures. All labels are "
            "derived from HelioMesh prototype safety rules, not NASA/NOAA standards."
        ),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    with open(FEATURE_PATH, "w") as f:
        json.dump(fi_sorted, f, indent=2)
    print(f"Features saved → {FEATURE_PATH}")

    print("\n✅ Supervised temporal predictor training complete.")
    return metrics


if __name__ == "__main__":
    train()
