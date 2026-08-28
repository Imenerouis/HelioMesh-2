"""
HelioMesh -- Real Telemetry Evaluation
=======================================
Runs the full anomaly-detection evaluation pipeline and saves results to
  validation/results/anomaly_methodology_metrics.json
  (SYNTHETIC_STANDIN — NOT real spacecraft telemetry validation)

Metrics computed
----------------
Point-level (standard binary classification):
  precision, recall, F1, balanced_accuracy, MCC, ROC-AUC

Event-level:
  detected_events   -- anomaly windows with >= 1 point flagged as anomalous
  missed_events     -- anomaly windows with 0 points flagged
  false_alarms      -- contiguous predicted-anomaly runs that don't overlap
                       any labelled window

Counts:
  total_anomaly_windows, false_alarm_count, missed_event_count

IMPORTANT: results here are labelled with prefix "real_telemetry_" and
kept in a namespace completely separate from the HelioMesh simulation
results (snapshot_validation.json, temporal_validation.json etc.).

This evaluation does NOT validate the RF / GB spacecraft-health models.
"""

import os
import sys
import json
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from validation.real_telemetry.dataset import load_dataset
from validation.real_telemetry.anomaly_model import train_detector, METHOD, THRESHOLD


# ---------------------------------------------------------------------------
# Pure-Python metrics (no sklearn required for these)
# ---------------------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def _balanced_accuracy(tp, tn, fp, fn):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return round((sensitivity + specificity) / 2, 4)


def _mcc(tp, tn, fp, fn):
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return round((tp * tn - fp * fn) / denom, 4)


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal AUC without sklearn."""
    # Sort by descending score
    order  = np.argsort(-scores)
    ys     = y_true[order]
    n_pos  = int(y_true.sum())
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tps, fps = 0, 0
    auc      = 0.0
    prev_fp  = 0
    prev_tp  = 0

    for y in ys:
        if y == 1:
            tps += 1
        else:
            fps += 1
            # Each new FP adds a trapezoid
            auc += (tps + prev_tp) / 2.0
            prev_tp = tps
            prev_fp = fps

    auc += (tps + prev_tp) / 2.0  # final rectangle
    auc /= n_pos * n_neg
    return round(float(auc), 4)


# ---------------------------------------------------------------------------
# Event-level helpers
# ---------------------------------------------------------------------------

def _contiguous_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive for contiguous 1-runs in arr."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(arr):
        if v == 1 and not in_run:
            in_run = True
            start = i
        elif v == 0 and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(arr) - 1))
    return runs


def _overlaps(run: tuple[int, int], windows: list[tuple[int, int]]) -> bool:
    s, e = run
    for ws, we in windows:
        if s <= we and e >= ws:
            return True
    return False


def event_level_metrics(
    y_pred: np.ndarray,
    anomaly_windows: list[tuple[int, int]],
) -> dict:
    """
    detected_events   -- labelled windows that have >=1 predicted-anomaly point
    missed_events     -- labelled windows with 0 predicted-anomaly points
    false_alarms      -- predicted-anomaly runs that don't overlap any labelled window
    """
    pred_runs = _contiguous_runs(y_pred)

    detected = 0
    missed   = 0
    for ws, we in anomaly_windows:
        window_labels = y_pred[ws : we + 1]
        if window_labels.sum() > 0:
            detected += 1
        else:
            missed += 1

    false_alarms = sum(
        1 for run in pred_runs if not _overlaps(run, anomaly_windows)
    )

    return {
        "total_anomaly_windows": len(anomaly_windows),
        "detected_events":       detected,
        "missed_events":         missed,
        "false_alarm_count":     false_alarms,
        "event_detection_rate":  round(detected / len(anomaly_windows), 4)
                                 if anomaly_windows else 0.0,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run() -> dict:
    print("\n[real_telemetry] Starting evaluation...")

    # 1. Load dataset
    dataset = load_dataset()

    # 2. Train detector
    detector, scores, y_pred = train_detector(dataset, method=METHOD)

    y_true = dataset["labels"]

    # 3. Point-level metrics
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    bal_acc = _balanced_accuracy(tp, tn, fp, fn)
    mcc     = _mcc(tp, tn, fp, fn)
    auc     = _roc_auc(y_true, scores)

    # 4. Event-level metrics
    evts = event_level_metrics(y_pred, dataset["anomaly_windows"])

    # 5. Print
    total = tp + tn + fp + fn
    print(f"\n  Source   : {dataset['source']}")
    print(f"  Method   : {METHOD} (threshold={THRESHOLD})")
    print(f"  Samples  : {total}  (anomalous={int(y_true.sum())}, normal={int((y_true==0).sum())})")
    print(f"\n  --- Point-level ---")
    print(f"  Precision        : {precision}")
    print(f"  Recall           : {recall}")
    print(f"  F1               : {f1}")
    print(f"  Balanced Acc     : {bal_acc}")
    print(f"  MCC              : {mcc}")
    print(f"  ROC-AUC          : {auc}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"\n  --- Event-level ---")
    print(f"  Anomaly windows  : {evts['total_anomaly_windows']}")
    print(f"  Detected events  : {evts['detected_events']}")
    print(f"  Missed events    : {evts['missed_events']}")
    print(f"  False alarms     : {evts['false_alarm_count']}")
    print(f"  Event detect rate: {evts['event_detection_rate']*100:.1f}%")

    result = {
        "evaluated_at":   datetime.now().isoformat(),
        "real_telemetry_source":     dataset["source"],
        "real_telemetry_channels":   dataset["channels"],
        "real_telemetry_n_timesteps": int(dataset["n_timesteps"]),
        "real_telemetry_method":     METHOD,
        "real_telemetry_threshold":  THRESHOLD,
        "real_telemetry_point_level": {
            "precision":         precision,
            "recall":            recall,
            "f1":                f1,
            "balanced_accuracy": bal_acc,
            "mcc":               mcc,
            "roc_auc":           auc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        },
        "real_telemetry_event_level": evts,
        "real_telemetry_disclaimer": dataset["disclaimer"],
        "namespace_note": (
            "Results prefixed 'real_telemetry_' are independent of "
            "HelioMesh simulation results. This benchmark evaluates "
            "unsupervised anomaly detection on telemetry channels. "
            "It does NOT validate the RF/GB spacecraft-health classifiers."
        ),
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "anomaly_methodology_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n  [OK] Saved -> {out_path}")
    return result


if __name__ == "__main__":
    run()
