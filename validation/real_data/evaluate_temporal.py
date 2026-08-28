"""
HelioMesh â€” Temporal Predictor Evaluation on OMNI2-Derived Sequences
======================================================================
Builds 6-step telemetry windows from OMNI2-derived hourly observations,
runs the Gradient Boosting temporal predictor, and compares predictions
against HelioMesh labeling rules applied to the t+6 step.

Temporal note:
  Window: steps[i : i+6]  (6 hourly steps = 6 hours of observations)
  Target: step[i+6] â€” the state 6 steps (6 hours) ahead

  This differs from the simulation training setup (5-min steps, 30-min window)
  because OMNI2 data is hourly. This difference is documented in results.

DISCLAIMER:
  This is an internal-consistency check only. Labels are derived from
  HelioMesh safety rules applied to simulated feature values. Not a
  predictor of real spacecraft failures.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pickle
import math
from datetime import datetime

from validation.real_data.loader import load_omni2
from validation.real_data.preprocess import preprocess
from validation.real_data.metrics import classification_report, confusion_matrix, label_distribution

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ml", "forecaster_model.pkl"
)


def _build_window_features(window: list[dict]) -> list:
    """
    Build the 42-element feature vector from a 6-row window.
    Mirrors ml/forecaster.py :: _build_features() exactly.
    Step features: [kp_index, solar_wind_speed, orbit_deviation,
                    power_output, solar_wind_density, b_field]
    """
    steps = []
    for row in window:
        steps.append([
            row["kp_index"],
            row["solar_wind_speed"],
            row["orbit_deviation"],
            row["power_output"],
            row["solar_wind_density"],
            row["b_field"],
        ])

    flat = []
    for step in steps:
        flat.extend(step)
    # Delta features: t5 - t0 for each of 6 features
    for i in range(6):
        flat.append(round(steps[-1][i] - steps[0][i], 4))
    return flat


def _critical_label(kp, orbit_dev, power):
    """Binary label: 1 = CRITICAL_AHEAD (STANDBY or SAFE_MODE)."""
    if kp > 6.0 or orbit_dev > 1.5 or power < 10.0:
        return 1
    if kp > 4.0 or orbit_dev > 0.8 or power < 40.0:
        return 1
    return 0


def run(n_records: int = 600) -> dict:
    """
    Run the temporal predictor evaluation on OMNI2-derived sequences.
    Returns result dict; saves to validation/results/temporal_validation.json.
    """
    print("  Loading OMNI2 data...", end=" ", flush=True)
    records, source = load_omni2(n_static=n_records)
    print(f"{len(records)} records ({source})")

    print("  Preprocessing features...", end=" ", flush=True)
    processed = preprocess(records)
    print(f"{len(processed)} rows ready")

    print("  Loading GB temporal model...", end=" ", flush=True)
    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("OK")

    WINDOW = 6
    y_true = []
    y_pred_label = []
    y_pred_prob  = []

    for i in range(len(processed) - WINDOW):
        window = processed[i : i + WINDOW]
        target = processed[i + WINDOW]

        feat_vec  = _build_window_features(window)
        proba     = model.predict_proba([feat_vec])[0]   # [P(0), P(1)]
        p_crit    = float(proba[1])
        predicted = 1 if p_crit >= 0.5 else 0

        true_label = _critical_label(
            target["kp_index"],
            target["orbit_deviation"],
            target["power_output"],
        )
        y_true.append(true_label)
        y_pred_label.append(predicted)
        y_pred_prob.append(p_crit)

    # String labels for report
    label_map = {0: "NOMINAL_AHEAD", 1: "CRITICAL_AHEAD"}
    yt_str = [label_map[v] for v in y_true]
    yp_str = [label_map[v] for v in y_pred_label]

    metrics  = classification_report(yt_str, yp_str)
    cm       = confusion_matrix(yt_str, yp_str)
    gt_dist  = label_distribution(yt_str)
    pd_dist  = label_distribution(yp_str)

    result = {
        "evaluated_at":       datetime.now().isoformat(),
        "data_source":        source,
        "n_records_raw":      len(processed),
        "n_sequences":        len(y_true),
        "window_steps":       WINDOW,
        "horizon_steps":      1,
        "step_duration_note": "OMNI2 hourly data â€” steps are 1 hour apart (training used 5-min steps)",
        "task":               "internal_consistency_check",
        "task_description": (
            "Temporal predictor vs HelioMesh labeling rules applied to OMNI2-derived features. "
            "Window cadence differs from training (hourly vs 5-min). "
            "NOT spacecraft failure prediction."
        ),
        "consistency_rate":          round(metrics["accuracy"], 4),
        "macro_f1":                  metrics["macro_f1"],
        "accuracy":                  metrics["accuracy"],
        "per_class_metrics":         metrics["per_class"],
        "confusion_matrix":          cm,
        "ground_truth_distribution": gt_dist,
        "predicted_distribution":    pd_dist,
        "disclaimer": (
            "Labels are synthetic, derived from HelioMesh operational safety rules. "
            "Hourly window differs from 5-min training window â€” results are indicative only."
        ),
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "temporal_validation.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  âœ“ Temporal consistency: {metrics['accuracy']*100:.1f}%  macro-F1: {metrics['macro_f1']}")
    print(f"  âœ“ Saved â†’ {out_path}")
    return result


if __name__ == "__main__":
    run()

