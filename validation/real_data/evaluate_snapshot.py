"""
HelioMesh â€” Snapshot Classifier Evaluation on OMNI2 Data
=========================================================
Runs the Random Forest snapshot classifier on OMNI2-derived inputs and
compares predictions against HelioMesh labeling rules applied to those
same inputs.

IMPORTANT â€” what this measures and what it does NOT measure:
  - MEASURES:   Internal consistency â€” does the RF model reproduce
                HelioMesh labeling rules when given real space-weather inputs?
  - DOES NOT:   Predict real spacecraft failures. Labels are synthetic,
                derived from HelioMesh rules, not from actual mission data.
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

_ML_DIR      = os.path.join(os.path.dirname(__file__), "..", "..", "ml")
_MODEL_PATH  = os.path.join(_ML_DIR, "risk_model.pkl")
_ENCODER_PATH = os.path.join(_ML_DIR, "label_encoder.pkl")


def _build_features(row: dict) -> list:
    """
    Build the 11-element feature vector for the RF snapshot classifier.
    Mirrors ml/predictor.py :: _build_features() exactly.
    """
    return [
        row["kp_index"],
        row["sail_angle"],
        row["solar_wind_speed"],
        row["solar_wind_density"],
        row["b_field"],
        row["drag_factor"],
        row["orbit_deviation"],
        row["power_output"],
        row["thrust_output"],
        row["geomagnetic_energy"],
        row["dynamic_pressure"],
    ]


def run(n_records: int = 500) -> dict:
    """
    Run the snapshot classifier evaluation.
    Returns a result dict saved to validation/results/snapshot_validation.json.
    """
    print("  Loading OMNI2 data...", end=" ", flush=True)
    records, source = load_omni2(n_static=n_records)
    print(f"{len(records)} records ({source})")

    print("  Preprocessing features...", end=" ", flush=True)
    processed = preprocess(records)
    print(f"{len(processed)} rows ready")

    print("  Loading RF model...", end=" ", flush=True)
    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(_ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    print("OK")

    y_true = []
    y_pred = []

    for row in processed:
        feat_vec  = _build_features(row)
        idx       = model.predict([feat_vec])[0]
        predicted = encoder.inverse_transform([idx])[0]
        y_true.append(row["ground_truth_label"])
        y_pred.append(predicted)

    metrics = classification_report(y_true, y_pred)
    cm      = confusion_matrix(y_true, y_pred)
    gt_dist = label_distribution(y_true)
    pd_dist = label_distribution(y_pred)

    # Consistency rate = fraction where model label matches rule-derived label
    consistency = metrics["accuracy"]

    result = {
        "evaluated_at":      datetime.now().isoformat(),
        "data_source":       source,
        "n_records":         len(processed),
        "task":              "internal_consistency_check",
        "task_description": (
            "Snapshot RF classifier vs HelioMesh labeling rules on space-weather inputs. "
            "NOT spacecraft failure prediction."
        ),
        "consistency_rate":       round(consistency, 4),
        "macro_f1":               metrics["macro_f1"],
        "accuracy":               metrics["accuracy"],
        "per_class_metrics":      metrics["per_class"],
        "confusion_matrix":       cm,
        "ground_truth_distribution": gt_dist,
        "predicted_distribution":    pd_dist,
        "disclaimer": (
            "Labels are synthetic, derived from HelioMesh operational safety rules. "
            "Consistency measures model-rule agreement, not real spacecraft performance."
        ),
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "snapshot_validation.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  âœ“ Snapshot consistency: {consistency*100:.1f}%  macro-F1: {metrics['macro_f1']}")
    print(f"  âœ“ Saved â†’ {out_path}")
    return result


if __name__ == "__main__":
    run()

