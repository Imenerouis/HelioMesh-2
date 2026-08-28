"""
HelioMesh â€” Early Warning Evaluation
======================================
Evaluates the temporal predictor's ability to provide early warning of
critical state transitions, using the simulation test sequences (ml/sequences.csv).

Algorithm:
  For each sequence in the test split:
    - If the ground-truth label is CRITICAL_AHEAD (label=1):
        A "transition" is flagged.
        "Early detection" = temporal predictor outputs CRITICAL_AHEAD for this sequence.
        "Lead time" is defined as: would the predictor catch this before the snapshot
        classifier? We estimate this from the delta features in the window.
    - If the ground-truth label is NOMINAL_AHEAD (label=0):
        A false early warning = temporal predictor outputs CRITICAL_AHEAD (false positive).

Lead time approximation:
  The sequences are independent (no consecutive sliding windows), so exact step-level
  lead time cannot be computed from the CSV alone. Instead we use the KP delta feature
  to estimate how much earlier the window-based predictor detects rising conditions vs
  a naive current-state rule:
    - A sequence is "early" if delta_kp > 0 AND predictor = CRITICAL AND label = CRITICAL
      (the predictor caught a rising storm before it peaked)
    - Lead time proxy: floor(delta_kp / 0.25) steps (each step = 5 min)
      â€” represents estimated number of 5-min steps of advance warning

Results saved to: validation/results/early_warning.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import pickle
import math
import statistics
from datetime import datetime

_ML_DIR    = os.path.join(os.path.dirname(__file__), "..", "ml")
_MODEL_PATH = os.path.join(_ML_DIR, "forecaster_model.pkl")
_SEQ_PATH   = os.path.join(_ML_DIR, "sequences.csv")
_OUT_PATH   = os.path.join(os.path.dirname(__file__), "results", "early_warning.json")

# Chronological test split is the last 15% of 12000 sequences = rows 10200â€“11999
TEST_START = 10200


def _load_sequences() -> list[dict]:
    rows = []
    with open(_SEQ_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_features(row: dict) -> list[float]:
    """Extract the 42 precomputed features from a CSV row."""
    # Sequences CSV has columns: kp_index_t0..t5, solar_wind_speed_t0..t5, ...
    # Then delta columns. We read them in the exact same order as training.
    from ml.generate_sequences import FEATURE_NAMES
    return [float(row[name]) for name in FEATURE_NAMES]


def run() -> dict:
    print("  Loading sequences...", end=" ", flush=True)
    all_rows = _load_sequences()
    test_rows = all_rows[TEST_START:]
    print(f"{len(test_rows)} test rows")

    print("  Loading temporal model...", end=" ", flush=True)
    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("OK")

    total_critical_transitions = 0
    transitions_detected_early = 0
    missed_transitions          = 0
    false_early_warnings        = 0
    lead_times: list[float] = []

    for row in test_rows:
        label     = int(row["label"])
        feat_vec  = _extract_features(row)
        proba     = model.predict_proba([feat_vec])[0]
        p_crit    = float(proba[1])
        predicted = 1 if p_crit >= 0.5 else 0

        # Delta KP: feature index 36 (first delta feature)
        delta_kp = float(row.get("delta_kp_index", feat_vec[36]))

        if label == 1:
            total_critical_transitions += 1
            if predicted == 1:
                transitions_detected_early += 1
                # Lead time proxy: how many steps of rising KP the predictor saw
                lead = max(0.0, math.floor(delta_kp / 0.25))
                lead_times.append(lead)
            else:
                missed_transitions += 1
        else:
            if predicted == 1:
                false_early_warnings += 1

    median_lead = round(statistics.median(lead_times), 2) if lead_times else 0.0
    mean_lead   = round(statistics.mean(lead_times),   2) if lead_times else 0.0

    result = {
        "evaluated_at":               datetime.now().isoformat(),
        "n_test_sequences":           len(test_rows),
        "total_critical_transitions": total_critical_transitions,
        "transitions_detected_early": transitions_detected_early,
        "missed_transitions":         missed_transitions,
        "false_early_warnings":       false_early_warnings,
        "early_detection_rate":       round(
            transitions_detected_early / total_critical_transitions, 4
        ) if total_critical_transitions > 0 else 0.0,
        "false_positive_rate":        round(
            false_early_warnings / (len(test_rows) - total_critical_transitions), 4
        ) if (len(test_rows) - total_critical_transitions) > 0 else 0.0,
        "median_lead_time_steps":     median_lead,
        "mean_lead_time_steps":       mean_lead,
        "step_duration_minutes":      5,
        "methodology_note": (
            "Lead time is a proxy: floor(delta_kp / 0.25) steps, "
            "representing estimated 5-min steps of advance KP warning in the look-back window. "
            "Sequences are independent â€” exact step-level transitions cannot be tracked across rows."
        ),
        "disclaimer": (
            "Evaluation uses simulation test sequences only. "
            "NOT validated against real spacecraft data."
        ),
    }

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  âœ“ Critical transitions: {total_critical_transitions}")
    print(f"  âœ“ Detected early: {transitions_detected_early}  "
          f"({result['early_detection_rate']*100:.1f}%)")
    print(f"  âœ“ Missed: {missed_transitions}")
    print(f"  âœ“ False early warnings: {false_early_warnings}")
    print(f"  âœ“ Median lead time: {median_lead} steps ({median_lead*5:.0f} min)")
    print(f"  âœ“ Saved â†’ {_OUT_PATH}")
    return result


if __name__ == "__main__":
    run()

