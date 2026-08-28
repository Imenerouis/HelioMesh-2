"""
HelioMesh â€” Data Drift / Compatibility Monitor
================================================
Computes whether incoming telemetry is compatible with the training
distribution of the simulation sequences dataset.

Method:
  1. Load training split of ml/sequences.csv (first 70% = rows 0..8399)
  2. Compute per-feature mean and std
  3. For incoming telemetry, compute:
       z_score     = |value - train_mean| / train_std
       spread_ratio = std(window) / train_std  (if window given)
  4. Classify:
       ALL z_scores < 2.5        â†’ STABLE
       ANY z_score in [2.5, 4.0) â†’ MODERATE_DRIFT
       ANY z_score >= 4.0        â†’ HIGH_DRIFT

Drift config saved to: validation/drift_config.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import math
import statistics
from datetime import datetime

_SEQ_PATH     = os.path.join(os.path.dirname(__file__), "..", "ml", "sequences.csv")
_CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "drift_config.json")

TRAIN_END = 8400   # chronological 70% of 12000

# Features to monitor (last-step values available from snapshot telemetry)
MONITOR_FEATURES = [
    "kp_index", "solar_wind_speed", "solar_wind_density",
    "b_field", "orbit_deviation", "power_output",
]

# Map from snapshot telemetry key â†’ sequence CSV last-step column name
_FEAT_COL_MAP = {
    "kp_index":           "kp_index_t5",
    "solar_wind_speed":   "solar_wind_speed_t5",
    "solar_wind_density": "solar_wind_density_t5",
    "b_field":            "b_field_t5",
    "orbit_deviation":    "orbit_deviation_t5",
    "power_output":       "power_output_t5",
}

# Drift thresholds
Z_MODERATE = 2.5
Z_HIGH     = 4.0


def build_training_stats() -> dict:
    """
    Compute mean and std for monitor features from the training split.
    Returns dict: {feature: {mean, std}}
    """
    values: dict[str, list[float]] = {f: [] for f in MONITOR_FEATURES}

    with open(_SEQ_PATH, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= TRAIN_END:
                break
            for feat in MONITOR_FEATURES:
                col = _FEAT_COL_MAP[feat]
                values[feat].append(float(row[col]))

    stats = {}
    for feat, vals in values.items():
        mu  = statistics.mean(vals)
        sig = statistics.stdev(vals) if len(vals) > 1 else 1.0
        stats[feat] = {
            "mean": round(mu,  4),
            "std":  round(sig, 4),
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
        }
    return stats


def save_drift_config() -> dict:
    """Build and persist drift config from training data."""
    print("  Building drift config from training split...", end=" ", flush=True)
    stats = build_training_stats()
    config = {
        "built_at":          datetime.now().isoformat(),
        "train_rows":        TRAIN_END,
        "monitor_features":  MONITOR_FEATURES,
        "z_threshold_moderate": Z_MODERATE,
        "z_threshold_high":     Z_HIGH,
        "training_stats":    stats,
        "status_definitions": {
            "STABLE":         f"All |z-scores| < {Z_MODERATE}",
            "MODERATE_DRIFT": f"Any |z-score| in [{Z_MODERATE}, {Z_HIGH})",
            "HIGH_DRIFT":     f"Any |z-score| >= {Z_HIGH}",
        },
    }
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print("OK")
    print(f"  âœ“ Saved â†’ {_CONFIG_PATH}")
    return config


def load_drift_config() -> dict:
    """Load drift config, building it if not present."""
    if not os.path.exists(_CONFIG_PATH):
        return save_drift_config()
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def assess_drift(telemetry: dict) -> dict:
    """
    Assess whether incoming telemetry is within the training distribution.

    Parameters
    ----------
    telemetry : dict
        Must contain keys: kp_index, solar_wind_speed, solar_wind_density,
        b_field, orbit_deviation, power_output

    Returns
    -------
    dict with keys:
        drift_status   : "STABLE" | "MODERATE_DRIFT" | "HIGH_DRIFT"
        z_scores       : {feature: z_score}
        max_z_score    : float
        drifted_features: list[str]
    """
    config = load_drift_config()
    stats  = config["training_stats"]

    z_scores: dict[str, float] = {}
    for feat in MONITOR_FEATURES:
        value = telemetry.get(feat)
        if value is None:
            continue
        mu  = stats[feat]["mean"]
        sig = stats[feat]["std"] if stats[feat]["std"] > 0 else 1.0
        z   = abs(float(value) - mu) / sig
        z_scores[feat] = round(z, 3)

    max_z   = max(z_scores.values()) if z_scores else 0.0
    drifted = [f for f, z in z_scores.items() if z >= Z_MODERATE]

    if max_z >= Z_HIGH:
        status = "HIGH_DRIFT"
    elif max_z >= Z_MODERATE:
        status = "MODERATE_DRIFT"
    else:
        status = "STABLE"

    return {
        "drift_status":      status,
        "z_scores":          z_scores,
        "max_z_score":       round(max_z, 3),
        "drifted_features":  drifted,
    }


if __name__ == "__main__":
    cfg = save_drift_config()
    print("\nTraining distribution stats:")
    for feat, s in cfg["training_stats"].items():
        print(f"  {feat:25s}  mean={s['mean']:8.3f}  std={s['std']:7.3f}")

    # Quick smoke test
    test_nominal = {
        "kp_index": 2.0, "solar_wind_speed": 420.0,
        "solar_wind_density": 5.0, "b_field": -5.0,
        "orbit_deviation": 0.3, "power_output": 70.0,
    }
    test_extreme = {
        "kp_index": 9.0, "solar_wind_speed": 1100.0,
        "solar_wind_density": 28.0, "b_field": -40.0,
        "orbit_deviation": 2.5, "power_output": 1.0,
    }
    print("\nDrift check (nominal):", assess_drift(test_nominal)["drift_status"])
    print("Drift check (extreme):", assess_drift(test_extreme)["drift_status"])

