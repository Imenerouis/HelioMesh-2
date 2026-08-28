"""
HelioMesh â€” OMNI2 â†’ HelioMesh Feature Mapping
===============================================
Maps OMNI2 space-weather fields to the HelioMesh feature schema used by
the snapshot classifier (Random Forest) and temporal predictor (Gradient Boosting).

Derived features (drag_factor, orbit_deviation, power_output, etc.) are
computed using the SAME formulas as ml/generate_dataset.py and ml/forecaster.py
so there is no train/inference mismatch.

Fixed simulation parameters:
  sail_angle = 45Â° (representative mid-mission value)

This is an internal-consistency validation: real space-weather inputs â†’ model
predictions â†’ compare against HelioMesh labeling rules applied to same inputs.
NOT a prediction of real spacecraft failures.
"""

import math
from typing import Any

# Fixed satellite configuration assumed for OMNI2 inputs
DEFAULT_SAIL_ANGLE = 45.0


def _derive_features(kp: float, sail: float, wind: float,
                     dens: float, bz: float) -> dict[str, Any]:
    """
    Compute all 11 HelioMesh features from 5 raw space-weather parameters.
    Formula source: ml/generate_dataset.py :: compute_features()
    """
    drag_factor      = (sail / 90) * (1 + kp * 0.1)
    orbit_deviation  = drag_factor * (wind / 400) * 0.5
    power_output     = math.cos(math.radians(sail)) * 100
    thrust_output    = drag_factor * 0.02
    geo_energy       = kp * abs(bz) / 9.0
    dyn_pressure     = 0.5 * dens * (wind ** 2) * 1e-6

    return {
        "kp_index":           round(kp,  2),
        "sail_angle":         round(sail, 2),
        "solar_wind_speed":   round(wind, 2),
        "solar_wind_density": round(dens, 2),
        "b_field":            round(bz,   2),
        "drag_factor":        round(drag_factor,     4),
        "orbit_deviation":    round(orbit_deviation,  4),
        "power_output":       round(power_output,     4),
        "thrust_output":      round(thrust_output,    6),
        "geomagnetic_energy": round(geo_energy,       4),
        "dynamic_pressure":   round(dyn_pressure,     4),
    }


def _derive_label(kp: float, orbit_dev: float, power: float) -> str:
    """
    HelioMesh operational safety rules (from ml/generate_dataset.py).
    Applied here to generate ground-truth labels for OMNI2 inputs.
    """
    if kp > 6.0 or orbit_dev > 1.5 or power < 10.0:
        return "SAFE_MODE"
    elif kp > 4.0 or orbit_dev > 0.8 or power < 40.0:
        return "STANDBY"
    return "NOMINAL"


def preprocess(records: list[dict],
               sail_angle: float = DEFAULT_SAIL_ANGLE) -> list[dict]:
    """
    Convert OMNI2 records into HelioMesh-compatible feature dicts with
    ground-truth labels derived from HelioMesh labeling rules.

    Returns list of dicts containing:
      - All 11 snapshot features
      - ground_truth_label (NOMINAL / STANDBY / SAFE_MODE)
      - original OMNI2 fields (timestamp, source)
    """
    processed = []
    for r in records:
        kp   = r["kp_index"]
        wind = r["solar_wind_speed"]
        dens = r["solar_wind_density"]
        bz   = r["b_field"]

        feats = _derive_features(kp, sail_angle, wind, dens, bz)
        label = _derive_label(
            feats["kp_index"],
            feats["orbit_deviation"],
            feats["power_output"],
        )
        processed.append({
            "timestamp":           r["timestamp"],
            "source":              r.get("source", "unknown"),
            "ground_truth_label":  label,
            **feats,
        })
    return processed

