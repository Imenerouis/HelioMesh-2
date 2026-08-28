"""
HelioMesh — Supervised Temporal Predictor (Inference)
======================================================
Loads the trained Gradient Boosting model and exposes forecast().

Pipeline position:
    Telemetry window (last 6 steps) → [Temporal Predictor] → Granite

What this produces:
    Given the last 30 minutes of telemetry (6 snapshots, 5 min apart),
    estimate whether the satellite will be in a critical operational
    state (SAFE_MODE or STANDBY) 30 minutes from now.

What this does NOT produce:
    A prediction of real satellite failures. All state definitions are
    HelioMesh prototype rules derived from simulation.

Input contract:
    window: list of 6 dicts, each containing:
        kp_index, solar_wind_speed, orbit_deviation,
        power_output, solar_wind_density, b_field
    ordered oldest-first (window[0] = t0, window[5] = t+25min).
    None of these values may come from the future horizon — callers
    must only pass observed/simulated past telemetry.
"""

import os
import math
import pickle
from typing import TypedDict

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "forecaster_model.pkl")

_model = None


def _load():
    global _model
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)


# Feature order must match generate_sequences.py exactly.
_STEP_FEATURES = [
    "kp_index", "solar_wind_speed", "orbit_deviation",
    "power_output", "solar_wind_density", "b_field",
]
_WINDOW_STEPS = 6


def _extract_step(snap: dict) -> list:
    """
    Extract the 6 raw feature values from one telemetry snapshot.
    orbit_deviation and power_output are computed if not present,
    matching the same formula used in generate_sequences.py to avoid
    any train/inference mismatch.
    """
    kp   = snap["kp_index"]
    wind = snap["solar_wind_speed"]
    sail = snap.get("sail_angle", 45)
    dens = snap.get("solar_wind_density", 5.0)
    bz   = snap.get("b_field", -5.0)

    drag      = (sail / 90) * (1 + kp * 0.1)
    orb_dev   = snap.get("orbit_deviation",
                         round(drag * (wind / 400) * 0.5, 4))
    power_out = snap.get("power_output",
                         round(math.cos(math.radians(sail)) * 100, 4))

    return [kp, wind, orb_dev, power_out, dens, bz]


def _build_features(window: list) -> list:
    """
    Flatten 6-step window into 42 features:
      36 raw values (6 steps × 6 features) + 6 delta values (t5 - t0).
    Strictly uses only the window steps — no future data.
    """
    steps = [_extract_step(snap) for snap in window]

    flat = []
    for step in steps:
        flat.extend(step)

    # Delta: direction of change over the window (t5 - t0)
    for i in range(len(_STEP_FEATURES)):
        flat.append(round(steps[-1][i] - steps[0][i], 4))

    return flat


class ForecastResult(TypedDict):
    critical_probability: float   # P(CRITICAL_AHEAD) at t+30
    nominal_probability:  float   # P(NOMINAL_AHEAD)  at t+30
    forecast_label:       str     # "CRITICAL_AHEAD" | "NOMINAL_AHEAD"
    forecast_confidence:  float   # max(P(critical), P(nominal))
    horizon_minutes:      int     # always 30
    look_back_minutes:    int     # always 30
    delta_kp:             float   # KP change over window (trend indicator)
    delta_power:          float   # power change over window


def forecast(window: list) -> ForecastResult:
    """
    Returns a 30-minute operational state forecast from a 6-step
    telemetry window (oldest first).

    Raises ValueError if window length != 6.
    """
    if len(window) != _WINDOW_STEPS:
        raise ValueError(
            f"forecast() requires exactly {_WINDOW_STEPS} telemetry snapshots; "
            f"got {len(window)}."
        )

    _load()

    features  = _build_features(window)
    proba     = _model.predict_proba([features])[0]   # [P(0), P(1)]
    p_nominal  = round(float(proba[0]), 4)
    p_critical = round(float(proba[1]), 4)

    label      = "CRITICAL_AHEAD" if p_critical >= 0.5 else "NOMINAL_AHEAD"
    confidence = round(max(p_nominal, p_critical), 4)

    # Human-readable trend indicators extracted from delta features
    # Delta features start at index 36 (after 6×6 raw values)
    delta_kp    = round(features[36], 3)   # delta_kp_index
    delta_power = round(features[39], 3)   # delta_power_output

    return ForecastResult(
        critical_probability=p_critical,
        nominal_probability=p_nominal,
        forecast_label=label,
        forecast_confidence=confidence,
        horizon_minutes=30,
        look_back_minutes=30,
        delta_kp=delta_kp,
        delta_power=delta_power,
    )
