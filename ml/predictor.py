"""
HelioMesh — ML Risk Predictor
==============================
Loads the trained Random Forest model and exposes a single
predict() function used by the Decision Engine pipeline.

Pipeline position:
    Telemetry → [ML Predictor] → Granite → Decision Engine
"""

import os
import math
import pickle
from typing import TypedDict

_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "risk_model.pkl")
_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

# Lazy-loaded singletons
_model   = None
_encoder = None


def _load():
    global _model, _encoder
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(_ENCODER_PATH, "rb") as f:
            _encoder = pickle.load(f)


def _build_features(telemetry: dict) -> list:
    kp   = telemetry["kp_index"]
    sail = telemetry["sail_angle"]
    wind = telemetry["solar_wind_speed"]
    dens = telemetry.get("solar_wind_density", 5.0)
    bz   = telemetry.get("b_field", -5.0)

    drag           = (sail / 90) * (1 + kp * 0.1)
    orbit_dev      = telemetry.get("orbit_deviation", drag * (wind / 400) * 0.5)
    power_out      = telemetry.get("power_output", math.cos(math.radians(sail)) * 100)
    thrust         = drag * 0.02
    geo_energy     = kp * abs(bz) / 9.0
    dyn_pressure   = 0.5 * dens * (wind ** 2) * 1e-6

    return [kp, sail, wind, dens, bz,
            drag, orbit_dev, power_out, thrust,
            geo_energy, dyn_pressure]


class MLPrediction(TypedDict):
    predicted_state:    str           # NOMINAL | STANDBY | SAFE_MODE
    probabilities:      dict          # {class: probability}
    risk_probability:   float         # P(SAFE_MODE) + P(STANDBY) combined
    model_confidence:   float         # max class probability
    feature_highlights: dict          # top drivers for this prediction


def predict(telemetry: dict) -> MLPrediction:
    """
    Returns ML-based mission risk prediction for a telemetry snapshot.
    Used by the pipeline BEFORE Granite reasoning.
    """
    _load()

    features = _build_features(telemetry)
    class_idx = _model.predict([features])[0]
    proba     = _model.predict_proba([features])[0]

    classes = list(_encoder.classes_)
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    predicted_state  = _encoder.inverse_transform([class_idx])[0]
    model_confidence = round(float(max(proba)), 4)

    # Combined risk probability (anything other than NOMINAL)
    risk_probability = round(1.0 - prob_dict.get("NOMINAL", 0.0), 4)

    # Feature highlights (top 3 drivers based on trained importances)
    importances = _model.feature_importances_
    feat_names  = [
        "kp_index", "sail_angle", "solar_wind_speed", "solar_wind_density",
        "b_field", "drag_factor", "orbit_deviation", "power_output",
        "thrust_output", "geomagnetic_energy", "dynamic_pressure",
    ]
    top3 = sorted(zip(feat_names, features, importances),
                  key=lambda x: x[2], reverse=True)[:3]
    feature_highlights = {
        name: {"value": round(float(val), 3), "importance": round(float(imp), 3)}
        for name, val, imp in top3
    }

    return MLPrediction(
        predicted_state=predicted_state,
        probabilities=prob_dict,
        risk_probability=risk_probability,
        model_confidence=model_confidence,
        feature_highlights=feature_highlights,
    )
