"""
HelioMesh â€” Model Agreement Analysis
======================================
Cross-tabulates the Random Forest snapshot classifier against the Gradient
Boosting temporal predictor on the simulation test sequences.

For each test sequence:
  - The RF classifier receives the LAST snapshot of the window (row t5 features)
    as a current-state assessment.
  - The GB predictor receives the full 6-step window for its 30-min forecast.

Both predictions are collected and compared.

Results saved to: validation/results/model_agreement.json
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import math
import pickle
from collections import Counter
from datetime import datetime

_ML_DIR        = os.path.join(os.path.dirname(__file__), "..", "ml")
_RF_MODEL_PATH = os.path.join(_ML_DIR, "risk_model.pkl")
_RF_ENC_PATH   = os.path.join(_ML_DIR, "label_encoder.pkl")
_GB_MODEL_PATH = os.path.join(_ML_DIR, "forecaster_model.pkl")
_SEQ_PATH      = os.path.join(_ML_DIR, "sequences.csv")
_OUT_PATH      = os.path.join(os.path.dirname(__file__), "results", "model_agreement.json")

TEST_START = 10200   # last 15% of 12000 sequences

DEFAULT_SAIL = 45.0  # assumed sail angle for RF snapshot reconstruction


def _build_rf_features(row: dict) -> list[float]:
    """
    Reconstruct the RF feature vector from the t5 (last step) values in the
    sequence row. Uses the same formula as ml/predictor.py :: _build_features().
    """
    kp   = float(row["kp_index_t5"])
    wind = float(row["solar_wind_speed_t5"])
    dens = float(row["solar_wind_density_t5"])
    bz   = float(row["b_field_t5"])
    sail = DEFAULT_SAIL

    drag       = (sail / 90) * (1 + kp * 0.1)
    orbit_dev  = float(row.get("orbit_deviation_t5",
                               round(drag * (wind / 400) * 0.5, 4)))
    power_out  = float(row.get("power_output_t5",
                               round(math.cos(math.radians(sail)) * 100, 4)))
    thrust     = drag * 0.02
    geo_energy = kp * abs(bz) / 9.0
    dyn_press  = 0.5 * dens * (wind ** 2) * 1e-6

    return [kp, sail, wind, dens, bz,
            drag, orbit_dev, power_out, thrust,
            geo_energy, dyn_press]


def _build_gb_features(row: dict) -> list[float]:
    """Extract the 42 precomputed GB features from a CSV row."""
    from ml.generate_sequences import FEATURE_NAMES
    return [float(row[name]) for name in FEATURE_NAMES]


def run() -> dict:
    print("  Loading models...", end=" ", flush=True)
    with open(_RF_MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
    with open(_RF_ENC_PATH, "rb") as f:
        rf_enc = pickle.load(f)
    with open(_GB_MODEL_PATH, "rb") as f:
        gb_model = pickle.load(f)
    print("OK")

    print("  Loading sequences...", end=" ", flush=True)
    with open(_SEQ_PATH, newline="") as f:
        all_rows = list(csv.DictReader(f))
    test_rows = all_rows[TEST_START:]
    print(f"{len(test_rows)} test rows")

    rf_preds  = []
    gb_labels = []   # NOMINAL_AHEAD / CRITICAL_AHEAD
    gt_labels = []   # ground truth from sequences

    for row in test_rows:
        # RF: current state
        rf_feat = _build_rf_features(row)
        rf_idx  = rf_model.predict([rf_feat])[0]
        rf_pred = rf_enc.inverse_transform([rf_idx])[0]

        # GB: future state
        gb_feat = _build_gb_features(row)
        gb_proba = gb_model.predict_proba([gb_feat])[0]
        gb_pred  = "CRITICAL_AHEAD" if gb_proba[1] >= 0.5 else "NOMINAL_AHEAD"

        rf_preds.append(rf_pred)
        gb_labels.append(gb_pred)
        gt_labels.append(int(row["label"]))

    n = len(rf_preds)

    # Agreement: RF says NOMINAL â†” GB says NOMINAL_AHEAD, or RF non-NOMINAL â†” GB CRITICAL_AHEAD
    def _risk_level(rf: str) -> str:
        return "HIGH_RISK" if rf in ("SAFE_MODE", "STANDBY") else "LOW_RISK"

    def _gb_risk_level(gb: str) -> str:
        return "HIGH_RISK" if gb == "CRITICAL_AHEAD" else "LOW_RISK"

    agree = sum(
        1 for r, g in zip(rf_preds, gb_labels)
        if _risk_level(r) == _gb_risk_level(g)
    )

    # Cross-tabulation counts
    rf_nom_gb_crit  = sum(
        1 for r, g in zip(rf_preds, gb_labels)
        if r == "NOMINAL" and g == "CRITICAL_AHEAD"
    )
    rf_nonnominal_gb_nom = sum(
        1 for r, g in zip(rf_preds, gb_labels)
        if r != "NOMINAL" and g == "NOMINAL_AHEAD"
    )

    # Full cross-tab
    crosstab = {}
    for r in ("NOMINAL", "STANDBY", "SAFE_MODE"):
        crosstab[r] = {}
        for g in ("NOMINAL_AHEAD", "CRITICAL_AHEAD"):
            crosstab[r][g] = sum(
                1 for rp, gp in zip(rf_preds, gb_labels)
                if rp == r and gp == g
            )

    result = {
        "evaluated_at":            datetime.now().isoformat(),
        "n_sequences":             n,
        "agreement_rate":          round(agree / n, 4),
        "disagreement_rate":       round(1 - agree / n, 4),
        "rf_nominal_gb_critical":  rf_nom_gb_crit,
        "rf_non_nominal_gb_nominal": rf_nonnominal_gb_nom,
        "cross_tabulation":        crosstab,
        "rf_distribution":         dict(Counter(rf_preds)),
        "gb_distribution":         dict(Counter(gb_labels)),
        "agreement_definition": (
            "Agreement = RF risk level (NOMINALâ†”LOW_RISK, otherâ†”HIGH_RISK) "
            "matches GB risk level (NOMINAL_AHEADâ†”LOW_RISK, CRITICAL_AHEADâ†”HIGH_RISK)."
        ),
        "disclaimer": (
            "Both models are evaluated on simulation test sequences. "
            "NOT validated against real spacecraft data."
        ),
    }

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  âœ“ Agreement rate:      {result['agreement_rate']*100:.1f}%")
    print(f"  âœ“ Disagreement rate:   {result['disagreement_rate']*100:.1f}%")
    print(f"  âœ“ RF-NOMINAL / GB-CRITICAL: {rf_nom_gb_crit}")
    print(f"  âœ“ RF-non-NOMINAL / GB-NOMINAL: {rf_nonnominal_gb_nom}")
    print(f"  âœ“ Saved â†’ {_OUT_PATH}")
    return result


if __name__ == "__main__":
    run()

