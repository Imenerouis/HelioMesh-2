"""
HelioMesh — Model Disagreement Demo (end-to-end pipeline run)
==============================================================
Re-runs the known RF=NOMINAL / GB=CRITICAL_AHEAD disagreement case through
the ACTUAL ml.predictor, ml.forecaster, engine.decision_engine pipeline
to verify that the disagreement autonomy policy routes correctly.

Scenario:
  KP window rising from 2.0 → 3.8 over 6 steps (30 min).
  RF snapshot at current KP=3.8 classifies NOMINAL.
  GB temporal model sees delta_kp=+1.8 → CRITICAL_AHEAD.
  Expected route BEFORE policy fix : auto_executed
  Expected route AFTER  policy fix : pending_approval

Output saved to: validation/results/model_disagreement_demo.json
"""

import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.predictor  import predict  as ml_predict
from ml.forecaster import forecast as ml_forecast
from engine.decision_engine import (
    route_decision,
    compute_risk_score,
    compute_risk_breakdown,
    DISAGREE_CRITICAL_THRESHOLD,
)

_OUT_PATH = os.path.join(os.path.dirname(__file__), "results", "model_disagreement_demo.json")

# ---------------------------------------------------------------------------
# Build the 6-step telemetry window (exact live-demo values)
# ---------------------------------------------------------------------------
KP_WINDOW   = [2.0, 2.3, 2.7, 3.0, 3.4, 3.8]
SAIL_ANGLE  = 45
WIND_SPEED  = 500
SW_DENSITY  = 5.0
B_FIELD     = -5.0

window = []
for kp in KP_WINDOW:
    window.append({
        "kp_index":           kp,
        "solar_wind_speed":   WIND_SPEED,
        "sail_angle":         SAIL_ANGLE,
        "solar_wind_density": SW_DENSITY,
        "b_field":            B_FIELD,
        "status":             "nominal",
    })

# Current-state snapshot = last step in window
current = window[-1]

# ---------------------------------------------------------------------------
# Step 1 — RF snapshot classification
# ---------------------------------------------------------------------------
print("Running RF snapshot classifier...", end=" ", flush=True)
rf_result = ml_predict(current)
print(f"{rf_result['predicted_state']}  "
      f"(conf={rf_result['model_confidence']*100:.1f}%)")

# ---------------------------------------------------------------------------
# Step 2 — GB temporal forecast
# ---------------------------------------------------------------------------
print("Running GB temporal forecaster...", end=" ", flush=True)
gb_result = ml_forecast(window)
print(f"{gb_result['forecast_label']}  "
      f"(p_crit={gb_result['critical_probability']*100:.1f}%)")

# ---------------------------------------------------------------------------
# Step 3 — Compute risk scores
# ---------------------------------------------------------------------------
risk_score     = compute_risk_score(
    current["kp_index"], 0.2, 75.0, WIND_SPEED)
risk_breakdown = compute_risk_breakdown(
    current["kp_index"], 0.2, 75.0, WIND_SPEED)

# ---------------------------------------------------------------------------
# Step 4 — Route through Decision Engine WITHOUT model_agreement (baseline)
# ---------------------------------------------------------------------------
telem = {**current, "orbit_deviation": 0.2, "power_output": 75.0}
decision_without = route_decision(telem, ai_trace="[disagree demo]",
                                  model_agreement=None)

# ---------------------------------------------------------------------------
# Step 5 — Build model_agreement dict and route WITH the policy
# ---------------------------------------------------------------------------
model_agreement_dict = {
    "rf_label":            rf_result["predicted_state"],
    "gb_label":            gb_result["forecast_label"],
    "critical_probability": gb_result["critical_probability"],
    "agreement":           (
        "AGREE" if (
            (rf_result["predicted_state"] in ("SAFE_MODE", "STANDBY"))
            == (gb_result["forecast_label"] == "CRITICAL_AHEAD")
        ) else "DISAGREE"
    ),
}

decision_with = route_decision(telem, ai_trace="[disagree demo]",
                               model_agreement=model_agreement_dict)

# ---------------------------------------------------------------------------
# Assemble result
# ---------------------------------------------------------------------------
policy_triggered_correctly = (
    decision_without["status"] == "auto_executed"
    and decision_with["status"]    == "pending_approval"
)

result = {
    "scenario":          "rf_nominal_gb_critical_disagreement",
    "evaluated_at":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "description": (
        "Window KP rising 2.0 → 3.8 over 6 steps (30 min). "
        "RF snapshot at current KP=3.8 classifies NOMINAL (below 4.0 threshold). "
        "GB temporal model sees rising KP trend (delta_kp=+1.8) → CRITICAL_AHEAD."
    ),
    "window_kp_steps": KP_WINDOW,
    "current_kp":      float(current["kp_index"]),

    # RF results
    "rf_predicted_state":      rf_result["predicted_state"],
    "rf_model_confidence_pct": round(rf_result["model_confidence"] * 100, 1),
    "rf_p_nominal_pct":        round(rf_result["probabilities"].get("NOMINAL", 0) * 100, 1),
    "rf_p_standby_pct":        round(rf_result["probabilities"].get("STANDBY", 0) * 100, 1),
    "rf_p_safe_mode_pct":      round(rf_result["probabilities"].get("SAFE_MODE", 0) * 100, 1),

    # GB results
    "gb_forecast_label":          gb_result["forecast_label"],
    "gb_critical_probability_pct": round(gb_result["critical_probability"] * 100, 1),
    "gb_forecast_confidence_pct":  round(gb_result["forecast_confidence"] * 100, 1),
    "gb_delta_kp":                 gb_result["delta_kp"],

    # Agreement
    "model_agreement":    model_agreement_dict["agreement"],

    # Risk
    "risk_score":     risk_score,
    "risk_breakdown": risk_breakdown,

    # Routing — before and after policy
    "route_without_policy": {
        "status":           decision_without["status"],
        "confidence_score": decision_without["confidence_score"],
        "action_message":   decision_without["action_message"],
        "disagree_override": decision_without["disagree_override"],
    },
    "route_with_policy": {
        "status":            decision_with["status"],
        "confidence_score":  decision_with["confidence_score"],
        "action_message":    decision_with["action_message"],
        "disagree_override": decision_with["disagree_override"],
        "decision_source":   decision_with["decision_source"],
    },

    "disagree_threshold_used": DISAGREE_CRITICAL_THRESHOLD,
    "policy_triggered_correctly": policy_triggered_correctly,
    "interpretation": (
        "RF snapshot (current state): KP=3.8 is below the 4.0 STANDBY threshold → NOMINAL. "
        "GB temporal predictor (30-min window): delta_kp=+1.8 signals a rapidly rising storm "
        "→ CRITICAL_AHEAD. "
        "The disagreement autonomy policy detects RF=NOMINAL ∧ GB=CRITICAL_AHEAD ∧ "
        f"p_crit ≥ {DISAGREE_CRITICAL_THRESHOLD:.0%} and forces PENDING_APPROVAL, "
        "blocking AUTO_EXECUTE so an operator must review before any action is taken. "
        "171 equivalent cases exist in the simulation test set "
        "(validation/results/model_agreement.json, rf_nominal_gb_critical=171)."
    ),
}

os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
with open(_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("  Model Disagreement Demo — End-to-End Pipeline")
print("=" * 60)
print(f"  RF predicted state : {rf_result['predicted_state']}  "
      f"({rf_result['model_confidence']*100:.1f}% conf)")
print(f"  GB forecast label  : {gb_result['forecast_label']}  "
      f"(p_crit={gb_result['critical_probability']*100:.1f}%)")
print(f"  Model agreement    : {model_agreement_dict['agreement']}")
print(f"  delta_kp           : +{gb_result['delta_kp']}")
print()
print(f"  Route WITHOUT policy : {decision_without['status']}"
      f"  (conf={decision_without['confidence_score']})")
print(f"  Route WITH policy    : {decision_with['status']}"
      f"  (override={decision_with['disagree_override']})")
print()
if policy_triggered_correctly:
    print("  [PASS] POLICY TRIGGERED CORRECTLY:")
    print("    auto_executed -> pending_approval (disagree_override=True)")
else:
    print("  [FAIL] POLICY TRIGGER FAILED -- check engine/decision_engine.py")
print(f"\n  Saved -> {_OUT_PATH}")
