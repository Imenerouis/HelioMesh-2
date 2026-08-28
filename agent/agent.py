import requests
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL")

def get_token():
    """Obtain an IBM Cloud IAM access token using the configured API key."""
    response = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": API_KEY
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        }
    )
    result = response.json()
    print("Token response status:", response.status_code)
    if "access_token" not in result:
        print("Error:", result)
        raise RuntimeError(f"IBM token error: {result}")
    return result["access_token"]

def build_decision_trace(telemetry, ml_result=None, forecast_result=None,
                         risk_score=None, risk_breakdown=None,
                         policy_route=None, drift_status=None):
    """
    Granite builds a 7-section Mission Decision Trace.
    Receives two independent ML inputs plus structured evidence from the
    Decision Engine and drift monitor.

    New parameters (all optional for backwards compatibility):
      risk_score     — numeric 0-100 from Decision Engine
      risk_breakdown — dict of component scores
      policy_route   — routing decision string (auto_executed / pending_approval / escalated)
      drift_status   — "STABLE" / "MODERATE_DRIFT" / "HIGH_DRIFT"
    """
    token = get_token()

    # ── Model agreement label ──
    model_agreement = "UNKNOWN"
    model_agreement_detail = "Insufficient ML data to assess agreement."
    if ml_result and forecast_result:
        rf_high = ml_result['predicted_state'] in ("SAFE_MODE", "STANDBY")
        gb_high = forecast_result['forecast_label'] == "CRITICAL_AHEAD"
        if rf_high == gb_high:
            model_agreement = "AGREE"
            if rf_high:
                model_agreement_detail = "Both models indicate elevated/critical risk."
            else:
                model_agreement_detail = "Both models indicate nominal/low risk."
        else:
            model_agreement = "DISAGREE"
            if gb_high and not rf_high:
                model_agreement_detail = (
                    f"RF predicts {ml_result['predicted_state']} (current nominal) "
                    f"but GB predicts CRITICAL_AHEAD (rising risk in 30 min). "
                    "Treat as early warning."
                )
            else:
                model_agreement_detail = (
                    f"RF predicts {ml_result['predicted_state']} (elevated current state) "
                    f"but GB predicts NOMINAL_AHEAD (conditions may improve). "
                    "Monitor closely."
                )

    # ── Block 1: Current-state ML (Random Forest snapshot) ──
    ml_context = ""
    if ml_result:
        ml_context = f"""
CURRENT-STATE ML (Random Forest — snapshot classifier):
  Answers: "What is the operational state RIGHT NOW?"
  - Predicted State:  {ml_result['predicted_state']}
  - Risk Probability: {ml_result['risk_probability'] * 100:.1f}%  (P(not NOMINAL))
  - Model Confidence: {ml_result['model_confidence'] * 100:.1f}%
  - P(SAFE_MODE):     {ml_result['probabilities'].get('SAFE_MODE', 0) * 100:.1f}%
  - P(STANDBY):       {ml_result['probabilities'].get('STANDBY', 0) * 100:.1f}%
  - P(NOMINAL):       {ml_result['probabilities'].get('NOMINAL', 0) * 100:.1f}%
"""

    # ── Block 2: Temporal ML (Gradient Boosting 30-min predictor) ──
    forecast_context = ""
    if forecast_result:
        trend_kp    = forecast_result.get('delta_kp', 0)
        trend_power = forecast_result.get('delta_power', 0)
        kp_dir    = f"+{trend_kp:.2f}" if trend_kp >= 0 else f"{trend_kp:.2f}"
        power_dir = f"+{trend_power:.2f}" if trend_power >= 0 else f"{trend_power:.2f}"
        padded_note = " (Note: window padded - limited history)" if forecast_result.get('window_padded') else ""
        kp_trend_str    = "rising" if trend_kp > 0.2 else "falling" if trend_kp < -0.2 else "stable"
        power_trend_str = "rising" if trend_power > 2 else "falling" if trend_power < -2 else "stable"
        forecast_context = f"""
30-MINUTE TEMPORAL PREDICTOR (Gradient Boosting — supervised window classifier):
  Answers: "What operational state is expected 30 minutes from now?"
  Method: supervised classification over last 30-min telemetry window{padded_note}
  - Forecast Label:        {forecast_result['forecast_label']}
  - P(CRITICAL in 30 min): {forecast_result['critical_probability'] * 100:.1f}%
  - P(NOMINAL  in 30 min): {forecast_result['nominal_probability'] * 100:.1f}%
  - Forecast Confidence:   {forecast_result['forecast_confidence'] * 100:.1f}%
  - KP trend over window:  {kp_dir} ({kp_trend_str})
  - Power trend over window: {power_dir} W ({power_trend_str})
  IMPORTANT: This predicts HelioMesh simulation operational states, not real satellite failures.
"""

    # ── Block 3: Structured evidence ──
    risk_context = ""
    if risk_score is not None:
        rb = risk_breakdown or {}
        risk_context = f"""
RISK SCORE (Decision Engine):
  - Total Risk Score:   {risk_score}/100
  - Solar Weather:      {rb.get('solar_weather', 'N/A')}
  - Orbital Instability:{rb.get('orbital_instability', 'N/A')}
  - Power Degradation:  {rb.get('power_degradation', 'N/A')}
  - Solar Wind:         {rb.get('solar_wind', 'N/A')}
"""

    policy_context = ""
    if policy_route:
        policy_context = f"""
POLICY ROUTE (AUTHORITATIVE): {policy_route.upper()}
  This route was determined by the deterministic Decision Engine rule-set.
  The deterministic policy route is authoritative. Do not claim that the LLM
  changed or selected the route. Your role is to EXPLAIN this route using the
  evidence above, not to override or re-derive it.
"""

    drift_context = ""
    if drift_status:
        drift_context = f"""
DATA DRIFT STATUS: {drift_status}
  (Indicates whether current telemetry is within the training data distribution.)
"""

    model_agreement_context = f"""
MODEL STATUS: {model_agreement}
  {model_agreement_detail}
"""

    prompt = f"""You are HelioMesh AI, an AI-assisted satellite mission control agent.
Based on the structured evidence below, build a Mission Decision Trace.
Reason ONLY over the supplied structured evidence. Do not invent telemetry values or model results.
The POLICY ROUTE shown above (if present) was determined by the deterministic Decision Engine.
Do not claim to have made or changed that routing decision — your role is explanation only.

TELEMETRY DATA:
- Timestamp:        {telemetry['timestamp']}
- KP Index:         {telemetry['kp_index']} (scale 0-9, above 6 is critical)
- Sail Angle:       {telemetry['sail_angle']} degrees
- Solar Wind Speed: {telemetry['solar_wind_speed']} km/s
- Orbit Deviation:  {telemetry['orbit_deviation']} km
- Power Output:     {telemetry['power_output']} watts
- Status:           {telemetry['status']}
{ml_context}{forecast_context}{risk_context}{policy_context}{drift_context}{model_agreement_context}
Generate a Mission Decision Trace with these exact section headings:

1. OBSERVATION: What is the current satellite state? Reference the current-state ML prediction and telemetry values.
2. PREDICTION: What will happen in 30 minutes? Reference the temporal predictor label, confidence, and trend indicators.
3. MODEL STATUS: State whether the two models AGREE or DISAGREE on risk level. Explain what this means operationally.
4. EVIDENCE: Summarise the key numerical evidence: risk score, top risk drivers, drift status.
5. RECOMMENDED ACTION: What specific action should be taken now?
6. CONFIDENCE: State High / Medium / Low and justify briefly.
7. REASON: Explain the overall reasoning chain, linking observation to prediction to recommended action.

Use only the data provided above. Do not add information not in the telemetry or evidence blocks.
"""

    response = requests.post(
        f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "model_id": "ibm/granite-4-h-small",
            "input": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3
            },
            "project_id": PROJECT_ID
        }
    )
    
    result = response.json()
    
    if "results" in result:
        ai_response = result["results"][0]["generated_text"]
    else:
        ai_response = str(result)
    
    trace = {
        "decision_id": f"DEC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": telemetry['timestamp'],
        "telemetry_status": telemetry['status'],
        "ai_trace": ai_response,
        "generated_at": datetime.now().isoformat()
    }
    
    return trace


GRANITE_STATUS_AVAILABLE = "AVAILABLE"
GRANITE_STATUS_UNAVAILABLE = "UNAVAILABLE"


def build_granite_fallback_trace(
    telemetry,
    ml_result=None,
    forecast_result=None,
    risk_score=None,
    risk_breakdown=None,
    policy_route=None,
    drift_status=None,
    model_agreement=None,
    error_message="",
):
    """
    Deterministic 7-section trace when Granite/watsonx is unavailable.
    Routing was already decided by the Decision Engine; this is explanation-only.
    """
    agreement = model_agreement or "UNKNOWN"
    rf_state = ml_result["predicted_state"] if ml_result else "N/A"
    gb_label = forecast_result["forecast_label"] if forecast_result else "N/A"
    crit_prob = (
        forecast_result["critical_probability"] * 100
        if forecast_result
        else 0.0
    )
    rb = risk_breakdown or {}

    observation = (
        f"KP={telemetry['kp_index']}, orbit_dev={telemetry['orbit_deviation']} km, "
        f"power={telemetry['power_output']} W, status={telemetry['status']}. "
        f"RF current-state prediction: {rf_state}."
    )
    prediction = (
        f"GB 30-min forecast: {gb_label} "
        f"(P(CRITICAL)={crit_prob:.1f}%)."
    )
    if forecast_result:
        prediction += (
            f" Delta-KP={forecast_result.get('delta_kp', 0)}, "
            f"delta-power={forecast_result.get('delta_power', 0)}."
        )
    model_status = f"Model agreement: {agreement}."
    evidence = (
        f"Risk score={risk_score}/100. "
        f"Solar={rb.get('solar_weather', 'N/A')}, "
        f"Orbital={rb.get('orbital_instability', 'N/A')}, "
        f"Power={rb.get('power_degradation', 'N/A')}, "
        f"Wind={rb.get('solar_wind', 'N/A')}. "
        f"Drift status={drift_status or 'N/A'}."
    )
    recommended = (
        f"Follow deterministic policy route: {policy_route.upper() if policy_route else 'UNKNOWN'}."
    )
    confidence = (
        "High" if policy_route == "auto_executed"
        else "Medium" if policy_route == "pending_approval"
        else "Low"
    )
    reason = (
        f"Policy route {policy_route} was set by the deterministic Decision Engine. "
        f"Granite/watsonx explanation was unavailable: {error_message}"
    )

    ai_trace = f"""GRANITE STATUS: UNAVAILABLE
Reason: {error_message}

OBSERVATION: {observation}
PREDICTION: {prediction}
MODEL STATUS: {model_status}
EVIDENCE: {evidence}
RECOMMENDED ACTION: {recommended}
CONFIDENCE: {confidence}
REASON: {reason}"""

    return {
        "decision_id": f"DEC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": telemetry["timestamp"],
        "telemetry_status": telemetry["status"],
        "ai_trace": ai_trace,
        "generated_at": datetime.now().isoformat(),
        "granite_status": GRANITE_STATUS_UNAVAILABLE,
        "granite_error": error_message,
    }


def build_decision_trace_safe(
    telemetry,
    ml_result=None,
    forecast_result=None,
    risk_score=None,
    risk_breakdown=None,
    policy_route=None,
    drift_status=None,
    model_agreement=None,
):
    """
    Call Granite trace generation; on failure return a structured fallback
    without blocking the decision pipeline.
    """
    try:
        trace = build_decision_trace(
            telemetry,
            ml_result=ml_result,
            forecast_result=forecast_result,
            risk_score=risk_score,
            risk_breakdown=risk_breakdown,
            policy_route=policy_route,
            drift_status=drift_status,
        )
        trace["granite_status"] = GRANITE_STATUS_AVAILABLE
        trace["granite_error"] = None
        return trace
    except Exception as exc:
        return build_granite_fallback_trace(
            telemetry,
            ml_result=ml_result,
            forecast_result=forecast_result,
            risk_score=risk_score,
            risk_breakdown=risk_breakdown,
            policy_route=policy_route,
            drift_status=drift_status,
            model_agreement=model_agreement,
            error_message=str(exc),
        )


if __name__ == "__main__":
    test_telemetry = {
        "timestamp": datetime.now().isoformat(),
        "kp_index": 7.5,
        "sail_angle": 90,
        "solar_wind_speed": 800,
        "orbit_deviation": 1.75,
        "power_output": 0.0,
        "status": "critical"
    }
    
    print("🧠 HelioMesh AI Agent — Building Decision Trace...")
    print("=" * 50)
    
    trace = build_decision_trace(test_telemetry)
    
    print(f"Decision ID: {trace['decision_id']}")
    print(f"Status: {trace['telemetry_status']}")
    print("\n📋 AI Decision Trace:")
    print(trace['ai_trace'])