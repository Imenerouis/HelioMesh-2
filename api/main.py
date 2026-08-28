from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.omni_data import get_omni_data
from data.simulator import simulate_orbit
from agent.agent import build_decision_trace_safe
from engine.decision_engine import (route_decision, compute_risk_score,
                                    compute_risk_breakdown)
from ml.predictor   import predict as ml_predict
from ml.forecaster  import forecast as ml_forecast
from validation.drift_monitor import assess_drift
from validation.opssat_evidence import get_opssat_evidence, get_opssat_summary

app = FastAPI(title="HelioMesh API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory database for now
decisions_db   = []
telemetry_db   = []
telemetry_window = []   # rolling 6-step window for temporal predictor


@app.get("/")
def root():
    return {
        "system": "HelioMesh",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }


@app.get("/telemetry")
def get_telemetry():
    """Get current telemetry snapshot"""
    omni = get_omni_data()
    telemetry = simulate_orbit(
        kp_index=omni['kp_index'],
        sail_angle=45,
        solar_wind_speed=omni['solar_wind_speed']
    )
    telemetry["solar_wind_density"] = omni.get("solar_wind_density", 5.0)
    telemetry["b_field"]            = omni.get("b_field", -5.0)
    telemetry_db.append(telemetry)
    # Keep rolling window for temporal predictor (max 6 steps)
    telemetry_window.append(telemetry)
    if len(telemetry_window) > 6:
        telemetry_window.pop(0)
    return telemetry


@app.get("/telemetry/history")
def get_telemetry_history():
    """Get historical telemetry"""
    return {"history": telemetry_db[-10:]}


@app.post("/decision/new")
def create_decision(sail_angle: float = 45, scenario: str = "normal"):
    """Create a new decision — runs full pipeline"""

    # Override data for storm scenario
    if scenario == "storm":
        omni = {
            "timestamp": datetime.now().isoformat(),
            "kp_index": 7.5,
            "solar_wind_speed": 800,
            "solar_wind_density": 15.0,
            "b_field": -25.0,
            "status": "storm"
        }
        sail_angle = 90
    elif scenario == "warning":
        omni = {
            "timestamp": datetime.now().isoformat(),
            "kp_index": 5.0,
            "solar_wind_speed": 550,
            "solar_wind_density": 8.0,
            "b_field": -12.0,
            "status": "warning"
        }
        sail_angle = 60
    else:
        omni = get_omni_data()

    # ── Step 1: Orbital simulation ──
    telemetry = simulate_orbit(
        kp_index=omni['kp_index'],
        sail_angle=sail_angle,
        solar_wind_speed=omni['solar_wind_speed']
    )
    telemetry["solar_wind_density"] = omni.get("solar_wind_density", 5.0)
    telemetry["b_field"]            = omni.get("b_field", -5.0)

    # Keep rolling window updated (used by temporal predictor)
    telemetry_window.append(telemetry)
    if len(telemetry_window) > 6:
        telemetry_window.pop(0)

    # ── Step 2a: Snapshot ML — "What is the current state?" ──
    ml_result = ml_predict(telemetry)

    # ── Step 2b: Temporal ML — "What happens in 30 minutes?" ──
    # Requires a full 6-step window. If we don't have enough history yet,
    # pad the window with copies of the current snapshot so inference is
    # always available (clearly documented as "limited history").
    window_to_use = telemetry_window.copy()
    window_padded = False
    if len(window_to_use) < 6:
        pad = [telemetry] * (6 - len(window_to_use))
        window_to_use = pad + window_to_use
        window_padded = True

    forecast_result = ml_forecast(window_to_use)
    forecast_result["window_padded"] = window_padded   # transparency flag

    # ── Step 2c: Drift assessment ──
    drift_result = assess_drift(telemetry)
    drift_status = drift_result["drift_status"]

    # ── Step 2d: Pre-compute risk scores for Granite context ──
    risk_score = compute_risk_score(
        telemetry['kp_index'], telemetry['orbit_deviation'],
        telemetry['power_output'], telemetry['solar_wind_speed']
    )
    risk_breakdown = compute_risk_breakdown(
        telemetry['kp_index'], telemetry['orbit_deviation'],
        telemetry['power_output'], telemetry['solar_wind_speed']
    )

    # Model agreement — compute structured dict for the policy engine
    rf_label_str = ml_result['predicted_state']          # NOMINAL | STANDBY | SAFE_MODE
    gb_label_str = forecast_result['forecast_label']     # CRITICAL_AHEAD | NOMINAL_AHEAD
    crit_prob    = forecast_result['critical_probability']

    rf_high = rf_label_str in ("SAFE_MODE", "STANDBY")
    gb_high = gb_label_str == "CRITICAL_AHEAD"
    model_agreement_str = "AGREE" if rf_high == gb_high else "DISAGREE"

    # Structured dict passed to route_decision for the disagreement policy
    model_agreement_dict = {
        "rf_label":            rf_label_str,
        "gb_label":            gb_label_str,
        "critical_probability": crit_prob,
        "agreement":           model_agreement_str,
    }

    # ── Step 3: Decision Engine (runs BEFORE Granite so policy_route is known) ──
    # Granite receives the authoritative route as structured context.
    decision = route_decision(telemetry, ai_trace="",
                              model_agreement=model_agreement_dict)

    # ── Step 4: Granite reasoning (non-blocking — fallback if watsonx unavailable) ──
    trace = build_decision_trace_safe(
        telemetry, ml_result, forecast_result,
        risk_score=risk_score,
        risk_breakdown=risk_breakdown,
        policy_route=decision["status"],   # authoritative route from Decision Engine
        drift_status=drift_status,
        model_agreement=model_agreement_str,
    )
    # Patch the ai_trace into the decision record
    decision["ai_trace_summary"] = (
        trace['ai_trace'][:200] if len(trace['ai_trace']) > 200 else trace['ai_trace']
    )

    # Store in memory
    full_record = {
        **decision,
        "telemetry":       telemetry,
        "ai_trace":        trace['ai_trace'],
        "granite_status":  trace.get("granite_status"),
        "granite_error":   trace.get("granite_error"),
        "ml_prediction":   ml_result,
        "ml_forecast":     forecast_result,
        "model_agreement": model_agreement_str,
        "drift_status":    drift_status,
        "drift_details":   drift_result,
    }
    decisions_db.append(full_record)

    return full_record


@app.get("/decision")
def list_decisions(status: Optional[str] = None):
    """List all decisions, optionally filtered by status"""
    if status:
        filtered = [d for d in decisions_db if d['status'] == status]
        return {"decisions": filtered}
    return {"decisions": decisions_db}


@app.get("/decision/{decision_id}")
def get_decision(decision_id: str):
    """Get full decision detail"""
    for d in decisions_db:
        if d['decision_id'] == decision_id:
            return d
    raise HTTPException(status_code=404, detail="Decision not found")


@app.post("/approve/{decision_id}")
def approve_decision(decision_id: str):
    """Operator approves a pending decision"""
    for d in decisions_db:
        if d['decision_id'] == decision_id:
            if d['status'] != "pending_approval":
                raise HTTPException(
                    status_code=400,
                    detail=f"Decision is {d['status']} — cannot approve"
                )
            d['status'] = "auto_executed"
            d['approved_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            d['approved_by'] = "operator"
            return {
                "decision_id": decision_id,
                "status": "auto_executed",
                "message": "Decision approved and executed",
                "approved_at": d['approved_at']
            }
    raise HTTPException(status_code=404, detail="Decision not found")


@app.post("/reject/{decision_id}")
def reject_decision(decision_id: str):
    """Operator rejects a pending decision"""
    for d in decisions_db:
        if d['decision_id'] == decision_id:
            if d['status'] != "pending_approval":
                raise HTTPException(
                    status_code=400,
                    detail=f"Decision is {d['status']} — cannot reject"
                )
            d['status'] = "rejected"
            d['rejected_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            return {
                "decision_id": decision_id,
                "status": "rejected",
                "message": "Decision rejected by operator",
                "rejected_at": d['rejected_at']
            }
    raise HTTPException(status_code=404, detail="Decision not found")


@app.post("/chat")
def chat(question: str, decision_id: Optional[str] = None):
    """Operator asks the AI agent a question"""
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    API_KEY = os.getenv("WATSONX_API_KEY")
    PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_URL = os.getenv("WATSONX_URL")

    # Get context
    context = ""
    if decision_id:
        for d in decisions_db:
            if d['decision_id'] == decision_id:
                context = f"Decision ID: {d['decision_id']}\nStatus: {d['status']}\nAI Trace: {d.get('ai_trace', '')[:300]}"
                break

    # Get token
    token_response = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": API_KEY
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token_data = token_response.json()
    if "access_token" not in token_data:
        raise HTTPException(status_code=503, detail=f"IBM token error: {token_data}")
    token = token_data["access_token"]

    prompt = f"""You are HelioMesh AI assistant. Answer the operator's question based only on the provided context.

Context:
{context if context else "No specific decision context provided."}

Operator question: {question}

Answer clearly and concisely:"""

    response = requests.post(
        f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "model_id": "ibm/granite-4-h-small",
            "input": prompt,
            "parameters": {"max_new_tokens": 200, "temperature": 0.3},
            "project_id": PROJECT_ID
        }
    )

    result = response.json()
    answer = result["results"][0]["generated_text"] if "results" in result else "Unable to process question"

    return {
        "question": question,
        "answer": answer,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    }


@app.get("/validation")
def get_validation_results():
    """Return combined validation results from all stages."""
    import json
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "validation", "results")
    summary_path = os.path.join(results_dir, "validation_summary.json")
    if not os.path.exists(summary_path):
        # Try to load individual files
        result = {}
        for fname, key in [
            ("snapshot_validation.json",  "snapshot_validation"),
            ("temporal_validation.json",  "temporal_validation"),
            ("early_warning.json",        "early_warning"),
            ("model_agreement.json",      "model_agreement"),
            ("policy_tests.json",         "policy_tests"),
        ]:
            path = os.path.join(results_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    result[key] = json.load(f)
        if not result:
            raise HTTPException(status_code=404, detail="Validation results not found. Run validation/run_validation.py first.")
        return result
    with open(summary_path) as f:
        return json.load(f)


@app.get("/report")
def get_report():
    """Get mission report summary"""
    if not decisions_db:
        return {"report": "No decisions recorded yet."}

    total = len(decisions_db)
    auto = len([d for d in decisions_db if d['status'] == 'auto_executed'])
    pending = len([d for d in decisions_db if d['status'] == 'pending_approval'])
    escalated = len([d for d in decisions_db if d['status'] == 'escalated'])
    rejected = len([d for d in decisions_db if d['status'] == 'rejected'])

    return {
        "total_decisions": total,
        "auto_executed": auto,
        "pending_approval": pending,
        "escalated": escalated,
        "rejected": rejected,
        "decisions": decisions_db[-5:]
    }


@app.get("/opssat/evidence")
def get_real_evidence():
    """
    Return the verified OPS-SAT-AD real-spacecraft evidence.

    This evidence is SEPARATE from the simulation benchmark.
    OPS-SAT-AD uses BINARY anomaly labels — NOT the HelioMesh
    four-class simulation taxonomy (NOMINAL/STANDBY/SAFE_MODE/CRITICAL_AHEAD).

    All metrics are pre-computed from the official held-out test partition.
    """
    return get_opssat_evidence()


@app.get("/opssat/summary")
def get_real_evidence_summary():
    """
    Compact OPS-SAT-AD evidence summary for dashboard display.
    Key verified metrics only.
    """
    return get_opssat_summary()


@app.get("/health")
def health():
    """Health check endpoint for production deployment."""
    return {
        "status": "healthy",
        "system": "HelioMesh",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "simulation_benchmark": "FROZEN",
        "real_spacecraft_benchmark": "VERIFIED",
    }