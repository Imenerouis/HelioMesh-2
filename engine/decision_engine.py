import uuid
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Disagreement autonomy policy
# ---------------------------------------------------------------------------
# When the current-state RF classifier predicts NOMINAL but the 30-minute
# gradient-boost forecaster predicts CRITICAL_AHEAD, the models disagree
# about the near-future trajectory.  If the forecaster's critical_probability
# also meets or exceeds DISAGREE_CRITICAL_THRESHOLD the pipeline must NOT
# auto-execute: the route is forced to at least PENDING_APPROVAL so a human
# operator can review before any action is taken.
#
# This rule is evaluated BEFORE the normal confidence-score tier check so
# that a high confidence_score cannot override it.
#
# Configurable threshold (default 0.50 = 50 %).  Lower values make the
# policy more conservative; raise only with documented justification.
DISAGREE_CRITICAL_THRESHOLD: float = 0.50


def compute_confidence(kp_index, orbit_deviation, power_output):
    score = 100
    if kp_index > 6:
        score -= 40
    elif kp_index > 4:
        score -= 20
    else:
        score -= 5
    if orbit_deviation > 1.5:
        score -= 30
    elif orbit_deviation > 0.5:
        score -= 15
    else:
        score -= 5
    if power_output < 10:
        score -= 20
    elif power_output < 50:
        score -= 10
    return max(0, min(100, score))


def compute_risk_score(kp_index, orbit_deviation, power_output, solar_wind_speed):
    score = 0
    score += min(kp_index / 9 * 40, 40)
    score += min(orbit_deviation / 2 * 30, 30)
    score += min((100 - power_output) / 100 * 20, 20)
    score += min(solar_wind_speed / 1000 * 10, 10)
    return round(min(100, score), 1)


def compute_risk_breakdown(kp_index, orbit_deviation, power_output, solar_wind_speed):
    solar   = round(min(kp_index / 9 * 40, 40), 1)
    orbital = round(min(orbit_deviation / 2 * 30, 30), 1)
    power   = round(min((100 - power_output) / 100 * 20, 20), 1)
    wind    = round(min(solar_wind_speed / 1000 * 10, 10), 1)
    total   = round(min(100, solar + orbital + power + wind), 1)
    return {
        "solar_weather":      solar,
        "orbital_instability": orbital,
        "power_degradation":  power,
        "solar_wind":         wind,
        "total":              total,
    }


def get_risk_label(risk_score):
    if risk_score <= 25:
        return "LOW"
    elif risk_score <= 50:
        return "MODERATE"
    elif risk_score <= 75:
        return "HIGH"
    else:
        return "CRITICAL"


def get_mission_mode(status, confidence_score):
    if status == "escalated":
        return "SAFE MODE"
    elif status == "pending_approval":
        return "STANDBY MODE"
    else:
        return "NOMINAL MODE"


def get_subsystem_commands(kp_index, orbit_deviation, power_output, sail_angle):
    recommendations = []

    if kp_index > 6:
        recommendations.append("Consider enabling radiation protection")
        recommendations.append(
            f"Recommended sail-angle target: {max(0, sail_angle - 30)}° (drag reduction)"
        )
    elif kp_index > 4:
        recommendations.append(
            f"Recommended sail-angle target: {sail_angle}° (hold current)"
        )
        recommendations.append("Monitor radiation levels")

    else:
        recommendations.append(
            f"Recommended sail-angle target: {min(90, sail_angle + 10)}° (optimize power)"
        )

    if orbit_deviation > 1.5:
        recommendations.append("Consider orbit-correction response")

    if power_output < 10:
        recommendations.append("Consider reducing payload power to minimum")
        recommendations.append("Consider switching to emergency power mode")

    return recommendations


def build_reason(kp_index, orbit_deviation, power_output, solar_wind_speed):
    reasons = []
    if kp_index > 6:
        reasons.append(f"KP Index = {kp_index} (CRITICAL — above 6)")
    elif kp_index > 4:
        reasons.append(f"KP Index = {kp_index} (WARNING — above 4)")
    else:
        reasons.append(f"KP Index = {kp_index} (nominal)")
    if orbit_deviation > 1.5:
        reasons.append(f"Orbit Deviation = {orbit_deviation} km (CRITICAL — above 1.5 km)")
    elif orbit_deviation > 0.5:
        reasons.append(f"Orbit Deviation = {orbit_deviation} km (WARNING — above 0.5 km)")
    else:
        reasons.append(f"Orbit Deviation = {orbit_deviation} km (nominal)")
    if power_output < 10:
        reasons.append(f"Power Output = {power_output} W (CRITICAL — below 10 W)")
    elif power_output < 50:
        reasons.append(f"Power Output = {power_output} W (WARNING — below 50 W)")
    else:
        reasons.append(f"Power Output = {power_output} W (nominal)")
    reasons.append(f"Solar Wind Speed = {solar_wind_speed} km/s")
    return reasons


def route_decision(telemetry, ai_trace, model_agreement=None):
    start_time = time.time()

    confidence_score = compute_confidence(
        kp_index=telemetry['kp_index'],
        orbit_deviation=telemetry['orbit_deviation'],
        power_output=telemetry['power_output']
    )

    risk_score = compute_risk_score(
        kp_index=telemetry['kp_index'],
        orbit_deviation=telemetry['orbit_deviation'],
        power_output=telemetry['power_output'],
        solar_wind_speed=telemetry['solar_wind_speed']
    )

    risk_breakdown = compute_risk_breakdown(
        kp_index=telemetry['kp_index'],
        orbit_deviation=telemetry['orbit_deviation'],
        power_output=telemetry['power_output'],
        solar_wind_speed=telemetry['solar_wind_speed']
    )

    risk_label = get_risk_label(risk_score)

    reasons = build_reason(
        kp_index=telemetry['kp_index'],
        orbit_deviation=telemetry['orbit_deviation'],
        power_output=telemetry['power_output'],
        solar_wind_speed=telemetry['solar_wind_speed']
    )

    # ------------------------------------------------------------------
    # Disagreement autonomy policy (evaluated first — takes precedence)
    # ------------------------------------------------------------------
    # Triggered when:
    #   RF label  == "NOMINAL"         (current state appears safe)
    #   GB label  == "CRITICAL_AHEAD"  (forecast detects approaching storm)
    #   critical_probability >= DISAGREE_CRITICAL_THRESHOLD
    #
    # In this configuration auto-execution is blocked regardless of
    # confidence_score; the minimum permitted route is PENDING_APPROVAL.
    disagree_override = False
    if model_agreement is not None:
        rf_label  = model_agreement.get("rf_label", "")
        gb_label  = model_agreement.get("gb_label", "")
        crit_prob = float(model_agreement.get("critical_probability", 0.0))
        if (rf_label == "NOMINAL"
                and gb_label == "CRITICAL_AHEAD"
                and crit_prob >= DISAGREE_CRITICAL_THRESHOLD):
            disagree_override = True

    # ------------------------------------------------------------------
    # Normal confidence-score tier routing
    # ------------------------------------------------------------------
    if confidence_score >= 70:
        tier = "HIGH"
        status = "auto_executed"
        action = (
            "PROTOTYPE ACTION RECOMMENDATION: "
            "High-confidence route; no spacecraft command executed."
        )
        source = "Deterministic Policy Engine"

    elif confidence_score >= 40:
        tier = "MEDIUM"
        status = "pending_approval"
        action = (
            "PENDING APPROVAL: "
            "Prototype action recommendation awaiting human review."
        )
        source = "Deterministic Policy Engine + Human Oversight"

    else:
        tier = "LOW"
        status = "escalated"
        action = (
            "ESCALATED: "
            "Prototype action recommendation requires operator review."
        )
        source = "Deterministic Policy Engine + Human Oversight"

    # ------------------------------------------------------------------
    # Apply disagreement override AFTER normal routing
    # ------------------------------------------------------------------
    if disagree_override and status == "auto_executed":
        status = "pending_approval"
        action = (
            "PENDING APPROVAL: Model disagreement — RF=NOMINAL but "
            "GB=CRITICAL_AHEAD (p≥{:.0f}%). Operator review required."
        ).format(DISAGREE_CRITICAL_THRESHOLD * 100)
        source = "Disagreement Safety Policy + Human Oversight"

    mission_mode = get_mission_mode(status, confidence_score)

    commands = get_subsystem_commands(
        kp_index=telemetry['kp_index'],
        orbit_deviation=telemetry['orbit_deviation'],
        power_output=telemetry['power_output'],
        sail_angle=telemetry.get('sail_angle', 45)
    )

    inference_time = round(time.time() - start_time, 3)

    decision = {
        "decision_id": f"DEC-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "inference_time_s": inference_time,
        "confidence_score": confidence_score,
        "confidence_tier": tier,
        "risk_score": risk_score,
        "risk_level": risk_label,
        "risk_breakdown": risk_breakdown,
        "status": status,
        "action_message": action,
        "decision_source": source,
        "disagree_override": disagree_override,
        "mission_mode": mission_mode,
        "subsystem_commands": commands,
        "telemetry_status": telemetry['status'],
        "reasons": reasons,
        "ai_trace_summary": ai_trace[:200] if len(ai_trace) > 200 else ai_trace
    }

    return decision


def print_decision(decision):
    print("\n" + "="*55)
    print("   HELIOMESH DECISION ENGINE")
    print("="*55)
    print(f"Decision ID     : {decision['decision_id']}")
    print(f"Timestamp       : {decision['timestamp']}")
    print(f"Inference Time  : {decision['inference_time_s']} s")
    print(f"Decision Source : {decision['decision_source']}")
    print("-"*55)
    print(f"Confidence      : {decision['confidence_score']}/100")
    print(f"Confidence Tier : {decision['confidence_tier']}")
    print(f"Risk Score      : {decision['risk_score']}/100")
    print(f"Risk Level      : {decision['risk_level']}")
    print(f"Status          : {decision['status'].upper()}")
    print(f"Mission Mode    : {decision['mission_mode']}")
    print(f"Action          : {decision['action_message']}")
    print("\nReason:")
    for r in decision['reasons']:
        print(f"  - {r}")
    print("\nPrototype Action Recommendations:")
    for cmd in decision['subsystem_commands']:
        print(f"  → {cmd}")
    print("="*55)


if __name__ == "__main__":

    print("\nTEST 1: Normal Conditions")
    telemetry_normal = {
        "kp_index": 2.0,
        "orbit_deviation": 0.1,
        "power_output": 80.0,
        "solar_wind_speed": 400,
        "sail_angle": 45,
        "status": "nominal"
    }
    print_decision(route_decision(telemetry_normal, "Normal operations."))

    print("\nTEST 2: Warning Conditions")
    telemetry_warning = {
        "kp_index": 5.0,
        "orbit_deviation": 0.8,
        "power_output": 40.0,
        "solar_wind_speed": 550,
        "sail_angle": 60,
        "status": "warning"
    }
    print_decision(route_decision(telemetry_warning, "Elevated solar activity."))

    print("\nTEST 3: Solar Storm")
    telemetry_storm = {
        "kp_index": 7.5,
        "orbit_deviation": 1.75,
        "power_output": 0.0,
        "solar_wind_speed": 800,
        "sail_angle": 90,
        "status": "critical"
    }
    print_decision(route_decision(telemetry_storm, "Critical solar storm detected."))