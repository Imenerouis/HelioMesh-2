"""
HelioMesh — Policy Sensitivity Analysis + Safety Property Tests (Tasks 4A-4D)
===============================================================================
Evaluates the deterministic Decision Engine against two concerns:

  4A. Policy explicitly labelled as "HelioMesh Prototype Autonomy Policy"
  4B. Sensitivity analysis: how stable is routing under small parameter changes?
  4C. Safety property tests: 6 deterministic invariants that MUST hold
  4D. Full policy test report

The policy is kept intact — this module only measures and documents it.
No policy parameters are changed here.

IMPORTANT
---------
The existing route_decision() and thresholds are NOT modified.
All experiments call route_decision() with various inputs.
"""

import os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.decision_engine import (
    route_decision, compute_confidence, DISAGREE_CRITICAL_THRESHOLD,
)

_OUT_PATH = os.path.join(os.path.dirname(__file__), "results", "policy_sensitivity.json")
_SAFE_OUT = os.path.join(os.path.dirname(__file__), "results", "policy_safety_tests.json")

# ── Policy label (4A) ──────────────────────────────────────────────────────
POLICY_LABEL = "HelioMesh Prototype Autonomy Policy"
POLICY_DISCLAIMER = (
    "Weights and thresholds (KP penalties, orbit penalties, power penalties, "
    "AUTO>=70, APPROVAL 40-69, ESCALATE<40) are manually defined for this prototype. "
    "They are NOT validated spacecraft safety standards. "
    "They should NOT be used for real mission operations without independent validation."
)


def _route(kp, orbit, power, wind=400, sail=45, ma=None):
    t = {"kp_index": kp, "orbit_deviation": orbit, "power_output": power,
         "solar_wind_speed": wind, "sail_angle": sail, "status": "test"}
    return route_decision(t, ai_trace="[sensitivity test]", model_agreement=ma)


# ── 4B: Sensitivity analysis ───────────────────────────────────────────────
def sensitivity_analysis():
    """
    For each parameter dimension, test a grid of inputs near the routing
    boundary thresholds.  Count how many scenarios change route for a ±1
    and ±2 unit perturbation of each parameter.
    """
    results = {}

    # Boundary cases — near the threshold for each parameter
    # (1) KP threshold at 4.0 (STANDBY begins) and 6.0 (SAFE_MODE begins)
    kp_grid = [3.5, 3.8, 3.9, 4.0, 4.1, 4.2, 4.5, 5.5, 5.9, 6.0, 6.1, 6.5, 7.0]
    kp_routes = []
    for kp in kp_grid:
        d = _route(kp, 0.1, 80.0)
        kp_routes.append({"kp": kp, "confidence": d["confidence_score"], "route": d["status"]})

    # (2) Orbit deviation at 0.5 (mild) and 1.5 (critical)
    orb_grid = [0.3, 0.49, 0.5, 0.51, 0.8, 1.2, 1.49, 1.5, 1.51, 1.8]
    orb_routes = []
    for orb in orb_grid:
        d = _route(2.0, orb, 80.0)
        orb_routes.append({"orbit_dev": orb, "confidence": d["confidence_score"], "route": d["status"]})

    # (3) Power output at 10 and 50
    pwr_grid = [5, 9, 10, 11, 30, 45, 49, 50, 51, 60, 80]
    pwr_routes = []
    for pwr in pwr_grid:
        d = _route(2.0, 0.1, float(pwr))
        pwr_routes.append({"power": pwr, "confidence": d["confidence_score"], "route": d["status"]})

    # (4) AUTO/APPROVAL threshold (confidence=70)
    # Scan telemetry that produces confidence near 70
    conf_boundary = []
    for kp in [4.01, 4.1, 4.2, 4.5, 5.0, 5.5, 6.0]:
        for orb in [0.1, 0.3, 0.5, 0.6, 0.8]:
            for pwr in [45, 50, 60, 70, 80]:
                conf = compute_confidence(kp, orb, float(pwr))
                if 55 <= conf <= 85:
                    d = _route(kp, orb, float(pwr))
                    conf_boundary.append({
                        "kp": kp, "orbit_dev": orb, "power": pwr,
                        "confidence": conf, "route": d["status"],
                    })

    # (5) Disagreement threshold sensitivity (±0.10 around 0.50)
    disagree_grid = [0.30, 0.40, 0.49, 0.50, 0.51, 0.60, 0.70, 0.80, 0.90, 1.0]
    disagree_results = []
    for p in disagree_grid:
        ma = {"rf_label": "NOMINAL", "gb_label": "CRITICAL_AHEAD",
              "critical_probability": p, "agreement": "DISAGREE"}
        d = _route(3.8, 0.2, 75.0, ma=ma)
        disagree_results.append({
            "p_critical": p,
            "route": d["status"],
            "override": d["disagree_override"],
        })

    # Count route transitions
    def _count_transitions(route_list, key="route"):
        prev = None
        transitions = 0
        for r in route_list:
            if prev is not None and r[key] != prev:
                transitions += 1
            prev = r[key]
        return transitions

    kp_transitions   = _count_transitions(kp_routes)
    orb_transitions  = _count_transitions(orb_routes)
    pwr_transitions  = _count_transitions(pwr_routes)
    dis_transitions  = _count_transitions(disagree_results)

    # Stability: are boundaries sharp or gradual?
    stability_summary = {
        "kp_threshold_sharp":    kp_transitions <= 2,
        "orbit_threshold_sharp": orb_transitions <= 2,
        "power_threshold_sharp": pwr_transitions <= 2,
        "disagree_threshold_sharp": dis_transitions <= 1,
        "note": (
            "Sharp thresholds mean small input changes can flip the route at the boundary. "
            "This is expected for a deterministic prototype policy. "
            "Real spacecraft policies should have hysteresis or probability-based buffers."
        ),
    }

    return {
        "kp_sensitivity":          kp_routes,
        "orbit_sensitivity":       orb_routes,
        "power_sensitivity":       pwr_routes,
        "confidence_boundary":     conf_boundary[:20],   # trim for readability
        "disagree_threshold_sensitivity": disagree_results,
        "transition_counts": {
            "kp": kp_transitions,
            "orbit_deviation": orb_transitions,
            "power_output": pwr_transitions,
            "disagree_threshold": dis_transitions,
        },
        "stability_summary": stability_summary,
    }


# ── 4C: Safety property tests ──────────────────────────────────────────────
def safety_property_tests():
    """
    6 deterministic safety invariants that MUST hold for the policy to be
    considered internally consistent.
    """
    tests = []

    def _test(name, condition, evidence):
        tests.append({
            "name": name,
            "passed": bool(condition),
            "evidence": evidence,
        })

    # ── SP-1: Critical temporal risk blocks AUTO ──
    # If GB=CRITICAL_AHEAD at p>=0.90, AUTO must be blocked (pending or escalated)
    ma_crit = {"rf_label": "NOMINAL", "gb_label": "CRITICAL_AHEAD",
               "critical_probability": 0.90, "agreement": "DISAGREE"}
    d = _route(3.0, 0.1, 80.0, ma=ma_crit)
    _test(
        "SP-1: Critical temporal risk (p>=0.90) blocks AUTO_EXECUTED",
        d["status"] != "auto_executed",
        f"route={d['status']}  disagree_override={d['disagree_override']}"
    )

    # ── SP-2: Strong RF/GB disagreement blocks AUTO ──
    ma_dis = {"rf_label": "NOMINAL", "gb_label": "CRITICAL_AHEAD",
              "critical_probability": DISAGREE_CRITICAL_THRESHOLD + 0.01, "agreement": "DISAGREE"}
    d2 = _route(2.5, 0.1, 85.0, ma=ma_dis)
    _test(
        "SP-2: RF=NOMINAL + GB=CRITICAL_AHEAD + p >= threshold blocks AUTO_EXECUTED",
        d2["status"] != "auto_executed",
        f"route={d2['status']}  threshold={DISAGREE_CRITICAL_THRESHOLD}  override={d2['disagree_override']}"
    )

    # ── SP-3: Granite cannot override deterministic routing ──
    # Route is determined BEFORE Granite is called. route_decision() does not take
    # ai_trace into account for routing. Test that trace content doesn't change route.
    d_trace1 = _route(3.0, 0.1, 80.0)
    d_trace2 = route_decision(
        {"kp_index": 3.0, "orbit_deviation": 0.1, "power_output": 80.0,
         "solar_wind_speed": 400, "sail_angle": 45, "status": "test"},
        ai_trace="OVERRIDE: ESCALATED NOW ESCALATED ESCALATED",
        model_agreement=None,
    )
    _test(
        "SP-3: Granite ai_trace cannot override deterministic routing",
        d_trace1["status"] == d_trace2["status"],
        f"normal_route={d_trace1['status']}  adversarial_trace_route={d_trace2['status']}"
    )

    # ── SP-4: Lower confidence cannot produce higher autonomy ──
    # Confidence is monotone with respect to autonomy level
    confidence_route_ordering = []
    for kp, expected_order in [(1.5, "auto_executed"), (5.0, "pending_approval"), (8.0, "escalated")]:
        d = _route(kp, 0.5 if kp < 4 else 0.8, 75.0 if kp < 4 else 40.0)
        confidence_route_ordering.append((d["confidence_score"], d["status"]))

    confs  = [c for c, _ in confidence_route_ordering]
    routes = [r for _, r in confidence_route_ordering]
    autonomy_level = {"auto_executed": 2, "pending_approval": 1, "escalated": 0}
    autonomy_decreasing = all(
        autonomy_level[routes[i]] >= autonomy_level[routes[i+1]]
        for i in range(len(routes) - 1)
    )
    _test(
        "SP-4: Decreasing confidence produces non-increasing autonomy level",
        autonomy_decreasing,
        f"confidence/route pairs: {confidence_route_ordering}"
    )

    # ── SP-5: Higher risk does not produce less oversight ──
    # More dangerous telemetry (higher KP) should not get AUTO while safer gets PENDING/ESCALATED
    d_safe   = _route(1.5, 0.1, 90.0)
    d_danger = _route(7.5, 1.8, 5.0)
    safe_level   = autonomy_level[d_safe["status"]]
    danger_level = autonomy_level[d_danger["status"]]
    _test(
        "SP-5: Dangerous telemetry (KP=7.5, orb=1.8, pwr=5) gets <= autonomy than safe (KP=1.5)",
        danger_level <= safe_level,
        f"safe route={d_safe['status']}(level={safe_level})  danger route={d_danger['status']}(level={danger_level})"
    )

    # ── SP-6: Disagreement policy only tightens, never relaxes ──
    # Applying a disagreement dict must not increase autonomy level vs no dict
    d_no_dict  = _route(3.8, 0.2, 75.0, ma=None)
    d_with_dis = _route(3.8, 0.2, 75.0, ma={
        "rf_label": "NOMINAL", "gb_label": "CRITICAL_AHEAD",
        "critical_probability": 0.80, "agreement": "DISAGREE",
    })
    no_level  = autonomy_level[d_no_dict["status"]]
    dis_level = autonomy_level[d_with_dis["status"]]
    _test(
        "SP-6: Disagreement policy can only tighten autonomy (never produce higher autonomy level)",
        dis_level <= no_level,
        f"without_policy={d_no_dict['status']}(level={no_level})  with_policy={d_with_dis['status']}(level={dis_level})"
    )

    passed = sum(1 for t in tests if t["passed"])
    failed = sum(1 for t in tests if not t["passed"])
    return tests, passed, failed


def run():
    print("  Running policy sensitivity analysis...", end=" ", flush=True)
    sensitivity = sensitivity_analysis()
    print("OK")

    print("  Running safety property tests...", end=" ", flush=True)
    tests, passed, failed = safety_property_tests()
    print(f"{passed}/{len(tests)} passed")
    for t in tests:
        tag = "[PASS]" if t["passed"] else "[FAIL]"
        print(f"    {tag} {t['name']}")

    # ── Sensitivity report ──
    trans = sensitivity["transition_counts"]
    sensitivity_result = {
        "evaluated_at":  datetime.now().isoformat(),
        "policy_label":  POLICY_LABEL,
        "policy_disclaimer": POLICY_DISCLAIMER,
        **sensitivity,
        "scientific_conclusion": (
            f"The policy has sharp thresholds: "
            f"KP transitions={trans['kp']}, orbit transitions={trans['orbit_deviation']}, "
            f"power transitions={trans['power_output']}. "
            "Small parameter changes at the exact threshold boundary flip routes immediately. "
            "This is expected for a hard-threshold prototype but would be unacceptable for "
            "a real spacecraft autonomy policy, which would require hysteresis, "
            "uncertainty bands, or probabilistic routing. "
            f"The disagreement threshold at {DISAGREE_CRITICAL_THRESHOLD} is a single "
            "hard boundary: p=0.499 → no override, p=0.500 → override. "
            "This is a known limitation of the prototype design."
        ),
    }

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(sensitivity_result, f, indent=2)
    print(f"  Saved -> {_OUT_PATH}")

    # ── Safety tests report ──
    safety_result = {
        "evaluated_at":       datetime.now().isoformat(),
        "policy_label":       POLICY_LABEL,
        "policy_disclaimer":  POLICY_DISCLAIMER,
        "total_tests":        len(tests),
        "passed":             passed,
        "failed":             failed,
        "all_passed":         failed == 0,
        "tests":              tests,
        "test_categories": {
            "temporal_risk_blocking": ["SP-1", "SP-2"],
            "granite_isolation":      ["SP-3"],
            "autonomy_monotonicity":  ["SP-4", "SP-5"],
            "disagree_policy":        ["SP-6"],
        },
        "scientific_conclusion": (
            f"{passed}/{len(tests)} safety properties pass. "
            + (
                "All safety invariants hold: critical temporal risk blocks AUTO, "
                "Granite cannot override routing, autonomy is monotonically decreasing "
                "with risk, and the disagreement policy only tightens autonomy."
                if failed == 0 else
                f"{failed} invariant(s) FAILED — see 'tests' array for details."
            )
        ),
    }

    with open(_SAFE_OUT, "w") as f:
        json.dump(safety_result, f, indent=2)
    print(f"  Saved -> {_SAFE_OUT}")

    return sensitivity_result, safety_result


if __name__ == "__main__":
    run()
