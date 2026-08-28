"""
decision_value_summary.py — Decision Intelligence Value Summary

Consolidates the 4-architecture ablation, policy calibration, and
disagreement analysis into a single auditable JSON artifact that
demonstrates what HelioMesh adds beyond bare anomaly detection.

ALL numbers are read from existing frozen result artifacts.
No new ML inference, no model retraining.

Outputs: validation/results/decision_value_summary.json
"""

import json
import os
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> dict:
    ablation  = load("real_decision_ablation.json")
    policy    = load("real_policy_calibration.json")
    disagree  = load("real_disagreement_value.json")

    # ── Pull ablation table ──────────────────────────────────────────────────
    tbl = {row["architecture"]: row for row in ablation["ablation_table"]}
    arch_A = tbl["A_current_only"]
    arch_C = tbl["C_disagreement_policy"]
    arch_D = tbl["D_full_heliomesh"]

    # ── Policy calibration test results ──────────────────────────────────────
    pol_test = policy["test_results"]
    baseline_pol  = pol_test["baseline"]
    calibrated_pol = pol_test["calibrated"]
    risk_aware_pol = pol_test["risk_aware"]

    # ── Disagreement analysis ─────────────────────────────────────────────────
    dis_exp = disagree["experiment_D_disagreement"]

    # ── Build consolidated summary ────────────────────────────────────────────
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": [
            "real_decision_ablation.json",
            "real_policy_calibration.json",
            "real_disagreement_value.json",
        ],
        "data_note": (
            "All numbers derived from frozen OPS-SAT-AD test partition "
            "(529 segments, 113 anomalous). No model retraining. "
            "Results are read-only consolidations of existing artifacts."
        ),

        # ── 1. Core question: what does each layer add? ───────────────────────
        "decision_layers": {
            "layer_1_detection_only": {
                "description": "RF anomaly detector, default 0.5 threshold, no policy",
                "recall":        arch_A["event_recall"],
                "precision":     arch_A["event_precision"],
                "f1":            arch_A["event_f1"],
                "missed":        arch_A["missed"],
                "false_alerts":  arch_A["false_alerts"],
                "unsafe_auto":   arch_A["unsafe_auto"],
                "utility":       arch_A["utility_baseline"],
                "what_it_does":  "Detects anomalies; all detections route to AUTO.",
            },
            "layer_2_threshold_calibration": {
                "description": (
                    "RF detector + calibrated threshold (p_esc=0.35, p_pend=0.20). "
                    "Thresholds selected by 5-fold CV on training partition only."
                ),
                "recall":       calibrated_pol["event_metrics"]["recall"],
                "precision":    calibrated_pol["event_metrics"]["precision"],
                "f1":           calibrated_pol["event_metrics"]["f1"],
                "missed":       calibrated_pol["event_metrics"]["missed"],
                "false_alerts": calibrated_pol["event_metrics"]["false_alerts"],
                "unsafe_auto":  calibrated_pol["decision_stats"]["unsafe_auto"],
                "utility":      calibrated_pol["decision_stats"]["utility_baseline"],
                "what_it_does": (
                    "Lower thresholds widen the detection net — more anomalies "
                    "are escalated or sent to PENDING, reducing unsafe AUTO decisions."
                ),
            },
            "layer_3_disagreement_gate": {
                "description": (
                    "Detection + disagreement policy "
                    "(RF=NOMINAL ∧ GB=CRITICAL_AHEAD ∧ p≥0.5 → PENDING_APPROVAL)"
                ),
                "recall":       arch_C["event_recall"],
                "precision":    arch_C["event_precision"],
                "f1":           arch_C["event_f1"],
                "missed":       arch_C["missed"],
                "false_alerts": arch_C["false_alerts"],
                "unsafe_auto":  arch_C["unsafe_auto"],
                "utility":      arch_C["utility_baseline"],
                "what_it_does": (
                    "Safety gate: blocks AUTO EXECUTE when models disagree on risk. "
                    "Recall increases because union-flag logic catches more anomalies. "
                    "False alerts increase significantly (75 vs 3 for detection-only)."
                ),
            },
            "layer_4_full_heliomesh": {
                "description": (
                    "Full architecture: calibrated threshold + disagreement gate + "
                    "deterministic policy + Granite explanation + human oversight"
                ),
                "recall":       arch_D["event_recall"],
                "precision":    arch_D["event_precision"],
                "f1":           arch_D["event_f1"],
                "missed":       arch_D["missed"],
                "false_alerts": arch_D["false_alerts"],
                "unsafe_auto":  arch_D["unsafe_auto"],
                "utility":      arch_D["utility_baseline"],
                "what_it_does": (
                    "Combined calibrated routing + disagreement blocking: "
                    "9 unsafe AUTO decisions (vs 14 baseline), recall 0.920. "
                    "Each non-trivial decision is explainable via Granite trace."
                ),
            },
        },

        # ── 2. Key improvements from baseline to full architecture ────────────
        "improvements_vs_detection_only": {
            "unsafe_auto_reduction":     arch_A["unsafe_auto"] - arch_D["unsafe_auto"],
            "unsafe_auto_reduction_pct": round(
                (arch_A["unsafe_auto"] - arch_D["unsafe_auto"]) /
                arch_A["unsafe_auto"] * 100, 1),
            "recall_gain":     round(arch_D["event_recall"] - arch_A["event_recall"], 4),
            "missed_reduction": arch_A["missed"] - arch_D["missed"],
            "false_alerts_increase": arch_D["false_alerts"] - arch_A["false_alerts"],
            "attribution_note": (
                "The 14→9 unsafe AUTO reduction and 0.876→0.920 recall improvement "
                "reflect the COMBINED effect of threshold calibration and the full "
                "policy architecture. They cannot be attributed to model disagreement alone."
            ),
        },

        # ── 3. Policy comparison ──────────────────────────────────────────────
        "policy_comparison": {
            "note": (
                "All configurations evaluated on the same frozen test partition. "
                "Utility weights are prototype values — NOT validated by mission operators."
            ),
            "baseline": {
                "thresholds": {"p_esc": 0.5, "p_pend": 0.5},
                "recall":       baseline_pol["event_metrics"]["recall"],
                "missed":       baseline_pol["event_metrics"]["missed"],
                "false_alerts": baseline_pol["event_metrics"]["false_alerts"],
                "unsafe_auto":  baseline_pol["decision_stats"]["unsafe_auto"],
                "utility":      baseline_pol["decision_stats"]["utility_baseline"],
            },
            "calibrated": {
                "thresholds": {"p_esc": 0.35, "p_pend": 0.20},
                "recall":       calibrated_pol["event_metrics"]["recall"],
                "missed":       calibrated_pol["event_metrics"]["missed"],
                "false_alerts": calibrated_pol["event_metrics"]["false_alerts"],
                "unsafe_auto":  calibrated_pol["decision_stats"]["unsafe_auto"],
                "utility":      calibrated_pol["decision_stats"]["utility_baseline"],
            },
            "risk_aware": {
                "thresholds": {"p_esc": 0.60, "p_pend": 0.35},
                "recall":       risk_aware_pol["event_metrics"]["recall"],
                "missed":       risk_aware_pol["event_metrics"]["missed"],
                "false_alerts": risk_aware_pol["event_metrics"]["false_alerts"],
                "unsafe_auto":  risk_aware_pol["decision_stats"]["unsafe_auto"],
                "utility":      risk_aware_pol["decision_stats"]["utility_baseline"],
            },
            "trade_off_summary": (
                "No policy is universally optimal. Calibrated policy minimises unsafe "
                "AUTO decisions (9) and maximises recall (0.920) at the cost of more "
                "false alerts (22 vs 3). Risk-aware achieves the highest utility score "
                "(0.804) with a balanced false-alert rate. The choice depends on "
                "mission risk tolerance and operator workload."
            ),
        },

        # ── 4. Disagreement analysis ──────────────────────────────────────────
        "disagreement_analysis": {
            "note": (
                "Disagreement = RF=NOMINAL AND GB predicts anomaly for the same segment. "
                "Source: real_disagreement_value.json, OPS-SAT test set n=529."
            ),
            "early_warning_cases":       dis_exp["early_warning"]["count"],
            "early_warning_true_pos":    dis_exp["early_warning"]["true_pos"],
            "early_warning_false_pos":   dis_exp["early_warning"]["false_pos"],
            "early_warning_precision":   dis_exp["early_warning"]["precision"],
            "late_signal_cases":         dis_exp["late_signal"]["count"],
            "late_signal_truly_anomalous": dis_exp["late_signal"]["truly_anomalous"],
            "agreement_nominal_correct": dis_exp["agreement_nominal"]["true_normal"],
            "agreement_nominal_missed":  dis_exp["agreement_nominal"]["missed"],
            "conclusion": (
                "Early-warning disagreement precision = 5.3% — not independently "
                "actionable as a standalone anomaly predictor. "
                "Disagreement is used as a CONSERVATIVE SAFETY GATE: it blocks "
                "AUTO EXECUTE when models diverge, routing to PENDING_APPROVAL "
                "for human review. This is a safety-control function, not a "
                "prediction function."
            ),
        },

        # ── 5. Central claim ─────────────────────────────────────────────────
        "central_claim": (
            "HelioMesh does not stop at detecting an anomaly. It converts detection "
            "evidence into an auditable, safety-controlled operational decision "
            "recommendation with a deterministic policy, model disagreement safety "
            "gate, calibrated routing, and Granite explanation. The result is a "
            "structured, human-in-the-loop decision pipeline — not a flag."
        ),
        "limitations": [
            "Utility weights are prototype values, not validated by real mission operators.",
            "AUTO EXECUTE is a prototype policy route label — no real spacecraft commands.",
            "Disagreement is a safety gate, not a validated standalone early-warning signal.",
            "Cross-mission generalization is unvalidated.",
        ],
    }

    out_path = os.path.join(RESULTS_DIR, "decision_value_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("decision_value_summary.json saved -> " + out_path)
    return summary


if __name__ == "__main__":
    main()
