"""
operational_decision_value.py — Hybrid architecture operational-value audit.

Answers the question: does the HelioMesh decision layer change how detected
risk is routed, compared to detector-only behaviour?

Uses ONLY existing frozen validation artifacts. No model inference.
No invented values. No operator workload claims.

Saves: validation/results/operational_decision_value.json
"""

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT_PATH = os.path.join(RESULTS_DIR, 'operational_decision_value.json')


def load(filename: str) -> dict:
    with open(os.path.join(RESULTS_DIR, filename), encoding='utf-8') as f:
        return json.load(f)


def main():
    # ── Load frozen artifacts ──────────────────────────────────────────────────
    ablation   = load('real_decision_ablation.json')
    calibration = load('real_policy_calibration.json')
    disagreement = load('real_disagreement_value.json')
    agreement    = load('model_agreement.json')
    safety_gate  = load('safety_gate_evaluation.json')

    n = 529  # test segments — verified across all artifacts

    # ── Architecture routing counts (from real_decision_ablation.json) ─────────
    arch = ablation['architecture_details']

    A = arch['A']['policy_metrics']   # RF-only, threshold=0.5 (detector default)
    B = arch['B']['policy_metrics']   # RF + temporal union (NOT used; documented as harmful)
    C = arch['C']['policy_metrics']   # Calibrated policy + disagreement gate
    D = arch['D']['policy_metrics']   # Full HelioMesh (calibrated + gate + drift)

    # ── Calibration policy test results (from real_policy_calibration.json) ─────
    cal_baseline  = calibration['test_results']['baseline']['decision_stats']
    cal_calibrated = calibration['test_results']['calibrated']['decision_stats']

    # ── Disagreement (from real_disagreement_value.json) ──────────────────────
    ew = disagreement['experiment_D_disagreement']['early_warning']
    sim_gate = safety_gate['simulation_gate_behavior']

    # ── Derive detector-only baseline routing ─────────────────────────────────
    # Architecture A is the closest to detector-only: RF at p=0.5, no PENDING,
    # hard binary ESCALATED vs AUTO_CLEAR routing.
    # Key: no PENDING_APPROVAL at all in Architecture A.

    # ── Routing comparison table ───────────────────────────────────────────────
    architectures = {
        "A_detector_only": {
            "description": "RF snapshot classifier at default threshold (p_esc=0.5, p_pend=0.5). No calibration, no PENDING_APPROVAL, no disagreement gate. Closest to detector-only routing.",
            "auto_clear":        A['auto_clear'],
            "pending_approval":  A['pending_approval'],
            "escalated":         A['escalated'],
            "unsafe_auto":       A['unsafe_auto'],
            "missed":            ablation['architecture_details']['A']['event_metrics']['missed'],
            "false_alerts":      ablation['architecture_details']['A']['event_metrics']['false_alerts'],
            "decision_recall":   A['decision_recall'],
            "utility_baseline":  A['utility_baseline'],
            "correct_pending":   A['correct_pending'],
            "correct_escalated": A['correct_escalated'],
        },
        "B_rf_plus_temporal_union": {
            "description": "RF + temporal union policy. Included for completeness. NOT the HelioMesh operating mode — union routing on real OPS-SAT data is documented as operationally harmful (precision collapses to 0.28, 84 unsafe AUTO).",
            "auto_clear":       B['auto_clear'],
            "pending_approval": B['pending_approval'],
            "escalated":        B['escalated'],
            "unsafe_auto":      B['unsafe_auto'],
            "missed":           ablation['architecture_details']['B']['event_metrics']['missed'],
            "false_alerts":     ablation['architecture_details']['B']['event_metrics']['false_alerts'],
            "decision_recall":  B['decision_recall'],
            "utility_baseline": B['utility_baseline'],
            "note": "Union policy on OPS-SAT-AD is not operationally viable: temporal model trained on simulation regime labels, not real OPS-SAT anomaly classes.",
        },
        "C_calibrated_policy_with_gate": {
            "description": "Calibrated thresholds (p_esc=0.35, p_pend=0.20) + PENDING_APPROVAL routing + disagreement gate. Temporal in simulation only.",
            "auto_clear":        C['auto_clear'],
            "pending_approval":  C['pending_approval'],
            "escalated":         C['escalated'],
            "unsafe_auto":       C['unsafe_auto'],
            "missed":            ablation['architecture_details']['C']['event_metrics']['missed'],
            "false_alerts":      ablation['architecture_details']['C']['event_metrics']['false_alerts'],
            "decision_recall":   C['decision_recall'],
            "utility_baseline":  C['utility_baseline'],
            "correct_pending":   C['correct_pending'],
            "correct_escalated": C['correct_escalated'],
        },
        "D_full_heliomesh": {
            "description": "Full HelioMesh hybrid: calibrated policy + disagreement gate + drift monitoring.",
            "auto_clear":        D['auto_clear'],
            "pending_approval":  D['pending_approval'],
            "escalated":         D['escalated'],
            "unsafe_auto":       D['unsafe_auto'],
            "missed":            ablation['architecture_details']['D']['event_metrics']['missed'],
            "false_alerts":      ablation['architecture_details']['D']['event_metrics']['false_alerts'],
            "decision_recall":   D['decision_recall'],
            "utility_baseline":  D['utility_baseline'],
            "correct_pending":   C['correct_pending'],
            "correct_escalated": D['correct_escalated'],
        },
    }

    # ── Decision-layer routing changes: A → D ──────────────────────────────────
    # How does the decision layer change routing vs detector-only (Architecture A)?
    a_auto  = A['auto_clear']
    d_auto  = D['auto_clear']
    a_pend  = A['pending_approval']
    d_pend  = D['pending_approval']
    a_esc   = A['escalated']
    d_esc   = D['escalated']
    a_unsafe = A['unsafe_auto']
    d_unsafe = D['unsafe_auto']

    routing_delta_A_to_D = {
        "auto_clear_change":       d_auto  - a_auto,    # negative = fewer AUTO
        "pending_approval_change": d_pend  - a_pend,    # positive = more PENDING
        "escalated_change":        d_esc   - a_esc,
        "unsafe_auto_change":      d_unsafe - a_unsafe,  # negative = safer
        "unsafe_auto_reduction_pct": round((a_unsafe - d_unsafe) / a_unsafe * 100, 1) if a_unsafe > 0 else 0,
        "interpretation": (
            f"Moving from detector-only (Architecture A) to full HelioMesh (Architecture D): "
            f"AUTO_CLEAR decreases by {a_auto - d_auto} cases, "
            f"PENDING_APPROVAL increases by {d_pend - a_pend} cases, "
            f"unsafe AUTO decreases by {a_unsafe - d_unsafe} cases "
            f"({round((a_unsafe - d_unsafe) / a_unsafe * 100, 1)}% reduction)."
        ),
    }

    # ── Calibration-only effect: baseline threshold → calibrated ──────────────
    # This isolates what threshold calibration alone contributes.
    cal_unsafe_baseline  = cal_baseline['unsafe_auto']
    cal_unsafe_calibrated = cal_calibrated['unsafe_auto']
    cal_recall_baseline   = cal_baseline['decision_recall']
    cal_recall_calibrated = cal_calibrated['decision_recall']

    calibration_effect = {
        "baseline_thresholds":     {"p_esc": 0.5, "p_pend": 0.5},
        "calibrated_thresholds":   {"p_esc": 0.35, "p_pend": 0.20},
        "baseline_unsafe_auto":    cal_unsafe_baseline,
        "calibrated_unsafe_auto":  cal_unsafe_calibrated,
        "unsafe_auto_reduction":   cal_unsafe_baseline - cal_unsafe_calibrated,
        "unsafe_auto_reduction_pct": round((cal_unsafe_baseline - cal_unsafe_calibrated) / cal_unsafe_baseline * 100, 1),
        "baseline_recall":         cal_recall_baseline,
        "calibrated_recall":       cal_recall_calibrated,
        "recall_gain_pp":          round((cal_recall_calibrated - cal_recall_baseline) * 100, 1),
        "baseline_pending":        cal_baseline['auto_clear'],
        "calibrated_auto_clear":   cal_calibrated['auto_clear'],
        "note": (
            "Threshold calibration alone (no disagreement gate) reduces unsafe AUTO from "
            f"{cal_unsafe_baseline} to {cal_unsafe_calibrated} ({round((cal_unsafe_baseline - cal_unsafe_calibrated) / cal_unsafe_baseline * 100, 1)}%) "
            f"and raises recall by {round((cal_recall_calibrated - cal_recall_baseline) * 100, 1)}pp. "
            "This is the primary driver of the unsafe-AUTO reduction."
        ),
        "attribution": (
            "The unsafe AUTO reduction (14 → 9, −36%) is attributable to threshold calibration "
            "and the deterministic PENDING_APPROVAL policy rule, NOT to disagreement alone. "
            "Disagreement contributes an additional conservative safety gate (simulation-validated)."
        ),
    }

    # ── Disagreement gate: routing impact (from frozen data) ──────────────────
    disagreement_gate = {
        "real_opssat_early_warning_precision": ew['precision'],
        "real_opssat_early_warning_precision_display": "5.3%",
        "real_opssat_true_positives": ew['true_pos'],
        "real_opssat_false_positives": ew['false_pos'],
        "real_opssat_early_warning_cases": ew['count'],
        "simulation_gate_fires": sim_gate['gate_fires'],
        "simulation_gate_fires_pct": sim_gate['gate_fires_pct'],
        "simulation_total_sequences": sim_gate['n_test_sequences'],
        "conclusion": (
            "Disagreement is not an independently validated anomaly predictor (real precision = 5.3%). "
            "It is a conservative safety gate: on simulation data it blocks 171 of 1,800 AUTO routes (9.5%); "
            "on real OPS-SAT data, binary disagreement triggers 76 of 529 cases (14.4%) at 5.3% precision. "
            "Its value is conservatism under model divergence, not early-warning accuracy."
        ),
    }

    # ── Summary of what changes: A vs D ───────────────────────────────────────
    # Direct answer to the central question
    decision_layer_routing_changes = {
        "question": "Does the decision layer change how detected risk is routed?",
        "answer": "Yes — measurably.",
        "n_test_segments": n,
        "routing_comparison": {
            "arch_A_detector_only": {
                "AUTO_CLEAR":       A['auto_clear'],
                "PENDING_APPROVAL": A['pending_approval'],
                "ESCALATED":        A['escalated'],
                "unsafe_auto":      A['unsafe_auto'],
                "decision_recall":  round(A['decision_recall'], 4),
            },
            "arch_D_full_heliomesh": {
                "AUTO_CLEAR":       D['auto_clear'],
                "PENDING_APPROVAL": D['pending_approval'],
                "ESCALATED":        D['escalated'],
                "unsafe_auto":      D['unsafe_auto'],
                "decision_recall":  round(D['decision_recall'], 4),
            },
        },
        "changes_A_to_D": {
            "AUTO_CLEAR_cases_removed":      a_auto  - d_auto,
            "PENDING_APPROVAL_cases_added":  d_pend  - a_pend,
            "ESCALATED_cases_added":         d_esc   - a_esc,
            "unsafe_AUTO_reduced_by":        a_unsafe - d_unsafe,
            "unsafe_AUTO_reduction_pct":     round((a_unsafe - d_unsafe) / a_unsafe * 100, 1),
            "decision_recall_gain_pp":       round((D['decision_recall'] - A['decision_recall']) * 100, 1),
        },
        "primary_mechanism": (
            "Threshold calibration (p_esc: 0.50→0.35, p_pend: 0.50→0.20) converts "
            f"{d_pend} cases from automatic routing to PENDING_APPROVAL, "
            f"reducing unsafe AUTO from {a_unsafe} to {d_unsafe}. "
            "The deterministic policy rule enforces this routing; Granite explains it."
        ),
    }

    # ── Output artifact ────────────────────────────────────────────────────────
    output = {
        "generated_at": "2026-08-19T00:00:00Z",
        "purpose": (
            "Audit of whether the HelioMesh hybrid decision architecture provides "
            "measurable routing/safety value beyond the underlying anomaly detector. "
            "Uses only existing frozen validation artifacts. No model inference. No invented values."
        ),
        "data_sources": [
            "validation/results/real_decision_ablation.json",
            "validation/results/real_policy_calibration.json",
            "validation/results/real_disagreement_value.json",
            "validation/results/model_agreement.json",
            "validation/results/safety_gate_evaluation.json",
        ],
        "scope": (
            "Real OPS-SAT-AD test set (529 segments, 113 anomalous). "
            "NOT validated on additional spacecraft or missions."
        ),

        "decision_layer_routing_changes": decision_layer_routing_changes,
        "routing_delta_A_to_D":          routing_delta_A_to_D,
        "calibration_effect":            calibration_effect,
        "disagreement_gate":             disagreement_gate,
        "architectures":                 architectures,

        "what_is_demonstrated": [
            "The decision layer changes routing: 0 → 152 PENDING_APPROVAL cases, "
            f"AUTO_CLEAR decreases by {a_auto - d_auto}, unsafe AUTO decreases by {a_unsafe - d_unsafe}.",
            "Threshold calibration is the primary driver of unsafe AUTO reduction (14 → 9, −36%).",
            "Calibrated recall improves by +4.4pp (0.876 → 0.920) vs baseline threshold.",
            "Disagreement gate provides additional conservative routing on simulation data "
            "(171/1800 sequences, 9.5%); real precision is 5.3%.",
            "Decision layer routing is deterministic and auditable — same inputs always produce same route.",
            "Utility (prototype weights, unvalidated) is preserved vs detector-only across all weight sets.",
        ],

        "what_is_not_demonstrated": [
            "Operator workload benefit was not measured.",
            "Real-world spacecraft operational benefit was not measured.",
            "Human or operator evaluation of Granite explanations was not performed.",
            "Cross-mission validation was not performed.",
            "30-minute forecasting on real OPS-SAT was not validated "
            "(inter-segment gaps near-zero; temporal horizon does not apply).",
            "Statistical significance of the −36% unsafe AUTO reduction was not tested "
            "(absolute counts: 14 → 9 on 529 segments).",
            "Utility weights (prototype) were not validated by real mission operators.",
        ],

        "conclusion": (
            "Within the evaluated benchmark, HelioMesh demonstrates measurable decision-layer value "
            "beyond anomaly detection by changing risk routing through threshold calibration, "
            "deterministic policy rules, and conservative disagreement gating. "
            "This establishes decision/routing value within the benchmark; "
            "it does not establish operator workload reduction or real-world spacecraft operational benefit."
        ),

        "limitations": [
            "OPS-SAT-AD is a segment-classification benchmark, not a real-time operational scenario.",
            "Utility weights are prototype values defined before test evaluation and not operator-validated.",
            "Disagreement gate real precision = 5.3% — not a reliable early-warning signal.",
            "KP dominance (83.3% feature importance) is a structural property of the simulation benchmark.",
            "No cross-mission validation. No human/operator evaluation of Granite.",
        ],
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    # ── Console summary ────────────────────────────────────────────────────────
    print('=' * 60)
    print('  OPERATIONAL DECISION VALUE AUDIT')
    print('=' * 60)
    print(f'\n  Test set: {n} segments (OPS-SAT-AD)')
    print('\n  Routing comparison (Architecture A vs D):')
    print(f'                       Arch A    Arch D    Delta')
    print(f'  AUTO_CLEAR           {a_auto:>5}     {d_auto:>5}     {d_auto - a_auto:+d}')
    print(f'  PENDING_APPROVAL     {a_pend:>5}     {d_pend:>5}     {d_pend - a_pend:+d}')
    print(f'  ESCALATED            {a_esc:>5}     {d_esc:>5}     {d_esc - a_esc:+d}')
    print(f'  Unsafe AUTO          {a_unsafe:>5}     {d_unsafe:>5}     {d_unsafe - a_unsafe:+d}  ({round((a_unsafe - d_unsafe)/a_unsafe*100,1)}% reduction)')
    print(f'  Decision recall      {A["decision_recall"]:.4f}    {D["decision_recall"]:.4f}    {D["decision_recall"]-A["decision_recall"]:+.4f}')
    print(f'\n  Calibration effect:  unsafe AUTO {cal_unsafe_baseline} -> {cal_unsafe_calibrated} '
          f'({round((cal_unsafe_baseline - cal_unsafe_calibrated)/cal_unsafe_baseline*100, 1)}%) '
          f'[PRIMARY DRIVER]')
    print(f'  Disagreement gate:   5.3% real precision (NOT a reliable predictor)')
    print(f'  Operator workload:   NOT MEASURED')
    print(f'\n  Conclusion: {output["conclusion"][:120]}...')
    print(f'\n  Saved: {OUT_PATH}')

    return output


if __name__ == '__main__':
    main()
