from pathlib import Path

path = Path(
    r"validation\real_spacecraft\opssat_ad\phase3_ablation.py"
)

text = path.read_text(encoding="utf-8")


# ============================================================
# 1) Fix C: calibrated RF + deterministic routing only
# ============================================================

old_c = '''    # ═══════════════════════════════════════════════════════════════════════
    # C — DISAGREEMENT POLICY
    # ═══════════════════════════════════════════════════════════════════════

    decisions_c = apply_deterministic_policy(
        y_pred_cal,
        y_pred_b,
        y_prob_cal,
        y_prob_b,
        disagree_threshold=0.50
    )

    binary_c = np.asarray([
        0
        if decision == "AUTO_CLEAR"
        else 1
        for decision in decisions_c
    ])

    event_c = event_metrics(
        y_true,
        binary_c,
        "arch_C_disagreement_policy"
    )

    policy_c = policy_decision_metrics(
        y_true,
        decisions_c,
        "arch_C"
    )
'''

new_c = '''    # ═══════════════════════════════════════════════════════════════════════
    # C — CALIBRATED RF + DETERMINISTIC ROUTING
    # ═══════════════════════════════════════════════════════════════════════

    # C deliberately uses ONLY the calibrated RF evidence.
    # Temporal / lead-time evidence is reserved for D.
    #
    # Rule:
    #   calibrated anomaly probability >= 0.50 -> PENDING_APPROVAL
    #   otherwise                            -> AUTO_CLEAR
    #
    # No threshold is tuned on the held-out test set.

    decisions_c = [
        "PENDING_APPROVAL"
        if float(probability) >= 0.50
        else "AUTO_CLEAR"
        for probability in y_prob_cal
    ]

    binary_c = np.asarray([
        0
        if decision == "AUTO_CLEAR"
        else 1
        for decision in decisions_c
    ])

    event_c = event_metrics(
        y_true,
        binary_c,
        "arch_C_calibrated_rf_deterministic_policy"
    )

    policy_c = policy_decision_metrics(
        y_true,
        decisions_c,
        "arch_C"
    )
'''

if old_c not in text:
    raise RuntimeError(
        "Old C block not found."
    )

text = text.replace(
    old_c,
    new_c,
    1
)


# ============================================================
# 2) D must use calibrated RF + temporal + lead-time
# ============================================================

old_d_prediction = '''        current_prediction = int(
            y_pred_cal[i]
        )
'''

new_d_prediction = '''        # D combines calibrated current-state evidence,
        # temporal evidence, and lead-time warnings.
        current_prediction = int(
            y_pred_cal[i]
        )
'''

if old_d_prediction in text:
    text = text.replace(
        old_d_prediction,
        new_d_prediction,
        1
    )


# ============================================================
# 3) Fix C -> D incremental lead-time analysis
#    It must compare D against the corrected C.
# ============================================================

old_lead = '''    lead_time_analysis = (
        compare_policy_without_vs_with_lead_time(
            y_true,
            y_pred_cal,
            y_prob_cal,
            y_pred_b,
            y_prob_b,
            lead_warning_segments,
            segment_ids
        )
    )
'''

new_lead = '''    # Incremental lead-time analysis:
    # compare the corrected C routing policy against D.
    lead_time_analysis = (
        compare_policy_without_vs_with_lead_time(
            y_true,
            y_pred_cal,
            y_prob_cal,
            y_pred_b,
            y_prob_b,
            lead_warning_segments,
            segment_ids
        )
    )
'''

if old_lead not in text:
    raise RuntimeError(
        "Lead-time analysis block not found."
    )

text = text.replace(
    old_lead,
    new_lead,
    1
)


# ============================================================
# 4) Fix the displayed lead_time field
#    It was incorrectly showing n=529 instead of 58.
# ============================================================

old_lead_field = '''            "lead_time":
                (
                    f'~{int(lead_time_analysis["D_with_lead_time"]["policy_metrics"]["n"])} segments'
                    if len(lead_warning_segments) > 0
                    else "N/A"
                ),
'''

new_lead_field = '''            "lead_time":
                (
                    f'{len(lead_warning_segments)} warning segments'
                    if len(lead_warning_segments) > 0
                    else "N/A"
                ),
'''

if old_lead_field not in text:
    raise RuntimeError(
        "Old lead_time table field not found."
    )

text = text.replace(
    old_lead_field,
    new_lead_field,
    1
)


# ============================================================
# 5) Rename C description in printed/report table
# ============================================================

text = text.replace(
    '"C_calibrated_rf_deterministic_policy"',
    '"C_calibrated_rf_deterministic_policy"',
    1
)


# ============================================================
# 6) Make experiment C comparison metadata explicit
# ============================================================

old_exp_c_label = '''    exp_c_result = experiment_c(
        y_true_aligned,
        y_pred_a_aligned,
        y_pred_b_aligned
    )
'''

new_exp_c_label = '''    # Supporting temporal experiment.
    # This is intentionally separate from the A-D decision ablation.
    exp_c_result = experiment_c(
        y_true_aligned,
        y_pred_a_aligned,
        y_pred_b_aligned
    )
'''

if old_exp_c_label in text:
    text = text.replace(
        old_exp_c_label,
        new_exp_c_label,
        1
    )


# ============================================================
# 7) Add explicit A-D definition to JSON
# ============================================================

old_definition = '''        "alignment": {
            "method":
                "segment_id_inner_join",

            "common_segments":
                int(n),
        },
'''

new_definition = '''        "alignment": {
            "method":
                "segment_id_inner_join",

            "common_segments":
                int(n),
        },

        "ablation_definition": {
            "A": "Raw RF + simple routing",
            "B": "Calibrated RF + simple routing",
            "C": "Calibrated RF + deterministic routing",
            "D": "Calibrated RF + temporal evidence + lead-time + deterministic routing",
            "test_partition": "official held-out OPS-SAT-AD test partition",
            "threshold_tuning_on_test": False,
        },
'''

if old_definition not in text:
    raise RuntimeError(
        "Ablation JSON insertion point not found."
    )

text = text.replace(
    old_definition,
    new_definition,
    1
)


# ============================================================
# 8) Update the top documentation
# ============================================================

old_header = '''Experiments:
  A — Real anomaly-only baseline
  B — Real current + temporal predictor
  C — Deterministic disagreement-aware policy
  D — Full HelioMesh policy with lead-time signal
'''

new_header = '''Architectures in the primary A→D ablation:
  A — Raw RF + simple routing
  B — Calibrated RF + simple routing
  C — Calibrated RF + deterministic routing
  D — Full HelioMesh: calibrated RF + temporal evidence + lead-time

Supporting analyses:
  - Temporal predictor
  - Model disagreement
  - C→D lead-time incremental analysis
'''

if old_header in text:
    text = text.replace(
        old_header,
        new_header,
        1
    )


# ============================================================
# 9) Save
# ============================================================

path.write_text(
    text,
    encoding="utf-8"
)

print("FIX_OK")
print(path)