from pathlib import Path

path = Path(
    r"validation\real_spacecraft\opssat_ad\phase3_ablation.py"
)

text = path.read_text(encoding="utf-8")


# ============================================================
# Replace old lead-time comparison call inside experiment_e
# ============================================================

old_call = '''    lead_time_analysis = (
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

new_call = '''    # ---------------------------------------------------------------
    # C -> D incremental analysis
    # Use the EXACT C policy already computed above.
    # This prevents the historical temporal-policy metrics from
    # contaminating the final A->D ablation.
    # ---------------------------------------------------------------

    c_event_for_lead = event_metrics(
        y_true,
        np.asarray([
            0 if decision == "AUTO_CLEAR" else 1
            for decision in decisions_c
        ]),
        "C_policy_without_lead"
    )

    c_policy_for_lead = policy_decision_metrics(
        y_true,
        decisions_c,
        "C_policy_without_lead"
    )

    d_event_for_lead = event_metrics(
        y_true,
        binary_d,
        "D_policy_with_lead"
    )

    d_policy_for_lead = policy_decision_metrics(
        y_true,
        decisions_d,
        "D_policy_with_lead"
    )

    delta_cd = {
        "recall":
            d_event_for_lead["event_recall"]
            - c_event_for_lead["event_recall"],

        "precision":
            d_event_for_lead["event_precision"]
            - c_event_for_lead["event_precision"],

        "f1":
            d_event_for_lead["event_f1"]
            - c_event_for_lead["event_f1"],

        "missed":
            d_event_for_lead["missed"]
            - c_event_for_lead["missed"],

        "false_alerts":
            d_event_for_lead["false_alerts"]
            - c_event_for_lead["false_alerts"],

        "unsafe_auto":
            d_policy_for_lead["unsafe_auto"]
            - c_policy_for_lead["unsafe_auto"],

        "utility_baseline":
            d_policy_for_lead["utility_baseline"]
            - c_policy_for_lead["utility_baseline"],
    }

    lead_time_analysis = {
        "C_without_lead_time": {
            "event_metrics":
                c_event_for_lead,

            "policy_metrics":
                c_policy_for_lead,
        },

        "D_with_lead_time": {
            "event_metrics":
                d_event_for_lead,

            "policy_metrics":
                d_policy_for_lead,
        },

        "delta_D_minus_C": {
            key:
                float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
            for key, value in delta_cd.items()
        },

        "lead_warning_segments":
            int(len(lead_warning_segments)),

        "n_test_segments":
            int(n),

        "interpretation": (
            "C is the calibrated-RF deterministic routing policy. "
            "D adds temporal evidence and lead-time warnings. "
            "Both are evaluated on the same aligned held-out "
            "OPS-SAT-AD test segments."
        ),
    }

    print("\n--- Lead-Time Incremental Value: C vs D ---")

    print(
        f'  C no lead: '
        f'recall={c_event_for_lead["event_recall"]:.4f} '
        f'F1={c_event_for_lead["event_f1"]:.4f} '
        f'missed={c_event_for_lead["missed"]} '
        f'FA={c_event_for_lead["false_alerts"]} '
        f'unsafe={c_policy_for_lead["unsafe_auto"]} '
        f'utility={c_policy_for_lead["utility_baseline"]:.4f}'
    )

    print(
        f'  D + lead: '
        f'recall={d_event_for_lead["event_recall"]:.4f} '
        f'F1={d_event_for_lead["event_f1"]:.4f} '
        f'missed={d_event_for_lead["missed"]} '
        f'FA={d_event_for_lead["false_alerts"]} '
        f'unsafe={d_policy_for_lead["unsafe_auto"]} '
        f'utility={d_policy_for_lead["utility_baseline"]:.4f}'
    )

    print(
        f'  Delta D-C: '
        f'recall={delta_cd["recall"]:+.4f} '
        f'F1={delta_cd["f1"]:+.4f} '
        f'missed={delta_cd["missed"]:+d} '
        f'FA={delta_cd["false_alerts"]:+d} '
        f'unsafe={delta_cd["unsafe_auto"]:+d} '
        f'utility={delta_cd["utility_baseline"]:+.4f}'
    )
'''

if old_call not in text:
    raise RuntimeError(
        "The old lead-time call was not found."
    )

text = text.replace(
    old_call,
    new_call,
    1
)


# ============================================================
# Ensure D row displays the real warning count
# ============================================================

old_lead_field = '''            "lead_time":
                (
                    f'{len(lead_warning_segments)} warning segments'
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

text = text.replace(
    old_lead_field,
    new_lead_field,
    1
)


# ============================================================
# Add the formal ablation definition to experiment_e result
# ============================================================

needle = '''        "alignment": {
            "method":
                "segment_id_inner_join",

            "common_segments":
                int(n),
        },
'''

replacement = '''        "alignment": {
            "method":
                "segment_id_inner_join",

            "common_segments":
                int(n),
        },

        "ablation_definition": {
            "A":
                "Raw RF + simple routing",

            "B":
                "Calibrated RF + simple routing",

            "C":
                "Calibrated RF + deterministic routing",

            "D":
                "Calibrated RF + temporal evidence + lead-time + deterministic routing",

            "test_partition":
                "official held-out OPS-SAT-AD test partition",

            "threshold_tuning_on_test":
                False,

            "C_policy_thresholds": {
                "pending_approval":
                    0.50,

                "escalated":
                    0.90,
            },
        },
'''

if needle not in text:
    raise RuntimeError(
        "The ablation definition insertion point was not found."
    )

text = text.replace(
    needle,
    replacement,
    1
)


# ============================================================
# Include the definition in saved JSON output
# ============================================================

old_output = '''        "alignment":
            ablation[
                "alignment"
            ],

        "provenance":
            provenance,
'''

new_output = '''        "alignment":
            ablation[
                "alignment"
            ],

        "ablation_definition":
            ablation[
                "ablation_definition"
            ],

        "provenance":
            provenance,
'''

if old_output not in text:
    raise RuntimeError(
        "The JSON output insertion point was not found."
    )

text = text.replace(
    old_output,
    new_output,
    1
)


# ============================================================
# Save patched source
# ============================================================

path.write_text(
    text,
    encoding="utf-8"
)

print("FINAL_PHASE3_FIX_OK")
print(path)