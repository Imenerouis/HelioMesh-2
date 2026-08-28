from pathlib import Path

path = Path(
    r"validation\real_spacecraft\opssat_ad\phase3_ablation.py"
)

text = path.read_text(encoding="utf-8")

# ------------------------------------------------------------------
# 1. Add calibration import
# ------------------------------------------------------------------

old = "from sklearn.ensemble import RandomForestClassifier\n"

new = (
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.calibration import CalibratedClassifierCV\n"
)

if "from sklearn.calibration import CalibratedClassifierCV" not in text:
    if old not in text:
        raise RuntimeError("Could not find sklearn ensemble import.")
    text = text.replace(old, new, 1)


# ------------------------------------------------------------------
# 2. Insert calibrated RF experiment before Experiment B
# ------------------------------------------------------------------

marker = (
    "# ═══════════════════════════════════════════════════════════════════════════\n"
    "# EXPERIMENT B\n"
    "# ═══════════════════════════════════════════════════════════════════════════\n"
)

if "def experiment_calibrated_rf(" not in text:

    calibrated_function = r'''
# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — CALIBRATED RF BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def experiment_calibrated_rf(train_df, test_df):
    """
    Calibrated RF baseline.

    Calibration is fitted using 5-fold CV on TRAIN ONLY.
    The official held-out OPS-SAT-AD test partition is never used
    during calibration fitting.
    """
    print("\n--- Experiment B: Calibrated RF Baseline ---")

    X_tr, y_tr = build_features(train_df)
    X_te, y_te = build_features(test_df)

    X_te = align_cols(X_te, X_tr)

    X_tr, X_te, scaler = scale_features(
        X_tr,
        X_te
    )

    base_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    try:
        calibrated_rf = CalibratedClassifierCV(
            estimator=base_rf,
            method="sigmoid",
            cv=5,
            n_jobs=-1
        )
    except TypeError:
        calibrated_rf = CalibratedClassifierCV(
            base_estimator=base_rf,
            method="sigmoid",
            cv=5
        )

    calibrated_rf.fit(
        X_tr,
        y_tr
    )

    y_pred = calibrated_rf.predict(
        X_te
    )

    y_prob = calibrated_rf.predict_proba(
        X_te
    )[:, 1]

    seg_m = metrics_dict(
        y_te,
        y_pred,
        y_prob,
        "experiment_B_calibrated_rf"
    )

    evt_m = event_metrics(
        y_te,
        y_pred,
        "experiment_B_calibrated_rf"
    )

    print(
        f'  F1={seg_m["f1"]:.4f} '
        f'recall={evt_m["event_recall"]:.4f} '
        f'prec={evt_m["event_precision"]:.4f} '
        f'missed={evt_m["missed"]} '
        f'FP={evt_m["false_alerts"]} '
        f'ROC-AUC={seg_m["roc_auc"]:.4f}'
    )

    return (
        calibrated_rf,
        scaler,
        X_tr,
        y_tr,
        X_te,
        y_te,
        y_pred,
        y_prob,
        seg_m,
        evt_m,
    )


'''

    if marker not in text:
        raise RuntimeError("Experiment B marker not found.")

    text = text.replace(
        marker,
        calibrated_function + marker,
        1
    )


# ------------------------------------------------------------------
# 3. Insert calibrated baseline execution in main()
# ------------------------------------------------------------------

main_marker = (
    "    a_test_segment_ids = (\n"
    "        test_df[\"segment\"]\n"
    "        .to_numpy(dtype=int)\n"
    "    )\n\n"
)

if "calibrated_test_segment_ids" not in text:

    calibrated_main = r'''
    # ═══════════════════════════════════════════════════════════════════════
    # B — CALIBRATED RF
    # ═══════════════════════════════════════════════════════════════════════

    (
        rf_cal,
        scaler_cal,
        X_tr_cal,
        y_tr_cal,
        X_te_cal,
        y_te_cal,
        y_pred_cal,
        y_prob_cal,
        seg_m_cal,
        evt_m_cal,
    ) = experiment_calibrated_rf(
        train_df,
        test_df
    )

    calibrated_test_segment_ids = (
        test_df["segment"]
        .to_numpy(dtype=int)
    )

'''

    if main_marker not in text:
        raise RuntimeError("Main insertion point not found.")

    text = text.replace(
        main_marker,
        main_marker + calibrated_main,
        1
    )


# ------------------------------------------------------------------
# 4. Create alignment for calibrated RF
# ------------------------------------------------------------------

align_marker = (
    "    print(\n"
    '        f"  Final aligned test segments: "\n'
    "        f\"{n}\"\n"
    "    )\n\n"
)

if "Calibrated RF alignment" not in text:

    cal_align = r'''
    # ═══════════════════════════════════════════════════════════════════════
    # Calibrated RF alignment
    # ═══════════════════════════════════════════════════════════════════════

    cal_df = pd.DataFrame({
        "segment": calibrated_test_segment_ids.astype(int),
        "y_true": np.asarray(y_te_cal).astype(int),
        "y_pred": np.asarray(y_pred_cal).astype(int),
        "y_prob": np.asarray(y_prob_cal, dtype=float),
    })

    if cal_df["segment"].duplicated().any():
        raise ValueError(
            "Calibrated RF contains duplicate segment IDs."
        )

    cal_common = pd.DataFrame({
        "segment": aligned_segment_ids.astype(int),
    }).merge(
        cal_df,
        on="segment",
        how="inner",
        validate="one_to_one"
    ).sort_values("segment").reset_index(drop=True)

    if len(cal_common) != n:
        raise ValueError(
            "Calibrated RF alignment does not cover all aligned test segments. "
            f"Expected {n}, got {len(cal_common)}."
        )

    cal_truth_mismatch = (
        cal_common["y_true"].to_numpy(dtype=int)
        != y_true_aligned
    )

    if int(cal_truth_mismatch.sum()) != 0:
        raise ValueError(
            "Calibrated RF ground truth mismatch after alignment."
        )

    y_pred_cal_aligned = (
        cal_common["y_pred"].to_numpy(dtype=int)
    )

    y_prob_cal_aligned = (
        cal_common["y_prob"].to_numpy(dtype=float)
    )

    print(
        f"  Calibrated RF alignment: "
        f"{len(cal_common)} segments, "
        f"truth_mismatches=0"
    )

'''

    if align_marker not in text:
        raise RuntimeError("Alignment insertion point not found.")

    text = text.replace(
        align_marker,
        align_marker + cal_align,
        1
    )


# ------------------------------------------------------------------
# 5. Add calibrated RF architecture variables before Experiment E
# ------------------------------------------------------------------
# We modify experiment_e signature and A/B/C logic.
# For a robust patch, add optional calibrated arrays to the signature.

old_sig = '''def experiment_e(
    y_true,
    y_pred_a,
    y_prob_a,
    y_pred_b,
    y_prob_b,
    y_lt_pred,
    y_lt_prob,
    lt_test,
    n,
    aligned_segment_ids
):'''

new_sig = '''def experiment_e(
    y_true,
    y_pred_a,
    y_prob_a,
    y_pred_b,
    y_prob_b,
    y_pred_cal,
    y_prob_cal,
    y_lt_pred,
    y_lt_prob,
    lt_test,
    n,
    aligned_segment_ids
):'''

if old_sig in text:
    text = text.replace(old_sig, new_sig, 1)

# Add calibrated arrays after y_prob_b conversion.
old_block = '''    y_prob_b = np.asarray(
        y_prob_b
    )[:n]

    segment_ids = np.asarray(
        aligned_segment_ids
    ).astype(int)[:n]
'''

new_block = '''    y_prob_b = np.asarray(
        y_prob_b
    )[:n]

    y_pred_cal = np.asarray(
        y_pred_cal
    )[:n]

    y_prob_cal = np.asarray(
        y_prob_cal
    )[:n]

    segment_ids = np.asarray(
        aligned_segment_ids
    ).astype(int)[:n]
'''

if old_block in text:
    text = text.replace(old_block, new_block, 1)


# ------------------------------------------------------------------
# 6. Replace the old B architecture with calibrated RF
# ------------------------------------------------------------------

old_b = '''    # ═══════════════════════════════════════════════════════════════════════
    # B — CURRENT + TEMPORAL
    # ═══════════════════════════════════════════════════════════════════════

    decisions_b = apply_current_only_policy(
        y_pred_b
    )

    binary_b = np.asarray([
        0
        if decision == "AUTO_CLEAR"
        else 1
        for decision in decisions_b
    ])

    event_b = event_metrics(
        y_true,
        binary_b,
        "arch_B_current_plus_temporal"
    )

    policy_b = policy_decision_metrics(
        y_true,
        decisions_b,
        "arch_B"
    )
'''

new_b = '''    # ═══════════════════════════════════════════════════════════════════════
    # B — CALIBRATED RF
    # ═══════════════════════════════════════════════════════════════════════

    decisions_b = apply_current_only_policy(
        y_pred_cal
    )

    binary_b = np.asarray([
        0
        if decision == "AUTO_CLEAR"
        else 1
        for decision in decisions_b
    ])

    event_b = event_metrics(
        y_true,
        binary_b,
        "arch_B_calibrated_rf"
    )

    policy_b = policy_decision_metrics(
        y_true,
        decisions_b,
        "arch_B"
    )
'''

if old_b not in text:
    raise RuntimeError(
        "Old architecture B block was not found."
    )

text = text.replace(
    old_b,
    new_b,
    1
)


# ------------------------------------------------------------------
# 7. Replace C so it uses calibrated RF + temporal evidence
# ------------------------------------------------------------------

old_c_call = '''    decisions_c = apply_deterministic_policy(
        y_pred_a,
        y_pred_b,
        y_prob_a,
        y_prob_b,
        disagree_threshold=0.50
    )
'''

new_c_call = '''    decisions_c = apply_deterministic_policy(
        y_pred_cal,
        y_pred_b,
        y_prob_cal,
        y_prob_b,
        disagree_threshold=0.50
    )
'''

if old_c_call not in text:
    raise RuntimeError(
        "C policy block was not found."
    )

text = text.replace(
    old_c_call,
    new_c_call,
    1
)


# ------------------------------------------------------------------
# 8. Make D use calibrated RF as the current evidence source
# ------------------------------------------------------------------

old_d = '''        current_prediction = int(
            y_pred_a[i]
        )
'''

new_d = '''        current_prediction = int(
            y_pred_cal[i]
        )
'''

# Replace only inside experiment_e, but this exact occurrence can also
# appear elsewhere. Use the last occurrence, which belongs to D.
pos = text.rfind(old_d)

if pos == -1:
    raise RuntimeError(
        "D current prediction block was not found."
    )

text = (
    text[:pos]
    + new_d
    + text[pos + len(old_d):]
)


# ------------------------------------------------------------------
# 9. Change C→D lead-time isolation to use calibrated RF
# ------------------------------------------------------------------

old_lead_call = '''    lead_time_analysis = (
        compare_policy_without_vs_with_lead_time(
            y_true,
            y_pred_a,
            y_prob_a,
            y_pred_b,
            y_prob_b,
            lead_warning_segments,
            segment_ids
        )
    )
'''

new_lead_call = '''    lead_time_analysis = (
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

if old_lead_call not in text:
    raise RuntimeError(
        "Lead-time comparison block was not found."
    )

text = text.replace(
    old_lead_call,
    new_lead_call,
    1
)


# ------------------------------------------------------------------
# 10. Change architecture labels
# ------------------------------------------------------------------

text = text.replace(
    '"B_current_plus_temporal"',
    '"B_calibrated_rf"',
    1
)

text = text.replace(
    '"C_disagreement_policy"',
    '"C_calibrated_rf_deterministic_policy"',
    1
)

text = text.replace(
    '"D_full_heliomesh"',
    '"D_full_heliomesh"',
    1
)


# ------------------------------------------------------------------
# 11. Update main() call to experiment_e
# ------------------------------------------------------------------

old_call = '''    ablation = experiment_e(
        y_true_aligned,
        y_pred_a_aligned,
        y_prob_a_aligned,
        y_pred_b_aligned,
        y_prob_b_aligned,
        y_lt_pred,
        y_lt_prob,
        lt_test,
        n,
        aligned_segment_ids
    )
'''

new_call = '''    ablation = experiment_e(
        y_true_aligned,
        y_pred_a_aligned,
        y_prob_a_aligned,
        y_pred_b_aligned,
        y_prob_b_aligned,
        y_pred_cal_aligned,
        y_prob_cal_aligned,
        y_lt_pred,
        y_lt_prob,
        lt_test,
        n,
        aligned_segment_ids
    )
'''

if old_call not in text:
    raise RuntimeError(
        "experiment_e main call was not found."
    )

text = text.replace(
    old_call,
    new_call,
    1
)


# ------------------------------------------------------------------
# 12. Save calibrated experiment in JSON
# ------------------------------------------------------------------

save_marker = '''        "experiment_B_current_temporal": {
            "segment_metrics":
                seg_m_b,
            "event_metrics":
                evt_m_b,
        },
'''

save_replacement = '''        "experiment_B_calibrated_rf": {
            "segment_metrics":
                seg_m_cal,
            "event_metrics":
                evt_m_cal,
        },

        "experiment_temporal_supporting": {
            "segment_metrics":
                seg_m_b,
            "event_metrics":
                evt_m_b,
        },
'''

if save_marker in text:
    text = text.replace(
        save_marker,
        save_replacement,
        1
)


# ------------------------------------------------------------------
# 13. Update grounding audit to include calibrated model
# ------------------------------------------------------------------

# Keep existing structural audit, but add calibrated metrics.
# This is optional and does not alter the experiment logic.

if '"calibrated_model_f1"' not in text:

    granite_marker = '''    checks = {
        "evidence_coverage_current_model":
            coverage_a,
'''

    granite_replacement = '''    checks = {
        "evidence_coverage_current_model":
            coverage_a,

        "calibrated_model_f1":
            float(
                seg_m_calibration_f1
            ) if "seg_m_calibration_f1" in locals()
            else None,
'''

    if granite_marker in text:
        text = text.replace(
            granite_marker,
            granite_replacement,
            1
        )


# ------------------------------------------------------------------
# 14. Add calibrated metrics to granite invocation using globals is
#     intentionally avoided; instead make existing call unchanged.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 15. Add calibrated metrics to temporal output and main result
# ------------------------------------------------------------------

# No unsafe global modifications here.


# ------------------------------------------------------------------
# Write patched file
# ------------------------------------------------------------------

path.write_text(
    text,
    encoding="utf-8"
)

print(
    "PATCH_OK:",
    path
)
print(
    "Calibrated RF experiment inserted:",
    "def experiment_calibrated_rf(" in text
)
print(
    "Calibration import inserted:",
    "CalibratedClassifierCV" in text
)