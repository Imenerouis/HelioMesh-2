"""
HelioMesh Contribution Ablation
================================

Goal:
    Separate the contribution of:
      C  = calibrated RF only
      CT = calibrated RF + temporal evidence
      CL = calibrated RF + lead-warning only
      D  = calibrated RF + temporal evidence + lead-warning

All four architectures are evaluated on the same aligned
official held-out OPS-SAT-AD test segments.

Important:
    - No test-threshold tuning.
    - No retraining of frozen artifacts.
    - Ground truth is used only for evaluation.
    - D is checked against the official Phase 3 result.
"""

import json
import os
import numpy as np
import pandas as pd

from validation.real_spacecraft.opssat_ad import phase3_ablation as p


RESULTS_DIR = os.path.join(
    "validation",
    "results"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "real_contribution_ablation.json"
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "real_contribution_ablation.csv"
)


def event_metrics(y_true, decisions, label):
    binary = np.asarray([
        0 if d == "AUTO_CLEAR" else 1
        for d in decisions
    ])

    return p.event_metrics(
        y_true,
        binary,
        label
    )


def policy_metrics(y_true, decisions, label):
    return p.policy_decision_metrics(
        y_true,
        decisions,
        label
    )


def build_lead_warning_segments(
    y_lt_pred,
    lt_test
):
    if y_lt_pred is None or lt_test is None:
        return set()

    lead_predictions = np.asarray(
        y_lt_pred
    ).astype(int)

    lt_work = lt_test.copy()

    if len(lead_predictions) != len(lt_work):
        raise ValueError(
            "Lead prediction length does not match lt_test."
        )

    lt_work["lt_pred"] = lead_predictions

    warned = (
        lt_work[
            lt_work["lt_pred"] == 1
        ]["segment"]
        .astype(int)
        .values
    )

    raw = pd.read_csv(
        p.DS_PATH
    )

    segment_times = pd.read_csv(
        p.SEG_PATH,
        parse_dates=["timestamp"]
    )

    start_times = (
        segment_times
        .groupby("segment")["timestamp"]
        .min()
        .rename("start_ts")
    )

    raw = raw.join(
        start_times,
        on="segment"
    )

    raw = (
        raw
        .sort_values(
            ["channel", "start_ts"]
        )
        .reset_index(drop=True)
    )

    raw["next_segment"] = (
        raw
        .groupby("channel")["segment"]
        .shift(-1)
    )

    next_map = (
        raw
        .set_index("segment")["next_segment"]
        .dropna()
        .to_dict()
    )

    lead_warning_segments = set()

    for segment_id in warned:
        next_segment = next_map.get(
            int(segment_id)
        )

        if next_segment is not None:
            lead_warning_segments.add(
                int(next_segment)
            )

    return lead_warning_segments


def route_ct(
    current_pred,
    temporal_pred,
    temporal_prob
):
    """
    C + temporal, without lead warnings.

    Same evidence logic as D, but lead-warning is disabled.
    """
    decisions = []

    for current, temporal, p_temporal in zip(
        current_pred,
        temporal_pred,
        temporal_prob
    ):

        if (
            current == 1
            and temporal == 1
        ):
            decisions.append(
                "ESCALATED"
            )

        elif current == 1:
            decisions.append(
                "PENDING_APPROVAL"
            )

        elif (
            temporal == 1
            and p_temporal >= 0.50
        ):
            decisions.append(
                "PENDING_APPROVAL"
            )

        else:
            decisions.append(
                "AUTO_CLEAR"
            )

    return decisions


def route_cl(
    current_pred,
    lead_warning_segments,
    segment_ids
):
    """
    C + lead-warning, without temporal model evidence.

    Lead warnings alone are allowed to move AUTO_CLEAR cases to
    PENDING_APPROVAL.

    A current anomaly remains PENDING_APPROVAL because this isolates
    the lead-warning contribution rather than changing the current
    detector's severity semantics.
    """
    decisions = []

    for current, segment_id in zip(
        current_pred,
        segment_ids
    ):

        warned = (
            int(segment_id)
            in lead_warning_segments
        )

        if current == 1:
            decisions.append(
                "PENDING_APPROVAL"
            )

        elif warned:
            decisions.append(
                "PENDING_APPROVAL"
            )

        else:
            decisions.append(
                "AUTO_CLEAR"
            )

    return decisions


def route_d(
    current_pred,
    temporal_pred,
    temporal_prob,
    segment_ids,
    lead_warning_segments
):
    """
    Full D routing logic.
    This must reproduce the official Phase 3 D policy.
    """
    decisions = []

    for current, temporal, p_temporal, segment_id in zip(
        current_pred,
        temporal_pred,
        temporal_prob,
        segment_ids
    ):

        current = int(current)
        temporal = int(temporal)
        p_temporal = float(p_temporal)
        segment_id = int(segment_id)

        lead_warned = (
            segment_id
            in lead_warning_segments
        )

        if (
            current == 1
            and temporal == 1
        ):
            decisions.append(
                "ESCALATED"
            )

        elif (
            lead_warned
            and (
                current == 1
                or temporal == 1
            )
        ):
            decisions.append(
                "ESCALATED"
            )

        elif (
            current == 1
            or (
                temporal == 1
                and p_temporal >= 0.50
            )
        ):
            decisions.append(
                "PENDING_APPROVAL"
            )

        elif lead_warned:
            decisions.append(
                "PENDING_APPROVAL"
            )

        else:
            decisions.append(
                "AUTO_CLEAR"
            )

    return decisions


def main():

    print("=" * 76)
    print("HELIOMESH — CONTRIBUTION ABLATION")
    print("=" * 76)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------------------
    # Load and reproduce the same Phase 3 models/alignment.
    # ------------------------------------------------------------------

    seg, ds = p.load_data()

    train_df, test_df, _ = p.get_splits()

    (
        rf_a,
        scaler_a,
        X_tr_a,
        y_tr_a,
        X_te_a,
        y_te_a,
        y_pred_a,
        y_prob_a,
        seg_m_a,
        evt_m_a
    ) = p.experiment_a(
        train_df,
        test_df
    )

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
    ) = p.experiment_calibrated_rf(
        train_df,
        test_df
    )

    ret_b = p.experiment_b(
        train_df,
        test_df,
        seg,
        rf_a,
        scaler_a,
        X_tr_a,
        y_tr_a,
        X_te_a,
        y_te_a
    )

    (
        rf_b,
        scaler_b,
        X_tr_b,
        y_tr_b,
        X_te_b,
        y_te_b,
        y_pred_b,
        y_prob_b,
        seg_m_b,
        evt_m_b,
        lead_model_result,
        rf_lt,
        lt_test,
        y_lt_te,
        y_lt_pred,
        y_lt_prob,
        b_test_segment_ids
    ) = ret_b

    a_test_segment_ids = (
        test_df["segment"]
        .to_numpy(dtype=int)
    )

    calibrated_test_segment_ids = (
        test_df["segment"]
        .to_numpy(dtype=int)
    )

    # ------------------------------------------------------------------
    # Align temporal model by actual segment ID.
    # ------------------------------------------------------------------

    (
        aligned_segment_ids,
        y_true,
        y_pred_a_aligned,
        y_prob_a_aligned,
        y_pred_temporal_aligned,
        y_prob_temporal_aligned
    ) = p.align_ablation_by_segment(
        y_true_a=y_te_a,
        segment_ids_a=a_test_segment_ids,
        y_pred_a=y_pred_a,
        y_prob_a=y_prob_a,
        y_true_b=y_te_b,
        segment_ids_b=b_test_segment_ids,
        y_pred_b=y_pred_b,
        y_prob_b=y_prob_b
    )

    n = len(
        aligned_segment_ids
    )

    # ------------------------------------------------------------------
    # Align calibrated RF.
    # ------------------------------------------------------------------

    cal_df = pd.DataFrame({
        "segment":
            calibrated_test_segment_ids.astype(int),

        "y_true":
            np.asarray(
                y_te_cal
            ).astype(int),

        "y_pred":
            np.asarray(
                y_pred_cal
            ).astype(int),

        "y_prob":
            np.asarray(
                y_prob_cal,
                dtype=float
            ),
    })

    cal_common = (
        pd.DataFrame({
            "segment":
                aligned_segment_ids.astype(int)
        })
        .merge(
            cal_df,
            on="segment",
            how="inner",
            validate="one_to_one"
        )
        .sort_values(
            "segment"
        )
        .reset_index(
            drop=True
        )
    )

    if len(cal_common) != n:
        raise ValueError(
            "Calibrated RF alignment does not cover "
            "all aligned segments."
        )

    y_true_cal = (
        cal_common["y_true"]
        .to_numpy(dtype=int)
    )

    if not np.array_equal(
        y_true,
        y_true_cal
    ):
        raise ValueError(
            "Ground-truth mismatch between calibrated RF "
            "and aligned test set."
        )

    y_pred_current = (
        cal_common["y_pred"]
        .to_numpy(dtype=int)
    )

    y_prob_current = (
        cal_common["y_prob"]
        .to_numpy(dtype=float)
    )

    # ------------------------------------------------------------------
    # Lead-warning set.
    # ------------------------------------------------------------------

    lead_warning_segments = build_lead_warning_segments(
        y_lt_pred,
        lt_test
    )

    print()
    print(
        f"Aligned test segments: {n}"
    )

    print(
        f"Lead-warning segments: "
        f"{len(lead_warning_segments)}"
    )

    # ------------------------------------------------------------------
    # C — calibrated RF only.
    # ------------------------------------------------------------------

    decisions_c = [
        "PENDING_APPROVAL"
        if probability >= 0.50
        else "AUTO_CLEAR"
        for probability in y_prob_current
    ]

    # ------------------------------------------------------------------
    # C + TEMPORAL
    # ------------------------------------------------------------------

    decisions_ct = route_ct(
        y_pred_current,
        y_pred_temporal_aligned,
        y_prob_temporal_aligned
    )

    # ------------------------------------------------------------------
    # C + LEAD WARNING
    # ------------------------------------------------------------------

    decisions_cl = route_cl(
        y_pred_current,
        lead_warning_segments,
        aligned_segment_ids
    )

    # ------------------------------------------------------------------
    # D — FULL HELIOMESH
    # ------------------------------------------------------------------

    decisions_d = route_d(
        y_pred_current,
        y_pred_temporal_aligned,
        y_prob_temporal_aligned,
        aligned_segment_ids,
        lead_warning_segments
    )

    architectures = {
        "C_calibrated_rf":
            decisions_c,

        "CT_calibrated_rf_plus_temporal":
            decisions_ct,

        "CL_calibrated_rf_plus_lead_warning":
            decisions_cl,

        "D_full_heliomesh":
            decisions_d,
    }

    # ------------------------------------------------------------------
    # Evaluate.
    # ------------------------------------------------------------------

    results = {}

    for name, decisions in architectures.items():

        evt = event_metrics(
            y_true,
            decisions,
            name
        )

        policy = policy_metrics(
            y_true,
            decisions,
            name
        )

        results[name] = {
            "event_metrics": evt,
            "policy_metrics": policy,
        }

    # ------------------------------------------------------------------
    # Verify official D result.
    # ------------------------------------------------------------------

    official_path = os.path.join(
        RESULTS_DIR,
        "real_decision_ablation.json"
    )

    d_verification = {
        "official_result_available": False,
        "verified": False,
    }

    if os.path.exists(
        official_path
    ):

        with open(
            official_path,
            "r",
            encoding="utf-8"
        ) as f:
            official = json.load(f)

        official_d = (
            official[
                "architecture_details"
            ]["D"]["event_metrics"]
        )

        reproduced_d = (
            results[
                "D_full_heliomesh"
            ]["event_metrics"]
        )

        checks = {
            "recall":
                abs(
                    official_d["event_recall"]
                    - reproduced_d["event_recall"]
                ) < 1e-12,

            "precision":
                abs(
                    official_d["event_precision"]
                    - reproduced_d["event_precision"]
                ) < 1e-12,

            "f1":
                abs(
                    official_d["event_f1"]
                    - reproduced_d["event_f1"]
                ) < 1e-12,

            "missed":
                official_d["missed"]
                == reproduced_d["missed"],

            "false_alerts":
                official_d["false_alerts"]
                == reproduced_d["false_alerts"],
        }

        d_verification = {
            "official_result_available":
                True,

            "verified":
                bool(
                    all(
                        checks.values()
                    )
                ),

            "checks":
                checks,

            "official":
                {
                    "recall":
                        official_d["event_recall"],
                    "precision":
                        official_d["event_precision"],
                    "f1":
                        official_d["event_f1"],
                    "missed":
                        official_d["missed"],
                    "false_alerts":
                        official_d["false_alerts"],
                },

            "reproduced":
                {
                    "recall":
                        reproduced_d["event_recall"],
                    "precision":
                        reproduced_d["event_precision"],
                    "f1":
                        reproduced_d["event_f1"],
                    "missed":
                        reproduced_d["missed"],
                    "false_alerts":
                        reproduced_d["false_alerts"],
                },
        }

    # ------------------------------------------------------------------
    # Incremental deltas.
    # ------------------------------------------------------------------

    c_evt = results[
        "C_calibrated_rf"
    ]["event_metrics"]

    ct_evt = results[
        "CT_calibrated_rf_plus_temporal"
    ]["event_metrics"]

    cl_evt = results[
        "CL_calibrated_rf_plus_lead_warning"
    ]["event_metrics"]

    d_evt = results[
        "D_full_heliomesh"
    ]["event_metrics"]

    contribution = {
        "C_to_CT_temporal_only": {
            "recall_delta":
                ct_evt["event_recall"]
                - c_evt["event_recall"],

            "precision_delta":
                ct_evt["event_precision"]
                - c_evt["event_precision"],

            "f1_delta":
                ct_evt["event_f1"]
                - c_evt["event_f1"],

            "missed_delta":
                ct_evt["missed"]
                - c_evt["missed"],

            "false_alert_delta":
                ct_evt["false_alerts"]
                - c_evt["false_alerts"],
        },

        "C_to_CL_lead_warning_only": {
            "recall_delta":
                cl_evt["event_recall"]
                - c_evt["event_recall"],

            "precision_delta":
                cl_evt["event_precision"]
                - c_evt["event_precision"],

            "f1_delta":
                cl_evt["event_f1"]
                - c_evt["event_f1"],

            "missed_delta":
                cl_evt["missed"]
                - c_evt["missed"],

            "false_alert_delta":
                cl_evt["false_alerts"]
                - c_evt["false_alerts"],
        },

        "C_to_D_full": {
            "recall_delta":
                d_evt["event_recall"]
                - c_evt["event_recall"],

            "precision_delta":
                d_evt["event_precision"]
                - c_evt["event_precision"],

            "f1_delta":
                d_evt["event_f1"]
                - c_evt["event_f1"],

            "missed_delta":
                d_evt["missed"]
                - c_evt["missed"],

            "false_alert_delta":
                d_evt["false_alerts"]
                - c_evt["false_alerts"],
        },
    }

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    output = {
        "analysis":
            (
                "Contribution ablation separating temporal evidence "
                "and lead-warning evidence using the same aligned "
                "held-out OPS-SAT-AD test partition."
            ),

        "n_test_segments":
            int(n),

        "threshold_tuning_on_test":
            False,

        "official_D_reproduction":
            d_verification,

        "lead_warning_segments":
            int(
                len(
                    lead_warning_segments
                )
            ),

        "architectures":
            results,

        "incremental_contribution":
            contribution,

        "interpretation_guardrail":
            (
                "These component comparisons identify benchmark-level "
                "incremental routing effects. They do not prove causal "
                "operational superiority and do not validate mission costs."
            ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    # CSV
    rows = []

    for name, value in results.items():

        evt = value["event_metrics"]
        pol = value["policy_metrics"]

        rows.append({
            "architecture":
                name,

            "recall":
                evt["event_recall"],

            "precision":
                evt["event_precision"],

            "f1":
                evt["event_f1"],

            "missed":
                evt["missed"],

            "false_alerts":
                evt["false_alerts"],

            "unsafe_auto":
                pol["unsafe_auto"],

            "utility":
                pol["utility_baseline"],
        })

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ------------------------------------------------------------------
    # Print.
    # ------------------------------------------------------------------

    print()
    print("=" * 76)
    print("CONTRIBUTION ABLATION RESULTS")
    print("=" * 76)

    print(
        f'{"Architecture":<38}'
        f'{"Recall":>9}'
        f'{"Prec":>9}'
        f'{"F1":>9}'
        f'{"Missed":>9}'
        f'{"FA":>7}'
    )

    print("-" * 76)

    for row in rows:

        print(
            f'{row["architecture"]:<38}'
            f'{row["recall"]:>9.4f}'
            f'{row["precision"]:>9.4f}'
            f'{row["f1"]:>9.4f}'
            f'{row["missed"]:>9}'
            f'{row["false_alerts"]:>7}'
        )

    print()
    print(
        "C -> C+Temporal:"
    )
    print(
        json.dumps(
            contribution[
                "C_to_CT_temporal_only"
            ],
            indent=2
        )
    )

    print()
    print(
        "C -> C+Lead:"
    )
    print(
        json.dumps(
            contribution[
                "C_to_CL_lead_warning_only"
            ],
            indent=2
        )
    )

    print()
    print(
        "C -> D:"
    )
    print(
        json.dumps(
            contribution[
                "C_to_D_full"
            ],
            indent=2
        )
    )

    print()
    print(
        "Official D reproduction:"
    )

    print(
        "  "
        + (
            "PASS"
            if d_verification["verified"]
            else "FAIL"
        )
    )

    if (
        d_verification[
            "official_result_available"
        ]
        and not d_verification["verified"]
    ):
        print(
            json.dumps(
                d_verification,
                indent=2
            )
        )

    print()
    print(
        f"Saved: {OUTPUT_JSON}"
    )

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()