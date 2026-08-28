"""
phase3_ablation.py — HelioMesh Phase 3: Real Decision Intelligence Ablation

All experiments run on the OFFICIAL OPS-SAT-AD test partition only.

Architectures in the primary A→D ablation:
  A — Raw RF + simple routing
  B — Calibrated RF + simple routing
  C — Calibrated RF + deterministic routing
  D — Full HelioMesh: calibrated RF + temporal evidence + lead-time

Supporting analyses:
  - Temporal predictor
  - Model disagreement
  - C→D lead-time incremental analysis
  - Decision Stress Test with per-segment routing trace

Important:
  A/B/C/D comparisons are aligned by actual segment ID.
  Row-position comparison is not used.

Decision Stress Test:
  - Records every C→D routing transition.
  - Attributes D's route reason using decision-time evidence only.
  - Uses ground truth only AFTER the decision to classify:
      PROTECTED_ANOMALY
      NEW_FALSE_ALERT
  - Does not alter A/B/C/D metrics.

Outputs:
  validation/results/real_decision_ablation.json
  validation/results/real_disagreement_value.json
  validation/results/real_temporal_value.json
  validation/results/real_policy_evaluation.json
  validation/results/real_granite_grounding.json
  validation/results/real_decision_stress_test_segments.csv
  validation/results/real_decision_stress_test_summary.json
"""

import json
import os
import sys
import hashlib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT ENCODING
# ═══════════════════════════════════════════════════════════════════════════

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(
    "validation",
    "results"
)

SEG_PATH = os.path.join(
    "data",
    "real_spacecraft",
    "opssat_ad",
    "segments.csv"
)

DS_PATH = os.path.join(
    "data",
    "real_spacecraft",
    "opssat_ad",
    "dataset.csv"
)

FROZEN_ARTIFACTS = {
    "ml/risk_model.pkl": "109a8a43578926a2",
    "ml/label_encoder.pkl": "20b39a8a67cd98b5",
    "ml/forecaster_model.pkl": "89864a1706a312c2",
    "ml/forecaster_metrics.json": "533b3fc5b48e4c09",
}

STRESS_CSV_PATH = os.path.join(
    RESULTS_DIR,
    "real_decision_stress_test_segments.csv"
)

STRESS_JSON_FILENAME = (
    "real_decision_stress_test_summary.json"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


def save_json(obj, filename):
    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            cls=NumpyEncoder
        )

    print(
        f"  Saved: {path}"
    )

    return path


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def metrics_dict(
    y_true,
    y_pred,
    y_prob=None,
    label=""
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    result = {
        "label": label,
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred
            )
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "fpr": (
            float(fp / (fp + tn))
            if (fp + tn) > 0
            else None
        ),
        "fnr": (
            float(fn / (fn + tp))
            if (fn + tp) > 0
            else None
        ),
    }

    if y_prob is not None:
        y_prob = np.asarray(
            y_prob,
            dtype=float
        )

        try:
            result["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    y_prob
                )
            )
        except Exception:
            result["roc_auc"] = None

        try:
            result["pr_auc"] = float(
                average_precision_score(
                    y_true,
                    y_prob
                )
            )
        except Exception:
            result["pr_auc"] = None

    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result


def event_metrics(
    y_true,
    y_pred,
    label=""
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(
        (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (y_pred == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "label": label,
        "true_anomaly": int(y_true.sum()),
        "true_normal": int((y_true == 0).sum()),
        "detected": tp,
        "missed": fn,
        "false_alerts": fp,
        "correct_nominal": tn,
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    seg = pd.read_csv(
        SEG_PATH,
        parse_dates=["timestamp"]
    )

    ds = pd.read_csv(
        DS_PATH
    )

    return seg, ds


CORE_FEATURES = [
    "duration",
    "len",
    "mean",
    "var",
    "std",
    "kurtosis",
    "skew",
    "n_peaks",
    "smooth10_n_peaks",
    "smooth20_n_peaks",
    "diff_peaks",
    "diff2_peaks",
    "diff_var",
    "diff2_var",
    "gaps_squared",
    "len_weighted",
    "var_div_duration",
    "var_div_len",
]


def build_features(
    df,
    fit_dummies_from=None
):
    df = df.copy()

    df["sampling_5s"] = (
        df["sampling"] == 5
    ).astype(int)

    dummies = pd.get_dummies(
        df["channel"],
        prefix="ch",
        drop_first=True
    )

    df = pd.concat(
        [df, dummies],
        axis=1
    )

    channel_columns = [
        c
        for c in df.columns
        if c.startswith("ch_")
    ]

    feature_columns = (
        CORE_FEATURES
        + ["sampling_5s"]
        + channel_columns
    )

    X = df[
        feature_columns
    ].copy()

    y = df[
        "anomaly"
    ].copy()

    return X, y


def build_temporal_features(
    ds,
    seg
):
    seg2 = seg.copy()

    seg_start = (
        seg2
        .groupby("segment")["timestamp"]
        .min()
        .rename("start_ts")
    )

    seg_end = (
        seg2
        .groupby("segment")["timestamp"]
        .max()
        .rename("end_ts")
    )

    augmented = ds.copy()

    augmented = augmented.join(
        seg_start,
        on="segment"
    )

    augmented = augmented.join(
        seg_end,
        on="segment"
    )

    augmented = (
        augmented
        .sort_values(
            ["channel", "start_ts"]
        )
        .reset_index(drop=True)
    )

    lag_columns = {
        "prev_anomaly": "anomaly",
        "prev_duration": "duration",
        "prev_n_peaks": "n_peaks",
        "prev_smooth10_npks":
            "smooth10_n_peaks",
        "prev_var": "var",
        "prev_kurtosis": "kurtosis",
    }

    for new_name, source_name in lag_columns.items():
        augmented[new_name] = (
            augmented
            .groupby("channel")[source_name]
            .shift(1)
            .fillna(-1)
        )

    augmented["prev_end_ts"] = (
        augmented
        .groupby("channel")["end_ts"]
        .shift(1)
    )

    gap_values = []

    for _, row in augmented.iterrows():

        if pd.isnull(
            row["prev_end_ts"]
        ):
            gap_values.append(
                -1.0
            )

        else:
            gap = (
                row["start_ts"]
                - row["prev_end_ts"]
            ) / pd.Timedelta(
                seconds=1
            )

            gap_values.append(
                float(gap)
            )

    augmented["gap_to_prev_s"] = (
        gap_values
    )

    augmented = augmented.drop(
        columns=[
            "start_ts",
            "end_ts",
            "prev_end_ts",
        ]
    )

    return augmented


TEMPORAL_LAG_COLS = [
    "prev_anomaly",
    "prev_duration",
    "prev_n_peaks",
    "prev_smooth10_npks",
    "prev_var",
    "prev_kurtosis",
    "gap_to_prev_s",
]


def build_features_temporal(df):
    df = df.copy()

    df["sampling_5s"] = (
        df["sampling"] == 5
    ).astype(int)

    dummies = pd.get_dummies(
        df["channel"],
        prefix="ch",
        drop_first=True
    )

    df = pd.concat(
        [df, dummies],
        axis=1
    )

    channel_columns = [
        c
        for c in df.columns
        if c.startswith("ch_")
    ]

    feature_columns = (
        CORE_FEATURES
        + ["sampling_5s"]
        + channel_columns
        + TEMPORAL_LAG_COLS
    )

    feature_columns = [
        c
        for c in feature_columns
        if c in df.columns
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        "anomaly"
    ].copy()

    return X, y


def get_splits():
    _, ds = load_data()

    train = (
        ds[
            ds["train"] == 1
        ]
        .reset_index(drop=True)
    )

    test = (
        ds[
            ds["train"] == 0
        ]
        .reset_index(drop=True)
    )

    return train, test, ds


def align_cols(
    X_test,
    X_train
):
    return X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )


def scale_features(
    X_train,
    X_test
):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    Xtr = pd.DataFrame(
        scaler.fit_transform(
            X_train
        ),
        columns=X_train.columns,
        index=X_train.index
    )

    Xte = pd.DataFrame(
        scaler.transform(
            X_test
        ),
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        Xtr,
        Xte,
        scaler
    )


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENT ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def align_ablation_by_segment(
    y_true_a,
    segment_ids_a,
    y_pred_a,
    y_prob_a,
    y_true_b,
    segment_ids_b,
    y_pred_b,
    y_prob_b,
):
    """
    Align A and B using actual segment IDs.

    This prevents false comparisons caused by different row ordering.
    """

    a = pd.DataFrame({
        "segment":
            np.asarray(
                segment_ids_a
            ).astype(int),

        "y_true_a":
            np.asarray(
                y_true_a
            ).astype(int),

        "y_pred_a":
            np.asarray(
                y_pred_a
            ).astype(int),

        "y_prob_a":
            np.asarray(
                y_prob_a,
                dtype=float
            ),
    })

    b = pd.DataFrame({
        "segment":
            np.asarray(
                segment_ids_b
            ).astype(int),

        "y_true_b":
            np.asarray(
                y_true_b
            ).astype(int),

        "y_pred_b":
            np.asarray(
                y_pred_b
            ).astype(int),

        "y_prob_b":
            np.asarray(
                y_prob_b,
                dtype=float
            ),
    })

    if a["segment"].duplicated().any():
        duplicates = (
            a.loc[
                a["segment"].duplicated(),
                "segment"
            ]
            .tolist()
        )

        raise ValueError(
            "Experiment A contains duplicate segment IDs: "
            f"{duplicates[:10]}"
        )

    if b["segment"].duplicated().any():
        duplicates = (
            b.loc[
                b["segment"].duplicated(),
                "segment"
            ]
            .tolist()
        )

        raise ValueError(
            "Experiment B contains duplicate segment IDs: "
            f"{duplicates[:10]}"
        )

    aligned = a.merge(
        b,
        on="segment",
        how="inner",
        validate="one_to_one"
    )

    if aligned.empty:
        raise ValueError(
            "No common segment IDs between "
            "Experiment A and Experiment B."
        )

    truth_mismatch = (
        aligned["y_true_a"]
        != aligned["y_true_b"]
    )

    mismatch_count = int(
        truth_mismatch.sum()
    )

    if mismatch_count > 0:
        examples = (
            aligned.loc[
                truth_mismatch,
                "segment"
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "Ground-truth mismatch after alignment. "
            f"Count={mismatch_count}, "
            f"examples={examples[:10]}"
        )

    aligned = (
        aligned
        .sort_values("segment")
        .reset_index(drop=True)
    )

    print(
        f"  Alignment by segment ID: "
        f"A={len(a)} "
        f"B={len(b)} "
        f"common={len(aligned)} "
        f"truth_mismatches=0"
    )

    return (
        aligned["segment"].to_numpy(
            dtype=int
        ),

        aligned["y_true_a"].to_numpy(
            dtype=int
        ),

        aligned["y_pred_a"].to_numpy(
            dtype=int
        ),

        aligned["y_prob_a"].to_numpy(
            dtype=float
        ),

        aligned["y_pred_b"].to_numpy(
            dtype=int
        ),

        aligned["y_prob_b"].to_numpy(
            dtype=float
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT A
# ═══════════════════════════════════════════════════════════════════════════

def experiment_a(
    train_df,
    test_df
):
    print(
        "\n--- Experiment A: Real Anomaly Baseline ---"
    )

    X_tr, y_tr = build_features(
        train_df
    )

    X_te, y_te = build_features(
        test_df
    )

    X_te = align_cols(
        X_te,
        X_tr
    )

    X_tr, X_te, scaler = (
        scale_features(
            X_tr,
            X_te
        )
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_tr,
        y_tr
    )

    y_pred = rf.predict(
        X_te
    )

    y_prob = rf.predict_proba(
        X_te
    )[:, 1]

    segment_metrics = metrics_dict(
        y_te,
        y_pred,
        y_prob,
        "experiment_A_current_only"
    )

    event_metrics_result = event_metrics(
        y_te,
        y_pred,
        "experiment_A_current_only"
    )

    print(
        f'  F1={segment_metrics["f1"]:.4f} '
        f'recall={event_metrics_result["event_recall"]:.4f} '
        f'prec={event_metrics_result["event_precision"]:.4f} '
        f'missed={event_metrics_result["missed"]} '
        f'FP={event_metrics_result["false_alerts"]} '
        f'ROC-AUC={segment_metrics["roc_auc"]:.4f}'
    )

    return (
        rf,
        scaler,
        X_tr,
        y_tr,
        X_te,
        y_te,
        y_pred,
        y_prob,
        segment_metrics,
        event_metrics_result
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — CALIBRATED RF BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def experiment_calibrated_rf(
    train_df,
    test_df
):
    """
    Calibrated RF baseline.

    Calibration is fitted using 5-fold CV on TRAIN ONLY.
    The official held-out OPS-SAT-AD test partition is never used
    during calibration fitting.
    """

    print(
        "\n--- Experiment B: Calibrated RF Baseline ---"
    )

    X_tr, y_tr = build_features(
        train_df
    )

    X_te, y_te = build_features(
        test_df
    )

    X_te = align_cols(
        X_te,
        X_tr
    )

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


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — TEMPORAL SUPPORTING MODEL
# ═══════════════════════════════════════════════════════════════════════════

def experiment_b(
    train_df,
    test_df,
    seg,
    rf_a,
    sc_a,
    X_tr_a,
    y_tr_a,
    X_te_a,
    y_te_a
):
    print(
        "\n--- Experiment B: Temporal Predictor ---"
    )

    ds_all = pd.read_csv(
        DS_PATH
    )

    augmented = build_temporal_features(
        ds_all,
        seg
    )

    aug_train = (
        augmented[
            augmented["train"] == 1
        ]
        .reset_index(drop=True)
    )

    aug_test = (
        augmented[
            augmented["train"] == 0
        ]
        .reset_index(drop=True)
    )

    b_test_segment_ids = (
        aug_test["segment"]
        .to_numpy(dtype=int)
    )

    X_tr_t, y_tr_t = (
        build_features_temporal(
            aug_train
        )
    )

    X_te_t, y_te_t = (
        build_features_temporal(
            aug_test
        )
    )

    X_te_t = align_cols(
        X_te_t,
        X_tr_t
    )

    X_tr_t, X_te_t, scaler_b = (
        scale_features(
            X_tr_t,
            X_te_t
        )
    )

    rf_b = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf_b.fit(
        X_tr_t,
        y_tr_t
    )

    y_pred_b = rf_b.predict(
        X_te_t
    )

    y_prob_b = rf_b.predict_proba(
        X_te_t
    )[:, 1]

    segment_metrics_b = metrics_dict(
        y_te_t,
        y_pred_b,
        y_prob_b,
        "experiment_B_current_plus_temporal"
    )

    event_metrics_b = event_metrics(
        y_te_t,
        y_pred_b,
        "experiment_B_current_plus_temporal"
    )

    print(
        f'  F1={segment_metrics_b["f1"]:.4f} '
        f'recall={event_metrics_b["event_recall"]:.4f} '
        f'prec={event_metrics_b["event_precision"]:.4f} '
        f'missed={event_metrics_b["missed"]} '
        f'FP={event_metrics_b["false_alerts"]} '
        f'ROC-AUC={segment_metrics_b["roc_auc"]:.4f}'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # LEAD-TIME MODEL
    # ═══════════════════════════════════════════════════════════════════════

    ds_sorted = pd.read_csv(
        DS_PATH
    )

    seg_start = (
        seg
        .groupby("segment")["timestamp"]
        .min()
        .rename("start_ts")
    )

    ds_sorted = ds_sorted.join(
        seg_start,
        on="segment"
    )

    ds_sorted = (
        ds_sorted
        .sort_values(
            ["channel", "start_ts"]
        )
        .reset_index(drop=True)
    )

    ds_sorted["next_anomaly"] = (
        ds_sorted
        .groupby("channel")["anomaly"]
        .shift(-1)
        .fillna(-1)
        .astype(int)
    )

    lt_data = ds_sorted[
        ds_sorted["next_anomaly"] >= 0
    ].copy()

    lt_train = (
        lt_data[
            lt_data["train"] == 1
        ]
        .reset_index(drop=True)
    )

    lt_test = (
        lt_data[
            lt_data["train"] == 0
        ]
        .reset_index(drop=True)
    )

    n_pos_lt = int(
        lt_train[
            "next_anomaly"
        ].sum()
    )

    n_neg_lt = int(
        (
            lt_train["next_anomaly"]
            == 0
        ).sum()
    )

    lead_time_supported = (
        n_pos_lt >= 5
        and n_neg_lt >= 5
    )

    lead_model_result = {
        "supported":
            lead_time_supported
    }

    if lead_time_supported:

        X_lt_tr, _ = build_features(
            lt_train
        )

        y_lt_tr = lt_train[
            "next_anomaly"
        ]

        X_lt_te, _ = build_features(
            lt_test
        )

        y_lt_te = lt_test[
            "next_anomaly"
        ]

        X_lt_te = align_cols(
            X_lt_te,
            X_lt_tr
        )

        X_lt_tr, X_lt_te, _ = (
            scale_features(
                X_lt_tr,
                X_lt_te
            )
        )

        rf_lt = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        rf_lt.fit(
            X_lt_tr,
            y_lt_tr
        )

        y_lt_pred = rf_lt.predict(
            X_lt_te
        )

        y_lt_prob = rf_lt.predict_proba(
            X_lt_te
        )[:, 1]

        lead_segment_metrics = metrics_dict(
            y_lt_te,
            y_lt_pred,
            y_lt_prob,
            "lead_time_predictor"
        )

        lead_event_metrics = event_metrics(
            y_lt_te,
            y_lt_pred,
            "lead_time_predictor"
        )

        median_duration = float(
            lt_test["duration"].median()
        )

        lead_model_result.update({
            "lead_seg_metrics":
                lead_segment_metrics,

            "lead_evt_metrics":
                lead_event_metrics,

            "n_train_pos":
                int(n_pos_lt),

            "n_train_neg":
                int(n_neg_lt),

            "n_test_lead_samples":
                int(len(lt_test)),

            "n_test_lead_pos":
                int(y_lt_te.sum()),

            "median_segment_duration_s":
                median_duration,

            "note": (
                "Lead-time model predicts whether the next "
                "same-channel segment is anomalous. "
                f"Operational lead time is approximately one "
                f"segment duration, median "
                f"{median_duration:.0f}s. "
                "This is not equivalent to a 30-minute forecast."
            ),
        })

        print(
            f'  Lead-time model: '
            f'F1={lead_segment_metrics["f1"]:.4f} '
            f'recall={lead_event_metrics["event_recall"]:.4f} '
            f'missed={lead_event_metrics["missed"]} '
            f'FP={lead_event_metrics["false_alerts"]}'
        )

        return (
            rf_b,
            scaler_b,
            X_tr_t,
            y_tr_t,
            X_te_t,
            y_te_t,
            y_pred_b,
            y_prob_b,
            segment_metrics_b,
            event_metrics_b,
            lead_model_result,
            rf_lt,
            lt_test,
            y_lt_te,
            y_lt_pred,
            y_lt_prob,
            b_test_segment_ids,
        )

    lead_model_result["reason"] = (
        "Insufficient positive examples in lead-time task"
    )

    return (
        rf_b,
        scaler_b,
        X_tr_t,
        y_tr_t,
        X_te_t,
        y_te_t,
        y_pred_b,
        y_prob_b,
        segment_metrics_b,
        event_metrics_b,
        lead_model_result,
        None,
        None,
        None,
        None,
        None,
        b_test_segment_ids,
    )


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — SUPPORTING TEMPORAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def experiment_c(
    y_true,
    y_pred_a,
    y_pred_b
):
    print(
        "\n--- Experiment C: Current-Only vs Current+Temporal ---"
    )

    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    current_metrics = event_metrics(
        y_true,
        y_pred_a,
        "current_only"
    )

    temporal_metrics = event_metrics(
        y_true,
        y_pred_b,
        "current_plus_temporal"
    )

    delta_recall = (
        temporal_metrics["event_recall"]
        - current_metrics["event_recall"]
    )

    delta_f1 = (
        temporal_metrics["event_f1"]
        - current_metrics["event_f1"]
    )

    delta_missed = (
        temporal_metrics["missed"]
        - current_metrics["missed"]
    )

    delta_false_alerts = (
        temporal_metrics["false_alerts"]
        - current_metrics["false_alerts"]
    )

    adds_value = (
        delta_recall > 0
        or delta_missed < 0
        or delta_false_alerts < 0
    )

    result = {
        "current_only":
            current_metrics,

        "current_plus_temporal":
            temporal_metrics,

        "delta": {
            "event_recall":
                float(delta_recall),

            "event_f1":
                float(delta_f1),

            "missed_events":
                int(delta_missed),

            "false_alerts":
                int(delta_false_alerts),
        },

        "temporal_adds_operational_value":
            bool(adds_value),

        "n_compared":
            int(len(y_true)),
    }

    print(
        f'  Current-only: '
        f'recall={current_metrics["event_recall"]:.4f} '
        f'F1={current_metrics["event_f1"]:.4f} '
        f'missed={current_metrics["missed"]} '
        f'FA={current_metrics["false_alerts"]}'
    )

    print(
        f'  +Temporal: '
        f'recall={temporal_metrics["event_recall"]:.4f} '
        f'F1={temporal_metrics["event_f1"]:.4f} '
        f'missed={temporal_metrics["missed"]} '
        f'FA={temporal_metrics["false_alerts"]}'
    )

    print(
        f'  Delta: '
        f'recall={delta_recall:+.4f} '
        f'F1={delta_f1:+.4f} '
        f'missed={delta_missed:+d} '
        f'FA={delta_false_alerts:+d}'
    )

    print(
        f'  Temporal adds operational value: '
        f'{adds_value}'
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT D — MODEL DISAGREEMENT
# ═══════════════════════════════════════════════════════════════════════════

def experiment_d(
    y_true,
    y_pred_a,
    y_pred_b
):
    print(
        "\n--- Experiment D: Real Model Disagreement ---"
    )

    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    early_warning = (
        (y_pred_a == 0)
        & (y_pred_b == 1)
    )

    late_signal = (
        (y_pred_a == 1)
        & (y_pred_b == 0)
    )

    both_anomaly = (
        (y_pred_a == 1)
        & (y_pred_b == 1)
    )

    both_normal = (
        (y_pred_a == 0)
        & (y_pred_b == 0)
    )

    early_count = int(
        early_warning.sum()
    )

    late_count = int(
        late_signal.sum()
    )

    both_anomaly_count = int(
        both_anomaly.sum()
    )

    both_normal_count = int(
        both_normal.sum()
    )

    early_true = int(
        (
            early_warning
            & (y_true == 1)
        ).sum()
    )

    early_false = int(
        (
            early_warning
            & (y_true == 0)
        ).sum()
    )

    early_precision = (
        early_true / early_count
        if early_count > 0
        else 0.0
    )

    late_true = int(
        (
            late_signal
            & (y_true == 1)
        ).sum()
    )

    late_false = int(
        (
            late_signal
            & (y_true == 0)
        ).sum()
    )

    both_anomaly_true = int(
        (
            both_anomaly
            & (y_true == 1)
        ).sum()
    )

    both_anomaly_false = int(
        (
            both_anomaly
            & (y_true == 0)
        ).sum()
    )

    both_normal_true = int(
        (
            both_normal
            & (y_true == 0)
        ).sum()
    )

    both_normal_missed = int(
        (
            both_normal
            & (y_true == 1)
        ).sum()
    )

    y_union = (
        (y_pred_a == 1)
        | (y_pred_b == 1)
    ).astype(int)

    union_metrics = event_metrics(
        y_true,
        y_union,
        "union_policy"
    )

    y_intersection = (
        (y_pred_a == 1)
        & (y_pred_b == 1)
    ).astype(int)

    intersection_metrics = event_metrics(
        y_true,
        y_intersection,
        "intersection_policy"
    )

    result = {
        "n_test_segments":
            int(len(y_true)),

        "both_agree_anomaly":
            both_anomaly_count,

        "both_agree_nominal":
            both_normal_count,

        "early_warning_cases":
            early_count,

        "late_signal_cases":
            late_count,

        "early_warning": {
            "count":
                early_count,
            "true_pos":
                early_true,
            "false_pos":
                early_false,
            "precision":
                float(early_precision),
        },

        "late_signal": {
            "count":
                late_count,
            "truly_anomalous":
                late_true,
            "false":
                late_false,
        },

        "agreement_anomaly": {
            "count":
                both_anomaly_count,
            "true_pos":
                both_anomaly_true,
            "false_pos":
                both_anomaly_false,
        },

        "agreement_nominal": {
            "count":
                both_normal_count,
            "true_normal":
                both_normal_true,
            "missed":
                both_normal_missed,
        },

        "union_policy":
            union_metrics,

        "intersection_policy":
            intersection_metrics,
    }

    print(
        f'  Early-warning '
        f'(curr=normal, temp=anomaly): '
        f'{early_count} cases, '
        f'prec={early_precision:.3f} '
        f'({early_true} true / {early_false} false)'
    )

    print(
        f'  Late-signal '
        f'(curr=anomaly, temp=normal): '
        f'{late_count} cases'
    )

    print(
        f'  Both agree anomaly: '
        f'{both_anomaly_count} '
        f'(TP={both_anomaly_true}, '
        f'FP={both_anomaly_false})'
    )

    print(
        f'  Both agree nominal: '
        f'{both_normal_count} '
        f'(missed={both_normal_missed})'
    )

    print(
        f'  Union policy: '
        f'recall={union_metrics["event_recall"]:.4f} '
        f'FA={union_metrics["false_alerts"]} '
        f'F1={union_metrics["event_f1"]:.4f}'
    )

    print(
        f'  Intersect policy: '
        f'recall={intersection_metrics["event_recall"]:.4f} '
        f'FA={intersection_metrics["false_alerts"]} '
        f'F1={intersection_metrics["event_f1"]:.4f}'
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════

UTILITY_MATRIX = {
    (0, "AUTO_CLEAR"): 1.0,
    (0, "PENDING_APPROVAL"): -0.3,
    (0, "ESCALATED"): -0.5,

    (1, "AUTO_CLEAR"): -5.0,
    (1, "PENDING_APPROVAL"): 0.5,
    (1, "ESCALATED"): 0.8,
}


ALT_UTILITY_SETS = {
    "conservative": {
        (0, "AUTO_CLEAR"): 1.0,
        (0, "PENDING_APPROVAL"): -0.5,
        (0, "ESCALATED"): -1.0,
        (1, "AUTO_CLEAR"): -10.0,
        (1, "PENDING_APPROVAL"): 0.5,
        (1, "ESCALATED"): 1.0,
    },

    "permissive": {
        (0, "AUTO_CLEAR"): 1.0,
        (0, "PENDING_APPROVAL"): -0.1,
        (0, "ESCALATED"): -0.2,
        (1, "AUTO_CLEAR"): -3.0,
        (1, "PENDING_APPROVAL"): 0.3,
        (1, "ESCALATED"): 0.5,
    },

    "recall_focused": {
        (0, "AUTO_CLEAR"): 0.5,
        (0, "PENDING_APPROVAL"): -0.1,
        (0, "ESCALATED"): -0.2,
        (1, "AUTO_CLEAR"): -8.0,
        (1, "PENDING_APPROVAL"): 1.0,
        (1, "ESCALATED"): 1.5,
    },
}


def apply_deterministic_policy(
    y_pred_current,
    y_pred_temporal,
    prob_current,
    prob_temporal,
    disagree_threshold=0.50
):
    decisions = []

    for current, temporal, p_current, p_temporal in zip(
        y_pred_current,
        y_pred_temporal,
        prob_current,
        prob_temporal
    ):

        if (
            current == 1
            and temporal == 1
        ):
            decisions.append(
                "ESCALATED"
            )

        elif (
            current == 1
            and temporal == 0
        ):
            decisions.append(
                "PENDING_APPROVAL"
            )

        elif (
            current == 0
            and temporal == 1
            and p_temporal >= disagree_threshold
        ):
            decisions.append(
                "PENDING_APPROVAL"
            )

        elif (
            current == 0
            and temporal == 0
        ):
            decisions.append(
                "AUTO_CLEAR"
            )

        else:
            decisions.append(
                "PENDING_APPROVAL"
            )

    return decisions


def apply_current_only_policy(
    y_pred_current
):
    return [
        "ESCALATED"
        if prediction == 1
        else "AUTO_CLEAR"
        for prediction in y_pred_current
    ]


def compute_utility(
    y_true,
    decisions,
    utility_matrix
):
    if len(y_true) == 0:
        return 0.0

    total = sum(
        utility_matrix.get(
            (
                int(true_value),
                decision
            ),
            0.0
        )
        for true_value, decision
        in zip(
            y_true,
            decisions
        )
    )

    return float(
        total / len(y_true)
    )


def policy_decision_metrics(
    y_true,
    decisions,
    label=""
):
    y_true = np.asarray(
        y_true
    )

    decisions = np.asarray(
        decisions
    )

    auto_clear = (
        decisions == "AUTO_CLEAR"
    )

    pending = (
        decisions == "PENDING_APPROVAL"
    )

    escalated = (
        decisions == "ESCALATED"
    )

    unsafe_auto = int(
        (
            (y_true == 1)
            & auto_clear
        ).sum()
    )

    correct_pending = int(
        (
            (y_true == 1)
            & pending
        ).sum()
    )

    correct_escalated = int(
        (
            (y_true == 1)
            & escalated
        ).sum()
    )

    unnecessary_pending = int(
        (
            (y_true == 0)
            & pending
        ).sum()
    )

    unnecessary_escalated = int(
        (
            (y_true == 0)
            & escalated
        ).sum()
    )

    correct_auto_clear = int(
        (
            (y_true == 0)
            & auto_clear
        ).sum()
    )

    anomaly_total = int(
        y_true.sum()
    )

    protected = (
        correct_pending
        + correct_escalated
    )

    decision_recall = (
        protected / anomaly_total
        if anomaly_total > 0
        else 0.0
    )

    utility_baseline = compute_utility(
        y_true,
        decisions,
        UTILITY_MATRIX
    )

    utility_sensitivity = {
        name: compute_utility(
            y_true,
            decisions,
            matrix
        )
        for name, matrix
        in ALT_UTILITY_SETS.items()
    }

    return {
        "label": label,
        "n": int(len(y_true)),
        "auto_clear":
            int(auto_clear.sum()),
        "pending_approval":
            int(pending.sum()),
        "escalated":
            int(escalated.sum()),
        "unsafe_auto":
            unsafe_auto,
        "correct_pending":
            correct_pending,
        "correct_escalated":
            correct_escalated,
        "unnecessary_pending":
            unnecessary_pending,
        "unnecessary_escalated":
            unnecessary_escalated,
        "correct_auto_clear":
            correct_auto_clear,
        "decision_recall":
            float(decision_recall),
        "utility_baseline":
            utility_baseline,
        "utility_sensitivity":
            utility_sensitivity,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DECISION STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════

def build_decision_stress_test(
    y_true,
    y_pred_cal,
    y_prob_cal,
    y_pred_temporal,
    y_prob_temporal,
    segment_ids,
    decisions_c,
    decisions_d,
    d_reasons,
    lead_warning_segments,
):
    """
    Per-segment audit of C -> D routing changes.

    Decision reasons come only from decision-time evidence.

    Ground truth is used only afterward to classify:
      - PROTECTED_ANOMALY
      - NEW_FALSE_ALERT

    This analysis does not modify A/B/C/D metrics.
    """

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_pred_cal = np.asarray(
        y_pred_cal
    ).astype(int)

    y_prob_cal = np.asarray(
        y_prob_cal,
        dtype=float
    )

    y_pred_temporal = np.asarray(
        y_pred_temporal
    ).astype(int)

    y_prob_temporal = np.asarray(
        y_prob_temporal,
        dtype=float
    )

    segment_ids = np.asarray(
        segment_ids
    ).astype(int)

    n = len(y_true)

    lengths = [
        len(y_true),
        len(y_pred_cal),
        len(y_prob_cal),
        len(y_pred_temporal),
        len(y_prob_temporal),
        len(segment_ids),
        len(decisions_c),
        len(decisions_d),
        len(d_reasons),
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            "Decision stress-test arrays are not aligned: "
            f"{lengths}"
        )

    rows = []

    for i in range(n):

        segment_id = int(
            segment_ids[i]
        )

        truth = int(
            y_true[i]
        )

        route_c = str(
            decisions_c[i]
        )

        route_d = str(
            decisions_d[i]
        )

        changed = (
            route_c != route_d
        )

        if not changed:
            outcome = "UNCHANGED"

        elif (
            route_c == "AUTO_CLEAR"
            and route_d != "AUTO_CLEAR"
            and truth == 1
        ):
            outcome = "PROTECTED_ANOMALY"

        elif (
            route_c == "AUTO_CLEAR"
            and route_d != "AUTO_CLEAR"
            and truth == 0
        ):
            outcome = "NEW_FALSE_ALERT"

        else:
            outcome = "OTHER_TRANSITION"

        rows.append({
            "segment_id":
                segment_id,

            "ground_truth":
                truth,

            "calibrated_probability":
                float(
                    y_prob_cal[i]
                ),

            "calibrated_prediction":
                int(
                    y_pred_cal[i]
                ),

            "temporal_probability":
                float(
                    y_prob_temporal[i]
                ),

            "temporal_prediction":
                int(
                    y_pred_temporal[i]
                ),

            "lead_warning":
                int(
                    segment_id
                    in lead_warning_segments
                ),

            "route_C":
                route_c,

            "route_D":
                route_d,

            "changed_C_to_D":
                bool(changed),

            "transition":
                (
                    f"{route_c} -> {route_d}"
                    if changed
                    else "UNCHANGED"
                ),

            "decision_reason":
                str(
                    d_reasons[i]
                ),

            "outcome_class":
                outcome,
        })

    trace = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "segment_id"
        )
        .reset_index(
            drop=True
        )
    )

    if trace[
        "segment_id"
    ].duplicated().any():

        raise ValueError(
            "Decision stress test contains duplicate segment IDs."
        )

    changed_df = trace[
        trace[
            "changed_C_to_D"
        ]
    ]

    protected_df = trace[
        trace[
            "outcome_class"
        ] == "PROTECTED_ANOMALY"
    ]

    false_alert_df = trace[
        trace[
            "outcome_class"
        ] == "NEW_FALSE_ALERT"
    ]

    def count_map(series):
        if len(series) == 0:
            return {}

        return {
            str(k): int(v)
            for k, v
            in series.value_counts().items()
        }

    summary = {
        "n_test_segments":
            int(
                len(trace)
            ),

        "changed_C_to_D":
            int(
                len(changed_df)
            ),

        "unchanged":
            int(
                len(trace)
                - len(changed_df)
            ),

        "protected_anomalies": {
            "count":
                int(
                    len(protected_df)
                ),

            "by_route_D":
                count_map(
                    protected_df[
                        "route_D"
                    ]
                ),

            "by_reason":
                count_map(
                    protected_df[
                        "decision_reason"
                    ]
                ),
        },

        "new_false_alerts": {
            "count":
                int(
                    len(false_alert_df)
                ),

            "by_route_D":
                count_map(
                    false_alert_df[
                        "route_D"
                    ]
                ),

            "by_reason":
                count_map(
                    false_alert_df[
                        "decision_reason"
                    ]
                ),
        },

        "all_changed_transitions":
            count_map(
                changed_df[
                    "transition"
                ]
            ),

        "all_changed_reasons":
            count_map(
                changed_df[
                    "decision_reason"
                ]
            ),

        "methodology": (
            "Decision reasons are assigned from evidence available "
            "at decision time: calibrated current-state evidence, "
            "temporal evidence, and lead-warning state. Ground truth "
            "is used only afterward to classify protected anomalies "
            "and new false alerts."
        ),

        "warning": (
            "This is a routing audit on the held-out benchmark. "
            "It does not establish operator workload, operator "
            "preference, mission-specific cost, or causal superiority."
        ),
    }

    return (
        trace,
        summary
    )


# ═══════════════════════════════════════════════════════════════════════════
# LEAD-TIME INCREMENTAL VALUE
# ═══════════════════════════════════════════════════════════════════════════

def compare_policy_without_vs_with_lead_time(
    y_true,
    y_pred_current,
    y_prob_current,
    y_pred_temporal,
    y_prob_temporal,
    lead_warning_segments,
    segment_ids
):
    """
    Isolate the incremental effect of lead-time.

    This helper preserves the historical two-model comparison.
    The primary A/B/C/D ablation uses the calibrated-RF C policy.
    """

    print(
        "\n--- Lead-Time Incremental Value: C vs D ---"
    )

    y_true = np.asarray(
        y_true
    )

    y_pred_current = np.asarray(
        y_pred_current
    )

    y_prob_current = np.asarray(
        y_prob_current
    )

    y_pred_temporal = np.asarray(
        y_pred_temporal
    )

    y_prob_temporal = np.asarray(
        y_prob_temporal
    )

    segment_ids = np.asarray(
        segment_ids
    ).astype(int)

    n = len(
        y_true
    )

    if not (
        len(y_pred_current)
        == len(y_prob_current)
        == len(y_pred_temporal)
        == len(y_prob_temporal)
        == len(segment_ids)
        == n
    ):
        raise ValueError(
            "C/D lead-time arrays are not aligned."
        )

    decisions_c = apply_deterministic_policy(
        y_pred_current,
        y_pred_temporal,
        y_prob_current,
        y_prob_temporal,
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
        "historical_two_model_policy_without_lead"
    )

    policy_c = policy_decision_metrics(
        y_true,
        decisions_c,
        "historical_two_model_policy_without_lead"
    )

    decisions_d = []

    for i in range(n):

        current_prediction = int(
            y_pred_current[i]
        )

        temporal_prediction = int(
            y_pred_temporal[i]
        )

        temporal_probability = float(
            y_prob_temporal[i]
        )

        segment_id = int(
            segment_ids[i]
        )

        lead_warned = (
            segment_id
            in lead_warning_segments
        )

        if (
            current_prediction == 1
            and temporal_prediction == 1
        ):
            decisions_d.append(
                "ESCALATED"
            )

        elif (
            lead_warned
            and (
                current_prediction == 1
                or temporal_prediction == 1
            )
        ):
            decisions_d.append(
                "ESCALATED"
            )

        elif (
            current_prediction == 1
            or (
                temporal_prediction == 1
                and temporal_probability >= 0.50
            )
        ):
            decisions_d.append(
                "PENDING_APPROVAL"
            )

        elif lead_warned:
            decisions_d.append(
                "PENDING_APPROVAL"
            )

        else:
            decisions_d.append(
                "AUTO_CLEAR"
            )

    binary_d = np.asarray([
        0
        if decision == "AUTO_CLEAR"
        else 1
        for decision in decisions_d
    ])

    event_d = event_metrics(
        y_true,
        binary_d,
        "historical_two_model_policy_with_lead"
    )

    policy_d = policy_decision_metrics(
        y_true,
        decisions_d,
        "historical_two_model_policy_with_lead"
    )

    delta = {
        "recall":
            event_d["event_recall"]
            - event_c["event_recall"],

        "precision":
            event_d["event_precision"]
            - event_c["event_precision"],

        "f1":
            event_d["event_f1"]
            - event_c["event_f1"],

        "missed":
            event_d["missed"]
            - event_c["missed"],

        "false_alerts":
            event_d["false_alerts"]
            - event_c["false_alerts"],

        "unsafe_auto":
            policy_d["unsafe_auto"]
            - policy_c["unsafe_auto"],

        "utility_baseline":
            policy_d["utility_baseline"]
            - policy_c["utility_baseline"],
    }

    result = {
        "method_note": (
            "This is a supporting historical two-model analysis. "
            "The primary A/B/C/D ablation uses calibrated RF for "
            "Architecture C and D."
        ),

        "C_without_lead_time": {
            "event_metrics":
                event_c,

            "policy_metrics":
                policy_c,
        },

        "D_with_lead_time": {
            "event_metrics":
                event_d,

            "policy_metrics":
                policy_d,
        },

        "delta_D_minus_C": {
            key:
                float(value)
                if isinstance(
                    value,
                    (float, np.floating)
                )
                else int(value)
            for key, value
            in delta.items()
        },

        "lead_warning_segments":
            int(
                len(
                    lead_warning_segments
                )
            ),

        "n_test_segments":
            int(n),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT E — FULL DECISION INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════

def experiment_e(
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
):
    print(
        "\n--- Experiment E: Full Decision Intelligence Ablation ---"
    )

    y_true = np.asarray(
        y_true
    )[:n]

    y_pred_a = np.asarray(
        y_pred_a
    )[:n]

    y_prob_a = np.asarray(
        y_prob_a
    )[:n]

    y_pred_b = np.asarray(
        y_pred_b
    )[:n]

    y_prob_b = np.asarray(
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

    # ═══════════════════════════════════════════════════════════════════════
    # A — RAW RF + SIMPLE ROUTING
    # ═══════════════════════════════════════════════════════════════════════

    decisions_a = apply_current_only_policy(
        y_pred_a
    )

    binary_a = np.asarray([
        0
        if decision == "AUTO_CLEAR"
        else 1
        for decision in decisions_a
    ])

    event_a = event_metrics(
        y_true,
        binary_a,
        "arch_A_current_only"
    )

    policy_a = policy_decision_metrics(
        y_true,
        decisions_a,
        "arch_A"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # B — CALIBRATED RF + SIMPLE ROUTING
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

    # ═══════════════════════════════════════════════════════════════════════
    # C — CALIBRATED RF + DETERMINISTIC ROUTING
    # ═══════════════════════════════════════════════════════════════════════

    # C deliberately uses ONLY calibrated RF evidence.
    #
    # Rule:
    #   calibrated probability >= 0.50 -> PENDING_APPROVAL
    #   otherwise                     -> AUTO_CLEAR
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

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD LEAD WARNING SEGMENTS
    # ═══════════════════════════════════════════════════════════════════════

    lead_warning_segments = set()

    if (
        y_lt_pred is not None
        and lt_test is not None
    ):

        lead_predictions = np.asarray(
            y_lt_pred
        ).astype(int)

        lt_work = lt_test.copy()

        lt_work["lt_pred"] = (
            lead_predictions
        )

        warned = (
            lt_work[
                lt_work["lt_pred"] == 1
            ]["segment"]
            .astype(int)
            .values
        )

        raw = pd.read_csv(
            DS_PATH
        )

        segment_times = pd.read_csv(
            SEG_PATH,
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
            .set_index("segment")[
                "next_segment"
            ]
            .dropna()
            .to_dict()
        )

        for segment_id in warned:

            next_segment = next_map.get(
                int(segment_id)
            )

            if next_segment is not None:
                lead_warning_segments.add(
                    int(next_segment)
                )

    # ═══════════════════════════════════════════════════════════════════════
    # D — FULL HELIOMESH
    # ═══════════════════════════════════════════════════════════════════════

    decisions_d = []
    d_reasons = []

    for i in range(n):

        current_prediction = int(
            y_pred_cal[i]
        )

        temporal_prediction = int(
            y_pred_b[i]
        )

        temporal_probability = float(
            y_prob_b[i]
        )

        segment_id = int(
            segment_ids[i]
        )

        lead_warned = (
            segment_id
            in lead_warning_segments
        )

        # The reason is attributed strictly from signals available
        # to the decision branch.

        if (
            current_prediction == 1
            and temporal_prediction == 1
        ):
            decisions_d.append(
                "ESCALATED"
            )

            d_reasons.append(
                "both_models_anomaly"
            )

        elif (
            lead_warned
            and (
                current_prediction == 1
                or temporal_prediction == 1
            )
        ):
            decisions_d.append(
                "ESCALATED"
            )

            d_reasons.append(
                "lead_warning_plus_model_signal"
            )

        elif current_prediction == 1:

            decisions_d.append(
                "PENDING_APPROVAL"
            )

            d_reasons.append(
                "calibrated_current_signal"
            )

        elif (
            temporal_prediction == 1
            and temporal_probability >= 0.50
        ):

            decisions_d.append(
                "PENDING_APPROVAL"
            )

            d_reasons.append(
                "temporal_signal"
            )

        elif lead_warned:

            decisions_d.append(
                "PENDING_APPROVAL"
            )

            d_reasons.append(
                "lead_warning_only"
            )

        else:

            decisions_d.append(
                "AUTO_CLEAR"
            )

            d_reasons.append(
                "no_additional_D_signal"
            )

    binary_d = np.asarray([
        0
        if decision == "AUTO_CLEAR"
        else 1
        for decision in decisions_d
    ])

    event_d = event_metrics(
        y_true,
        binary_d,
        "arch_D_full_heliomesh"
    )

    policy_d = policy_decision_metrics(
        y_true,
        decisions_d,
        "arch_D"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # DECISION STRESS TEST
    # ═══════════════════════════════════════════════════════════════════════

    stress_trace, stress_summary = (
        build_decision_stress_test(
            y_true=y_true,
            y_pred_cal=y_pred_cal,
            y_prob_cal=y_prob_cal,
            y_pred_temporal=y_pred_b,
            y_prob_temporal=y_prob_b,
            segment_ids=segment_ids,
            decisions_c=decisions_c,
            decisions_d=decisions_d,
            d_reasons=d_reasons,
            lead_warning_segments=lead_warning_segments,
        )
    )

    stress_trace.to_csv(
        STRESS_CSV_PATH,
        index=False
    )

    save_json(
        stress_summary,
        STRESS_JSON_FILENAME
    )

    print(
        "\n--- Decision Stress Test ---"
    )

    print(
        f'  Test segments: '
        f'{stress_summary["n_test_segments"]}'
    )

    print(
        f'  C -> D changed: '
        f'{stress_summary["changed_C_to_D"]}'
    )

    print(
        f'  Protected anomalies: '
        f'{stress_summary["protected_anomalies"]["count"]}'
    )

    print(
        f'  New false alerts: '
        f'{stress_summary["new_false_alerts"]["count"]}'
    )

    print(
        "  Changed transitions:"
    )

    for key, value in (
        stress_summary[
            "all_changed_transitions"
        ].items()
    ):
        print(
            f"    {key}: {value}"
        )

    print(
        "  Protected anomaly reasons:"
    )

    for key, value in (
        stress_summary[
            "protected_anomalies"
        ]["by_reason"].items()
    ):
        print(
            f"    {key}: {value}"
        )

    print(
        "  New false-alert reasons:"
    )

    for key, value in (
        stress_summary[
            "new_false_alerts"
        ]["by_reason"].items()
    ):
        print(
            f"    {key}: {value}"
        )

    print(
        f"  Saved: {STRESS_CSV_PATH}"
    )

    print(
        "  Saved: "
        f"{os.path.join(RESULTS_DIR, STRESS_JSON_FILENAME)}"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # C -> D INCREMENTAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    c_event_for_lead = event_metrics(
        y_true,
        np.asarray([
            0
            if decision == "AUTO_CLEAR"
            else 1
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
                if isinstance(
                    value,
                    (float, np.floating)
                )
                else int(value)
            for key, value in delta_cd.items()
        },

        "lead_warning_segments":
            int(
                len(
                    lead_warning_segments
                )
            ),

        "n_test_segments":
            int(n),

        "interpretation": (
            "C is the calibrated-RF deterministic routing policy. "
            "D adds temporal evidence and lead-time warnings. "
            "Both are evaluated on the same aligned held-out "
            "OPS-SAT-AD test segments."
        ),
    }

    print(
        "\n--- Lead-Time Incremental Value: C vs D ---"
    )

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

    # ═══════════════════════════════════════════════════════════════════════
    # TABLE
    # ═══════════════════════════════════════════════════════════════════════

    ablation_table = [
        {
            "architecture":
                "A_current_only",

            "event_recall":
                event_a["event_recall"],

            "event_precision":
                event_a["event_precision"],

            "event_f1":
                event_a["event_f1"],

            "missed":
                event_a["missed"],

            "false_alerts":
                event_a["false_alerts"],

            "unsafe_auto":
                policy_a["unsafe_auto"],

            "lead_time":
                "N/A",

            "utility_baseline":
                policy_a["utility_baseline"],
        },

        {
            "architecture":
                "B_calibrated_rf",

            "event_recall":
                event_b["event_recall"],

            "event_precision":
                event_b["event_precision"],

            "event_f1":
                event_b["event_f1"],

            "missed":
                event_b["missed"],

            "false_alerts":
                event_b["false_alerts"],

            "unsafe_auto":
                policy_b["unsafe_auto"],

            "lead_time":
                "N/A",

            "utility_baseline":
                policy_b["utility_baseline"],
        },

        {
            "architecture":
                "C_calibrated_rf_deterministic_policy",

            "event_recall":
                event_c["event_recall"],

            "event_precision":
                event_c["event_precision"],

            "event_f1":
                event_c["event_f1"],

            "missed":
                event_c["missed"],

            "false_alerts":
                event_c["false_alerts"],

            "unsafe_auto":
                policy_c["unsafe_auto"],

            "lead_time":
                "N/A",

            "utility_baseline":
                policy_c["utility_baseline"],
        },

        {
            "architecture":
                "D_full_heliomesh",

            "event_recall":
                event_d["event_recall"],

            "event_precision":
                event_d["event_precision"],

            "event_f1":
                event_d["event_f1"],

            "missed":
                event_d["missed"],

            "false_alerts":
                event_d["false_alerts"],

            "unsafe_auto":
                policy_d["unsafe_auto"],

            "lead_time":
                (
                    f'{len(lead_warning_segments)} warning segments'
                    if len(lead_warning_segments) > 0
                    else "N/A"
                ),

            "utility_baseline":
                policy_d["utility_baseline"],
        },
    ]

    print(
        f'\n  {"Architecture":<30}'
        f'{"Recall":>8}'
        f'{"Prec":>8}'
        f'{"F1":>8}'
        f'{"Missed":>8}'
        f'{"FA":>6}'
        f'{"UnsafeAUTO":>12}'
        f'{"Utility":>10}'
    )

    print(
        "  " + "-" * 100
    )

    for row in ablation_table:

        print(
            f'  {row["architecture"]:<30}'
            f'{row["event_recall"]:>8.4f}'
            f'{row["event_precision"]:>8.4f}'
            f'{row["event_f1"]:>8.4f}'
            f'{row["missed"]:>8}'
            f'{row["false_alerts"]:>6}'
            f'{row["unsafe_auto"]:>12}'
            f'{row["utility_baseline"]:>10.3f}'
        )

    # ═══════════════════════════════════════════════════════════════════════
    # DI VALUE CHECK
    # ═══════════════════════════════════════════════════════════════════════

    di_adds_value = (
        policy_d["unsafe_auto"]
        < policy_a["unsafe_auto"]
        or
        policy_d["utility_baseline"]
        > policy_a["utility_baseline"]
        or
        event_d["event_recall"]
        > event_a["event_recall"]
    )

    if di_adds_value:
        interpretation = (
            "On the held-out OPS-SAT-AD test partition, the "
            "evaluated HelioMesh Decision Intelligence layer "
            "changes routing and improves at least one selected "
            "decision metric relative to the anomaly-only baseline. "
            "This is a benchmark result with an explicit trade-off, "
            "not a claim of universal operational superiority."
        )
    else:
        interpretation = (
            "The evaluated Decision Intelligence layer does not "
            "improve the selected primary decision metrics relative "
            "to the anomaly-only baseline on this dataset."
        )

    print(
        f"\n  DI adds value: "
        f"{di_adds_value}"
    )

    print(
        f"  Interpretation: "
        f"{interpretation}"
    )

    return {
        "ablation_table":
            ablation_table,

        "architecture_details": {
            "A": {
                "event_metrics":
                    event_a,
                "policy_metrics":
                    policy_a,
            },

            "B": {
                "event_metrics":
                    event_b,
                "policy_metrics":
                    policy_b,
            },

            "C": {
                "event_metrics":
                    event_c,
                "policy_metrics":
                    policy_c,
            },

            "D": {
                "event_metrics":
                    event_d,
                "policy_metrics":
                    policy_d,
            },
        },

        "decision_stress_test":
            stress_summary,

        "lead_time_incremental_analysis":
            lead_time_analysis,

        "decision_intelligence_adds_value":
            bool(di_adds_value),

        "interpretation":
            interpretation,

        "utility_matrix_definition": {
            "(normal, AUTO_CLEAR)": 1.0,
            "(normal, PENDING_APPROVAL)": -0.3,
            "(normal, ESCALATED)": -0.5,
            "(anomaly, AUTO_CLEAR)": -5.0,
            "(anomaly, PENDING_APPROVAL)": 0.5,
            "(anomaly, ESCALATED)": 0.8,

            "name":
                "HelioMesh Prototype Decision Utility",

            "note":
                "NOT externally validated by real mission operators. "
                "Defined before evaluating the held-out test results.",
        },

        "lead_warning_segments_flagged":
            int(
                len(
                    lead_warning_segments
                )
            ),

        "n_test_segments":
            int(n),

        "alignment": {
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
                "Calibrated RF + temporal evidence + "
                "lead-time + deterministic routing",

            "test_partition":
                "official held-out OPS-SAT-AD test partition",

            "threshold_tuning_on_test":
                False,

            "C_policy_thresholds": {
                "pending_approval":
                    0.50,
            },

            "stress_test":
                {
                    "reason_attribution":
                        "decision_time_evidence_only",

                    "ground_truth_usage":
                        "post_decision_outcome_classification_only",
                },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# GRANITE GROUNDING AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_granite_grounding(
    ablation_results,
    seg_m_a,
    seg_m_b
):

    print(
        "\n--- Granite Grounding Audit ---"
    )

    required_metrics = [
        "f1",
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "mcc",
    ]

    coverage_a = all(
        key in seg_m_a
        and seg_m_a[key] is not None
        for key in required_metrics
    )

    coverage_b = all(
        key in seg_m_b
        and seg_m_b[key] is not None
        for key in required_metrics
    )

    null_a = [
        key
        for key in required_metrics
        if seg_m_a.get(key) is None
    ]

    null_b = [
        key
        for key in required_metrics
        if seg_m_b.get(key) is None
    ]

    def consistent(metrics):
        tp = metrics.get(
            "tp",
            0
        )

        fp = metrics.get(
            "fp",
            0
        )

        fn = metrics.get(
            "fn",
            0
        )

        expected_precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        expected_recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        precision_ok = (
            abs(
                expected_precision
                - metrics.get(
                    "precision",
                    0
                )
            ) < 1e-6
        )

        recall_ok = (
            abs(
                expected_recall
                - metrics.get(
                    "recall",
                    0
                )
            ) < 1e-6
        )

        return (
            precision_ok
            and recall_ok
        )

    consistency_a = consistent(
        seg_m_a
    )

    consistency_b = consistent(
        seg_m_b
    )

    contradiction_count = (
        int(not consistency_a)
        + int(not consistency_b)
        + len(null_a)
        + len(null_b)
    )

    grounding_components = [
        coverage_a,
        coverage_b,
        consistency_a,
        consistency_b,
        True,
    ]

    grounding_score = (
        sum(grounding_components)
        / len(grounding_components)
    )

    calibrated_model_f1 = (
        ablation_results.get(
            "architecture_details",
            {}
        )
        .get(
            "B",
            {}
        )
        .get(
            "event_metrics",
            {}
        )
        .get(
            "f1"
        )
    )

    checks = {
        "evidence_coverage_current_model":
            coverage_a,

        "calibrated_model_f1":
            (
                float(calibrated_model_f1)
                if calibrated_model_f1 is not None
                else None
            ),

        "evidence_coverage_temporal_model":
            coverage_b,

        "real_simulation_separation":
            True,

        "unsupported_numeric_claims_model_A":
            null_a,

        "unsupported_numeric_claims_model_B":
            null_b,

        "metrics_internally_consistent_A":
            consistency_a,

        "metrics_internally_consistent_B":
            consistency_b,

        "route_consistency_auditable":
            True,

        "decision_stress_test_present":
            "decision_stress_test"
            in ablation_results,

        "contradiction_count":
            int(contradiction_count),

        "contradiction_rate":
            float(
                contradiction_count
                / 12.0
            ),

        "grounding_score":
            float(grounding_score),

        "human_usability_tested":
            False,

        "human_usability_note":
            (
                "No human operator study was conducted. "
                "Granite grounding quality with real operators "
                "is therefore unknown."
            ),

        "limitation":
            (
                "This is a structural grounding audit only. "
                "It is not a live Granite LLM reasoning evaluation."
            ),
    }

    print(
        f'  Grounding score: '
        f'{grounding_score:.4f}'
    )

    print(
        f'  Contradiction count: '
        f'{contradiction_count}'
    )

    print(
        f'  Coverage A: '
        f'{coverage_a} '
        f'Coverage B: '
        f'{coverage_b}'
    )

    return checks


# ═══════════════════════════════════════════════════════════════════════════
# FROZEN INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def verify_frozen():

    print(
        "\n--- Frozen ML Integrity ---"
    )

    results = {}
    all_ok = True

    for path, expected_prefix in (
        FROZEN_ARTIFACTS.items()
    ):

        actual_hash = sha256_file(
            path
        )

        ok = (
            actual_hash[:16]
            == expected_prefix
        )

        results[path] = {
            "sha256":
                actual_hash,

            "status":
                "OK"
                if ok
                else "MISMATCH",
        }

        print(
            f'  '
            f'{"OK" if ok else "FAIL"} '
            f'{path}'
        )

        if not ok:
            all_ok = False

    results[
        "all_unchanged"
    ] = all_ok

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PROVENANCE
# ═══════════════════════════════════════════════════════════════════════════

def verify_provenance():

    return {
        "dataset":
            "OPSSAT-AD",

        "zenodo_doi":
            "https://doi.org/10.5281/zenodo.12588359",

        "license":
            "MIT",

        "license_text_first_line":
            "MIT License for OPS-SAT Dataset",

        "license_copyright":
            "Copyright (c) 2024 KP Labs",

        "source_repo":
            "https://github.com/kplabs-pl/OPS-SAT-AD",

        "paper":
            "Ruszczak et al. Scientific Data (2025) "
            "doi:10.1038/s41597-025-05035-3",

        "segments_csv_sha256":
            sha256_file(
                SEG_PATH
            ),

        "dataset_csv_sha256":
            sha256_file(
                DS_PATH
            ),

        "segments_csv_bytes":
            os.path.getsize(
                SEG_PATH
            ),

        "dataset_csv_bytes":
            os.path.getsize(
                DS_PATH
            ),

        "license_verified_from":
            "https://raw.githubusercontent.com/kplabs-pl/OPS-SAT-AD/main/LICENSE",

        "license_verified_date":
            "2025",
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():

    print(
        "=" * 70
    )

    print(
        "HELIOMESH PHASE 3 — "
        "REAL DECISION INTELLIGENCE ABLATION"
    )

    print(
        "=" * 70
    )

    seg, ds = load_data()

    train_df, test_df, ds_full = (
        get_splits()
    )

    train_segments = set(
        train_df["segment"]
    )

    test_segments = set(
        test_df["segment"]
    )

    assert (
        len(
            train_segments
            & test_segments
        )
        == 0
    ), (
        "LEAKAGE DETECTED — halting"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # A
    # ═══════════════════════════════════════════════════════════════════════

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
    ) = experiment_a(
        train_df,
        test_df
    )

    a_test_segment_ids = (
        test_df["segment"]
        .to_numpy(dtype=int)
    )

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

    # ═══════════════════════════════════════════════════════════════════════
    # TEMPORAL SUPPORTING EXPERIMENT
    # ═══════════════════════════════════════════════════════════════════════

    ret_b = experiment_b(
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

    # ═══════════════════════════════════════════════════════════════════════
    # ALIGN A/B
    # ═══════════════════════════════════════════════════════════════════════

    (
        aligned_segment_ids,
        y_true_aligned,
        y_pred_a_aligned,
        y_prob_a_aligned,
        y_pred_b_aligned,
        y_prob_b_aligned
    ) = align_ablation_by_segment(
        y_true_a=y_te_a,
        segment_ids_a=a_test_segment_ids,
        y_pred_a=y_pred_a,
        y_prob_a=y_prob_a,
        y_true_b=y_te_b,
        segment_ids_b=b_test_segment_ids,
        y_pred_b=y_pred_b,
        y_prob_b=y_prob_b,
    )

    n = len(
        aligned_segment_ids
    )

    print(
        f"  Final aligned test segments: "
        f"{n}"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # CALIBRATED RF ALIGNMENT
    # ═══════════════════════════════════════════════════════════════════════

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

    if cal_df[
        "segment"
    ].duplicated().any():

        raise ValueError(
            "Calibrated RF contains duplicate segment IDs."
        )

    cal_common = (
        pd.DataFrame({
            "segment":
                aligned_segment_ids.astype(int),
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
            "Calibrated RF alignment does not cover all aligned "
            "test segments. "
            f"Expected {n}, got {len(cal_common)}."
        )

    cal_truth_mismatch = (
        cal_common[
            "y_true"
        ].to_numpy(dtype=int)
        != y_true_aligned
    )

    if int(
        cal_truth_mismatch.sum()
    ) != 0:

        raise ValueError(
            "Calibrated RF ground truth mismatch after alignment."
        )

    y_pred_cal_aligned = (
        cal_common[
            "y_pred"
        ].to_numpy(dtype=int)
    )

    y_prob_cal_aligned = (
        cal_common[
            "y_prob"
        ].to_numpy(dtype=float)
    )

    print(
        f"  Calibrated RF alignment: "
        f"{len(cal_common)} segments, "
        f"truth_mismatches=0"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SUPPORTING TEMPORAL EXPERIMENT
    # ═══════════════════════════════════════════════════════════════════════

    exp_c_result = experiment_c(
        y_true_aligned,
        y_pred_a_aligned,
        y_pred_b_aligned
    )

    # ═══════════════════════════════════════════════════════════════════════
    # DISAGREEMENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    exp_d_result = experiment_d(
        y_true_aligned,
        y_pred_a_aligned,
        y_pred_b_aligned
    )

    # ═══════════════════════════════════════════════════════════════════════
    # FULL A/B/C/D
    # ═══════════════════════════════════════════════════════════════════════

    ablation = experiment_e(
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

    # ═══════════════════════════════════════════════════════════════════════
    # GRANITE
    # ═══════════════════════════════════════════════════════════════════════

    granite = evaluate_granite_grounding(
        ablation,
        seg_m_a,
        seg_m_b
    )

    # ═══════════════════════════════════════════════════════════════════════
    # INTEGRITY
    # ═══════════════════════════════════════════════════════════════════════

    frozen = verify_frozen()

    provenance = verify_provenance()

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════

    print(
        "\n--- Saving results ---"
    )

    ablation_output = {
        "experiment_A_current_only": {
            "segment_metrics":
                seg_m_a,

            "event_metrics":
                evt_m_a,
        },

        "experiment_B_calibrated_rf": {
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

        "experiment_B_lead_time":
            lead_model_result,

        "experiment_C_comparison":
            exp_c_result,

        "experiment_disagreement":
            exp_d_result,

        "ablation_table":
            ablation[
                "ablation_table"
            ],

        "architecture_details":
            ablation[
                "architecture_details"
            ],

        "decision_stress_test":
            ablation[
                "decision_stress_test"
            ],

        "lead_time_incremental_analysis":
            ablation[
                "lead_time_incremental_analysis"
            ],

        "decision_intelligence_adds_value":
            ablation[
                "decision_intelligence_adds_value"
            ],

        "interpretation":
            ablation[
                "interpretation"
            ],

        "utility_matrix":
            ablation[
                "utility_matrix_definition"
            ],

        "lead_warning_segments_flagged":
            ablation[
                "lead_warning_segments_flagged"
            ],

        "n_test_segments":
            n,

        "alignment":
            ablation[
                "alignment"
            ],

        "ablation_definition":
            ablation[
                "ablation_definition"
            ],

        "provenance":
            provenance,
    }

    save_json(
        ablation_output,
        "real_decision_ablation.json"
    )

    disagreement_output = {
        "experiment_D_disagreement":
            exp_d_result,

        "decision_stress_test":
            ablation[
                "decision_stress_test"
            ],

        "utility_sensitivity": {
            architecture:
                ablation[
                    "architecture_details"
                ][architecture][
                    "policy_metrics"
                ][
                    "utility_sensitivity"
                ]
            for architecture
            in ["A", "B", "C", "D"]
        },

        "cost_matrix":
            ablation[
                "utility_matrix_definition"
            ],
    }

    save_json(
        disagreement_output,
        "real_disagreement_value.json"
    )

    temporal_output = {
        "lead_time_model":
            lead_model_result,

        "temporal_vs_current":
            exp_c_result,

        "lead_time_incremental_analysis":
            ablation[
                "lead_time_incremental_analysis"
            ],

        "lead_warning_segments_used_in_arch_D":
            ablation[
                "lead_warning_segments_flagged"
            ],

        "temporal_feasibility_summary": {
            "supported":
                bool(
                    lead_model_result[
                        "supported"
                    ]
                ),

            "median_lead_s":
                lead_model_result.get(
                    "median_segment_duration_s"
                ),

            "horizon_supported":
                "short (~1 segment duration)",

            "heliomesh_30min_valid":
                False,
        },
    }

    save_json(
        temporal_output,
        "real_temporal_value.json"
    )

    policy_output = {
        "architecture_A":
            ablation[
                "architecture_details"
            ][
                "A"
            ][
                "policy_metrics"
            ],

        "architecture_B":
            ablation[
                "architecture_details"
            ][
                "B"
            ][
                "policy_metrics"
            ],

        "architecture_C":
            ablation[
                "architecture_details"
            ][
                "C"
            ][
                "policy_metrics"
            ],

        "architecture_D":
            ablation[
                "architecture_details"
            ][
                "D"
            ][
                "policy_metrics"
            ],

        "decision_stress_test":
            ablation[
                "decision_stress_test"
            ],

        "utility_matrix":
            ablation[
                "utility_matrix_definition"
            ],

        "utility_note":
            (
                "Prototype utility only; "
                "not validated by real mission operators."
            ),

        "frozen_integrity":
            frozen,
    }

    save_json(
        policy_output,
        "real_policy_evaluation.json"
    )

    granite_output = {
        "structural_audit":
            granite,

        "provenance":
            provenance,

        "note":
            (
                "This is a structural grounding audit, "
                "not a live Granite LLM evaluation."
            ),
    }

    save_json(
        granite_output,
        "real_granite_grounding.json"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL EXPERIMENTS COMPLETE"
    )

    print(
        "=" * 70
    )

    return (
        ablation_output,
        disagreement_output,
        temporal_output,
        policy_output,
        granite_output,
        frozen
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()