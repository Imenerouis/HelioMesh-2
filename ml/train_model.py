"""
HelioMesh — ML Risk Predictor Training
=======================================
Trains a Random Forest classifier to predict HelioMesh operational risk states.

ML Task:
  Given telemetry features, predict the operational state defined by
  HelioMesh safety rules: NOMINAL | STANDBY | SAFE_MODE.

  The model learns the mapping between continuous telemetry values
  and the discrete safety states — it does NOT predict real satellite failures.
  Labels are generated from HelioMesh Operational Safety Rules (see dataset_metadata.json).

Evaluation metrics:
  - Precision, Recall, F1 per class
  - Macro-averaged F1
  - ROC-AUC (One-vs-Rest, macro)
  - 5-fold cross-validation F1
  - Confusion Matrix
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

# ── Config ────────────────────────────────────────────────────
DATASET_PATH  = "ml/dataset.csv"
MODEL_PATH    = "ml/risk_model.pkl"
ENCODER_PATH  = "ml/label_encoder.pkl"
METRICS_PATH  = "ml/training_metrics.json"
FEATURE_PATH  = "ml/feature_importance.json"

FEATURES = [
    "kp_index",
    "sail_angle",
    "solar_wind_speed",
    "solar_wind_density",
    "b_field",
    "drag_factor",
    "orbit_deviation",
    "power_output",
    "thrust_output",
    "geomagnetic_energy",
    "dynamic_pressure",
]

TARGET = "label"


def train():
    # ── Load dataset ──
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Rows: {len(df)}")
    print(f"  Label distribution:\n{df[TARGET].value_counts().to_string()}\n")

    X = df[FEATURES].values
    y_raw = df[TARGET].values

    # ── Encode labels ──
    le = LabelEncoder()
    y = le.fit_transform(y_raw)   # NOMINAL=0, SAFE_MODE=1, STANDBY=2 (alphabetical)
    print(f"Classes: {list(le.classes_)}\n")

    # ── Train / test split (stratified) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Train Random Forest ──
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",    # handles class imbalance
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Training complete.\n")

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_, output_dict=True)

    macro_f1  = f1_score(y_test, y_pred, average="macro")
    roc_auc   = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    cm        = confusion_matrix(y_test, y_pred).tolist()

    # Cross-validation F1 (5-fold)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro", n_jobs=-1)

    # ── Print results ──
    print("=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Macro F1 Score  : {macro_f1:.4f}")
    print(f"ROC-AUC (OvR)   : {roc_auc:.4f}")
    print(f"CV F1 (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  Classes: {list(le.classes_)}")
    for row in cm:
        print(f"  {row}")

    # ── Feature importance ──
    fi = dict(zip(FEATURES, model.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    print(f"\nTop Feature Importances:")
    for feat, imp in list(fi_sorted.items())[:5]:
        print(f"  {feat:25s}: {imp:.4f}")

    # ── Save model + encoder ──
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"\nModel saved   → {MODEL_PATH}")
    print(f"Encoder saved → {ENCODER_PATH}")

    # ── Save metrics JSON ──
    metrics = {
        "trained_at": datetime.now().isoformat(),
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "classes": list(le.classes_),
        "macro_f1":         round(macro_f1, 4),
        "roc_auc_ovr":      round(roc_auc, 4),
        "cv_f1_mean":       round(float(cv_scores.mean()), 4),
        "cv_f1_std":        round(float(cv_scores.std()), 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1_score":  round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in le.classes_
        },
        "confusion_matrix": cm,
        "model_params": model.get_params(),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")

    # ── Save feature importance ──
    with open(FEATURE_PATH, "w") as f:
        json.dump(fi_sorted, f, indent=2)
    print(f"Features saved → {FEATURE_PATH}")

    print("\n✅ Training complete.")
    return metrics


if __name__ == "__main__":
    train()
