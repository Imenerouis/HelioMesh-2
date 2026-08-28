"""
HelioMesh — Temporal Predictor: fast evaluation script.
Trains with n_estimators=100 for speed, saves model + metrics.
For production use train_forecaster.py (n_estimators=300).
"""
import json, pickle, sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, average_precision_score, accuracy_score
)

TARGET    = "label"
DROP_COLS = ["timestamp", "regime", "label"]
STEP_FEATURES = ["kp_index","solar_wind_speed","orbit_deviation",
                 "power_output","solar_wind_density","b_field"]

def snapshot_label(kp, orb, pwr):
    if kp > 6.0 or orb > 1.5 or pwr < 10.0: return 1
    if kp > 4.0 or orb > 0.8 or pwr < 40.0: return 1
    return 0

def dist(arr):
    u, c = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}

print("Loading dataset...", flush=True)
df = pd.read_csv("ml/sequences.csv")
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].values
y = df[TARGET].values
n = len(X)

train_end = int(n * 0.70)
val_end   = int(n * 0.85)
X_train, y_train = X[:train_end], y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:],   y[val_end:]

print(f"Split  Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}", flush=True)
print(f"       Train dist={dist(y_train)}", flush=True)
print(f"       Val   dist={dist(y_val)}", flush=True)
print(f"       Test  dist={dist(y_test)}", flush=True)

print("Training (n_estimators=100 for speed)...", flush=True)
model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.08, max_depth=5,
    min_samples_leaf=8, subsample=0.85, random_state=42
)
model.fit(X_train, y_train)
print("Done.", flush=True)

# Validation
val_pred = model.predict(X_val)
val_f1   = f1_score(y_val, val_pred, average="macro")
val_auc  = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"VAL  Macro F1={val_f1:.4f}  ROC-AUC={val_auc:.4f}", flush=True)

# Test
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
roc_auc  = roc_auc_score(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)
cm       = confusion_matrix(y_test, y_pred).tolist()
rep      = classification_report(y_test, y_pred,
           target_names=["NOMINAL_AHEAD","CRITICAL_AHEAD"], output_dict=True)

# Baseline 1: last-known-state — full snapshot rule on t5 (kp + orbit + power)
LAST = 5 * len(STEP_FEATURES)  # index 30 = start of t5 block
base_pred = np.array([snapshot_label(row[LAST], row[LAST+2], row[LAST+3])
                      for row in X_test])
base_f1   = f1_score(y_test, base_pred, average="macro")
base_acc  = accuracy_score(y_test, base_pred)
base_auc  = roc_auc_score(y_test, base_pred.astype(float))
base_rep  = classification_report(y_test, base_pred,
            target_names=["NOMINAL_AHEAD","CRITICAL_AHEAD"], output_dict=True)
base_cm   = confusion_matrix(y_test, base_pred).tolist()

# Baseline 2: KP-only threshold — uses only kp_index_t5
# Audit found kp_index_t5 alone carries 83% of model importance.
# Best single threshold from audit (full dataset): kp_t5 > 3.5 gave acc=0.936
# We use the HelioMesh rule threshold of 4.0 (matches STANDBY rule exactly)
# so the comparison is apples-to-apples with the label definition.
KP_T5_IDX = LAST + 0   # kp_index is first feature in each step block
kp_only_pred = (X_test[:, KP_T5_IDX] > 4.0).astype(int)
kp_f1   = f1_score(y_test, kp_only_pred, average="macro")
kp_acc  = accuracy_score(y_test, kp_only_pred)
kp_auc  = roc_auc_score(y_test, kp_only_pred.astype(float))
kp_rep  = classification_report(y_test, kp_only_pred,
          target_names=["NOMINAL_AHEAD","CRITICAL_AHEAD"], output_dict=True)
kp_cm   = confusion_matrix(y_test, kp_only_pred).tolist()

print("\n" + "="*62, flush=True)
print("  TEST SET  (final 15% -- never seen during training)", flush=True)
print("="*62, flush=True)
print(classification_report(y_test, y_pred,
      target_names=["NOMINAL_AHEAD","CRITICAL_AHEAD"]), flush=True)
print(f"Accuracy   : {acc:.4f}", flush=True)
print(f"Macro F1   : {macro_f1:.4f}", flush=True)
print(f"ROC-AUC    : {roc_auc:.4f}", flush=True)
print(f"Avg Prec   : {avg_prec:.4f}", flush=True)
print(f"Conf Matrix (rows=actual, cols=predicted):", flush=True)
print(f"  Classes  : [NOMINAL_AHEAD, CRITICAL_AHEAD]", flush=True)
for row in cm:
    print(f"  {row}", flush=True)

print("\n" + "="*62, flush=True)
print("  BASELINE 1  (last-known-state full rule on t5)", flush=True)
print("="*62, flush=True)
print(classification_report(y_test, base_pred,
      target_names=["NOMINAL_AHEAD","CRITICAL_AHEAD"]), flush=True)
print(f"Base Acc   : {base_acc:.4f}", flush=True)
print(f"Base F1    : {base_f1:.4f}", flush=True)
print(f"Base AUC   : {base_auc:.4f}", flush=True)
for row in base_cm:
    print(f"  {row}", flush=True)

print("\n" + "="*62, flush=True)
print("  BASELINE 2  (KP-only: kp_index_t5 > 4.0)", flush=True)
print("  Strongest single-feature shortcut found in audit", flush=True)
print("="*62, flush=True)
print(classification_report(y_test, kp_only_pred,
      target_names=["NOMINAL_AHEAD","CRITICAL_AHEAD"]), flush=True)
print(f"KP Acc     : {kp_acc:.4f}", flush=True)
print(f"KP F1      : {kp_f1:.4f}", flush=True)
print(f"KP AUC     : {kp_auc:.4f}", flush=True)
for row in kp_cm:
    print(f"  {row}", flush=True)

print("\n" + "="*62, flush=True)
print("  THREE-WAY COMPARISON  (test set)", flush=True)
print("="*62, flush=True)
f1g_vs_base = macro_f1 - base_f1
f1g_vs_kp   = macro_f1 - kp_f1
aucg_vs_base = roc_auc - base_auc
aucg_vs_kp   = roc_auc - kp_auc
print(f"  {'Method':<35} {'Acc':>7}  {'Macro F1':>9}  {'CRITICAL recall':>16}", flush=True)
print(f"  {'-'*70}", flush=True)
crit_rec_base = base_rep['CRITICAL_AHEAD']['recall']
crit_rec_kp   = kp_rep['CRITICAL_AHEAD']['recall']
crit_rec_mod  = rep['CRITICAL_AHEAD']['recall']
print(f"  {'Last-known-state baseline':<35} {base_acc:>7.4f}  {base_f1:>9.4f}  {crit_rec_base:>16.4f}", flush=True)
print(f"  {'KP-only baseline (kp_t5 > 4.0)':<35} {kp_acc:>7.4f}  {kp_f1:>9.4f}  {crit_rec_kp:>16.4f}", flush=True)
print(f"  {'Temporal Predictor (GB)':<35} {acc:>7.4f}  {macro_f1:>9.4f}  {crit_rec_mod:>16.4f}", flush=True)
print(f"\n  Model vs last-known-state : F1 {f1g_vs_base:+.4f}   AUC {aucg_vs_base:+.4f}", flush=True)
print(f"  Model vs KP-only          : F1 {f1g_vs_kp:+.4f}   AUC {aucg_vs_kp:+.4f}", flush=True)

# Save
with open("ml/forecaster_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved: ml/forecaster_model.pkl", flush=True)

metrics = {
    "trained_at": datetime.now().isoformat(),
    "model_type": "GradientBoostingClassifier",
    "n_estimators_note": "100 (fast eval); use train_forecaster.py for n=300",
    "task": "Supervised classification: 30-min look-back -> state at t+30min",
    "honest_description": (
        "HelioMesh predicts the satellite future operational safety state "
        "from simulated telemetry sequences. NOT a prediction of real satellite failures."
    ),
    "timing": {
        "look_back_minutes": 30, "look_ahead_minutes": 30,
        "total_span_minutes": 60, "step_minutes": 5,
        "window_steps": 6, "horizon_steps": 6,
        "timing_note": (
            "Input: t0..t25 (6 steps x 5min). "
            "Horizon: t25..t55 (not in features). "
            "Target: state at t55 = 30min beyond last input snapshot."
        ),
    },
    "split": {
        "strategy": "chronological 70/15/15",
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "train_label_dist": dist(y_train),
        "val_label_dist":   dist(y_val),
        "test_label_dist":  dist(y_test),
    },
    "sequence_overlap": "none",
    "sequence_overlap_note": (
        "Each sequence has independent random initial conditions. "
        "No sliding-window overlap between sequences."
    ),
    "val_metrics": {"macro_f1": round(val_f1, 4), "roc_auc": round(val_auc, 4)},
    "test_metrics": {
        "accuracy":          round(acc, 4),
        "macro_f1":          round(macro_f1, 4),
        "roc_auc":           round(roc_auc, 4),
        "average_precision": round(avg_prec, 4),
        "per_class": {
            "NOMINAL_AHEAD": {
                "precision": round(rep["NOMINAL_AHEAD"]["precision"], 4),
                "recall":    round(rep["NOMINAL_AHEAD"]["recall"],    4),
                "f1":        round(rep["NOMINAL_AHEAD"]["f1-score"],  4),
                "support":   int(rep["NOMINAL_AHEAD"]["support"]),
            },
            "CRITICAL_AHEAD": {
                "precision": round(rep["CRITICAL_AHEAD"]["precision"], 4),
                "recall":    round(rep["CRITICAL_AHEAD"]["recall"],    4),
                "f1":        round(rep["CRITICAL_AHEAD"]["f1-score"],  4),
                "support":   int(rep["CRITICAL_AHEAD"]["support"]),
            },
        },
        "confusion_matrix": cm,
        "confusion_matrix_note": "rows = actual label, cols = predicted label, order = [NOMINAL_AHEAD, CRITICAL_AHEAD]",
    },
    "baseline_last_known_state": {
        "name": "last-known-state (full snapshot rule on t5)",
        "description": (
            "Apply HelioMesh safety rules to the LAST window snapshot (t5). "
            "Zero temporal learning — represents the trivial current-state predictor."
        ),
        "accuracy":  round(base_acc, 4),
        "macro_f1":  round(base_f1, 4),
        "roc_auc":   round(base_auc, 4),
        "critical_recall": round(base_rep["CRITICAL_AHEAD"]["recall"], 4),
        "per_class": {
            "NOMINAL_AHEAD":  {"precision": round(base_rep["NOMINAL_AHEAD"]["precision"],  4), "recall": round(base_rep["NOMINAL_AHEAD"]["recall"],  4), "f1": round(base_rep["NOMINAL_AHEAD"]["f1-score"],  4)},
            "CRITICAL_AHEAD": {"precision": round(base_rep["CRITICAL_AHEAD"]["precision"], 4), "recall": round(base_rep["CRITICAL_AHEAD"]["recall"], 4), "f1": round(base_rep["CRITICAL_AHEAD"]["f1-score"], 4)},
        },
        "confusion_matrix": base_cm,
    },
    "baseline_kp_only": {
        "name": "KP-only threshold (kp_index_t5 > 4.0)",
        "description": (
            "Strongest single-feature shortcut identified in audit. "
            "Uses only kp_index at last window step with the STANDBY rule threshold. "
            "Tests whether the temporal model adds value beyond the strongest shortcut."
        ),
        "threshold": "kp_index_t5 > 4.0",
        "accuracy":  round(kp_acc, 4),
        "macro_f1":  round(kp_f1,  4),
        "roc_auc":   round(kp_auc, 4),
        "critical_recall": round(kp_rep["CRITICAL_AHEAD"]["recall"], 4),
        "per_class": {
            "NOMINAL_AHEAD":  {"precision": round(kp_rep["NOMINAL_AHEAD"]["precision"],  4), "recall": round(kp_rep["NOMINAL_AHEAD"]["recall"],  4), "f1": round(kp_rep["NOMINAL_AHEAD"]["f1-score"],  4)},
            "CRITICAL_AHEAD": {"precision": round(kp_rep["CRITICAL_AHEAD"]["precision"], 4), "recall": round(kp_rep["CRITICAL_AHEAD"]["recall"], 4), "f1": round(kp_rep["CRITICAL_AHEAD"]["f1-score"], 4)},
        },
        "confusion_matrix": kp_cm,
    },
    "model_vs_baselines": {
        "vs_last_known_state": {
            "accuracy_gain": round(acc - base_acc, 4),
            "macro_f1_gain": round(macro_f1 - base_f1, 4),
            "roc_auc_gain":  round(roc_auc - base_auc, 4),
            "critical_recall_gain": round(crit_rec_mod - crit_rec_base, 4),
        },
        "vs_kp_only": {
            "accuracy_gain": round(acc - kp_acc, 4),
            "macro_f1_gain": round(macro_f1 - kp_f1, 4),
            "roc_auc_gain":  round(roc_auc - kp_auc, 4),
            "critical_recall_gain": round(crit_rec_mod - crit_rec_kp, 4),
        },
        "interpretation": (
            "The temporal model must beat BOTH baselines to demonstrate genuine value. "
            "KP-only is the strongest shortcut found in the audit (83% feature importance). "
            "Gains vs KP-only show what the full temporal window adds beyond that shortcut."
        ),
    },
    "disclaimer": (
        "Does NOT predict real satellite failures. "
        "Labels derived from HelioMesh prototype safety rules, not NASA/NOAA standards."
    ),
}
with open("ml/forecaster_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved: ml/forecaster_metrics.json", flush=True)
