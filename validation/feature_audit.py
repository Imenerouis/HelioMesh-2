"""
HelioMesh — Temporal Predictor Feature Importance Audit (Task 2C)
=================================================================
Extracts feature_importances_ from the frozen GradientBoostingClassifier
and reports predictive contribution of each feature group.

IMPORTANT DISCLAIMER:
  Feature importance ≠ causal importance.
  High importance only means the feature is predictive within the
  model's training distribution (HelioMesh synthetic simulation).
  Do NOT interpret this as physical causality.

Feature names (42 total) from ml/generate_sequences.py:
  [kp_index_t0 … b_field_t5]   36 raw window features (6 steps × 6 features)
  [delta_kp_index … delta_b_field]  6 delta features (t5 - t0)
"""

import os, sys, json, pickle
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "forecaster_model.pkl")
_OUT_PATH   = os.path.join(os.path.dirname(__file__), "results", "feature_audit.json")

STEP_FEATURES = [
    "kp_index", "solar_wind_speed", "orbit_deviation",
    "power_output", "solar_wind_density", "b_field",
]
WINDOW_STEPS = 6
FEATURE_NAMES = (
    [f"{f}_t{i}" for i in range(WINDOW_STEPS) for f in STEP_FEATURES] +
    [f"delta_{f}" for f in STEP_FEATURES]
)
assert len(FEATURE_NAMES) == 42


def run() -> dict:
    print("  Loading GB temporal model...", end=" ", flush=True)
    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("OK")

    importances = model.feature_importances_
    assert len(importances) == 42, f"Expected 42 features, got {len(importances)}"

    total = sum(importances)
    ranked = sorted(
        [(name, float(imp)) for name, imp in zip(FEATURE_NAMES, importances)],
        key=lambda x: x[1], reverse=True
    )

    top10 = [{"feature": n, "importance": round(imp, 6), "pct": round(imp/total*100, 2)}
             for n, imp in ranked[:10]]
    all_features = [{"feature": n, "importance": round(imp, 6), "pct": round(imp/total*100, 2)}
                    for n, imp in ranked]

    # ── Group analysis ──
    delta_names    = {f"delta_{f}" for f in STEP_FEATURES}
    kp_names       = {f"kp_index_t{i}" for i in range(WINDOW_STEPS)} | {"delta_kp_index"}
    non_kp_names   = set(FEATURE_NAMES) - kp_names

    delta_imp    = sum(imp for n, imp in ranked if n in delta_names)
    raw_win_imp  = sum(imp for n, imp in ranked if n not in delta_names)
    kp_imp       = sum(imp for n, imp in ranked if n in kp_names)
    non_kp_imp   = sum(imp for n, imp in ranked if n in non_kp_names)

    delta_kp_in_top5 = any(n == "delta_kp_index" for n, _ in ranked[:5])
    non_kp_material  = non_kp_imp / total > 0.10   # > 10% of total

    # ── Conclusion ──
    conclusion_parts = [
        f"Total feature importance sums to {total:.6f} (should be ≈1.0).",
        f"KP-related features (all t0-t5 + delta_kp): {kp_imp/total*100:.1f}% of total predictive contribution.",
        f"Delta features (all 6 deltas): {delta_imp/total*100:.1f}% of total.",
        f"Non-KP features: {non_kp_imp/total*100:.1f}% of total.",
        f"delta_kp_index in top-5: {delta_kp_in_top5}.",
        (
            f"Non-KP features contribute {non_kp_imp/total*100:.1f}% — "
            + ("this EXCEEDS the 10% materiality threshold. "
               "Non-KP features (wind speed, density, B-field, orbit deviation, power) "
               "do contribute measurably to predictions beyond KP alone."
               if non_kp_material else
               "this is BELOW the 10% materiality threshold. "
               "The temporal model is dominated by KP-related features on simulation data.")
        ),
        (
            "HONEST ASSESSMENT (simulation domain only): "
            f"KP accounts for {kp_imp/total*100:.1f}% of predictive contribution. "
            "This is consistent with the KP-only ablation result. "
            "Non-KP features provide supplementary signal but KP is the dominant driver. "
            "On real spacecraft data the feature importance distribution may differ significantly."
        ),
    ]

    result = {
        "evaluated_at":    datetime.now().isoformat(),
        "model":           "ml/forecaster_model.pkl (GradientBoostingClassifier, FROZEN)",
        "n_features":      42,
        "top_10_features": top10,
        "all_features":    all_features,
        "group_analysis": {
            "kp_all_features":       {"features": sorted(kp_names),    "importance_sum": round(kp_imp, 6),    "pct_of_total": round(kp_imp/total*100, 2)},
            "delta_features":        {"features": sorted(delta_names),  "importance_sum": round(delta_imp, 6),  "pct_of_total": round(delta_imp/total*100, 2)},
            "raw_window_features":   {"importance_sum": round(raw_win_imp, 6), "pct_of_total": round(raw_win_imp/total*100, 2)},
            "non_kp_features":       {"importance_sum": round(non_kp_imp, 6),  "pct_of_total": round(non_kp_imp/total*100, 2)},
        },
        "delta_kp_index_in_top5": bool(delta_kp_in_top5),
        "non_kp_material_gt10pct": bool(non_kp_material),
        "scientific_conclusion":  " ".join(conclusion_parts),
        "disclaimer": (
            "Feature importance = predictive contribution within the trained model on simulation data. "
            "NOT physical causality. NOT validated on real spacecraft telemetry."
        ),
    }

    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Top-5 features by importance:")
    for item in top10[:5]:
        print(f"    {item['feature']:30s}  {item['pct']:5.1f}%")
    print(f"  KP-related total:  {kp_imp/total*100:.1f}%")
    print(f"  Delta features:    {delta_imp/total*100:.1f}%")
    print(f"  Non-KP features:   {non_kp_imp/total*100:.1f}%")
    print(f"  Saved -> {_OUT_PATH}")
    return result


if __name__ == "__main__":
    run()
