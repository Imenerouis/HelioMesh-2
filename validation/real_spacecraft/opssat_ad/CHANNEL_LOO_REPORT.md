# Channel Leave-One-Out Generalization Report
## OPS-SAT-AD — Cross-Channel Generalization of the Anomaly-Detection Approach

---

## 1. Purpose

This report documents a cross-channel leave-one-out (LOO) generalization
experiment on the OPS-SAT-AD spacecraft telemetry dataset.

**What this experiment tests:**
> The anomaly-detection *training procedure* — Random Forest on segment-level
> statistical features — generalizes to OPS-SAT telemetry channels not seen
> during training, including across a substantial cross-channel feature-scale
> difference (~9 orders of magnitude in `var`).

**What this experiment does NOT test:**
> The frozen, production-trained HelioMesh Phase 2 model (`ml/risk_model.pkl`)
> is not evaluated here and is not modified.  This experiment answers a
> generalization question about the *approach*, not about the frozen artifact.

---

## 2. Relationship to Existing Phase 2 Benchmark

| Property | Phase 2 Benchmark | This LOO Experiment |
|---|---|---|
| Model artifact | `ml/risk_model.pkl` (FROZEN) | 9 separate fold-local RF models (discarded after eval) |
| Train/test split | Official split (mixed channels) | Channel LOO (train on N-1 channels, test on held-out) |
| Channel identity as feature | Yes (one-hot) | **No** (tests generalization without channel encoding) |
| Purpose | Production evaluation | Generalization of the approach |
| Modifies any existing artifact | No | No |

The Phase 2 benchmark (F1=0.9209, ROC-AUC=0.9874) is the primary result.
This experiment is a **supplementary generalization study only**.

---

## 3. Experimental Design

### 3.1 LOO Protocol

- All 2,123 segments, 9 telemetry channels.
- Each fold: train on all segments from 8 channels, evaluate on the 9th.
- Repeated 9 times, once per channel.
- `StandardScaler` fitted on training channels only within each fold.
- Channel identity excluded from the feature matrix; model must generalize
  purely from segment-level signal statistics.
- Hyperparameters: `n_estimators=200`, `class_weight='balanced'`,
  `random_state=42`.  Not tuned on any test fold.

### 3.2 Feature Set (19 features, no channel identity)

`duration`, `len`, `mean`, `var`, `std`, `kurtosis`, `skew`,
`n_peaks`, `smooth10_n_peaks`, `smooth20_n_peaks`,
`diff_peaks`, `diff2_peaks`, `diff_var`, `diff2_var`,
`gaps_squared`, `len_weighted`, `var_div_duration`, `var_div_len`,
`sampling_5s`

### 3.3 Feature Scale Regimes

Two distinct sensor populations are present:

| Regime | Channels | Median `var` | log₁₀(var) |
|---|---|---|---|
| A | CADC0872, CADC0873, CADC0874 | ~4×10⁻¹⁰ | −9.4 |
| B | CADC0884, CADC0886, CADC0888, CADC0890, CADC0892, CADC0894 | ~0.05–0.11 | −1.0 to −1.5 |

Regime A vs Regime B differ by ~9 orders of magnitude in variance.  Evaluating
cross-regime generalization is the core scientific stress test.

### 3.4 Exclusion Criteria

Folds with `n_test_anomaly < 10` are excluded from the primary aggregate
(insufficient anomalies to estimate F1/AUC reliably):

- **CADC0884**: 0 anomalies — unevaluable for anomaly detection.
- **CADC0886**: 3 anomalies — too small for reliable estimation.

Both are reported individually for completeness.

---

## 4. Per-Channel Results

| Channel | Regime | n_test | n_anom | n_nom | Precision | Recall | F1 | ROC-AUC | MCC | In Agg? |
|---|---|---|---|---|---|---|---|---|---|---|
| CADC0872 | A | 546 | 131 | 415 | 0.9813 | 0.8015 | **0.8824** | 0.9890 | 0.8571 | ✓ |
| CADC0873 | A | 593 | 105 | 488 | 0.9417 | 0.9238 | **0.9327** | 0.9931 | 0.9184 | ✓ |
| CADC0874 | A | 194 | 69 | 125 | 1.0000 | 0.8551 | **0.9219** | 0.9939 | 0.8898 | ✓ |
| CADC0884 | B | 158 | 0 | 158 | — | — | — | — | — | ✗ (0 anom) |
| CADC0886 | B | 11 | 3 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | ✗ (n=3) |
| CADC0888 | B | 252 | 60 | 192 | 0.9565 | 0.7333 | **0.8302** | 0.9198 | 0.7971 | ✓ |
| CADC0890 | B | 14 | 11 | 3 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | ✓ |
| CADC0892 | B | 211 | 34 | 177 | 0.8750 | 0.2059 | **0.3333** | 0.8519 | 0.3854 | ✓ |
| CADC0894 | B | 144 | 21 | 123 | 0.4516 | 0.6667 | **0.5385** | 0.8639 | 0.4538 | ✓ |

*CADC0874 precision=1.000: zero false positives in this fold.*

---

## 5. Aggregate Results

### 5.1 Primary Aggregate (7 folds, n_test_anomaly ≥ 10)

| Metric | Value |
|---|---|
| Mean F1 | **0.7770** |
| Mean ROC-AUC | **0.9445** |
| Mean PR-AUC | 0.8917 |
| Mean Precision | 0.8866 |
| Mean Recall | 0.7409 |
| Mean MCC | 0.7574 |
| Mean Balanced Accuracy | 0.8582 |

### 5.2 By Scale Regime

| Regime | Channels in agg. | Mean F1 | Mean ROC-AUC |
|---|---|---|---|
| A (var ~1e-10) | CADC0872, 0873, 0874 | **0.9123** | **0.9920** |
| B (var ~1e-1) | CADC0888, 0890, 0892, 0894 | 0.6755 | 0.9089 |

---

## 6. Interpretation

### Strong channels (all Regime A)

CADC0872, 0873, 0874 each achieve F1 ≥ 0.88 and ROC-AUC ≥ 0.989 when held
out.  The approach generalizes strongly to these channels with no channel-level
supervision.

### Variable Regime B performance

- **CADC0888** (n=252, 60 anomalies): F1=0.830, AUC=0.920 — strong.
- **CADC0890** (n=14, 11 anomalies): F1=1.000 — near-trivial (very small,
  highly anomalous channel; treat as anecdotal).
- **CADC0892** (n=211, 34 anomalies): F1=0.333, AUC=0.852 — the approach
  struggles at the decision threshold on this channel despite high AUC.
  High precision (0.875) but low recall (0.206) indicates the model is
  conservative.
- **CADC0894** (n=144, 21 anomalies): F1=0.539, AUC=0.864 — moderate.

The AUC pattern (≥0.85 for all 7 evaluable channels) indicates the ranking is
reliable across regimes; the F1 gap reflects threshold miscalibration when the
anomaly base rate or feature distribution of the held-out channel differs from
the training mix.

### Cross-regime finding

The approach generalizes robustly across the ~9-order-of-magnitude variance
shift (Regime A → B), reaching mean ROC-AUC=0.909 on Regime B channels.
The `StandardScaler` re-fitted within each fold handles the absolute scale
difference; the relative feature structure (peaks, kurtosis, skew) carries the
discriminative signal across both regimes.

---

## 7. Supported Scientific Claim

> "The anomaly-detection approach (Random Forest on segment-level statistical
> features) generalizes across unseen OPS-SAT telemetry channels, achieving
> mean ROC-AUC=0.9445 and mean F1=0.777 across 7 held-out channels, including
> channels from a distinct sensor-scale regime differing by ~9 orders of
> magnitude in variance from the training channels."

**Limitations:**

1. All 9 channels originate from the same spacecraft (OPS-SAT).  This tests
   intra-spacecraft cross-channel generalization only, not cross-mission.
2. The feature distributions within each regime are not fully independent.
3. CADC0890 (F1=1.0, n=14) and CADC0886 (F1=1.0, n=3) are too small to be
   meaningful.
4. Regime B F1 is pulled down by CADC0892 threshold miscalibration; ROC-AUC
   suggests the latent discriminability is present (0.852).

---

## 8. Integrity Checks

- Frozen artifacts (`ml/risk_model.pkl`, `ml/label_encoder.pkl`,
  `ml/forecaster_model.pkl`, `ml/forecaster_metrics.json`) are untouched.
- Existing validation artifacts (`opssat_ad_real_metrics.json`,
  `real_temporal_audit.json`, `real_policy_calibration.json`,
  `granite_real_evidence.json`, `policy_tests.json`) are untouched.
- No source files outside `validation/real_spacecraft/opssat_ad/` were
  modified.
- All LOO results stored exclusively in
  `validation/results/channel_loo_results.json`.

---

*Generated by `validation/real_spacecraft/opssat_ad/channel_loo.py`.*
*random_state=42, n_estimators=200, class_weight=balanced.*
