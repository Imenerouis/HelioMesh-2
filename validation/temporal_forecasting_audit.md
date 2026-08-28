# Temporal Forecasting Benchmark Audit

## Overall Result

**SCIENTIFICALLY DEFENSIBLE — WITH CONDITIONS**

**NO MODEL CHANGE RECOMMENDED.**

The frozen HelioMesh temporal forecasting benchmark was audited for:

- Data leakage
- Future-information leakage
- Sequence contamination
- Target construction
- Chronological train/validation/test splitting
- Baseline validity

All six areas passed.

## Audit Results

| Audit Area | Result |
|---|---|
| Data leakage | PASS |
| Future-information leakage | PASS |
| Sequence overlap / contamination | PASS |
| Target construction | PASS |
| Chronological split | PASS |
| Baseline validity | PASS |

## Key Findings

### 1. Data Leakage

No data leakage was detected.

The dataset is split chronologically before model fitting. No preprocessing is fitted across train, validation, and test partitions.

### 2. Future-Information Leakage

No future-information leakage was detected.

The forecasting features contain only the six-step input window. Future simulation states are used only to construct the target.

The 30-minute horizon is therefore not present in the input features.

### 3. Sequence Contamination

No sequence overlap or contamination was detected.

Sequences are generated independently from separate random initial conditions. There are no overlapping sliding windows between train, validation, and test partitions.

The synthetic timestamps establish generation order for the chronological split; they do not represent a continuous physical spacecraft trajectory.

### 4. Target Construction

The target is generated from the simulated future state 30 minutes ahead.

The model therefore predicts whether the HelioMesh simulation's deterministic safety rules will classify that future simulated state as CRITICAL.

This is a controlled simulation benchmark, **not real spacecraft failure prediction**.

### 5. Chronological Split

The reported 70/15/15 split is implemented without shuffling:

- Training: 8,400 sequences
- Validation: 1,800 sequences
- Test: 1,800 sequences
- Total: 12,000 sequences

The model is fitted on the training partition and evaluated on the held-out test partition.

### 6. Baseline Validity

Both baselines are evaluated on the same test partition:

- Last-Known-State baseline F1: **0.9015**
- KP-only baseline F1: **0.8977**

The temporal model achieves:

- Macro-F1: **0.9708**
- ROC-AUC: **0.9975**
- Accuracy: **0.9717**
- CRITICAL recall: **0.9819**

The temporal model therefore improves over both baselines.

The strongest shortcut identified was `kp_index_t5`, accounting for approximately 83.3% of feature importance. The KP-only baseline is retained to quantify the additional value provided by the temporal window.

## Important Limitations

This benchmark does **not** establish:

- Real OPS-SAT 30-minute forecasting
- Real spacecraft failure prediction
- Validation of the four-class simulation taxonomy on OPS-SAT-AD
- Cross-mission generalization

OPS-SAT-AD is evaluated separately as a real-spacecraft binary anomaly-detection track.

## Frozen Artifacts

The following artifacts remain unchanged:

- `ml/risk_model.pkl`
- `ml/label_encoder.pkl`
- `ml/forecaster_model.pkl`
- `ml/forecaster_metrics.json`

No model retraining or artifact regeneration was performed as a result of this audit.

## Audit Conclusion

The frozen temporal forecasting benchmark passed all six methodological audit areas.

**NO MODEL CHANGE RECOMMENDED.**