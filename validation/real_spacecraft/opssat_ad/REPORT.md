# OPS-SAT-AD Real-Spacecraft Benchmark — Phase 2 Report

**HelioMesh | Real-Data Validation**  
**Status: B — REAL TELEMETRY AVAILABLE BUT EVENT/TEMPORAL TASK LIMITED**

---

## DATASET VERIFIED

Both canonical files were inspected and verified from disk. All counts below
are measured directly from the files — not taken from documentation.

---

## EXACT FILE STATISTICS

### File Hashes (provenance anchor)

| File | Size (bytes) | SHA-256 |
|------|-------------|---------|
| `segments.csv` | 18,987,091 | `d5201e9e751eb2a53a0ff7c11567dc4239f594ea4b479b2aa66fe67ddcbcb9ba` |
| `dataset.csv`  | 507,550 | `b524177da6f516d5c9f63c7acbc385341f0ad42046ef20c1bed2d25e51b98f02` |

### segments.csv

| Property | Verified Value |
|----------|----------------|
| Rows | **303,493** |
| Columns | **8** |
| Column names | `channel`, `timestamp`, `value`, `label`, `sampling`, `anomaly`, `segment`, `train` |
| Null values | **None** |
| Duplicate rows | **None** |
| Unique channels | **9** (`CADC0872`, `CADC0873`, `CADC0874`, `CADC0884`, `CADC0886`, `CADC0888`, `CADC0890`, `CADC0892`, `CADC0894`) |
| Unique segments | **2,123** |
| Timestamp range | `2022-01-04 20:00:50 UTC` → `2022-06-02 15:10:42 UTC` |
| Span | 148 days 19 h 9 min |
| Sampling cadence | 1 s (237,165 samples) or 5 s (66,328 samples) |
| Train samples | **225,178** |
| Test samples | **78,315** |
| Anomaly samples (`anomaly=1`) | **100,264** |
| Normal samples (`anomaly=0`) | **203,229** |
| Samples per segment | min=8, max=1,040, mean=143.0, median=70 |
| Channels per segment | always exactly **1** |
| Timestamps monotonic within segment | **True** (all 2,123 segments) |

**Important note on `label` column:**  
The `label` column contains the string `"anomaly"` for every single row. This is
a dataset-source tag, not the target variable. The ground-truth classification
target is the `anomaly` column (integer 0 or 1).

### Anomaly distribution by channel (segments.csv)

| Channel | Anomalous samples | Total samples | Anomaly rate |
|---------|-------------------|---------------|--------------|
| CADC0872 | 24,366 | 66,819 | 36.5% |
| CADC0873 | 22,449 | 68,289 | 32.9% |
| CADC0874 | 35,444 | 58,719 | 60.4% |
| CADC0884 | 0 | 11,810 | 0.0% |
| CADC0886 | 87 | 373 | 23.3% |
| CADC0888 | 2,754 | 11,640 | 23.7% |
| CADC0890 | 402 | 549 | 73.2% |
| CADC0892 | 7,234 | 49,782 | 14.5% |
| CADC0894 | 7,528 | 35,512 | 21.2% |

### dataset.csv

| Property | Verified Value |
|----------|----------------|
| Rows | **2,123** (one row per segment) |
| Columns | **23** |
| Column names | `segment`, `anomaly`, `train`, `channel`, `sampling`, `duration`, `len`, `mean`, `var`, `std`, `kurtosis`, `skew`, `n_peaks`, `smooth10_n_peaks`, `smooth20_n_peaks`, `diff_peaks`, `diff2_peaks`, `diff_var`, `diff2_var`, `gaps_squared`, `len_weighted`, `var_div_duration`, `var_div_len` |
| Null values | **None** |
| Duplicate rows | **None** |
| Anomaly=1 segments | **434** |
| Anomaly=0 segments | **1,689** |
| Train segments | **1,594** |
| Test segments | **529** |
| Train anomaly=1 | **321** |
| Train anomaly=0 | **1,273** |
| Test anomaly=1 | **113** |
| Test anomaly=0 | **416** |
| Unique segment IDs matching segments.csv | **2,123** (perfect 1-to-1) |

**Relationship between files:**  
`dataset.csv` is a summary feature table — each row corresponds to exactly one
segment in `segments.csv`. Features (`mean`, `var`, `n_peaks`, etc.) are
pre-computed statistical descriptors of the raw telemetry within that segment.
The `anomaly` and `train` columns are identical between the two files for each
segment ID.

---

## TRAIN/TEST INTEGRITY

| Property | Result |
|----------|--------|
| Official split column | `train` (1=train, 0=test) |
| Segment leakage between train and test | **0 segments** |
| Train set size | 1,594 segments |
| Test set size | 529 segments |
| Train anomaly rate | 20.14% |
| Test anomaly rate | 21.36% |
| Class distribution stable across splits | **Yes** (within 1.2 pp) |

**No custom or random splitting was performed. The official `train` column was
used as-is throughout all experiments.**

For model development, only the TRAIN partition was used for fitting and the
TEST partition was held out untouched until final evaluation.

---

## REAL-SPACECRAFT BASELINE RESULTS

Evaluated on the **official test partition only (529 segments, 113 anomalous)**.

### Features used
18 engineered features from `dataset.csv` plus 8 channel dummy variables and 1
sampling flag = 27 total features. All features pre-computed from real
ESA OPS-SAT telemetry by the dataset authors. StandardScaler fitted on TRAIN
only.

### Model 1 — Majority Baseline (always predict normal)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7864 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| MCC | 0.0000 |
| ROC-AUC | N/A |
| PR-AUC | N/A |
| TP / TN / FP / FN | 0 / 416 / 0 / 113 |

### Model 2 — Random Forest (200 trees, class_weight='balanced')

| Metric | Value |
|--------|-------|
| Accuracy | **0.9679** |
| Balanced Accuracy | **0.9344** |
| Precision | **0.9706** |
| Recall | **0.8761** |
| F1 | **0.9209** |
| MCC | **0.9027** |
| ROC-AUC | **0.9874** |
| PR-AUC | **0.9677** |
| FPR | 0.0072 |
| FNR | 0.1239 |
| TP / TN / FP / FN | **99 / 413 / 3 / 14** |

### Model 3 — Isolation Forest (200 trees, contamination=0.2014)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7051 |
| Balanced Accuracy | 0.5418 |
| Precision | 0.2871 |
| Recall | 0.2566 |
| F1 | 0.2710 |
| MCC | 0.0871 |
| ROC-AUC | 0.6420 |
| PR-AUC | 0.3823 |
| TP / TN / FP / FN | 29 / 344 / 72 / 84 |

### Top-10 Random Forest Feature Importances

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | n_peaks | 0.2325 |
| 2 | diff2_peaks | 0.1002 |
| 3 | smooth10_n_peaks | 0.0758 |
| 4 | kurtosis | 0.0727 |
| 5 | diff_peaks | 0.0645 |
| 6 | len | 0.0509 |
| 7 | duration | 0.0473 |
| 8 | gaps_squared | 0.0422 |
| 9 | std | 0.0401 |
| 10 | skew | 0.0324 |

Peak-count features dominate, reflecting that anomalous telemetry segments
exhibit distinctly different oscillation patterns.

---

## EVENT-LEVEL RESULTS

**Methodology note:** In OPS-SAT-AD each segment is a single annotated event.
Segment boundaries ARE event boundaries by dataset design — no temporal boundary
reconstruction is needed or scientifically valid. Event-level evaluation is
therefore equivalent to segment-level evaluation.

| Metric | Value |
|--------|-------|
| Total test events | 529 |
| True anomaly events | 113 |
| True normal events | 416 |
| Detected events (TP) | **99** |
| Missed events (FN) | **14** |
| False alerts (FP) | **3** |
| Correct nominals (TN) | 413 |
| Event Precision | **0.9706** |
| Event Recall | **0.8761** |
| Event F1 | **0.9209** |

### Per-channel event breakdown (test partition)

| Channel | Total | Anomaly | Detected | Missed | False Alerts |
|---------|-------|---------|----------|--------|--------------|
| CADC0872 | 132 | 32 | 29 | 3 | 0 |
| CADC0873 | 153 | 31 | 29 | 2 | 1 |
| CADC0874 | 52 | 23 | 18 | 5 | 0 |
| CADC0884 | 36 | 0 | 0 | 0 | 0 |
| CADC0886 | 4 | 1 | 1 | 0 | 0 |
| CADC0888 | 64 | 12 | 11 | 1 | 1 |
| CADC0890 | 2 | 2 | 2 | 0 | 0 |
| CADC0892 | 53 | 7 | 6 | 1 | 0 |
| CADC0894 | 33 | 5 | 3 | 2 | 1 |

CADC0884 has zero anomalies in both train and test — it represents a
channel that was nominal throughout the mission window.

---

## TEMPORAL FEASIBILITY

| Property | Value |
|----------|-------|
| Task assessment | **LIMITED TEMPORAL TASK POSSIBLE** |
| Primary sampling cadence | 1 s (78.2%) and 5 s (21.8%) |
| Segment duration: median | 225 s (~3.75 min) |
| Segment duration: p5/p95 | 75 s / 605 s |
| Inter-segment gap: median | 5 s |
| Inter-segment gap: p95 | 2,185 s |
| Nominal→Anomaly transitions | **275 found** |
| Lead time (nominal before anomaly): median | **255 s (~4.25 min)** |
| Lead time: min/max | 45 s / 10,655,449 s |
| Supports future-anomaly prediction | **Conditionally** (short horizon only) |
| HelioMesh 30-min prediction target valid | **No** |

**Assessment rationale:** There are 275 nominal→anomaly segment transitions
in the dataset. The median lead time (time from start of a nominal segment to
the start of the following anomaly segment) is 255 s (~4.25 min). This means
a short-horizon early-warning system (≤ 4 min) could potentially be evaluated
on this data. The HelioMesh 30-minute prediction horizon is *not* validated
here — that horizon is grounded in the separate simulation benchmark.

**What cannot be done with OPS-SAT-AD:**
- Validate 30-minute ahead anomaly forecasting
- Evaluate continuous real-time anomaly onset prediction
- Reconstruct spacecraft mode transitions (no mode labels)
- Produce NOMINAL/STANDBY/SAFE_MODE/CRITICAL_AHEAD state predictions

---

## PROVENANCE

| Property | Value |
|----------|-------|
| Dataset name | OPSSAT-AD — Anomaly Detection Dataset for Satellite Telemetry |
| Zenodo DOI | [10.5281/zenodo.12588359](https://doi.org/10.5281/zenodo.12588359) |
| Zenodo record | 12588359 |
| Source repository | [kplabs-pl/OPS-SAT-AD](https://github.com/kplabs-pl/OPS-SAT-AD) |
| License | **MIT** |
| Paper | Ruszczak B., Kotowski K., Evans D., Nalepa J. (2025). *The OPS-SAT benchmark for detecting anomalies in satellite telemetry.* **Scientific Data**, Springer Nature. [doi:10.1038/s41597-025-05035-3](https://doi.org/10.1038/s41597-025-05035-3) |
| Mission | ESA OPS-SAT — first ESA flying nanosatellite laboratory |
| Segment count | 2,123 (55 channels observed — 9 selected for benchmark) |
| Temporal coverage | 2022-01-04 to 2022-06-02 |
| Dataset version | Zenodo 12588359 (v1 release, 2024) |
| segments.csv SHA-256 | `d5201e9e751eb2a53a0ff7c11567dc4239f594ea4b479b2aa66fe67ddcbcb9ba` |
| dataset.csv SHA-256 | `b524177da6f516d5c9f63c7acbc385341f0ad42046ef20c1bed2d25e51b98f02` |
| Local acquisition | Manually downloaded from Zenodo record and placed in `data/real_spacecraft/opssat_ad/` |

---

## FROZEN ML INTEGRITY

All four HelioMesh frozen simulation artifacts verified **unchanged** after Phase 2.

| Artifact | SHA-256 (full) | Status |
|----------|----------------|--------|
| `ml/risk_model.pkl` | `109a8a43578926a2a4168b2eba404e09e1aa0ec81f459d91380d5cc49a27dadb` | ✓ UNCHANGED |
| `ml/label_encoder.pkl` | `20b39a8a67cd98b5b6e6b464eb2842048337fec8ed0cbfe11bd02346c16c3c02` | ✓ UNCHANGED |
| `ml/forecaster_model.pkl` | `89864a1706a312c21a5fb86b67d925f549fd00d30e8bdda070d0ca3f325c23f1` | ✓ UNCHANGED |
| `ml/forecaster_metrics.json` | `533b3fc5b48e4c095f9356e4dbd0311ab2cc9212a2b7efb7fdefda0592ad613a` | ✓ UNCHANGED |

No retraining, no overwriting, no modification of any simulation benchmark asset.

---

## WHAT THIS PROVES

1. **Real ESA spacecraft telemetry is now part of the HelioMesh validation suite.**
   The OPS-SAT-AD dataset provides 303,493 timestamped samples from 9 telemetry
   channels of a real operating nanosatellite.

2. **Anomaly detection on real spacecraft data is tractable.**
   The Random Forest baseline achieves F1=0.9209, ROC-AUC=0.9874, PR-AUC=0.9677
   on the official held-out test set — a strong, honest result on unseen real data.

3. **Event-level performance is high.**
   99 of 113 anomalous events detected (87.6% recall) with only 3 false alerts
   (0.72% FPR) across the 529-segment test partition.

4. **Feature engineering is validated.**
   The 18 pre-computed segment statistics discriminate real anomalies effectively;
   peak-count and kurtosis features are the most informative.

5. **The official train/test split has zero leakage.**
   1,594 train segments and 529 test segments are fully disjoint.

6. **All frozen HelioMesh simulation artifacts remain intact.**
   The simulation benchmark (accuracy=0.9717, macro_F1=0.9708, ROC-AUC=0.9975)
   is unchanged; this phase adds a separate real-data layer of evidence.

---

## WHAT THIS DOES NOT PROVE

1. **This does not validate HelioMesh's 30-minute ahead prediction.**
   OPS-SAT-AD is a segment-level classification task; the median lead time is
   ~255 s. The 30-minute horizon is grounded solely in the simulation benchmark.

2. **This does not validate HelioMesh's NOMINAL/STANDBY/SAFE_MODE/CRITICAL_AHEAD
   risk classification scheme.**
   OPS-SAT-AD uses binary anomaly labels only (0=normal, 1=anomaly). The
   four-class risk taxonomy is specific to HelioMesh's simulation training data.

3. **This does not validate the Granite AI grounding pipeline.**
   The grounding score (0.9444) and contradiction rate (0.00) are properties
   of the simulation-based decision chain, not evaluated against OPS-SAT labels.

4. **The Isolation Forest baseline did not perform well (F1=0.2710).**
   Unsupervised anomaly detection is significantly harder on this dataset without
   access to labeled training data — the RF's supervised advantage is large.

5. **This is a segment-classification benchmark, not a streaming-telemetry benchmark.**
   The dataset is pre-segmented; a deployed system would also need to solve
   the segmentation problem in real time.

---

## FINAL STATUS

**B — REAL TELEMETRY AVAILABLE BUT EVENT/TEMPORAL TASK LIMITED**

- Real ESA spacecraft telemetry: ✓ confirmed
- Binary anomaly labels from satellite operators: ✓ confirmed  
- Official train/test split with zero leakage: ✓ confirmed
- Strong supervised performance (RF F1=0.9209): ✓ confirmed
- Event-level detection (F1=0.9209): ✓ confirmed
- 30-minute prediction horizon validation: ✗ not supported by this dataset
- Four-class risk classification validation: ✗ not supported by this dataset
- Streaming/real-time evaluation: ✗ pre-segmented data only

The next phase (Phase 3) should focus on connecting this real-data evidence
to the HelioMesh decision-intelligence pipeline — mapping the segment anomaly
signal into the risk decision chain and demonstrating end-to-end consistency
between the simulation benchmark and the real-spacecraft baseline.

---

*Generated by HelioMesh validation pipeline — all metrics produced by code from
the raw data files. No metrics were manually imputed or adjusted.*
