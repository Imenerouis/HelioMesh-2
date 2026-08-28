# Simulation Benchmark — Scientific Presentation Note

## What the Simulation Benchmark Is

The HelioMesh temporal forecasting benchmark is a **supervised classification task** defined entirely within the HelioMesh simulation environment.

### Task Definition

> Given 6 simulated telemetry snapshots taken at 5-minute intervals (a 30-minute look-back window), predict whether the simulated satellite will be in a critical operational state — STANDBY or SAFE_MODE — 30 minutes after the last observed snapshot.

This is supervised classification over a fixed-length feature window. It is not classical ARIMA or RNN time-series forecasting.

### What the Model Predicts

The Gradient Boosting model predicts **a HelioMesh-defined simulation label**, derived by applying the same deterministic safety rules to a future simulated state:

```
if kp > 6.0 OR orbit_dev > 1.5 OR power < 10.0  → SAFE_MODE  → CRITICAL_AHEAD (1)
if kp > 4.0 OR orbit_dev > 0.8 OR power < 40.0  → STANDBY    → CRITICAL_AHEAD (1)
otherwise                                         → NOMINAL    → NOMINAL_AHEAD  (0)
```

The target is **not** a ground-truth physical event. It is the simulation's own rule applied to a future simulated state.

### What the Model Does NOT Predict

- Real spacecraft failures
- Real satellite anomalies
- Real geomagnetic storm impact on an actual spacecraft
- Any event validated against real mission telemetry

### Honest Statement of Purpose

> "HelioMesh predicts the satellite's future operational safety state from a sequence of simulated telemetry observations. This is NOT a prediction of real satellite failures."

---

## Benchmark Results

| Metric | Value |
|--------|-------|
| Temporal GB macro-F1 | **0.9708** |
| ROC-AUC | **0.9975** |
| Accuracy | 0.9717 |
| CRITICAL class recall | **0.9819** |
| Last-known-state baseline F1 | 0.9015 |
| KP-only baseline F1 | 0.8977 |
| Sequences | 12,000 |
| Split | 70/15/15 chronological |

---

## KP Dominance — Disclosed and Baselined

`kp_index_t5` carries **83.3% of the model's total feature importance**. This is the strongest single-feature shortcut in the dataset.

Two baselines are provided to quantify what the full temporal window adds beyond this shortcut:

| Model | Macro-F1 | CRITICAL recall |
|-------|----------|-----------------|
| KP-only threshold (kp_t5 > 4.0) | 0.8977 | 0.8382 |
| Last-known-state (full snapshot rules on t5) | 0.9015 | 0.8449 |
| **Temporal GB (30-min window)** | **0.9708** | **0.9819** |

**Gain vs KP-only: +7.3pp F1, +14.4pp CRITICAL recall**

This gain demonstrates **additional predictive value within the controlled simulation benchmark**. The temporal window's main contribution is detecting rising KP trends not yet above threshold at the last observed step — storm-approach scenarios that the KP-only baseline misses.

The improvement must be understood as simulation-internal. It cannot be extrapolated to real spacecraft performance without real data.

---

## Methodology Audit Summary

All six methodological audit areas pass:

| Area | Result |
|------|--------|
| Data leakage | PASS — no cross-partition preprocessing |
| Future-information leakage | PASS — horizon gap never in features |
| Sequence overlap/contamination | PASS — independent random initial conditions |
| Target construction | PASS — future label from simulation, not window |
| Chronological split (70/15/15) | PASS — strict ordering, no shuffle |
| Baseline validity | PASS — both baselines on identical test partition |

**Audit conclusion: NO MODEL CHANGE RECOMMENDED.**

---

## Limitations

1. All labels are HelioMesh prototype simulation states — not NASA/NOAA engineering standards.
2. KP dominance (83.3%) is a structural property of the simulation — not hidden, baselined.
3. The 30-minute horizon is defined within the simulation and is NOT validated on real OPS-SAT-AD telemetry.
4. Performance may differ on real spacecraft telemetry; the simulation does not reproduce all real-world noise.
5. `power_output` has zero feature importance (sail angle fixed per sequence — a known simulation design property).
