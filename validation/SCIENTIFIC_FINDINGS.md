# HelioMesh — Scientific Findings

> Generated from reproducible experiments on the frozen simulation test split
> (rows 10200–11999 of `ml/sequences.csv`) unless otherwise stated.
> All benchmark numbers are measured values. No claim is aspirational.

---

## 1. What the ML Actually Learns

### RF Snapshot Classifier
- **RF vs rules agreement rate: 99.2%** on 1,800 test rows.
- **RF accuracy vs rule-derived ground truth: 99.2%**, macro-F1: 0.9911.
- **Honest assessment:** The RF classifier was trained on rule-generated data and accurately reproduces those rules. The primary additional value is a **calibrated probability output** (confidence score) that the hard rules cannot produce. On simulation data, the RF is largely a probabilistic re-implementation of the safety rules — this is expected and not a design flaw.

### GB Temporal Predictor
- **Simulation test benchmark (frozen):** accuracy=0.9717, macro-F1=0.9708, CRITICAL recall=0.9819.
- KP-only threshold baseline: macro-F1=0.8977, CRITICAL recall=0.8382.
- Multi-feature threshold (kp_t5>4 OR delta_kp>1.0): macro-F1=0.9045.
- **GB gain vs KP-only:** +0.0731 macro-F1, +0.1437 CRITICAL recall.
- The temporal window adds measurable and reproducible value beyond KP alone, primarily in reducing missed anomalies.

---

## 2. Synthetic Stand-In Anomaly Methodology (NOT Real Telemetry Validation)

> **Important:** No real spacecraft telemetry dataset was available. The execution environment has no outbound network access and the SMAP S3 bucket returns HTTP 403. The results below are from a fully synthetic stand-in and demonstrate methodology only. They are **not** evidence of real spacecraft anomaly detection capability.

- A methodology demonstration was built using a **SYNTHETIC_STANDIN** (seed=42, `anomaly_methodology_metrics.json`) that mimics SMAP structure (3 channels, injected anomaly windows at known positions).
- Evaluation method: Isolation Forest trained on normal-only split.
- Methodology results: precision=0.818, recall=0.990, F1=0.896, ROC-AUC=0.9998, event detection=4/4 (100%).
- The evidence bridge produces `real_telemetry_status` (NOMINAL/ANOMALOUS/UNCERTAIN), `real_telemetry_anomaly_score`, and `real_telemetry_confidence` as **evidence fields** — not simulation state labels.

---

## 3. What Is NOT Proven

- **Real spacecraft generalization:** The RF and GB models have never been tested on real mission telemetry. All training and test data is simulation-generated.
- **Real telemetry anomaly benchmark:** The `anomaly_methodology_metrics.json` result uses a fully synthetic stand-in — not the real SMAP dataset. It proves the anomaly detection pipeline works on synthetic data with known anomaly positions. Nothing more.
- **Real telemetry temporal validation:** No real temporal anomaly prediction was performed.
- The OMNI2 internal-consistency check is **not spacecraft failure prediction**. It confirms the RF model is self-consistent when applied to real space-weather inputs — nothing more.
- The anomaly detection task (NORMAL vs ANOMALOUS) is **entirely separate** from the HelioMesh simulation states. No mapping exists between them.

---

## 4. Does ML Add Value Beyond the Rules?

**Answer: Yes, with important caveats.**

| Condition | Macro-F1 | CRITICAL Recall |
|-----------|----------|-----------------|
| Rules-only (ceiling) | 1.0000 | 1.0000 |
| RF model | 0.9911 | ~0.99 |
| GB temporal model | **0.9708** | **0.9819** |
| Multi-feature threshold | 0.9045 | 0.8630 |
| KP-only threshold | 0.8977 | 0.8382 |
| Linear-trend KP (post-hoc) | 0.6715 | 0.6403 |
| Always-CRITICAL | 0.3686 | 1.0000 |

**RF:** Adds a calibrated probability estimate; prediction accuracy is essentially identical to the rules on simulation data (by construction).

**GB temporal model:** Adds +0.073 macro-F1 and +0.144 CRITICAL recall over the best single-feature shortcut. The benefit is real and reproducible. The mechanism is the 30-minute trend window (delta features), especially `delta_kp_index` (83.3% top-feature importance) combined with non-KP features that contribute 14.2% of total importance.

**Caveat:** All evidence is on simulation data. The same conclusion may not hold on real spacecraft telemetry. The RF/GB models should not be applied to real missions without independent validation.

---

## 5. Why OMNI2 Temporal Transfer Fails (F1 = 0.4839)

**Controlled cadence experiment results:**

| Experiment | Cadence | Macro-F1 |
|------------|---------|----------|
| Simulation (frozen) | 5 min | 0.9708 |
| Simulation downsampled to hourly | 1 hour | **0.4329** |
| OMNI2 hourly evaluation | 1 hour | 0.4839 |

**Cadence mismatch is a major demonstrated contributor to the degradation; additional domain differences may also contribute.**

- Applying hourly cadence to the *same simulation data* produces 0.4329 — lower than the OMNI2 result (0.4839), showing that cadence alone accounts for most of the measured drop.
- The delta features (trained on 5-min differences) have completely different magnitudes when computed over 6 hours.
- The OMNI2 evaluation combines **three simultaneous mismatches** whose individual contributions cannot be fully isolated without real hourly-cadence training data:
  1. Cadence mismatch (hourly vs 5-min training) — demonstrated as primary contributor
  2. Domain shift (real solar-wind distributions vs simulation) — present but not isolated
  3. Target mismatch (t+1h ahead vs t+30min ahead) — present but not isolated
- **Classification: DOMAIN-SHIFT / TRANSFER TEST — not a temporal prediction benchmark.**
- The low F1 reflects these combined mismatches. Cadence alone is a sufficient explanation, but the other factors are not ruled out.

---

## 6. Is the Policy Stable?

**Policy label:** HelioMesh Prototype Autonomy Policy

**Safety property tests: 6/6 pass.**

| Property | Result |
|----------|--------|
| SP-1: Critical temporal risk blocks AUTO | PASS |
| SP-2: RF/GB disagreement blocks AUTO | PASS |
| SP-3: Granite cannot override routing | PASS |
| SP-4: Lower confidence → non-increasing autonomy | PASS |
| SP-5: Dangerous telemetry gets ≤ autonomy | PASS |
| SP-6: Disagreement policy only tightens | PASS |

**Sensitivity finding:** The policy has **sharp thresholds** — small changes at the exact boundary (e.g., KP=3.999 vs 4.001) flip the route immediately. This is expected for a hard-threshold prototype but would be unacceptable for a real spacecraft autonomy policy. Real policies would require hysteresis, uncertainty bands, or probabilistic routing.

**Policy test suite: 25/25 pass** (100%), including 10 disagreement-specific scenarios.

**Disclaimer:** Weights and thresholds (KP penalties, orbit penalties, power penalties, AUTO≥70, APPROVAL 40–69, ESCALATE<40) are manually defined for this prototype. They are **not validated spacecraft safety standards** and must not be used for real mission operations without independent validation.

---

## 7. Does Granite Stay Grounded?

**Grounded Decision Trace Evaluation on 3 live scenarios:**

- **Overall grounding score: 0.94** (17/18 checks pass)
- **Contradiction rate: 0.00** (no cases where Granite output contradicts supplied evidence)
- All sections complete (7/7) for all 3 scenarios
- No authorship claims detected (Granite never claims it selected the route)
- Route consistency: all traces recommend actions compatible with the deterministic route
- One partial failure: "warning" scenario evidence_coverage — trace uses different terminology for GB label but contains no contradiction

**Limitation:** This evaluation only checks internal consistency (does the trace contradict its own evidence?). It does not verify whether the reasoning is scientifically correct. A fully grounded trace can still contain incorrect inferences if the underlying evidence is ambiguous.

---

## 8. Remaining Limitations

1. **All ML training and testing is on simulation data.** The RF and GB models have not been tested on real spacecraft telemetry. The 99%+ simulation metrics do not generalise to real missions.

2. **Real telemetry benchmark uses a synthetic stand-in.** The actual NASA SMAP dataset was unavailable (S3 bucket no longer public). Results demonstrate methodology only.

3. **The policy thresholds are prototype values.** KP=4, KP=6, orbit=0.5km, orbit=1.5km, power=10W, power=50W and the confidence tier thresholds (70, 40) were chosen for demonstration. They are not derived from spacecraft engineering standards.

4. **Feature importance is simulation-domain only.** The 83.3% KP importance reflects the simulation physics. On real spacecraft, the relative importance of telemetry channels would differ.

5. **Granite evaluation is based on 3 live decision traces.** A larger evaluation set (50+ traces) would be needed to draw statistically meaningful conclusions about grounding rate.

6. **OMNI2 cadence mismatch is a fundamental limitation.** Fixing it would require retraining the GB model on hourly data, which is outside this prototype's scope. Until then, OMNI2 results should not be reported as "temporal prediction accuracy."

7. **Disagreement policy threshold (0.50) is arbitrary.** The choice of 50% as the GB critical-probability threshold for requiring human approval was not derived from operational risk analysis.

---

## Artifacts Produced

```
validation/results/
    anomaly_methodology_metrics.json — SYNTHETIC STANDIN anomaly methodology demo (NOT real spacecraft data)
    rf_ablation.json                — 4-condition RF vs rules ablation
    temporal_ablation.json          — temporal predictor vs 4 new baselines
    feature_audit.json              — GB feature importance breakdown
    cadence_experiment.json         — 3-experiment cadence controlled test
    domain_shift.json               — simulation vs OMNI2 distribution comparison
    policy_sensitivity.json         — boundary sensitivity analysis
    policy_safety_tests.json        — 6 safety property tests (6/6 pass)
    granite_grounding.json          — grounded trace evaluation (0.94 score)
    final_validation_summary.json   — aggregated summary of all experiments
```

## Frozen ML Artifact Integrity

| File | SHA-256 [:16] | Status |
|------|---------------|--------|
| `risk_model.pkl` | `109a8a43578926a2` | UNCHANGED |
| `label_encoder.pkl` | `20b39a8a67cd98b5` | UNCHANGED |
| `forecaster_model.pkl` | `89864a1706a312c2` | UNCHANGED |
| `forecaster_metrics.json` | `533b3fc5b48e4c09` | UNCHANGED |
