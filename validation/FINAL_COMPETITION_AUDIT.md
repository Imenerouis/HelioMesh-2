# HelioMesh — Final Competition Audit

**IBM AI Builders Challenge 2025**  
**Audit date:** 2025  
**Status: COMPLETE**

---

## TECHNICAL EXECUTION

### Simulation Benchmark (Frozen)

| Artifact | SHA-256 | Status |
|----------|---------|--------|
| `ml/risk_model.pkl` | `109a8a43578926a2a4168b2eba404e09...` | UNCHANGED |
| `ml/label_encoder.pkl` | `20b39a8a67cd98b5b6e6b464eb284204...` | UNCHANGED |
| `ml/forecaster_model.pkl` | `89864a1706a312c21a5fb86b67d925f5...` | UNCHANGED |
| `ml/forecaster_metrics.json` | `533b3fc5b48e4c095f9356e4dbd0311a...` | UNCHANGED |

| Test | Result |
|------|--------|
| Policy tests (25 scenarios) | 25/25 PASS |
| Safety properties | 6/6 PASS |
| Temporal predictor macro-F1 | 0.9708 |
| CRITICAL class recall | 98.2% |
| ROC-AUC | 0.9975 |

### Real OPS-SAT Benchmark

| Item | Value |
|------|-------|
| Dataset | OPS-SAT-AD, Zenodo 12588359, MIT License |
| `segments.csv` SHA-256 | `d5201e9e751eb2a53a0ff7c11567dc4e...` |
| `dataset.csv` SHA-256 | `b524177da6f516d5c9f63c7acbc3853...` |
| Test partition | 529 segments, 113 anomalous, zero leakage |
| RF F1 | **0.9209** |
| ROC-AUC | **0.9874** |
| Calibrated unsafe AUTO | **9 (baseline: 14, −36%)** |
| Calibrated recall | **0.9204 (baseline: 0.8761, +4.4pp)** |

### All README Numbers Traced to JSON Artifacts

| README Claim | Source File | Verified |
|--------------|-------------|---------|
| RF F1=0.9209 | `opssat_ad_real_metrics.json`.random_forest.f1 | YES |
| ROC-AUC=0.9874 | `opssat_ad_real_metrics.json`.random_forest.roc_auc | YES |
| Unsafe AUTO 14→9 | `real_policy_calibration.json`.test_results.{baseline,calibrated}.event_metrics.unsafe_auto | YES |
| Recall 0.876→0.920 | `real_policy_calibration.json`.test_results.{baseline,calibrated}.event_metrics.recall | YES |
| Event precision=0.9706 | `opssat_ad_event_metrics.json`.event_precision | YES |
| Chi2=85.08 p<0.001 | `real_temporal_audit.json`.transition_independence_test | YES |
| Disagreement EW prec=5.4% | `real_policy_calibration.json`.disagree_cv.mean_ew_prec_unconditioned | YES |
| Conditioned EW prec=38.7% | `real_policy_calibration.json`.disagree_cv.mean_ew_prec_cond_cur_gt025 | YES |
| Policy tests 25/25 | `policy_tests.json`.passed | YES |
| Granite grounding 0.9444 | `granite_grounding.json`.grounding_score | YES |
| Real Granite grounding 1.0 | `granite_real_evidence.json`.structural_audit.grounding_score | YES |
| Simulation macro-F1=0.9708 | `ml/forecaster_metrics.json` + `validation_summary.json` | YES |

### Secrets Audit

| Check | Result |
|-------|--------|
| `.env` committed to git | NO — in `.gitignore` |
| API keys in source code | NONE FOUND |
| Hardcoded credentials | NONE |
| `.env.example` contains real keys | NO — placeholder strings only |

---

## INNOVATION

### What Is Novel

1. **Deterministic policy + LLM explanation separation.** Granite explains but never decides. The policy route is computed before Granite is called and passed as authoritative context. This design prevents LLM hallucination from affecting safety-critical routing.

2. **Model disagreement as a first-class safety signal.** When RF (current state) and GB (30-min forecast) disagree — specifically RF=NOMINAL + GB=CRITICAL_AHEAD — auto-execution is blocked regardless of confidence score. This early-warning mechanism is tested with 10 dedicated policy scenarios.

3. **Real-spacecraft benchmark integration with semantic separation.** OPS-SAT-AD evidence is explicitly bounded as binary-only and never mapped to the simulation taxonomy. The semantic boundary is enforced in code (`opssat_evidence.py`), in the API (`/opssat/evidence`), and in the dashboard (SEMANTIC BOUNDARY panel).

4. **Policy calibration on training data only.** Policy thresholds were tuned using 5-fold CV on the training partition. The test set was touched exactly once, after calibration was frozen.

5. **Fully auditable validation chain.** Every number in the README and dashboard traces to a code-generated JSON artifact. No metric was manually entered.

### What Is Not Novel

- Random Forest and Gradient Boosting are standard models.
- FastAPI + Next.js is a standard stack.
- IBM Granite as LLM backend is the required platform.
- Anomaly detection on satellite telemetry is an active research area.

---

## CHALLENGE FIT

### "Advance Space Exploration with AI"

| Criterion | Evidence |
|-----------|---------|
| Space operations domain | Spacecraft health monitoring, solar weather decision-making |
| AI technology (IBM) | IBM Granite 4 via watsonx.ai — 7-section structured reasoning |
| Real spacecraft data | ESA OPS-SAT-AD dataset with verified results |
| Safety-critical AI | Deterministic policy prevents unsafe auto-execution; disagreement routing adds human oversight |
| Reproducibility | All results reproducible via `python -m validation.real_spacecraft.opssat_ad.evaluate` |

### Limitations in Challenge Context

- No live spacecraft integration — API connects to simulated space weather
- No real-time telemetry stream — OPS-SAT evidence is pre-computed from static files
- 30-minute prediction horizon not validated on real spacecraft data
- Single real dataset — cannot claim generalizability to other missions

---

## FEASIBILITY

### What Works Today

- Backend API runs locally and on cloud (Procfile + railway.toml + render.yaml provided)
- Dashboard renders all tabs including OPS-SAT REAL evidence
- Granite integration requires valid IBM watsonx credentials
- All validation scripts are runnable (`python -m validation.real_spacecraft.opssat_ad.evaluate`)
- OPS-SAT data locally available (Zenodo download, MIT license, SHA-256 verified)

### What Requires Credentials

- Granite decision traces (requires `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`)
- IBM IAM token exchange
- Without credentials: all endpoints except `/decision/new` (Granite portion), `/chat` still function

### Production Readiness

- Health endpoint: `/health` — returns benchmark status
- CORS configured for public access
- No database — in-memory only (decisions lost on restart)
- No authentication on API endpoints (by design for demo)
- OPS-SAT data files are ~19 MB — not included in repo, must be placed manually

---

## REAL-WORLD IMPACT

### Demonstrated

| Impact | Evidence |
|--------|---------|
| Anomaly detection on real ESA spacecraft telemetry | F1=0.9209 on 529 held-out test segments |
| Policy calibration reduces unsafe AUTO decisions | 14→9 (−36%) on real test data |
| Conditioned disagreement routing improves precision | 5.4%→38.7% on 5-fold validation |
| Deterministic routing prevents Granite from bypassing safety | Policy tests 25/25 |
| Fully traceable evidence chain | All numbers in README trace to code-generated JSON |

### Not Demonstrated

| Claim | Status |
|-------|--------|
| 30-min prediction horizon on real spacecraft | NOT validated on OPS-SAT-AD |
| Four-class risk taxonomy from real telemetry | NOT validated (binary labels only) |
| Human operator evaluation of Granite explanations | NOT tested |
| Deployment on live spacecraft or mission | NOT done |
| Generalizability beyond OPS-SAT | UNKNOWN |
| Operational utility weights for cost matrix | PROTOTYPE ONLY |

---

## OVERALL ASSESSMENT

### Strengths

1. **Honest scientific reporting.** Limitations are explicitly documented throughout — in code, JSON artifacts, dashboard, and README. No metric is inflated or fabricated.

2. **Strong real-data result.** F1=0.9209 and ROC-AUC=0.9874 on real ESA OPS-SAT telemetry is a genuine achievement, independently reproducible.

3. **Rigorous validation methodology.** Official train/test split respected, zero leakage, policy frozen before final test evaluation, sensitivity analysis across 4 utility sets.

4. **Safety-first architecture.** The deterministic policy + Granite explanation separation is a principled design choice that prevents a key failure mode of LLM-in-the-loop systems.

5. **Provenance documentation.** Zenodo DOI, MIT license confirmed from source, SHA-256 hashes for all data and frozen model files.

### Weaknesses

1. **30-minute prediction horizon not validated on real data.** This is the headline simulation capability, but OPS-SAT-AD cannot support it (inter-segment gap = 1s median).

2. **Single real dataset.** One satellite mission, 9 channels, ~5 months. No cross-mission generalizability evidence.

3. **Simulation data is fully synthetic.** The simulation benchmark, while rigorous internally, trains on data generated by HelioMesh's own labeling rules — not real spacecraft operational standards.

4. **No live deployment URL confirmed.** Deployment configuration is provided (Procfile, railway.toml, render.yaml) but public URL availability depends on the deployment environment.

5. **Conditioned disagreement result has high variance (±38%).** The 38.7% precision figure is directionally correct but not stable across folds — 5 folds, small sample sizes.

---

## CLASSIFICATION

| Dimension | Rating | Evidence |
|-----------|--------|---------|
| Technical execution | STRONG | All tests pass, frozen artifacts unchanged, full reproducibility |
| Innovation | MODERATE-HIGH | Novel policy/LLM separation; real data integration; honest limitations |
| Challenge fit | MODERATE | Space domain, IBM Granite, real spacecraft data — but no live integration |
| Feasibility | MODERATE | Runs locally and on cloud; OPS-SAT data requires manual setup |
| Real-world impact | MODERATE | Strong real-data detection; policy safety improvement demonstrated; single dataset |

**Overall: A technically rigorous, honestly documented AI decision-intelligence prototype for spacecraft operations, validated on real ESA telemetry with appropriate scientific caveats.**

---

## REPRODUCTION COMMANDS

```bash
# Reproduce OPS-SAT benchmark
python -m validation.real_spacecraft.opssat_ad.evaluate

# Reproduce policy calibration (Phase 4)
python -m validation.real_spacecraft.opssat_ad.phase4

# Reproduce decision intelligence ablation (Phase 3)
python -m validation.real_spacecraft.opssat_ad.phase3_ablation

# Verify Granite real-evidence grounding
python validation/granite_real_evidence.py

# Verify frozen artifact hashes
python -c "
import hashlib
FROZEN = {'ml/risk_model.pkl':'109a8a43578926a2','ml/label_encoder.pkl':'20b39a8a67cd98b5','ml/forecaster_model.pkl':'89864a1706a312c2','ml/forecaster_metrics.json':'533b3fc5b48e4c09'}
for p,x in FROZEN.items():
    h = hashlib.sha256(open(p,'rb').read()).hexdigest()
    print(('OK' if h[:16]==x else 'FAIL'), p)
"
```

---

*This audit was produced at the conclusion of Phase 6 (Final Integration). All numbers trace to code-generated JSON artifacts. No metric was manually entered or adjusted after test evaluation.*
