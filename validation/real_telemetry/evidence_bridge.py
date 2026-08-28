"""
HelioMesh -- Evidence Bridge: Real Telemetry -> Decision Engine
===============================================================
Produces a structured evidence dict from the real-telemetry anomaly
detector for consumption by downstream HelioMesh components.

CRITICAL DESIGN CONSTRAINTS
-----------------------------
1. This module does NOT map anomaly detector output to HelioMesh
   simulation states (NOMINAL / STANDBY / SAFE_MODE / CRITICAL_AHEAD).
   Those states are outputs of the RF/GB models trained on SYNTHETIC
   simulation data — they have a completely different domain.

2. The dict returned here is EVIDENCE INPUT, not a routing decision.
   The decision engine MAY use this as one factor among many but MUST
   NOT treat it as a direct state label.

3. `real_telemetry_source` clearly identifies whether the underlying
   data is real SMAP data or the synthetic stand-in so that downstream
   logging and human operators always know which regime they are in.

Output fields
-------------
  real_telemetry_status      : "NOMINAL" | "ANOMALOUS" | "UNCERTAIN"
  real_telemetry_anomaly_score : float in [0.0, 1.0]
  real_telemetry_confidence  : float in [0.0, 1.0]
  real_telemetry_source      : "SMAP_REAL" | "SMAP_STANDIN"

Thresholds
----------
  score >= ANOMALOUS_THRESHOLD  -> ANOMALOUS
  score <= NOMINAL_THRESHOLD    -> NOMINAL
  otherwise                     -> UNCERTAIN
"""

import numpy as np

from validation.real_telemetry.anomaly_model import THRESHOLD

# Evidence thresholds (deliberately wider band than the binary threshold to
# express genuine uncertainty in the middle range)
ANOMALOUS_THRESHOLD = THRESHOLD + 0.10   # e.g. 0.60
NOMINAL_THRESHOLD   = THRESHOLD - 0.15   # e.g. 0.35


def _confidence_from_score(score: float, status: str) -> float:
    """
    Map anomaly score to a confidence in [0.0, 1.0] for the assigned status.

    - ANOMALOUS: confidence scales from 0.5 at threshold to 1.0 at max score.
    - NOMINAL:   confidence scales from 0.5 at threshold to 1.0 at score=0.
    - UNCERTAIN: fixed low confidence (0.45) — we genuinely don't know.
    """
    if status == "ANOMALOUS":
        # linear from (ANOMALOUS_THRESHOLD, 0.5) -> (1.0, 1.0)
        span = 1.0 - ANOMALOUS_THRESHOLD
        if span <= 0:
            return 1.0
        return round(min(1.0, 0.5 + 0.5 * (score - ANOMALOUS_THRESHOLD) / span), 4)
    elif status == "NOMINAL":
        # linear from (0.0, 1.0) -> (NOMINAL_THRESHOLD, 0.5)
        span = NOMINAL_THRESHOLD
        if span <= 0:
            return 1.0
        return round(min(1.0, 1.0 - 0.5 * score / span), 4)
    else:  # UNCERTAIN
        return 0.45


def make_evidence(
    anomaly_score: float,
    source: str,
    *,
    anomalous_threshold: float = ANOMALOUS_THRESHOLD,
    nominal_threshold: float   = NOMINAL_THRESHOLD,
) -> dict:
    """
    Convert a single-timestep anomaly score into an evidence dict.

    Parameters
    ----------
    anomaly_score : float
        Normalised anomaly score in [0, 1] from the detector.
    source : str
        "SMAP_REAL" or "SMAP_STANDIN" as returned by dataset.load_dataset().
    anomalous_threshold : float, optional
    nominal_threshold : float, optional

    Returns
    -------
    dict with four fields (see module docstring).
    """
    score = float(np.clip(anomaly_score, 0.0, 1.0))

    if score >= anomalous_threshold:
        status = "ANOMALOUS"
    elif score <= nominal_threshold:
        status = "NOMINAL"
    else:
        status = "UNCERTAIN"

    confidence = _confidence_from_score(score, status)

    return {
        "real_telemetry_status":        status,
        "real_telemetry_anomaly_score": round(score, 4),
        "real_telemetry_confidence":    confidence,
        "real_telemetry_source":        source,
    }


def make_evidence_batch(
    scores: np.ndarray,
    source: str,
) -> list[dict]:
    """
    Vectorised version of make_evidence for a full time-series.

    Parameters
    ----------
    scores : np.ndarray, shape (T,)
    source : str

    Returns
    -------
    List of T evidence dicts.
    """
    return [make_evidence(float(s), source) for s in scores]


# ---------------------------------------------------------------------------
# Quick smoke-test (not part of the main evaluate pipeline)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_scores = [0.05, 0.35, 0.50, 0.62, 0.90]
    source = "SMAP_STANDIN"
    print("Evidence bridge smoke test")
    print(f"{'Score':>6}  {'Status':10}  {'Confidence':>10}")
    print("-" * 35)
    for s in test_scores:
        ev = make_evidence(s, source)
        print(
            f"{s:>6.2f}  "
            f"{ev['real_telemetry_status']:10}  "
            f"{ev['real_telemetry_confidence']:>10.4f}"
        )
