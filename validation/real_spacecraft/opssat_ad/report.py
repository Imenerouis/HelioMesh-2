"""
report.py — OPS-SAT-AD event-level analysis and temporal feasibility.

Tasks:
  1. Event-level evaluation (TASK 5)
  2. Temporal feasibility analysis (TASK 6)
  3. Save opssat_ad_event_metrics.json
  4. Save opssat_ad_temporal_feasibility.json

Usage:
  python -m validation.real_spacecraft.opssat_ad.report
"""

import json
import os
import sys
import numpy as np
import pandas as pd

from .loader import load_segments, load_dataset, get_official_split
from .preprocess import prepare_train_test
from .train import train_all_models
from .metrics import compute_all_metrics

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'results',
)


# ============================================================
# EVENT-LEVEL ANALYSIS
# ============================================================

def run_event_analysis(rf_model=None, X_test=None, y_test_segs=None,
                       test_df=None, verbose: bool = True) -> dict:
    """
    Evaluate anomaly detection at the segment (event) level.

    An "event" in OPS-SAT-AD is a single anomalous segment.
    Since each segment is already a contiguous block of telemetry with a
    single binary label, the segment IS the event unit — no boundary
    reconstruction is needed.

    Event-level metrics are therefore identical to segment-level metrics.
    This is scientifically valid because:
      - The dataset was designed with segment as the unit of annotation
      - Each segment belongs to one channel and has one ground-truth label
      - Anomaly boundaries coincide with segment boundaries by design
    """
    ds = load_dataset()
    _, test_df_full = get_official_split(ds)

    # Ground truth event indicators
    y_event_true = test_df_full['anomaly'].values

    # We need predictions — use the RF model passed in or re-train
    if rf_model is None or X_test is None:
        models = train_all_models()
        rf_model = models['random_forest']
        X_test   = models['X_test']
        y_test   = models['y_test']
    else:
        y_test = y_event_true

    y_event_pred = rf_model.predict(X_test).astype(int)
    y_event_prob = rf_model.predict_proba(X_test)[:, 1]

    # True/detected/missed/false-alert events
    true_anom_events  = int(y_event_true.sum())
    detected_events   = int(((y_event_true == 1) & (y_event_pred == 1)).sum())
    missed_events     = int(((y_event_true == 1) & (y_event_pred == 0)).sum())
    false_alert_events = int(((y_event_true == 0) & (y_event_pred == 1)).sum())
    true_normal_events = int((y_event_true == 0).sum())
    correct_nominal    = int(((y_event_true == 0) & (y_event_pred == 0)).sum())

    event_precision = detected_events / (detected_events + false_alert_events) \
        if (detected_events + false_alert_events) > 0 else 0.0
    event_recall = detected_events / true_anom_events \
        if true_anom_events > 0 else 0.0
    event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) \
        if (event_precision + event_recall) > 0 else 0.0

    # Anomaly rate by channel in test set
    channel_breakdown = {}
    for ch in sorted(test_df_full['channel'].unique()):
        mask = test_df_full['channel'] == ch
        ch_true = y_event_true[mask.values]
        ch_pred = y_event_pred[mask.values]
        channel_breakdown[ch] = {
            'total_segments'  : int(len(ch_true)),
            'anomaly_segments': int(ch_true.sum()),
            'detected'        : int(((ch_true == 1) & (ch_pred == 1)).sum()),
            'missed'          : int(((ch_true == 1) & (ch_pred == 0)).sum()),
            'false_alerts'    : int(((ch_true == 0) & (ch_pred == 1)).sum()),
        }

    result = {
        'methodology_note': (
            "In OPS-SAT-AD each segment is a single annotated event. "
            "No boundary reconstruction is required — segment boundaries "
            "ARE event boundaries by dataset design."
        ),
        'total_test_segments'   : int(len(y_event_true)),
        'true_anomaly_events'   : true_anom_events,
        'true_normal_events'    : true_normal_events,
        'detected_events'       : detected_events,
        'missed_events'         : missed_events,
        'false_alert_events'    : false_alert_events,
        'correct_nominal'       : correct_nominal,
        'event_precision'       : float(event_precision),
        'event_recall'          : float(event_recall),
        'event_f1'              : float(event_f1),
        'channel_breakdown'     : channel_breakdown,
        'model_used'            : 'random_forest (official test partition)',
    }

    if verbose:
        print(f"  Event precision: {event_precision:.4f}")
        print(f"  Event recall:    {event_recall:.4f}")
        print(f"  Event F1:        {event_f1:.4f}")
        print(f"  Detected:        {detected_events}/{true_anom_events}")
        print(f"  Missed:          {missed_events}")
        print(f"  False alerts:    {false_alert_events}")

    return result


# ============================================================
# TEMPORAL FEASIBILITY
# ============================================================

def run_temporal_analysis(verbose: bool = True) -> dict:
    """
    Analyse whether OPS-SAT-AD supports future-anomaly prediction tasks.

    Examines:
      - actual sampling cadence
      - segment duration distribution
      - time gaps between segments
      - whether anomaly onset can be predicted from prior nominal data
    """
    seg = load_segments(parse_timestamps=True)
    ds  = load_dataset()

    # Cadence analysis
    cadence_counts = seg['sampling'].value_counts().to_dict()

    # Segment duration stats
    durations = ds['duration']  # seconds

    # Gaps between consecutive same-channel segments
    all_gaps = []
    for ch in seg['channel'].unique():
        ch_segs = seg[seg['channel'] == ch].sort_values('timestamp')
        # Get the last timestamp of each segment
        seg_end_times = ch_segs.groupby('segment')['timestamp'].max().sort_values()
        seg_start_times = ch_segs.groupby('segment')['timestamp'].min().sort_values()
        # Gaps = start[i+1] - end[i]  for consecutive segments on same channel
        ends   = seg_end_times.values
        starts = seg_start_times.values
        # Match them by ordering
        order = np.argsort(ends)
        ends_sorted   = ends[order]
        starts_sorted = starts[order]
        for i in range(len(ends_sorted) - 1):
            gap = (starts_sorted[i+1] - ends_sorted[i]) / np.timedelta64(1, 's')
            if gap >= 0:
                all_gaps.append(gap)

    all_gaps = np.array(all_gaps)

    # Can we predict future anomalies?
    # Key question: given a window of nominal segments, is there enough
    # lead time before the next anomalous segment to issue a warning?
    # Assess by finding transitions nominal→anomalous in time-ordered segments.

    lead_times = []
    for ch in ds['channel'].unique():
        ch_ds = ds[ds['channel'] == ch].copy()
        # Get segment start times
        ch_seg = seg[seg['channel'] == ch]
        start_times = ch_seg.groupby('segment')['timestamp'].min()
        ch_ds = ch_ds.join(start_times.rename('start_ts'), on='segment')
        ch_ds = ch_ds.sort_values('start_ts')
        ch_ds = ch_ds.reset_index(drop=True)
        # Find nominal → anomaly transitions
        for i in range(len(ch_ds) - 1):
            if ch_ds.loc[i, 'anomaly'] == 0 and ch_ds.loc[i+1, 'anomaly'] == 1:
                gap_s = (ch_ds.loc[i+1, 'start_ts'] - ch_ds.loc[i, 'start_ts'])
                if pd.notnull(gap_s):
                    lead_times.append(float(gap_s / pd.Timedelta(seconds=1)))

    lead_times = np.array(lead_times)

    # Summary
    supports_future_prediction = False
    reason = ""
    if len(lead_times) == 0:
        reason = ("No nominal→anomaly transitions found. Segments are individually "
                  "labelled — dataset does not capture continuous telemetry streams "
                  "that cross anomaly onset boundaries. Segment-level classification "
                  "is the valid task; future-event prediction requires continuity "
                  "of telemetry across the normal-to-anomaly transition.")
    else:
        median_lead = float(np.median(lead_times))
        if median_lead < 60:
            reason = (f"Median lead time between nominal and anomaly segments is "
                      f"{median_lead:.0f}s — insufficient for reliable early warning "
                      f"at operationally meaningful horizons (e.g., 30 min).")
        else:
            supports_future_prediction = True
            reason = (f"Median lead time = {median_lead:.0f}s. "
                      f"Early-warning evaluation may be feasible at limited horizons.")

    result = {
        'task_assessment': (
            'TEMPORAL TASK NOT SUPPORTED — see details'
            if not supports_future_prediction
            else 'LIMITED TEMPORAL TASK POSSIBLE — see details'
        ),
        'rationale': reason,
        'cadence_distribution_seconds': {str(k): int(v) for k, v in cadence_counts.items()},
        'segment_duration_stats': {
            'min_s'   : int(durations.min()),
            'max_s'   : int(durations.max()),
            'mean_s'  : float(durations.mean()),
            'median_s': float(durations.median()),
            'p5_s'    : float(durations.quantile(0.05)),
            'p95_s'   : float(durations.quantile(0.95)),
        },
        'inter_segment_gap_stats': {
            'n_gaps'    : int(len(all_gaps)),
            'min_s'     : float(all_gaps.min()) if len(all_gaps) > 0 else None,
            'max_s'     : float(all_gaps.max()) if len(all_gaps) > 0 else None,
            'median_s'  : float(np.median(all_gaps)) if len(all_gaps) > 0 else None,
            'p95_s'     : float(np.percentile(all_gaps, 95)) if len(all_gaps) > 0 else None,
        },
        'nominal_to_anomaly_transitions': {
            'count'            : int(len(lead_times)),
            'lead_time_min_s'  : float(lead_times.min())    if len(lead_times) > 0 else None,
            'lead_time_max_s'  : float(lead_times.max())    if len(lead_times) > 0 else None,
            'lead_time_median_s': float(np.median(lead_times)) if len(lead_times) > 0 else None,
        },
        'supports_future_prediction'  : supports_future_prediction,
        'supports_early_warning'      : supports_future_prediction,
        'heliomesh_30min_target_valid': False,
        'heliomesh_note': (
            "HelioMesh's 30-minute prediction horizon is based on the "
            "simulation benchmark. OPSSAT-AD is a segment-classification task "
            "and cannot be used to validate that horizon directly."
        ),
    }

    if verbose:
        print(f"  Task assessment: {result['task_assessment']}")
        print(f"  Rationale: {reason[:120]}...")

    return result


def save_event_metrics(result: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'opssat_ad_event_metrics.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    return path


def save_temporal_metrics(result: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'opssat_ad_temporal_feasibility.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    return path


if __name__ == '__main__':
    print("=== EVENT-LEVEL ANALYSIS ===")
    event_result = run_event_analysis(verbose=True)
    p1 = save_event_metrics(event_result)
    print(f"  Saved: {p1}\n")

    print("=== TEMPORAL FEASIBILITY ===")
    temp_result = run_temporal_analysis(verbose=True)
    p2 = save_temporal_metrics(temp_result)
    print(f"  Saved: {p2}\n")
