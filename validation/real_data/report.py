"""
HelioMesh â€” Validation Report Generator
=========================================
Assembles all validation JSON files into a single human-readable text report
and a combined JSON summary.
"""

import os
import json
from datetime import datetime


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _load(filename: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def generate_report() -> dict:
    snap    = _load("snapshot_validation.json")
    temp    = _load("temporal_validation.json")
    early   = _load("early_warning.json")
    agree   = _load("model_agreement.json")
    policy  = _load("policy_tests.json")
    drift   = _load("drift_config.json")

    lines = [
        "=" * 60,
        "  HelioMesh â€” Validation Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "â”€â”€ STAGE 2: Real-Data Validation (OMNI2 Internal Consistency) â”€â”€",
    ]

    if snap:
        lines += [
            f"  Source:             {snap['data_source']}",
            f"  Records:            {snap['n_records']}",
            f"  Snapshot RF",
            f"    Consistency rate: {snap['consistency_rate']*100:.1f}%",
            f"    Macro F1:         {snap['macro_f1']:.4f}",
            f"    Per class:",
        ]
        for cls, m in snap["per_class_metrics"].items():
            lines.append(
                f"      {cls:15s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
            )
    else:
        lines.append("  Snapshot validation: NOT RUN")

    lines.append("")

    if temp:
        lines += [
            f"  Temporal GB",
            f"    Sequences:        {temp['n_sequences']}",
            f"    Consistency rate: {temp['consistency_rate']*100:.1f}%",
            f"    Macro F1:         {temp['macro_f1']:.4f}",
            f"    Window note:      {temp['step_duration_note']}",
        ]
    else:
        lines.append("  Temporal validation: NOT RUN")

    lines += [
        "",
        "â”€â”€ STAGE 3: Early Warning Evaluation â”€â”€",
    ]
    if early:
        lines += [
            f"  Critical transitions:         {early['total_critical_transitions']}",
            f"  Detected early:               {early['transitions_detected_early']}",
            f"  Missed:                       {early['missed_transitions']}",
            f"  False early warnings:         {early['false_early_warnings']}",
            f"  Median lead time (steps):     {early['median_lead_time_steps']}",
            f"  Mean lead time (steps):       {early['mean_lead_time_steps']}",
        ]
    else:
        lines.append("  Early warning: NOT RUN")

    lines += [
        "",
        "â”€â”€ STAGE 4: Model Agreement Analysis â”€â”€",
    ]
    if agree:
        lines += [
            f"  Sequences analysed:           {agree['n_sequences']}",
            f"  Agreement rate:               {agree['agreement_rate']*100:.1f}%",
            f"  Disagreement rate:            {agree['disagreement_rate']*100:.1f}%",
            f"  RF-NOMINAL / GB-CRITICAL:     {agree['rf_nominal_gb_critical']}",
            f"  RF-SAFE_MODE / GB-NOMINAL:    {agree['rf_non_nominal_gb_nominal']}",
        ]
    else:
        lines.append("  Model agreement: NOT RUN")

    lines += [
        "",
        "â”€â”€ STAGE 5: Policy Test Suite â”€â”€",
    ]
    if policy:
        lines += [
            f"  Total scenarios:              {policy['total_scenarios']}",
            f"  Passed:                       {policy['passed']}",
            f"  Failed:                       {policy['failed']}",
            f"  Consistency:                  {policy['consistency_pct']:.1f}%",
        ]
    else:
        lines.append("  Policy tests: NOT RUN")

    lines += ["", "=" * 60]

    report_text = "\n".join(lines)
    print(report_text)

    summary = {
        "report_generated_at": datetime.now().isoformat(),
        "snapshot_validation":  snap,
        "temporal_validation":  temp,
        "early_warning":        early,
        "model_agreement":      agree,
        "policy_tests":         policy,
    }
    out = os.path.join(RESULTS_DIR, "validation_summary.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved â†’ {out}")
    return summary


if __name__ == "__main__":
    generate_report()

