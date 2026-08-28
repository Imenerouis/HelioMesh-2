"""
HelioMesh â€” Validation Entry Point
=====================================
Runs all validation stages in sequence and prints a final summary.

Usage:
    python validation/run_validation.py [--stage STAGE]

Stages:
    all       Run all stages (default)
    stage2    Real-data (OMNI2) validation
    stage3    Early warning evaluation
    stage4    Model agreement analysis
    stage5    Policy test suite
    stage6    Drift monitor config
    report    Generate combined report only
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_stage2():
    print("\n" + "="*55)
    print("STAGE 2 -- Real-Data Validation (OMNI2)")
    print("="*55)
    from validation.real_data.evaluate_snapshot import run as snap_run
    from validation.real_data.evaluate_temporal import run as temp_run
    snap_result = snap_run()
    temp_result = temp_run()
    return snap_result, temp_result


def run_stage3():
    print("\n" + "="*55)
    print("STAGE 3 â€” Early Warning Evaluation")
    print("="*55)
    from validation.early_warning import run
    return run()


def run_stage4():
    print("\n" + "="*55)
    print("STAGE 4 â€” Model Agreement Analysis")
    print("="*55)
    from validation.model_agreement import run
    return run()


def run_stage5():
    print("\n" + "="*55)
    print("STAGE 5 â€” Policy Test Suite")
    print("="*55)
    from validation.policy_tests import run
    return run()


def run_stage6():
    print("\n" + "="*55)
    print("STAGE 6 â€” Drift Monitor Config")
    print("="*55)
    from validation.drift_monitor import save_drift_config
    return save_drift_config()


def run_report():
    print("\n" + "="*55)
    print("FINAL REPORT")
    print("="*55)
    from validation.real_data.report import generate_report
    return generate_report()


def run_hardening():
    """Run all scientific hardening experiments (Problems 1-5)."""
    print("\n" + "="*55)
    print("HARDENING -- Scientific Hardening Experiments")
    print("="*55)

    from validation.real_telemetry.evaluate import run as rt_run
    print("\n  [1/6] Real telemetry anomaly benchmark...")
    rt_run()

    from validation.rf_ablation import run as rf_run
    print("\n  [2/6] RF ablation study...")
    rf_run()

    from validation.temporal_ablation import run as ta_run
    print("\n  [3/6] Temporal ablation study...")
    ta_run()

    from validation.feature_audit import run as fa_run
    print("\n  [4/6] Feature importance audit...")
    fa_run()

    from validation.cadence_experiment import run as ce_run
    print("\n  [5/6] Cadence + domain shift experiment...")
    ce_run()

    from validation.policy_sensitivity import run as ps_run
    print("\n  [6/6] Policy sensitivity + safety tests...")
    ps_run()

    from validation.granite_grounding import run as gg_run
    print("\n  [7/7] Granite grounding evaluation...")
    gg_run()

    from validation.generate_final_summary import run as fs_run
    fs_run()


def main():
    parser = argparse.ArgumentParser(description="HelioMesh Validation Pipeline")
    parser.add_argument("--stage", default="all",
                        choices=["all", "stage2", "stage3", "stage4",
                                 "stage5", "stage6", "report", "hardening"])
    args = parser.parse_args()

    stage = args.stage
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

    if stage == "stage2":
        run_stage2()
    elif stage == "stage3":
        run_stage3()
    elif stage == "stage4":
        run_stage4()
    elif stage == "stage5":
        run_stage5()
    elif stage == "stage6":
        run_stage6()
    elif stage == "report":
        run_report()
    elif stage == "hardening":
        run_hardening()
    else:
        # Run all
        run_stage2()
        run_stage3()
        run_stage4()
        run_stage5()
        run_stage6()
        run_report()

    print("\nValidation pipeline complete.")


if __name__ == "__main__":
    main()

