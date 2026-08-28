"""
HelioMesh Utility Sensitivity Analysis
======================================

Purpose:
    Test whether Architecture D remains preferable to Architecture C
    under multiple explicitly declared decision-cost assumptions.

Important:
    - Uses the already-generated held-out OPS-SAT-AD routing results.
    - Does NOT retrain any model.
    - Does NOT tune thresholds.
    - Does NOT alter the official A/B/C/D results.
    - Does NOT claim that any utility matrix is mission-validated.

Input:
    validation/results/real_decision_ablation.json

Outputs:
    validation/results/real_utility_sensitivity.json
    validation/results/real_utility_sensitivity.csv
"""

import json
import os
import csv


RESULTS_DIR = os.path.join(
    "validation",
    "results"
)

INPUT_PATH = os.path.join(
    RESULTS_DIR,
    "real_decision_ablation.json"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "real_utility_sensitivity.json"
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "real_utility_sensitivity.csv"
)


# ═══════════════════════════════════════════════════════════════════════════
# DECLARED COST SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

UTILITY_SCENARIOS = {

    "prototype_baseline": {
        "normal_auto": 1.0,
        "normal_pending": -0.3,
        "normal_escalated": -0.5,
        "anomaly_auto": -5.0,
        "anomaly_pending": 0.5,
        "anomaly_escalated": 0.8,
    },

    "conservative_safety": {
        "normal_auto": 1.0,
        "normal_pending": -0.5,
        "normal_escalated": -1.0,
        "anomaly_auto": -10.0,
        "anomaly_pending": 0.5,
        "anomaly_escalated": 1.0,
    },

    "permissive": {
        "normal_auto": 1.0,
        "normal_pending": -0.1,
        "normal_escalated": -0.2,
        "anomaly_auto": -3.0,
        "anomaly_pending": 0.3,
        "anomaly_escalated": 0.5,
    },

    "recall_focused": {
        "normal_auto": 0.5,
        "normal_pending": -0.1,
        "normal_escalated": -0.2,
        "anomaly_auto": -8.0,
        "anomaly_pending": 1.0,
        "anomaly_escalated": 1.5,
    },

    "false_alert_sensitive": {
        "normal_auto": 1.0,
        "normal_pending": -1.0,
        "normal_escalated": -2.0,
        "anomaly_auto": -5.0,
        "anomaly_pending": 0.5,
        "anomaly_escalated": 0.8,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ROUTING COUNTS
# ═══════════════════════════════════════════════════════════════════════════

def load_policy_counts():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_PATH}\n\n"
            "Run phase3_ablation.py first."
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    architecture_details = data[
        "architecture_details"
    ]

    counts = {}

    for architecture in ["C", "D"]:

        policy = architecture_details[
            architecture
        ][
            "policy_metrics"
        ]

        anomaly_total = (
            policy["unsafe_auto"]
            + policy["correct_pending"]
            + policy["correct_escalated"]
        )

        normal_total = (
            policy["correct_auto_clear"]
            + policy["unnecessary_pending"]
            + policy["unnecessary_escalated"]
        )

        counts[architecture] = {
            "normal_auto":
                int(policy["correct_auto_clear"]),

            "normal_pending":
                int(policy["unnecessary_pending"]),

            "normal_escalated":
                int(policy["unnecessary_escalated"]),

            "anomaly_auto":
                int(policy["unsafe_auto"]),

            "anomaly_pending":
                int(policy["correct_pending"]),

            "anomaly_escalated":
                int(policy["correct_escalated"]),

            "n":
                int(policy["n"]),

            "anomaly_total":
                int(anomaly_total),

            "normal_total":
                int(normal_total),
        }

        if (
            anomaly_total
            + normal_total
            != policy["n"]
        ):
            raise ValueError(
                f"{architecture}: routing counts do not sum to n."
            )

    return counts


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_utility(
    counts,
    matrix
):
    total = (
        counts["normal_auto"]
        * matrix["normal_auto"]
        +
        counts["normal_pending"]
        * matrix["normal_pending"]
        +
        counts["normal_escalated"]
        * matrix["normal_escalated"]
        +
        counts["anomaly_auto"]
        * matrix["anomaly_auto"]
        +
        counts["anomaly_pending"]
        * matrix["anomaly_pending"]
        +
        counts["anomaly_escalated"]
        * matrix["anomaly_escalated"]
    )

    return float(
        total / counts["n"]
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def main():

    print("=" * 72)
    print("HELIOMESH — UTILITY SENSITIVITY ANALYSIS")
    print("=" * 72)

    counts = load_policy_counts()

    rows = []

    for scenario_name, matrix in UTILITY_SCENARIOS.items():

        utility_c = compute_utility(
            counts["C"],
            matrix
        )

        utility_d = compute_utility(
            counts["D"],
            matrix
        )

        delta = (
            utility_d
            - utility_c
        )

        if delta > 0:
            winner = "D"
        elif delta < 0:
            winner = "C"
        else:
            winner = "TIE"

        rows.append({
            "scenario":
                scenario_name,

            "C_utility":
                utility_c,

            "D_utility":
                utility_d,

            "D_minus_C":
                delta,

            "winner":
                winner,
        })

    d_wins = sum(
        row["winner"] == "D"
        for row in rows
    )

    c_wins = sum(
        row["winner"] == "C"
        for row in rows
    )

    ties = sum(
        row["winner"] == "TIE"
        for row in rows
    )

    summary = {
        "analysis": (
            "Sensitivity analysis of C versus D using routing counts "
            "from the official held-out OPS-SAT-AD test partition."
        ),

        "test_partition":
            "official held-out OPS-SAT-AD test partition",

        "n_test_segments":
            int(counts["C"]["n"]),

        "architectures_compared":
            {
                "C":
                    "Calibrated RF + deterministic routing",

                "D":
                    (
                        "Calibrated RF + temporal evidence + "
                        "lead-time + deterministic routing"
                    ),
            },

        "threshold_tuning_on_test":
            False,

        "mission_cost_validation":
            False,

        "scenario_count":
            len(rows),

        "D_wins":
            int(d_wins),

        "C_wins":
            int(c_wins),

        "ties":
            int(ties),

        "D_wins_all_declared_scenarios":
            bool(
                d_wins == len(rows)
            ),

        "interpretation": (
            "D is utility-preferred under a scenario only when its "
            "declared utility exceeds C. These utilities are analytical "
            "sensitivity assumptions, not validated mission costs."
        ),

        "routing_counts": counts,

        "scenarios": rows,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "C_utility",
                "D_utility",
                "D_minus_C",
                "winner",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Routing counts used:")
    print()

    for architecture in ["C", "D"]:
        print(
            f"{architecture}: "
            f"normal_auto={counts[architecture]['normal_auto']} "
            f"normal_pending={counts[architecture]['normal_pending']} "
            f"normal_escalated={counts[architecture]['normal_escalated']} "
            f"anomaly_auto={counts[architecture]['anomaly_auto']} "
            f"anomaly_pending={counts[architecture]['anomaly_pending']} "
            f"anomaly_escalated={counts[architecture]['anomaly_escalated']}"
        )

    print()
    print(
        f'{"Scenario":<28}'
        f'{"C":>10}'
        f'{"D":>10}'
        f'{"D-C":>12}'
        f'{"Winner":>10}'
    )

    print("-" * 72)

    for row in rows:
        print(
            f'{row["scenario"]:<28}'
            f'{row["C_utility"]:>10.4f}'
            f'{row["D_utility"]:>10.4f}'
            f'{row["D_minus_C"]:>+12.4f}'
            f'{row["winner"]:>10}'
        )

    print()
    print(
        f"D wins: {d_wins}/{len(rows)}"
    )

    print(
        f"C wins: {c_wins}/{len(rows)}"
    )

    print(
        f"Ties: {ties}/{len(rows)}"
    )

    print()
    print(
        f"Saved: {OUTPUT_JSON}"
    )

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()