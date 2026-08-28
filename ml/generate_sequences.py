"""
HelioMesh — Temporal Sequence Dataset Generator
================================================
Generates fixed-window sequences for the supervised temporal predictor.

NAMING NOTE:
  This is a supervised temporal predictor, not a classical time-series
  forecasting model (no RNN, no ARIMA, no autoregression).
  Gradient Boosting receives a flattened fixed-length window of past
  telemetry as features and predicts a single future label.
  Accurate description:
      "Supervised classification over a fixed look-back window
       to predict the operational state 30 minutes ahead."

WHAT THIS MODEL DOES (and does NOT do):
  - DOES:  Given 6 simulated telemetry snapshots (t0 … t+25min, each 5 min apart),
           predict whether the simulated satellite will be in a CRITICAL
           operational state (SAFE_MODE or STANDBY) at t+30min.
  - DOES NOT: Predict real satellite failures. All inputs and the target label
           are simulation-generated. Labels represent HelioMesh operational
           safety states, NOT validated spacecraft engineering thresholds.

Why this is different from the snapshot classifier (predictor.py):
  - Snapshot classifier: current telemetry → current label  (reproduces same rules)
  - Temporal predictor:  window of past telemetry → future label (learns trends)

  A system with slowly rising KP and dropping power will be flagged earlier
  by the temporal predictor than by the snapshot classifier, because the
  predictor sees directional change via delta features, not just the current value.

TIMING:
  Each sequence spans 60 minutes total from first observation to target:
    t=0 … t=25   INPUT window (6 steps × 5 min = 30 min look-back)
    t=25 … t=55  Horizon gap  (6 steps × 5 min — NOT in features)
    t=55         TARGET label (state 30 min beyond the last input snapshot)

  The model predicts the operational state 30 full minutes after the
  last observed snapshot — not just the next step.

SEQUENCE INDEPENDENCE (no sliding-window overlap):
  Each sequence is generated with its own independent random initial
  conditions sampled from the regime distribution. This dataset does NOT
  extract sequence[k] = t_k … t_{k+5} and sequence[k+1] = t_{k+1} … t_{k+6}
  from the same continuous simulation trajectory.
  Every sequence starts fresh → no overlapping windows → no cross-sequence leakage.

LEAKAGE PREVENTION:
  1. Feature vector contains t0…t5 only (window). t(5+HORIZON) never appears in features.
  2. Sequences are independent — no shared state between adjacent rows in the CSV.
  3. Training split is chronological 70/15/15 (see train_forecaster.py).

Sequence design:
  - Window: 6 steps × 5 min = 30 min look-back  (INPUT only)
  - Horizon: 6 steps forward = 30 min look-ahead (TARGET only — never in features)
  - Features per step: kp_index, solar_wind_speed, orbit_deviation, power_output,
                       solar_wind_density, b_field  (6 features × 6 steps = 36)
  - Delta features: change from t0 to t5 for each of the 6 features (+6)
  - Total input features: 42
  - Target: binary — 1 = CRITICAL_AHEAD (STANDBY or SAFE_MODE at t+30)
                     0 = NOMINAL_AHEAD
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
import csv
import json
from datetime import datetime, timedelta

random.seed(42)

WINDOW_STEPS  = 6      # look-back window (6 × 5 min = 30 min)
HORIZON_STEPS = 6      # look-ahead horizon (6 × 5 min = 30 min)
STEP_MINUTES  = 5

# ── Space-weather evolution model ────────────────────────────
def _next_kp(kp: float, regime: str) -> float:
    """Simulate KP index evolution with regime-dependent drift."""
    if regime == "rising":
        drift = random.uniform(0.05, 0.25)
    elif regime == "falling":
        drift = random.uniform(-0.20, -0.02)
    else:  # stable
        drift = random.uniform(-0.10, 0.10)
    noise = random.gauss(0, 0.08)
    return max(0.0, min(9.0, kp + drift + noise))


def _next_wind(wind: float, regime: str) -> float:
    if regime == "rising":
        drift = random.uniform(5, 25)
    elif regime == "falling":
        drift = random.uniform(-15, -2)
    else:
        drift = random.uniform(-8, 8)
    return max(250.0, min(1200.0, wind + drift + random.gauss(0, 5)))


def _next_density(dens: float, regime: str) -> float:
    if regime == "rising":
        drift = random.uniform(0.3, 1.2)
    elif regime == "falling":
        drift = random.uniform(-0.8, -0.1)
    else:
        drift = random.uniform(-0.3, 0.3)
    return max(1.0, min(30.0, dens + drift + random.gauss(0, 0.15)))


def _next_bz(bz: float, regime: str) -> float:
    if regime == "rising":   # more southward = more negative
        drift = random.uniform(-2.0, -0.2)
    elif regime == "falling":
        drift = random.uniform(0.2, 1.5)
    else:
        drift = random.uniform(-0.5, 0.5)
    return max(-40.0, min(5.0, bz + drift + random.gauss(0, 0.3)))


def _simulate_step(kp, sail, wind, dens, bz):
    """Compute derived telemetry for one step (mirrors simulator.py logic)."""
    drag        = (sail / 90) * (1 + kp * 0.1)
    orbit_dev   = drag * (wind / 400) * 0.5
    power_out   = math.cos(math.radians(sail)) * 100
    return kp, wind, round(orbit_dev, 4), round(power_out, 4), dens, bz


def _label(kp, orbit_dev, power_out) -> int:
    """
    HelioMesh operational safety label at a future time step.
    Returns 1 (CRITICAL_AHEAD) if SAFE_MODE or STANDBY would be triggered.
    Returns 0 (NOMINAL_AHEAD) otherwise.
    """
    if kp > 6.0 or orbit_dev > 1.5 or power_out < 10.0:
        return 1   # SAFE_MODE → CRITICAL
    if kp > 4.0 or orbit_dev > 0.8 or power_out < 40.0:
        return 1   # STANDBY → CRITICAL
    return 0       # NOMINAL


# ── Sequence generator ────────────────────────────────────────
def _generate_sequence(regime: str):
    """
    Generate one (window, horizon_label) pair.

    - Window:  WINDOW_STEPS snapshots evolving under `regime`
    - Future:  HORIZON_STEPS more steps to determine the t+30 label
    """
    # Initial conditions vary by regime
    if regime == "nominal_stable":
        kp   = random.uniform(0.5, 3.5)
        wind = random.uniform(300, 500)
        dens = random.uniform(2.0, 7.0)
        bz   = random.uniform(-8, 2)
        sail = random.uniform(30, 55)
        evo  = "stable"
    elif regime == "nominal_rising":
        kp   = random.uniform(1.0, 3.5)
        wind = random.uniform(350, 520)
        dens = random.uniform(3.0, 8.0)
        bz   = random.uniform(-10, 0)
        sail = random.uniform(35, 60)
        evo  = "rising"
    elif regime == "warning_stable":
        kp   = random.uniform(4.0, 5.5)
        wind = random.uniform(500, 680)
        dens = random.uniform(7.0, 14.0)
        bz   = random.uniform(-18, -6)
        sail = random.uniform(50, 72)
        evo  = "stable"
    elif regime == "warning_rising":
        kp   = random.uniform(3.5, 5.0)
        wind = random.uniform(450, 620)
        dens = random.uniform(6.0, 12.0)
        bz   = random.uniform(-15, -4)
        sail = random.uniform(45, 68)
        evo  = "rising"
    elif regime == "storm_building":
        kp   = random.uniform(4.5, 6.5)
        wind = random.uniform(580, 800)
        dens = random.uniform(10.0, 20.0)
        bz   = random.uniform(-28, -10)
        sail = random.uniform(60, 85)
        evo  = "rising"
    else:  # storm_active
        kp   = random.uniform(6.5, 9.0)
        wind = random.uniform(700, 1100)
        dens = random.uniform(14.0, 28.0)
        bz   = random.uniform(-38, -15)
        sail = random.uniform(70, 90)
        evo  = "stable"

    # Build the look-back window
    window_feats = []
    for _ in range(WINDOW_STEPS):
        kp_s, wind_s, orb_s, pwr_s, dens_s, bz_s = _simulate_step(kp, sail, wind, dens, bz)
        window_feats.append([kp_s, wind_s, orb_s, pwr_s, dens_s, bz_s])
        kp   = _next_kp(kp, evo)
        wind = _next_wind(wind, evo)
        dens = _next_density(dens, evo)
        bz   = _next_bz(bz, evo)

    # Advance HORIZON_STEPS to get future state
    for _ in range(HORIZON_STEPS):
        kp   = _next_kp(kp, evo)
        wind = _next_wind(wind, evo)
        dens = _next_density(dens, evo)
        bz   = _next_bz(bz, evo)
    _, wind_f, orb_f, pwr_f, _, _ = _simulate_step(kp, sail, wind, dens, bz)
    future_label = _label(kp, orb_f, pwr_f)

    return window_feats, future_label


def _flatten(window: list) -> list:
    """
    Flatten 6×6 window into 36 features + 6 delta features (t5 - t0).
    Total: 42 features.
    """
    flat = []
    for step in window:
        flat.extend(step)
    # Delta features: last step minus first step
    for i in range(len(window[0])):
        flat.append(round(window[-1][i] - window[0][i], 4))
    return flat


# ── Feature names ─────────────────────────────────────────────
STEP_FEATURES = ["kp_index", "solar_wind_speed", "orbit_deviation",
                 "power_output", "solar_wind_density", "b_field"]

FEATURE_NAMES = (
    [f"{f}_t{i}" for i in range(WINDOW_STEPS) for f in STEP_FEATURES] +
    [f"delta_{f}" for f in STEP_FEATURES]
)


# ── Dataset generation ────────────────────────────────────────
def generate(n: int = 12_000):
    """
    Stratified sequence generation.
    Distribution:
      nominal_stable   30% — stays nominal through horizon
      nominal_rising   20% — starts nominal, KP rising (may tip into warning)
      warning_stable   15% — in warning zone, stays there
      warning_rising   15% — in warning, rising toward storm
      storm_building   10% — approaching storm onset
      storm_active     10% — already in storm regime
    """
    regimes = ["nominal_stable", "nominal_rising", "warning_stable",
               "warning_rising", "storm_building", "storm_active"]
    weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]

    records = []
    base_time = datetime(2026, 1, 1)

    for i in range(n):
        regime = random.choices(regimes, weights=weights)[0]
        window, label = _generate_sequence(regime)
        flat = _flatten(window)
        ts = (base_time + timedelta(minutes=i * STEP_MINUTES)).isoformat()
        records.append({
            "timestamp": ts,
            "regime": regime,
            **{name: round(val, 4) for name, val in zip(FEATURE_NAMES, flat)},
            "label": label,
        })

    return records


def main():
    os.makedirs("ml", exist_ok=True)
    print("Generating temporal sequence dataset...")
    records = generate(12_000)

    from collections import Counter
    dist = Counter(r["label"] for r in records)
    regime_dist = Counter(r["regime"] for r in records)
    total = len(records)

    print(f"\nDataset: {total} sequences")
    print(f"Window: {WINDOW_STEPS} steps × {STEP_MINUTES} min = {WINDOW_STEPS*STEP_MINUTES} min look-back")
    print(f"Horizon: {HORIZON_STEPS} steps × {STEP_MINUTES} min = {HORIZON_STEPS*STEP_MINUTES} min look-ahead")
    print(f"\nLabel distribution:")
    print(f"  NOMINAL_AHEAD   (0): {dist[0]:5d}  ({dist[0]/total*100:.1f}%)")
    print(f"  CRITICAL_AHEAD  (1): {dist[1]:5d}  ({dist[1]/total*100:.1f}%)")
    print(f"\nRegime distribution:")
    for r, c in sorted(regime_dist.items()):
        print(f"  {r:20s}: {c:5d}  ({c/total*100:.1f}%)")

    # Save CSV
    csv_path = "ml/sequences.csv"
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved -> {csv_path}")

    # Save metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "n_sequences": total,
        "window_steps": WINDOW_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "step_minutes": STEP_MINUTES,
        "look_back_minutes": WINDOW_STEPS * STEP_MINUTES,
        "look_ahead_minutes": HORIZON_STEPS * STEP_MINUTES,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "label_definition": {
            "0": "NOMINAL_AHEAD — satellite expected in nominal state at t+30min",
            "1": "CRITICAL_AHEAD — satellite expected in STANDBY or SAFE_MODE at t+30min",
        },
        "label_source": (
            "HelioMesh operational safety rules applied to simulated future state. "
            "NOT a predictor of real satellite failures."
        ),
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "regime_distribution": {k: int(v) for k, v in regime_dist.items()},
        "regimes": {
            "nominal_stable":  "KP 0.5–3.5, stable evolution",
            "nominal_rising":  "KP 1.0–3.5, rising toward warning zone",
            "warning_stable":  "KP 4.0–5.5, stable in warning zone",
            "warning_rising":  "KP 3.5–5.0, rising toward storm onset",
            "storm_building":  "KP 4.5–6.5, rapidly building toward storm",
            "storm_active":    "KP 6.5–9.0, active storm conditions",
        },
    }
    with open("ml/sequences_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved → ml/sequences_metadata.json")


if __name__ == "__main__":
    main()
