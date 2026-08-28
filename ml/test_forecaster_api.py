"""Quick smoke-test for forecaster.py inference API."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.forecaster import forecast

def make_snap(kp, wind, sail=45):
    drag = (sail / 90) * (1 + kp * 0.1)
    orb  = round(drag * (wind / 400) * 0.5, 4)
    pwr  = round(math.cos(math.radians(sail)) * 100, 4)
    return {
        "kp_index": kp, "solar_wind_speed": wind, "sail_angle": sail,
        "orbit_deviation": orb, "power_output": pwr,
        "solar_wind_density": 5.0, "b_field": -5.0
    }

# --- Normal: stable KP ~2 ---
normal_window = [make_snap(2.0 + i * 0.05, 400 + i * 2) for i in range(6)]
r = forecast(normal_window)
print("NORMAL window:")
print(f"  label={r['forecast_label']}  P(critical)={r['critical_probability']}  conf={r['forecast_confidence']}")
print(f"  delta_kp={r['delta_kp']}  delta_power={r['delta_power']}")

# --- Storm: rising KP 5 -> 7.5 ---
storm_window = [make_snap(5.0 + i * 0.5, 550 + i * 40, sail=60 + i * 5) for i in range(6)]
r2 = forecast(storm_window)
print("STORM rising window:")
print(f"  label={r2['forecast_label']}  P(critical)={r2['critical_probability']}  conf={r2['forecast_confidence']}")
print(f"  delta_kp={r2['delta_kp']}  delta_power={r2['delta_power']}")

print("forecaster.py API: OK")
