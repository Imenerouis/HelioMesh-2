import math
import json
from datetime import datetime

def simulate_orbit(kp_index, sail_angle, solar_wind_speed):
    """
    Simple orbital simulation model
    kp_index: geomagnetic activity index (0-9)
    sail_angle: mesh angle in degrees (0-90)
    solar_wind_speed: solar wind speed in km/s
    """
    
    drag_factor = (sail_angle / 90) * (1 + kp_index * 0.1)
    orbit_deviation = drag_factor * (solar_wind_speed / 400) * 0.5
    power_output = math.cos(math.radians(sail_angle)) * 100
    thrust_output = drag_factor * 0.02

    result = {
        "timestamp": datetime.now().isoformat(),
        "sail_angle": sail_angle,
        "kp_index": kp_index,
        "solar_wind_speed": solar_wind_speed,
        "drag_factor": round(drag_factor, 3),
        "orbit_deviation": round(orbit_deviation, 3),
        "power_output": round(power_output, 2),
        "thrust_output": round(thrust_output, 4),
        "status": "critical" if kp_index > 6 else "warning" if kp_index > 4 else "nominal"
    }

    return result

if __name__ == "__main__":
    print("=== Normal Conditions ===")
    print(json.dumps(simulate_orbit(kp_index=3.5, sail_angle=45, solar_wind_speed=450), indent=2))
    
    print("\n=== Solar Storm ===")
    print(json.dumps(simulate_orbit(kp_index=7.5, sail_angle=90, solar_wind_speed=800), indent=2))
