import sys
sys.path.append('.')
from ml.predictor import predict

scenarios = [
    ("NORMAL",  {"kp_index":2.0,"sail_angle":45,"solar_wind_speed":400,"solar_wind_density":4.0,"b_field":-3.0,"orbit_deviation":0.1,"power_output":70.7,"status":"nominal"}),
    ("WARNING", {"kp_index":5.0,"sail_angle":60,"solar_wind_speed":550,"solar_wind_density":8.0,"b_field":-12.0,"orbit_deviation":0.8,"power_output":50.0,"status":"warning"}),
    ("STORM",   {"kp_index":7.5,"sail_angle":90,"solar_wind_speed":800,"solar_wind_density":15.0,"b_field":-25.0,"orbit_deviation":1.75,"power_output":0.0,"status":"critical"}),
]

for name, t in scenarios:
    r = predict(t)
    state = r["predicted_state"]
    risk  = r["risk_probability"] * 100
    conf  = r["model_confidence"] * 100
    print(name + ": predicted=" + state + " risk=" + str(round(risk,1)) + "% confidence=" + str(round(conf,1)) + "%")

print("ML predictor working.")
