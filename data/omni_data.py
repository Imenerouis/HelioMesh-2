import json
from datetime import datetime

def get_omni_data():
    """Return simulated space weather telemetry (mock data).

    This implementation does NOT connect to any external API or NASA OMNI-2 feed.
    All values are hardcoded to represent a baseline scenario for the HelioMesh simulation.
    To use real data, replace the mock_data dict with a live OMNI-2 API call.
    """

    mock_data = {
        "timestamp": datetime.now().isoformat(),
        "kp_index": 3.5,
        "solar_wind_speed": 450,
        "solar_wind_density": 5.2,
        "b_field": -8.3,
        "status": "nominal"
    }
    
    print("Simulated OMNI-2 telemetry loaded:")
    print(json.dumps(mock_data, indent=2))
    return mock_data

if __name__ == "__main__":
    get_omni_data()
