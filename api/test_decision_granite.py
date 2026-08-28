"""
Tests for Granite decoupling on POST /decision/new.

Verifies that the deterministic decision path succeeds whether or not
Granite/watsonx is available.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from api.main import app, decisions_db, telemetry_window
from engine.decision_engine import route_decision

MOCK_GRANITE_TRACE = """
OBSERVATION: KP index is 2.1. System nominal.
PREDICTION: Mild geomagnetic activity expected.
MODEL STATUS: RF=NOMINAL GB=NOMINAL Agreement=HIGH
EVIDENCE: KP stable, orbit deviation minimal.
RECOMMENDED ACTION: Continue nominal operations.
CONFIDENCE: High
REASON: All indicators within safe bounds.
""".strip()


def _required_fields(data: dict) -> None:
    for key in (
        "telemetry",
        "ml_prediction",
        "ml_forecast",
        "drift_status",
        "drift_details",
        "status",
        "confidence_score",
        "ai_trace",
        "granite_status",
    ):
        assert key in data, f"missing required field: {key}"


class TestDecisionGranite(unittest.TestCase):
    def setUp(self):
        telemetry_window.clear()
        decisions_db.clear()

    @patch("agent.agent.requests.post")
    @patch("agent.agent.get_token")
    def test_granite_available(self, mock_token, mock_post):
        mock_token.return_value = "fake-token"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"generated_text": MOCK_GRANITE_TRACE}]
        }
        mock_post.return_value = mock_response

        client = TestClient(app)
        resp = client.post("/decision/new?scenario=normal")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        _required_fields(data)
        self.assertEqual(data["granite_status"], "AVAILABLE")
        self.assertIsNone(data["granite_error"])
        self.assertIn("KP index is 2.1", data["ai_trace"])
        self.assertNotIn("GRANITE STATUS: UNAVAILABLE", data["ai_trace"])

    @patch("agent.agent.get_token")
    def test_granite_unavailable(self, mock_token):
        mock_token.side_effect = RuntimeError("IBM token error: invalid key")

        client = TestClient(app)
        resp = client.post("/decision/new?scenario=normal")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        _required_fields(data)
        self.assertEqual(data["granite_status"], "UNAVAILABLE")
        self.assertIsNotNone(data["granite_error"])
        self.assertIn("IBM token error", data["granite_error"])
        self.assertIn("GRANITE STATUS: UNAVAILABLE", data["ai_trace"])

    @patch("agent.agent.get_token")
    def test_routing_unchanged_when_granite_unavailable(self, mock_token):
        mock_token.side_effect = RuntimeError("IBM token error: invalid key")

        client = TestClient(app)
        resp = client.post("/decision/new?scenario=normal")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        model_agreement_dict = {
            "rf_label": data["ml_prediction"]["predicted_state"],
            "gb_label": data["ml_forecast"]["forecast_label"],
            "critical_probability": data["ml_forecast"]["critical_probability"],
            "agreement": data["model_agreement"],
        }
        expected = route_decision(
            data["telemetry"],
            ai_trace="",
            model_agreement=model_agreement_dict,
        )

        self.assertEqual(data["status"], expected["status"])
        self.assertEqual(data["confidence_score"], expected["confidence_score"])
        self.assertEqual(data["disagree_override"], expected["disagree_override"])
        self.assertEqual(data["risk_score"], expected["risk_score"])


if __name__ == "__main__":
    unittest.main()
