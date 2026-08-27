"""
test_api.py
-------------
Basic integration tests for the FastAPI backend using TestClient.
These test the API's request/response behavior without needing a
running server — FastAPI's TestClient calls the app directly.

Run:  pytest tests/test_api.py -v

Note: requires models/fraud_model.pkl and models/scaler.pkl to exist
(run the training pipeline first) since api/main.py loads them at import time.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_FEATURES = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_returns_expected_fields():
    payload = {"Amount": 149.62, "Time": 0, "features": SAMPLE_FEATURES}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_batch_returns_correct_count():
    payload = [
        {"Amount": 149.62, "Time": 0, "features": SAMPLE_FEATURES},
        {"Amount": 50.00, "Time": 100, "features": SAMPLE_FEATURES},
    ]
    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    assert "flagged" in data


def test_predict_missing_field_returns_422():
    """Sending an incomplete payload should trigger FastAPI's validation error."""
    payload = {"Amount": 100.0}  # missing Time and features
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_history_endpoint_returns_list():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)