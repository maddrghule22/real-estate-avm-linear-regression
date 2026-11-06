"""Unit and integration tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_PROPERTY_PAYLOAD = {
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "YrSold": 2008,
    "TotalBsmtSF": 856.0,
    "1stFlrSF": 856.0,
    "2ndFlrSF": 854.0,
    "GrLivArea": 1710.0,
    "FullBath": 2,
    "HalfBath": 1,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "TotRmsAbvGrd": 8,
    "Fireplaces": 1,
    "GarageCars": 2,
    "GarageArea": 548.0,
    "LotArea": 8450.0,
    "Neighborhood": "CollgCr",
    "BldgType": "1Fam",
    "HouseStyle": "2Story",
    "ExterQual": "Gd",
    "KitchenQual": "Gd",
    "Foundation": "PConc",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "SaleCondition": "Normal",
}


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single_property_success():
    response = client.post("/predict", json=SAMPLE_PROPERTY_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    pred_price = data["predicted_price_usd"]
    assert 150000.0 < pred_price < 250000.0

    ci = data["confidence_interval"]
    assert ci["lower_bound_usd"] < pred_price < ci["upper_bound_usd"]


def test_predict_single_property_invalid_payload():
    # OverallQual out of bounds (> 10)
    bad_payload = SAMPLE_PROPERTY_PAYLOAD.copy()
    bad_payload["OverallQual"] = 15
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_batch_properties():
    batch_payload = {"properties": [SAMPLE_PROPERTY_PAYLOAD, SAMPLE_PROPERTY_PAYLOAD]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
