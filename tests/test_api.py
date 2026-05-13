from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nlr_fraud.synthetic_data import generate_transactions
from nlr_fraud.train import train_models


@pytest.fixture()
def client_with_model(tmp_path, monkeypatch):
    df = generate_transactions(n_samples=2000, fraud_rate=0.05, random_seed=7)
    art = tmp_path / "artifacts"
    train_models(df, art, random_seed=7)
    model_path = art / "advanced.joblib"
    monkeypatch.setenv("NLR_MODEL_PATH", str(model_path))
    import importlib

    import api.main as api_main

    importlib.reload(api_main)
    with TestClient(api_main.app) as client:
        yield client


def test_health(client_with_model):
    r = client_with_model.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_configured"] is True


def test_score(client_with_model):
    payload = {
        "hour_of_day": 3,
        "amount": 250.0,
        "merchant_risk_score": 0.8,
        "device_new_to_user": 1,
        "country_mismatch": 1,
        "txn_count_24h": 12,
    }
    r = client_with_model.post("/score", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert Path(data["model_path"]).exists()
