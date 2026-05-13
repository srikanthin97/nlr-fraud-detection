"""Optional Flask mirror of the scoring API (same contract as FastAPI for simple deployments)."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from nlr_fraud.preprocess import NUMERIC_COLS

ARTIFACT = Path(os.environ.get("NLR_MODEL_PATH", "models/artifacts/advanced.joblib"))

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_configured": ARTIFACT.exists()})


@app.post("/score")
def score():
    payload = request.get_json(force=True, silent=False) or {}
    row = pd.DataFrame([payload])[NUMERIC_COLS]
    if not ARTIFACT.exists():
        return jsonify({"error": "Model artifact missing"}), 503
    model = joblib.load(ARTIFACT)
    proba = float(model.predict_proba(row)[0, 1])
    return jsonify({"fraud_probability": proba, "model_path": str(ARTIFACT.resolve())})
