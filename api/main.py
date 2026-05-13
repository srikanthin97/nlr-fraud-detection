from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nlr_fraud.preprocess import NUMERIC_COLS

ARTIFACT = Path(os.environ.get("NLR_MODEL_PATH", "models/artifacts/advanced.joblib"))

app = FastAPI(title="NLR Fraud Detection API", version="0.1.0")


class TransactionPayload(BaseModel):
    hour_of_day: int = Field(ge=0, le=23)
    amount: float = Field(gt=0)
    merchant_risk_score: float = Field(ge=0, le=1)
    device_new_to_user: int = Field(ge=0, le=1)
    country_mismatch: int = Field(ge=0, le=1)
    txn_count_24h: int = Field(ge=0)


class ScoreResponse(BaseModel):
    fraud_probability: float
    model_path: str


def _load_model():
    if not ARTIFACT.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact missing. Train models or set NLR_MODEL_PATH.",
        )
    return joblib.load(ARTIFACT)


@app.get("/health")
def health():
    return {"status": "ok", "model_configured": ARTIFACT.exists()}


@app.post("/score", response_model=ScoreResponse)
def score(txn: TransactionPayload):
    model = _load_model()
    row = pd.DataFrame([txn.model_dump()])[NUMERIC_COLS]
    proba = float(model.predict_proba(row)[0, 1])
    return ScoreResponse(fraud_probability=proba, model_path=str(ARTIFACT.resolve()))
