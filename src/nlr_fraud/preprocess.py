"""Train-time preprocessing for fraud features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = [
    "hour_of_day",
    "amount",
    "merchant_risk_score",
    "device_new_to_user",
    "country_mismatch",
    "txn_count_24h",
]


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(transformers=[("num", numeric, NUMERIC_COLS)])


def dataframe_to_xy(df: pd.DataFrame) -> FeatureMatrix:
    missing = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "is_fraud" not in df.columns:
        raise ValueError("Expected column 'is_fraud'")
    X = df[NUMERIC_COLS].to_numpy(dtype=float)
    y = df["is_fraud"].to_numpy(dtype=int)
    return FeatureMatrix(X=X, y=y, feature_names=list(NUMERIC_COLS))
