"""Train baseline and advanced models; persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nlr_fraud.preprocess import NUMERIC_COLS, build_preprocessor, dataframe_to_xy


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }


def train_models(
    df: pd.DataFrame,
    artifacts_dir: Path,
    random_seed: int = 42,
) -> dict:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fm = dataframe_to_xy(df)
    X_train, X_temp, y_train, y_temp = train_test_split(
        fm.X,
        fm.y,
        test_size=0.3,
        random_state=random_seed,
        stratify=fm.y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_seed,
        stratify=y_temp,
    )

    baseline = Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )
    advanced = Pipeline(
        steps=[
            ("pre", build_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.06,
                    max_iter=200,
                    random_state=random_seed,
                ),
            ),
        ]
    )

    baseline.fit(pd.DataFrame(X_train, columns=NUMERIC_COLS), y_train)
    advanced.fit(pd.DataFrame(X_train, columns=NUMERIC_COLS), y_train)

    def _eval(name: str, model: Pipeline, split: str, X: np.ndarray, y: np.ndarray) -> dict:
        proba = model.predict_proba(pd.DataFrame(X, columns=NUMERIC_COLS))[:, 1]
        m = _metrics(y, proba)
        m.update({"model": name, "split": split})
        return m

    report = [
        _eval("baseline", baseline, "val", X_val, y_val),
        _eval("baseline", baseline, "test", X_test, y_test),
        _eval("advanced", advanced, "val", X_val, y_val),
        _eval("advanced", advanced, "test", X_test, y_test),
    ]

    joblib.dump(baseline, artifacts_dir / "baseline.joblib")
    joblib.dump(advanced, artifacts_dir / "advanced.joblib")
    (artifacts_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"metrics": report, "artifacts": [str(artifacts_dir / "baseline.joblib")]}
