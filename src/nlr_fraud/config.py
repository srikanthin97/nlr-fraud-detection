from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Paths(BaseModel):
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    artifacts_dir: Path = Field(default=Path("models/artifacts"))


class DataConfig(BaseModel):
    random_seed: int = 42
    n_samples: int = 50_000
    fraud_rate: float = 0.02


class ModelConfig(BaseModel):
    baseline: str = "logistic_regression"
    advanced: str = "hist_gradient_boosting"
    test_size: float = 0.2
    val_size: float = 0.1


class AppConfig(BaseModel):
    paths: Paths = Field(default_factory=Paths)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
