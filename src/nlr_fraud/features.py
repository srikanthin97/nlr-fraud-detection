"""Optional feature engineering hooks (domain-specific transforms)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
    return out


def add_night_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_night"] = ((out["hour_of_day"] >= 0) & (out["hour_of_day"] <= 5)).astype(int)
    return out
