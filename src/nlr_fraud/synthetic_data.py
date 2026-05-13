"""Synthetic transaction data for NLR fraud detection demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_transactions(
    n_samples: int = 50_000,
    fraud_rate: float = 0.02,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a labeled transaction table with mild fraud/non-fraud separation.

    Features approximate card-not-present risk signals: amount skew, velocity,
    device mismatch, and off-hours activity for fraudulent rows.
    """
    rng = np.random.default_rng(random_seed)
    n_fraud = int(round(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    def _legit_block(n: int) -> pd.DataFrame:
        hour = rng.integers(0, 24, size=n)
        amount = rng.lognormal(mean=3.2, sigma=1.1, size=n)
        merchant_risk = rng.beta(2, 5, size=n)
        device_new = rng.binomial(1, 0.08, size=n)
        country_mismatch = rng.binomial(1, 0.02, size=n)
        txn_count_24h = rng.poisson(2.5, size=n)
        return pd.DataFrame(
            {
                "hour_of_day": hour,
                "amount": amount,
                "merchant_risk_score": merchant_risk,
                "device_new_to_user": device_new,
                "country_mismatch": country_mismatch,
                "txn_count_24h": txn_count_24h,
            }
        )

    def _fraud_block(n: int) -> pd.DataFrame:
        hour = rng.integers(0, 24, size=n)
        amount = rng.lognormal(mean=4.4, sigma=1.3, size=n)
        merchant_risk = rng.beta(4, 3, size=n)
        device_new = rng.binomial(1, 0.35, size=n)
        country_mismatch = rng.binomial(1, 0.22, size=n)
        txn_count_24h = rng.poisson(8.0, size=n)
        df = pd.DataFrame(
            {
                "hour_of_day": hour,
                "amount": amount,
                "merchant_risk_score": merchant_risk,
                "device_new_to_user": device_new,
                "country_mismatch": country_mismatch,
                "txn_count_24h": txn_count_24h,
            }
        )
        night = rng.random(n) < 0.45
        df.loc[night, "hour_of_day"] = rng.integers(0, 6, size=int(night.sum()))
        return df

    legit = _legit_block(n_legit)
    fraud = _fraud_block(n_fraud)
    legit["is_fraud"] = 0
    fraud["is_fraud"] = 1
    out = pd.concat([legit, fraud], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    out.insert(0, "transaction_id", np.arange(1, len(out) + 1))
    return out
