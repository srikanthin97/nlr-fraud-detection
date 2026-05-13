#!/usr/bin/env python3
"""Train models from synthetic or CSV data and write artifacts."""

from __future__ import annotations

import argparse

from nlr_fraud.config import load_config
from nlr_fraud.synthetic_data import generate_transactions
from nlr_fraud.train import train_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional CSV with the same schema as the synthetic generator.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input_csv:
        import pandas as pd

        df = pd.read_csv(args.input_csv)
    else:
        df = generate_transactions(
            n_samples=cfg.data.n_samples,
            fraud_rate=cfg.data.fraud_rate,
            random_seed=cfg.data.random_seed,
        )
    out = train_models(df, cfg.paths.artifacts_dir, random_seed=cfg.data.random_seed)
    print(out)


if __name__ == "__main__":
    main()
