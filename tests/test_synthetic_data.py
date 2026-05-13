from nlr_fraud.preprocess import NUMERIC_COLS
from nlr_fraud.synthetic_data import generate_transactions


def test_generate_transactions_shape_and_label():
    df = generate_transactions(n_samples=1000, fraud_rate=0.05, random_seed=1)
    assert len(df) == 1000
    assert set(df["is_fraud"].unique()) <= {0, 1}
    fraud_share = df["is_fraud"].mean()
    assert 0.03 < fraud_share < 0.07
    assert "transaction_id" in df.columns


def test_numeric_columns_present():
    df = generate_transactions(n_samples=200, fraud_rate=0.1, random_seed=2)
    for col in NUMERIC_COLS:
        assert col in df.columns
    assert not df[NUMERIC_COLS].isna().any().any()
