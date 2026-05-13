# Testing and quality assurance

## Automated tests

Run from the repository root with the virtual environment activated:

```bash
pytest
```

Coverage (optional):

```bash
pytest --cov=nlr_fraud --cov=api --cov-report=term-missing
```

## Linting

```bash
ruff check src api tests scripts
```

CI runs the same commands on Ubuntu with Python 3.11.

## Manual API smoke test

After training:

```bash
export NLR_MODEL_PATH=models/artifacts/advanced.joblib
uvicorn api.main:app --port 8000
curl -s localhost:8000/health
curl -s localhost:8000/score -H 'content-type: application/json' \
  -d '{"hour_of_day":2,"amount":400,"merchant_risk_score":0.9,"device_new_to_user":1,"country_mismatch":1,"txn_count_24h":15}'
```

## Release checklist (suggested)

- [ ] Refresh EDA notebook after schema changes.
- [ ] Confirm PR-AUC on a holdout matches deployment expectations.
- [ ] Verify Docker image includes the intended model artifact or mounted volume.
- [ ] Run load tests if latency SLOs apply.
