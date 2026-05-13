# Success criteria and deliverables

## Minimum viable deliverables

| Deliverable | Definition of done |
|-------------|--------------------|
| Project scaffold | Installable package, configs, data directories, Docker + CI wired |
| Synthetic + EDA | Notebook generates data, visualizes separation, exports CSV |
| Training | `scripts/train.py` writes baseline/advanced joblib + `metrics.json` |
| API | `/health` and `/score` with Pydantic validation (FastAPI) |
| Quality gate | `ruff` + `pytest` pass in CI |

## Model quality targets (synthetic baseline — illustrative)

On the shipped synthetic generator with default config, advanced model **PR-AUC should exceed** baseline PR-AUC on the held-out test split (see `metrics.json` after training). Replace these thresholds with business-driven numbers when real labels are available.

## Operational readiness (when moving beyond synthetic data)

- Documented data lineage and refresh cadence.
- Monitoring for score distribution drift, latency, and error rate on `/score`.
- Incident runbooks for rollback to the prior artifact version.
