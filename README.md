# NR Fraud Detection (End-to-End)

Production-style scaffold for **NLR** payment fraud detection: synthetic data and EDA, preprocessing and feature hooks, **scikit-learn** baselines and gradient boosting, a **FastAPI** scoring service (with an optional **Flask** mirror), **Docker** packaging, and **GitHub Actions** CI.

## Repository layout

| Path | Purpose |
|------|---------|
| `configs/default.yaml` | Data volume, fraud rate, seeds, artifact paths |
| `data/raw`, `data/processed` | Local datasets (CSV outputs are gitignored) |
| `notebooks/01_eda_synthetic.ipynb` | Synthetic generation, profiling, exports |
| `src/nlr_fraud/` | Config loader, synthetic data, preprocess, training |
| `api/main.py` | FastAPI `/health` and `/score` |
| `api/flask_app.py` | Flask equivalents for simple deployments |
| `scripts/train.py` | CLI to materialize data and fit models |
| `docker/` | `Dockerfile` and `docker-compose.yml` |
| `.github/workflows/ci.yml` | Lint + tests on push/PR |
| `docs/` | Architecture, QA, success criteria |
| `cursor_fraud_detection_prompt.md` | Full product/engineering prompt |

## Quickstart

```bash
cd /Users/srikanth/projects/nlr-fraud-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Train models (synthetic data by default) and write `models/artifacts/*.joblib` plus `metrics.json`:

```bash
python scripts/train.py --config configs/default.yaml
```

Run the API locally:

```bash
export NLR_MODEL_PATH=models/artifacts/advanced.joblib
uvicorn api.main:app --reload --port 8000
```

Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Testing and quality gates

- `pytest` exercises synthetic data invariants and the FastAPI surface with a freshly trained temp model.
- `ruff check src api tests scripts` is enforced in CI (`.github/workflows/ci.yml`).

## Documentation map

- [Architecture](docs/ARCHITECTURE.md)
- [Testing and QA](docs/TESTING.md)
- [Success criteria and deliverables](docs/SUCCESS_CRITERIA.md)
- [Master prompt](cursor_fraud_detection_prompt.md)

## License

Proprietary / internal NLR use unless otherwise specified.
