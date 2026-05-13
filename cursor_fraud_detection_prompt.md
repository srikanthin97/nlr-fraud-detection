# NLR Fraud Detection — Master Prompt (Cursor / Engineering Spec)

This document is the authoritative description for building and hardening the NLR fraud detection platform. It complements the runnable code in this repository.

## 1. Project structure and setup

- **Language:** Python 3.11+
- **Layout:** `src/nlr_fraud` for library code, top-level `api/` for HTTP services, `notebooks/` for EDA, `configs/` for YAML, `data/` for local datasets, `models/artifacts/` for serialized estimators, `docker/` for container assets, `.github/workflows/` for CI.
- **Packaging:** `pyproject.toml` with setuptools `src` layout; `pip install -e .` is the default developer install.
- **Dependencies:** NumPy, pandas, scikit-learn, matplotlib, seaborn, PyYAML, FastAPI, Uvicorn, Pydantic, Joblib, Flask (optional mirror).

## 2. Data and EDA phase

- **Sources:** Start from `nlr_fraud.synthetic_data.generate_transactions` to produce a reproducible CSV under `data/raw/` (gitignored).
- **EDA goals:** Class balance, univariate fraud vs. legit comparisons, correlation review, basic data quality (missing IDs, impossible ranges).
- **Artifact:** Notebook `notebooks/01_eda_synthetic.ipynb` encodes the narrative and plots.

## 3. Preprocessing and feature engineering

- **Preprocess:** Median imputation + scaling on the numeric transaction vector (`nlr_fraud.preprocess`).
- **Feature hooks:** `nlr_fraud.features` demonstrates log amount and night-hour flags; extend here for domain-specific signals (velocity windows, graph features, device reputation, etc.).

## 4. Model development

- **Baseline:** `LogisticRegression` with `class_weight="balanced"` inside a sklearn `Pipeline` with the column preprocessor.
- **Advanced:** `HistGradientBoostingClassifier` with conservative depth/iterations; tune via held-out validation.
- **Metrics:** Track ROC-AUC and PR-AUC (critical for imbalance); persist `models/artifacts/metrics.json` via `nlr_fraud.train.train_models`.
- **Training entrypoint:** `python scripts/train.py [--input-csv PATH]`.

## 5. API development (FastAPI + Flask)

- **FastAPI:** `POST /score` accepts JSON transaction features and returns `fraud_probability`. `GET /health` reports model availability. `NLR_MODEL_PATH` selects the joblib artifact.
- **Flask:** `api/flask_app.py` mirrors `/health` and `/score` for environments where ASGI hosting is not preferred.

## 6. Docker containerization

- **Image:** `docker/Dockerfile` installs requirements, copies `src` and `api`, exposes port 8000, runs Uvicorn.
- **Compose:** `docker/docker-compose.yml` builds from repo root and sets `NLR_MODEL_PATH` for the container. Mount trained weights in real deployments if they are not baked into the image.

## 7. GitHub CI/CD workflows

- **CI:** On push/PR to `main`/`master`, install dev dependencies, run `ruff check`, then `pytest`.
- **CD (extension point):** Add a workflow to build/push the Docker image to your registry on tags, and optionally deploy to staging with integration tests against `/health`.

## 8. Testing and quality assurance

- **Unit:** Synthetic generator invariants; API scoring with a model trained in a temporary directory.
- **Static analysis:** Ruff in CI; optional mypy tightening over time.
- **Operational checks:** Latency budgets, schema validation (Pydantic), and canary evaluation against shadow traffic when moving beyond synthetic data.

## 9. Documentation requirements

- `README.md` for onboarding and commands.
- `docs/ARCHITECTURE.md` for component diagram and data flow.
- `docs/TESTING.md` for how to run tests locally and in CI.
- `docs/SUCCESS_CRITERIA.md` for measurable outcomes.
- **Model card** (extend): intended use, metrics, limitations, bias/fairness notes when using real PII.

## 10. Success criteria and deliverables

- **Reproducible data path:** One command or notebook path from config → CSV snapshot.
- **Model artifacts:** Baseline and advanced joblib files plus JSON metrics after `scripts/train.py`.
- **API contract:** Documented JSON schema, validated inputs, health endpoint for orchestrators.
- **Container:** `docker compose up` serves scoring on port 8000 when artifacts are present.
- **CI green:** Lint + tests on default branch.
- **Security posture (when handling real data):** Secrets via environment/secret manager, no credentials in git, TLS termination at the edge, audit logging for scores.

This prompt should stay synchronized with implementation changes (training schema, API fields, metrics).
