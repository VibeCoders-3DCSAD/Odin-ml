# Odin ML

Python microservice for machine learning APIs and inference, plus the complete model development pipeline for the Odin thesis.

## What's Here

| Directory | Purpose |
|-----------|---------|
| `app/` | FastAPI microservice for model serving (PFP, forecaster, anomaly, budget) |
| `models/` | Canonical home for new-scope FINAL model artifacts + `metadata.json` |
| `tests/` | Pytest coverage for the serving API |
| `training/scripts/` | Data collection, preprocessing, feature engineering, and training pipeline |
| `training/docs/phases/` | Model development runbooks (Phase 7 evaluation through 10 monitoring) |
| `training/datasets/` | Processed + engineered feature matrices (Parquet, gitignored) |
| `training/synth/` | Generated personas and transactions (Parquet, gitignored) |
| `training/models/` | Intermediate model checkpoints (joblib, gitignored) |
| `docs/models/` | RRL-grounded model candidate roster + model-lifecycle guidance |
| `training/figures/` | EDA plots and analysis outputs (gitignored) |

> **Model artifacts:** the top-level `models/` directory is the canonical home for **final**
> model artifacts — every family is committed with `evaluation.json`,
> `evaluation_report.md`, and `metadata.json` (winner contract documented in
> `models/README.md`). Training lives in `training/` as scripts + docs only.

> **New contributor?** Start with `docs/models/TEAMMATE-GUIDE.md` — the trusted path for
> humans and AI agents: decided model scope, per-family training runbooks, CPU/GPU toggles,
> metadata/eval contract, and verification commands.

## Model Development Pipeline

```
FIES 2023 → collector.py → preprocessor.py → feature engineering → train_* → models/
     ↓            ↓              ↓                 ↓                 ↓
datasets/    datasets/      datasets/         datasets/        models/
  raw/     unprocessed/     processed/      engineered/        pfp/
                                 ↓              forecaster/     forecaster/
                           datasets/synth/      anomaly/        anomaly/
                           personas +           dimension-
                           transactions         discovery/
```

All pipeline commands run from the repository root with the virtualenv activated:

```bash
python training/scripts/collector.py \
  --input training/datasets/raw/ \
  --output training/datasets/unprocessed/
```

```bash
python training/scripts/preprocessor.py \
  --input training/datasets/unprocessed/puf.parquet \
  --output training/datasets/processed/
```

`preprocessor.py` also writes the synthetic personas, transactions, and monthly
summaries used by the model-specific feature engineering steps:

```text
training/synth/{personas.json, personas.parquet, transactions.parquet, monthly_summaries.parquet}
```

Feature engineering (one step per model family; each outputs a gitignored feature matrix):

```bash
python training/scripts/feature_engineering.py \
  --input training/datasets/processed/ \
  --output training/datasets/engineered/

python training/scripts/feature_engineering_forecaster.py
python training/scripts/feature_engineering_anomaly.py
python training/scripts/dimension_discovery.py
```

Defaults: `feature_engineering_forecaster.py` reads `training/synth/` +
`training/datasets/processed/split_metadata.json` → `training/datasets/forecaster/`;
`feature_engineering_anomaly.py` reads `training/synth/transactions.parquet` +
splits → `training/datasets/anomaly/`; `dimension_discovery.py` reads `training/synth/`
→ `training/datasets/dimension-discovery/`.

EDA (reads `training/datasets/processed/`, writes `training/figures/`):

```bash
python training/scripts/eda.py \
  --input training/datasets/processed/ \
  --output training/figures/ \
  --seed 42
```

Training (each model reads its own feature matrix and writes `models/<family>/`):

```bash
python training/scripts/train_pfp.py
python training/scripts/train_forecaster.py
python training/scripts/train_anomaly.py
```

Defaults: `train_pfp.py` reads `training/datasets/engineered/` → `models/pfp/`;
`train_forecaster.py` reads `training/datasets/forecaster/` → `models/forecaster/`;
`train_anomaly.py` reads `training/datasets/anomaly/` → `models/anomaly/`. All three
consume `training/datasets/processed/temporal_folds.json` for walk-forward validation.

The Budget Optimizer is a constraint-optimization module (LP via `scipy.linprog`); see the Budget Optimizer MDD v1.0 in `../Odin-Paper/docs/ml/1_problem-statement/module-design-document.md`.

## Tech Stack

- Python `3.14.4` (runtime pinned by `.python-version`)
- FastAPI `0.135.3`
- Uvicorn
- PyTorch `2.13.0` (primary deep learning framework — forecaster LSTM/GRU, anomaly autoencoder)
- scikit-learn `1.9.0`
- scipy, joblib, pandas, numpy, pyarrow
- matplotlib, seaborn (visualization)
- Pytest, HTTPX

## Prerequisites

- Python `3.14.4`
- `pip`
- `venv`

## Repository Layout

```text
odin-ml/
├─ app/
│  ├─ api/                        # FastAPI route modules (health, pfp, forecast, anomaly, budget)
│  ├─ services/                   # Inference and business logic (reuses training feature builders)
│  ├─ models/                     # Model loading and artifact registry
│  ├─ schemas/                    # Pydantic request and response models
│  ├─ core/                       # Settings, startup wiring
│  └─ main.py                     # FastAPI entrypoint
├─ models/                        # Canonical home for new-scope FINAL artifacts + metadata.json
├─ tests/                         # Pytest coverage
├─ training/
│  ├─ scripts/                    # collector, preprocessor, feature engineering, train_* scripts
│  ├─ docs/                       # ML design documents + phases/ runbooks (7–10)
│  ├─ datasets/                   # raw/, unprocessed/, processed/, engineered/, forecaster/, anomaly/, dimension-discovery/ (Parquet, gitignored)
│  ├─ synth/                      # Generated personas + transactions (Parquet, gitignored)
│  ├─ figures/                    # EDA plots + old-scope model artifacts (gitignored)
│  └─ models/                     # Intermediate trained artifacts pfp/, forecaster/, anomaly/ (gitignored)
├─ docs/
│  ├─ standards/                  # Enforceable Python/ML + format + commit standards
│  └─ models/                     # Model candidate roster + lifecycle guidance
├─ pyproject.toml                 # Ruff / mypy / pytest / package metadata
├─ .pre-commit-config.yaml
├─ requirements.txt
├─ requirements-dev.txt
├─ AGENTS.md
└─ README.md
```

## First-Time Setup

### Windows

```powershell
cd C:\path\to\App\odin-ml
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Bash

```bash
cd /path/to/App/odin-ml
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Fish

```fish
cd /path/to/App/odin-ml
python3.14 -m venv .venv
source .venv/bin/activate.fish
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Start the FastAPI dev server:

```bash
uvicorn app.main:app --reload --port 8000
```

## Common Commands

```bash
python3.14 -m venv .venv
source .venv/bin/activate        # Fish: source .venv/bin/activate.fish
pip install -r requirements.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000
pytest
ruff check .
ruff format --check .
mypy app
python -m py_compile app/main.py
```

## Current Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service banner |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (models loaded) |
| `GET` | `/metrics` | Module/winner metadata |
| `POST` | `/api/v1/pfp/classify` | PFP classification (STANDARD / QUESTIONNAIRE) |
| `POST` | `/api/v1/pfp/classify/batch` | Batch PFP classification |
| `POST` | `/api/v1/forecast/predict` | Next-month expense forecast |
| `POST` | `/api/v1/forecast/predict/batch` | Batch forecast |
| `POST` | `/api/v1/anomaly/detect` | Transaction anomaly detection |
| `POST` | `/api/v1/anomaly/detect/batch` | Batch anomaly detection |
| `POST` | `/api/v1/budget/recommend` | Budget allocation recommendation (LP) |
| `POST` | `/api/v1/budget/recommend/batch` | Batch budget recommendation |

Default local URL:

```text
http://localhost:8000
```

Interactive docs: `http://localhost:8000/docs`

## Troubleshooting

### `source .venv/bin/activate` fails in Fish

That is expected. Fish cannot source the Bash activation script; use `source .venv/bin/activate.fish`.

### PowerShell blocks script activation

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the venv again.

### `python3.14` is not found

Confirm Python `3.14.4` is installed and on your `PATH`. On Windows use `py -3.14 --version`; on Bash or Fish use `python3.14 --version`.

### PyTorch not installed

The anomaly autoencoder artifact requires PyTorch at serve time. Install the CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Recommended Next Steps

- Containerize the service (per-module `Dockerfile` + `docker-compose.yml`, ports 8000–8005) matching `../Odin-Paper/docs/ml/1_problem-statement/deployment-architecture.md` v1.1
- Wire the Odin app to `/api/v1/anomaly/detect` (alerts) and `/api/v1/budget/recommend` (budget scheduling); integration plans live in the `../Odin/plans/` repo
- Add auth/rate-limiting to the batch endpoints (`/classify/batch`, `/predict/batch`, `/detect/batch`, `/recommend/batch`) before public exposure
