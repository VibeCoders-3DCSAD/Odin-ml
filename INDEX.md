# Odin-ML — Repository Index

- **Project:** Development of BUDI: A Personalized Intelligent Finance Management Application for Filipinos Using Classification, Forecasting, Optimization, and Anomaly Detection Models for Improving Savings and Debt
- **Institution:** University of Makati | Group 4, III-DCSAD
- **Last indexed:** 2026-09-20

---

## How to Use This Index

| Need | Go to |
| :--- | :--- |
| Project overview, setup, and usage | `README.md` |
| Pen-and-paper algorithms (PFP, forecast v2, IQR, LP) | `training/docs/pen-and-paper-algorithms.md` |
| FastAPI service code (endpoints, models, schemas) | `app/` |
| Run the test suite | `tests/` (run `pytest`) |
| Model training pipeline and phase docs | `training/` |
| **Trusted path for teammates/agents to train & serve** | `docs/models/TEAMMATE-GUIDE.md` |
| ML design documents (data collection, EDA, thresholds) | `training/docs/` |
| Engineering and documentation standards | `docs/standards/` |
| Enforceable Python/ML code standards | `docs/standards/REPOSITORY-STANDARDS.md` |
| Git commit message conventions | `docs/standards/git-commit-standards.md` |
| Thesis documentation (system spec, PRD, chapters) | **Odin-Paper** |
| RRL corpus and scoring | **Odin-Literature** |

---

## Repository Map

| Path | Purpose |
| :--- | :--- |
| `AGENTS.md` | Agent navigation guide, standards, and repository conventions. |
| `INDEX.md` | This file. Master navigation index. |
| `README.md` | Project overview, pipeline, setup, and usage. |
| `app/` | FastAPI microservice for classification, model serving (PFP, forecaster, anomaly, budget), and optimization. |
| `models/` | Canonical home for new-scope FINAL model artifacts + `metadata.json`. |
| `tests/` | Pytest coverage for the service. |
| `training/` | Model development pipeline (scripts, datasets, models, docs). |
| `docs/` | Standards, documentation, and model-lifecycle docs. |
| `pyproject.toml` | Ruff / mypy / pytest / package metadata. |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, format, hygiene). |

---

## app/ — FastAPI Microservice

| Path | Purpose |
| :--- | :--- |
| `app/main.py` | FastAPI entrypoint. |
| `app/api/` | Route modules (health, pfp, forecast, anomaly, budget). |
| `app/services/` | Inference and business logic. |
| `app/models/` | Model loading and artifact registry. |
| `app/schemas/` | Pydantic request and response models. |
| `app/core/` | Settings and startup wiring. |

---

## tests/

Pytest coverage for the serving API (health, PFP, forecast, anomaly, budget). See `conftest.py` for shared fixtures. Run with `pytest`.

---

## models/

Canonical home for **new-scope** final model artifacts, each with `metadata.json`
(schema in `models/README.md`).

| Path | Purpose |
| :--- | :--- |
| `models/README.md` | Artifact layout + `metadata.json` schema. |
| `models/pfp/` | PFP classifier final artifact + metadata. |
| `models/forecaster/` | Spending forecaster v1 artifact + metadata (12-month Gaussian-noise corpus). |
| `models/forecaster_v2/` | Spending forecaster v2 artifact + metadata (HFCE-calibrated 36-month corpus). |
| `models/anomaly/` | Anomaly detector final artifact + metadata. |
| `models/budget/` | Budget optimizer config + metadata. |

## training/

The model development pipeline. Large generated artifacts are gitignored; scripts, docs, and evaluation reports are committed.

| Path | Purpose |
| :--- | :--- |
| `training/scripts/` | Collector, preprocessor, feature engineering, and training scripts. |
| `training/docs/` | ML design documents (data collection, EDA, dimension discovery). |
| `training/docs/pen-and-paper-algorithms.md` | Hand-executable serve-time algorithms: PFP, forecast v2 SARIMA, IQR alerts, budget LP. |
| `training/docs/pen-and-paper-simulation.md` | End-to-end pipeline simulation from a FIES row through one persona. |
| `training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md` | FIES-anchored, HFCE-calibrated temporal disaggregation methodology. |
| `training/docs/data-collection/synthetic-generation-v2.md` | Synthetic Generation v2 operator runbook (parallel pipeline; v1 untouched). |
| `training/docs/data-collection/survey-only-forecast-v3.md` | Survey-only v3 selection, pseudonymization, allocation, and release contract. |
| `docs/superpowers/plans/2026-09-17-synthetic-generation-v2.md` | Implementation plan for Synthetic Generation v2. |
| `training/docs/phases/` | Model development runbooks: Phase 7 evaluation, 8 selection/versioning, 9 deployment, 10 monitoring. |
| `training/figures/` | EDA plots and `eda_report.md`. |
| `training/datasets/` | Processed/engineered feature matrices (gitignored). |
| `training/synth/` | Generated personas and transactions (gitignored). |
| `synth_v2/` | Synthetic Generation v2 outputs — personas, transactions, `synthesis_report.json` (gitignored). |

## docs/

| Path | Purpose |
| :--- | :--- |
| `docs/README.md` | Clarifies the split between `docs/standards/`, `docs/models/`, and `training/docs/`. |
| `docs/models/TEAMMATE-GUIDE.md` | Trusted path for human teammates and AI agents: scope, runbooks, verification. |
| `docs/models/forecaster.md` | Plain-language serving explanation; v1 evaluation figures. |
| `docs/models/forecaster-v2.md` | HFCE-calibrated v2 training note: empirical backing, SARIMA vs ARIMA, 2026-09-17 metrics. |
| `docs/models/forecaster-v3.md` | Unserved survey-only v3 forecast pipeline and external-validation release gate. |
| `docs/models/anomaly-alerts-v2.md` | V2 personalized dual-channel anomaly-alert methodology: transaction and forecast-residual IQR baselines from user history. |
| `docs/models/classification-v2.md` | Deterministic financial classification rules, service contract, input sources, and literature provenance. |
| `docs/models/model-candidate-roster.md` | RRL-grounded candidate algorithms per model family. |
| `docs/standards/REPOSITORY-STANDARDS.md` | Enforceable Python/ML engineering standards. |
| `docs/standards/git-commit-standards.md` | Git commit message format and scopes. |
| `docs/standards/documentation-format.md` | Shared documentation formatting rules. |
| `docs/standards/archive/` | Deprecated frontend/design standards (for the `odin` app). |

---

## Cross-References

| Task | Use |
| :--- | :--- |
| Model design documents (MDD, feature sets) | `../Odin-Paper/docs/ml/` |
| System specification and PRD | `../Odin-Paper/docs/requirements-engineering/` |
| RRL corpus, scoring, and pipeline | **Odin-Literature** |
