# Teammate Guide — Odin-ML

> For human teammates **and** AI agents (e.g. opencode) working on the BUDI ML service.
> This file is the single "trusted path" doc: it explains what is already decided,
> what is still open, and the exact commands to train/serve/verify each model family.

**Last updated:** 2026-09-08

---

## 1. Repository role

Odin-ML is the Python ML service for BUDI: FastAPI microservice (`app/`) + training
pipeline (`training/`). Sibling repos are **read-only** here: `Odin` (app), `Odin-QA` (QA),
`Odin-Paper` (thesis docs — verify against Google Drive before citing), `Odin-Literature`
(RRL corpus).

| Repo | Write | Use |
| :--- | :--- | :--- |
| Odin-ML | Yes | ML service + training. |
| Odin / Odin-QA | No | App code, QA content. Do not modify. |
| Odin-Paper | No | Thesis chapters. Ground truth is Google Drive. |
| Odin-Literature | No | RRL corpus + scoring pipeline. |

---

## 2. Scope decided (do not re-litigate)

Models are picked under a **pre-registered decision rule** per family, grounded in
`docs/models/model-candidate-roster.md` + `docs/models/model-candidate-report.md`.

| Family | Currently trained / committed | Rule |
| :--- | :--- | :--- |
| **PFP** (8-class profile) | Full light tier set on this CPU box: majority, rule-based (Tier 1), logistic, Gaussian NB, RF, calibrated SVM, xgboost | Winner must beat Tier 1 by **> 2 pts Macro-F1**, else fall back to Tier 1. Winner artifact resolved by name at serve time. |
| **Forecaster** | Pooled user-normalized **ARIMA** (`tier3_arima`, 9.40% MAPE vs 37.45% naive, −74.9%), plus **SARIMA** variant (`tier3_sarima`, degrades to plain ARIMA when pool < 24 months) | Learned model must cut MAPE **≥ 20%** vs naive, else naive. Heavy neural/RF tiers are CPU-heavy — see §8. |
| **Anomaly** | IQR baseline (Tier 1); **adaptive threshold**, Isolation Forest, One-Class SVM, Autoencoder (Tier 2); **HybridEnsemble** (Tier 3) | Winner must reach **PR-AUC ≥ 1.5× IQR** and **PR-AUC ≥ 0.15** on the held-out test split, else fall back to IQR (Option A, revised 2026.09.10 — see `docs/thesis/anomaly-decision-rule-rationale.md`). Serving compares **raw** scores vs raw threshold. |

Hybrid ensemble and adaptive threshold are **back on the table** (they were briefly
dropped; the roster/report still list them under Tier 3 / Tier 2). Do not remove them
again without updating the roster + report together.

---

## 3. Environment

- Python pinned to `3.14.4` (`.python-version`).
- All commands below assume the repo root and the local `.venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

- Use `.venv/bin/python` for everything. Torch import is slow (~60–70 s) — expect it in script startup.
- On CPU-limited hosts, scripts support toggles: `--skip-torch` (forecaster), `--skip-ae` (anomaly). See §7.

---

## 4. Data pipeline

Large generated artifacts are gitignored. To regenerate from scratch (this replaces
`training/datasets/`, `training/synth/`):

```bash
# 1. Raw FIES → HSP: collector
python training/scripts/collector.py --input <fies.csv> --output training/datasets/unprocessed/
# 2. Synthesize personas (1000/archetype) + transactions
python training/scripts/synthesizer.py --input training/datasets/unprocessed/puf.parquet \
    --output training/datasets/ --months 12
# 3. Preprocess → temporal-folds + split_metadata
python training/scripts/preprocessor.py --input training/datasets/unprocessed/puf.parquet \
    --output training/datasets/processed/ --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
# 4. Feature engineering per family
python training/scripts/feature_engineering.py \
    --input training/datasets/processed/ --synth-dir training/synth/ \
    --output training/datasets/engineered/
```

The three family datasets are:
- `training/datasets/pfp/` — alias of `engineered/` (persona-month features × 35).
- `training/datasets/forecaster/` — monthly summaries + lookback features (from the same engineered data).
- `training/datasets/anomaly/` — transaction-level feature rows (also derived/engineered).
- `training/datasets/processed/temporal_folds.json` — 5 growing-window folds shared by all families.

If a parquet exists, **you do not normally need to rebuild data**; only rebuild when you changed the generator.

**Synthetic Generation v2 (optional, parallel):** a second, HFCE-calibrated synth
pipeline is available at `training/scripts/synthesizer_v2.py` / `generate_transactions_v2.py`
/ `temporal_disaggregation.py` / `preprocessor_v2.py`. It is **parallel** to the pipeline
above — `synthesizer.py`, `generate_personas.py`, `generate_transactions.py`, and
`preprocessor.py` are unedited and remain the default. v2 writes to `synth_v2/` (never
`training/synth/`) and replaces v1's flat Gaussian monthly expense noise with a PSA HFCE
quarterly-weighted schedule. `preprocessor_v2.py` splits v2 personas + builds temporal
folds into `training/datasets/processed_v2/`, using v1's exact split/fold algorithm
(imported, read-only). See `training/docs/data-collection/synthetic-generation-v2.md`
for commands and validation.

---

## 5. Model lifecycle (new scope)

- Final artifacts live in top-level `models/<family>/` and **are committed** (with `metadata.json`).
- Intermediate checkpoints belong in `training/models/` (gitignored).
- Every training script emits:
  - `evaluation.json` — fold metrics, aggregate metrics, and the uniform **winner contract** (`winner`, `winner_artifact`, `winner_reason`, `winner_params`, `threshold`). **Runtime source of truth** for serving resolution.
  - `evaluation_report.md` — human-readable report.
  - `metadata.json` — `build_metadata` schema per `models/README.md` (model_id, family, created_at, metrics, decision_rule, framework, feature_columns, artifacts, winner contract). Re-emitted from `evaluation.json`; must not diverge from it.
  - family artifacts (`.joblib` / `.pth` / `anomaly_detector.joblib` / `budget_config.json`).
- `training/scripts/regenerate_artifacts.py` re-emits reports + metadata from an existing `evaluation.json` without retraining (all four families).

**Registry contract (app/models/registry.py):**
- `load_all()` reads each family's `evaluation.json` and resolves the **winner artifact from the winner contract** (`_resolve_pfp_artifact`, `_resolve_forecaster_artifact`, `_load_anomaly`) — no hardcoded winner names. `metadata.json` (when present) supplies the serve-time `model_id`.
- API responses report the resolved winner at request time — `model_version` (`<family>-<winner>`, e.g. `forecaster-tier3_sarima`, `pfp-tier3_svm`, `anomaly-tier1_iqr`) and pfp's `tier_used`/`model_name` — no hardcoded version strings in `app/api/`.
- pfp is **optional**: missing artifacts → `pfp=None`, PFP `STANDARD` endpoint returns 503 ("pending training"). Questionnaire mode still works.
- `is_ready` ⇔ forecaster + anomaly loaded. `/ready` lists `loaded_modules`.

---

## 6. Train a single family (CPU runtime)

All scripts bootstrap `sys.path` to the repo root and run from the repo root directory
(note the command paths are root-relative).

### 6a. PFP classifier (~5–10 min)

```bash
.venv/bin/python training/scripts/train_pfp.py \
  --input training/datasets/engineered/ \
  --output models/pfp/ \
  --temporal-folds training/datasets/processed/temporal_folds.json
```

Winner artifact is auto-selected; `evaluation.json["winner_artifact"]` names the file
the server will load.

### 6b. Forecaster — ARIMA + SARIMA (+ naive) (~1–2 min, light profile)

```bash
.venv/bin/python training/scripts/train_forecaster.py \
  --input training/datasets/forecaster/ \
  --output models/forecaster/ \
  --folds training/datasets/processed/temporal_folds.json \
  --skip-torch --skip-rf
```

Heavy tiers (GRU/RF) are evaluated separately — see §8. The default scope keeps
ARIMA/SARIMA as serving candidates because they are far lighter and beat the naive rule.

### 6c. Anomaly — IQR + adaptive + IF + OCSVM + AE + Hybrid (~80 min with AE; ~15 min without)

```bash
.venv/bin/python training/scripts/train_anomaly.py \
  --input training/datasets/anomaly/ \
  --output models/anomaly/ \
  --folds training/datasets/processed/temporal_folds.json
# CPU host w/o PyTorch (drops AE + hybrid AE member):
.venv/bin/python training/scripts/train_anomaly.py \
  --input training/datasets/anomaly/ \
  --output models/anomaly/ --skip-ae
```

Note: SARIMA and Adaptive-Threshold produce near-identical numbers to their parents on
the current 12-month data — that is expected (SARIMA cannot fit a seasonal component
below 24 pooled months; adaptive threshold trades the ~0-param baseline interpretation).

---

## 7. Reliable command tips

- Long runs: launch with `setsid nohup ... > /tmp/opencode/<family>.log 2>&1 &`, add `-u`
  for unbuffered output, then **poll the log** instead of running in the foreground (the
  shell waits on the child otherwise).
- Do not run two training jobs at once on a CPU box — they contend (ruff/pytest get
  starved too).
- `--skip-*` toggles only skip heavy tiers; the decision rule and reports still cover
  everything that ran.

---

## 8. Heavy-tier runbook (teammate GPU box — RF / GRU / LSTM / BiLSTM)

Scope decision: on the shared/thesis CPU host we run the **light tiers** (ARIMA/SARIMA,
IQR/adaptive/IF/OCSVM/AE/hybrid, all PFP models). The following remain **open syncs**:

- **Forecaster heavy:** RF (`--no-skip-rf`), GRU (`--no-skip-torch`). LSTM/BiLSTM are
  "hold" candidates — flip `for variant in ["gru"]` in `train_forecaster.py` to run them,
  they are documented, not urgent.

```bash
.venv/bin/python training/scripts/train_forecaster.py \
  --input training/datasets/forecaster/ --output models/forecaster/ \
  --folds training/datasets/processed/temporal_folds.json
# run_rf + run_gru default when toggles absent
```

The decision rule then picks the best tier (ARIMA is expected to stay winner on the
synthetic 12-month horizon; heavy tiers are the evidence trail for the thesis).

---

## 9. Verify after training

```bash
# Lint / type / test (run when no training job is active)
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy app && .venv/bin/python -m pytest
```

Known debt (pre-existing, don't "fix" without a ticket):
- `training/scripts/*`: legacy `B028/B007/B905/E731/SIM118/F841` on trainer scripts (kept to match existing style).
- `app/services/features.py`: mypy import-not-found for `feature_engineering` modules.
- `budget_service.py`: `tuple[float, None]`, `SIM114/SIM102`; pandas stubs missing.

Smoke test (started API): `GET /health` (200), `GET /ready` (expect `["pfp","forecaster","anomaly","budget"]`
after all families trained), `POST /api/v1/forecast/predict` (200 SUCCESS or FALLBACK),
`POST /api/v1/anomaly/detect` (200 with `anomalous_transactions`/`overspending_transactions`),
`POST /api/v1/pfp/classify` QUESTIONNAIRE (200; STANDARD requires pfp artifact).

Also verify the metadata:eval reading is consistent:
`models/<family>/evaluation.json` `winner` == the artifact actually loaded; run `pytest tests/`.

---

## 10. Git conventions

- Format: `<type>(<scope>): <msg>` per `docs/standards/git-commit-standards.md`.
- Scopes: `ml`, `api`, `training`, `data`, `docs`, `config`, `deps`, `tests`, `standards`.
- Commit **only** tracked, intended files. `training/datasets/`, `training/synth/`,
  `training/models/`, `training/figures/` are gitignored — never force them in.
- Artifacts in `models/` **are** committed (final + metadata only, < ~50 MB).

---

## 11. Current status (2026-09-08)

- All **four** families evaluated, verified, and committed in `models/`, each with the uniform
  winner contract + `evaluation.json` / `evaluation_report.md` / `metadata.json`:
  - `models/forecaster/` → winner `tier3_sarima.joblib` (kind `sarima`; degrades to ARIMA on pools < 24 months).
  - `models/anomaly/` → winner `tier1_iqr` (`anomaly_detector.joblib`); adaptive-threshold + hybrid tiers
    trained and evidenced in `evaluation.json`, kept under the pre-registered fallback rule.
  - `models/pfp/` → winner `tier3_svm.joblib` (`CalibratedClassifierCV`); NB tier + winner-resolution serving.
  - `models/budget/` → deterministic LP (`scipy_linprog`) evaluated over 600 synthetic personas
    (`budget_config.json`, feasibility breakdown in `evaluation.json`).
- Reports + metadata are regenerated from `evaluation.json` only — `training/scripts/regenerate_artifacts.py`
  (no retraining). Winner resolution at serve time is fully metadata-driven (`app/models/registry.py`).
- 17 tests pass (full scope: PFP STANDARD → 200, metrics set includes all three modules, plus
  SARIMA-branch and adaptive-threshold unit regressions). `mypy app` shows only pre-existing debt;
  `ruff format --check` clean for all changed files (2 unrelated legacy files remain unformatted).
- Heavy tiers (RF/GRU/LSTM/BiLSTM) remain delegated to the teammate GPU box (see §9), per roster.
- See `INDEX.md` + `README.md` for the master index; update both when this guide changes.