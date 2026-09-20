# Spending Forecaster v2 — HFCE-Calibrated Training

```json
{
  "document-type": "model-evaluation",
  "version": "1.1.0",
  "date": "2026.09.20",
  "authors": ["Group 4, III-DCSAD"]
}
```

> **What this is:** Companion to [`forecaster.md`](forecaster.md) for the Synthetic Generation v2 corpus. It records the 2026-09-17 walk-forward run, what is empirically backed in the training data, and what the v2 SARIMA artifact actually is. Serving behaviour is unchanged and is still described in `forecaster.md`.

---

## Purpose

v1's within-year expense path used `Normal(1.0, 0.15)` noise with 0.5×–2.0× caps. That variation had no Philippine calendar behind it. Synthetic Generation v2 replaces that noise with FIES-anchored annual amounts and PSA quarterly HFCE weights.

This note answers three questions that the v1 forecaster write-up cannot:

1. Did the v2 training run pass the pre-registered decision rule?
2. Is the saved winner a seasonal SARIMA, or ARIMA under a SARIMA name?
3. What, exactly, is empirically backed in the v2 series — and what is still assumed?

It does **not** claim that the HFCE methodology beat v1 in a controlled bake-off. Horizon, fold count, and expense-generating process all changed together.

---

## Relation to v1

| Item | v1 (`models/forecaster/`) | v2 (`models/forecaster_v2/`) |
| :--- | :--- | :--- |
| Expense months | Gaussian noise around a flat monthly mean | HFCE quarterly weights, equal thirds inside each quarter |
| Series length | 12 months (seasonal term never identified) | 36 months (`2023-01`–`2025-12`), chosen so SARIMA with `s=12` can fit |
| Walk-forward folds | 5 | 29 |
| Seasonal ARIMA gate | Never reached (`< 24` pooled months) | On for folds 19–29 and the final artifact |
| Naive MAPE | 37.45% | 36.48% |
| Winner MAPE | 9.40% (label `tier3_sarima`, fit was plain ARIMA) | 6.12% (label `tier3_sarima`, fit is SARIMA) |
| MAPE reduction vs naive | 74.9% | 83.2% |

The serving explanation, feature list, and inference path in [`forecaster.md`](forecaster.md) still apply. Only the training corpus, fold file, and committed artifact directory differ.

---

## Empirical backing

The data methodology is **FIES-anchored, HFCE-calibrated temporal disaggregation with proportional benchmarking**. Full derivation: [`training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md`](../../training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md). Operator path: [`training/docs/data-collection/synthetic-generation-v2.md`](../../training/docs/data-collection/synthetic-generation-v2.md).

### Observed

| Component | Source |
| :--- | :--- |
| Annual household-category magnitude | 2023 FIES, via FIES-calibrated personas (not a 1:1 `SEQ_NO` replay) |
| Within-year **quarter** shape, by category | Original artifact: PSA 2023 HFCE by purpose, constant 2018 prices; next corpus: calendar-year current-price profiles |
| Q4 national spike (food, residual `other`) | Same HFCE series |

HFCE is a **population-level** temporal calibration source. Every persona receives the same quarterly share of its own annual amount. That is intentional: the generator is not claiming that each household's calendar matches the national accounts.

### Persona archetypes

v2 does not invent a new persona generator. It reuses v1's 12 archetypes (A–L) from `generate_all_personas`. Magnitudes come from 2023 FIES NCR; behavioral and attitudinal patterns are informed by the BSP 2021 Consumer Finance Survey. The archetype roster was reviewed by a **general-finance subject-matter expert** (Asst. Prof. Pamela A. Go, College of Business and Financial Science) against the SME draft in Odin-Paper (`docs/ml/1_problem-statement/persona-validation-list-SME-draft.md`). Numeric PFP cutoffs (income CV, obligation ratio, runway) remain researcher-defined pending threshold discovery; expert review of the roster is not the same as a signed calibration of those cutoffs.

### Explicit modeling assumptions (not observed)

| Component | Status |
| :--- | :--- |
| Month inside a quarter | Equal thirds (`Jan = Feb = Mar = Q1 / 3`) |
| Years after 2023 | Original artifact replays the 2023 profile. The next corpus uses configured 2024-2025 current-price profiles and 2026 Q1-Q2 only; FIES remains the 2023 household anchor. |
| Income | Still the v1 income path; HFCE is expenditure-only |
| Intra-month dates and splits | Synthetic |
| Random Gaussian expense noise | **Removed** |

The committed artifact described here was trained before the year-specific current-price update. The next training corpus spans `2023-01` through `2026-06`, using 2023-2025 full-year profiles and 2026 Q1-Q2 only; it requires a separate evaluation and artifact-release run. Generated months remain synthetic allocations calibrated to observed annual FIES magnitudes and observed national quarterly consumption, not reconstructed FIES household monthly histories.

---

## Why 36 months

The SARIMA variant uses `seasonal_order=(1, 0, 0, 12)`. A seasonal autoregressive lag of 12 months is unidentified with fewer than 24 monthly observations, so `train_forecaster.py` disables the seasonal term until the pooled series has at least 24 months.

v1 was 12 months long; that gate never opened and the saved `tier3_sarima` fit was plain ARIMA. The v2 training corpus is therefore **36 months** (three copies of the 2023 HFCE year, labelled `2023-01`–`2025-12`): 24 months is the identification floor for `s=12`; the third year supplies walk-forward test folds after the seasonal term is on. Those later years still wrap 2023 HFCE because 2024 and 2025 FIES public-use files remain locked by the PSA — they are calendar labels, not additional observed vintages.

---

## Model: ARIMA vs SARIMA

Both tiers share the same pooled, user-normalized monthly series and the same non-seasonal order `ARIMA(1, 1, 0)`. The SARIMA variant adds `seasonal_order=(1, 0, 0, 12)` **only when the pooled series has at least 24 months**. Below that gate it falls back to plain ARIMA. The gate lives in `training/scripts/train_forecaster.py` (`forecast_sarima_pool` and the final `save_models` refit).

On this 36-month corpus:

| Stage | Pooled months | What `tier3_sarima` actually fit |
| :--- | :---: | :--- |
| Walk-forward folds 1–18 | 6–23 | Plain ARIMA (metrics identical to `tier3_arima`) |
| Walk-forward folds 19–29 | 24–34 | SARIMA `(1,1,0)(1,0,0,12)` |
| Final artifact | 35 observations | SARIMA |

The saved `models/forecaster_v2/tier3_sarima.joblib` is a `SARIMAXResultsWrapper` with `kind: sarima`. Fitted seasonal AR at lag 12 is approximately **0.998**; the non-seasonal AR coefficient is approximately **0**. That is the statistical signature of a repeating annual cycle after first differencing — which is how v2 generates 2024 and 2025 (the 2023 HFCE year, copied).

Statsmodels may emit `EstimationWarning: Non-stationary starting seasonal autoregressive`. That warning is a start-parameter issue: the seasonal AR wants to sit on the stationarity boundary, so the optimiser starts at zero and still converges. It is not a failed fit.

The largest fold-level gap after the gate turns on is fold 27 (test `2025-10`): ARIMA MAPE 15.46% versus SARIMA MAPE 4.07%. October is a high-spend HFCE month; without a seasonal term the model cannot copy the previous October.

---

## Evaluation results

**Run (2026-09-17):**

```bash
python training/scripts/train_forecaster.py \
  --input training/datasets/forecaster_v2/ \
  --output models/forecaster_v2/ \
  --folds training/datasets/processed_v2/temporal_folds.json \
  --skip-torch --skip-rf
```

Random Forest and GRU were not trained on this run. The comparison is naive baseline versus pooled ARIMA versus pooled SARIMA.

**Decision rule (unchanged):** the winning learned model must cut naive MAPE by at least 20%, else retain naive.

| Tier | MAPE (mean ± std) | Result |
| :--- | :--- | :--- |
| Naive baseline | 36.48% ± 2.15% | Floor |
| `tier3_arima` | 6.60% ± 5.30% | Passes the rule |
| **`tier3_sarima`** | **6.12% ± 5.07%** | **Winner** (83.2% MAPE reduction) |

Secondary metrics for the winner: MAE 2194.51 ± 1640.58, SMAPE 6.23 ± 4.84, MDA 0.32 ± 0.24, RMSE 4638.37 ± 1233.96, R² 0.83 ± 0.10. Source of truth: [`models/forecaster_v2/evaluation.json`](../../models/forecaster_v2/evaluation.json) and [`evaluation_report.md`](../../models/forecaster_v2/evaluation_report.md).

The run **passed**. The committed artifact is real SARIMA, not a renamed ARIMA.

---

## What this does not claim

- **Not a controlled v1-versus-v2 model bake-off.** Lower MAPE is expected once a deterministic annual HFCE cycle is injected and the series is long enough for `s=12` to exist. Horizon (12 → 36 months) and fold count (5 → 29) changed at the same time as the generator.
- **Not household-level monthly truth.** Equal-thirds months and a shared national quarter shape remain assumptions.
- **The committed artifact does not contain year-specific seasonality after 2023.** Its 2024-2025 inputs replay the 2023 HFCE calendar. The pending 42-month corpus corrects this using supplied current-price profiles, but no replacement artifact is claimed until it is retrained and evaluated.
- **Not a full-tier bake-off.** `--skip-rf` and `--skip-torch` excluded Random Forest and GRU from this run.

The defensible claim is narrower: v2 training succeeded on an empirically calibrated quarterly pattern; the seasonal term now has something real to fit; the v1 Gaussian expense path is no longer the within-year engine.

---

## Artifacts

| Path | Role |
| :--- | :--- |
| `models/forecaster_v2/tier3_sarima.joblib` | Winner artifact (`kind: sarima`) |
| `models/forecaster_v2/evaluation.json` | Fold metrics and winner contract |
| `models/forecaster_v2/evaluation_report.md` | Human-readable report |
| `models/forecaster_v2/metadata.json` | Provenance and serve-time winner fields |
| `training/datasets/forecaster_v2/` | Feature matrices for this run (gitignored) |
| `training/datasets/processed_v2/temporal_folds.json` | 29-fold expanding window |

v1 artifacts remain in `models/forecaster/` and are not overwritten by this pipeline.
