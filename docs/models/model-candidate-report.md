# Model Candidate Report

```json
{
  "document-type": "model-candidate-report",
  "version": "1.0.0",
  "date": "2026.09.07",
  "authors": ["Group 4, III-DCSAD"]
}
```

---

## 1. Purpose and Scope

This report pre-registers the candidate models for the revised BUDI ML scope. It is
grounded exclusively in the RRL corpus processed in `../Odin-Literature/` and in the
current serving stack in this repository. It supersedes the evenly-weighted round of
candidate screening (Rev-1 roster, `../Odin-Literature/docs/standards/batch-<n>-algorithm-screening.md`)
by introducing an explicit **ascending-weight ordering** as the organizing axis for each
intelligent module.

Two modules are in **core scope**:

1. **Budget Optimization / Recommendation / Allocation engine** — propose budget
   allocations that respect obligations, constraints, and user preferences.
2. **Financial Forecasting** — estimate future income and expenditure from historical
   transaction data.

Two modules are **extended scope**, included because the RRL corpus supports a candidate
roster for each:

3. **Anomaly Detection** — flag transactions that deviate from a user's baseline.
4. **PFP (Profile) Classification** — assign a personal financial profile from recorded
   income, expenses, obligations, and behavioral features.

Modules 3 and 4 are only de-scoped if the Phase 7 evaluation shows the extended families
cannot reach their pre-registered acceptance rules; the evidence collected in
`batch-<n>-algorithm-screening.md` (B1–B6) supports retaining both.

## 2. How This Report Was Built

Sources used:

| Source | Role here |
| :--- | :--- |
| `Odin-Literature/docs/standards/batch-1..6-algorithm-screening.md` | RRL evidence (cited as `B<n>` below) for which models the related literature actually uses |
| `Odin-Literature/config/modules.yaml` | Module ground truth: `budget_recommendation`, `forecasting`, `anomaly_detection`, `financial_profile_classification` |
| `Odin-Literature/literature/conversions/batch-7/` | Canonical stems for the intake papers (evidence papers map to these stems) |
| Rev-1 roster `docs/models/model-candidate-roster.md` | Prior screening verdicts and tier assignments |
| `app/services/` + `training/scripts/train_*.py` | Current implementation and tier stack this report constrains |

## 3. Weight-Ordering Principle

Chapter-1 scope language requires candidacy to be assessed on "evaluation metrics,
computational requirements, interpretability, and suitability for integration"
(Phase 8 selection chain: **primary metric → interpretability → artifact size /
inference cost → integration effort**). This report materializes the *artifact size /
inference cost* leg as an **ascending-weight roster**: for every module the candidates
are listed from **lightest** (fewest learned parameters, smallest artifact, cheapest
inference) to **heaviest**.

Weights below are **relative classes**, not absolute byte counts — BUDI is an
offline-first, mobile-scale service and deep sequence models are the heaviest tier.
Exact parameter counts, artifact sizes, and inference latencies are measured in
Phase 7 and recorded in each `models/<family>/metadata.json`.

## 4. Module 1 — Budget Optimization / Recommendation / Allocation

**Task:** propose budget allocations respecting obligations, constraints, and preferences.

**Metrics (Budget Optimizer MDD v1.0):** Constraint Satisfaction Rate, Budget
Utilization Rate, Deviation from User Preferences.

Candidates are optimization formulations rather than parameterized networks; "weight"
here means solver/runtime cost and interface complexity, ascending lightest → heaviest.

| Rank | Candidate | Weight class | RRL evidence | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Rule-based / heuristic allocation (envelope, 50-30-20, payday) | ~0 params, no solver | Alenazi & Sas 2023 budgeting-app study (B1, `A--Alenazi-2023`); mental-accounting framing in Pretnar et al. 2025 (B4, `A--Pretnar-2025`) | Baseline (fallback) |
| 2 | Constraint LP via `scipy.linprog` | ~0 learned params, single exact solve | Lu et al. 2025 constrained data-driven budgeting framework (B3, crucial, `A--Lu-2025`) | **Shortlist (current v1)** |
| 3 | Multi-criteria optimization (preference-weight allocation) | small preference-vector weights | Gulbakyt et al. 2025 dynamic budget allocation via multi-criteria optimization (B2, crucial, `A--Gulbakyt-2025`) | Shortlist (v2 candidate) |
| 4 | Two-stage / mental-accounting behavioral model | calibration weights only | Pretnar et al. 2025 two-stage budgeting under bounded rationality (B4, supporting, `A--Pretnar-2025`) | Research only |
| 5 | LLM-driven personal budgeting | very heavy (LLM) | de Zarza et al. (LLM budgeting optimization, B2, supporting; `A--DeZarza-2023` intake, roster cites 2024) | Out of scope for v1 (offline-first, no token API) |

Decision rule: the LP must meet all hard constraints (Constraint Satisfaction Rate =
1.0) with minimal deviation from the user's target ratios; `REDUCED`/`INFEASIBLE`
fallbacks are explicitly modeled in `budget_service.py`. Multi-criteria v2 becomes the
winner only if it exceeds the LP's deviation-from-preferences KPI without violating
constraints.

## 5. Module 2 — Financial Forecasting

**Task:** estimate future total/category-level expenses from chronological transaction history.

**Metrics (Phase 7):** MAE, SMAPE, MDA, RMSE.

**Decision rule (pre-registered):** a model must beat the naive baseline by **≥ 20% MAPE
reduction**, evaluated on a 5-fold expanding window (`training/datasets/processed/temporal_folds.json`).

| Rank | Candidate | Weight class | RRL evidence | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Naive / trailing-mean baseline (FO-02 cold start) | 0 params | Kara & Senguler 2025 budget-forecasting SLR (B3, crucial, `A--KaraSenguler-2025`); Tjostheim 2025 statistical-vs-ML review (B5, `A--Tjostheim-2025` if intaked) | Baseline |
| 2 | ARIMA / SARIMA (statistical, small `p,d,q`) | small | Kontopoulou 2023 ARIMA-vs-ML forecasting (B3, crucial); Tjostheim 2025 (B5) | Baseline (sanity floor) / shortlist if lightweight |
| 3 | Random Forest regressor | medium (tree ensemble) | Current T2 (selected previous winner); Yunita et al. 2025 NN comparison incl. ML baselines (B5, `A--Yunita-2025` if intaked) | Shortlist (current T2) |
| 4 | GRU | light recurrent (3 gates) | Yunita et al. 2025 (B5, if intaked); Ghonaim & El-Sharawy 2025 RNN budget app (B2, crucial, `A--Ghonaim-2025`) | Shortlist (current T3b) |
| 5 | LSTM | medium recurrent (4 gates) | Yunita et al. 2025 (B5); Ghonaim & El-Sharawy 2025 (B2); Ciric 2023 multi-output LSTM (B2, `A--Ciric-2023` if intaked) | Shortlist (current T3a) |
| 6 | BiLSTM | heavy recurrent (~2× LSTM) | Yunita et al. 2025 bidirectional analysis (B5) | Hold (current T3c; heavy) |
| 7 | Hybrid ensembles / sequence-hybrid | heaviest | Shuryhin & Zinovatna 2024 financial-advice recommender, IF + ARIMA/LSTM (B5, `A--Shuryhin-2024`) | Research only |

Rationale: the RRL's forecasting family is dominated by **statistical baselines
(ARIMA) + Random Forest + RNN family (LSTM/GRU/BiLSTM) + hybrids** — exactly the tier
stack already implemented in `train_forecaster.py`. Development proceeds by ascending
weight: naive → ARIMA → RF → GRU → LSTM → BiLSTM, adding the heavier tier only when the
lighter tier fails the MAPE-reduction rule.

## 6. Module 3 — Anomaly Detection (extended candidate)

**Task:** flag transactions deviating from the user's baseline.

**Metrics (Phase 7, revised):** PR-AUC (primary, imbalance-safe); Accuracy, Precision,
Recall, F1 at the F2/val operating point (+ ROC-AUC).

**Decision rule (pre-registered, Option A — revised 2026.09.10):** the winner must reach
**PR-AUC ≥ 1.5× the IQR baseline** and **PR-AUC ≥ 0.15** on the held-out test split;
otherwise fall back to IQR (see `docs/thesis/anomaly-decision-rule-rationale.md`).

| Rank | Candidate | Weight class | RRL evidence | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| 1 | IQR (statistical, per-feature) | 0 params | Baseline floor (current serving fallback) | Baseline (T1) |
| 2 | Adaptive threshold calibration | ~0 params, simple transform | Zhong 2025 adaptive financial data-quality thresholds (B6, crucial, `A--Zhong-2025`) | Shortlist (next iteration; cheap, high-value) |
| 3 | Isolation Forest | small tree ensemble | Shuryhin & Zinovatna 2024 IF in finance recommender (B5, `A--Shuryhin-2024`); Zhang et al. 2023 17-algo benchmark (B6, `A--ZhangEtAl-2023`) | Shortlist (current T2) |
| 4 | One-Class SVM | kernel / support vectors | Zhang et al. 2023 experimental benchmark incl. OCSVM (B6, `A--ZhangEtAl-2023`) | Shortlist (current T2) |
| 5 | Autoencoder (sequence reconstruction, PyTorch) | medium (dense encoder) | Zhang & Duan 2025 self-supervised accounting anomaly + prediction (B6, crucial); deep-anomaly survey Wang F. 2025 (B5) | Shortlist (current T2) |
| 6 | Hybrid ensemble (voting T1–T3) | largest (n models) | Zhang et al. 2023 taxonomy/benchmark (B6) | Shortlist (current T3) |

Development aligns with the RRL: the anomaly family in the corpus is a
**statistical baseline → tree/kernel one-class → deep reconstruction → ensemble**
progression, matched 1:1 by the current tier stack in `train_anomaly.py`.

## 7. Module 4 — PFP / Profile Classification (extended candidate)

**Task:** assign a personal financial profile from recorded income, expenses,
obligations, and behavioral features.

**Metrics (Phase 7):** Macro-F1 (primary), Accuracy.

**Decision rule (pre-registered):** a learned model must beat the Tier-1 rule-based
floor by **≥ 0.02 Macro-F1** (5-fold expanding window).

| Rank | Candidate | Weight class | RRL evidence | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Rule-based tier (deterministic thresholds) | 0 params | Dimension & threshold discovery (`training/docs/dimension-threshold-discovery/`) | Baseline (T1) |
| 2 | Naive Bayes | lightest learned (class priors + per-feature moments) | Apus et al. 2023 Filipino household income, NB (B1, local, `L--Apus-2023` if intaked); Pandiin & Matias 2025 LR/RF/SVM loan eligibility (B4, crucial, local) | Baseline (T1/T2 candidate) |
| 3 | Logistic Regression | light (n_features + 1) | Pandiin & Matias 2025 (B4); Zhang & Hou 2026 LR/SVM/RF/XGB comparison (B6, `A--ZhangHou-2026`) | Shortlist (current T2) |
| 4 | SVM (RBF) | support-vector set | Pandiin & Matias 2025 (B4) | Shortlist (current T3 winner, calibrated) |
| 5 | Random Forest | tree ensemble | Pandiin & Matias 2025 (B4); Thakur & Jadhav 2025 expense-tracker ML (B5, `A--Thakur-2025`) | Shortlist (current T3) |
| 6 | XGBoost | boosted trees | Thakur & Jadhav 2025: XGBoost best of MLP/XGB/SVM/bagging/boosting (B5); Zhang & Hou 2026 (B6) | Shortlist (current T4) |
| 7 | CatBoost | boosted trees (ordered) | Salvador 2024 PH wealth-quintile boosting, CatBoost-based (B5, local, `L--Salvador-2024`) | Hold (boosted path follow-up) |
| 8 | TabTransformer | heavy deep-tabular | Hartomo et al. 2025 weighted-loss credit risk + XAI (B2, `A--Hartomo-2025` if intaked) | Research only |

The RRL's classification family is **rule floor → NB/LR → SVM/RF → gradient boosting
(XGB/CatBoost) → deep tabular (research)**. The development path is ascending weight:
NB and LR as the cheapest learned floor, then SVM/RF, then XGBoost only if a lighter
tier fails the Macro-F1 rule.

## 8. RRL Alignment Matrix

| Module | Candidate | Evidence paper(s) | B-number | Weight |
| :--- | :--- | :--- | :--- | :--- |
| Budget | Rule-based | Alenazi & Sas 2023; Pretnar et al. 2025 | B1, B4 | ~0 |
| Budget | LP (`linprog`) | Lu et al. 2025 | B3 (crucial) | ~0 |
| Budget | Multi-criteria | Gulbakyt et al. 2025 | B2 (crucial) | small |
| Budget | Two-stage | Pretnar et al. 2025 | B4 | small |
| Budget | LLM-driven | de Zarza et al. | B2 (supporting) | very heavy |
| Forecasting | Naive | Kara & Senguler 2025 | B3 (crucial) | 0 |
| Forecasting | ARIMA | Kontopoulou 2023; Tjostheim 2025 | B3, B5 | small |
| Forecasting | RF regressor | Yunita et al. 2025; current T2 | B5 | medium |
| Forecasting | GRU | Yunita et al. 2025; Ghonaim & El-Sharawy 2025 | B5, B2 (crucial) | light |
| Forecasting | LSTM | Yunita et al. 2025; Ghonaim & El-Sharawy 2025; Ciric 2023 | B5, B2, B2 | medium |
| Forecasting | BiLSTM | Yunita et al. 2025 | B5 | heavy |
| Forecasting | Hybrid | Shuryhin & Zinovatna 2024 | B5 | heaviest |
| Anomaly | IQR | Baseline floor | — | 0 |
| Anomaly | Adaptive threshold | Zhong 2025 | B6 (crucial) | ~0 |
| Anomaly | Isolation Forest | Shuryhin & Zinovatna 2024; Zhang et al. 2023 | B5, B6 | small |
| Anomaly | One-Class SVM | Zhang et al. 2023 | B6 | small |
| Anomaly | Autoencoder | Zhang & Duan 2025; Wang F. 2025 | B6 (crucial), B5 | medium |
| Anomaly | Hybrid ensemble | Zhang et al. 2023 | B6 | largest |
| PFP | Rule-based | dimension-threshold discovery | — | 0 |
| PFP | Naive Bayes | Apus et al. 2023; Pandiin & Matias 2025 | B1 (local), B4 | lightest |
| PFP | Logistic Regression | Pandiin & Matias 2025; Zhang & Hou 2026 | B4, B6 | light |
| PFP | SVM (RBF) | Pandiin & Matias 2025 | B4 | small |
| PFP | Random Forest | Pandiin & Matias 2025; Thakur & Jadhav 2025 | B4, B5 | medium |
| PFP | XGBoost | Thakur & Jadhav 2025; Zhang & Hou 2026 | B5, B6 | heavy |
| PFP | CatBoost | Salvador 2024 | B5 (local) | heavy |
| PFP | TabTransformer | Hartomo et al. 2025 | B2 | very heavy |

## 9. Reconciliation and Open Items

1. **Evidence refresh.** Several papers cited above as "if intaked" are part of the
   batch-7 intake (`literature/conversions/batch-7/`) and are not yet summarized or
   scored. Once summarized, run `embed.py --force` + `score.py` and re-check each
   candidate's B-number evidence line against the regenerated `scores/`.
2. **Rev-1 roster relationship.** This report is the weight-ordered companion to
   `model-candidate-roster.md` (Rev 1). The Rev-1 tier verdicts (`current T0–T4`) are
   preserved verbatim as the *current tier* column here.
3. **Budget v2.** Gulbakyt (B2) is the recommended next budget candidate after the LP
   is validated end-to-end (`models/budget/metadata.json`).
4. **Anomaly adaptive thresholds.** Zhong (B6) is scheduled as the cheap follow-up
   iteration to the current T1–T3 run.
5. **Odin-Paper reconciliation.** Reconcile this report with
   `../Odin-Paper/docs/ml/TODO.md` Task 4 (ground candidates in literature + cost
   tags) on the Odin-Paper side.