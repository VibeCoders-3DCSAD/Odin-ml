# Model Candidate Roster (Rev 1, 2026-09-07)

Candidate algorithms for each of the four BUDI ML model families, grounded in the
RRL corpus processed in `../Odin-Literature/`. This roster is the **scope frame** for the
revised model development pipeline (phases 7–10); the pre-registered decision rules and
metrics for the previous training run live in the `training/scripts/train_*.py` docstrings
and in `evaluation.json` files.

Sources cited as `B<n> <paper> (<Year>)` refer to `Odin-Literature/docs/standards/batch-<n>-algorithm-screening.md`.

## Roster at a glance

| Family | Candidate algorithms | Evidence (RRL) | Current tier | Cost |
|---|---|---|---|---|
| **PFP Classification** | Naive Bayes; Logistic Regression; Random Forest; SVM (RBF); XGBoost; (CatBoost, TabTransformer — review) | Apus 2023 (B1); Pandiin & Matias 2025 (B4); Thakur & Jadhav 2025 (B5); Zhang & Hou 2026 (B6) | T0 majority / T1 rules / T2 LR / T3 RF+SVM / T4 XGB | Low–Med |
| **Financial Forecasting** | ARIMA family; Random Forest regressor; LSTM; GRU; BiLSTM; hybrid | Yunita et al. 2025 (B5); Ghonaim & El-Sharawy 2025 (B2); Kontopoulou 2023 (B3); Kara & Senguler 2025 (B3) | T2 RF / T3a LSTM / T3b GRU / T3c BiLSTM | Med–High |
| **Anomaly Detection** | IQR (statistical); Isolation Forest; One-Class SVM; Autoencoder; hybrid ensemble | Zhong 2025 (B6); Zhang & Duan 2025 (B6); Zhang et al. 2023 17-algo benchmark (B6); Vijayanand & Smrithy 2025 (B5) | T0 majority / T1 IQR / T2 IF+OCSVM+AE / T3 hybrid | Med |
| **Budget Optimization** | Constraint LP (scipy.linprog); multi-criteria; two-stage/mental-accounting; (RNN-driver — forecasting-adjacent) | Gulbakyt et al. 2025 (B2); Lu et al. 2025 (B3); Pretnar et al. 2025 (B4); de Zarza et al. 2024 (B2) | scipy LP (v1) | Low |

---

## 1. PFP Classification

**Task:** assign a personal financial profile from recorded income, expenses, obligations,
and behavioral features. Pre-registered family-level decision rule: a learned model must beat
the Tier-1 rule-based floor by ≥ 2 points Macro-F1 (5-fold expanding window).

| Candidate | RRL evidence | Why it fits | Verdict |
|---|---|---|---|
| Naive Bayes | Apus et al. (2023) predicts Filipino household income (B1, Maintain) | PH-domain precedent; cheap baseline | Baseline (T1/T2 candidate) |
| Logistic Regression | Pandiin & Matias (2025) LR/RF/SVM loan-eligibility comparison (B4, Maintain/crucial) | Strong interpretable baseline; matches current T2 | Shortlist (current T2) |
| Random Forest | Pandiin & Matias (2025) (B4); Thakur & Jadhav (2025) expense tracker (B5) | Robust to tabular features; current T3 | Shortlist (current T3) |
| SVM (RBF) | Pandiin & Matias (2025) (B4) | Current T3 winner; strong on engineered features | Shortlist (current T3 winner) |
| XGBoost | Thakur & Jadhav (2025): XGBoost best of MLP/XGB/SVM/bagging/boosting (B5); Zhang & Hou (2026) LR/SVM/RF/XGBoost (B6) | Literature favors gradient boosting on tabular finance data | Shortlist (current T4) |
| CatBoost | Salvador (2024) wealth-quintile boosting (B5, Maintain) | Boosting variant; only if XGB underperforms | Hold (revisit if boosted path) |
| TabTransformer | Hartomo et al. (2025) weighted-loss credit risk + XAI (B2, Maintain/Supporting) | Deep tabular + explainability; higher cost | Research only (not in v1 scope) |

---

## 2. Financial Forecasting

**Task:** estimate future total/category-level expenses from chronological transaction history.
Pre-registered family-level decision rule: a model must beat the naive baseline by ≥ 20% MAPE
reduction. Primary metrics: MAE, SMAPE, MDA, RMSE.

| Candidate | RRL evidence | Why it fits | Verdict |
|---|---|---|---|
| ARIMA family | Kontopoulou (2023) ARIMA-vs-ML forecasting (B3, crucial); Tjostheim (2025) statistical-vs-ML review (B5) | Statistical baseline; useful sanity floor | Baseline (compare against) |
| Random Forest regressor | Current T2 (selected previous winner) | Strong tabular performance on monthly aggregates | Shortlist (current T2) |
| LSTM | Yunita et al. (2025) RNN/LSTM/GRU/hybrid (B5, Maintain); Ghonaim & El-Sharawy (2025) RNN budget app (B2) | Day-level sequence modeling; current T3a | Shortlist (current T3a) |
| GRU | Yunita et al. (2025) (B5); current T3b | Cheaper recurrent alternative; current T3b has an artifact | Shortlist (current T3b) |
| BiLSTM | Yunita et al. (2025) (B5); Ciric (2023) multi-output LSTM (B2, review) | Bidirectional context; current T3c | Hold (heavy) |
| Hybrid (e.g., IF+ARIMA/LSTM) | Shuryhin & Zinovatna (2024) financial-advice recommender (B5, supporting) | If single models underperform | Research only |

---

## 3. Anomaly Detection

**Task:** flag transactions deviating from the user's baseline.
Pre-registered family-level decision rule (Option A, revised 2026.09.10 — see
`docs/thesis/anomaly-decision-rule-rationale.md`): the winner must reach PR-AUC ≥ 1.5× the
IQR baseline and PR-AUC ≥ 0.15 on the held-out test split, else fall back to IQR. Primary
metric: PR-AUC (imbalance-safe). Secondary: Accuracy, Precision, Recall, F1 at the F2
operating point (+ ROC-AUC).

| Candidate | RRL evidence | Why it fits | Verdict |
|---|---|---|---|
| IQR (statistical) | Baseline floor; per-feature | Cheap, interpretable | Baseline (T1) |
| Isolation Forest | Shuryhin & Zinovatna (2024) IF in finance recommender (B5); background in Zhang et al. (2023) 17-algo benchmark | Standard, efficient; current T2 | Shortlist (current T2) |
| One-Class SVM | Zhang et al. (2023) experimental evaluation incl. OCSVM (B6) | Kernel alternative; current T2 | Shortlist (current T2) |
| Autoencoder | Zhang & Duan (2025) self-supervised accounting anomaly + prediction (B6, crucial); deep-anomaly survey (Wang F. 2025, B5) | Sequence reconstruction; current T2 (PyTorch) | Shortlist (current T2) |
| Adaptive threshold calibration | Zhong (2025) financial-data-quality adaptive thresholds (B6, crucial) | Direct improvement over static thresholds; cheap | Shortlist (next iteration) |
| Hybrid ensemble | Current T3 (voting T1–T2) | Robustness | Shortlist (current T3) |

---

## 4. Budget Optimization

**Task:** propose budget allocations respecting obligations, constraints, and preferences.
Budget metrics: Constraint Satisfaction Rate, Budget Utilization Rate, Deviation from User Preferences.

| Candidate | RRL evidence | Why it fits | Verdict |
|---|---|---|---|
| Constraint optimization (scipy.linprog) | Lu et al. (2025) constrained data-driven budgeting framework (B3, crucial) | Matches current implementation; exact, fast | Shortlist (current v1) |
| Multi-criteria optimization | Gulbakyt et al. (2025) dynamic budget allocation via multi-criteria optimization (B2, crucial) | Natural next step for preference weighting | Shortlist (v2 candidate) |
| Two-stage / mental accounting | Pretnar et al. (2025) two-stage budgeting under bounded rationality (B4, supporting) | Aligns with the app's obligation-vs-savings framing | Research only |
| LLM-driven personal budgeting | de Zarza et al. (2024) LLM budgeting optimization (B2, supporting) | Not in v1 scope (offline-first, no token API dependency) | Out of scope for v1 |

---

## Open revision items

1. **CatBoost / TabTransformer** — revisit only if the boosted path underperforms or explainability
   becomes a hard requirement. Cost tags above are relative to the thesis timeline.
2. **Anomaly adaptive thresholds (Zhong 2025)** — cheap, high-value; schedule as a follow-up
   iteration to the current T1–T3 run.
3. **Budget v2** — multi-criteria allocation (Gulbakyt 2025) is the recommended next candidate
   after the current LP implementation is validated end-to-end.
4. Reconcile with `Odin-Paper/docs/ml/TODO.md` Task 4 (ground candidates in literature + cost
   tags) — this roster is the Odin-ML-side artifact for that task.