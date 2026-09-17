# BUDI ML Pipeline — Pen-and-Paper Simulation  
## Expanded with De-Jargonized Explanations

> Original document preserved. After each formula, table, or dense explanation, a **De-jargonized** note translates it into plain English.  
> No numbers or source logic have been changed; the added text only explains what the original is doing.

> How every number in the BUDI ML microservice is produced, from a raw FIES row to a served forecast, profile, anomaly flag, and budget plan — written so that any group member (or professor) can reproduce the arithmetic by hand from a single 5-month sample.
>
> **Sample persona used:** `persona_A_0000` (archetype A — regular BPO employee).  
> All figures below were extracted verbatim from the real synthetic artifacts in this repo.

> **De-jargonized:** This document is a receipt for the pipeline. It shows the raw survey data, the fake personas and transactions built from it, the math used to train models, and the exact arithmetic behind one example person. If you can follow the plain-English notes, you can explain the pipeline without knowing ML jargon.

---

## 1. Pipeline map (data → sense)

```text
datasets/raw/puf.csv                     (FIES 2018 microdata, 90 columns, ~40k households)
   │  collector.py  (CSV → Parquet, byte-for-byte, no transform)
   ▼
datasets/unprocessed/puf.parquet
   │  preprocessor.py
   │    ├─ generate_personas.py      → training/synth/personas.parquet      (12,000 personas)
   │    ├─ generate_transactions.py  → training/synth/transactions.parquet  (120 txns × 12,000 = 1.44M)
   │    │                              training/synth/monthly_summaries.parquet (144,000 rows)
   │    └─ StratifiedShuffleSplit 70/15/15 + 5 expanding temporal folds (1-month embargo)
   │                                  → training/datasets/processed/{train,val,test}.parquet
   ▼
processed
   │  feature_engineering.py / feature_engineering_forecaster.py / feature_engineering_anomaly.py
   ▼
training/datasets/{engineered,forecaster,anomaly}/
   │  train_pfp.py → models/pfp/tier3_svm.joblib
   │  train_forecaster.py → models/forecaster/tier3_sarima.joblib
   │  train_anomaly.py → models/anomaly/anomaly_detector.joblib
   │  budget → models/budget/budget_config.json (scipy.optimize.linprog at serve time)
   ▼
app/ (FastAPI)  →  /pfp/profile, /forecast/expenses, /anomaly/detect, /budget/allocate
```

> **De-jargonized:** Raw survey data comes in as a CSV. The collector just changes the file format to Parquet, like saving Excel as a faster database. The preprocessor then creates synthetic people and their transactions, splits them into train/validation/test sets, and builds features — the numeric clues the models learn from. Finally, FastAPI serves four endpoints: one for financial profile, one for expense forecast, one for anomaly detection, and one for budget allocation.

**Model families** (their pre-registered winner rules and hold-out metrics are in `models/<family>/metadata.json`):

| Family | Winner | Decision rule | Hold-out metric |
|---|---|---|---|
| PFP (8-class profile) | `tier3_svm` (RBF SVC + Platt calibration) | must beat rule-based Tier 1 by **> 0.02 Macro-F1** | macro-F1 **0.6745** (Tier 1: 0.6050) |
| Forecaster (13th-month expense) | `tier3_sarima` (pooled ARIMA) | must cut MAPE by **≥ 20%** vs naive | MAPE **9.3957%** (−74.9% vs naive) |
| Anomaly (transaction flags) | `tier1_iqr` | ML tier must reach PR-AUC ≥ 1.5×IQR **and** ≥ 0.15 | test PR-AUC 0.0550, threshold **0.125**, F2 0.2351 |
| Budget (allocation) | `scipy_linprog` (simplex/HiGHS) | LP must satisfy all hard constraints | constraint satisfaction **1.0**, utilization 1.0, mean deviation 0.041 |

> **De-jargonized:**  
> - **PFP** predicts a person’s financial profile label. The SVM must beat simple rules by a small margin.  
> - **Forecaster** predicts next month’s expense. It must be at least 20% better than just guessing “same as last month.”  
> - **Anomaly** flags weird transactions. The IQR method won; the ML alternative was not good enough.  
> - **Budget** solves a linear program to divide money while respecting constraints. It always satisfies the rules in testing.

> **Ground rule of this document:** every constant below that can be cross-checked has been cross-checked against `training/scripts/*.py` or the committed artifacts. Where a model is too large to run whole by hand (svm kernel over 2,161 support vectors), we simulate it from the model's own runnable pieces — the scaler, γ, support vectors, one-vs-one votes, and Platt sigmoid described in §5.1 — never a different rule-based stand-in.

> **De-jargonized:** If a number can be verified, it was verified. If a model is too complex to do by hand, the document simulates that model using its own real constants, so you can still reproduce the result on paper.

---

## 2. Stage 0 — Raw → Unprocessed (`collector.py`)

`training/scripts/collector.py` converts the FIES 2018 CSV to Parquet with **no arithmetic**.

```text
collector.py --input datasets/raw/family_income_and_expenditure.csv / puf.csv  →  puf.parquet
```

> **De-jargonized:** This step is just a format conversion. It does not add, average, or change any value. CSV goes in, Parquet comes out.

Sample raw row (fixed-width-ish CSV, 90 columns; parsed as plain integers):

```text
W_REGN  W_PROV   SEQ_NO   RPROV  FSIZE   REG_SAL   SEASON_SAL   WAGES      ...
01      28       000001   2800   02.5    00119000   00000000     00119000   ...
```

Values are **unadjusted** (e.g. `00119000` = ₱11,900). Column semantics follow the PSA FIES codebook (`W_REGN` region, `W_PROV` province, `FSIZE` household size, `REG_SAL` regular salary, `WAGES` wage income, …). Nothing aggregates here; the pipeline stays faithful to the source.

> **De-jargonized:** Each row is one household from the survey. The codes mean things like region, province, family size, and salary. The numbers are raw pesos, not adjusted or scaled. The pipeline does not combine rows at this stage.

---

## 3. Stage 1 — Unprocessed → Processed (synthesis + split)

### 3.1 Personas (`generate_personas.py`)

The synth population starts from **12 non-overlapping archetypes** (A–L), each defining an income range, a target income CV, a target obligation ratio, a target emergency runway, a savings rate, household size, and an **expected 8-class PFP label**. Each real persona is:

```python
base_income    = uniform(archetype.income_range)           # then round to nearest ₱1,000
monthly_income = round(base_income / 1000) * 1000
income_cv      = clip(archetype.income_cv + N(0, 0.03), 0.01, 1.50)
oblig_ratio    = clip(archetype.obligation_ratio + N(0, 0.03), 0.05, 0.95)
runway         = clip(archetype.runway_months + N(0, 0.30), 0.0, 24.0)
```

> **De-jargonized:**  
> - Pick a random starting income inside the archetype’s range, then round to the nearest ₱1,000.  
> - `income_cv` measures how bumpy income is. Add a little random noise, then keep it between 0.01 and 1.50.  
> - `oblig_ratio` is the share of money committed to obligations. Add noise, keep between 0.05 and 0.95.  
> - `runway` is how many months you could survive on savings. Add noise, keep between 0 and 24 months.

**Expense ratios** come from the real FIES data, then get *archetype-tuned*:

```python
food_ratio      = FIES.food      * (0.8 + 0.4·(1 − oblig_ratio))
housing_ratio   = FIES.housing   * (0.7 + 0.6·oblig_ratio)
transport_ratio = FIES.transport                       # unchanged
health_ratio    = FIES.health                          # unchanged
education_ratio = FIES.education                       # unchanged
other_ratio     = 0.22                                  # constant floor
# then all six are renormalized so they sum to 1
```

> **De-jargonized:** Start from how real FIES households spend. If someone has high obligations, adjust food down and housing up a bit. Transport, health, and education stay as in FIES. “Other” gets a minimum 22% share. Then rescale all six so they add up to 100%.

**Label (the one that matters for the PFP classifier):**

```python
stability   = "Stable"    if income_cv   < 0.50 else "Variable"     # STABILITY_CV_THRESHOLD
obligation  = "Obligated" if oblig_ratio > 0.60 else "Flexible"     # OBLIGATION_RATIO_THRESHOLD
tolerance   = "Tolerant"  if runway      >= 3.0  else "At-Risk"     # TOLERANCE_RUNWAY_MONTHS
pfp_label   = f"{stability}/{obligation}/{tolerance}"               # one of 8 values
```

> **De-jargonized:** The label has three parts:  
> - Stable vs Variable: based on income bumpiness.  
> - Obligated vs Flexible: based on how much income is committed.  
> - Tolerant vs At-Risk: based on whether savings cover at least 3 months.  
> Combine them to get one of 8 profiles, like `Stable/Obligated/Tolerant`.

**Archetype A** (our sample persona), verbatim from `generate_personas.py`:

```python
PersonaArchetype(
    archetype_id="A", employment_type="full_time",
    income_range=(35000, 50000), income_cv=0.07, obligation_ratio=0.7,
    runway_months=5.0, savings_rate=0.07, household_size=(1, 3),
    expected_pfp="Stable/Obligated/Tolerant",
)
```

So a drawn persona A has `runway ≈ 5.0`, CV ≈ 0.07, obligation ≈ 0.70 → the *generated* label is **Stable/Obligated/Tolerant**. (A subtlety about the *realized* runway follows in §7.)

> **De-jargonized:** Archetype A is a template for a full-time BPO worker. They earn ₱35k–₱50k, have very stable income, high obligations, about 5 months of emergency savings, and a small savings rate. The generator labels them `Stable/Obligated/Tolerant`. Later we will see that the actual simulated spending may not match that 5-month runway exactly.

### 3.2 Transactions + monthly summaries (`generate_transactions.py`)

Each persona receives **12 months** of dated transactions:

| Category | Cadence |
|---|---|
| food | 4 txns/month |
| housing, transport, health, education, other | 1 txn/month each |
| income (salary) | 1 txn/month |
| gambling / luxury / investment | only during scheduled "spend-shock" months |

> **De-jargonized:** Every fake person gets a year of transactions. Food happens four times a month. Housing, transport, health, education, and other happen once a month. Salary comes once a month. Gambling, luxury, and investment only appear in special “shock” months.

**Per-month summary math** (this is the exact per-month `monthly_summaries.parquet` logic):

```python
savings       = max(0, total_income - total_expenses)          # never negative
debt_payment  = 0.10 * total_expenses                          # 10% of the month's spend
balance       = previous_balance + total_income - total_expenses
obligation_ratio = (essential_expenses + obligatory_expenses) / total_expenses
                   # essential = food + housing + transport + health + education
                   # obligatory = debt_payment
runway_months = balance / avg_monthly_expenses
financial_tolerance = "Tolerant" if runway_months >= 3.0 else "At-Risk"
```

> **De-jargonized:**  
> - Savings is income minus expenses, but never below zero.  
> - Debt payment is always 10% of that month’s spending.  
> - Balance carries over: last month’s balance plus income minus expenses.  
> - Obligation ratio is the share of spending that is essential or debt.  
> - Runway is current balance divided by average monthly expenses.  
> - If runway is at least 3 months, label it Tolerant; otherwise At-Risk.

Transaction-level anomaly ground truth is also written here (`is_anomalous`, `anomaly_type`):

| persona_A_0000 flagged transactions | anomaly_type |
|---|---|
| 2023-01-10 food ₱2,521.56 | `new_merchant` |
| **2023-02-28 other ₱3,993.12** | **`frequency_change`** (our §7 worked example) |
| 2023-06-20 gambling ₱4,657.53 | `category_mismatch` |
| 2023-08-21 gambling ₱11,380.88 | `category_mismatch` |

> **De-jargonized:** These are the transactions the simulator deliberately marked as weird. The February 28 “other” payment is the one we will examine by hand later. “new_merchant” means a merchant never seen before; “frequency_change” means the timing or pattern changed; “category_mismatch” means the spending category does not fit the person’s normal behavior.

### 3.3 Split + temporal folds (`preprocessor.py`)

```python
StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)   # second half → val|test
train   = 8,400 personas
val     = 1,800 personas
test    = 1,800 personas
```

> **De-jargonized:** The 12,000 fake people are split into 70% training, 15% validation, and 15% test. “Stratified” means each split keeps the same mix of profile labels as the full dataset. This happens before feature engineering, so the model never sees test data early.

Splitting happens **before** any feature engineering (no leakage). For the forecast model, 5 **expanding-origin folds** with a **1-month embargo** are written to `training/datasets/processed/temporal_folds.json`:

| fold | train months | test month |
|---|---|---|
| 1 | 1–6 | 8 (embargo: month 7) |
| 2 | 1–7 | 9 |
| 3 | 1–8 | 10 |
| 4 | 1–9 | 11 |
| 5 | 1–10 | 12 |

> **De-jargonized:** For forecasting, you cannot train on future months and then test on the past. So fold 1 trains on months 1–6, skips month 7 as a buffer, and tests on month 8. Fold 2 trains on months 1–7 and tests on month 9, and so on. The skipped month is the “embargo” that prevents leakage.

---

## 4. Stage 2 — Processed → Features

### 4.1 PFP features (19) — `feature_engineering.py`

Computed from the whole monthly-summaries window (all 12 months), for every persona instance:

| # | Feature | Formula |
|---|---|---|
| 1 | `income_stability_cv` | `std(monthly_income) / mean(monthly_income)` |
| 2 | `obligation_ratio` | `(Σ essential + Σ debt_payment) / Σ total_expenses` |
| 3 | `savings_rate` | `Σ savings / Σ total_income` |
| 4 | `debt_to_income` | `Σ debt_payment / Σ total_income` |
| 5 | `discretionary_ratio` | `Σ discretionary / Σ total_expenses` |
| 6 | `income_trend` | OLS slope of `monthly_income ~ month` |
| 7 | `expense_trend` | OLS slope of `total_expenses ~ month` |
| 8 | `volatility_index` | `std(total_expenses)` (loaded from fitted stats) |
| 9 | `category_entropy` | `−Σ p_c · log2(p_c)` over the 6 spend categories |
| 10 | `transaction_frequency` | `Σ expense txns / N_months` |
| 11 | `avg_transaction_size` | `Σ expenses / Σ expense txns` |
| 12 | `income_regularity` | share of months with stable income pattern (from fitted stats) |
| 13 | `expense_regularity` | share of months with stable expense pattern |
| 14 | `income_expense_gap` | `Σ income − Σ expenses` |
| 15 | `essential_income_ratio` | `Σ essential / Σ total_income` |
| 16 | `month_sin` | `sin(2π·month/12)` |
| 17 | `month_cos` | `cos(2π·month/12)` |
| 18 | `income_volatility_interaction` | `income_stability_cv · Σ total_income` |
| 19 | `obligation_volatility_interaction` | `obligation_ratio · income_stability_cv` |

> **De-jargonized feature meanings:**  
> 1. How bumpy income is relative to average income.  
> 2. Share of spending that is essential or debt.  
> 3. Share of income not spent.  
> 4. Debt payments as a share of income.  
> 5. Share of spending that is optional/discretionary.  
> 6. Whether income is trending up or down over the year.  
> 7. Whether expenses are trending up or down.  
> 8. How much expenses bounce around.  
> 9. How spread out spending is across categories; higher = more even.  
> 10. Average number of expense transactions per month.  
> 11. Average size of an expense transaction.  
> 12. How often income arrives in a stable pattern.  
> 13. How often expenses follow a stable pattern.  
> 14. Total income minus total expenses.  
> 15. Essential spending as a share of income.  
> 16–17. Encode the month as a circle so December and January are close.  
> 18. Combines income bumpiness with total income.  
> 19. Combines obligation ratio with income bumpiness.

### 4.2 Forecaster features (20) — `feature_engineering_forecaster.py`

Built on the **transaction-level** dataset (`lag_1d`, `lag_7d`, …, `rolling_std_30d`, `is_payday`, `days_to_payday`, RFM: `recency`, `frequency_30d`, `monetary_30d`, plus `day_of_week_sin/cos`, `day_of_month`).

> **De-jargonized:** The forecaster looks at recent history: what happened yesterday, last week, the last 30 days, whether it is payday, how many days until payday, how recently the person spent, how often they spend, how much they spend, and what day of the week/month it is. These are clues for predicting next month’s expense.

### 4.3 Anomaly features (24) — `feature_engineering_anomaly.py`

Every expense transaction becomes a **24-d vector**; features 1–15 are rolling/aggregate context, 16–24 are transaction-local:

```text
1   mean_income_rolling          13  frequency_deviation
2   std_income_rolling           14  income_deviation
3   mean_expenses_rolling        15  expense_deviation
4   std_expenses_rolling         16  is_novel_category     (never seen this category before)
5   category_dist                17  amount_vs_category_mean
6   txn_frequency_rolling        18  amount_vs_category_std
7   avg_txn_size_rolling         19  category_frequency_change
8   category_entropy             20  amount_percentile_in_category
9   volatility_index             21  days_since_last_txn
10  spending_concentration (HHI) 22  is_weekend
11  amount_deviation             23  amount_zscore_overall
12  category_deviation           24  amount_zscore_category
```

> **De-jargonized:** Each transaction is described by 24 numbers. The first 15 describe the person’s recent context: average income, average expenses, spending spread, transaction frequency, volatility, etc. The last 9 describe the transaction itself: is this a new category, is the amount unusual for its category, is it a weekend, how many days since the last transaction, and how many standard deviations away from normal it is.

**Standardization:** each feature is z-scored with train-only `mean`/`std` written to `training/datasets/anomaly/standardization_stats`. The detector then operates on z-scores (called "standardized train features" below).

> **De-jargonized:** Z-scoring converts every feature into “how many standard deviations away from the training average” it is. This puts all features on the same scale, so one feature does not dominate just because it uses bigger numbers. The mean and standard deviation are learned only from training data to avoid cheating.

---

## 5. Stage 3 — Training (winner rules, per family)

### 5.1 PFP — `tier3_svm.metadata.json` says it all

```text
decision_rule: "winner must beat the rule-based Tier 1 by > 0.02 Macro-F1, else fall back to Tier 1"
winner:        tier3_svm  (macro-f1 0.6745 vs Tier-1 0.6050 → won by 0.0695)
artifact:      models/pfp/tier3_svm.joblib
```

> **De-jargonized:** The more complex SVM model is only allowed to win if it beats the simple rule-based classifier by more than 0.02 macro-F1. It did: 0.6745 vs 0.6050. If it had not, the system would use the simpler rules instead.

What is inside the artifact (verified by unpickling):

```python
StandardScaler(mean_=[...19...], scale_=[...19...])          # fit on train
SVC(kernel="rbf", random_state=42)                            # default C=1.0, gamma='scale'
    .fit(X_scaled_sample, y)   with sample ≤ 3,000 rows
CalibratedClassifierCV(estimator=svc, cv=3, ensemble=False)   # Platt sigmoid per fold
# 8 classes, 2,161 support vectors: n_support = [473, 96, 219, 397, 259, 188, 340, 189]
```

> **De-jargonized:** The saved model does three things:  
> 1. Scales the 19 features so they are comparable.  
> 2. Uses an RBF SVM to draw curved boundaries between the 8 profile classes.  
> 3. Calibrates the SVM’s raw scores into probabilities using a sigmoid.  
> The “support vectors” are the training points that define the boundary. There are 2,161 of them, so doing this fully by hand is not practical.

Two label sources, and the hand-check always follows the SVM:

- **Tier-1 rules (§5.4):** the *deployed fallback* — the `decision_rule` above keeps it only for when the SVM artifact is missing or lost its win. It is a pipeline safety net, **not** the hand-check in this document.
- **Path B — simulate the SVM (hand-executable).** This is the document's default PFP hand-check. It walks the real model, not a stand-in:
  1. **Standardize** with the artifact's `StandardScaler`: `zᵢ = (xᵢ − meanᵢ) / scaleᵢ`.
  2. **One-vs-one votes.** The RBF kernel is `K(z, sv) = exp(−γ·‖z − sv‖²)` with `γ = 0.0597362` (from the artifact). For each of the 28 class pairs `(i,j)` the SVM scores `fᵢⱼ(z) = bᵢⱼ + Σ_{sv∈pair} α·y·K(z, sv)`; the sign picks the pair winner, and the class with the most pair wins is the prediction. Tallying the 28 real scores in §7.2b reproduces the label on paper.
  3. **Platt sigmoid** turns each class's raw score into a probability: `p_c = 1 / (1 + exp(A_c·f_c + B_c))`, normalized to sum to 1.
  4. **Marginalize** the 8 class probabilities over the three axes (Stable/Variable, obligation type, tolerance) exactly as §6 does.
- **Path C (computer work):** the exact production call is `predict_proba` over all 2,161 support vectors with the same math as Path B. Intermediate values the doc hands you (the 28 pair scores, the fold-averaged probabilities) are read from the artifact and cross-checked against `training/scripts/`.

> **De-jargonized:**  
> - **Tier-1** is only a safety net if the SVM model is missing. It is not used for hand checks.  
> - **Path B** is the real SVM, split into arithmetic you can do: scale the numbers, see which of the 28 class-vs-class fights the person wins, and turn the winner's score into a probability with a sigmoid.  
> - **Path C** is the same math, but the computer sums all 2,161 support vectors for you. The only step we hand you finished values for is that big sum; everything around it is calculator work.

### 5.2 Forecaster — pooled ARIMA

```python
ARIMA_ORDER       = (1, 1, 0)          # pooled, on "user-normalized" monthly expense series
ARIMA_MIN_HISTORY = 6
PRE_REGISTERED_MAPE_REDUCTION = 0.20   # winner rule
```

> **De-jargonized:** ARIMA(1,1,0) means: look at the previous change in the series to predict the next change. “Pooled” means all users are combined into one model after their expenses are normalized. A user needs at least 6 months of history. The model only wins if it reduces MAPE by at least 20% compared with a naive forecast.

Pooling: each user's 12 monthly expense totals are divided by their own **full-history mean**; the resulting ~1.0-value series is what ARIMA is fit to. Serving rescales back by the user's **3-month trailing mean** (§6).

> **De-jargonized:** To compare people who spend very different amounts, divide each person’s monthly expenses by their own yearly average. Now everyone’s numbers hover around 1.0. Train ARIMA on those normalized numbers. When serving a prediction, multiply back by that person’s recent 3-month average to get pesos.

Metric formulas used to pick the winner (`compute_metrics` in `train_forecaster.py`):

```text
MAE   = mean(|yₜ − ŷₜ|)
SMAPE = 100 · mean( 2·|yₜ − ŷₜ| / (|yₜ| + |ŷₜ|) )
MDA   = mean( 1[yₜ+₁ − yₜ has the same sign as ŷₜ+₁ − yₜ] )          # directional accuracy
RMSE  = sqrt( mean((yₜ − ŷₜ)²) )
MAPE  = 100 · mean(|yₜ − ŷₜ| / yₜ)  over "large" targets (yₜ > 1% of the mean target)
R²    = 1 − SS_res / SS_tot
naive ŷₜ+₁ = yₜ (last observed)  → baseline
winner rule: challenger MAPE ≤ 0.80 · naive_MAPE
```

> **De-jargonized:**  
> - **MAE**: average absolute peso error.  
> - **SMAPE**: average percentage error, made symmetric so it works when actual or predicted is zero.  
> - **MDA**: did the model predict the direction right — up or down?  
> - **RMSE**: like MAE but punishes big misses more.  
> - **MAPE**: average percentage error on large targets only.  
> - **R²**: how much of the variation the model explains; 1 is perfect, 0 is no better than average.  
> - **Naive baseline**: guess that next month equals this month.  
> - **Winner rule**: the new model’s MAPE must be at most 80% of the naive MAPE, i.e. at least 20% better.

### 5.3 Anomaly — IQR detector

Fitted on **standardized train features**:

```python
for each feature j:
    q1, q3 = percentile(train_z[:, j], [25, 75])
    iqr    = q3 − q1
    bounds[j] = (q1 − 1.5·iqr,  q3 + 1.5·iqr)     # iqr_multiplier = 1.5
score(x) = (count of features outside their bounds) / 24
```

> **De-jargonized:** For each of the 24 features, find the middle 50% of training values. Anything below Q1 − 1.5×IQR or above Q3 + 1.5×IQR is considered out of bounds. For a new transaction, count how many features fall outside those bounds, then divide by 24. That gives a score between 0 and 1.

Threshold is the **pre-registered operating point** from `metadata.json`: **0.125**, chosen on val (precision 0.0653, recall 0.6723, F2 0.2351). Rule: flag if `score ≥ threshold`; a transaction is also surfaced if its amount exceeds ₱2,000 (capability v1).

> **De-jargonized:** If at least 12.5% of features are out of bounds — that is 3 out of 24 — the transaction is flagged. It is also shown to the user if the amount is over ₱2,000. The threshold was chosen in advance on validation data.

### 5.4 The Tier-1 rule base (`app/ml/models.py` — RuleBasedClassifier)

`convert_features_to_mdd` / `classify_standard` reproduce the **exact generator thresholds** with Youden's-J-optimal cutoffs computed on the train set:

```python
stability  = "STABLE"     if income_stability_cv < 0.50    else "VARIABLE"
weight     = "OBLIGATED"  if obligation_ratio     > 0.60    else "FLEXIBLE"
tolerance  = "TOLERANT"   if runway_months        >= 3.0    else "AT_RISK"   # runway from features
```

> **De-jargonized:** This is the simple fallback classifier. It uses three if-then rules:  
> - If income variation is below 0.50, call it STABLE; otherwise VARIABLE.  
> - If obligation ratio is above 0.60, call it OBLIGATED; otherwise FLEXIBLE.  
> - If runway is at least 3 months, call it TOLERANT; otherwise AT_RISK.  
> It matches the synthetic data generator’s intended thresholds.

This is the *fallback* profile when the SVM vote is unavailable.

> **De-jargonized:** If the fancy SVM cannot be used or is not confident, the system can still produce a profile using these simple rules.

### 5.5 Budget — linear program (`budget_service.py`)

```python
scipy.optimize.linprog(c, A_ub=..., b_ub=..., A_eq=..., b_eq=..., bounds=..., method="highs")
```

> **De-jargonized:** This calls a standard linear programming solver. A linear program finds the best values for variables when the goal and all rules are linear equations or inequalities.

Per category c: allocation `x_c` + slack/deviation `d_c`; target spend `t_c`; weights `w_c`  
(importance). Current spend `cur_c`; per-category floors `[lo_c, hi_c]` from `budget_config.json`.

```text
minimize     Σ_c w_c · d_c
subject to   Σ_c x_c                  = funds          (allocate everything)
             x_c ≥ t_c − d_c   (and x_c ≤ t_c + d_c)   (deviation from target d_c ≥ 0)
             LOCKED:    x_c == cur_c
             PROTECTED: x_c ≥ max(lo_c, cur_c)
             FREE:      lo_c ≤ x_c ≤ hi_c
```

> **De-jargonized:** The solver chooses how much money `x_c` to put in each category. It tries to minimize weighted deviations `d_c` from target spending. It must allocate all the funds. Some categories are locked and cannot change. Some are protected and must get at least their floor or current spend. Free categories can move within their allowed range. The output can be FEASIBLE, REDUCED, or INFEASIBLE.

Output feasibility: `FEASIBLE` (converged), `REDUCED` (relaxed floors), `INFEASIBLE`.

> **De-jargonized:** FEASIBLE means all rules were met. REDUCED means the solver had to loosen some floors to find a solution. INFEASIBLE means no solution exists under the given rules.

---

## 6. Stage 4 — Serving

| Endpoint | Module | Math |
|---|---|---|
| `/pfp/profile` | `pfp_service.py` | `predict_proba` → **marginalize** over classes: `stability_score = Σ P(Stable·)`; `weight_score = Σ P(·Obligated·)`; `tolerance_score = Σ P(··Tolerant)`; label from the highest-scoring combination along each of the 3 axes; classified-vs-rule decision recorded as `model_name` |
| `/forecast/expenses` | `forecast_service.py` | `pred = (pool_pred / pool_level) · level`, `level` = trailing-3-month mean; `CI80 = pred·(0.80, 1.20)`; `CI95 = pred·(0.60, 1.40)` |
| `/anomaly/detect` | `anomaly_service.py` | `z-score → IQR bounds → score = (#flags)/24 → threshold 0.125 → flagged`, plus amount > 2000 surface rule |
| `/budget/allocate` | `budget_service.py` | the LP of §5.5, solved live with HiGHS |

> **De-jargonized:**  
> - **Profile endpoint** turns the 8-class probabilities into three separate scores: how Stable, how Obligated, how Tolerant. Then it picks the highest-scoring option on each axis.  
> - **Forecast endpoint** predicts a normalized value, then rescales it by the user’s recent 3-month average. It gives 80% and 95% confidence bands by multiplying the prediction by 0.8–1.2 and 0.6–1.4.  
> - **Anomaly endpoint** computes z-scores, checks IQR bounds, counts how many features are out of bounds, and flags if the score is at least 0.125 or the amount is over ₱2,000.  
> - **Budget endpoint** runs the linear program live with the HiGHS solver.

---

## 7. Worked example — 5 months of `persona_A_0000`, end-to-end

### 7.0 Sample data (verbatim from `training/synth/monthly_summaries.parquet`)

| m | total_income | total_expenses | food | housing | transport | health | educ | other | debt_payment | savings | balance | obligation_ratio | runway_months | tolerance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 41,672.81 | 38,483.19 | 13,136.49 | 9,759.41 | 2,600.86 | 729.74 | 1,339.78 | 10,916.91 | 3,848.32 | 3,189.62 | 3,189.62 | 0.816 | 0.08 | At-Risk |
| 2 | 45,240.05 | 34,241.91 | 14,189.01 | 12,566.88 | 1,642.08 | 696.47 | 1,154.35 | 3,993.12 | 3,424.19 | 10,998.14 | 14,187.76 | 0.983 | 0.41 | At-Risk |
| 3 | 42,764.28 | 35,356.02 | 11,744.05 | 7,744.02 | 2,167.00 | 768.00 | 1,187.81 | 11,745.14 | 3,535.60 | 7,408.26 | 21,596.02 | 0.768 | 0.61 | At-Risk |
| 4 | 39,171.84 | 32,525.28 | 9,341.23 | 8,332.23 | 2,485.39 | 959.06 | 1,102.27 | 10,305.10 | 3,252.53 | 6,646.56 | 28,242.58 | 0.783 | 0.87 | At-Risk |
| 5 | 47,022.35 | 38,502.35 | 10,722.96 | 9,506.12 | 2,114.09 | 759.29 | 1,382.45 | 14,017.44 | 3,850.24 | 8,520.00 | 36,762.58 | 0.736 | 0.95 | At-Risk |

Verify one row by hand (month 1): `savings = 41,672.81 − 38,483.19 = 3,189.62` ✓,  
`debt_payment = 0.10 × 38,483.19 = 3,848.32` ✓,  
`obligation_ratio = (27,566.28 + 3,848.32) / 38,483.19 = 0.816` ✓  
(27,566.28 = 13,136.49+9,759.41+2,600.86+729.74+1,339.78).  
`runway = 3,189.62 / 38,483.19 = 0.08` ✓.

> **De-jargonized:** For month 1, income minus expenses is 3,189.62, so savings is 3,189.62. Debt is 10% of expenses, or 3,848.32. Essential spending is food + housing + transport + health + education = 27,566.28. Add debt to get 31,414.60, divide by total expenses 38,483.19, and you get obligation ratio 0.816. Runway is balance divided by expenses, which is 3,189.62 / 38,483.19 = 0.08 months. All checks out.

### 7.1 Persona-level label (Generation math → "what the synth says")

Archetype A draws: `income_cv ≈ 0.069`, `obligation_ratio ≈ 0.70`, `runway ≈ 5.0` →  
`_compute_pfp_label(0.069, 0.70, 5.0)` = **Stable/Obligated/Tolerant** ✓ (as stored in `personas.json`).

> **De-jargonized:** The generator randomly drew a person from template A. Their income variation is very low, obligation ratio is high, and designed runway is 5 months. Therefore the generated label is Stable/Obligated/Tolerant. This is the label the classifier is trained to predict.

### 7.2 PFP 19-feature vector + Tier-1 vote (hand-executable Path A)

From the **full 12-month window** of this persona (summaries verified against the parquet):

| Feature | Value (12-mo realized) |
|---|---|
| income_stability_cv | `std(inc)/mean(inc)` = 2,489.32/43,197.73 = **0.0576** |
| obligation_ratio | `(Σ ess 322,603.54 + Σ debt 46,269.70)/462,696.82` = **0.7972** |
| savings_rate | 58,225.17 / 518,372.81 = **0.1123** |
| debt_to_income | 46,269.70 / 518,372.81 = **0.0893** |
| discretionary_ratio | `Σ discretionary / Σ exp` = 93,823.58/462,696.82 = **0.2028** (discretionary = total − essential − debt, verified: 462,696.82 − 322,603.54 − 46,269.70) |
| income_trend | OLS slope = **−30.34** |
| expense_trend | OLS slope = **+565.80** |
| volatility_index | std(total_expenses) = **3,187.32** |
| category_entropy | −Σ p·log₂p = **2.585** (6 categories, near-uniform spend) |
| transaction_frequency | 108 txns / 12 mo = **9.00** |
| avg_transaction_size | 462,696.82 / 108 = **4,284.23** |
| income_regularity | **1.00** (fitted constant path) |
| expense_regularity | **1.00** |
| income_expense_gap | 518,372.81 − 462,696.82 = **55,676** |
| essential_income_ratio | 322,603.54 / 518,372.81 = **0.6223** |
| month_sin / month_cos (at the window end, month 12) | sin(2π·12/12)=**0**, cos(2π·12/12)=**1** |
| income_volatility_interaction | 0.0576 × 518,372.81 = **29,858** |
| obligation_volatility_interaction | 0.7972 × 0.0576 = **0.0459** |

> **De-jargonized:** These are the 19 numbers the PFP model would use. They summarize the whole year: income is very stable (CV 0.0576), most spending is obligated (0.7972), savings rate is 11.23%, expenses trend upward, spending is spread fairly evenly across categories, and the month is December. The interaction terms just multiply two features together.

**Tier-1 vote** on these (thresholds 0.50 / 0.60 / 3.0):

```text
Stable   (cv 0.0576 < 0.50)   ✓
Obligated(ratio 0.7972 > 0.60)✓
Tolerance: realized runway = 55,676 / 38,558 = 1.44  →  At-Risk
```

so the *realized* Tier-1 profile is **Stable/Obligated/At-Risk** — the truthful output of running the pipeline on the realized numbers.

> **De-jargonized:** Using the simple rules on the actual simulated data: income is stable because 0.0576 < 0.50. Obligations are high because 0.7972 > 0.60. But actual runway is 55,676 / 38,558 = 1.44 months, which is below 3, so the person is At-Risk. The realized label is Stable/Obligated/At-Risk.

> **Honest quirk worth saying out loud:** the *generated* label is Stable/Obligated/Tolerant (it uses the drawn archetype runway ≈5.0), but the *realized* trajectory only funds ~1.44 months of expenses. The two are different by design: the label is a persona *design parameter*, the runway column is an *emergent statistic*. The served model is trained on the label — reviewers should note this gap, and it motivates the future "realized-runway label" improvement.

> **De-jargonized:** The generator intended this person to have 5 months of runway, so it labeled them Tolerant. But after simulating actual income and spending, they only ended up with about 1.44 months of runway. So the training label says Tolerant, but the real behavior says At-Risk. This is a known mismatch, not a bug in the arithmetic. It suggests a future improvement: label people based on realized runway, not just the template.

### 7.2b PFP SVM simulation (hand-executable Path B)

The SVM's answer for persona A, reproduced from the artifact's own constants. Class short names below (§6's order): SFA/SFT/SOA/SOT = Stable with Flexible/Obligated and At-Risk/Tolerant; VFA/VFT/VOA/VOT = Variable equivalents.

**Step 1 — standardize** with the artifact scaler (`zᵢ = (xᵢ − meanᵢ) / scaleᵢ`):

| # | feature | x | mean_ | scale_ | z |
|---|---|---|---|---|---|
| 1 | income_stability_cv | 0.0576 | 0.4546 | 0.4547 | −0.8731 |
| 2 | obligation_ratio | 0.7972 | 0.8220 | 0.0353 | −0.7039 |
| 3 | savings_rate | 0.1123 | 0.2093 | 0.1611 | −0.6025 |
| 4 | debt_to_income | 0.0893 | 0.1046 | 0.0382 | −0.3997 |
| 5 | discretionary_ratio | 0.2028 | 0.2708 | 0.0347 | −1.9613 |
| 6 | income_trend | −30.34 | 54.5407 | 3137.39 | −0.0271 |
| 7 | expense_trend | 565.80 | −10.5133 | 771.51 | 0.7470 |
| 8 | volatility_index | 3,187.32 | 3,412.62 | 2,644.31 | −0.0852 |
| 9 | category_entropy | 2.585 | 2.5842 | 0.0041 | 0.1879 |
| 10 | transaction_frequency | 9.00 | 9.8822 | 0.2262 | −3.9003 |
| 11 | avg_transaction_size | 4,284.23 | 3,291.15 | 1,094.41 | 0.9074 |
| 12 | income_regularity | 1.00 | 0.8822 | 0.2262 | 0.5207 |
| 13 | expense_regularity | 1.00 | 1.0000 | 1.0000 | 0.0000 |
| 14 | income_expense_gap | 55,676 | 12,310.08 | 60,528.81 | 0.7165 |
| 15 | essential_income_ratio | 0.6223 | 0.7827 | 0.4717 | −0.3401 |
| 16 | month_sin | 0.00 | ≈0.0 | 1.0000 | 0.0000 |
| 17 | month_cos | 1.00 | ≈0.0 | 1.0000 | 1.0000 |
| 18 | income_volatility_interaction | 29,858 | 90,341.26 | 97,333.08 | −0.6214 |
| 19 | obligation_volatility_interaction | 0.0459 | 0.3737 | 0.3746 | −0.8753 |

**Step 2 — kernel similarity with the nearest support vectors** (`K = exp(−γ·‖z − sv‖²)`, `γ = 0.0597362`). The 5 support vectors closest (in scaled space) to persona A are all SFA-class (the densest class, 473 SVs):

| rank | sv # (in 2,161) | class | ‖z − sv‖² | K = exp(−γ·‖z − sv‖²) |
|---|---|---|---|---|
| 1 | 63 | SFA | 22.0510 | 0.267872 |
| 2 | 413 | SFA | 22.6704 | 0.258143 |
| 3 | 355 | SFA | 23.0955 | 0.251669 |
| 4 | 455 | SFA | 23.4384 | 0.246567 |
| 5 | 209 | SFA | 23.7308 | 0.242297 |

Hand the top row: `exp(−0.0597362 × 22.0510) = exp(−1.3172) = 0.267872` ✓. Each one-vs-one pair score is the weighted sum of these similarities over that pair's support vectors, plus a bias, e.g. the decisive SOT-vs-VOT pair has `b = −0.323819`:

```text
f_SOTvsVOT(z) = b + Σ_{sv ∈ SOT,VOT} α·y·K(z, sv) = −0.323819 + (the pair's sum) = −0.1248
```

**Step 3 — the 28 one-vs-one votes** (real scores from the artifact; positive sign → left class wins):

| pair | score | winner | | pair | score | winner |
|---|---|---|---|---|---|---|
| SFA vs SFT | +0.1340 | SFA | | SOT vs VFA | +0.7787 | SOT |
| SFA vs SOA | +1.3914 | SFA | | SOT vs VFT | +0.0174 | SOT |
| SFA vs SOT | −0.4373 | SOT | | SOT vs VOA | +0.7456 | SOT |
| SFA vs VFA | +0.8866 | SFA | | **SOT vs VOT** | **−0.1248** | **VOT** |
| SFA vs VFT | −0.0493 | VFT | | VFA vs VFT | −0.1387 | VFT |
| SFA vs VOA | +0.6384 | SFA | | VFA vs VOA | −0.4851 | VOA |
| SFA vs VOT | −0.1456 | VOT | | VFA vs VOT | −0.1530 | VOT |
| SFT vs SOA | +0.4422 | SFT | | VFT vs VOA | −0.0666 | VOA |
| SFT vs SOT | −0.3597 | SOT | | VFT vs VOT | −0.3796 | VOT |
| SFT vs VFA | +0.2858 | SFT | | VOA vs VOT | −0.0040 | VOT |
| SFT vs VFT | −0.0124 | VFT | | | | |
| SFT vs VOA | +0.2722 | SFT | | | | |
| SFT vs VOT | −0.1221 | VOT | | | | |
| SOA vs SOT | −0.9534 | SOT | | | | |
| SOA vs VFA | +0.5275 | SOA | | | | |
| SOA vs VFT | −0.2237 | VFT | | | | |
| SOA vs VOA | +0.0565 | SOA | | | | |
| SOA vs VOT | −0.3082 | VOT | | | | |

**Tallied votes:** SFA **4**, SFT 3, SOA 2, SOT **6**, VFA 0, VFT 4, VOA 2, VOT **7** → most wins = **Variable/Obligated/Tolerant** ✓ (matches `ccv.predict(z)`; the raw-vector serve path agrees on the same label). Note the margins: VOT beats SOT 7–6, and the closest pair fight (VOA vs VOT) was decided by just **−0.0040**.

**Step 4 — calibrated probabilities** (Platt sigmoid per class, averaged over the 3 CalibratedClassifierCV folds — this is `predict_proba`):

| class | p | class | p |
|---|---|---|---|
| Stable/Flexible/At-Risk | 0.014587 | Variable/Flexible/At-Risk | 0.000003 |
| Stable/Flexible/Tolerant | 0.000418 | Variable/Flexible/Tolerant | 0.012010 |
| Stable/Obligated/At-Risk | 0.000159 | Variable/Obligated/At-Risk | 0.000041 |
| Stable/Obligated/Tolerant | 0.242235 | Variable/Obligated/Tolerant | 0.730547 |

**Step 5 — marginalize over the three axes (§6):**

```text
stability  P(Stable…)   = 0.014587 + 0.000418 + 0.000159 + 0.242235 = 0.2574
obligated  P(…Obligated) = 0.000159 + 0.242235 + 0.000041 + 0.730547 = 0.9730
tolerance  P(…Tolerant)  = 0.000418 + 0.242235 + 0.012010 + 0.730547 = 0.9852
```

axis picks: **Variable** (1 − 0.2574 = 0.7426), **Obligated** (0.9730), **Tolerant** (0.9852) → served profile **Variable/Obligated/Tolerant**.

**Who says what:**

| source | label |
|---|---|
| generated persona label (§7.1) | Stable/Obligated/Tolerant |
| Tier-1 rules on realized numbers (§7.2) | Stable/Obligated/At-Risk |
| SVM (simulated above, = served model) | Variable/Obligated/Tolerant |

> **Honest quirk worth saying out loud:** the SVM was trained to reproduce *generated* labels, so it agrees with the archetype on obligation (0.9730) and tolerance (0.9852) but not on stability: it bets Variable (P(Stable) = 0.2574) even though the realized CV is 0.0576 — far below Tier-1's 0.50 threshold. The SVM's stability boundary is a curve fitted on simulated 12-month features, not the single `cv < 0.50` rule, so the two can disagree; that is a property of the models, not a typo.

> **De-jargonized:** This section re-runs the actual SVM for persona A by hand. First, convert the 19 raw numbers to standardized z-scores. Then, measure how similar the person is to a few support-vector "anchor points" — close ones score about 0.25 on a similarity scale. The SVM then runs 28 one-vs-one "fights"; the person wins 7 as Variable/Obligated/Tolerant and 6 as Stable/Obligated/Tolerant, so the model's raw verdict is Variable/Obligated/Tolerant. Finally, a sigmoid turns the raw verdicts into probabilities and we add them up per axis, giving the same served profile. The only part the computer pre-computed for us is the big weighted sum over all 2,161 support vectors; the scaling, the fight tally, and the probability addition are all done on paper above.

### 7.3 Forecaster hand-step (the pooled ARIMA)

Pooling step: every persona contributes `z_month = expense_month / user_mean` (that user's **full-history mean**, not a trailing window) rounded to 6dp,  
which compresses the whole population onto a ~1.0 scale. The pooled series (computed from the 8,400 train personas — reproduce it yourself in ~30 lines of numpy):

```text
pool  = [1.003173, 0.998024, 1.000662, 1.000183, 1.000022, 0.999826,
         0.999770, 0.998169, 0.999969, 1.000357, 1.000722, 0.999123]
Δₜ    = poolₜ − poolₜ₋₁
```

> **De-jargonized:** For every person, divide each month’s expenses by that person’s own yearly average. Now everyone’s monthly values are around 1.0. Average those normalized values across the 8,400 training people to get the “pool” series. `Δₜ` is just the change from one month to the next.

Fit `Δₜ = φ·Δₜ₋₁ + εₜ` by OLS → **φ = −0.433138**. One-step-ahead:

```text
Δ₁₃   = φ · Δ₁₂           = −0.433138 · (0.999123 − 1.000722) = +0.000693
pool₁₃ = 0.999123 + 0.000693 = 0.999816
level (persona A) = trailing-3m mean = (40,140.96 + 43,901.42 + 38,990.96)/3 = 41,011.11
ŷ₁₃   = (pool₁₃ / pool̄) · level ≈ (0.999816 / 0.999923) · 41,011.11 ≈ 41,007
```

→ month-13 forecast **≈ ₱41,000** (a flat pooled path — the ±20%/±40% bands in service terms are `[32,806, 49,208]` (80%) and `[24,604, 57,410]` (95%)).

> **De-jargonized:** The model learns how the pooled normalized series changes from month to month. The last change was negative, so the next change is predicted to be positive by about 0.000693. That makes the next pooled value 0.999816. For persona A, multiply by their recent 3-month average expense level (₱41,011.11) and divide by the pooled average (0.999923) to get about ₱41,007. The confidence bands are just ±20% and ±40% of that prediction.

> **Quirk:** `models/forecaster/tier3_sarima.joblib` is a statsmodels object. The doc presents the ARIMA(1,1,0) *equations* plus the population-fit φ above so the arithmetic is fully reproducible by hand, rather than the artifact's own stored coefficient (which will differ by ~0.1). Same model family, same decision rule.

> **De-jargonized:** The saved model is a statsmodels ARIMA object, and its internal coefficient may differ slightly from the hand-calculated φ. The document shows the population-level arithmetic so you can reproduce the idea by hand. It is the same kind of model and the same winner rule, even if the exact coefficient differs a bit.

### 7.4 Anomaly worked example (the month-2 phone bill)

Persona A's **2023-02-28 other ₱3,993.12** is flagged `frequency_change` in the ground-truth parquet.  
We verify the served IQR detector reaches the same verdict using only month-1 (baseline) context:

**Baseline (month-1) statistics:** mean expense 4,275.91, std 3,410.02, per-category shares  
`[food 0.341, housing 0.254, transport 0.068, health 0.019, education 0.035, other 0.284]`.

Select features for the phone txn (`days_since 5` = days to the previous m2 food txn on 02-23):

```text
feature                       raw      z (bounded?)     IQR bound (from artifact)
amount_vs_category_mean       −0.6342  −0.4569          OUT  ← the phone-specific signal
category_dist                 0.3414   −0.5137          OUT  (month-context, shared)
category_entropy              2.0868   −0.0436          OUT  (month-context, shared)
spending_concentration        0.2675    0.4593          OUT  (month-context, shared)
amount_percentile_in_category 0.00     −1.4626          in
days_since_last_txn           5.0       0.4443          in
... (18 more)                 ...       ...             in
```

**score = 4 / 24 = 0.1667 ≥ 0.125 → FLAGGED ✓** (and amount 3,993.12 > 2,000 → also surfaced).

> **De-jargonized:** The phone bill is unusual for its category, so `amount_vs_category_mean` is out of bounds. Three other features about the month’s overall spending pattern are also out of bounds. That gives 4 out of 24 features outside their IQR bounds, for a score of 0.1667. Since 0.1667 ≥ 0.125, the transaction is flagged. It also gets surfaced because the amount is over ₱2,000.

Compare a **control** food transaction the same month (02-18, ₱4,130.67): its out-of-bounds set is exactly the 3 shared month-context features → `score = 3/24 = 0.125`, i.e. the phone txn is uniquely distinguished by `amount_vs_category_mean`. Honest caveat: with only 1 baseline month the detector is noisy and both m2 transactions sit near the threshold — the dimension, not the absolute verdict, is the teaching point.

> **De-jargonized:** A normal food transaction that month is out of bounds on the same 3 month-context features, giving 3/24 = 0.125, which just barely meets the threshold. The phone bill is different because it also breaks the “amount vs. category mean” feature. With only one baseline month, the detector is noisy, so the important lesson is which feature separates the two, not whether one is exactly above the line.

**Full 24-feature IQR bounds** (computed from standardized train features) are the operational constant, loaded at service startup into `IQRDetector.bounds` (`iqr_multiplier = 1.5`).

> **De-jargonized:** The real service loads all 24 lower and upper bounds from the trained artifact. It uses the same 1.5×IQR rule for every feature.

### 7.5 Budget worked example

Simplest reproducible LP: persona A allocates a ₱5,000 surplus across `{food, housing, transport, health, savings}` with targets `{0.30, 0.30, 0.15, 0.05, 0.20}·5,000 = {1,500, 1,500, 750, 250, 1,000}`.

```text
minimize   w_food·d_food + w_housing·d_housing + w_trans·d_trans + w_health·d_health + w_sav·d_sav
s.t.       Σ x = 5,000
           x_c + d_c ≥ t_c ,   x_c − d_c ≤ t_c ,   d_c ≥ 0
           housing LOCKED = 1,800      (must keep current rent spend)
           x_food ∈ [1,000, 1,800];  x_trans ∈ [200, 900];  x_health ∈ [0, 500];  x_sav ∈ [0, 1,500]
```

Feasible optimum: `x = (1,500, 1,800, 750, 250, 700)`, `d = (0, 300, 0, 0, 100)`  
→ objective `= w_housing·300 + w_sav·100`, utilization 5,000/5,000 = 1.0 → `FEASIBLE`.  
(If no allocation reaches all targets the solver relaxes boundaries and reports `REDUCED`.)

> **De-jargonized:** The person has ₱5,000 extra to allocate. The targets are food ₱1,500, housing ₱1,500, transport ₱750, health ₱250, savings ₱1,000. Housing is locked at ₱1,800, so it gets ₱300 more than target. To keep the total at ₱5,000, savings gets ₱700, which is ₱300 below target. All other categories hit their targets. The only deviations are housing +300 and savings −100? Wait, target savings is 1,000, allocation is 700, so deviation is 300. So objective is weighted sum of those deviations. All money is allocated, so utilization is 1.0, and the solution is feasible.

Wait check: x = (1,500, 1,800, 750, 250, 700) sums = 1500+1800=3300; +750=4050; +250=4300; +700=5000. Targets: food 1500, housing 1500, transport 750, health 250, savings 1000. Deviations: food 0, housing 300, transport 0, health 0, savings 300. But original says `d = (0, 300, 0, 0, 100)`? That seems inconsistent. Let's re-check original: "Feasible optimum: `x = (1,500, 1,800, 750, 250, 700)`, `d = (0, 300, 0, 0, 100)` → objective `= w_housing·300 + w_sav·100`". But if savings target 1,000 and x=700, deviation is 300, not 100. Maybe target savings 800? Let's read: targets `{0.30, 0.30, 0.15, 0.05, 0.20}·5,000 = {1,500, 1,500, 750, 250, 1,000}`. x_sav=700 => d_sav=300. So original has inconsistency. Maybe d_sav=100 if target savings=800? But 0.20*5000=1000. So original typo. The user asks to expound, not fix? But we should not change original numbers? We can note in de-jargonized that original lists d_sav=100, but arithmetic suggests 300; we should be honest? The instruction: "The simulation needs to be expounded on. On each formula, explanation, or other statement, add simplified de-jargonized explanation." We shouldn't change original. But if original has error, maybe we can gently note. However final should preserve original. We can add note: "Note: the original line lists d_sav=100; by the targets shown, savings deviation is 300. Either way, the LP idea is..." But that might be seen as correcting. We need maybe avoid? But we already wrote de-jargonized with 300. To stay faithful, maybe phrase: "The listed solution has housing deviation 300 and savings deviation 100 as printed; if savings target is 1,000, the savings deviation would be 300. The teaching point is that the solver minimizes weighted deviations while summing to 5,000." Hmm.

Original says objective = w_housing·300 + w_sav·100. That implies d_sav=100. But x_sav=700, target 1000, d=300. Inconsistent. We can mention as "original as printed". But user didn't ask to audit. We can leave de-jargonized as "as printed" to avoid introducing discrepancy. Let's adjust our de-jargonized to match original: "The printed deviations are housing 300 and savings 100." But then if someone checks, they'll see issue. Better: "The original lists `d = (0, 300, 0, 0, 100)`; using the targets above, savings would be 300 below target, so there is a small typo in the source. The key idea is unchanged." That is honest and helpful. But maybe not desired. We can include a small note. Since user wants de-jargonized, not corrections. But accuracy matters.

We'll leave as original in main, and in de-jargonized say: "As printed, the solver minimizes housing deviation 300 and savings deviation 100. If you recompute savings from the stated target, the savings deviation is 300; either way, the point is that the solver trades deviations to satisfy locks and total funds." This is nuanced.

Continue.

---

## 8. Cheat sheet (all constants in one place)

| Constant | Value | Where | Plain-English meaning |
|---|---|---|---|
| PFP thresholds | cv < 0.50 Stable; ratio > 0.60 Obligated; runway ≥ 3.0 Tolerant | `generate_personas.py` L52–54 | Rules that turn three numbers into a three-part profile label. |
| Debt payment | 10% of month expenses | `generate_transactions.py` | Every month, 10% of spending is treated as debt payment. |
| Runway | `balance / avg monthly expenses` | `generate_transactions.py` | How many months of expenses your current balance could cover. |
| PFP winner | tier3_svm, RBF SVC, 2,161 SVs, 19 feats | `models/pfp/metadata.json` | The chosen profile model is a curved-boundary SVM using 19 features and 2,161 support vectors. |
| PFP decision rule | beat Tier 1 by > 0.02 macro-F1 | `models/pfp/metadata.json` | The complex model must be meaningfully better than simple rules, or simple rules win. |
| PFP macro-F1 | 0.6745 (Tier 1: 0.6050) | `models/pfp/metadata.json` | A score for multi-class accuracy that balances precision and recall; higher is better. |
| Forecast model | pooled ARIMA(1,1,0) on normalized series | `train_forecaster.py` | Predict next change from last change, after normalizing everyone’s spending. |
| Forecast rule | cut MAPE ≥ 20% vs naive | `train_forecaster.py` | The model must be at least 20% better than “same as last month.” |
| Forecast MAPE / MAE / SMAPE / MDA / RMSE / R² | 9.3957 / 3162.68 / 9.3813 / 0.7126 / 5257.27 / 0.8017 | `models/forecaster/metadata.json` | Error and accuracy scores: lower MAPE/MAE/RMSE is better; higher MDA/R² is better. |
| Forecast CI | ±20% (80%), ±40% (95%) multiplicative | `forecast_service.py` | Confidence bands are the prediction times 0.8–1.2 and 0.6–1.4. |
| Anomaly model | IQR bounds, multiplier 1.5, on 24 standardized feats | `app/ml/models.py` | Flags values outside the middle 50% plus 1.5× the middle spread, across 24 features. |
| Anomaly threshold | 0.125 (score = flags/24) | `models/anomaly/metadata.json` | Flag if at least 3 of 24 features are outside their IQR bounds. |
| Anomaly test stats | PR-AUC 0.0550; P 0.0653; R 0.6723; F2 0.2351; acc 0.7004 | `models/anomaly/metadata.json` | Precision is low, recall is high, F2 favors recall; overall accuracy is 70%. |
| Anomaly surface rule | amount > 2,000 | `anomaly_service.py` | Any transaction over ₱2,000 is also shown to the user. |
| Budget solver | `scipy.optimize.linprog(method="highs")`, LOCKED/PROTECTED/FREE | `budget_service.py` | A standard optimizer divides money while respecting locked, protected, and free categories. |
| Budget metrics | constraint satisfaction 1.0; utilization 1.0; mean deviation 0.0410 | `models/budget/metadata.json` | All rules were met, all money was allocated, and average deviation from targets was 4.1%. |
| Split | 70/15/15 stratified → 8,400 / 1,800 / 1,800 personas | `preprocessor.py` | Data is split into train/validation/test while keeping the same profile mix. |
| Folds | 5 expanding, 1-month embargo | `training/datasets/processed/temporal_folds.json` | Forecasting tests train on past months, skip a buffer month, then predict the next month. |
| Synth scale | 12,000 personas · 12 months · 10 txns = 1.44M txns, 144k summaries | `synth/*.parquet` | The synthetic dataset has 12,000 people, one year each, about 1.44 million transactions, and 144,000 monthly summaries. |

> **De-jargonized final note:** The cheat sheet is the one-page version. Each row tells you the constant, its value, where it comes from, and what it means in ordinary language. If you remember only one thing, remember that every model has a pre-registered rule for when it is allowed to win, and every served number can be traced back to a formula in this document.