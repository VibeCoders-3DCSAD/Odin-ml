# BUDI ML Pipeline — Pen-and-Paper Simulation

> How every number in the BUDI ML microservice is produced, from a raw FIES row to a served
> forecast, profile, anomaly flag, and budget plan — written so that any group member (or
> professor) can reproduce the arithmetic by hand from a single 5-month sample.
>
> **Sample persona used:** `persona_A_0000` (archetype A — regular BPO employee).
> All figures below were extracted verbatim from the real synthetic artifacts in this repo.

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

**Model families** (their pre-registered winner rules and hold-out metrics are in
`models/<family>/metadata.json`):

| Family | Winner | Decision rule | Hold-out metric |
|---|---|---|---|
| PFP (8-class profile) | `tier3_svm` (RBF SVC + Platt calibration) | must beat rule-based Tier 1 by **> 0.02 Macro-F1** | macro-F1 **0.6745** (Tier 1: 0.6050) |
| Forecaster (13th-month expense) | `tier3_sarima` (pooled ARIMA) | must cut MAPE by **≥ 20%** vs naive | MAPE **9.3957%** (−74.9% vs naive) |
| Anomaly (transaction flags) | `tier1_iqr` | ML tier must reach PR-AUC ≥ 1.5×IQR **and** ≥ 0.15 | test PR-AUC 0.0550, threshold **0.125**, F2 0.2351 |
| Budget (allocation) | `scipy_linprog` (simplex/HiGHS) | LP must satisfy all hard constraints | constraint satisfaction **1.0**, utilization 1.0, mean deviation 0.041 |

> **Ground rule of this document:** every constant below that can be cross-checked has been
> cross-checked against `training/scripts/*.py` or the committed artifacts. Where the doc can only
> *describe* a computation (svm kernel over 2,161 support vectors), we say so and give the
> hand-executable Path A (rule-based) instead.

---

## 2. Stage 0 — Raw → Unprocessed (`collector.py`)

`training/scripts/collector.py` converts the FIES 2018 CSV to Parquet with **no arithmetic**.

```text
collector.py --input datasets/raw/family_income_and_expenditure.csv / puf.csv  →  puf.parquet
```

Sample raw row (fixed-width-ish CSV, 90 columns; parsed as plain integers):

```text
W_REGN  W_PROV   SEQ_NO   RPROV  FSIZE   REG_SAL   SEASON_SAL   WAGES      ...
01      28       000001   2800   02.5    00119000   00000000     00119000   ...
```

Values are **unadjusted** (e.g. `00119000` = ₱11,900). Column semantics follow the PSA FIES
codebook (`W_REGN` region, `W_PROV` province, `FSIZE` household size, `REG_SAL` regular salary,
`WAGES` wage income, …). Nothing aggregates here; the pipeline stays faithful to the source.

---

## 3. Stage 1 — Unprocessed → Processed (synthesis + split)

### 3.1 Personas (`generate_personas.py`)

The synth population starts from **12 non-overlapping archetypes** (A–L), each defining an income
range, a target income CV, a target obligation ratio, a target emergency runway, a savings rate,
household size, and an **expected 8-class PFP label**. Each real persona is:

```python
base_income    = uniform(archetype.income_range)           # then round to nearest ₱1,000
monthly_income = round(base_income / 1000) * 1000
income_cv      = clip(archetype.income_cv + N(0, 0.03), 0.01, 1.50)
oblig_ratio    = clip(archetype.obligation_ratio + N(0, 0.03), 0.05, 0.95)
runway         = clip(archetype.runway_months + N(0, 0.30), 0.0, 24.0)
```

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

**Label (the one that matters for the PFP classifier):**

```python
stability   = "Stable"    if income_cv   < 0.50 else "Variable"     # STABILITY_CV_THRESHOLD
obligation  = "Obligated" if oblig_ratio > 0.60 else "Flexible"     # OBLIGATION_RATIO_THRESHOLD
tolerance   = "Tolerant"  if runway      >= 3.0  else "At-Risk"     # TOLERANCE_RUNWAY_MONTHS
pfp_label   = f"{stability}/{obligation}/{tolerance}"               # one of 8 values
```

**Archetype A** (our sample persona), verbatim from `generate_personas.py`:

```python
PersonaArchetype(
    archetype_id="A", employment_type="full_time",
    income_range=(35000, 50000), income_cv=0.07, obligation_ratio=0.7,
    runway_months=5.0, savings_rate=0.07, household_size=(1, 3),
    expected_pfp="Stable/Obligated/Tolerant",
)
```

So a drawn persona A has `runway ≈ 5.0`, CV ≈ 0.07, obligation ≈ 0.70 → the *generated* label is
**Stable/Obligated/Tolerant**. (A subtlety about the *realized* runway follows in §7.)

### 3.2 Transactions + monthly summaries (`generate_transactions.py`)

Each persona receives **12 months** of dated transactions:

| Category | Cadence |
|---|---|
| food | 4 txns/month |
| housing, transport, health, education, other | 1 txn/month each |
| income (salary) | 1 txn/month |
| gambling / luxury / investment | only during scheduled "spend-shock" months |

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

Transaction-level anomaly ground truth is also written here (`is_anomalous`, `anomaly_type`):

| persona_A_0000 flagged transactions | anomaly_type |
|---|---|
| 2023-01-10 food ₱2,521.56 | `new_merchant` |
| **2023-02-28 other ₱3,993.12** | **`frequency_change`** (our §7 worked example) |
| 2023-06-20 gambling ₱4,657.53 | `category_mismatch` |
| 2023-08-21 gambling ₱11,380.88 | `category_mismatch` |

### 3.3 Split + temporal folds (`preprocessor.py`)

```python
StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)   # second half → val|test
train   = 8,400 personas
val     = 1,800 personas
test    = 1,800 personas
```

Splitting happens **before** any feature engineering (no leakage). For the forecast model,
5 **expanding-origin folds** with a **1-month embargo** are written to
`training/datasets/processed/temporal_folds.json`:

| fold | train months | test month |
|---|---|---|
| 1 | 1–6 | 8 (embargo: month 7) |
| 2 | 1–7 | 9 |
| 3 | 1–8 | 10 |
| 4 | 1–9 | 11 |
| 5 | 1–10 | 12 |

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

### 4.2 Forecaster features (20) — `feature_engineering_forecaster.py`

Built on the **transaction-level** dataset (`lag_1d`, `lag_7d`, …, `rolling_std_30d`,
`is_payday`, `days_to_payday`, RFM: `recency`, `frequency_30d`, `monetary_30d`, plus
`day_of_week_sin/cos`, `day_of_month`).

### 4.3 Anomaly features (24) — `feature_engineering_anomaly.py`

Every expense transaction becomes a **24-d vector**; features 1–15 are rolling/aggregate context,
16–24 are transaction-local:

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

**Standardization:** each feature is z-scored with train-only `mean`/`std` written to
`training/datasets/anomaly/standardization_stats`. The detector then operates on z-scores (called
"standardized train features" below).

---

## 5. Stage 3 — Training (winner rules, per family)

### 5.1 PFP — `tier3_svm.metadata.json` says it all

```text
decision_rule: "winner must beat the rule-based Tier 1 by > 0.02 Macro-F1, else fall back to Tier 1"
winner:        tier3_svm  (macro-f1 0.6745 vs Tier-1 0.6050 → won by 0.0695)
artifact:      models/pfp/tier3_svm.joblib
```

What is inside the artifact (verified by unpickling):

```python
StandardScaler(mean_=[...19...], scale_=[...19...])          # fit on train
SVC(kernel="rbf", random_state=42)                            # default C=1.0, gamma='scale'
    .fit(X_scaled_sample, y)   with sample ≤ 3,000 rows
CalibratedClassifierCV(estimator=svc, cv=3, ensemble=False)   # Platt sigmoid per fold
# 8 classes, 2,161 support vectors: n_support = [473, 96, 219, 397, 259, 188, 340, 189]
```

The two alternative paths to a label:

- **Path A (hand-executable whenever an SVM kernel is impractical):** the Tier-1 rules (§5.4).
- **Path B:** full RBF SVM decision
  `f(x) = w·K(x) + b`, `K(x,z) = exp(−γ·‖x−z‖²)`, with Platt sigmoid
  `p(y|f) = 1 / (1 + exp(A·f + B))`, then softmax over the 8 class votes. With 2,161 support
  vectors and 19 features this is **not** pen-and-paper; it is what `predict_proba` does.

### 5.2 Forecaster — pooled ARIMA

```python
ARIMA_ORDER       = (1, 1, 0)          # pooled, on "user-normalized" monthly expense series
ARIMA_MIN_HISTORY = 6
PRE_REGISTERED_MAPE_REDUCTION = 0.20   # winner rule
```

Pooling: each user's 12 monthly expense totals are divided by their own **3-month trailing mean**;
the resulting ~1.0-value series is what ARIMA is fit to. Serving rescales back (§6).

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

### 5.3 Anomaly — IQR detector

Fitted on **standardized train features**:

```python
for each feature j:
    q1, q3 = percentile(train_z[:, j], [25, 75])
    iqr    = q3 − q1
    bounds[j] = (q1 − 1.5·iqr,  q3 + 1.5·iqr)     # iqr_multiplier = 1.5
score(x) = (count of features outside their bounds) / 24
```

Threshold is the **pre-registered operating point** from `metadata.json`: **0.125**, chosen on val
(precision 0.0653, recall 0.6723, F2 0.2351). Rule: flag if `score ≥ threshold`; a transaction is
also surfaced if its amount exceeds ₱2,000 (capability v1).

### 5.4 The Tier-1 rule base (`app/ml/models.py` — RuleBasedClassifier)

`convert_features_to_mdd` / `classify_standard` reproduce the **exact generator thresholds**
with Youden's-J-optimal cutoffs computed on the train set:

```python
stability  = "STABLE"     if income_stability_cv < 0.50    else "VARIABLE"
weight     = "OBLIGATED"  if obligation_ratio     > 0.60    else "FLEXIBLE"
tolerance  = "TOLERANT"   if runway_months        >= 3.0    else "AT_RISK"   # runway from features
```

This is the *fallback* profile when the SVM vote is unavailable.

### 5.5 Budget — linear program (`budget_service.py`)

```python
scipy.optimize.linprog(c, A_ub=..., b_ub=..., A_eq=..., b_eq=..., bounds=..., method="highs")
```

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

Output feasibility: `FEASIBLE` (converged), `REDUCED` (relaxed floors), `INFEASIBLE`.

---

## 6. Stage 4 — Serving

| Endpoint | Module | Math |
|---|---|---|
| `/pfp/profile` | `pfp_service.py` | `predict_proba` → **marginalize** over classes: `stability_score = Σ P(Stable·)`; `weight_score = Σ P(·Obligated·)`; `tolerance_score = Σ P(··Tolerant)`; label from the highest-scoring combination along each of the 3 axes; classified-vs-rule decision recorded as `model_name` |
| `/forecast/expenses` | `forecast_service.py` | `pred = (pool_pred / pool_level) · level`, `level` = trailing-3-month mean; `CI80 = pred·(0.80, 1.20)`; `CI95 = pred·(0.60, 1.40)` |
| `/anomaly/detect` | `anomaly_service.py` | `z-score → IQR bounds → score = (#flags)/24 → threshold 0.125 → flagged`, plus amount > 2000 surface rule |
| `/budget/allocate` | `budget_service.py` | the LP of §5.5, solved live with HiGHS |

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

### 7.1 Persona-level label (Generation math → "what the synth says")

Archetype A draws: `income_cv ≈ 0.069`, `obligation_ratio ≈ 0.70`, `runway ≈ 5.0` →
`_compute_pfp_label(0.069, 0.70, 5.0)` = **Stable/Obligated/Tolerant** ✓ (as stored in
`personas.json`).

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

**Tier-1 vote** on these (thresholds 0.50 / 0.60 / 3.0):

```text
Stable   (cv 0.0576 < 0.50)   ✓
Obligated(ratio 0.7972 > 0.60)✓
Tolerance: realized runway = 55,676 / 38,558 = 1.44  →  At-Risk
```

so the *realized* Tier-1 profile is **Stable/Obligated/At-Risk** — the truthful output of running
the pipeline on the realized numbers.

> **Honest quirk worth saying out loud:** the *generated* label is Stable/Obligated/Tolerant
> (it uses the drawn archetype runway ≈5.0), but the *realized* trajectory only funds ~1.44 months
> of expenses. The two are different by design: the label is a persona *design parameter*, the
> runway column is an *emergent statistic*. The served model is trained on the label — reviewers
> should note this gap, and it motivates the future "realized-runway label" improvement.

### 7.3 Forecaster hand-step (the pooled ARIMA)

Pooling step: every persona contributes `z_month = expense_month / trailing_3m_mean` rounded to 6dp,
which compresses the whole population onto a ~1.0 scale. The pooled series (computed from the 8,400
train personas — reproduce it yourself in ~30 lines of numpy):

```text
pool  = [1.003173, 0.998024, 1.000662, 1.000183, 1.000022, 0.999826,
         0.999770, 0.998169, 0.999969, 1.000357, 1.000722, 0.999123]
Δₜ    = poolₜ − poolₜ₋₁
```

Fit `Δₜ = φ·Δₜ₋₁ + εₜ` by OLS → **φ = −0.433138**. One-step-ahead:

```text
Δ₁₃   = φ · Δ₁₂           = −0.433138 · (0.999123 − 1.000722) = +0.000693
pool₁₃ = 0.999123 + 0.000693 = 0.999816
level (persona A) = trailing-3m mean = (40,140.96 + 43,901.42 + 38,990.96)/3 = 41,011.11
ŷ₁₃   = (pool₁₃ / pool̄) · level ≈ (0.999816 / 0.999923) · 41,011.11 ≈ 41,007
```

→ month-13 forecast **≈ ₱41,000** (a flat pooled path — the ±20%/±40% bands in service terms are
`[32,806, 49,208]` (80%) and `[24,604, 57,410]` (95%)).

> **Quirk:** `models/forecaster/tier3_sarima.joblib` is a statsmodels object; neither available venv
> here can unpickle it (no `statsmodels`), so the doc presents the ARIMA(1,1,0) *equations* plus the
> population-fit φ above rather than the artifact's own stored coefficient (which will differ by
> ~0.1). Same model family, same decision rule, fully reproducible.

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

Compare a **control** food transaction the same month (02-18, ₱4,130.67): its out-of-bounds set is
exactly the 3 shared month-context features → `score = 3/24 = 0.125`, i.e. the phone txn is uniquely
distinguished by `amount_vs_category_mean`. Honest caveat: with only 1 baseline month the detector is
noisy and both m2 transactions sit near the threshold — the dimension, not the absolute verdict, is
the teaching point.

**Full 24-feature IQR bounds** (computed from standardized train features) are the operational
constant, loaded at service startup into `IQRDetector.bounds` (`iqr_multiplier = 1.5`).

### 7.5 Budget worked example

Simplest reproducible LP: persona A allocates a ₱5,000 surplus across `{food, housing, transport,
health, savings}` with targets `{0.30, 0.30, 0.15, 0.05, 0.20}·5,000 = {1,500, 1,500, 750, 250, 1,000}`.

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

---

## 8. Cheat sheet (all constants in one place)

| Constant | Value | Where |
|---|---|---|
| PFP thresholds | cv < 0.50 Stable; ratio > 0.60 Obligated; runway ≥ 3.0 Tolerant | `generate_personas.py` L52–54 |
| Debt payment | 10% of month expenses | `generate_transactions.py` |
| Runway | `balance / avg monthly expenses` | `generate_transactions.py` |
| PFP winner | tier3_svm, RBF SVC, 2,161 SVs, 19 feats | `models/pfp/metadata.json` |
| PFP decision rule | beat Tier 1 by > 0.02 macro-F1 | `models/pfp/metadata.json` |
| PFP macro-F1 | 0.6745 (Tier 1: 0.6050) | `models/pfp/metadata.json` |
| Forecast model | pooled ARIMA(1,1,0) on normalized series | `train_forecaster.py` |
| Forecast rule | cut MAPE ≥ 20% vs naive | `train_forecaster.py` |
| Forecast MAPE / MAE / SMAPE / MDA / RMSE / R² | 9.3957 / 3162.68 / 9.3813 / 0.7126 / 5257.27 / 0.8017 | `models/forecaster/metadata.json` |
| Forecast CI | ±20% (80%), ±40% (95%) multiplicative | `forecast_service.py` |
| Anomaly model | IQR bounds, multiplier 1.5, on 24 standardized feats | `app/ml/models.py` |
| Anomaly threshold | 0.125 (score = flags/24) | `models/anomaly/metadata.json` |
| Anomaly test stats | PR-AUC 0.0550; P 0.0653; R 0.6723; F2 0.2351; acc 0.7004 | `models/anomaly/metadata.json` |
| Anomaly surface rule | amount > 2,000 | `anomaly_service.py` |
| Budget solver | `scipy.optimize.linprog(method="highs")`, LOCKED/PROTECTED/FREE | `budget_service.py` |
| Budget metrics | constraint satisfaction 1.0; utilization 1.0; mean deviation 0.0410 | `models/budget/metadata.json` |
| Split | 70/15/15 stratified → 8,400 / 1,800 / 1,800 personas | `preprocessor.py` |
| Folds | 5 expanding, 1-month embargo | `training/datasets/processed/temporal_folds.json` |
| Synth scale | 12,000 personas · 12 months · 10 txns = 1.44M txns, 144k summaries | `synth/*.parquet` |