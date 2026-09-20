# BUDI Serve-Time Algorithms — Pen-and-Paper Instructions

```json
{
  "document-type": "worked-example",
  "version": "1.0.0",
  "date": "2026.09.17",
  "authors": ["Group 4, III-DCSAD"]
}
```

> Numbered algorithms for the four served families, written so a calculator and a sheet of paper are enough to reproduce the arithmetic. Each section states the inputs, the steps the service actually runs, a worked example, and the boxed output.

**Scope of this document**

| Family | Served winner | What you compute by hand |
| :--- | :--- | :--- |
| Classification (PFP) | `tier3_svm` (RBF SVM + Platt), with a questionnaire cold start and a Tier-1 rule fallback | Features, three-axis rules, questionnaire mapping. Full SVM kernel over 2,161 support vectors is not hand-complete; a 2-vector kernel step is included so the served formula is still executable. |
| Forecasting v2 | Pooled `SARIMA(1,1,0)(1,0,0,12)` in `models/forecaster_v2/` | One-step pooled forecast from the committed coefficients, then peso rescaling. |
| Anomaly alerts | `tier1_iqr` (1.5×IQR on 24 standardized features) | Quartiles, bounds, flag count, score, threshold. |
| Budgeting | `scipy.optimize.linprog` (HiGHS) | Write the LP, bind LOCKED/PROTECTED/FREE, recover the unique optimum on a 4-category instance. |

Companion documents: full pipeline walkthrough in [`pen-and-paper-simulation.md`](pen-and-paper-simulation.md); v2 training note in [`docs/models/forecaster-v2.md`](../../docs/models/forecaster-v2.md).

**Notation.** ₱ amounts are pesos. `Σ` is a sum over the months (or categories) in the working window. Sample standard deviation uses `n − 1`. Numpy-style linear percentiles are used for Q1/Q3 (the same method `IQRDetector.fit` calls).

---

## 1. Classification (PFP)

**What it does.** Maps a user onto one of eight labels `STABILITY/OBLIGATION/TOLERANCE`, plus three scores in `[0, 1]`.

| Axis | Values | Score meaning |
| :--- | :--- | :--- |
| Stability | `Stable` / `Variable` | Probability mass on classes that start with `Stable` |
| Obligation | `Obligated` / `Flexible` | Probability mass on classes that contain `Obligated` |
| Tolerance | `Tolerant` / `At-Risk` | Probability mass on classes that contain `Tolerant` |

Served label format: `STABLE_OBLIGATED_AT_RISK` (underscores, uppercase).

Two serve paths exist in `app/services/pfp_service.py`:

1. `QUESTIONNAIRE` — no model, three answers.
2. `STANDARD` — 19 features through the calibrated SVM.

Tier-1 rules are the pre-registered fallback in the winner contract (the SVM must beat them by more than 0.02 macro-F1). They are the fully hand-complete classifier.

### 1.1 Algorithm A — Questionnaire (cold start)

**Inputs.** Three strings from the user: `income_variability`, `obligation_level`, `emergency_runway`.

**Steps.**

1. Stability score:
   - `0.2` if the variability text contains `variable` or `irregular`
   - else `0.8`
2. Weight score:
   - `0.8` if obligation contains `high`
   - else `0.4` if it contains `medium`
   - else `0.2`
3. Tolerance score:
   - `0.2` if runway contains `low`
   - else `0.8` if it contains `high` or the character `3`
   - else `0.5`
4. Label parts:
   - `Variable` if stability `< 0.5`, else `Stable`
   - `Obligated` if weight `≥ 0.5`, else `Flexible`
   - `At-Risk` if tolerance `< 0.5`, else `Tolerant`
5. Confidence = `max(stability, weight, tolerance)`.

**Worked example.** Answers: variability = `stable`, obligation = `high`, runway = `low`.

```text
stability = 0.8          (no "variable"/"irregular")
weight    = 0.8          ("high")
tolerance = 0.2          ("low")
label     = Stable / Obligated / At-Risk
served    = STABLE_OBLIGATED_AT_RISK
confidence = 0.8
```

### 1.2 Algorithm B — Monthly summaries (shared by C and D)

**Inputs.** A list of dated income and expense transactions. The worked window below is five months of `persona_A_0000` (same figures as the pipeline simulation).

| m | income | expenses | food | housing | transport | health | educ | other |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41,672.81 | 38,483.19 | 13,136.49 | 9,759.41 | 2,600.86 | 729.74 | 1,339.78 | 10,916.91 |
| 2 | 45,240.05 | 34,241.91 | 14,189.01 | 12,566.88 | 1,642.08 | 696.47 | 1,154.35 | 3,993.12 |
| 3 | 42,764.28 | 35,356.02 | 11,744.05 | 7,744.02 | 2,167.00 | 768.00 | 1,187.81 | 11,745.14 |
| 4 | 39,171.84 | 32,525.28 | 9,341.23 | 8,332.23 | 2,485.39 | 959.06 | 1,102.27 | 10,305.10 |
| 5 | 47,022.35 | 38,502.35 | 10,722.96 | 9,506.12 | 2,114.09 | 759.29 | 1,382.45 | 14,017.44 |

For each month compute:

```text
savings_m      = max(0, income_m − expenses_m)
debt_m         = 0.10 × expenses_m
essential_m    = food_m + housing_m + transport_m + health_m + educ_m
balance_m      = balance_{m−1} + income_m − expenses_m     (start at 0)
```

Month 1 check: savings = 41,672.81 − 38,483.19 = **3,189.62**, debt = 0.10 × 38,483.19 = **3,848.32**.

Window totals:

```text
Σ income     = 215,871.33
Σ expenses   = 179,108.75
Σ essential  = 128,131.04
Σ other      =  50,977.71
Σ debt       =  17,910.88     (= 0.10 × Σ expenses)
Σ savings    =  36,762.58     (= end balance, because we started at 0)
n            = 5
mean income  = 215,871.33 / 5 = 43,174.27
mean expense = 179,108.75 / 5 = 35,821.75
```

### 1.3 Algorithm C — Tier-1 rules (hand-complete classifier)

**Inputs.** Three scalars from the window.

1. **Income CV** (sample std / mean):

```text
s_inc = sqrt( Σ (income_m − 43,174.27)²  /  4 ) = 3,062.56
CV    = 3,062.56 / 43,174.27 = 0.0709
```

2. **Obligation ratio**

```text
(Σ essential + Σ debt) / Σ expenses
= (128,131.04 + 17,910.88) / 179,108.75
= 0.8154
```

3. **Runway (months)**

```text
end_balance / mean_expense
= 36,762.58 / 35,821.75
= 1.026
```

4. **Apply the generator thresholds** (`generate_personas.py` / `RuleBasedClassifier` defaults):

```text
stability  = Stable    if CV < 0.50 else Variable          → Stable    (0.0709 < 0.50)
obligation = Obligated if ratio > 0.60 else Flexible       → Obligated (0.8154 > 0.60)
tolerance  = Tolerant  if runway ≥ 3.0 else At-Risk        → At-Risk   (1.026 < 3.0)
```

**Boxed answer (Tier-1 on this 5-month window):** `STABLE_OBLIGATED_AT_RISK`

The generated persona label for archetype A is `Stable/Obligated/Tolerant` because the generator draws a *design* runway of about 5 months. The realized trajectory only funds ~1 month. The rules follow the realized numbers.

### 1.4 Algorithm D — Served SVM (standard path)

**Feature vector (19 numbers).** Compute the window features, then the last-month cyclical pair and the two interactions. OLS slope uses `x = 0,1,2,3,4`:

```text
slope = (n Σ xy − Σx Σy) / (n Σx² − (Σx)²)
Σx = 10,   Σx² = 30,   denominator = 50

income slope  = (5 × 436,373.53 − 10 × 215,871.33) / 50 = 463.09
expense slope = (5 × 356,539.19 − 10 × 179,108.75) / 50 = −167.83
```

| # | Feature | Formula on this window | Value |
| :---: | :--- | :--- | ---: |
| 1 | `income_stability_cv` | `s_inc / mean_inc` | 0.0709 |
| 2 | `obligation_ratio` | `(Σ ess + Σ debt) / Σ exp` | 0.8154 |
| 3 | `savings_rate` | `Σ savings / Σ income` | 0.1703 |
| 4 | `debt_to_income` | `Σ debt / Σ income` | 0.0830 |
| 5 | `discretionary_ratio` | `Σ other / Σ exp` | 0.2846 |
| 6 | `income_trend` | OLS slope of income | 463.09 |
| 7 | `expense_trend` | OLS slope of expenses | −167.83 |
| 8 | `volatility_index` | sample std of expenses | 2,638.58 |
| 9 | `category_entropy` | `−Σ p_c log₂ p_c` over categories with any spend | 2.585 |
| 10 | `transaction_frequency` | expense txns / n (synth cadence = 9 / month) | 9.00 |
| 11 | `avg_transaction_size` | `Σ exp / (9n)` = 179,108.75 / 45 | 3,980.19 |
| 12 | `income_regularity` | share of months with income > 0 | 1.00 |
| 13 | `expense_regularity` | share of months with expenses > 0 | 1.00 |
| 14 | `income_expense_gap` | `Σ income − Σ exp` | 36,762.58 |
| 15 | `essential_income_ratio` | `Σ ess / Σ income` | 0.5936 |
| 16 | `month_sin` | `sin(2π · 5 / 12)` | 0.5000 |
| 17 | `month_cos` | `cos(2π · 5 / 12)` | −0.8660 |
| 18 | `income_volatility_interaction` | CV × Σ income | 15,312.80 |
| 19 | `obligation_volatility_interaction` | obligation × CV | 0.0578 |

**Standardize.** For each feature, `z_i = (x_i − mean_i) / scale_i` using the artifact `StandardScaler` (train-only). That step is ordinary arithmetic; the 19 means and scales live in `models/pfp/tier3_svm.joblib`.

**RBF kernel (the formula the SVM actually uses).**

```text
K(z, sv) = exp(−γ ‖z − sv‖²)
γ        = 0.0597362     (artifact, sklearn `gamma='scale'`)
```

Hand-sized illustration with two support vectors in 2-D (`z = (0, 0)`, `γ` as above):

```text
sv₁ = (0, 0)     K = exp(−γ · 0) = 1.0000
sv₂ = (3, 0)     K = exp(−γ · 9) = exp(−0.537626) = 0.5841
```

A pair score is `f = b + Σ_sv α y K(z, sv)`. Sign(`f`) votes for one of the two classes. The production model has **2,161 support vectors** and **28** one-vs-one pairs (8 classes). Tallying those 28 scores is the same arithmetic; the sums themselves are read from the artifact. A full vote table for a 12-month persona A vector is in [`pen-and-paper-simulation.md`](pen-and-paper-simulation.md) §7.2b.

**Calibrate and serve.**

1. Platt sigmoid per class: `p_c = 1 / (1 + exp(A_c f_c + B_c))`, then renormalize to sum to 1. This is `predict_proba`.
2. Served class = `argmax_c p_c` (what `CalibratedClassifierCV.predict` returns).
3. Dimension scores:
   - `financial_stability_score  = Σ p_c` over classes starting with `Stable`
   - `financial_weight_score     = Σ p_c` over classes containing `Obligated`
   - `financial_tolerance_score  = Σ p_c` over classes containing `Tolerant`
4. `confidence = max p_c`.

**Decision rule (training, not serve-time).** Keep the SVM only if macro-F1 beats Tier-1 by more than 0.02. Held-out: SVM 0.6745 vs Tier-1 0.6050. Winner stands.

---

## 2. Forecasting v2 (pooled SARIMA)

**What it does.** Predicts the next month's *total expense* in pesos, then optionally stretches that total across a weekly / semi-monthly / yearly horizon and across categories.

v1 was a 12-month Gaussian series, so the seasonal term never identified and the saved `tier3_sarima` was plain ARIMA. v2 trains on 36 HFCE-calibrated months. The committed artifact `models/forecaster_v2/tier3_sarima.joblib` is a real `SARIMAX` with order `(1,1,0)×(1,0,0,12)`.

### 2.1 Training-time pool (context, not re-fit by hand)

For each user and month:

```text
z_{u,t} = expense_{u,t} / mean_t(expense_u)     (user's own full-history mean)
pool_t  = average of z_{u,t} over users that have month t
```

The artifact stores `pool_level = 1.0` (the pooled series mean) and `profile_level ≈ 32,615` (mean peso expense, used only for cold start).

Seasonal gate: if the pooled series spans fewer than 24 distinct months, drop `(P,D,Q,s)` and fit `ARIMA(1,1,0)`. v2's final fit has 35 observations, so the seasonal term is on.

### 2.2 Algorithm — one-step pooled forecast

**Model.**

```text
(1 − φ B) (1 − Φ B¹²) (1 − B) y_t  =  ε_t
```

with committed coefficients

```text
φ  = ar.L1    = −0.001141
Φ  = ar.S.L12 =  0.997774
```

Let `Δ_t = y_t − y_{t−1}`. Expanding and setting future noise to 0 gives the one-step recursion:

```text
Δ̂_{T+1} = φ Δ_T  +  Φ Δ_{T−11}  −  φ Φ Δ_{T−12}
ŷ_{T+1}  = y_T + Δ̂_{T+1}
```

**Inputs (last 14 pooled observations from the artifact).** `T = 35`.

| t | y_t | t | y_t |
| :---: | ---: | :---: | ---: |
| 22 | 1.128542 | 29 | 0.973728 |
| 23 | 1.127581 | 30 | 0.946455 |
| 24 | 0.945454 | 31 | 0.947820 |
| 25 | 0.947166 | 32 | 0.946631 |
| 26 | 0.946591 | 33 | 1.127001 |
| 27 | 0.974492 | 34 | 1.125866 |
| 28 | 0.972441 | **35** | **1.126788** |

**Step 1 — differences**

```text
Δ_T      = y_35 − y_34 = 1.126788 − 1.125866 =  0.000922
Δ_{T−11} = y_24 − y_23 = 0.945454 − 1.127581 = −0.182127
Δ_{T−12} = y_23 − y_22 = 1.127581 − 1.128542 = −0.000961
```

**Step 2 — recursion**

Because `|φ| ≈ 0`, the first and third terms are ~10⁻⁶. The seasonal term dominates:

```text
Φ Δ_{T−11} = 0.997774 × (−0.182127) = −0.181721
Δ̂_{T+1}    ≈ −0.181723
ŷ_{T+1}     = 1.126788 − 0.181723 = 0.945065
```

This matches `artifact["model"].forecast(1)` exactly (0.945065). Plain-language: next month ≈ this month plus almost the same month-over-month jump as twelve months ago. On this series that jump is the drop from a high-spend month into a low-spend month.

### 2.3 Algorithm — serve-time peso rescaling

Code: `app/services/forecast_service.py` (`_predict_monthly_total`, `_horizon_factor`, `_projection_weights`).

**Inputs.** User expense transactions; pooled forecast `pool_pred = 0.945065`; `pool_level = 1.0`.

1. **User level.** Sum expenses by real `(year, month)`, drop non-positive months, take the mean of the last three chronological months. If that mean is 0 (cold start), replace it with `profile_level`.
2. **Monthly total.**

```text
pred = (pool_pred / pool_level) × level
     = 0.945065 × level
```

3. **Multiplicative bands** (not statistical SARIMA intervals):

```text
CI80 = [0.80 × pred,  1.20 × pred]
CI95 = [0.60 × pred,  1.40 × pred]
```

4. **Horizon factor.**

| Horizon | Factor |
| :--- | ---: |
| `WEEKLY` | 7 / 30.44 |
| `SEMI_MONTHLY` | 15 / 30.44 |
| `MONTHLY` | 1 |
| `YEARLY` | 12 |

`predicted = pred × factor`. The same factor scales the bands.

5. **Spread across dates.** Weights are the user's historical share of spend in each bucket (weekday 0–6, first/second half of month, week-of-month 0–3, or calendar month 0–11). If the history is empty, use equal weights. Yearly weights are rolled so the first forecast month is next calendar month.
6. **Category split (optional).** `amount_{c,date} = predicted × share_c × weight_date`, where `share_c` is that category's share of historical expense.

**Worked example.** Last three positive-expense months: ₱38,000, ₱40,000, ₱42,000. Horizon `MONTHLY`, level `TOTAL`.

```text
level = (38,000 + 40,000 + 42,000) / 3 = 40,000
pred  = 0.945065 × 40,000 = 37,802.60
CI80  = [30,242.08,  45,363.12]
CI95  = [22,681.56,  52,923.64]
```

If the four week-of-month weights from history are `(0.20, 0.30, 0.25, 0.25)`, the four `ForecastPoint` amounts are 7,560.52, 11,340.78, 9,450.65, 9,450.65.

**Boxed answer:** next-month total **₱37,802.60**, 80% band **[₱30,242.08, ₱45,363.12]**.

**Winner rule (training).** Challenger MAPE must be at most 80% of naive MAPE. v2: naive 36.48%, SARIMA 6.12% (83.2% reduction). Winner stands.

---

## 3. Anomaly alerts (IQR)

**What it does.** Scores each expense transaction; operating rule is `score ≥ 0.125`. Winner is the univariate IQR detector (`app/ml/models.py`, `IQRDetector`). Learned alternatives failed the pre-registered test-set gate (PR-AUC ≥ 1.5× IQR and ≥ 0.15).

### 3.1 Feature vector (24 numbers)

For a candidate transaction, build a 3-month *baseline* from strictly earlier months (`feature_engineering_anomaly.py`, `baseline_months=3`). Then compute:

**Baseline (10)**

| Feature | Formula |
| :--- | :--- |
| `mean_income_rolling` | mean of baseline monthly income |
| `std_income_rolling` | std of baseline monthly income |
| `mean_expenses_rolling` | mean of baseline expense amounts |
| `std_expenses_rolling` | std of baseline expense amounts |
| `category_dist` | max category share in the baseline |
| `txn_frequency_rolling` | mean monthly expense-txn count in the baseline |
| `avg_txn_size_rolling` | same as mean baseline expense amount |
| `category_entropy` | `−Σ p_c log₂ p_c` over baseline category shares |
| `volatility_index` | `std_expenses / mean_expenses` |
| `spending_concentration` | Herfindahl `Σ p_c²` |

**Detection (14)**

| Feature | Formula |
| :--- | :--- |
| `amount_deviation` | `(amount − mean_exp) / std_exp` |
| `category_deviation` | `|p_category − 1/10|` |
| `frequency_deviation` | `(this-month count − mean_freq) / std_freq` |
| `income_deviation` | `(this-month income − mean_inc) / std_inc` |
| `expense_deviation` | `(this-month total exp − mean monthly total) / std` |
| `is_novel_category` | 1 if this category never appeared before this month |
| `amount_vs_category_mean` | `(amount − cat_mean) / cat_mean` |
| `amount_vs_category_std` | `(amount − cat_mean) / cat_std` |
| `category_frequency_change` | relative change in this category's monthly count |
| `amount_percentile_in_category` | share of prior amounts in this category that are smaller |
| `days_since_last_txn` | calendar days since the previous expense |
| `is_weekend` | 1 if Saturday or Sunday |
| `amount_zscore_overall` | `|amount_deviation|` |
| `amount_zscore_category` | `|amount_vs_category_std|` |

If there is no baseline month, every feature is 0 and the service refuses the request (`anomaly detection requires baseline transaction history`).

**Standardize.** `z_j = (x_j − μ_j) / σ_j` with train-only mean/std. The IQR bounds below are fitted on those z-scores.

### 3.2 Algorithm — IQR detector

**Fit (once, on the train matrix).** For each feature `j = 1 … 24`:

```text
Q1_j = 25th percentile of train column j
Q3_j = 75th percentile of train column j
IQR_j = Q3_j − Q1_j
lo_j  = Q1_j − 1.5 × IQR_j
hi_j  = Q3_j + 1.5 × IQR_j
```

Percentiles use numpy's linear method: for `n` sorted values the `p`-quantile sits at index `p(n−1)` (0-based) and interpolates.

**Score (each transaction).**

```text
flags = number of features with  z_j < lo_j  or  z_j > hi_j
score = flags / 24
```

**Alert.**

```text
flag if score ≥ 0.125     (that is 3 or more of 24 features out of bounds)
```

`0.125` is the pre-registered operating point in `models/anomaly/metadata.json` (val F2, precision floor 0.30 not met). Amount `> ₱2,000` is attached as an explanation tag in `_explanation`; it does not change the IQR score.

Whitelist: skip a transaction whose `transaction_id` or `category` is on the request whitelist before features are built.

### 3.3 Worked example (4 features, same rule)

Production uses 24 columns. The arithmetic is identical on a 4-feature slice, which is what you can quartile by hand. Training values (`n = 8`):

| i | A `amount_z` | B `amt vs cat mean` | C `days since` | D `entropy` |
| :---: | ---: | ---: | ---: | ---: |
| 1 | −0.8 | −0.5 | 1 | 2.3 |
| 2 | −0.4 | −0.2 | 2 | 2.4 |
| 3 | −0.2 | −0.1 | 3 | 2.4 |
| 4 | 0.0 | 0.0 | 3 | 2.5 |
| 5 | 0.1 | 0.0 | 4 | 2.5 |
| 6 | 0.3 | 0.1 | 5 | 2.5 |
| 7 | 0.5 | 0.2 | 6 | 2.6 |
| 8 | 0.9 | 0.3 | 8 | 2.6 |

Q1 sits at index `0.25 × 7 = 1.75`; Q3 at `0.75 × 7 = 5.25`.

**Feature A**

```text
Q1 = −0.4 + 0.75 × (−0.2 − (−0.4)) = −0.25
Q3 =  0.3 + 0.25 × ( 0.5 −  0.3)  =  0.35
IQR = 0.60
lo, hi = −0.25 − 0.90,  0.35 + 0.90 = [−1.15, 1.25]
```

**Feature B**

```text
Q1 = −0.125,  Q3 = 0.125,  IQR = 0.25
lo, hi = [−0.50, 0.50]
```

**Feature C**

```text
Q1 = 2.75,  Q3 = 5.25,  IQR = 2.50
lo, hi = [−1.00, 9.00]
```

**Feature D**

```text
Q1 = 2.400,  Q3 = 2.525,  IQR = 0.125
lo, hi = [2.2125, 2.7125]
```

Two test rows:

| txn | A | B | C | D | out-of-bounds | score |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| Phone bill | −0.46 | **−0.63** | 5 | **2.09** | B, D | 2/4 = 0.50 |
| Ordinary food | 0.10 | 0.05 | 3 | 2.40 | none | 0/4 = 0.00 |

`0.50 ≥ 0.125` → **flag the phone bill**. `0.00 < 0.125` → food is clean.

On the real 24-feature detector the same phone-shaped `frequency_change` example in the pipeline simulation lands at `4/24 = 0.1667 ≥ 0.125` (flagged). A same-month food control lands at `3/24 = 0.125` (exactly at the operating point). The distinguishing signal is the extra out-of-bounds feature.

**Boxed answer:** phone score **0.50**, **FLAG**; food score **0.00**, **not flagged**.

---

## 4. Budgeting (linear program)

**What it does.** Splits `available_funds` across categories. There is no training. Same inputs always produce the same allocation (`app/services/budget_service.py`).

### 4.1 Algorithm — write the LP

**Decision variables** (two per category `i = 1 … n`):

- `x_i` — pesos allocated to category `i`
- `d_i ≥ 0` — absolute deviation from that category's target

**Target.** `t_i = available_funds × target_ratio_i`. Missing ratios count as 0.

**Weights.** `w_i = max(priority_weight_i, 10⁻⁹)` so a zero weight cannot drop a category from the objective.

**Objective.**

```text
minimize   Σ_i  w_i d_i
```

**Always-on constraints.**

```text
Σ_i x_i  =  available_funds
d_i ≥ x_i − t_i
d_i ≥ t_i − x_i
d_i ≥ 0
```

In matrix form the two deviation inequalities are `x_i − d_i ≤ t_i` and `−x_i − d_i ≤ −t_i`.

**Bounds from restriction level.**

| Level | Bound on `x_i` | Meaning |
| :--- | :--- | :--- |
| `LOCKED` | `x_i = current_spend` | Already committed; cannot move |
| `PROTECTED` | `max(floor, current_spend) ≤ x_i ≤ max(ceiling, that lower bound)` | Cannot go below what is already spent or the floor |
| `FREE` | `floor ≤ x_i ≤ ceiling` | Optimizer may move inside the box |

**Solve.** HiGHS simplex / interior-point. On paper, bind the LOCKED equalities first, meet every target that still fits inside its bounds, and assign any residual pesos to the lowest-weight category that still has unused ceiling.

**Feasibility label** (after a successful solve):

| Status | Test |
| :--- | :--- |
| `INFEASIBLE` | solver failed, or a bound/lock is violated after rounding |
| `REDUCED` | constraints hold but `Σ x_i / funds < 1` |
| `FEASIBLE` | constraints hold and utilization ≈ 1 |

### 4.2 Worked example

Funds = ₱25,000. Four categories (this is the payload in `tests/test_budget.py`).

| Category | Level | floor | ceiling | current | weight | ratio | target `t` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| food | PROTECTED | 6,000 | 9,000 | 6,800 | 0.4 | 0.28 | 7,000 |
| rent | LOCKED | 8,000 | 8,000 | 8,000 | 0.3 | 0.32 | 8,000 |
| leisure | FREE | 0 | 5,000 | 2,000 | 0.2 | 0.15 | 3,750 |
| savings | FREE | 0 | 6,000 | 0 | 0.1 | 0.25 | 6,250 |

**Step 1 — bounds**

```text
food:    lo = max(6,000, 6,800) = 6,800,   hi = 9,000
rent:    lo = hi = 8,000
leisure: [0, 5,000]
savings: [0, 6,000]
```

**Step 2 — drop LOCKED rent**

```text
x_food + x_leisure + x_savings = 25,000 − 8,000 = 17,000
```

Unconstrained targets on those three sum to 7,000 + 3,750 + 6,250 = 17,000, but savings' ceiling is 6,000, so the ₱6,250 savings target is infeasible. Bind savings at 6,000 (`d_savings = 250`). Remaining ₱11,000 must cover food (target 7,000) and leisure (target 3,750). That is ₱250 extra.

**Step 3 — where the extra ₱250 goes**

Putting it on food costs `0.4 × 250 = 100`. Putting it on leisure costs `0.2 × 250 = 50`. Leisure is cheaper, and 3,750 + 250 = 4,000 is still under the 5,000 ceiling.

```text
x = (food 7,000,  rent 8,000,  leisure 4,000,  savings 6,000)
d = (0,            0,          250,            250)
objective = 0.2×250 + 0.1×250 = 75
```

**Step 4 — verify**

```text
sum x              = 25,000                         ✓
rent locked        = 8,000                          ✓
food in [6800, 9000]                                ✓
leisure in [0, 5000]                                ✓
savings in [0, 6000]                                ✓
d_i = |x_i − t_i|                                   ✓
```

Utilization = 1.0, constraint satisfaction = 1.0 → **FEASIBLE**. HiGHS returns this same point.

**Boxed answer:** `(7,000, 8,000, 4,000, 6,000)`, status **FEASIBLE**, objective **75**.

If the solver cannot find any `x` inside the bounds that sums to funds, the service raises `budget LP infeasible`. There is no silent floor-relaxation in the current code path; `REDUCED` is reserved for a successful solve that fails to spend everything because ceilings cap the sum.

---

## 5. Constant sheet

| Constant | Value | Where it lives |
| :--- | :--- | :--- |
| PFP rule cutoffs | CV `< 0.50` Stable; obligation `> 0.60` Obligated; runway `≥ 3` Tolerant | `generate_personas.py`, `RuleBasedClassifier` |
| PFP winner | calibrated RBF SVM, 19 features, 2,161 SVs, γ = 0.0597362 | `models/pfp/` |
| PFP decision rule | beat Tier-1 by `> 0.02` macro-F1 | `models/pfp/metadata.json` |
| Forecast v2 order | `(1,1,0)×(1,0,0,12)` | `train_forecaster.py`, artifact `kind: sarima` |
| Forecast v2 φ, Φ | `−0.001141`, `0.997774` | `models/forecaster_v2/tier3_sarima.joblib` |
| Forecast v2 pool_level | `1.0` | same artifact |
| Forecast serve formula | `(pool_pred / pool_level) × trailing-3-month mean` | `forecast_service.py` |
| Forecast bands | ×0.8/1.2 (80%), ×0.6/1.4 (95%) | `forecast_service.py` |
| Forecast decision rule | cut naive MAPE by `≥ 20%` | v2: 6.12% vs 36.48% |
| IQR multiplier | 1.5 | `IQRDetector` |
| IQR score | flags / 24 | `IQRDetector.score` |
| IQR threshold | 0.125 | `models/anomaly/metadata.json` |
| Anomaly decision rule | ML must reach PR-AUC `≥ 1.5×IQR` and `≥ 0.15` on test, else IQR | fallback fired |
| Budget solver | `linprog(..., method="highs")` | `budget_service.py` |
| Budget objective | `Σ w_i \|x_i − t_i\|` | same |
| Debt in summaries | `0.10 × monthly expenses` | `build_monthly_summaries` |

---

## 6. What is *not* claimed

- The SVM kernel over 2,161 vectors is not a reasonable hand sum. Algorithm D gives the exact formula and a 2-vector evaluation; the 28 pair scores for a production vector are read from the artifact.
- Forecast confidence bands are multiplicative service conventions, not SARIMA analytic intervals.
- The IQR detector is univariate. It does not model feature interactions.
- The budget LP is single-period. It does not plan across months.
- v2 forecast metrics are not a controlled bake-off against v1: horizon, fold count, and the expense generator all changed together.
