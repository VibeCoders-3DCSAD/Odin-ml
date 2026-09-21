# Survey-Only Forecast v3 Data Pipeline

```json
{
  "document-type": "workflow",
  "version": "1.0.0",
  "date": "2026.09.20",
  "authors": ["Group 4, III-DCSAD"]
}
```

## Overview

V3 is an unserved forecasting research pipeline. It begins with observed 2023 FIES Volume 2 annual household expenditure, allocates those annual amounts across synthetic months using published PSA HFCE quarterly profiles, and evaluates next-month expenditure candidates. It does not generate personas, PFP labels, income, debt, savings, or behavioral variables.

---

## Data Flow

```text
FIES Volume 2 Household Summary
  -> household cleaning and pseudonymization
  -> six observed annual expenditure categories
  -> PSA HFCE quarterly temporal allocation
  -> 42 synthetic household-month totals
  -> household-disjoint train / validation / test splits
  -> strictly-prior normalized lag features
  -> global Random Forest ratio training and PHP evaluation
```

---

## Household Cleaning

Source file:

```text
FIES PUF 2023 Volume2 Household Summary.CSV
```

A household is retained when `SEQ_NO` is unique and present, `TOTEX` and household size are positive, and the five direct expenditure categories are non-negative. Missing, negative, duplicate, and invalid rows are excluded; no expenditure value is imputed or winsorized.

The raw FIES identifier is never exported:

```text
SEQ_NO -> SHA-256 -> household_id_v3
```

The retained annual categories are:

| V3 category | FIES source | Construction |
| :--- | :--- | :--- |
| `food` | `FOOD` | Direct annual FIES value. |
| `housing_water` | `HOUSING_WATER` | Direct annual FIES value. |
| `health` | `HEALTH` | Direct annual FIES value. |
| `transport` | `TRANSPORT` | Direct annual FIES value. |
| `education` | `EDUCATION` | Direct annual FIES value. |
| `other` | `TOTEX` | `TOTEX - five direct categories`. |

The required accounting invariant is:

```text
food + housing_water + health + transport + education + other = TOTEX
```

See [the v3 selection contract](survey-only-forecast-v3.md) for the locked rules.

---

## Monthly Allocation

The annual FIES category anchor is allocated separately for every supported calendar year using PSA current-price HFCE category values from `training/config/hfce_quarterly_indices.json`.

For category `c`, year `y`, and quarter `q`:

```text
quarter_weight[c, y, q] = HFCE[c, y, q] / sum(HFCE[c, y, available quarters])

quarter_amount[h, c, y, q] = annual_amount[h, c] * quarter_weight[c, y, q]

monthly_amount[h, c, y, m] = quarter_amount[h, c, y, q] / 3
```

`other` uses the sum of PSA alcohol/tobacco, clothing, furnishings, communication, recreation, restaurants/hotels, and miscellaneous series before quarter weights are calculated.

The supported timeline is:

```text
2023-01 through 2026-06
```

Each household has 42 monthly rows. Complete 2023, 2024, and 2025 category totals reconcile to the observed annual FIES amount. The 2026 Q1-Q2 allocation uses only the available six PSA-supported months; July through December 2026 are never created.

For each household month:

```text
total_expenses =
  food + housing_water + health + transport + education + other
```

These are synthetic monthly allocations of observed annual expenditure, not observed household transaction histories.

---

## Splits And Features

Splits are assigned by `household_id_v3`, never by individual row:

```text
70% train households
15% validation households
15% test households
```

All months for one household remain in exactly one split.

For target month `T`, the RF row contains only data known before `T`:

| Field | Definition |
| :--- | :--- |
| `user_scale` | Mean expense across `T-1`, `T-2`, and `T-3`. |
| `lag_1_ratio` | `expenses[T-1] / user_scale`. |
| `lag_2_ratio` | `expenses[T-2] / user_scale`. |
| `lag_3_ratio` | `expenses[T-3] / user_scale`. |
| `rolling_std_3_ratio` | Standard deviation of the three prior expenses divided by `user_scale`. |
| `target_expenses` | `total_expenses` at `T`, retained for PHP evaluation. |
| `target_ratio` | `target_expenses / user_scale`, the RF label. |

Rows without three prior completed months, a positive scale, or a target are excluded. A complete 42-month household series therefore yields 39 RF-eligible rows.

---

## Random Forest Training

The Random Forest receives every RF-eligible row from train households and predicts `target_ratio`, not an absolute PHP amount. It builds decision-tree rules such as:

```text
if lag_1_ratio is high and rolling_std_3_ratio is low,
predict a next-month ratio slightly above 1.0
```

Each resulting prediction is converted to PHP only after prediction:

```text
forecast_php[T] = forest(features[T]) * user_scale[T]
```

Random Forest does not need `StandardScaler`: tree threshold splits are unchanged by monotonic feature rescaling. User-relative ratios are used to make household behavior comparable, not to satisfy an RF numerical requirement. Full-corpus RF uses a disk-backed feature matrix to avoid loading the full train matrix as a pandas DataFrame. Trees are added in five-tree warm-start blocks so logs report `Completed RF trees: N/100`.

---

## Evaluation And Limits

RF and naive MAE are evaluated on the same eligible test-household rows and reported in PHP:

```text
MAE = mean(abs(predicted PHP total - target_expenses))
```

The full-corpus commands write separate records under:

```text
models/forecaster_v3/candidates/
```

`compare_forecaster_v3_full.py` ranks only the naive and RF reports recorded under the same normalized three-prior-month evaluation contract. Legacy ARIMA, SARIMA, and LSTM reports are research comparisons only; they do not alter the RF feature or scaling contract.

These results are internal synthetic-target evaluation only. A lower MAE demonstrates better prediction of the PSA-allocated target under this pipeline; it does not establish real household transaction forecasting accuracy or authorize replacement of the served v2 forecaster. External validation remains required.
