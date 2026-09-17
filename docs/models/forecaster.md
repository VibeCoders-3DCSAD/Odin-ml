# Spending Forecaster — Plain-Language Explanation

> **What it does:** Predicts how much a user will spend in the future, broken down by category and spread across a time horizon they choose.
>
> **v1 evaluation.** The figures below (MAPE 9.40%, five folds, 12-month Gaussian-noise corpus) describe the original synthetic generator. The HFCE-calibrated v2 training run — including what is empirically backed, when SARIMA actually fits a seasonal term, and the 2026-09-17 metrics — is in [`forecaster-v2.md`](forecaster-v2.md).

---

## In one sentence

The Spending Forecaster looks at a user's past transactions, figures out their spending rhythm, and projects it forward — telling them (and the rest of BUDI) what their expenses are likely to look like next week, next month, or next year.

---

## Why it exists

You can't budget well if you don't know what's coming. The forecaster gives BUDI a forward-looking view of spending so that the budget optimizer can plan ahead, the anomaly detector can compare actuals against expectations, and the user gets a realistic picture of their financial trajectory.

---

## Input — What goes in

The forecaster works with a user's **transaction history** — a chronological list of income and expense entries. From these, it computes **20 daily-level features** that capture the shape of spending over time:

| Feature | What it measures |
|---|---|
| Day-of-week sin/cos | Captures weekly rhythm (e.g., higher spending on weekends). |
| Day of month | Where in the month are we? (Payday effects, end-of-month tightening.) |
| Lag values (1, 7, 14, 15, 30, 60 days) | How much was spent N days ago? Captures recent and cyclical patterns. |
| Rolling mean (7, 14, 30 days) | Average spending over recent windows — smooths out daily noise. |
| Rolling std (7, 14, 30 days) | How volatile is spending in those same windows? |
| Is payday | Boolean flag — spending often spikes on paydays. |
| Days to payday | How many days until next expected paycheck? (Spending often drops as payday approaches.) |
| Recency | Days since the last transaction. |
| Frequency (30d) | Number of transactions in the past 30 days. |
| Monetary (30d) | Total amount spent in the past 30 days. |

---

## Process — How it thinks

### 1. Data normalization (the clever part)

Directly comparing raw peso amounts across users doesn't work well — a ₱50,000/month spender and a ₱10,000/month spender live in different scales. The forecaster solves this with **user normalization**:

- For each user, compute their **trailing 3-month average positive expense** (their personal spending scale).
- Divide all of that user's monthly expenses by this personal scale.
- This puts everyone on roughly the same ~1.0 scale, letting the model learn patterns that generalize across income levels.

### 2. Training (one-time, offline)

The training pipeline tested several approaches:

| Tier | Approach | What it does |
|---|---|---|
| Baseline | Naive / trailing mean | "Next month will be like last month" |
| 2 | Random Forest regressor | Learns from the 20 daily features to predict monthly totals |
| 3a | **SARIMA** (Seasonal ARIMA) | A statistical time-series model that captures trends and seasonality |
| 3b | GRU neural network | A lightweight recurrent neural network (hidden size 32, sequence length 3, 30-day lookback) |

A **decision rule** enforced at training time: the winning learned model must reduce the naive baseline's MAPE (Mean Absolute Percentage Error) by at least 20%.

**Winner: Tier 3a SARIMA** — a pooled, user-normalized Seasonal ARIMA model. Here's how "pooled" works:

1. Take all users' normalized monthly expense series and stack them into one long pooled series (marked by user boundaries so the model knows where one user ends and the next begins).
2. Fit a **low-order ARIMA** model — specifically `ARIMA(1,1,0)` with a seasonal component `Seasonal Order (1,0,0,12)` (monthly seasonal pattern). If the pooled series spans fewer than 24 months, it degrades to a plain ARIMA without the seasonal component.
3. The model learns aggregate spending dynamics (mean reversion, trend, seasonal effects) that are **shared across all users**, then gets personalized at inference time through rescaling.

**Performance (v1 corpus):** MAPE of **9.40%** — a **74.9% improvement over the naive baseline**. Also strong on secondary metrics: R² = 0.80, MDA (Mean Directional Accuracy) = 0.71 (it picks the right up/down direction 71% of the time). That v1 series was only 12 months long, so the seasonal ARIMA gate never opened and the saved `tier3_sarima` fit was plain ARIMA. See [`forecaster-v2.md`](forecaster-v2.md) for the HFCE 36-month run.

> **Current behavior callout (so nobody is surprised):**
> - The model order is `(p, d, q) = (1, 1, 0)` (d = 1) with seasonal order
>   `(P, D, Q, m) = (1, 0, 0, 12)` (D = 0, m = 12) — see `training/scripts/train_forecaster.py:129,411,938`.
> - The seasonal (SARIMAX) branch only activates when the pooled series spans **≥ 24 distinct
>   absolute months**. The current 12-month synthetic corpus (single year, 2023) therefore always
>   fits a plain `ARIMA(1,1,0)` — SARIMA and ARIMA are, by construction, the same model on today's
>   data (`models/forecaster/evaluation_report.md` shows identical metrics for both tiers).
> - Pooling groups by an **absolute month index** (`(year−2023)·12 + month`), so two Januaries from
>   different years never collapse into one row; the seasonal term becomes reachable once the data
>   spans two years.
> - At serving, the "level" multiplier is the user's mean of the **last 3 chronological**
>   positive-expense months (not calendar months 10–12).

### 3. Inference (live, per user)

When a forecast is requested:

1. **Build the user's normalized monthly series** from their transaction history.
2. **Feed it through the pooled SARIMA model**, which produces a point forecast and confidence intervals in normalized space.
3. **Rescale** the forecast back to the user's actual peso amounts by multiplying by their personal trailing expense average. If the user has no positive expense months (cold start), a **profile-level prior** from the training data is used instead — so the forecast is never literally zero.
4. **Choose the horizon** based on the requested period:
   - Weekly → multiply monthly forecast by an appropriate factor
   - Semi-monthly, monthly, yearly → similar scaling
5. **Spread the total across dates** using the user's observed spending rhythm (`_projection_weights`): if they tend to spend more on weekends, the weekend dates in the forecast get proportionally more.
6. **Split by category** using the user's historical category proportions (`_category_proportions`): if they normally spend 40% on food and 20% on transport, the forecast inherits those ratios.

---

## Output — What comes out

The API returns a structured forecast with:

| Field | Description |
|---|---|
| **ForecastPoint** (per date) | A predicted spending amount for each date in the forecast horizon. |
| **ConfidenceInterval** | An 80% and 95% range around each point — "we're 80% sure spending will be between X and Y." |
| **ForecastLevel** | Granularity: `TOTAL` (one number for the whole period), `CATEGORY_GROUP` (by broad category like "Essentials"), or `CATEGORY` (specific: food, transport, bills, etc.). |
| **Period info** | Start date, end date, frequency (weekly/monthly/etc.). |

### Example output

```
Period: October 2026 (monthly)
Total forecast: ₱32,450
  80% CI: ₱27,100 – ₱37,800
  95% CI: ₱24,200 – ₱40,700

By category:
  Food & Groceries:  ₱12,980 (40%)
  Transport:         ₱ 6,490 (20%)
  Bills & Utilities: ₱ 8,115 (25%)
  Discretionary:     ₱ 4,865 (15%)
```

---

## How other modules use this

- **Budget Optimizer** uses the forecast to set realistic category ceilings — "based on your pattern, you'll likely spend ₱13K on food next month, so let's budget around that."
- **Anomaly Detector** compares each day's actual spending against the forecast's expected values to flag deviations.
- **PFP Classifier** benefits indirectly — the forecaster's confidence intervals inform stability assessments.
- **User dashboard** displays the forecast as a "what to expect" widget, helping users see whether they're on track.

---

## Honest limitations

- **9.40% MAPE** is strong for a pooled model, but it means individual predictions can be off by roughly ₱3K–5K per ₱35K spender. The confidence intervals are there to communicate this uncertainty.
- The model assumes **past patterns repeat** — it can't predict one-time events (a medical emergency, a holiday splurge, a new recurring expense).
- **Cold-start users** with very little history get a generic forecast based on their income bracket, which won't be personalized until a few months of data accumulate.
- Category splits are **ratio-based** (they inherit historical proportions), not independently predicted. If a user shifts spending categories, the forecast will lag.
