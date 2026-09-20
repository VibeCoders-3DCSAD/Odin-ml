# Anomaly Alerts V2 Methodology

```json
{
  "document-type": "system-spec",
  "version": "2.0.0",
  "date": "2026.09.20",
  "authors": ["Group 4, III-DCSAD"]
}
```

> **Purpose:** V2 detects statistically unusual spending using each user's own transaction history. It does not use a trained anomaly-classification model, a global anomaly baseline, synthetic transactions, or another user's financial data for production anomaly decisions.

---

## Methodology Overview

V2 uses Personalized Dual-Channel Anomaly Detection:

1. Transaction-Level IQR Detector.
2. Forecast-Residual IQR Detector.

```text
Actual user transaction history
            |
     +------+------+
     |             |
Transaction IQR  Forecast-residual IQR
     |             |
     +------+------+
            |
       Anomaly alert
```

The system identifies statistical deviations from a user's historical behaviour. An alert does not mean that a transaction is fraudulent, irresponsible, or unauthorized.

---

## Production Data Boundary

Every production anomaly decision is based only on the current user's chronologically earlier transactions and, for the forecast channel, that user's historical forecast residuals.

| Input | Permitted in production anomaly decision | Notes |
| :--- | :---: | :--- |
| Current user's prior transactions | Yes | The personal transaction-IQR baseline. |
| Current user's prior monthly expenditure | Yes | The forecast input and residual-IQR baseline. |
| Current user's prior forecast residuals | Yes | The forecast-residual IQR baseline. |
| Other users' transactions | No | No population-wide anomaly reference is used. |
| Synthetic transactions | No | They must not form a production personal baseline. |
| Precomputed anomaly-model thresholds | No | Boundaries are calculated from the user history. |

Forecast-development data remains separate from production anomaly data. It may be used to develop and evaluate a forecasting model, but it must not become a substitute for the user's personal anomaly baseline.

---

## Cold Start Policy

Anomaly alerts are disabled until a user has sufficient usable history for stable quartile and forecast-residual estimates:

```text
n_u < N_min -> anomaly alerts disabled
```

Where `n_u` is the usable-observation count for user `u`, and `N_min` is the validated minimum history requirement. The value must be chosen experimentally, not arbitrarily, by testing the stability of Q1, Q3, IQR, forecast errors, and forecast accuracy across candidate windows such as 3, 6, 9, 12, 18, and 24 months.

When history is insufficient, the service must return an explicit insufficient-history state and must not manufacture an alert, a normal result, or a population-based fallback.

---

## Channel 1: Transaction-Level IQR

### Purpose

This channel answers:

> Is this transaction unusual compared with this user's earlier transactions in the same category?

Represent a transaction as:

```text
T_i = (a_i, c_i, t_i)
```

Where `a_i` is amount, `c_i` is category, and `t_i` is timestamp.

### Personal Category Baseline

For user `u` and category `c`, the historical amount set is:

```text
A_(u,c) = {a_1, a_2, ..., a_n}
```

The detector uses category-specific personal IQRs, `IQR_(u,c)`, rather than comparing categories with materially different expected scales, such as housing and transport. The baseline must contain only transactions that occurred before the candidate transaction.

### Log Transformation

Because transaction amounts may be positively skewed, the detector transforms each amount before computing quartiles:

```text
x_i = ln(1 + a_i)
```

The transformed personal category history is:

```text
X_(u,c) = {x_1, x_2, ..., x_n}
```

### IQR Boundaries

For each eligible user-category history:

```text
Q1_(u,c) = P25(X_(u,c))
Q3_(u,c) = P75(X_(u,c))
IQR_(u,c) = Q3_(u,c) - Q1_(u,c)

L_(u,c) = Q1_(u,c) - 1.5 * IQR_(u,c)
U_(u,c) = Q3_(u,c) + 1.5 * IQR_(u,c)
```

The candidate transaction is anomalous when:

```text
A_i_txn = 1 when x_i < L_(u,c) or x_i > U_(u,c)
A_i_txn = 0 otherwise
```

### Severity

The channel may calculate a continuous severity for prioritization and explanation:

```text
D_i_txn = max(L_(u,c) - x_i, 0, x_i - U_(u,c)) / (IQR_(u,c) + epsilon)
```

`D_i_txn = 0` means the transaction is within the personal historical range. Larger values indicate increasing distance beyond the applicable IQR fence.

---

## Channel 2: Forecast-Residual IQR

### Purpose

This channel answers:

> Is aggregate spending unusually different from what Odin expected from this user's historical spending pattern?

It identifies unusual aggregate spending that may not be visible in any single transaction.

### Monthly Expenditure and Forecast

Aggregate a user's actual transactions by category and month:

```text
E_(u,c,m) = sum(amount_i for user u, category c, month m)
```

The forecasting service produces a category-month forecast:

```text
E_hat_(u,c,t)
```

The forecast residual is:

```text
r_(u,c,t) = E_(u,c,t) - E_hat_(u,c,t)
```

A positive residual means actual spending exceeded forecast; a negative residual means actual spending was below forecast. A large residual alone is not an anomaly. It is compared with the user's own earlier forecast errors for that category.

### Residual IQR Boundaries

For historical residuals `R_(u,c) = {r_1, r_2, ..., r_n}`:

```text
Q1_r = P25(R_(u,c))
Q3_r = P75(R_(u,c))
IQR_r = Q3_r - Q1_r

L_r = Q1_r - 1.5 * IQR_r
U_r = Q3_r + 1.5 * IQR_r
```

The forecast channel triggers when:

```text
A_(u,c,t)_forecast = 1 when r_(u,c,t) < L_r or r_(u,c,t) > U_r
A_(u,c,t)_forecast = 0 otherwise
```

This means the forecasting error is unusual for that user and category. It avoids arbitrary rules such as declaring every expenditure 30% above forecast anomalous.

---

## Combined Alert Decision

The channels remain independent and are not combined with arbitrary weighted scores. The overall decision is:

```text
A_overall = A_txn OR A_forecast
```

| Transaction IQR | Forecast-residual IQR | Meaning |
| :---: | :---: | :--- |
| 0 | 0 | No detected anomaly. |
| 1 | 0 | Unusual individual transaction. |
| 0 | 1 | Unusual aggregate spending pattern. |
| 1 | 1 | Both transaction and aggregate-spending deviation. |

---

## Alert Language

Alerts must describe statistical deviation rather than motive or blame.

| Channel | Recommended language |
| :--- | :--- |
| Transaction-level | "This transaction is outside your usual spending range for this category." |
| Forecast-level | "Your spending in this category is outside the normal range of your historical forecast errors." |
| Both channels | "This transaction is unusually large compared with your normal spending, and your total category expenditure is also different from its expected pattern." |

The system must not claim that a transaction is fraudulent, that a user is spending irresponsibly, that a purchase is impulsive, or that money was stolen.

---

## Evaluation and Dataset Separation

Maintain separate datasets and purposes:

| Dataset | Purpose |
| :--- | :--- |
| `forecast_train` | Train and select the forecasting model using clean synthetic monthly expenditure. |
| `forecast_test` | Evaluate forecasting accuracy on unseen clean monthly expenditure. |
| `anomaly_test` | Evaluate the transaction and forecast-residual IQR rules with controlled deviations and known labels. |
| Production user history | Build each user's live transaction and residual baselines. |

Forecast training data must contain no anomaly injection, arbitrary Gaussian monthly noise, arbitrary percentage variation, fabricated transaction frequency, or fabricated transaction-size distributions. Anomalies must never be injected into forecast-training data.

Evaluation must be chronological:

```text
Historical observations
|
+-- training: fit the forecasting model and estimate initial transaction IQRs
+-- validation: tune the forecasting model and establish residual IQRs
`-- test: evaluate final anomaly detection
```

The test period must not recompute boundaries used for its own evaluation. Report precision, recall, and F1 as primary anomaly metrics; accuracy is secondary because anomalies are uncommon.

---

## Training Clarification

IQR does not train like an SVM, random forest, neural network, or global anomaly classifier. It statistically fits quartiles and Tukey fences from a user's own eligible history:

```text
user history -> Q1, Q3, IQR, lower fence, upper fence -> alert decision
```

This per-user boundary estimation is sometimes informally called IQR "training," but it is not machine-learning model training. The system does not save global IQR bounds and apply them to unrelated users.

The forecast-residual channel is different: it depends on a forecasting model. That model may require offline training and evaluation, but the residual IQR threshold used to decide whether a user's spending pattern is unusual remains personal and is calculated from that user's historical forecast errors.

---

## Final Methodology Label

**Personalized Dual-Channel Expenditure Anomaly Detection using Transaction-Level IQR and Forecast-Residual IQR**

This methodology separates individual transaction deviation from aggregate spending-pattern deviation while preserving the distinction between forecast-development data, anomaly-evaluation data, and actual production user history.
