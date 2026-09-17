# PFP Classifier — Plain-Language Explanation

> **What it does:** Sorts each user into one of eight financial personality types based on their income, expenses, debts, and spending habits.

---

## In one sentence

The PFP (Personal Financial Profile) Classifier looks at how money flows in and out of a person's life and assigns them a three-part financial label that tells the app *who this person is financially*.

---

## Why it exists

Not everyone manages money the same way. A salaried employee with a fixed paycheck and a freelancer with irregular income need different advice. The PFP Classifier figures out which "financial personality" a user has so the rest of BUDI can tailor its recommendations accordingly.

---

## Input — What goes in

The classifier receives **19 numbers** that describe one month of the user's financial life. These are computed from their transaction history (income entries and expense entries). Here's what they measure in plain terms:

| Feature | What it actually measures |
|---|---|
| Income stability | How predictable is the user's income from month to month? (coefficient of variation) |
| Obligation ratio | What fraction of income goes toward fixed obligations like rent, loans, and bills? |
| Savings rate | How much of income is left over after all spending? |
| Debt-to-income | How large are existing debts compared to income? |
| Discretionary ratio | How much spending is on "wants" (food out, entertainment) vs. "needs"? |
| Income trend | Is income going up, down, or staying flat over recent months? |
| Expense trend | Same question, but for spending. |
| Volatility index | How "spiky" or unpredictable is the overall spending pattern? |
| Category entropy | How spread out is spending across categories (food, transport, bills, etc.)? |
| Transaction frequency | How many transactions happen per day on average? |
| Average transaction size | Typical amount per transaction. |
| Income regularity | How evenly spaced are income events? (lumpy freelance vs. smooth salary.) |
| Expense regularity | Same, for expenses. |
| Income-expense gap | The raw difference between money in and money out. |
| Essential-income ratio | What fraction of income covers essential (non-discretionary) spending? |
| Month-of-year sin/cos | Captures seasonal patterns (e.g., bigger spending around holidays). |
| Income × Volatility interaction | Captures the compounding effect of unstable income on overall volatility. |
| Obligation × Volatility interaction | Same, but for obligations — someone with high fixed costs and volatile income is in a different situation than someone with low fixed costs and volatile income. |

---

## Process — How it thinks

### 1. Training (one-time, offline)

The model was trained on thousands of synthetic financial profiles derived from real Philippine household survey data (FIES 2023, NCR households). The training pipeline tested five "tiers" of increasing complexity:

| Tier | Approach | Example |
|---|---|---|
| 0 | Just pick the most common label | "Everyone is Stable/Flexible/Tolerant" |
| 1 | Simple human-written rules | "If savings rate < 0 and obligations > 60%, you're At-Risk" |
| 2 | Basic machine learning | Logistic Regression, Naive Bayes |
| 3 | Stronger classifiers | Random Forest, Support Vector Machine (SVM) |
| 4 | Boosted trees | XGBoost |

A **decision rule** enforced at training time: the winning model must beat the simple rule-based Tier 1 by a meaningful margin (> 0.02 F1 score). If no learned model clears that bar, the system would fall back to rules.

**Winner: Tier 3 SVM** — a Support Vector Machine with a radial-basis-function (RBF) kernel, wrapped in probability calibration (`CalibratedClassifierCV`), with all features scaled to zero-mean/unit-variance via `StandardScaler`. It scored **0.6745 Macro-F1** on 5-fold cross-validation.

### 2. Inference (live, per user)

When a user logs in, the system:

1. **Computes the 19 features** from the user's recent transaction history.
2. **Feeds them through the SVM**, which outputs a predicted class and a probability for each of the eight classes.
3. **Collapses the eight probabilities into three dimension scores:**
   - **Financial Stability Score** — how stable vs. variable is their income/spending?
   - **Financial Weight Score** — how burdened are they by obligations?
   - **Financial Tolerance Score** — how much risk can they absorb?
4. **Reports confidence** as the highest class probability.

### Cold-start fallback

If the user has no transaction history yet (brand new account), the system uses a **questionnaire path**: three simple questions (income variability, obligation level, emergency fund runway) are mapped deterministically to one of the eight classes. No model involved — pure rules. This ensures every user gets a profile immediately, even before they start recording transactions.

---

## Output — What comes out

The classifier assigns one of **8 classes**, formed by combining three binary dimensions:

| Dimension | Options |
|---|---|
| **Stability** | `Stable` or `Variable` |
| **Obligation** | `Obligated` or `Flexible` |
| **Tolerance** | `Tolerant` or `At-Risk` |

Examples: `Stable/Flexible/Tolerant`, `Variable/Obligated/At-Risk`, etc.

In the API, these appear as `STABLE_FLEXIBLE_AT_RISK` (underscore-separated uppercase).

Alongside the class, the API returns:

- **Three dimension scores** (0–1 each): `financial_stability_score`, `financial_weight_score`, `financial_tolerance_score` — these are calibrated probabilities marginalized from the full eight-class output, giving downstream services a continuous measure of each dimension.
- **Confidence** (0–1): the model's certainty in its top pick.

### What the eight classes look like in practice

| Class | Typical user |
|---|---|
| `Stable/Flexible/Tolerant` | Regular income with low debt, healthy savings, and room to invest. |
| `Stable/Obligated/At-Risk` | Steady income but heavy loan payments — one emergency away from trouble. |
| `Variable/Flexible/Tolerant` | Variable income with few fixed commitments and a healthy financial buffer. |
| `Variable/Obligated/At-Risk` | Irregular income, high debts, thin margin — the most vulnerable profile. |

(And four more combinations fill the remaining rows of the 2×2×2 grid.)

---

## How other modules use this

The PFP class feeds into almost everything else in BUDI:

- **Forecaster** adjusts its confidence windows based on income stability.
- **Anomaly detector** calibrates its sensitivity — a "Variable" user naturally has more fluctuation, so the bar for what counts as "unusual" is different.
- **Budget optimizer** uses the profile to set initial constraint levels (how strictly to lock vs. protect categories).
- **Recommendation engine** selects which tips and nudges to surface.

---

## Honest limitations

- The **Macro-F1 of 0.6745** means the model gets roughly a third of classifications wrong on the held-out folds. This is respectable for an 8-class problem on noisy financial data, but it's not perfect. The dimension scores (continuous) are often more useful than the hard class label.
- The model was trained on **synthetic data derived from real survey statistics**, not on real app users. Real-world performance may differ, and the model will benefit from retraining on actual user data as the app matures.
- The questionnare cold-start path is **purely rule-based** and doesn't learn from outcomes.
