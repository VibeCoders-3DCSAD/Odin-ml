# Financial Classification V2

```json
{
  "document-type": "system-spec",
  "version": "1.0.0",
  "date": "2026.09.20",
  "authors": ["Group 4, III-DCSAD"]
}
```

> **Purpose:** Classification V2 is ODIN's deterministic financial-condition assessment. It applies published financial-resilience thresholds to authoritative user data. It does not use data procurement, synthetic data, model training, model artifacts, or probabilistic confidence scores.

---

## Service Contract

The microservice exposes Classification V2 at `POST /api/v1/classification/v2`.

Classification V2 is distinct from the legacy PFP classifier. The PFP classifier returns an eight-class personal financial profile from trained-model or questionnaire inference. Classification V2 returns four independent, explainable assessments of the user's current financial condition.

Each response records the calculation inputs, applied threshold, source or rule-set version, and a data-completeness state. If required inputs are absent or history is incomplete, the affected assessment returns `INSUFFICIENT_DATA`; it must not infer a favourable result.

| Assessment | Authoritative input source | Required data |
| :--- | :--- | :--- |
| Emergency savings | Emergency Savings feature | Liquid emergency-savings balance and recurring essential monthly expenses |
| Debt burden | Budget and Debt Manager | Monthly disposable income and required monthly debt payments |
| Financial margin | Budget and Debt Manager | Monthly disposable income, required monthly debt payments, and basic living costs |
| Credit-card behaviour | Credit-card account history | Twelve consecutive months of account balance history |

The application defines which user-entered budget categories are essential and which constitute basic living costs. Debt payments must be sourced from the Debt Manager's required or minimum payment values. Classification V2 must not derive debt payments from a proportion of total expenses.

---

## Scope and Threshold Provenance

Classification V2 adopts published international thresholds as ODIN operational decision-support defaults. Equivalent validated Philippine household thresholds were not identified in the available literature; ODIN does not create or calibrate substitute thresholds from synthetic or unvalidated data.

The threshold set is versioned in the service configuration for reproducibility, not empirical tuning. The initial rule-set is `financial_rules_v1`:

| Rule | Value |
| :--- | :--- |
| Emergency fund adequacy | 3 months of recurring essential expenses |
| DSTI overburdened boundary | Greater than 40% |
| Financial margin boundary | Less than 0 |
| Credit-card observation window | 12 months |

---

## 1. Emergency Savings Classification

Emergency Fund Coverage (EFC) measures how many months of recurring essential expenses can be covered by available liquid emergency savings. Bhutta et al. (2023) use three months of non-discretionary expenses as an emergency-savings adequacy benchmark. The Board of Governors of the Federal Reserve System (2024) also uses three months of expenses as a measure of financial resilience.

$$
\text{EFC} =
\frac{\text{Liquid Emergency Savings}}
{\text{Average Monthly Recurring Essential Expenses}}
$$

*Formula adapted from the emergency-savings coverage measure discussed by Bhutta et al. (2023).*

| Classification | Quantifiable Rule |
| :--- | :--- |
| **Emergency Fund Inadequate** | $EFC < 3$ months |
| **Emergency Fund Adequate** | $EFC \geq 3$ months |

*Classification threshold based on Bhutta et al. (2023) and Board of Governors of the Federal Reserve System (2024).*

### References

Bhutta, N., Blair, J., & Dettling, L. J. (2023). The smart money is in cash? Financial literacy and liquid savings among U.S. families. *Journal of Accounting and Public Policy, 42*(2), 107000. https://doi.org/10.1016/j.jaccpubpol.2022.107000

Board of Governors of the Federal Reserve System. (2024). *Report on the economic well-being of U.S. households in 2023*. https://www.federalreserve.gov/publications/2024-economic-well-being-of-us-households-in-2023.htm

---

## 2. Debt Burden (DSTI) Classification

The Debt-Service-to-Income Ratio (DSTI) measures the proportion of monthly disposable income committed to required debt payments. He and Zhou (2022) discuss DSTI as a household financial-vulnerability indicator and report prior research using a 40% threshold to identify financially vulnerable households.

$$
\text{DSTI} =
\frac{\text{Required Monthly Debt Payments}}
{\text{Monthly Disposable Income}}
\times 100
$$

*Formula based on the DSTI measure discussed by He and Zhou (2022). The 40% classification boundary is adopted from prior research discussed in that study.*

| Classification | Quantifiable Rule |
| :--- | :--- |
| **Not Financially Overburdened** | $DSTI \leq 40\%$ |
| **Financially Overburdened** | $DSTI > 40\%$ |

Monthly disposable income is the value used by the user's budget. An absent or non-positive denominator produces `INSUFFICIENT_DATA`, not a ratio.

*Classification threshold based on He and Zhou (2022); the labels are ODIN's operational classifications of debt burden.*

### Reference

He, L., & Zhou, S. (2022). Household financial vulnerability to income and medical expenditure shocks: Measurement and determinants. *International Journal of Environmental Research and Public Health, 19*(8), 4480. https://doi.org/10.3390/ijerph19084480

---

## 3. Financial Margin Classification

Financial Margin (FM) measures the disposable income remaining after required debt payments and basic living costs. He and Zhou (2022) discuss negative financial margin as an indicator used in household financial-vulnerability research.

$$
\begin{aligned}
\text{FM} ={}& \text{Monthly Disposable Income} \\
&- \text{Required Monthly Debt Payments} \\
&- \text{Monthly Basic Living Costs}
\end{aligned}
$$

*Formula based on the financial-margin measure discussed by He and Zhou (2022).*

| Classification | Quantifiable Rule |
| :--- | :--- |
| **Positive Financial Margin** | $FM \geq 0$ |
| **Negative Financial Margin** | $FM < 0$ |

*The classifications apply the sign of the financial margin discussed by He and Zhou (2022). For ODIN, these categories describe the financial-margin state rather than overall financial vulnerability.*

### Reference

He, L., & Zhou, S. (2022). Household financial vulnerability to income and medical expenditure shocks: Measurement and determinants. *International Journal of Environmental Research and Public Health, 19*(8), 4480. https://doi.org/10.3390/ijerph19084480

---

## 4. Credit Card Behavior Classification

Credit-card behavior is classified according to the number of months an account maintains a revolving balance during the preceding 12 months. Adams et al. (2022) classify accounts as transactors, light revolvers, and heavy revolvers based on their revolving-balance history.

$$
R = \sum_{m=1}^{12} I(\text{Revolving Balance}_m > 0)
$$

Where:

- $R$ is the number of months with a revolving balance during the preceding 12 months.
- $I(\cdot)$ is an indicator function that returns 1 when a revolving balance exists and 0 otherwise.

*The formula is ODIN's mathematical representation of the 12-month revolving-frequency classification described by Adams et al. (2022); it is not presented as a formula printed verbatim in that publication.*

| Classification | Quantifiable Rule |
| :--- | :--- |
| **Transactor** | $R = 0$ |
| **Light Revolver** | $1 \leq R \leq 11$ |
| **Heavy Revolver** | $R = 12$ |

The classification requires twelve consecutive observed balance months. Missing months produce `INSUFFICIENT_HISTORY`; they must not be treated as zero revolving balances.

*Classification rules based on Adams et al. (2022), who distinguish the three account types using the number of months with revolving balances during the preceding 12 months.*

### Reference

Adams, R. M., Bord, V. M., & Katcher, B. (2022). Credit card profitability. *FEDS Notes*. Board of Governors of the Federal Reserve System. https://doi.org/10.17016/2380-7172.3100
