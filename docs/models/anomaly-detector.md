# Anomaly Detector — Plain-Language Explanation

> **What it does:** Watches each incoming transaction and flags the ones that look unusual compared to the user's normal spending behavior.

---

## In one sentence

The Anomaly Detector is a watchful eye over every transaction — it compares each one against what "normal" looks like for that specific user and raises a flag when something doesn't fit.

---

## Why it exists

Financial fraud, accidental double-charges, subscription price hikes, and impulse purchases that don't match a user's patterns — these all show up as transactions that look "off." The anomaly detector catches them early so the user can review and take action, and so other modules (like the budget optimizer) aren't thrown off by one-time spikes.

---

## Input — What goes in

Each time a transaction is scored, the detector receives **24 features** split into two groups:

### Baseline features (10 numbers — "what's normal for this user?")

These describe the user's *ongoing spending profile*:

| Feature | What it measures |
|---|---|
| Rolling income mean/std | Average and variability of income over recent windows. |
| Rolling expense mean/std | Same for spending. |
| Category distribution | The user's typical spending breakdown across categories. |
| Transaction frequency (rolling) | How many transactions per day, averaged over time. |
| Average transaction size (rolling) | Typical amount per transaction. |
| Category entropy | How evenly spread is spending across categories? |
| Volatility index | Overall unpredictability of the spending pattern. |
| Spending concentration | How much of total spending is dominated by one or a few categories. |

### Detection features (14 numbers — "how different is *this* transaction?")

These describe the *specific transaction* relative to the user's baseline:

| Feature | What it measures |
|---|---|
| Amount deviation | How far is this transaction's amount from the user's average? |
| Category deviation | Is this category one the user normally spends in? |
| Frequency deviation | Is the user transacting more or less often than usual recently? |
| Income/expense deviation | Does the overall spending level look elevated or depressed? |
| Is novel category | Has the user ever spent in this category before? |
| Amount vs. category mean | Is this amount normal *for this specific category*? |
| Amount vs. category std | Same, but measuring spread. |
| Category frequency change | Is the user suddenly buying in a category they rarely use? |
| Amount percentile in category | Where does this amount rank among the user's past transactions in this category? |
| Days since last transaction | How long since the previous transaction? (Very short gaps can be suspicious.) |
| Is weekend | Weekends often have different spending patterns. |
| Amount z-score (overall) | Standard deviations from the user's overall mean amount. |
| Amount z-score (category) | Standard deviations from the user's mean for this category. |

---

## Process — How it thinks

### 1. Training (one-time, offline)

The training pipeline tested multiple approaches in tiers:

| Tier | Approach | How it works |
|---|---|---|
| 0 | Majority vote | "Flag nothing" — always predict normal. |
| 1 | **IQR (Inter-Quartile Range)** | A simple statistical rule: if a value falls far outside the middle 50% of the data, it's anomalous. |
| 2 | Adaptive Threshold, Isolation Forest, One-Class SVM, Autoencoder | Four different learned approaches that try to model the user's "normal" and detect deviations from it. |
| 3 | Hybrid Ensemble | Average the scores of multiple models for a more robust signal. |

A **decision rule** enforced at training time (revised 2026.09.10 under Option A — see
`docs/thesis/anomaly-decision-rule-rationale.md`): the winner must reach **≥ 1.5× the IQR
baseline's PR-AUC** AND a **PR-AUC of at least 0.15**; otherwise fall back to IQR. The rule
is applied to the **held-out test split** (not just folds), so a candidate that looks good
on training folds must also prove it on unseen users. Operating point: the validation
threshold maximizing **F2** subject to **precision ≥ 0.30**.

**Winner: Tier 1 IQR** — the simple Inter-Quartile Range rule. Here's what happened honestly:

The learned models (Isolation Forest, One-Class SVM, Autoencoder, Ensemble) showed the best
PR-AUC on the training folds (hybrid 0.2753 vs IQR 0.0635) — a promising ranking gain. But
on the **held-out test split that gain collapsed**: the hybrid reached PR-AUC 0.0703 vs IQR
0.0550 (~1.28×, below the 1.5× gate) and did not reach 0.15. The pre-registered rule
therefore falls back to IQR, and this **fold-vs-test generalization gap is reported in
`models/anomaly/evaluation.json` → `test_validation`, not hidden**.

**Performance (honest state):**

| Metric | Value | What it means |
|---|---|---|
| F1 Score | 0.116 | Low — the model misses many anomalies and flags some normals. |
| Accuracy | 0.693 | Seems decent, but is misleading because most transactions are normal. |
| Precision | 0.064 | Only 6.4% of flagged transactions are actually anomalous. |
| Recall | 0.668 | It catches about 67% of real anomalies. |
| ROC-AUC | 0.708 | Moderate discrimination ability. |
| PR-AUC | 0.055 | Low, but the primary ranking metric (imbalance-safe; random = 3%). |

The low F1 is partly explained by the **synthetic anomaly injection** used during training — anomalies are planted in the data in specific patterns (amount spikes, new merchants, frequency changes, category mismatches) that may not perfectly reflect real-world fraud. As the app collects real user data, retraining is expected to improve these numbers significantly.

### 2. Inference (live, per transaction)

When a new transaction arrives:

1. **Compute the 24 features** — the 10 baseline features from the user's accumulated history, and the 14 detection features comparing this specific transaction against that baseline.
2. **Score the transaction** through the IQR detector:
   - For each feature, the IQR rule checks if the value falls outside the range `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` (the standard "box plot outlier" boundaries).
   - The overall anomaly score is a **per-feature deviation** combined into one raw score.
3. **Apply the threshold** — transactions with raw scores above the threshold (stored in the model artifact, calibrated on validation; current 0.125) are flagged as anomalous. Scores are compared in **raw units** — no per-request re-normalization, so the threshold's calibration units match the served scores.
4. **Generate an explanation** — the detector doesn't just say "anomalous," it explains *why*:
   - "Unusual amount" — the transaction amount is much higher/lower than normal.
   - "Novel category" — the user has never (or rarely) spent in this category.
   - "Frequency spike" — the user is transacting much more often than usual.
   - Feature contribution scores show which specific features drove the decision.
5. **Whitelist filtering** — transactions the user has previously approved as "not anomalous" are automatically skipped.
6. **Overspending detection** (separate sub-system) — compares category-level spending against budget allocations and flags when a category is being overspent relative to plan.

---

## Output — What comes out

The API returns for each scored transaction:

| Field | Description |
|---|---|
| **is_anomaly** (boolean) | `true` if the score exceeds the threshold. |
| **anomaly_score** (0–1) | Continuous score — higher means more anomalous. |
| **confidence** (0–1) | How sure the model is about its classification. |
| **explanation** (text) | A human-readable reason: "Unusual amount detected" or "Spending in a novel category." |
| **feature_contributions** | Which of the 24 features contributed most to the anomaly score, and by how much. |
| **category** | The type of anomaly detected (amount spike, novel category, frequency change, etc.). |

### What the user sees

When BUDI's frontend displays an anomaly alert, it shows:

- The flagged transaction (amount, merchant, date, category).
- A plain-language explanation ("This ₱15,000 charge at Electronics Mart is 4× your usual spending in that category.").
- Options: "This is normal" (whitelist it) or "This looks wrong" (flag for follow-up).

---

## How other modules use this

- **Budget Optimizer** can temporarily exclude flagged transactions from category totals to prevent one-time spikes from distorting allocations.
- **Forecasting** benefits from knowing which transactions are anomalous — excluding them from historical data produces cleaner forecasts.
- **PFP Classifier** uses anomaly rates as an input signal — a user with frequent anomalies may have a different financial profile than one with stable spending.

---

## Honest limitations

- **F1 of 0.116 is low.** The model is currently in a "better safe than sorry" mode — it catches most real anomalies (67% recall) but at the cost of many false alarms (6.4% precision). This is a known state, and the team chose transparency over inflating metrics.
- The IQR rule is **univariate** — it checks each feature independently and doesn't capture interactions (e.g., a moderate amount in a novel category at an unusual time might be more suspicious than any single feature alone).
- **Synthetic anomaly patterns** may not match real-world fraud, which is often more subtle and adversarial.
- The model requires **enough history** to establish a reliable baseline — very new users may see more (or fewer) false positives.
- The **whitelist system** relies on user judgment — if users whitelist real anomalies, the model can't learn from them.
