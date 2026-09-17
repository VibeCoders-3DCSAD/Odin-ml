# Synthetic Household Expenditure Generation Methodology

```json
{
  "document-type": "methodology",
  "version": "1.1.3",
  "date": "2026.09.17",
  "authors": ["Group 4, III-DCSAD"]
}
```

## Implementation Bridge (Synthetic Generation v2)

This methodology is implemented in code as **Synthetic Generation v2**, a **parallel**
pipeline living alongside the original (v1) synthetic generator:

1. **Parallel, not a replacement.** `training/scripts/generate_personas.py`,
   `generate_transactions.py`, `synthesizer.py`, and `preprocessor.py` (v1) remain
   available and behaviorally unchanged. Synthetic Generation v2 ships as new modules
   (`temporal_disaggregation.py`, `generate_transactions_v2.py`, `synthesizer_v2.py`)
   that implement the FIES-anchored, HFCE-calibrated temporal disaggregation described
   below. Nothing in v1 is edited, deprecated, or deleted by v2.
2. **\(A_c\) is FIES-calibrated, not a 1:1 FIES `SEQ_NO` replay.** In v2.0, the annual
   household-category benchmark \(A_{h,c}\) used for disaggregation is
   \(A_{h,c} = \text{persona monthly category expense} \times 12\), where the persona's
   monthly category expense is itself derived from FIES-calibrated archetypes (via v1's
   `generate_all_personas`). Those 12 archetypes (A–L) take numerical baselines from
   2023 FIES NCR and behavioral patterns from the BSP 2021 Consumer Finance Survey;
   the roster was reviewed by a general-finance subject-matter expert (Asst. Prof.
   Pamela A. Go, CBFS) against the Odin-Paper SME draft
   (`docs/ml/1_problem-statement/persona-validation-list-SME-draft.md`). This is
   **not** a direct row-level replay of individual FIES `SEQ_NO` household records —
   it is the FIES-calibrated, SME-reviewed annual anchor for that persona.
3. **`Other` is the residual.** \(H_{\text{Other},q}\) = quarterly Total HFCE minus the
   five essential categories (Food, Housing, Health, Transport, Education), per §10 below.
4. **Default output location.** v2 artifacts (personas, transactions, monthly summaries,
   synthesis report) are written to `synth_v2/` by default — never `synth/` (v1's
   default) — so the two pipelines' outputs never collide.
5. See `training/docs/data-collection/synthetic-generation-v2.md` for the operator
   runbook (commands, output layout, validation, non-claims).

## Methodology Name

**FIES-anchored, HFCE-calibrated temporal disaggregation with proportional benchmarking**

In simple terms:

> **FIES tells us how much a household spends in a year. PSA HFCE tells us how Philippine household spending is distributed across the four quarters. We use the HFCE quarterly pattern to split each household’s annual FIES expenditure into quarterly and then monthly synthetic values, while ensuring that the generated values still sum exactly to the original FIES annual amount.**

The methodology follows established temporal-disaggregation and benchmarking principles used in official statistics. Eurostat describes temporal disaggregation as the derivation of higher-frequency observations, such as quarterly or monthly estimates, from lower-frequency data while respecting temporal and accounting constraints (Eurostat, 2018).

---

## 1. Primary Household Source: 2023 FIES

The **2023 Family Income and Expenditure Survey (FIES)** is used as the household-level source of annual expenditure. The 2023 round is the latest FIES public-use file released by the PSA at the time of this study; the 2024 and 2025 FIES microdata remain locked and are not available for public research use. Multi-year synthetic calendars therefore replay the 2023 within-year HFCE pattern rather than claiming unpublished FIES vintages. The 36-month horizon used for the v2 forecaster training corpus is a downstream modelling choice: Seasonal ARIMA with period `s=12` requires at least 24 monthly observations to identify the seasonal term. The extra year beyond that floor is calendar labels wrapping 2023 HFCE, not additional FIES vintages.

For household \(h\) and expenditure category \(c\):

\[
\boxed{
A_{h,c}
=
\text{annual FIES expenditure of household }h
\text{ for category }c
}
\]

The expenditure categories used in the synthetic generator are:

\[
c\in
\{
\text{Food},
\text{Housing},
\text{Health},
\text{Transport},
\text{Education},
\text{Other}
\}
\]

The publicly released FIES household summary provides annual household-level expenditure values. Although the 2023 FIES was collected through two six-month visits, the public-use files used in this study do not expose separate household-level first-half and second-half expenditure histories. Therefore, the annual household-category expenditure is treated as the household-level benchmark.

The Philippine Statistics Authority defines family expenditure as household expenditure for personal consumption during the calendar year and documents the two-visit 2023 FIES design (Philippine Statistics Authority [PSA], 2026a).

---

## 2. External Temporal Source: PSA Quarterly HFCE

To obtain a Philippine within-year spending pattern, the generator uses **2023 Household Final Consumption Expenditure (HFCE) by purpose at constant 2018 prices** from the PSA National Accounts.

Define:

\[
H_{c,q}
=
\text{2023 HFCE for category }c
\text{ in quarter }q
\]

The 2023 quarterly HFCE values used are:

| Category | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Food and non-alcoholic beverages | 1,265,574 | 1,362,290 | 1,230,791 | 1,626,806 |
| Housing, water, electricity, gas and other fuels | 440,944 | 553,704 | 459,676 | 483,347 |
| Health | 163,223 | 138,034 | 202,630 | 182,361 |
| Transport | 357,466 | 308,874 | 372,911 | 333,613 |
| Education | 209,215 | 186,084 | 213,148 | 230,920 |
| Miscellaneous goods and services | 517,071 | 472,035 | 551,053 | 705,407 |

Constant-price HFCE is preferred because the goal is to capture the **shape of real consumption across quarters** while reducing the direct influence of price inflation.

HFCE is used only as a **population-level temporal calibration source**. It does not imply that every individual household follows exactly the same quarterly pattern.

---

## 3. Average Quarterly HFCE

For each category, calculate the average quarterly HFCE:

\[
\boxed{
\bar H_c
=
\frac{
H_{c,Q1}+H_{c,Q2}+H_{c,Q3}+H_{c,Q4}
}{4}
}
\]

Equivalent notation:

\[
\boxed{
\bar H_c
=
\frac14\sum_{q=1}^{4}H_{c,q}
}
\]

### Example: Food

\[
\bar H_{\text{food}}
=
\frac{
1{,}265{,}574+
1{,}362{,}290+
1{,}230{,}791+
1{,}626{,}806
}{4}
\]

\[
=
1{,}371{,}365.25
\]

---

## 4. Quarterly Seasonal Factor

The quarterly seasonal factor measures how high or low a specific quarter is relative to the average quarter for the same category.

\[
\boxed{
S_{c,q}
=
\frac{H_{c,q}}{\bar H_c}
}
\]

Expanded form:

\[
\boxed{
S_{c,q}
=
\frac{
H_{c,q}
}{
\frac14\sum_{j=1}^{4}H_{c,j}
}
}
\]

Interpretation:

- \(S=1.00\): average quarter
- \(S>1.00\): above-average quarter
- \(S<1.00\): below-average quarter

### Example: Food Q4

\[
S_{\text{food,Q4}}
=
\frac{
1{,}626{,}806
}{
1{,}371{,}365.25
}
\]

\[
=
1.186267
\]

Therefore:

\[
(1.186267-1)\times100
=
18.6267\%
\]

So Q4 food expenditure in 2023 was approximately **18.63% above the average quarter**.

### Seasonal Factors

| Category | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Food | 0.922857 | 0.993382 | 0.897493 | 1.186267 |
| Housing | 0.910256 | 1.143030 | 0.948925 | 0.997790 |
| Health | 0.951394 | 0.804572 | 1.181089 | 1.062945 |
| Transport | 1.041519 | 0.899941 | 1.086520 | 0.972021 |
| Education | 0.997013 | 0.886783 | 1.015756 | 1.100448 |
| Miscellaneous | 0.921052 | 0.840830 | 0.981584 | 1.256533 |

---

## 5. Quarterly Allocation Weight

For synthetic generation, the more useful quantity is the **quarterly allocation weight**.

\[
\boxed{
W_{c,q}
=
\frac{
H_{c,q}
}{
\sum_{j=1}^{4}H_{c,j}
}
}
\]

This measures the share of the category's annual HFCE that belongs to a specific quarter.

Because:

\[
\sum_{q=1}^{4}W_{c,q}=1
\]

the four quarterly weights always distribute 100% of the annual amount.

### Example: Food Q4

\[
W_{\text{food,Q4}}
=
\frac{
1{,}626{,}806
}{
1{,}265{,}574+
1{,}362{,}290+
1{,}230{,}791+
1{,}626{,}806
}
\]

\[
=
0.296567
\]

or:

\[
29.6567\%
\]

This means **29.66% of annual food spending is allocated to Q4**.

This is different from the seasonal factor interpretation:

- **18.63%** = Q4 is 18.63% above the average quarter
- **29.66%** = Q4 receives 29.66% of the annual food expenditure

### Quarterly Allocation Weights

| Category | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Food | 23.0714% | 24.8346% | 22.4373% | 29.6567% |
| Housing | 22.7564% | 28.5757% | 23.7231% | 24.9447% |
| Health | 23.7848% | 20.1143% | 29.5272% | 26.5736% |
| Transport | 26.0380% | 22.4985% | 27.1630% | 24.3005% |
| Education | 24.9253% | 22.1696% | 25.3939% | 27.5112% |
| Miscellaneous | 23.0263% | 21.0208% | 24.5396% | 31.4133% |

---

## 6. Generate Synthetic Quarterly Household Expenditure

For household \(h\), category \(c\), and quarter \(q\):

\[
\boxed{
Q_{h,c,q}
=
A_{h,c}W_{c,q}
}
\]

where:

- \(A_{h,c}\) = observed annual FIES expenditure
- \(W_{c,q}\) = PSA-derived quarterly allocation weight
- \(Q_{h,c,q}\) = synthetic quarterly expenditure

### Example

Assume annual food expenditure is:

\[
A_{h,\text{food}}
=
₱120{,}000
\]

and:

\[
W_{\text{food,Q4}}
=
0.296567
\]

then:

\[
Q_{h,\text{food,Q4}}
=
120{,}000(0.296567)
\]

\[
=
₱35{,}588.04
\]

Because the quarterly weights sum to 1:

\[
\sum_{q=1}^{4}Q_{h,c,q}
=
A_{h,c}
\]

Thus, no annual expenditure is created or removed.

---

## 7. Generate Synthetic Monthly Expenditure

The available FIES public-use data and quarterly HFCE do not provide household-level monthly observations.

Therefore, the baseline method distributes each quarterly amount equally among its three constituent months:

\[
\boxed{
E_{h,c,m}
=
\frac{
Q_{h,c,q(m)}
}{3}
}
\]

where \(q(m)\) identifies the quarter containing month \(m\).

Thus:

\[
Jan=Feb=Mar=\frac{Q1}{3}
\]

\[
Apr=May=Jun=\frac{Q2}{3}
\]

\[
Jul=Aug=Sep=\frac{Q3}{3}
\]

\[
Oct=Nov=Dec=\frac{Q4}{3}
\]

### Example

For:

\[
Q4=₱35{,}588.04
\]

then:

\[
Oct=Nov=Dec
=
\frac{35{,}588.04}{3}
\]

\[
=
₱11{,}862.68
\]

The equal split within a quarter is an **explicit modeling assumption**, not an observed household-level monthly pattern.

---

## 8. Annual Reconciliation

In theory, the quarterly-weight approach already guarantees:

\[
\sum_{m=1}^{12}E_{h,c,m}
=
A_{h,c}
\]

because the quarterly weights sum to 1.

However, an explicit reconciliation step can be retained for numerical and rounding safety.

Let the provisional monthly values be:

\[
E^{(0)}_{h,c,m}
\]

Calculate:

\[
\boxed{
K_{h,c}
=
\frac{
A_{h,c}
}{
\sum_{m=1}^{12}E^{(0)}_{h,c,m}
}
}
\]

Then:

\[
\boxed{
E_{h,c,m}
=
K_{h,c}E^{(0)}_{h,c,m}
}
\]

Equivalent form:

\[
\boxed{
E_{h,c,m}
=
A_{h,c}
\frac{
E^{(0)}_{h,c,m}
}{
\sum_{j=1}^{12}E^{(0)}_{h,c,j}
}
}
\]

This guarantees:

\[
\boxed{
\sum_{m=1}^{12}E_{h,c,m}
=
A_{h,c}
}
\]

The method follows the general principle of **temporal benchmarking**, where higher-frequency estimates remain consistent with authoritative lower-frequency totals (Eurostat, 2018).

The Australian Bureau of Statistics applies the same general principle when benchmarking higher-frequency household-spending indicators against HFCE totals (Australian Bureau of Statistics, 2025; 2026).

---

## 9. Random Monthly Variation

The baseline generator does not introduce arbitrary random monthly noise.

\[
\boxed{
\sigma_{\text{random}}=0
}
\]

This does **not** mean that real households have zero month-to-month variation.

It means that no unsupported household-specific random variation is added because no Philippine household-level monthly dataset has been identified from which a defensible monthly variance parameter can be estimated.

The previous assumptions:

\[
X\sim N(1,0.15)
\]

and:

\[
0.50\leq X\leq2.00
\]

are removed from the baseline model.

The baseline therefore uses only:

\[
\boxed{
\text{observed FIES annual magnitude}
+
\text{observed PSA quarterly pattern}
}
\]

---

## 10. Treatment of the "Other" Category

In this generator, expenditure categories are split by **essential vs non-essential** purpose:

\[
C_{\text{essential}}
=
\{
\text{Food},
\text{Housing},
\text{Health},
\text{Transport},
\text{Education}
\}
\]

\[
\text{Other}
=
\text{non-essential expenditure outside } C_{\text{essential}}
\]

The five essential categories are the empirically backed household needs used for obligation / PFP modeling. `Other` is the residual lifestyle / discretionary bucket, not a sixth essential class.

### HFCE mapping for `Other`

`Other` is **not** mapped to PSA Miscellaneous alone. It is the quarterly residual of Total HFCE after removing the five essentials:

\[
\boxed{
H_{\text{Other},q}
=
H_{\text{Total},q}
-
\sum_{c\in C_{\text{essential}}}H_{c,q}
}
\]

Equivalently (up to published rounding), \(H_{\text{Other},q}\) aggregates the non-essential HFCE-by-purpose series:

- Alcoholic beverages and tobacco
- Clothing and footwear
- Furnishings, household equipment and routine household maintenance
- Communication
- Recreation and culture
- Restaurants and hotels
- Miscellaneous goods and services

The quarterly weight is then:

\[
\boxed{
W_{\text{Other},q}
=
\frac{
H_{\text{Other},q}
}{
\sum_{j=1}^{4}H_{\text{Other},j}
}
}
\]

### 2023 residual levels (constant 2018 prices, million PHP)

| | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| \(H_{\text{Other},q}\) | 1,255,736 | 1,032,737 | 1,204,677 | 1,562,440 |
| \(W_{\text{Other},q}\) | 24.84% | 20.43% | 23.83% | **30.91%** |

This keeps `Other` seasonality aligned with the generator’s semantic definition (non-essentials), rather than with the narrower Miscellaneous series alone.

---

## 11. Complete Synthetic Generation Pipeline

The complete household-level generation flow is:

\[
\boxed{
A_{h,c}
\rightarrow
H_{c,q}
\rightarrow
W_{c,q}
\rightarrow
Q_{h,c,q}
\rightarrow
E_{h,c,m}
}
\]

### Step 1 — FIES Annual Benchmark

\[
A_{h,c}
=
\text{observed annual household-category expenditure}
\]

### Step 2 — PSA Quarterly HFCE

\[
H_{c,q}
=
\text{observed national quarterly consumption}
\]

### Step 3 — Quarterly Allocation Weight

\[
W_{c,q}
=
\frac{H_{c,q}}
{\sum_jH_{c,j}}
\]

### Step 4 — Synthetic Quarterly Household Expenditure

\[
Q_{h,c,q}
=
A_{h,c}W_{c,q}
\]

### Step 5 — Synthetic Monthly Expenditure

\[
E_{h,c,m}
=
\frac{Q_{h,c,q(m)}}{3}
\]

### Step 6 — Final Validation

\[
\sum_{m=1}^{12}E_{h,c,m}
=
A_{h,c}
\]

---

## 12. Transaction-Level Synthetic Generation

If the application requires individual transaction records, monthly expenditure totals should be treated as hard constraints.

For household \(h\), category \(c\), and month \(m\):

\[
E_{h,c,m}
=
\text{monthly category expenditure}
\]

Let:

\[
N_{h,c,m}
\]

be the number of generated transactions.

The transaction values are:

\[
T_{h,c,m,1},
T_{h,c,m,2},
\dots,
T_{h,c,m,N}
\]

The generator must enforce:

\[
\boxed{
\sum_{i=1}^{N_{h,c,m}}
T_{h,c,m,i}
=
E_{h,c,m}
}
\]

Therefore:

\[
\sum_m\sum_iT_{h,c,m,i}
=
A_{h,c}
\]

The accounting hierarchy is:

\[
\boxed{
\text{Transactions}
\rightarrow
\text{Monthly totals}
\rightarrow
\text{Quarterly totals}
\rightarrow
\text{Annual FIES totals}
}
\]

Transaction-count and transaction-size distributions should not be assigned arbitrary parameters unless a defensible empirical source is identified.

---

## 13. Validation

### 13.1 Annual Consistency

For every household and category:

\[
\boxed{
Error^{annual}_{h,c}
=
\left|
A_{h,c}
-
\sum_{m=1}^{12}E_{h,c,m}
\right|
}
\]

Target:

\[
Error^{annual}_{h,c}\approx0
\]

---

### 13.2 Quarterly Calibration

For the generated population:

\[
\hat Q_{c,q}
=
\sum_hQ_{h,c,q}
\]

Calculate:

\[
\boxed{
\hat W_{c,q}
=
\frac{
\hat Q_{c,q}
}{
\sum_{r=1}^{4}\hat Q_{c,r}
}
}
\]

Compare:

\[
\hat W_{c,q}
\approx
W^{PSA}_{c,q}
\]

Without additional random perturbation:

\[
\hat W_{c,q}
=
W^{PSA}_{c,q}
\]

apart from rounding.

---

### 13.3 Annual FIES Distribution Preservation

Because the annual values are retained:

\[
A^{synthetic}_{h,c}
=
A^{FIES}_{h,c}
\]

the generated data should preserve annual:

\[
Mean(A_c)
\]

\[
Median(A_c)
\]

\[
SD(A_c)
\]

\[
P_{25}(A_c)
\]

\[
P_{75}(A_c)
\]

for households that are directly anchored to FIES observations.

---

## 14. Empirical vs. Synthetic Components

| Component | Classification |
|---|---|
| Annual household category expenditure | **Observed FIES** |
| Total household expenditure | **Observed FIES** |
| Quarterly national HFCE | **Observed PSA** |
| Seasonal factor \(S_{c,q}\) | **Calculated from PSA HFCE** |
| Quarterly allocation weight \(W_{c,q}\) | **Calculated from PSA HFCE** |
| Household quarterly expenditure | **Synthetic / calibrated** |
| Household monthly expenditure | **Synthetic** |
| Equal split within quarter | **Explicit modeling assumption** |
| Random Gaussian noise | **Not used** |
| 15% coefficient of variation | **Removed** |
| 0.5×–2.0× bounds | **Removed** |
| Annual reconciliation | **Mathematical constraint** |
| Generated transaction dates | **Synthetic** |
| Generated transaction values | **Synthetic and constrained** |

---

## 15. Methodological Justification

The methodology is supported by three main foundations.

First, FIES provides the household-level annual expenditure benchmark. The PSA documents FIES as the primary source of Philippine family income and expenditure statistics and defines family expenditure over the calendar year (PSA, 2026a).

Second, quarterly PSA HFCE provides an observed Philippine within-year consumption pattern. The quarterly HFCE by purpose series allows category-specific national consumption patterns to be estimated instead of inventing arbitrary seasonal multipliers (PSA, 2026b).

Third, temporal disaggregation and benchmarking are established practices in official statistics. Eurostat provides formal guidelines for constructing higher-frequency statistics from lower-frequency benchmarks while maintaining temporal and accounting consistency (Eurostat, 2018). The Australian Bureau of Statistics similarly benchmarks higher-frequency household-spending indicators to HFCE totals in its Monthly Household Spending Indicator methodology (Australian Bureau of Statistics, 2025; 2026).

The generated monthly values must therefore be interpreted as **synthetic temporal allocations calibrated to observed annual household expenditure and observed national quarterly consumption patterns**, not as reconstructed historical monthly transactions of the original FIES households.

---

## 16. Thesis-Ready Methodology Summary

> The synthetic expenditure generator applies a FIES-anchored, HFCE-calibrated temporal disaggregation methodology. Annual household-category expenditures from the 2023 Family Income and Expenditure Survey are retained as authoritative household-level benchmarks. Because the publicly available FIES microdata do not provide twelve household-level monthly observations, quarterly Household Final Consumption Expenditure by purpose at constant 2018 prices from the Philippine Statistics Authority is used as a population-level temporal calibration source. For each expenditure category, the share of annual HFCE occurring in each quarter is calculated as \(W_{c,q}=H_{c,q}/\sum_{j=1}^{4}H_{c,j}\). Each household's annual FIES expenditure is then multiplied by the corresponding quarterly weight to obtain synthetic quarterly expenditures. In the absence of an appropriate household-level monthly Philippine expenditure indicator, each quarterly amount is distributed equally among its three constituent months. A proportional reconciliation check ensures that the twelve synthetic monthly observations sum exactly to the original annual FIES expenditure. This approach follows established temporal-disaggregation and benchmarking principles in official statistics while clearly distinguishing observed FIES and HFCE values from synthetically generated household-level monthly observations.

---

## References

Australian Bureau of Statistics. (2022). *Development of the new experimental monthly household spending indicator*. Australian Bureau of Statistics. https://www.abs.gov.au/articles/development-new-experimental-monthly-household-spending-indicator

Australian Bureau of Statistics. (2025). *Monthly Household Spending Indicator methodology*. Australian Bureau of Statistics. https://www.abs.gov.au/methodologies/monthly-household-spending-indicator-methodology

Australian Bureau of Statistics. (2026). *Monthly Household Spending Indicator methodology*. Australian Bureau of Statistics. https://www.abs.gov.au/methodologies/monthly-household-spending-indicator-methodology

Eurostat. (2018). *European Statistical System guidelines on temporal disaggregation, benchmarking and reconciliation: 2018 edition*. Publications Office of the European Union. https://doi.org/10.2785/743998

Philippine Statistics Authority. (2026a). *2023 Family Income and Expenditure Survey: Technical notes*. Philippine Statistics Authority. https://psa.gov.ph/statistics/income-expenditure/fies/technical-notes

Philippine Statistics Authority. (2026b). *Household Final Consumption Expenditure by purpose*. PSA OpenSTAT / National Accounts of the Philippines. https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2B__NA__QT__2HFCE/

---

## Final Methodology Label

\[
\boxed{
\textbf{FIES-anchored, HFCE-calibrated temporal disaggregation with proportional benchmarking}
}
\]

The critical distinction is:

\[
\boxed{
\text{FIES determines the household's annual expenditure;}
\quad
\text{HFCE determines the quarterly population-level allocation;}
\quad
\text{the monthly observations are synthetic.}
}
\]
