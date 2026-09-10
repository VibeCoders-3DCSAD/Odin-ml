# Anomaly — Anomaly Detection Evaluation Report

**Generated:** 2026-09-08T03:44:32.752924Z
**Folds:** 5
**Decision rule:** winner must beat the IQR baseline by ≥ 50% F1 improvement and reach F1 ≥ 0.85; otherwise fall back to IQR

## Winner

- **Tier:** tier1_iqr
- **Artifact:** anomaly_detector.joblib
- **Reason:** Best learned tier (tier2_adaptive_threshold) improved F1 by 171.0% over IQR but reached only 0.3718 (target >= 0.85); pre-registered rule failed, retaining the interpretable IQR baseline.

## Approval Criteria Result

**Result:** FAIL — pre-registered fallback to IQR — F1 improvement over IQR 171.0% (target ≥ 50%), F1 target ≥ 0.85, passed: False

## Aggregate Results

| Model | F1 (mean ± std) | Accuracy (mean) | PR-AUC (mean ± std) |
|-------|-----------------|-----------------|----------------------|
| tier2_adaptive_threshold | 0.3718 ± 0.0106 | 0.9766 | 0.2734 ± 0.0080 |
| tier3_hybrid | 0.3718 ± 0.0106 | 0.9766 | 0.2753 ± 0.0082 |
| tier2_autoencoder | 0.2272 ± 0.1096 | 0.9175 | 0.1238 ± 0.0672 |
| tier2_ocsvm | 0.2271 ± 0.0292 | 0.9458 | 0.1195 ± 0.0075 |
| tier2_isolation_forest | 0.2107 ± 0.0181 | 0.9134 | 0.1061 ± 0.0070 |
| tier1_iqr | 0.1372 ± 0.0085 | 0.7710 | 0.0635 ± 0.0031 |
| baseline | 0.0000 ± 0.0000 | 0.9698 | N/A |

## Per-Fold Results

### Fold 1

| Model | F1 | Accuracy | PR-AUC |
|-------|----|----------|--------|
| baseline | 0.0000 | 0.9700 | 0.0300 |
| tier1_iqr | 0.0055 | 0.9696 | 0.0659 |
| tier2_adaptive_threshold | 0.0780 | 0.5251 | 0.2840 |
| tier2_autoencoder | 0.0000 | 0.9699 | 0.2388 |
| tier2_isolation_forest | 0.1898 | 0.9429 | 0.1168 |
| tier2_ocsvm | 0.1462 | 0.9596 | 0.1273 |
| tier3_hybrid | 0.0749 | 0.5287 | 0.2856 |

### Fold 2

| Model | F1 | Accuracy | PR-AUC |
|-------|----|----------|--------|
| baseline | 0.0000 | 0.9701 | 0.0299 |
| tier1_iqr | 0.0016 | 0.9698 | 0.0617 |
| tier2_adaptive_threshold | 0.0749 | 0.5211 | 0.2750 |
| tier2_autoencoder | 0.0000 | 0.9699 | 0.1602 |
| tier2_isolation_forest | 0.1592 | 0.9384 | 0.1003 |
| tier2_ocsvm | 0.1530 | 0.9549 | 0.1074 |
| tier3_hybrid | 0.0717 | 0.5237 | 0.2763 |

### Fold 3

| Model | F1 | Accuracy | PR-AUC |
|-------|----|----------|--------|
| baseline | 0.0000 | 0.9696 | 0.0304 |
| tier1_iqr | 0.0016 | 0.9693 | 0.0630 |
| tier2_adaptive_threshold | 0.0766 | 0.5234 | 0.2597 |
| tier2_autoencoder | 0.0008 | 0.9694 | 0.0629 |
| tier2_isolation_forest | 0.1451 | 0.9424 | 0.0984 |
| tier2_ocsvm | 0.0838 | 0.9639 | 0.1182 |
| tier3_hybrid | 0.0724 | 0.5251 | 0.2610 |

### Fold 4

| Model | F1 | Accuracy | PR-AUC |
|-------|----|----------|--------|
| baseline | 0.0000 | 0.9682 | 0.0318 |
| tier1_iqr | 0.0008 | 0.9682 | 0.0677 |
| tier2_adaptive_threshold | 0.1017 | 0.7216 | 0.2713 |
| tier2_autoencoder | 0.0000 | 0.9682 | 0.0897 |
| tier2_isolation_forest | 0.1397 | 0.9420 | 0.1033 |
| tier2_ocsvm | 0.1651 | 0.9547 | 0.1170 |
| tier3_hybrid | 0.0948 | 0.7234 | 0.2738 |

### Fold 5

| Model | F1 | Accuracy | PR-AUC |
|-------|----|----------|--------|
| baseline | 0.0000 | 0.9713 | 0.0287 |
| tier1_iqr | 0.0017 | 0.9713 | 0.0590 |
| tier2_adaptive_threshold | 0.0938 | 0.7217 | 0.2771 |
| tier2_autoencoder | 0.0008 | 0.9713 | 0.0674 |
| tier2_isolation_forest | 0.1466 | 0.9490 | 0.1118 |
| tier2_ocsvm | 0.1958 | 0.9609 | 0.1277 |
| tier3_hybrid | 0.0872 | 0.7237 | 0.2798 |


## Supplementary

### Final Test Metrics (threshold selected on held-out val)

- **Operating threshold:** 0.1250
- **Accuracy:** 0.6934
- **Precision:** 0.0635
- **Recall:** 0.6684
- **F1:** 0.1159
- **PR-AUC (supplementary):** 0.0550
- **ROC-AUC (supplementary):** 0.7084
- **TP/FP/FN/TN:** 4292/63323/2129/143712

### Key Findings

- **Class imbalance:** ~3.0% anomaly rate
- **Primary metrics are Accuracy/Precision/Recall/F1** (MDD v2.3); PR-AUC/ROC retained as supplementary
- **IQR provides interpretable statistical baseline** with per-feature thresholds
- **Isolation Forest handles unsupervised detection**; contamination set to the observed training anomaly rate
- **Operating threshold is selected on the held-out val split** to avoid test leakage

### Anomaly Types

Synthetic data injects 4 anomaly types (`anomaly_type` column):

1. **amount_spike** — unusually high transaction amount
2. **new_merchant** — first transaction with a new merchant
3. **frequency_change** — abnormal transaction frequency
4. **category_mismatch** — transaction category inconsistent with expectation

### Recommendations

1. Deploy the winning model for real-time scoring
2. Set anomaly threshold based on business tolerance (precision vs recall)
3. Monitor model performance on incoming data for drift
4. Consider ensemble approach for production robustness
