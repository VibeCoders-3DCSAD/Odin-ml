# Forecaster v3

```json
{
  "document-type": "model-evaluation",
  "version": "1.1.0",
  "date": "2026.09.21",
  "authors": ["Group 4, III-DCSAD"]
}
```

V3 is an unserved research pipeline using selected 2023 FIES households and PSA HFCE temporal allocation. Its monthly targets are synthetic allocations, so internal metrics do not establish real household forecast accuracy. The served forecaster remains v2 until external validation and an explicit artifact-release approval.

## Candidate Contract

The v3 Random Forest is a global model: it learns normalized next-month spending behavior across training households, rather than an absolute PHP forecast per household.

For target month `T`, only the three completed months before `T` are used:

```text
user_scale[T] = mean(expenses[T-3], expenses[T-2], expenses[T-1])
target_ratio[T] = expenses[T] / user_scale[T]
```

The input features are the three prior expense lags and their three-month standard deviation, each divided by `user_scale[T]`. The forest predicts `target_ratio`; serving or evaluation converts that result back to PHP:

```text
forecast_php[T] = predicted_ratio[T] * user_scale[T]
```

This requires three completed positive-expense months. Shorter histories use a non-model fallback and are excluded from RF MAE. The naive comparator uses `user_scale[T]` as its PHP prediction on the identical eligible rows.

V3 deliberately excludes calendar month and quarter features from this RF contract. The source allocates each PSA quarter equally over its three months, so calendar features could primarily memorize a synthetic allocation assumption rather than learn user behavior.

## Evaluation Limit

Evaluation is household-disjoint and reports MAE in PHP after rescaling. It tests whether the RF predicts the pipeline's synthetic monthly allocation for held-out households. It does not establish accuracy on observed household transaction history and must not authorize deployment by itself.
