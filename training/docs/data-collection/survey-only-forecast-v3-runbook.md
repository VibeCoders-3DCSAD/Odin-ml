# Survey-Only Forecast v3 Runbook

Run `python training/scripts/survey_pipeline_v3.py --input <2023-fies.csv>`. It writes only `survey_v3/` and `training/datasets/processed_v3/`.

Then run `python training/scripts/feature_engineering_forecaster_v3.py`. It writes RF-eligible rows only: target month `T` needs three strictly prior completed expense months. The output stores normalized lag and rolling features, the three-month `user_scale`, the normalized `target_ratio`, and the original PHP target for evaluation.

Run the final candidate and its same-row naive baseline:

```bash
python training/scripts/train_forecaster_v3_full.py --candidate naive
python training/scripts/train_forecaster_v3_full.py --candidate random_forest
python training/scripts/compare_forecaster_v3_full.py
```

The RF predicts `target_ratio` and the evaluator multiplies each prediction by that row's strictly-prior `user_scale` before calculating PHP MAE. This keeps the global learned behavior separate from each household's actual spending level. The resulting metrics are internal synthetic-target evidence only. Do not copy an artifact into the serving forecaster directory without the required external-validation decision.
