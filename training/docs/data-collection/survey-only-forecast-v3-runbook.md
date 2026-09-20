# Survey-Only Forecast v3 Runbook

Run `python training/scripts/survey_pipeline_v3.py --input <2023-fies.csv>`. It writes only `survey_v3/` and `training/datasets/processed_v3/`.

Then run `python training/scripts/feature_engineering_forecaster_v3.py` and `train_forecaster_v3.py`. The v3 trainer compares naive, pooled ARIMA/SARIMA, Random Forest, and PyTorch LSTM candidates; use `--skip-rf` or `--skip-torch` on constrained machines. The resulting metrics are internal synthetic-target evidence only. Do not copy an artifact into the serving forecaster directory without the required external-validation decision.

For full-corpus candidate training, run `train_forecaster_v3_full.py` once per candidate and then use `compare_forecaster_v3_full.py`. This keeps only one learned candidate in memory at a time and records independent reports under `models/forecaster_v3/candidates/`.
