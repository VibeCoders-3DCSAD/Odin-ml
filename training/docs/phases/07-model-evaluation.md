# Phase 7 — Model Evaluation

**Status:** Complete (new-scope evaluation executed for all four families; artifacts live in
`models/<family>/`).

## Acceptance criteria (pre-registered decision rules)

| Family | Rule | Primary metric | Source of truth |
|---|---|---|---|
| PFP | Winner must beat Tier-1 rule-based floor by ≥ 0.02 Macro-F1 | Macro-F1 | `train_pfp.py` docstring + `models/pfp/evaluation.json` |
| Forecaster | Winner must beat naive baseline by ≥ 20% MAPE reduction | MAE, SMAPE, MDA, RMSE | `train_forecaster.py` docstring + `models/forecaster/evaluation.json` |
| Anomaly | Winner must reach PR-AUC ≥ 1.5× the IQR baseline and PR-AUC ≥ 0.15 on the held-out test split; else fall back to IQR. Operating point: F2 (β=2) maximized on the held-out val split subject to precision ≥ 0.30 | PR-AUC (primary, imbalance-safe); Accuracy/Precision/Recall/F1 at the operating point | `train_anomaly.py` docstring + `docs/thesis/anomaly-decision-rule-rationale.md` + `models/anomaly/evaluation.json` |
| Budget | Constraint Satisfaction Rate, Budget Utilization Rate, Deviation from User Preferences | (LP feasibility / utilization) | Budget Optimizer MDD v1.0 |

## Evaluation protocol (fixed for all new-scope runs)

1. **Split integrity:** 5-fold expanding window from `training/datasets/processed/temporal_folds.json`, keyed by canonical ISO `year_month` (`YYYY-MM`); embargo periods excluded from training labels (forecaster).
2. **No test leakage:** operating thresholds selected on the held-out val split only; test used exactly once at the end. For anomaly, the fold-selected candidate is additionally **validated on the held-out test split** before adoption — the served winner is whichever candidate passes the rule on unseen-user test evidence, and the fold-vs-test PR-AUC gap is recorded in `evaluation.json` (`test_validation`), not hidden.
3. **Write `evaluation.json` + `evaluation_report.md`** next to the artifact in `models/<family>/`.
4. **Record:** per-tier metrics, winner, winner reason (rule restated), feature columns, timestamp, training-data hash, git commit.

## New-scope deliverable

Re-run `train_*.py` after any feature/label change. Output lands in `models/<family>/` with
`evaluation.json`, `evaluation_report.md`, and its `metadata.json` (schema in
`models/README.md`). To re-emit reports/metadata from an existing `evaluation.json` without
retraining: `python training/scripts/regenerate_artifacts.py`.
