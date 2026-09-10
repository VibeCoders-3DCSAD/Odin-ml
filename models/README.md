# models/

Canonical home for **new-scope final model artifacts** for the revised BUDI ML service.

## Layout

```text
models/
  README.md            # This file
  pfp/                 # PFP classifier — winner .joblib + evaluation.json + metadata.json
  forecaster/          # Spending forecaster — winner artifact + evaluation.json + metadata.json
  anomaly/             # Anomaly detector — winner artifact + evaluation.json + metadata.json
  budget/              # Budget optimizer (deterministic LP) — budget_config.json + evaluation.json + metadata.json
```

Every family directory is self-contained and uniform:

- `evaluation.json` — raw evaluation results (fold metrics, winner). **Source of truth** for
  serving resolution.
- `evaluation_report.md` — human-readable report, regenerated from `evaluation.json`.
- `metadata.json` — reproducible provenance + the **winner contract**, re-emitted from
  `evaluation.json` (adds `model_id`, `created_at`, `training_commit`, `training_data_hash`).
- `metadata.example.json` — schema template for contributors.

Reports and metadata can be re-emitted from an existing `evaluation.json` without retraining:

```bash
python training/scripts/regenerate_artifacts.py
```

## Model artifact standards

Every final model committed to `models/` **must** carry `metadata.json` per this schema:

```json
{
  "model_id": "pfp-tier3_svm",
  "family": "pfp",
  "created_at": "2026-09-08T03:44:35Z",
  "training_commit": "<git sha of training code>",
  "training_data_hash": "<sha256 of source parquet or feature matrix>",
  "framework": "scikit-learn",
  "framework_version": "1.9.0",
  "python_version": "3.14.4",
  "artifacts": ["tier3_svm.joblib"],
  "feature_columns": ["income_stability_cv", "obligation_ratio", "..."],
  "metrics": {
    "primary": {"name": "macro_f1", "value": 0.675, "folds": 5},
    "secondary": {"accuracy": 0.678}
  },
  "decision_rule": "winner must beat tier1_rule_based by 0.02 macro-F1",
  "winner": "tier3_svm",
  "winner_artifact": "tier3_svm.joblib",
  "winner_reason": "...",
  "winner_params": {"C": 1.0, "kernel": "rbf"},
  "threshold": null,
  "fitted": true,
  "serving_note": "loaded by app/models/loader.py"
}
```

### Winner contract (uniform across families)

`winner`, `winner_artifact`, `winner_reason`, `winner_params`, `threshold` form the **winner
contract**. **`evaluation.json` is the runtime source of truth**: the serving layer
(`app/models/registry.py`) resolves a family's artifact from it, and
`training/scripts/regenerate_artifacts.py` re-emits `metadata.json` from it (`model_id` =
`<family>-<winner>`). Fresh training runs write both; they must never diverge. At request time
the API reports the resolved winner (e.g. `model_version`: `forecaster-tier3_sarima`,
`tier_used`/`model_name` for pfp) rather than hardcoded strings.

- `winner_artifact` names the exact artifact file to load (fallback: `winner` + extension).
- `winner_params` records how the winner was produced.
- `threshold` is the operational threshold where the family has one (e.g. anomaly detector;
  `null` otherwise).
- Budget's "winner" is the deterministic formulation `scipy_linprog`, so
  `fitted: false` and the artifact is `budget_config.json`.

Because resolution is winner-contract-driven, swapping, adding, or removing a winner is a
**metadata change, not a code change**.

### Minimum required keys

`model_id`, `family`, `created_at`, `training_commit`, `training_data_hash`, `framework`,
`framework_version`, `python_version`, `artifacts`, `feature_columns`, `metrics`,
`decision_rule`, `winner`, `winner_artifact`, `winner_reason`, `winner_params`, `threshold`,
`fitted`, `serving_note`.

## Gitignoring

Large checkpoints and intermediate artifacts belong in `training/models/` (gitignored).
Only the **final selected model** for each family plus `evaluation.json`,
`evaluation_report.md`, and `metadata.json` is committed here.
Keep total committed artifact size small (< ~50 MB per model) — else prefer a release
binary store and reference it from `metadata.json`.