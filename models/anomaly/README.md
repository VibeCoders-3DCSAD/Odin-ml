# models/anomaly/

Canonical home for the **new-scope** anomaly detector artifact (`metadata.json` + final
`.joblib`). Current winner: **`tier1_iqr`** (`anomaly_detector.joblib`) — IQR baseline kept
after Option A rule validation on the held-out test split; the fold-nominated hybrid
(generalization gap) is documented in `evaluation.json` → `test_validation`
(see `evaluation.json` / `evaluation_report.md` / `docs/thesis/anomaly-decision-rule-rationale.md`).

## Decision rule (pre-registered, Option A — revised 2026.09.10)

Winner must reach **PR-AUC ≥ 1.5× the IQR baseline** **and** **PR-AUC ≥ 0.15** on the
held-out test split; otherwise fall back to IQR.

## Primary metrics

- PR-AUC (primary, imbalance-safe); Accuracy, Precision, Recall, F1 at the working point
  (+ ROC-AUC). Operating point: val threshold maximizing F2 subject to precision ≥ 0.30.

## Target artifact

- `anomaly_detector.joblib` — committed here (winner resolved from `metadata.json` at serve
  time).
- `app/models/registry.py` (`ANOMALY_MODULE = "anomaly"`) loads it at serve time via
  `_resolve_anomaly`/metadata-based resolution.

## Committed state

- `metadata.json` — winner contract (`winner`, `winner_artifact`, `winner_params`, `threshold`).
- `evaluation.json` — raw fold metrics (source of truth for the report).
- `evaluation_report.md` — regenerated from `evaluation.json`.

Regenerate reports/metadata from existing `evaluation.json` without retraining:

```bash
python training/scripts/regenerate_artifacts.py
```
