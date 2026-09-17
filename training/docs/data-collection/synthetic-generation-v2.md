# Synthetic Generation v2 — Operator Runbook

```json
{
  "document-type": "runbook",
  "version": "1.0.0",
  "date": "2026.09.17",
  "authors": ["Group 4, III-DCSAD"]
}
```

**Methodology:** FIES-anchored, HFCE-calibrated temporal disaggregation with
proportional benchmarking. See
[`fies-hfce-synthetic-data-generation-methodology.md`](fies-hfce-synthetic-data-generation-methodology.md)
for the full derivation.

**`synth_version`:** `"2.0.0"`

---

## 1. Isolation Rule (v1 is untouched)

Synthetic Generation v2 is a **parallel** pipeline. It does **not** modify, deprecate, or
delete:

- `training/scripts/generate_personas.py`
- `training/scripts/generate_transactions.py`
- `training/scripts/synthesizer.py`
- `training/scripts/preprocessor.py`

v2 only *imports* read-only helpers from these modules (e.g. `generate_all_personas`,
`Transaction`, `MonthlySummary`, `inject_anomalies`, `generate_income_transactions`,
`split_personas`, `generate_temporal_folds`, `export_results`). v1's legacy
expense-noise path (`rng.normal(1.0, 0.15)` bounded to `0.5×–2.0×`) is guarded by
`tests/test_synth_v1_untouched.py` and remains in place for anyone who still wants the
v1 (non-HFCE) generator.

If you want the original v1 behavior, keep using `synthesizer.py` / `preprocessor.py`
exactly as before — nothing here changes their defaults or outputs.

---

## 2. New v2 Modules

| Module | Responsibility |
| :--- | :--- |
| `training/scripts/temporal_disaggregation.py` | Load HFCE config, compute quarterly weights, disaggregate annual → quarterly → monthly, reconcile to annual. |
| `training/scripts/generate_transactions_v2.py` | Generate v2 transactions/monthly summaries from HFCE-derived monthly schedules (no Gaussian expense noise). |
| `training/scripts/synthesizer_v2.py` | Orchestrate personas (imported from v1, unchanged) + v2 transactions → `synth_v2/`. |
| `training/scripts/preprocessor_v2.py` | Run `synthesizer_v2.py`, then split personas + build temporal folds using v1's exact `split_personas`/`generate_temporal_folds`/`export_results` (imported, read-only) → `training/datasets/processed_v2/`. |

---

## 3. Commands

Run all commands from the repository root with the virtualenv activated:

```bash
source .venv/bin/activate
```

### Full pipeline (personas + v2 transactions)

```bash
PYTHONPATH=training/scripts python training/scripts/synthesizer_v2.py \
  --input training/datasets/unprocessed/puf.parquet \
  --output synth_v2/ \
  --hfce training/config/hfce_quarterly_indices.json \
  --personas-per-archetype 1000 \
  --months 12 \
  --seed 42
```

### Full pipeline + train/val/test split + temporal folds (recommended for training)

`preprocessor_v2.py` runs the above generation step and then splits personas
(stratified by `pfp_label`) and builds walk-forward temporal folds, using the
exact same `split_personas()` / `generate_temporal_folds()` / `export_results()`
functions v1's `preprocessor.py` uses (imported, read-only — see §1):

```bash
python training/scripts/preprocessor_v2.py \
  --input training/datasets/unprocessed/puf.parquet \
  --output training/datasets/processed_v2/ \
  --synth-output synth_v2/ \
  --hfce training/config/hfce_quarterly_indices.json \
  --personas-per-archetype 1000 \
  --months 36 \
  --seed 42
```

This writes `training/datasets/processed_v2/{train,val,test}.parquet`,
`split_metadata.json` (stamped with `synth_version = "2.0.0"`),
`temporal_folds.json`, `feature_columns.json`, and `pipeline_report.json` — the
same layout v1's `preprocessor.py` writes to `training/datasets/processed/`.

### Skip FIES loading (use default archetype statistics)

```bash
PYTHONPATH=training/scripts python training/scripts/synthesizer_v2.py \
  --skip-fies \
  --personas-per-archetype 50 \
  --output synth_v2/
```

### v2 transactions only (personas already generated)

```bash
PYTHONPATH=training/scripts python training/scripts/generate_transactions_v2.py \
  --input synth_v2/personas.json \
  --output synth_v2/ \
  --hfce training/config/hfce_quarterly_indices.json \
  --months 12 \
  --seed 42
```

### Disable anomaly injection

```bash
PYTHONPATH=training/scripts python training/scripts/generate_transactions_v2.py \
  --input synth_v2/personas.json \
  --output synth_v2/ \
  --no-anomalies
```

---

## 4. Output Layout

```text
synth_v2/
  personas.json
  personas.parquet
  transactions.parquet
  monthly_summaries.parquet
  synthesis_report.json        # stamps synth_version = "2.0.0"
  expense_ratios.json          # if FIES data was loaded (not --skip-fies)
  fies_stats.json              # if FIES data was loaded (not --skip-fies)
```

`synth_v2/` is gitignored, same as v1's `synth/` — regenerate locally with the commands
above.

---

## 5. Pointing Downstream Tools at `synth_v2/` / `processed_v2/`

`feature_engineering_forecaster.py`, `feature_engineering_anomaly.py`, and
`dimension_discovery.py` all default to reading from `synth/` and
`training/datasets/processed/` (v1). **v2 does not change those defaults.** These
scripts do not take a single `--input` flag — pass their actual file-level flags
explicitly, pointing at `preprocessor_v2.py`'s output, e.g. for the forecaster:

```bash
python training/scripts/feature_engineering_forecaster.py \
  --transactions synth_v2/transactions.parquet \
  --summaries synth_v2/monthly_summaries.parquet \
  --splits training/datasets/processed_v2/split_metadata.json \
  --output training/datasets/forecaster_v2/
```

And for training (e.g. `train_forecaster.py`'s `--folds`):

```bash
python training/scripts/train_forecaster.py \
  --input training/datasets/forecaster_v2/ \
  --output models/forecaster_v2/ \
  --folds training/datasets/processed_v2/temporal_folds.json
```

(Check each script's `--help` output for its exact flags — this repo's isolation rule
forbids changing their *defaults*, but every script accepts explicit paths.)

---

## 6. Validation Commands

```bash
source .venv/bin/activate
pytest tests/test_temporal_disaggregation.py tests/test_synth_v1_untouched.py \
  tests/test_synth_v2_transactions.py tests/test_preprocessor_v2.py -v
```

Checks covered (methodology §13, mirrored in `tests/test_synth_v2_transactions.py`):

| ID | Rule |
| :--- | :--- |
| V0 | v1's `generate_transactions.py` still contains the legacy `Normal(1, 0.15)` noise path (untouched guard). |
| V1 | Per persona × category: annual sum of generated monthly amounts ≈ \(A_{h,c}\). |
| V2 | Population quarterly shares reproduce the HFCE quarterly weights \(W_{c,q}\). |
| V3 | Monthly transaction sums equal the HFCE-derived monthly schedule amount. |
| V4 | Deterministic output given the same seed + HFCE config. |
| V5 | `synthesis_report.json` stamps `synth_version == "2.0.0"`. |

Checks covered for `preprocessor_v2.py` (`tests/test_preprocessor_v2.py`):

| ID | Rule |
| :--- | :--- |
| P0 | v1's `preprocessor.py` still defines `split_personas`, `generate_temporal_folds`, `export_results` (untouched guard). |
| P1 | Full pipeline smoke test produces the v1-shaped output layout (`train/val/test.parquet`, `split_metadata.json`, `temporal_folds.json`, `feature_columns.json`, `pipeline_report.json`) plus `synth_v2/` artifacts. |
| P2 | `split_metadata.json` stamps `synth_version == "2.0.0"`. |
| P3 | Persona split is deterministic given the same seed. |
| P4 | Split sizes respect the requested train/val/test ratios. |

---

## 7. Non-Claims

To avoid overstating what the synthetic data represents:

- **No Visit 1 / Visit 2 semester anchors.** The public 2023 FIES PUF does not expose
  separate first-half / second-half household expenditure. v2 does not claim or
  reconstruct semester-level household observations.
- **Income is not HFCE-calibrated.** HFCE is an expenditure series; v2 reuses v1's
  income-generation helpers unchanged. Only expense amounts are HFCE-disaggregated.
- **Equal-thirds within a quarter is a modeling assumption**, not an observed
  household-level monthly pattern (methodology §7).
- **Multi-year runs reuse the same 12-month HFCE pattern every year** (calendar month →
  HFCE month index via `(month - 1) % 12`). This is a documented limitation, not a
  claim of year-specific seasonality beyond 2023.
- **`Other` is a residual bucket** (Total HFCE − essentials), not the narrower PSA
  "Miscellaneous goods and services" series alone (methodology §10).
