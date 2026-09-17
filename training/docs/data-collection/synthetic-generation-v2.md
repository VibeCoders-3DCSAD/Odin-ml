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
`Transaction`, `MonthlySummary`, `inject_anomalies`, `generate_income_transactions`).
v1's legacy expense-noise path (`rng.normal(1.0, 0.15)` bounded to `0.5×–2.0×`) is
guarded by `tests/test_synth_v1_untouched.py` and remains in place for anyone who still
wants the v1 (non-HFCE) generator.

If you want the original v1 behavior, keep using `synthesizer.py` / `preprocessor.py`
exactly as before — nothing here changes their defaults or outputs.

---

## 2. New v2 Modules

| Module | Responsibility |
| :--- | :--- |
| `training/scripts/temporal_disaggregation.py` | Load HFCE config, compute quarterly weights, disaggregate annual → quarterly → monthly, reconcile to annual. |
| `training/scripts/generate_transactions_v2.py` | Generate v2 transactions/monthly summaries from HFCE-derived monthly schedules (no Gaussian expense noise). |
| `training/scripts/synthesizer_v2.py` | Orchestrate personas (imported from v1, unchanged) + v2 transactions → `synth_v2/`. |

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

## 5. Pointing Downstream Tools at `synth_v2/`

`feature_engineering_forecaster.py`, `feature_engineering_anomaly.py`, and
`dimension_discovery.py` all default to reading from `synth/` (v1). **v2 does not
change those defaults.** To evaluate v2 data with the existing feature-engineering
scripts, pass the input path explicitly, e.g.:

```bash
python training/scripts/feature_engineering_forecaster.py \
  --input synth_v2/ \
  --output training/datasets/forecaster_v2/
```

(Check each script's `--input`/`--output` flags — this repo's isolation rule forbids
changing their *defaults*, but every script accepts explicit paths.)

---

## 6. Validation Commands

```bash
source .venv/bin/activate
pytest tests/test_temporal_disaggregation.py tests/test_synth_v1_untouched.py tests/test_synth_v2_transactions.py -v
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
