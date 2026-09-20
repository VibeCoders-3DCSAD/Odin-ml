# Year-Specific Current-Price HFCE Disaggregation Plan

## Goal

Replace Synthetic Generation v2's repeated 2023 constant-2018-price HFCE calendar with year-specific PSA HFCE-by-purpose values at current prices: all quarters for 2023, 2024, and 2025, then Q1-Q2 only for 2026. Generate the minimum forecasting history from `2023-01` through `2026-06` (42 months) without creating unobserved 2026 Q3-Q4 inputs.

## Source Of Truth

- User-supplied PSA quarterly HFCE-by-purpose values, at current prices, for 2023-Q1 through 2026-Q2.
- `training/config/hfce_quarterly_indices.json`
- `training/scripts/temporal_disaggregation.py`
- `training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md`

## Non-Goals

- Do not modify the v1 generator, preprocessor, or its outputs.
- Do not create PSA values for 2026-Q3 or 2026-Q4, or extend the dataset past `2026-06`.
- Do not claim the 2023 FIES household annual anchor has been replaced by newer FIES microdata.
- Do not retrain or replace the committed forecaster artifact in this change.

## Execution Order

## PR Stacking Strategy

```text
main
  -> data/year-specific-current-price-hfce-config
  -> training/calendar-year-disaggregation
  -> docs/current-price-hfce-runbook
```

Merge in the listed order. The first PR changes only the audited data/config and its validation; the second consumes that contract; the final PR updates methodology and operator commands after the implementation is verified.

## Linear Sub-Issue Tracking

Create sub-issues from this plan when ready.

### 1. Add the Audited Year-Specific HFCE Source Config

- Touch `training/config/hfce_quarterly_indices.json` and `tests/test_temporal_disaggregation.py`.
- Replace the single 2023 constant-price structure with a current-price, year-keyed schema containing all supplied PSA series for 2023-2025 and Q1-Q2 for 2026; retain the six generator categories and derive `other` as total less the five essentials for each available quarter.
- Record the source, unit, price basis, covered periods, and explicitly available quarters per year. Tests must assert representative supplied levels, including 2023 food Q1 (`1,536,533`), 2025 food Q4 (`2,257,594`), and 2026 food Q2 (`2,058,217`), plus correct residual construction.

### 2. Make Disaggregation Calendar-Year Aware

- Touch `training/scripts/temporal_disaggregation.py`, `training/scripts/generate_transactions_v2.py`, `training/scripts/synthesizer_v2.py`, `training/scripts/preprocessor_v2.py`, `tests/test_temporal_disaggregation.py`, `tests/test_synth_v2_transactions.py`, and `tests/test_preprocessor_v2.py`.
- Change HFCE loading and schedule construction to require a calendar year and return only that year's available quarters; make the transaction generator select the schedule using each generated `year`, rather than wrapping a single January-December schedule.
- For complete years, preserve the existing accounting contract: allocate the twelve-month persona benchmark across Q1-Q4 and reconcile to that annual benchmark. For 2026, permit only months 1-6, normalize Q1-Q2 over their available levels, and reconcile to `persona_monthly_category_expense * 6`; reject generation requests that cross into a year or month without configured HFCE data.
- Set the v2 training/default invocation to 42 months beginning in January 2023 so outputs end at `2026-06`; retain CLI overrides only when their requested date range is fully present in the config.
- Add regression coverage proving 42 distinct ISO `year_month` values from `2023-01` to `2026-06`, different year-specific quarterly shares for the same category, no 2026-Q3/Q4 summaries, complete-year annual reconciliation, partial-2026 six-month reconciliation, deterministic results, and untouched v1 scripts.

### 3. Document the Revised Data Contract and Regenerate Training Inputs

- Touch `training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md`, `training/docs/data-collection/synthetic-generation-v2.md`, `docs/models/forecaster-v2.md`, and `INDEX.md`.
- Replace claims that multi-year runs replay the 2023 constant-price pattern with the year-specific current-price contract, clearly distinguishing observed national quarterly values from the synthetic household allocation and documenting the 2026 Q1-Q2 cutoff.
- Update run commands to use `--months 42`, state that the training corpus spans `2023-01` to `2026-06`, and regenerate ignored `synth_v2/`, processed v2, and forecaster feature datasets only after the unit and integration tests pass. Record any subsequent retraining/evaluation as a separate artifact-release task.

## Acceptance Criteria

- The HFCE config uses current prices and covers 2023-Q1 through 2026-Q2 exactly as supplied.
- 2023, 2024, and 2025 use their own four-quarter profiles; no year reuses the former 2023 constant-price profile.
- The generated minimum dataset contains every month from `2023-01` through `2026-06`, and no later month.
- 2026's six generated months use only Q1-Q2 and reconcile to six persona-months per category.
- Complete-year category totals still reconcile to each persona's twelve-month category benchmark before anomaly injection.
- `pytest tests/test_temporal_disaggregation.py tests/test_synth_v1_untouched.py tests/test_synth_v2_transactions.py tests/test_preprocessor_v2.py tests/test_multi_year_pipeline.py -v`, `ruff check training tests`, and `ruff format --check training tests` pass.
