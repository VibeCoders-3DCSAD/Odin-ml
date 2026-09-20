# Survey-Only Forecast v3 Execution Plan

```json
{
  "document-type": "execution-plan",
  "version": "1.0.0",
  "date": "2026.09.20",
  "authors": ["Group 4, III-DCSAD"]
}
```

## Goal

Build a parallel v3 forecasting-data pipeline in which each training unit is an eligible 2023 FIES household, not a generated persona. V3 preserves observed annual household expenditure categories, allocates them to synthetic months using published PSA HFCE quarterly profiles, and produces survey-only forecasting inputs with complete provenance and accounting validation.

## Source Of Truth

- `training/datasets/raw/PHL-PSA-FIES-2023-V1-PUF/` for the 2023 FIES public-use microdata and data dictionary.
- `training/scripts/fies_columns.py` for the existing FIES variable-ID mapping.
- `training/config/hfce_quarterly_indices.json` for PSA current-price quarterly HFCE values from 2023-Q1 through 2026-Q2.
- `training/scripts/temporal_disaggregation.py` for the existing calendar-aware allocation and coverage contract.
- `training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md` for the established temporal-disaggregation methodology and its limitations.

## Non-Goals

- Do not modify, remove, regenerate, or overwrite any v1 or v2 source file, configuration, dataset, generated artifact, model artifact, evaluation report, metadata file, or documentation.
- Do not generate personas, archetypes, PFP labels, synthetic income, debt, savings, demographic values, or behavioral variables in v3.
- Do not expose `SEQ_NO` or any raw household identifier in v3 artifacts; use a deterministic pseudonymous household ID.
- Do not claim that the generated monthly series are observed household histories or that synthetic-target metrics demonstrate real household forecasting accuracy.
- Do not change the served forecaster until a separately approved v3 artifact-release decision.

## Hard Isolation Rule

- V3 is additive. It may read public FIES inputs and the existing PSA HFCE configuration, but it must write only v3-scoped files and directories.
- Create v3-specific training, feature-engineering, evaluation, and documentation modules; do not extend a v1 or v2 module for v3 behavior.
- V3 output paths are `survey_v3/`, `training/datasets/processed_v3/`, `training/datasets/forecaster_v3/`, and `models/forecaster_v3/` only.
- Tests must snapshot or checksum the v1/v2 tracked files before a v3 regeneration smoke test and fail if any change occurs.

## Execution Order

## PR Stacking Strategy

```text
main
  -> data/v3-fies-household-anchor
  -> training/v3-survey-monthly-generation
  -> training/v3-survey-preprocessing
  -> training/v3-forecaster-evaluation
  -> docs/v3-methodology-and-release-record
```

Merge in the listed order. Each branch has one data contract boundary: selected annual households, monthly allocations, leak-free feature data, model evaluation, then documentation and artifact provenance.

## Linear Sub-Issue Tracking

Create sub-issues from this plan when ready.

### 1. Lock the v3 Survey Contract

- Create `training/docs/data-collection/survey-only-forecast-v3.md` and `tests/test_fies_households_v3.py`.
- Record the approved geographic scope, inclusion and exclusion rules, duplicate-`SEQ_NO` policy, handling of missing and negative expenditures, outlier policy, and whether `RFACT` is used for aggregate diagnostics only or for weighted training; these decisions must be explicit before implementation because they define the training population.
- Define the six-category mapping: `FOOD`, `HOUSING_WATER`, `HEALTH`, `TRANSPORT`, and `EDUCATION` map directly; `other` is a documented residual derived from observed FIES annual expenditure, with a hard validation that the six categories reconcile to the chosen FIES total definition.

### 2. Build the FIES Household Cleaning and Selection Module

- Create `training/scripts/fies_households_v3.py` and extend `tests/test_fies_households_v3.py`.
- Implement a typed, deterministic loader and selection function that uses `FIESColumns`, validates required columns, coerces numeric fields, applies the locked selection rules, maps annual expenditures, and emits only approved observed covariates plus `household_id_v3` derived from `SEQ_NO` with a stable one-way hash.
- Write `survey_selection_report.json` beside the output with source file fingerprint, row counts at each exclusion step, column-resolution report, geography distribution, missingness, category distributions, and weighted diagnostics when `RFACT` is approved for that purpose.

### 3. Validate the Observed Annual Anchor

- Extend `training/scripts/fies_households_v3.py` and `tests/test_fies_households_v3.py`.
- Add hard validation for unique pseudonymous household IDs, non-negative category values, non-zero total annual expenditure, valid household size when retained, and exact or explicitly tolerated reconciliation between `food + housing + health + transport + education + other` and the selected observed expenditure total.
- Add distribution-preservation checks for count, mean, median, standard deviation, and P25/P75 by category before and after export; reject any cleaning rule that silently turns an observed value into a generated replacement.

### 4. Generate Household Monthly Expenditure Histories

- Create `training/scripts/generate_monthly_expenditures_v3.py`, `training/scripts/survey_pipeline_v3.py`, and `tests/test_survey_monthly_v3.py`.
- For every selected household and category, use the observed annual FIES amount as the benchmark, allocate it with the existing year-specific HFCE profile, split each quarter into equal monthly thirds, and produce only the supported timeline `2023-01` through `2026-06`.
- Retain annual accounting for every complete calendar year and six-month accounting for 2026; use a stable hash-derived seed only if optional transaction dates or splits are generated, never to vary category monthly totals.

### 5. Add Optional Constrained Transaction Rendering

- Create `training/scripts/generate_transactions_v3.py` and extend `tests/test_survey_monthly_v3.py` only if daily or transaction-level forecaster features remain necessary after the v3 feature-design review.
- Render transaction records from monthly category totals as a downstream synthetic representation, enforce transaction-to-month exactness, and keep generated dates, subcategories, and counts clearly marked as synthetic.
- Prefer a monthly forecaster input if the current daily feature set supplies no independently observed signal; do not add arbitrary transaction-count or transaction-size distributions merely to preserve the old interface.

### 6. Build Survey-Only Preprocessing and Split Contracts

- Create `training/scripts/preprocessor_v3.py`, `tests/test_preprocessor_v3.py`, and `training/datasets/processed_v3/` as the gitignored output directory.
- Split by `household_id_v3` before feature construction so no household appears in more than one train, validation, or test set; stratify only with retained observed FIES fields if the locked contract requires it, never with persona/PFP labels.
- Produce `train.parquet`, `val.parquet`, `test.parquet`, `temporal_folds.json`, `split_metadata.json`, and `pipeline_report.json`; persist source fingerprints, selection-contract version, HFCE configuration fingerprint, timeline, seed, and household IDs per split in the metadata.

### 7. Redesign Forecast Features for Survey-Only Inputs

- Create `training/scripts/feature_engineering_forecaster_v3.py` and `tests/test_forecaster_features_v3.py`.
- Remove persona-only features and audit each retained feature for availability at inference time; build lag and rolling features strictly from prior months and define the forecasting target as the next synthetic monthly total.
- Export `training/datasets/forecaster_v3/` with feature provenance, train-only imputation statistics, and tests that fail when a target-period value, a future month, or a held-out household contributes to a feature.

### 8. Train and Evaluate Without Overclaiming

- Create `training/scripts/train_forecaster_v3.py`, `training/scripts/evaluate_forecaster_v3.py`, and `tests/test_forecaster_v3_training.py`; do not modify the existing v1/v2 training or evaluation scripts.
- Compare the same candidate families against naive baselines using walk-forward folds and unseen-household evaluation, but label the result as internal synthetic-target evaluation because the monthly targets were produced by the allocation rule.
- Add an external validation gate before artifact release: use real household transactions when available, or a strictly withheld later PSA HFCE period for aggregate validation; record the validation level and prohibit a household-level accuracy claim from aggregate-only evidence.

### 9. Regenerate, Audit, and Document the v3 Release Candidate

- Create `training/docs/data-collection/survey-only-forecast-v3-runbook.md`, `docs/models/forecaster-v3.md`, and `tests/test_survey_v3_isolation.py`; update `README.md` and `INDEX.md` with additive v3 links only after the pipeline passes verification.
- Regenerate ignored `survey_v3/`, `training/datasets/processed_v3/`, and `training/datasets/forecaster_v3/`; write a v3 provenance report that distinguishes observed FIES, observed PSA HFCE, and synthetic monthly or transaction fields.
- The isolation test must prove all v1/v2 tracked source, config, artifact, and documentation paths remain unmodified, and the model note must state that model serving remains unchanged until a separate artifact-release approval.

## Acceptance Criteria

- V3 contains no generated persona, archetype, PFP, synthetic-income, debt, savings, or behavioral fields.
- No v1 or v2 source file, configuration, generated dataset, model artifact, evaluation record, metadata file, or documentation file changes during v3 implementation or regeneration.
- Every v3 household is selected from the 2023 FIES PUF under a documented, reproducible contract and is exported only under a deterministic pseudonymous ID.
- The six annual expenditure categories are observed FIES values or a documented residual that reconciles to the chosen observed FIES total definition.
- For each household and category, complete-year generated monthly totals reconcile to its observed annual benchmark; 2026 Q1-Q2 totals reconcile to six benchmark months and no later 2026 month exists.
- Aggregate generated quarter shares match the configured year-specific PSA HFCE category weights within rounding tolerance.
- Train, validation, and test household IDs are disjoint, and features use no target-period, future-period, or held-out-household information.
- All generated artifacts carry FIES source and config fingerprints, selection-contract version, timeline, and code revision.
- `pytest tests/test_fies_households_v3.py tests/test_survey_monthly_v3.py tests/test_preprocessor_v3.py tests/test_forecaster_features_v3.py tests/test_forecaster_v3_training.py -v`, `ruff check training tests`, `ruff format --check training tests`, and `mypy training/scripts` pass.
- No v3 model artifact replaces the served forecaster without the external-validation gate and explicit release approval.
