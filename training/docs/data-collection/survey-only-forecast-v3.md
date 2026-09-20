# Survey-Only Forecast v3 Contract

V3 selects eligible 2023 FIES public-use households and retains observed annual expenditure only. The output has a deterministic SHA-256-derived `household_id_v3`; raw `SEQ_NO` is never exported.

- Geographic scope: all reported 2023 FIES regions.
- Duplicate or missing `SEQ_NO`: exclude every row in the duplicate/missing group.
- Missing required numeric expenditure, non-positive `TOTEX`, non-positive household size, negative mapped expenditure, or negative residual: exclude the row. No values are imputed or winsorized.
- `RFACT`: not used for training or diagnostics in this version.
- Categories: `food`, `housing_water`, `health`, `transport`, and `education` map directly from FIES. `other = TOTEX - five direct categories`; all six categories must reconcile to `TOTEX` within PHP 0.005.

Monthly amounts are synthetic allocations, not observed household transaction histories. Each available calendar-year/category allocation uses published current-price PSA HFCE quarterly shares and equal monthly splits within a quarter. V3 covers 2023-01 through 2026-06 only.
