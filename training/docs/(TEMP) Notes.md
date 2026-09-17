Ung d=0, tsaka D=1 values
Di ko nakita san nakuha
Tsaka mali rin ung nasa code
ARIMA lang ung ginagamjt
Kasi lahat ng months pinapagsama, nagiging 1 year lang ung span ng data
Jan2024, at Jan2025 datas nagiging isang January lang
Di nagagamit ung SARIMA kasi minimum 24 months of data nakalagay sa code

Tapos pinoproduce nito 12 months lang

---

## Teammate Q&A — Forecaster (mga sagot, Sep 2026)

### 1. Saan nanggaling ang d=0 at D=1?
Wala pong `d=0`/`D=1` kahit saan sa code. Ang ginamit ay:
- `ARIMA_ORDER = (1, 1, 0)` → `(p, d, q) = (1, 1, 0)` → **d = 1** (`training/scripts/train_forecaster.py:129`)
- `seasonal_order = (1, 0, 0, 12)` → `(P, D, Q, m) = (1, 0, 0, 12)` → **D = 0**, period 12 (`train_forecaster.py:411` at `:938`)
- Kapareho rin ang nakasulat sa `docs/models/forecaster.md:64`. Kung may pinagkukunan na d=0/D=1, galing iyon sa labas ng repo (o binaligtad lang ang transcription).

### 2. "ARIMA lang ang ginagamit" — tama
Ang tier-3 winner ay pinangalanang `sarima`, pero ang seasonal SARIMAX branch ay naka-gate sa `len(pooled) >= 24` (`train_forecaster.py:408` at `:935`). Sa kasalukuyang 12-month pool (≤ 12 rows), laging fallback sa plain `ARIMA(1,1,0)` (`:416`, `:942`). Patunay: `models/forecaster/evaluation_report.md:23-24` — magkapareho ang metrics ng `tier3_arima` at `tier3_sarima` (9.3957% MAPE ang pareho).

### 3. "Lahat ng months pinagsasama = 1 year lang ang span / Jan2024+Jan2025 → isang January"
Tama, at structural ito:
- Pooling: `hist.groupby("month")["norm"].mean()` (`train_forecaster.py:377,405`) — naka-group siya by **month-of-year (1–12)**.
- Ang corpus ay single-year lang (2023): `build_daily_grid(..., year=2023)` (`feature_engineering_forecaster.py:135-147`), kaya 12 months max ang pool.
- Kahit pa multi-year ang data, ang `groupby("month")` ay pinagsasama ang Jan-2024 at Jan-2025 sa iisang January (ini-average) → nawawala ang seasonal axis bago pa makita ng ARIMA.
- Serve side: `build_monthly_summaries` ay 12 buckets (1–12) at nagsusuma across years (`app/services/features.py:62-90`); ang `tail(3)` ay months 10–12 lamang (`forecast_service.py:32-34`).

### 4. "Min 24 months, kaya di nagagamit ang SARIMA" — tama
`if len(pooled) >= 24:` (`:408`, `:935`). Hindi ito umaabot kasi ≤ 12 rows ang pool. Kahit dagdagan pa ng 24 months ang raw data, hindi pa rin sapat ang `groupby("month")` — kailangan ng absolute month index (ito ang ginawang fix — tingnan ang #6).

### 5. "12 months lang ang pinoproduce"
Sa API, ang YEARLY horizon ay 12 monthly points (`pd.date_range(..., periods=12, freq="MS")`, `forecast_service.py:141`). Ang MONTHLY → 4 weeks, ang SEMI_MONTHLY → 2, ang WEEKLY → 7. Ang model mismo ay 1-month pooled forecast lang. Design choice ito ng produkto, hindi seasonal cycle.

### 6. Fix na na-apply (kasama sa commit na ito): absolute month index
- `month_num = (year − 2023)·12 + month` sa forecaster grid at sa pooling — hindi na magsasama ang Jan-2024 at Jan-2025.
- SARIMA gate = span ≥ 24 distinct absolute months (hindi na `len(pooled)`).
- Serve `level` = last 3 **chronological** positive-expense months (hindi na months 10–12 bucket).
- Behavior: identical pa rin sa 2023 single-year data; maa-activate lang ang SARIMA kapag ≥ 24 absolute months ang data.