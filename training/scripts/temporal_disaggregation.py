"""Calendar-year HFCE temporal disaggregation for Synthetic Generation v2."""

from __future__ import annotations

import json
from pathlib import Path

SYNTH_VERSION = "2.1.0"

HFCE_CATEGORIES: tuple[str, ...] = (
    "food",
    "housing",
    "health",
    "transport",
    "education",
    "other",
)
ESSENTIAL_CATEGORIES: tuple[str, ...] = (
    "food",
    "housing",
    "health",
    "transport",
    "education",
)
NON_ESSENTIAL_SERIES: tuple[str, ...] = (
    "alcohol_tobacco",
    "clothing",
    "furnishings",
    "communication",
    "recreation",
    "restaurants_hotels",
    "miscellaneous",
)
QUARTER_KEYS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")
QUARTER_MONTHS: dict[str, tuple[int, int, int]] = {
    "Q1": (1, 2, 3),
    "Q2": (4, 5, 6),
    "Q3": (7, 8, 9),
    "Q4": (10, 11, 12),
}
DEFAULT_HFCE_PATH = Path(__file__).resolve().parent.parent / "config" / "hfce_quarterly_indices.json"


class HFCEConfigError(Exception):
    """Raised when the HFCE config is missing, malformed, or lacks a requested period."""


def _load_config(hfce_path: str | Path | None) -> dict:
    path = Path(hfce_path) if hfce_path is not None else DEFAULT_HFCE_PATH
    if not path.exists():
        raise HFCEConfigError(f"HFCE config not found: {path}")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HFCEConfigError(f"Unable to read HFCE config {path}: {error}") from error
    if not isinstance(data, dict) or data.get("price_basis") != "current_prices":
        raise HFCEConfigError(f"HFCE config {path} must declare current_prices")
    return data


def available_quarters(year: int, hfce_path: str | Path | None = None) -> tuple[str, ...]:
    """Return the published sequential quarters available for a calendar year."""
    data = _load_config(hfce_path)
    try:
        quarters = tuple(data["years"][str(year)]["quarters"])
    except (KeyError, TypeError) as error:
        raise HFCEConfigError(f"HFCE config has no data for year {year}") from error
    expected = QUARTER_KEYS[: len(quarters)]
    if not quarters or quarters != expected:
        raise HFCEConfigError(f"HFCE year {year} must contain sequential quarters from Q1")
    return quarters


def available_months(year: int, hfce_path: str | Path | None = None) -> tuple[int, ...]:
    """Return calendar months supported by the published quarters for ``year``."""
    return tuple(month for quarter in available_quarters(year, hfce_path) for month in QUARTER_MONTHS[quarter])


def validate_hfce_coverage(
    start_year: int, start_month: int, num_months: int, hfce_path: str | Path | None = None
) -> None:
    """Reject a requested timeline containing a month without published HFCE data."""
    for month_offset in range(num_months):
        month = ((start_month - 1 + month_offset) % 12) + 1
        year = start_year + (start_month - 1 + month_offset) // 12
        if month not in available_months(year, hfce_path):
            raise HFCEConfigError(f"HFCE data is unavailable for {year}-{month:02d}")


def load_hfce_levels(year: int, hfce_path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load current-price HFCE levels for a year and derive the residual ``other`` series."""
    data = _load_config(hfce_path)
    quarters = available_quarters(year, hfce_path)
    try:
        series = data["years"][str(year)]["psa_series"]
    except (KeyError, TypeError) as error:
        raise HFCEConfigError(f"HFCE config has no PSA series for year {year}") from error

    levels: dict[str, dict[str, float]] = {}
    for category in ESSENTIAL_CATEGORIES:
        try:
            levels[category] = {quarter: float(series[category][quarter]) for quarter in quarters}
        except (KeyError, TypeError, ValueError) as error:
            raise HFCEConfigError(f"HFCE {year} is missing {category} data") from error
    try:
        levels["other"] = {
            quarter: sum(float(series[name][quarter]) for name in NON_ESSENTIAL_SERIES)
            for quarter in quarters
        }
    except (KeyError, TypeError, ValueError) as error:
        raise HFCEConfigError(f"HFCE {year} cannot derive the other residual") from error

    for category, category_levels in levels.items():
        if any(value <= 0 for value in category_levels.values()):
            raise HFCEConfigError(f"HFCE {year} has non-positive {category} levels")
    return levels


def quarter_weights(levels: dict[str, float]) -> dict[str, float]:
    """Convert one year's available quarterly levels into allocation weights."""
    quarters = tuple(levels)
    if not quarters or quarters != QUARTER_KEYS[: len(quarters)]:
        raise HFCEConfigError("HFCE levels must contain sequential quarters from Q1")
    total = sum(levels.values())
    if total <= 0:
        raise HFCEConfigError(f"Cannot compute quarterly weights from non-positive total {total}")
    return {quarter: levels[quarter] / total for quarter in quarters}


def disaggregate_annual(annual_amount: float, weights: dict[str, float]) -> dict[str, float]:
    """Allocate a benchmark amount across the supplied quarter weights."""
    return {quarter: annual_amount * weight for quarter, weight in weights.items()}


def reconcile_to_annual(annual_amount: float, provisional_monthly: dict[int, float]) -> dict[int, float]:
    """Scale provisional monthly values to exactly match the applicable benchmark."""
    total = sum(provisional_monthly.values())
    if total == 0:
        if annual_amount == 0:
            return dict.fromkeys(provisional_monthly, 0.0)
        return dict.fromkeys(provisional_monthly, annual_amount / len(provisional_monthly))
    factor = annual_amount / total
    return {month: value * factor for month, value in provisional_monthly.items()}


def build_year_schedule(
    annual_by_category: dict[str, float], year: int, hfce_path: str | Path | None = None
) -> dict[str, dict[int, float]]:
    """Build a schedule for the published months of one calendar year.

    Complete years reconcile to a twelve-month benchmark. A partial 2026 schedule
    reconciles Q1-Q2 to its six-month benchmark; it never fabricates Q3-Q4 values.
    """
    levels_by_category = load_hfce_levels(year, hfce_path)
    schedule: dict[str, dict[int, float]] = {}
    for category, annual_amount in annual_by_category.items():
        if category not in levels_by_category:
            raise HFCEConfigError(f"Unknown HFCE category: {category!r}")
        quarterly = disaggregate_annual(float(annual_amount), quarter_weights(levels_by_category[category]))
        provisional = {
            month: quarterly[quarter] / len(QUARTER_MONTHS[quarter])
            for quarter in quarterly
            for month in QUARTER_MONTHS[quarter]
        }
        schedule[category] = reconcile_to_annual(float(annual_amount), provisional)
    return schedule


def month_amount(schedule: dict[str, dict[int, float]], category: str, calendar_month: int) -> float:
    """Look up a schedule amount, rejecting months outside the published coverage."""
    if category not in schedule:
        raise HFCEConfigError(f"Category '{category}' not present in schedule")
    try:
        return schedule[category][calendar_month]
    except KeyError as error:
        raise HFCEConfigError(
            f"Month {calendar_month} is not available in this year's HFCE schedule"
        ) from error
