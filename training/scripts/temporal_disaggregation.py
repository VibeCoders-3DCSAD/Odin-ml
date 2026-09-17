"""
Temporal Disaggregation — Synthetic Generation v2

Implements the core math of the FIES-anchored, HFCE-calibrated temporal
disaggregation with proportional benchmarking methodology:

    A_c (annual)  -->  H_{c,q} (PSA quarterly HFCE)  -->  W_{c,q} (quarterly weight)
        --> Q_{c,q} = A_c * W_{c,q}  --> E_{c,m} = Q_{c,q(m)} / 3  --> reconcile to A_c

See: training/docs/data-collection/fies-hfce-synthetic-data-generation-methodology.md

This module is part of **Synthetic Generation v2**, a pipeline that runs in
*parallel* to the original (v1) synthetic generator
(`generate_personas.py`, `generate_transactions.py`, `synthesizer.py`,
`preprocessor.py`). It does not modify any v1 module.

Usage:
    from temporal_disaggregation import build_year_schedule, month_amount
"""

from __future__ import annotations

import json
from pathlib import Path

SYNTH_VERSION = "2.0.0"

# The six expenditure categories used by the v2 generator. "other" is the
# non-essential residual (Total HFCE - essentials), not the narrower PSA
# "Miscellaneous goods and services" series alone.
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

QUARTER_KEYS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

# Equal-thirds month membership per quarter (methodology §7).
QUARTER_MONTHS: dict[str, tuple[int, int, int]] = {
    "Q1": (1, 2, 3),
    "Q2": (4, 5, 6),
    "Q3": (7, 8, 9),
    "Q4": (10, 11, 12),
}

DEFAULT_HFCE_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "hfce_quarterly_indices.json"
)


class HFCEConfigError(Exception):
    """Raised when the HFCE config file is missing, malformed, or incomplete."""

    pass


def load_hfce_levels(
    hfce_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """
    Load PSA HFCE quarterly levels per category from the HFCE config JSON.

    Args:
        hfce_path: Path to the HFCE config JSON. Defaults to
            ``training/config/hfce_quarterly_indices.json``.

    Returns:
        Mapping of ``category -> {"Q1": float, "Q2": float, "Q3": float, "Q4": float}``
        for each category in ``HFCE_CATEGORIES``.

    Raises:
        HFCEConfigError: If the file is missing, malformed, or a required
            category/quarter is missing or non-numeric.
    """
    path = Path(hfce_path) if hfce_path is not None else DEFAULT_HFCE_PATH

    if not path.exists():
        raise HFCEConfigError(f"HFCE config not found: {path}")

    try:
        raw = path.read_text()
    except OSError as e:
        raise HFCEConfigError(f"Failed to read HFCE config {path}: {e}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HFCEConfigError(f"Invalid JSON in HFCE config {path}: {e}")

    if not isinstance(data, dict) or "categories" not in data:
        raise HFCEConfigError(f"HFCE config {path} is missing the 'categories' key")

    categories = data["categories"]

    levels: dict[str, dict[str, float]] = {}
    for category in HFCE_CATEGORIES:
        if category not in categories:
            raise HFCEConfigError(
                f"HFCE config {path} is missing category '{category}'"
            )

        entry = categories[category]
        if not isinstance(entry, dict):
            raise HFCEConfigError(
                f"HFCE config {path} category '{category}' is not an object"
            )

        quarters: dict[str, float] = {}
        for quarter in QUARTER_KEYS:
            if quarter not in entry:
                raise HFCEConfigError(
                    f"HFCE config {path} category '{category}' is missing {quarter}"
                )
            try:
                quarters[quarter] = float(entry[quarter])
            except (TypeError, ValueError):
                raise HFCEConfigError(
                    f"HFCE config {path} category '{category}' has non-numeric "
                    f"{quarter}: {entry[quarter]!r}"
                )

        levels[category] = quarters

    return levels


def quarter_weights(levels: dict[str, float]) -> dict[str, float]:
    """
    Convert quarterly HFCE levels into quarterly allocation weights.

    W_{c,q} = H_{c,q} / sum_j(H_{c,j})

    Args:
        levels: ``{"Q1": float, "Q2": float, "Q3": float, "Q4": float}``.

    Returns:
        Weights summing to 1.0 (up to floating-point precision).

    Raises:
        HFCEConfigError: If the total of the levels is non-positive.
    """
    total = sum(levels.get(q, 0.0) for q in QUARTER_KEYS)
    if total <= 0:
        raise HFCEConfigError(
            f"Cannot compute quarterly weights: total HFCE level is non-positive "
            f"({total}) for levels={levels}"
        )
    return {q: levels[q] / total for q in QUARTER_KEYS}


def disaggregate_annual(
    annual_amount: float, weights: dict[str, float]
) -> dict[str, float]:
    """
    Split an annual amount into quarterly amounts using quarterly weights.

    Q_{c,q} = A_c * W_{c,q}

    Because sum_q(W_{c,q}) == 1, sum_q(Q_{c,q}) == A_c (up to rounding).
    """
    return {q: annual_amount * weights[q] for q in QUARTER_KEYS}


def reconcile_to_annual(
    annual_amount: float, provisional_monthly: dict[int, float]
) -> dict[int, float]:
    """
    Scale provisional monthly values so they sum exactly to ``annual_amount``.

    K_c = A_c / sum_m(E^(0)_{c,m})
    E_{c,m} = K_c * E^(0)_{c,m}

    Args:
        annual_amount: The authoritative annual total (A_c).
        provisional_monthly: ``{month(1-12): provisional_amount}``.

    Returns:
        Reconciled monthly amounts whose sum equals ``annual_amount``
        (up to floating-point precision).
    """
    total = sum(provisional_monthly.values())

    if total == 0:
        # Degenerate case: provisional values are all zero (e.g. annual_amount == 0,
        # or a zero-weight edge case). If the annual target is also zero, zeros are
        # already correct. Otherwise, fall back to an equal split across all months
        # to avoid a division-by-zero crash while still summing to annual_amount.
        if annual_amount == 0:
            return {m: 0.0 for m in provisional_monthly}
        n = len(provisional_monthly) or 1
        return {m: annual_amount / n for m in provisional_monthly}

    k = annual_amount / total
    return {m: v * k for m, v in provisional_monthly.items()}


def build_year_schedule(
    annual_by_category: dict[str, float],
    hfce_path: str | Path | None = None,
) -> dict[str, dict[int, float]]:
    """
    Build a 12-month (calendar months 1-12) expenditure schedule per category.

    For each category, the annual amount is disaggregated into quarters using the
    PSA HFCE quarterly weights, split equally across each quarter's three months,
    and reconciled so the twelve monthly values sum exactly to the annual amount.

    Args:
        annual_by_category: ``{category: annual_amount}`` for categories in
            ``HFCE_CATEGORIES``.
        hfce_path: Optional override for the HFCE config path.

    Returns:
        ``{category: {month(1-12): amount}}``.

    Raises:
        HFCEConfigError: If a category is not a known HFCE category, or if the
            HFCE config itself is invalid.
    """
    levels_by_category = load_hfce_levels(hfce_path)

    schedule: dict[str, dict[int, float]] = {}

    for category, annual_amount in annual_by_category.items():
        if category not in levels_by_category:
            raise HFCEConfigError(f"Unknown HFCE category: {category!r}")

        weights = quarter_weights(levels_by_category[category])
        quarterly = disaggregate_annual(float(annual_amount), weights)

        provisional: dict[int, float] = {}
        for quarter, months in QUARTER_MONTHS.items():
            month_provisional = quarterly[quarter] / 3
            for month in months:
                provisional[month] = month_provisional

        schedule[category] = reconcile_to_annual(float(annual_amount), provisional)

    return schedule


def month_amount(
    schedule: dict[str, dict[int, float]],
    category: str,
    calendar_month: int,
) -> float:
    """
    Look up the scheduled amount for a category at an arbitrary calendar month.

    Calendar months beyond 12 (multi-year runs) wrap onto the same 12-month HFCE
    pattern: ``month_index = ((calendar_month - 1) % 12) + 1``. This is a
    documented limitation — v2.0 does not model year-over-year seasonality drift.

    Args:
        schedule: Output of ``build_year_schedule``.
        category: HFCE category name.
        calendar_month: 1-based calendar month, may exceed 12.

    Returns:
        The scheduled amount for that category/month.

    Raises:
        HFCEConfigError: If ``category`` is not present in ``schedule``.
    """
    if category not in schedule:
        raise HFCEConfigError(f"Category '{category}' not present in schedule")

    month_index = ((calendar_month - 1) % 12) + 1
    return schedule[category][month_index]
