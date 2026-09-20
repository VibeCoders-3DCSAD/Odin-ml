"""Unit tests for calendar-year current-price HFCE disaggregation."""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "training" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from temporal_disaggregation import (  # noqa: E402
    DEFAULT_HFCE_PATH,
    HFCE_CATEGORIES,
    HFCEConfigError,
    SYNTH_VERSION,
    available_months,
    available_quarters,
    build_year_schedule,
    load_hfce_levels,
    month_amount,
    quarter_weights,
    validate_hfce_coverage,
)


def test_current_price_config_covers_requested_period() -> None:
    assert DEFAULT_HFCE_PATH.is_file()
    assert SYNTH_VERSION == "2.1.0"
    assert available_quarters(2023) == ("Q1", "Q2", "Q3", "Q4")
    assert available_quarters(2026) == ("Q1", "Q2")
    assert available_months(2026) == (1, 2, 3, 4, 5, 6)


def test_loads_supplied_current_price_levels_and_other_residual() -> None:
    levels_2023 = load_hfce_levels(2023)
    levels_2025 = load_hfce_levels(2025)
    levels_2026 = load_hfce_levels(2026)

    assert levels_2023["food"]["Q1"] == 1536533
    assert levels_2025["food"]["Q4"] == 2257594
    assert levels_2026["food"]["Q2"] == 2058217
    assert levels_2026["other"]["Q1"] == 1930257
    assert set(levels_2026["food"]) == {"Q1", "Q2"}


def test_year_specific_weights_are_not_replayed() -> None:
    assert quarter_weights(load_hfce_levels(2023)["food"]) != quarter_weights(
        load_hfce_levels(2024)["food"]
    )


def test_complete_year_schedule_reconciles_to_twelve_month_benchmark() -> None:
    annual_by_category = dict.fromkeys(HFCE_CATEGORIES, 12000.0)
    schedule = build_year_schedule(annual_by_category, 2025)

    for category in HFCE_CATEGORIES:
        assert set(schedule[category]) == set(range(1, 13))
        assert sum(schedule[category].values()) == pytest.approx(12000.0)


def test_partial_2026_schedule_reconciles_to_six_month_benchmark() -> None:
    schedule = build_year_schedule({"food": 6000.0}, 2026)
    assert set(schedule["food"]) == {1, 2, 3, 4, 5, 6}
    assert sum(schedule["food"].values()) == pytest.approx(6000.0)
    with pytest.raises(HFCEConfigError, match="not available"):
        month_amount(schedule, "food", 7)
    with pytest.raises(HFCEConfigError, match="2026-07"):
        validate_hfce_coverage(2023, 1, 43)


def test_missing_year_and_unknown_category_raise() -> None:
    with pytest.raises(HFCEConfigError, match="no data for year"):
        load_hfce_levels(2027)
    with pytest.raises(HFCEConfigError, match="Unknown HFCE category"):
        build_year_schedule({"not_a_category": 1000.0}, 2023)
