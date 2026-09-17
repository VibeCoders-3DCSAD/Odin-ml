"""Unit tests for training/scripts/temporal_disaggregation.py (Synthetic Generation v2)."""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "training" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from temporal_disaggregation import (  # noqa: E402
    DEFAULT_HFCE_PATH,
    ESSENTIAL_CATEGORIES,
    HFCE_CATEGORIES,
    SYNTH_VERSION,
    HFCEConfigError,
    build_year_schedule,
    disaggregate_annual,
    load_hfce_levels,
    month_amount,
    quarter_weights,
    reconcile_to_annual,
)


def test_synth_version_is_2_0_0() -> None:
    assert SYNTH_VERSION == "2.0.0"


def test_categories_and_essentials() -> None:
    assert set(ESSENTIAL_CATEGORIES) == {
        "food",
        "housing",
        "health",
        "transport",
        "education",
    }
    assert set(HFCE_CATEGORIES) == set(ESSENTIAL_CATEGORIES) | {"other"}


def test_default_hfce_path_exists() -> None:
    assert DEFAULT_HFCE_PATH.is_file()


def test_load_real_hfce_other_residual_q1() -> None:
    levels = load_hfce_levels()
    assert levels["other"]["Q1"] == pytest.approx(1255736)
    assert levels["other"]["Q2"] == pytest.approx(1032737)
    assert levels["other"]["Q3"] == pytest.approx(1204677)
    assert levels["other"]["Q4"] == pytest.approx(1562440)


def test_load_real_hfce_all_categories_present() -> None:
    levels = load_hfce_levels()
    assert set(levels.keys()) == set(HFCE_CATEGORIES)
    for category in HFCE_CATEGORIES:
        assert set(levels[category].keys()) == {"Q1", "Q2", "Q3", "Q4"}
        for q in ("Q1", "Q2", "Q3", "Q4"):
            assert levels[category][q] > 0


def test_load_hfce_missing_file_raises() -> None:
    with pytest.raises(HFCEConfigError):
        load_hfce_levels("/nonexistent/path/hfce.json")


def test_load_hfce_malformed_json_raises(tmp_path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json")
    with pytest.raises(HFCEConfigError):
        load_hfce_levels(bad_file)


def test_load_hfce_missing_category_raises(tmp_path) -> None:
    bad_file = tmp_path / "incomplete.json"
    bad_file.write_text('{"categories": {"food": {"Q1": 1, "Q2": 1, "Q3": 1, "Q4": 1}}}')
    with pytest.raises(HFCEConfigError):
        load_hfce_levels(bad_file)


def test_quarter_weights_sum_to_one() -> None:
    levels = load_hfce_levels()
    for category in HFCE_CATEGORIES:
        weights = quarter_weights(levels[category])
        assert sum(weights.values()) == pytest.approx(1.0)


def test_quarter_weights_food_q4_matches_methodology() -> None:
    levels = load_hfce_levels()
    weights = quarter_weights(levels["food"])
    # Methodology §5: W_food,Q4 = 0.296567 (29.6567%)
    assert weights["Q4"] == pytest.approx(0.296567, abs=1e-5)


def test_quarter_weights_non_positive_total_raises() -> None:
    with pytest.raises(HFCEConfigError):
        quarter_weights({"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0})


def test_disaggregate_annual_sums_to_annual() -> None:
    weights = {"Q1": 0.25, "Q2": 0.25, "Q3": 0.25, "Q4": 0.25}
    quarterly = disaggregate_annual(120000.0, weights)
    assert sum(quarterly.values()) == pytest.approx(120000.0)


def test_disaggregate_annual_food_q4_example() -> None:
    # Methodology §6 example: A=120,000, W_food_Q4=0.296567 -> Q4 = 35,588.04
    levels = load_hfce_levels()
    weights = quarter_weights(levels["food"])
    quarterly = disaggregate_annual(120000.0, weights)
    assert quarterly["Q4"] == pytest.approx(35588.04, abs=0.5)


def test_reconcile_to_annual_exact() -> None:
    provisional = dict.fromkeys(range(1, 13), 100.0)  # provisional total = 1200
    reconciled = reconcile_to_annual(1000.0, provisional)
    assert sum(reconciled.values()) == pytest.approx(1000.0)


def test_reconcile_to_annual_zero_annual_returns_zeros() -> None:
    provisional = dict.fromkeys(range(1, 13), 0.0)
    reconciled = reconcile_to_annual(0.0, provisional)
    assert all(v == 0.0 for v in reconciled.values())


def test_reconcile_to_annual_zero_provisional_nonzero_annual_no_crash() -> None:
    provisional = dict.fromkeys(range(1, 13), 0.0)
    reconciled = reconcile_to_annual(1200.0, provisional)
    assert sum(reconciled.values()) == pytest.approx(1200.0)


def test_build_year_schedule_annual_sum_matches_input() -> None:
    annual_by_category = dict.fromkeys(HFCE_CATEGORIES, 12000.0)
    schedule = build_year_schedule(annual_by_category)

    for category, annual_amount in annual_by_category.items():
        assert sum(schedule[category].values()) == pytest.approx(annual_amount, abs=1e-6)
        assert set(schedule[category].keys()) == set(range(1, 13))


def test_build_year_schedule_zero_annual() -> None:
    annual_by_category = dict.fromkeys(HFCE_CATEGORIES, 0.0)
    schedule = build_year_schedule(annual_by_category)
    for category in HFCE_CATEGORIES:
        assert all(v == 0.0 for v in schedule[category].values())


def test_build_year_schedule_unknown_category_raises() -> None:
    with pytest.raises(HFCEConfigError):
        build_year_schedule({"not_a_category": 1000.0})


def test_month_amount_matches_schedule() -> None:
    annual_by_category = {"food": 120000.0}
    schedule = build_year_schedule(annual_by_category)
    for month in range(1, 13):
        assert month_amount(schedule, "food", month) == pytest.approx(schedule["food"][month])


def test_month_amount_wraps_month_13_equals_month_1() -> None:
    annual_by_category = {"food": 120000.0}
    schedule = build_year_schedule(annual_by_category)
    assert month_amount(schedule, "food", 13) == pytest.approx(month_amount(schedule, "food", 1))
    assert month_amount(schedule, "food", 25) == pytest.approx(month_amount(schedule, "food", 1))
    assert month_amount(schedule, "food", 24) == pytest.approx(month_amount(schedule, "food", 12))


def test_month_amount_unknown_category_raises() -> None:
    schedule = build_year_schedule({"food": 1000.0})
    with pytest.raises(HFCEConfigError):
        month_amount(schedule, "housing", 1)
