"""
Integration tests for training/scripts/generate_transactions_v2.py
(Synthetic Generation v2).

Validation IDs mirror methodology §13 / the v2 implementation plan:

    V0 - v1 untouched guard still green
    V1 - Per persona x category annual consistency
    V2 - Population quarterly shares match HFCE weights
    V3 - Monthly transaction sums == HFCE-derived schedule
    V4 - Deterministic expenses given seed + HFCE config
    V5 - Report synth_version == "2.0.0"

All exactness checks (V1-V3) run with anomalies disabled, since anomaly
injection intentionally perturbs individual transaction amounts *after*
reconciliation (methodology: "anomalies after reconcile only; no
re-reconcile after anomalies").
"""

import sys
from collections import defaultdict
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "training" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_transactions_v2 import (  # noqa: E402
    build_synth_v2_report,
    generate_persona_transactions_v2,
)
from temporal_disaggregation import (  # noqa: E402
    HFCE_CATEGORIES,
    QUARTER_MONTHS,
    build_year_schedule,
    load_hfce_levels,
    quarter_weights,
)


def _persona() -> dict:
    return {
        "persona_id": "persona-v2-test",
        "archetype_id": "archetype-A",
        "archetype_name": "Test Archetype",
        "monthly_income": 40000.0,
        "income_cv": 0.1,
        "obligation_ratio": 0.6,
        "savings_rate": 0.1,
        "runway_months": 4.0,
        "household_size": 2,
        "income_pattern": "regular",
        "food_expense": 8000.0,
        "housing_expense": 6000.0,
        "transport_expense": 2500.0,
        "health_expense": 1200.0,
        "education_expense": 1500.0,
        "other_expense": 3000.0,
    }


def _category_monthly_expense_totals(transactions) -> dict[tuple[int, str], float]:
    """Sum expense-transaction amounts by (month, category)."""
    totals: dict[tuple[int, str], float] = defaultdict(float)
    for txn in transactions:
        if txn.transaction_type == "expense":
            totals[(txn.month, txn.category)] += txn.amount
    return totals


# ---------------------------------------------------------------------------
# V0 — v1 untouched guard still green (see also test_synth_v1_untouched.py)
# ---------------------------------------------------------------------------


def test_v0_v1_generate_transactions_untouched_guard_still_green() -> None:
    src = (SCRIPTS_DIR / "generate_transactions.py").read_text()
    assert "rng.normal(1.0, 0.15)" in src
    assert "max(0.5, min(2.0, variation))" in src


# ---------------------------------------------------------------------------
# V1 — Per persona x category annual consistency
# ---------------------------------------------------------------------------


def test_v1_annual_sum_matches_persona_benchmark() -> None:
    persona = _persona()
    transactions, summaries = generate_persona_transactions_v2(
        persona, num_months=12, seed=42, inject_anomalies_flag=False
    )

    monthly_totals = _category_monthly_expense_totals(transactions)

    for category in HFCE_CATEGORIES:
        annual_benchmark = persona.get(f"{category}_expense", 0) * 12
        annual_generated = sum(
            amount for (month, cat), amount in monthly_totals.items() if cat == category
        )
        assert annual_generated == pytest.approx(annual_benchmark, abs=0.05)

    assert len(summaries) == 12


# ---------------------------------------------------------------------------
# V2 — Population quarterly shares match HFCE weights
# ---------------------------------------------------------------------------


def test_v2_quarterly_shares_match_hfce_weights() -> None:
    persona = _persona()
    transactions, _ = generate_persona_transactions_v2(
        persona, num_months=12, seed=42, inject_anomalies_flag=False
    )
    monthly_totals = _category_monthly_expense_totals(transactions)

    levels = load_hfce_levels()

    for category in HFCE_CATEGORIES:
        annual_generated = sum(
            amount for (month, cat), amount in monthly_totals.items() if cat == category
        )
        if annual_generated == 0:
            continue

        expected_weights = quarter_weights(levels[category])

        for quarter, months in QUARTER_MONTHS.items():
            quarter_amount = sum(
                amount
                for (month, cat), amount in monthly_totals.items()
                if cat == category and month in months
            )
            observed_share = quarter_amount / annual_generated
            assert observed_share == pytest.approx(expected_weights[quarter], abs=1e-3)


# ---------------------------------------------------------------------------
# V3 — Monthly transaction sums == HFCE-derived schedule
# ---------------------------------------------------------------------------


def test_v3_monthly_sums_equal_schedule() -> None:
    persona = _persona()
    transactions, _ = generate_persona_transactions_v2(
        persona, num_months=12, seed=42, inject_anomalies_flag=False
    )
    monthly_totals = _category_monthly_expense_totals(transactions)

    annual_by_category = {
        category: persona.get(f"{category}_expense", 0) * 12 for category in HFCE_CATEGORIES
    }
    schedule = build_year_schedule(annual_by_category)

    for category in HFCE_CATEGORIES:
        for month in range(1, 13):
            expected = schedule[category][month]
            observed = monthly_totals.get((month, category), 0.0)
            assert observed == pytest.approx(expected, abs=0.05)


def test_weekly_category_split_sums_exactly() -> None:
    """Food is a 'weekly' category (4 transactions/month); residual on the
    last transaction must make the 4 amounts sum exactly to the month's
    scheduled amount."""
    persona = _persona()
    transactions, _ = generate_persona_transactions_v2(
        persona, num_months=1, seed=42, inject_anomalies_flag=False
    )

    food_txns = [t for t in transactions if t.category == "food"]
    assert len(food_txns) == 4

    annual_by_category = {"food": persona["food_expense"] * 12}
    schedule = build_year_schedule(annual_by_category)
    expected = schedule["food"][1]

    assert sum(t.amount for t in food_txns) == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# V4 — Deterministic expenses given seed + HFCE config
# ---------------------------------------------------------------------------


def test_v4_deterministic_given_same_seed() -> None:
    persona = _persona()

    transactions_a, summaries_a = generate_persona_transactions_v2(
        persona, num_months=12, seed=7
    )
    transactions_b, summaries_b = generate_persona_transactions_v2(
        persona, num_months=12, seed=7
    )

    assert [t.transaction_id for t in transactions_a] == [
        t.transaction_id for t in transactions_b
    ]
    assert [t.amount for t in transactions_a] == [t.amount for t in transactions_b]
    assert [s.total_expenses for s in summaries_a] == [s.total_expenses for s in summaries_b]


def test_v4_different_seed_can_differ() -> None:
    persona = _persona()

    transactions_a, _ = generate_persona_transactions_v2(persona, num_months=12, seed=1)
    transactions_b, _ = generate_persona_transactions_v2(persona, num_months=12, seed=2)

    dates_a = [t.date for t in transactions_a]
    dates_b = [t.date for t in transactions_b]
    assert dates_a != dates_b


# ---------------------------------------------------------------------------
# V5 — Report synth_version == "2.0.0"
# ---------------------------------------------------------------------------


def test_v5_report_stamps_synth_version() -> None:
    persona = _persona()
    _, summaries = generate_persona_transactions_v2(persona, num_months=12, seed=42)

    report = build_synth_v2_report(summaries)

    assert report["synth_version"] == "2.0.0"
    assert report["v1_untouched"] is True
    assert report["parallel_to"] == "training/scripts/generate_transactions.py"
    assert report["total_months"] == 12


def test_v5_report_empty_summaries_no_crash() -> None:
    report = build_synth_v2_report([])
    assert report["synth_version"] == "2.0.0"
    assert report["total_months"] == 0
    assert report["total_transaction_count"] == 0
