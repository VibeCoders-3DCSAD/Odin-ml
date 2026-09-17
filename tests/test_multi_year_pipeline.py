"""Regression coverage for multi-year synthetic time-series handling."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "training" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import train_forecaster  # noqa: E402
from feature_engineering import compute_derived_features  # noqa: E402
from feature_engineering_forecaster import process_persona  # noqa: E402
from generate_transactions import generate_persona_transactions  # noqa: E402
from preprocessor import generate_temporal_folds, load_anomaly_info  # noqa: E402


def _persona() -> dict:
    return {
        "persona_id": "persona-1",
        "archetype_id": "archetype-1",
        "archetype_name": "Test",
        "monthly_income": 20000.0,
        "income_cv": 0.1,
        "obligation_ratio": 0.5,
        "savings_rate": 0.2,
        "runway_months": 4.0,
        "household_size": 2,
        "income_pattern": "regular",
        "food_expense": 3000.0,
        "housing_expense": 5000.0,
        "transport_expense": 1000.0,
        "health_expense": 500.0,
        "education_expense": 0.0,
        "other_expense": 500.0,
    }


def _periods() -> list[str]:
    return pd.period_range("2023-01", periods=24, freq="M").astype(str).tolist()


def test_generator_preserves_24_distinct_months_and_transaction_ids() -> None:
    transactions, summaries = generate_persona_transactions(_persona(), num_months=24, seed=7)

    assert [summary.year_month for summary in summaries] == _periods()
    transaction_ids = [transaction.transaction_id for transaction in transactions]
    assert len(transaction_ids) == len(set(transaction_ids))
    assert all(
        f"_{transaction.year}_" in transaction.transaction_id for transaction in transactions
    )


def test_anomaly_info_and_derived_features_do_not_cross_match_years(tmp_path: Path) -> None:
    transactions = pd.DataFrame(
        [
            {
                "persona_id": "persona-1",
                "year_month": "2023-01",
                "month": 1,
                "is_anomalous": True,
                "anomaly_type": "amount_spike",
            },
            {
                "persona_id": "persona-1",
                "year_month": "2024-01",
                "month": 1,
                "is_anomalous": False,
                "anomaly_type": None,
            },
        ]
    )
    transactions.to_parquet(tmp_path / "transactions.parquet", index=False)
    anomaly_info = load_anomaly_info(str(tmp_path))

    assert anomaly_info is not None
    assert anomaly_info["year_month"].tolist() == ["2023-01", "2024-01"]

    summaries = pd.DataFrame(
        [
            {
                "persona_id": "persona-1",
                "year_month": period,
                "month": int(period[-2:]),
                "total_income": 100.0,
                "total_expenses": 50.0,
            }
            for period in ["2023-01", "2024-01"]
        ]
    )
    rows = compute_derived_features("persona-1", {"pfp_label": "Stable"}, summaries, anomaly_info)
    assert [(row["year_month"], row["is_anomalous"]) for row in rows] == [
        ("2023-01", True),
        ("2024-01", False),
    ]


def test_temporal_folds_use_ordered_year_month_periods() -> None:
    folds = generate_temporal_folds(_periods(), min_train_months=6, embargo_months=1)

    assert folds[0] == {
        "fold": 1,
        "train_periods": _periods()[:6],
        "embargo_periods": ["2023-07"],
        "test_periods": ["2023-08"],
    }
    assert any("2024-01" in fold["train_periods"] for fold in folds)
    assert len({period for fold in folds for period in fold["train_periods"]}) == 22


def test_forecaster_grid_and_pool_keep_all_24_periods(monkeypatch) -> None:
    periods = _periods()
    summaries = pd.DataFrame(
        {
            "persona_id": "persona-1",
            "year_month": periods,
            "month": [int(period[-2:]) for period in periods],
            "total_expenses": np.arange(1, 25, dtype=float),
        }
    )
    transactions = pd.DataFrame(
        {
            "transaction_type": ["expense"],
            "date": [pd.Timestamp("2024-02-29")],
            "amount": [10.0],
            "transaction_id": ["txn_persona-1_2024_02_food_0"],
        }
    )
    grid = process_persona("persona-1", transactions, summaries)

    assert grid["date"].min() == pd.Timestamp("2023-01-01")
    assert grid["date"].max() == pd.Timestamp("2024-12-31")
    assert pd.Timestamp("2024-02-29") in set(grid["date"])
    assert grid.loc[grid["year_month"] == "2023-12", "target_expenses"].iloc[0] == 13.0

    daily = pd.DataFrame(
        {
            "user_id": "persona-1",
            "year_month": np.repeat(periods, 2),
            "target_expenses": np.repeat(np.arange(1, 25, dtype=float), 2),
            "has_target": 1.0,
            "has_transaction": 0.0,
        }
    )
    monthly = train_forecaster.aggregate_to_monthly(daily, [])
    captured = []

    class FakeArima:
        def __init__(self, series, order):
            captured.append(series)

        def fit(self, method):
            return self

        def forecast(self, steps):
            return np.ones(steps)

    monkeypatch.setattr(train_forecaster, "HAS_STATSMODELS", True)
    monkeypatch.setattr(train_forecaster, "ARIMA", FakeArima)
    train_forecaster.forecast_arima_pool(monthly, ["2025-01"])

    assert len(monthly) == 24
    assert len(captured[0]) == 24
