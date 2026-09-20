from __future__ import annotations

import pandas as pd
from app.schemas.forecast import ForecastHorizon, ForecastLevel, ForecastRequest
from app.services import forecast_service

from tests.conftest import load_transactions


def test_forecast_predict(client):
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "historical_transactions": txns,
               "forecast_horizon": "MONTHLY", "forecast_level": "TOTAL"}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("SUCCESS", "FALLBACK")
    assert len(body["forecasts"]) == 4
    assert body["forecasts"][0]["amount"] >= 0
    assert body["confidence_intervals"]["lower_95"] <= body["confidence_intervals"]["upper_95"]


def test_forecast_predict_arima_success(client):
    """ARIMA (current winner) must serve a real prediction: SUCCESS, not fallback."""
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "historical_transactions": txns,
               "forecast_horizon": "MONTHLY", "forecast_level": "TOTAL"}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "SUCCESS"
    assert body["forecasts"][0]["amount"] > 0
    assert body["model_version"] == "forecaster-v2-tier3_sarima"


class _StubPooled:
    def forecast(self, steps):
        return _StubSeries()


class _StubSeries:
    @property
    def iloc(self):
        # Mirror pandas: result.iloc[0] indexes the first forecast point.
        return self

    def __getitem__(self, _):
        return 5.0


def test_forecast_sarima_branch_served(client):
    """SARIMA-kind artifacts resolve through the pooled forecast branch."""
    from app.models.registry import ModuleModel
    from app.services.forecast_service import _predict_monthly_total

    model = ModuleModel(
        module="forecaster",
        model={"kind": "sarima", "model": _StubPooled(), "pool_level": 1.0},
        evaluation={},
        feature_columns=[],
    )
    txns = [
        {"date": "2023-01-05", "amount": 100.0, "category": "food", "transaction_type": "debit"},
        {"date": "2023-01-20", "amount": 200.0, "category": "transport", "transaction_type": "debit"},
    ]
    pred, intervals = _predict_monthly_total(model, txns)

    import numpy as np

    assert np.isfinite(pred)
    assert pred > 0
    assert intervals["lower_95"] <= intervals["upper_95"]


def test_forecast_rejects_short_history(client):
    txns = load_transactions(n=1)
    payload = {"user_id": "test-user-2", "historical_transactions": txns,
               "forecast_horizon": "MONTHLY", "forecast_level": "TOTAL"}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    # Falls back (FALLBACK) or serves; never a 5xx, and the total is non-zero.
    assert resp.status_code in (200, 422)
    if resp.status_code == 200:
        body = resp.json()
        assert sum(p["amount"] for p in body["forecasts"]) > 0


def test_forecast_sparse_user_nonzero(client):
    """A real user with very few transactions must not get a zero total forecast."""
    txns = load_transactions(n=3)
    payload = {"user_id": "test-user-sparse", "historical_transactions": txns,
               "forecast_horizon": "MONTHLY", "forecast_level": "TOTAL"}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] in ("SUCCESS", "FALLBACK")
    assert sum(p["amount"] for p in body["forecasts"]) > 0


def test_cold_start_estimate_uses_profile_prior_when_no_expenses():
    """With no positive-expense months, the fallback returns the profile prior, not 0."""
    from app.services.forecast_service import cold_start_estimate

    transactions = [
        {"date": "2026-09-01", "amount": 500.0, "category": "income",
         "transaction_type": "income"},
    ]
    total, std = cold_start_estimate(transactions, profile_level=32559.8)
    assert total == 32559.8
    assert std == 32559.8 * 0.3
    assert total > 0


def test_forecast_rejects_empty_transactions(client):
    payload = {"user_id": "test-user-3", "historical_transactions": []}
    resp = client.post("/api/v1/forecast/predict", json=payload)
    assert resp.status_code == 422


def test_category_group_forecast_uses_group_labels_from_transactions(monkeypatch):
    monkeypatch.setattr(
        forecast_service,
        "_predict_monthly_total",
        lambda _model, _transactions: (1_000.0, {"lower_80": 800.0, "upper_80": 1_200.0, "lower_95": 600.0, "upper_95": 1_400.0}),
    )
    request = ForecastRequest(
        user_id="test-user-4",
        historical_transactions=[
            {"date": "2026-09-01", "amount": 600.0, "category": "Essentials", "transaction_type": "expense"},
            {"date": "2026-09-02", "amount": 400.0, "category": "Discretionary", "transaction_type": "expense"},
        ],
        forecast_level=ForecastLevel.CATEGORY_GROUP,
    )

    points, _, level = forecast_service.forecast(object(), request)

    assert level == "category_group"
    assert len(points) == 8
    assert sum(point.amount for point in points if point.category == "Essentials") == 600.0
    assert sum(point.amount for point in points if point.category == "Discretionary") == 400.0


def test_total_forecast_returns_four_weekly_points_for_a_month(monkeypatch):
    monkeypatch.setattr(
        forecast_service,
        "_predict_monthly_total",
        lambda _model, _transactions: (1_000.0, {"lower_80": 800.0, "upper_80": 1_200.0, "lower_95": 600.0, "upper_95": 1_400.0}),
    )
    request = ForecastRequest(
        user_id="test-user-5",
        historical_transactions=[
            {"date": "2026-09-01", "amount": 100.0, "category": "Essentials", "transaction_type": "expense"},
            {"date": "2026-09-08", "amount": 200.0, "category": "Essentials", "transaction_type": "expense"},
            {"date": "2026-09-15", "amount": 300.0, "category": "Essentials", "transaction_type": "expense"},
            {"date": "2026-09-22", "amount": 400.0, "category": "Essentials", "transaction_type": "expense"},
        ],
        forecast_horizon=ForecastHorizon.MONTHLY,
    )

    points, _, level = forecast_service.forecast(object(), request)

    assert level == "total"
    assert [point.amount for point in points] == [100.0, 200.0, 300.0, 400.0]
    assert [point.date for point in points] == ["2026-09-29", "2026-10-06", "2026-10-13", "2026-10-20"]


def test_total_forecast_returns_twelve_monthly_points_for_a_year(monkeypatch):
    monkeypatch.setattr(
        forecast_service,
        "_predict_monthly_total",
        lambda _model, _transactions: (1_000.0, {"lower_80": 800.0, "upper_80": 1_200.0, "lower_95": 600.0, "upper_95": 1_400.0}),
    )
    request = ForecastRequest(
        user_id="test-user-6",
        historical_transactions=[
            {"date": "2026-09-01", "amount": 100.0, "category": "Essentials", "transaction_type": "expense"}
        ],
        forecast_horizon=ForecastHorizon.YEARLY,
    )

    points, _, _ = forecast_service.forecast(object(), request)

    assert len(points) == 12
    assert sum(point.amount for point in points) == 12_000.0
    assert points[0].date == "2026-10-01"
    assert points[-1].date == "2027-09-01"


def test_yearly_category_group_weights_align_with_forecast_months(monkeypatch):
    monkeypatch.setattr(
        forecast_service,
        "_predict_monthly_total",
        lambda _model, _transactions: (1_000.0, {"lower_80": 800.0, "upper_80": 1_200.0, "lower_95": 600.0, "upper_95": 1_400.0}),
    )
    request = ForecastRequest(
        user_id="test-user-7",
        historical_transactions=[
            {"date": "2026-07-02", "amount": 100.0, "category": "Obligatory", "transaction_type": "expense"},
            {"date": "2026-09-01", "amount": 1_000.0, "category": "income", "transaction_type": "income"},
        ],
        forecast_horizon=ForecastHorizon.YEARLY,
        forecast_level=ForecastLevel.CATEGORY_GROUP,
    )

    points, _, _ = forecast_service.forecast(object(), request)

    amounts_by_date = {point.date: point.amount for point in points}
    assert amounts_by_date["2027-07-01"] == 12_000.0
    assert amounts_by_date["2027-04-01"] == 0.0


def test_yearly_forecast_excludes_the_in_progress_month_from_seasonal_weights(monkeypatch):
    monkeypatch.setattr(
        forecast_service,
        "_predict_monthly_total",
        lambda _model, _transactions: (1_000.0, {"lower_80": 800.0, "upper_80": 1_200.0, "lower_95": 600.0, "upper_95": 1_400.0}),
    )
    request = ForecastRequest(
        user_id="test-user-8",
        historical_transactions=[
            {"date": "2025-09-02", "amount": 100.0, "category": "Essentials", "transaction_type": "expense"},
            {"date": "2026-08-02", "amount": 100.0, "category": "Essentials", "transaction_type": "expense"},
            {"date": "2026-09-15", "amount": 100.0, "category": "Essentials", "transaction_type": "expense"},
        ],
        forecast_horizon=ForecastHorizon.YEARLY,
        forecast_level=ForecastLevel.CATEGORY_GROUP,
    )

    points, _, _ = forecast_service.forecast(object(), request)

    amounts_by_date = {point.date: point.amount for point in points}
    assert amounts_by_date["2027-09-01"] == 6_000.0


def test_category_group_forecast_returns_weekly_points_for_a_month(monkeypatch):
    monkeypatch.setattr(
        forecast_service,
        "_predict_monthly_total",
        lambda _model, _transactions: (1_000.0, {"lower_80": 800.0, "upper_80": 1_200.0, "lower_95": 600.0, "upper_95": 1_400.0}),
    )
    request = ForecastRequest(
        user_id="test-user-7",
        historical_transactions=[
            {"date": "2026-09-01", "amount": 60.0, "category": "Essentials", "transaction_type": "expense"},
            {"date": "2026-09-08", "amount": 40.0, "category": "Discretionary", "transaction_type": "expense"},
        ],
        forecast_level=ForecastLevel.CATEGORY_GROUP,
    )

    points, _, level = forecast_service.forecast(object(), request)

    assert level == "category_group"
    assert len(points) == 8
    assert [(point.category, point.amount) for point in points[:4]] == [
        ("Essentials", 600.0),
        ("Essentials", 0.0),
        ("Essentials", 0.0),
        ("Essentials", 0.0),
    ]
    assert [(point.category, point.amount) for point in points[4:]] == [
        ("Discretionary", 0.0),
        ("Discretionary", 400.0),
        ("Discretionary", 0.0),
        ("Discretionary", 0.0),
    ]


def _pool_hist(users: dict, n_years: int = 1) -> pd.DataFrame:
    """Synthetic monthly history with an absolute month index (1..12n)."""
    rows = []
    for uid, scale in users.items():
        for y in range(n_years):
            for m in range(1, 13):
                rows.append({
                    "user_id": uid,
                    "target_expenses": scale * (1.0 + 0.05 * (m % 3)),
                    "month_num": y * 12 + m,
                    "month": m,
                })
    return pd.DataFrame(rows)


def test_chronological_expense_level_keeps_years_distinct():
    """A >12-month history must use its last three REAL months, not buckets 1..12."""
    from app.services.forecast_service import _chronological_expense_level

    txns = [
        {"date": f"{y}-{m:02d}-10", "amount": 100.0, "category": "food",
         "transaction_type": "expense"}
        for y in (2023, 2024)
        for m in range(1, 13)
    ]
    # Last three real months: 2024-10, 2024-11, 2024-12 -> 100.0 each.
    assert _chronological_expense_level(txns) == 100.0

    no_expenses = [
        {"date": "2023-01-10", "amount": 500.0, "category": "salary",
         "transaction_type": "income"}
    ]
    assert _chronological_expense_level(no_expenses) == 0.0
    assert _chronological_expense_level([]) == 0.0


def test_chronological_expense_level_crossing_year_boundary():
    from app.services.forecast_service import _chronological_expense_level

    amounts = {("2023", 11): 100.0, ("2024", 1): 200.0, ("2024", 2): 300.0, ("2024", 3): 400.0}
    txns = [
        {"date": f"{y}-{int(m):02d}-10", "amount": amt, "category": "food",
         "transaction_type": "expense"}
        for (y, m), amt in amounts.items()
    ]
    # tail(3) chronological = 200, 300, 400 -> 300 (Jan-2024 is NOT bucket 1 of Dec-2023).
    assert _chronological_expense_level(txns) == 300.0


def test_pooled_abs_span_counts_real_months_not_rows():
    from training.scripts.train_forecaster import _pooled_abs_span

    hist = _pool_hist({"u1": 1000.0}, n_years=2)
    norm = hist["target_expenses"] / hist["target_expenses"].mean()
    pooled = norm.groupby(hist["month_num"]).mean().sort_index()
    assert _pooled_abs_span(pooled) == 24
    assert len(pooled) == 24  # Jan-2024 (13) and Jan-2025 (1) stay separate rows.


def test_forecast_sarima_pool_degrades_to_arima_on_single_year(monkeypatch):
    import numpy as np
    from training.scripts import train_forecaster as tf

    if not tf.HAS_STATSMODELS:
        import pytest

        pytest.skip("statsmodels unavailable")

    hist = _pool_hist({"u1": 1000.0, "u2": 400.0}, n_years=1)

    import statsmodels.tsa.statespace.sarimax as _sarimax_mod

    sarimax_calls = []
    real_sarimax = _sarimax_mod.SARIMAX

    def _wrapped_sarimax(*args, **kwargs):
        sarimax_calls.append(args)
        return real_sarimax(*args, **kwargs)

    arima_calls = []
    real_arima = tf.ARIMA

    def _wrapped_arima(*args, **kwargs):
        arima_calls.append(args)
        return real_arima(*args, **kwargs)

    monkeypatch.setattr(_sarimax_mod, "SARIMAX", _wrapped_sarimax)
    monkeypatch.setattr(tf, "ARIMA", _wrapped_arima)

    path, n_fits = tf.forecast_sarima_pool(hist, [13, 14])

    assert n_fits == 1
    assert not sarimax_calls, "a 12-month pool must NOT engage the seasonal branch"
    assert arima_calls, "a 12-month pool must fall back to plain ARIMA"
    assert np.isfinite(path[13]) and np.isfinite(path[14])


def test_forecast_sarima_pool_engages_on_two_years(monkeypatch):
    import numpy as np
    from training.scripts import train_forecaster as tf

    if not tf.HAS_STATSMODELS:
        import pytest

        pytest.skip("statsmodels unavailable")

    hist = _pool_hist({"u1": 1000.0, "u2": 400.0}, n_years=2)

    import statsmodels.tsa.statespace.sarimax as _sarimax_mod

    sarimax_calls = []
    real_sarimax = _sarimax_mod.SARIMAX

    def _wrapped_sarimax(*args, **kwargs):
        sarimax_calls.append(args)
        return real_sarimax(*args, **kwargs)

    monkeypatch.setattr(_sarimax_mod, "SARIMAX", _wrapped_sarimax)

    path, n_fits = tf.forecast_sarima_pool(hist, [25])

    assert n_fits == 1
    assert sarimax_calls, "a >= 24-absolute-month pool must use the seasonal branch"
    assert np.isfinite(path[25])


def test_forecast_arima_pool_consistent_with_month_of_year_on_single_year(monkeypatch):
    """Metamorphic regression: on 2023-only data, month_num pooling equals the
    old month-of-year pooling exactly (both feed plain ARIMA)."""
    import numpy as np
    from training.scripts import train_forecaster as tf

    if not tf.HAS_STATSMODELS:
        import pytest

        pytest.skip("statsmodels unavailable")

    hist = _pool_hist({"u1": 1000.0, "u2": 400.0, "u3": 250.0}, n_years=1)

    new_path, _ = tf.forecast_arima_pool(hist, [13, 14, 15])

    # Old behavior: group by month-of-year (1..12) on the same rows.
    h = hist[hist["target_expenses"] > 0].copy()
    user_mean = h.groupby("user_id")["target_expenses"].transform("mean")
    h["norm"] = h["target_expenses"] / user_mean
    pooled = h.groupby("month")["norm"].mean().sort_index().reset_index(drop=True)
    model = tf.ARIMA(pooled, order=tf.ARIMA_ORDER).fit(method="burg")
    old_forecast = np.asarray(model.forecast(3), dtype=float)

    assert np.allclose(list(new_path.values()), old_forecast, atol=1e-6)
