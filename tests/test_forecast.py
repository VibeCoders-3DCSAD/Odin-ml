from __future__ import annotations

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
    assert body["model_version"] == "forecaster-tier3_sarima"


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
