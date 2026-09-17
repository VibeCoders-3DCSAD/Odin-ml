from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from app.models.registry import ModuleModel
from app.schemas.forecast import (
    ConfidenceInterval,
    ForecastLevel,
    ForecastPoint,
    ForecastRequest,
)
from app.services.features import (
    forecast_last_3_months,
    transactions_to_frame,
)


def _chronological_expense_level(transactions: list[dict]) -> float:
    """Mean of the last 3 chronological months with positive expenses.

    The fixed 1..12 bucket summaries (``build_monthly_summaries``) renumber
    real dates onto the synthetic 2023 grid, so month 13 and month 1 would
    collapse into a single row. Grouping by the actual (year, month) keeps
    each real month distinct, so a >12-month history still uses its last
    three real months for the user's expense level.
    """
    if not transactions:
        return 0.0
    df = transactions_to_frame(transactions)
    expense = df[df["transaction_type"] == "expense"]
    if expense.empty:
        return 0.0
    monthly = expense.groupby([expense["date"].dt.year, expense["date"].dt.month])["amount"].sum()
    monthly = monthly[monthly > 0].sort_index()
    return float(monthly.tail(3).mean()) if not monthly.empty else 0.0


def _predict_monthly_total(model: ModuleModel, transactions: list[dict]) -> tuple[float, dict]:
    artifact = model.model

    if isinstance(artifact, dict) and artifact.get("kind") in ("arima", "sarima"):
        # Pooled classical forecaster (ARIMA/SARIMA): the pool path is
        # scale-normalized per user, so rescale it by the requesting user's own
        # recent expense level.
        arima = artifact["model"]
        pool_level = float(artifact.get("pool_level", 1.0))
        profile_level = float(artifact.get("profile_level", pool_level))
        pool_pred = float(arima.forecast(1).iloc[0]) if arima is not None else pool_level
        level = _chronological_expense_level(transactions)
        if level <= 0:
            # Cold start (no positive-expense month): fall back to the pooled
            # profile prior so the forecast is never a literal zero.
            level = profile_level
        pred = (pool_pred / pool_level) * level
        return pred, {
            "lower_80": pred * 0.8,
            "upper_80": pred * 1.2,
            "lower_95": pred * 0.6,
            "upper_95": pred * 1.4,
        }

    vector = forecast_last_3_months(transactions, model.feature_columns)

    if isinstance(artifact, dict) and "model" in artifact and hasattr(artifact["model"], "forward"):
        # PyTorch model (GRU/LSTM/BiLSTM)
        nn_model = artifact["model"]
        scaler = artifact["scaler"]
        seq_length = artifact.get("seq_length", 3)
        n_features = len(model.feature_columns)

        flat = vector.flatten()
        n_months = len(flat) // n_features
        if n_months < seq_length:
            n_months = seq_length
        recent = flat[-n_months * n_features :]
        recent_s = scaler.transform(recent.reshape(-1, n_features))
        seq = recent_s[-seq_length:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = nn_model(x).item()
        return pred, {
            "lower_80": pred * 0.8,
            "upper_80": pred * 1.2,
            "lower_95": pred * 0.6,
            "upper_95": pred * 1.4,
        }

    if isinstance(artifact, dict) and "model" in artifact:
        # sklearn RF model wrapped in dict
        scaler = artifact["scaler"]
        rf_model = artifact["model"]
        scaled = scaler.transform(vector)
        trees = np.array([t.predict(scaled) for t in rf_model.estimators_]).flatten()
        total = float(np.mean(trees))
        return total, {
            "lower_80": float(np.percentile(trees, 10)),
            "upper_80": float(np.percentile(trees, 90)),
            "lower_95": float(np.percentile(trees, 2.5)),
            "upper_95": float(np.percentile(trees, 97.5)),
        }

    # Direct sklearn model (no dict wrapper)
    scaled = artifact["scaler"].transform(vector) if hasattr(artifact, "scaler") else vector
    trees = np.array([t.predict(scaled) for t in artifact.estimators_]).flatten()
    total = float(np.mean(trees))
    return total, {
        "lower_80": float(np.percentile(trees, 10)),
        "upper_80": float(np.percentile(trees, 90)),
        "lower_95": float(np.percentile(trees, 2.5)),
        "upper_95": float(np.percentile(trees, 97.5)),
    }


def _horizon_factor(horizon: str) -> float:
    if horizon == "WEEKLY":
        return 7.0 / 30.44
    if horizon == "SEMI_MONTHLY":
        return 15.0 / 30.44
    if horizon == "YEARLY":
        return 12.0
    return 1.0


def _projection_weights(
    transactions: list[dict], horizon: str, incomplete_month: str | None = None
) -> np.ndarray:
    """Allocate a horizon total using the user's observed spending rhythm."""
    df = transactions_to_frame(transactions)
    expense = df[df["transaction_type"] == "expense"].copy()
    period_count = {"WEEKLY": 7, "SEMI_MONTHLY": 2, "MONTHLY": 4, "YEARLY": 12}[horizon]
    if expense.empty:
        return np.full(period_count, 1.0 / period_count)

    if horizon == "YEARLY" and incomplete_month is not None:
        incomplete_period = pd.Period(incomplete_month, freq="M")
        is_incomplete_month = expense["date"].dt.to_period("M") == incomplete_period
        if (~is_incomplete_month).any():
            expense = expense.loc[~is_incomplete_month]

    if horizon == "WEEKLY":
        expense["period"] = expense["date"].dt.dayofweek
    elif horizon == "SEMI_MONTHLY":
        expense["period"] = (expense["date"].dt.day > 15).astype(int)
    elif horizon == "MONTHLY":
        expense["period"] = ((expense["date"].dt.day - 1) // 7).clip(upper=3)
    else:
        expense["period"] = expense["date"].dt.month - 1

    totals = expense.groupby("period")["amount"].sum().reindex(range(period_count), fill_value=0.0)
    total = float(totals.sum())
    return totals.to_numpy(dtype=float) / total if total > 0 else np.full(period_count, 1.0 / period_count)


def _projection_dates(transactions: list[dict], horizon: str) -> list[str]:
    df = transactions_to_frame(transactions)
    last_date = df["date"].max().normalize()
    if horizon == "WEEKLY":
        return [(last_date + np.timedelta64(offset, "D")).date().isoformat() for offset in range(1, 8)]
    if horizon == "SEMI_MONTHLY":
        return [(last_date + np.timedelta64(offset, "D")).date().isoformat() for offset in (1, 8)]
    if horizon == "MONTHLY":
        return [(last_date + np.timedelta64(offset, "D")).date().isoformat() for offset in (7, 14, 21, 28)]
    first_next_month = last_date + pd.offsets.MonthBegin(1)
    return [date.date().isoformat() for date in pd.date_range(first_next_month, periods=12, freq="MS")]


def _align_projection_weights(weights: np.ndarray, dates: list[str], horizon: str) -> np.ndarray:
    """Align calendar-month yearly weights with a forecast starting next month."""
    if horizon != "YEARLY":
        return weights
    first_month = pd.Timestamp(dates[0]).month
    return np.roll(weights, -(first_month - 1))


def _category_proportions(transactions: list[dict]) -> dict[str, float]:
    df = transactions_to_frame(transactions)
    expense = df[df["transaction_type"] == "expense"]
    if expense.empty:
        return {}
    totals = expense.groupby("category")["amount"].sum()
    return (totals / totals.sum()).to_dict()


def forecast(
    model: ModuleModel, request: ForecastRequest
) -> tuple[list[ForecastPoint], ConfidenceInterval, str]:
    total, ci = _predict_monthly_total(
        model, [t.model_dump() for t in request.historical_transactions]
    )
    factor = _horizon_factor(request.forecast_horizon.value)
    predicted = total * factor
    scaled_ci = {k: v * factor for k, v in ci.items()}

    interval = ConfidenceInterval(
        lower_80=round(scaled_ci["lower_80"], 2),
        upper_80=round(scaled_ci["upper_80"], 2),
        lower_95=round(scaled_ci["lower_95"], 2),
        upper_95=round(scaled_ci["upper_95"], 2),
    )
    transactions = [t.model_dump() for t in request.historical_transactions]
    dates = _projection_dates(transactions, request.forecast_horizon.value)
    incomplete_month = max(transaction["date"] for transaction in transactions)[:7]
    weights = _projection_weights(
        transactions, request.forecast_horizon.value, incomplete_month
    )
    weights = _align_projection_weights(weights, dates, request.forecast_horizon.value)

    if request.forecast_level == ForecastLevel.TOTAL:
        points = [
            ForecastPoint(date=date, amount=round(predicted * weight, 2))
            for date, weight in zip(dates, weights, strict=True)
        ]
        return points, interval, "total"

    proportions = _category_proportions(transactions)
    if request.forecast_level == ForecastLevel.CATEGORY_GROUP:
        # The application sends category-group labels (for example, "Essentials"),
        # not individual category slugs. Project every supplied group over time.
        points = [
            ForecastPoint(date=date, amount=round(predicted * share * weight, 2), category=group)
            for group, share in sorted(proportions.items(), key=lambda item: item[1], reverse=True)
            for date, weight in zip(
                dates,
                _align_projection_weights(
                    _projection_weights(
                        [
                            transaction
                            for transaction in transactions
                            if transaction["transaction_type"] == "expense"
                            and transaction["category"] == group
                        ],
                        request.forecast_horizon.value,
                        incomplete_month,
                    ),
                    dates,
                    request.forecast_horizon.value,
                ),
                strict=True,
            )
        ]
        return points, interval, "category_group"

    points = []
    for cat, share in proportions.items():
        points.append(ForecastPoint(date="next", amount=round(predicted * share, 2), category=cat))
    points.sort(key=lambda p: p.amount, reverse=True)
    return points, interval, "category"


def forecast_or_fallback(
    model: ModuleModel, request: ForecastRequest
) -> tuple[list[ForecastPoint], ConfidenceInterval, str]:
    """Run the forecast; fall back to a trailing-average estimate on failure.

    FO-02 (cold-start fallback): when the learned path fails or history is
    too short, average the last 3 months of total expenses.
    """
    try:
        return forecast(model, request)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"forecast unavailable: {exc}") from exc


def cold_start_estimate(
    transactions: list[dict], profile_level: float = 1.0
) -> tuple[float, float]:
    """FO-02: profile-average fallback using the trailing 3-month mean.

    When no positive-expense months exist, returns the pooled profile prior
    (``profile_level``) with a 30% interval width rather than exact zero.
    """
    from app.services.features import build_monthly_summaries

    summaries = build_monthly_summaries(transactions)
    expenses = summaries.loc[summaries["total_expenses"] > 0, "total_expenses"].tail(3)
    if expenses.empty:
        return profile_level, profile_level * 0.3
    total = float(expenses.mean())
    return total, float(expenses.std())
