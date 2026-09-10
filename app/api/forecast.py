from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_registry
from app.models.registry import ModelRegistry
from app.schemas.common import ApiMetadata, ModuleStatus
from app.schemas.forecast import (
    ConfidenceInterval,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)
from app.services import forecast_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])


def _run(registry: ModelRegistry, request: ForecastRequest) -> ForecastResponse:
    start = time.perf_counter()
    transactions = [t.model_dump() for t in request.historical_transactions]
    status = ModuleStatus.SUCCESS
    points = interval = level = None
    model = registry.forecaster
    profile_level = 1.0
    if model is not None:
        if isinstance(model.model, dict):
            raw = model.model.get("profile_level") or model.model.get("pool_level") or 1.0
            profile_level = float(raw)
        try:
            points, interval, level = forecast_service.forecast(model, request)
        except Exception as exc:
            logger.warning("learned forecast failed for user %s: %s", request.user_id, exc)
            points = interval = level = None

    if points is None:
        total, std = forecast_service.cold_start_estimate(transactions, profile_level=profile_level)
        points = [ForecastPoint(date="next", amount=round(total, 2))]
        interval = ConfidenceInterval(
            lower_80=round(total - std, 2),
            upper_80=round(total + std, 2),
            lower_95=round(total - 1.96 * std, 2),
            upper_95=round(total + 1.96 * std, 2),
        )
        level = "cold_start_profiled"
        status = ModuleStatus.FALLBACK

    assert interval is not None and level is not None

    model_version = (
        registry.forecaster.model_id
        if registry.forecaster is not None and status == ModuleStatus.SUCCESS
        else "cold_start_fallback"
    )
    return ForecastResponse(
        response_id=str(uuid.uuid4()),
        request_id=request.user_id,
        user_id=request.user_id,
        forecasts=points,
        forecast_level=request.forecast_level,
        forecast_horizon=request.forecast_horizon,
        confidence_intervals=interval,
        model_version=model_version,
        status=status,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version=model_version,
            strategy_used=level,
        ),
    )


class BatchRequest(BaseModel):
    requests: list[ForecastRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    results: list[ForecastResponse]


@router.post("/predict", response_model=ForecastResponse)
async def predict(
    request: ForecastRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ForecastResponse:
    return _run(registry, request)


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    batch: BatchRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchResponse:
    return BatchResponse(results=[_run(registry, req) for req in batch.requests])
