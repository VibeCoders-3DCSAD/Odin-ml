from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_registry
from app.models.registry import ModelRegistry
from app.schemas.budget import BudgetRequest, BudgetResponse
from app.schemas.common import ApiMetadata, ModuleStatus
from app.services import budget_service

router = APIRouter(prefix="/api/v1/budget", tags=["budget"])


def _run(request: BudgetRequest, registry: ModelRegistry) -> BudgetResponse:
    start = time.perf_counter()
    try:
        recommendation, explanations = budget_service.optimize(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model_version = registry.budget.model_id if registry.budget is not None else "n/a"
    return BudgetResponse(
        response_id=str(uuid.uuid4()),
        request_id=request.request_id,
        user_id=request.user_id,
        recommendation=recommendation,
        explanations=explanations if request.include_reasoning else None,
        status=ModuleStatus.SUCCESS,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version=model_version,
            strategy_used="tier2_lp",
        ),
    )


class BatchRequest(BaseModel):
    requests: list[BudgetRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    results: list[BudgetResponse]


@router.post("/recommend", response_model=BudgetResponse)
async def recommend(
    request: BudgetRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BudgetResponse:
    return _run(request, registry)


@router.post("/recommend/batch", response_model=BatchResponse)
async def recommend_batch(
    batch: BatchRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchResponse:
    return BatchResponse(results=[_run(req, registry) for req in batch.requests])
