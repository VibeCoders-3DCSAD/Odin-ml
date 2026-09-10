from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_registry
from app.models.registry import ModelRegistry
from app.schemas.common import ApiMetadata
from app.schemas.pfp import (
    PFPClassifyRequest,
    PFPClassifyResponse,
)
from app.services import pfp_service

router = APIRouter(prefix="/api/v1/pfp", tags=["pfp"])


def _run(registry: ModelRegistry, request: PFPClassifyRequest) -> PFPClassifyResponse:
    if request.classification_mode.value == "STANDARD" and registry.pfp is None:
        raise HTTPException(status_code=503, detail="pfp model not loaded; pending training")
    start = time.perf_counter()
    try:
        classification = pfp_service.classify(registry.pfp, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="classification failed") from exc

    model_version = (
        registry.pfp.model_id
        if request.classification_mode.value == "STANDARD" and registry.pfp is not None
        else "questionnaire_rule"
    )
    return PFPClassifyResponse(
        response_id=str(uuid.uuid4()),
        request_id=request.user_id,
        user_id=request.user_id,
        classification=classification,
        metadata=ApiMetadata(
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            model_version=model_version,
            strategy_used=request.classification_mode.value,
        ),
    )


class BatchRequest(BaseModel):
    requests: list[PFPClassifyRequest] = Field(min_length=1)


class BatchResponse(BaseModel):
    results: list[PFPClassifyResponse]


@router.post("/classify", response_model=PFPClassifyResponse)
async def classify(
    request: PFPClassifyRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> PFPClassifyResponse:
    return _run(registry, request)


@router.post("/classify/batch", response_model=BatchResponse)
async def classify_batch(
    batch: BatchRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchResponse:
    return BatchResponse(results=[_run(registry, req) for req in batch.requests])
