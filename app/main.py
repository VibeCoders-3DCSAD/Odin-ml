from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import anomaly, budget, forecast, health, pfp
from app.core.config import SERVICE_VERSION
from app.models.registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    registry.load_all()
    app.state.registry = registry
    yield
    app.state.registry = None


app = FastAPI(
    title="Odin ML Service",
    version=SERVICE_VERSION,
    description=(
        "Machine learning microservice for PFP classification, expense "
        "forecasting, anomaly detection, and budget optimization."
    ),
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(pfp.router)
app.include_router(forecast.router)
app.include_router(anomaly.router)
app.include_router(budget.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Odin ML service is running.", "version": SERVICE_VERSION}
