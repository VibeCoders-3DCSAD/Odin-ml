from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from app.models.artifact_classes import (
    _SequenceForecaster,
)
from app.models.loader import ModelLoader

PFP_MODULE = "pfp"
FORECASTER_MODULE = "forecaster"
ANOMALY_MODULE = "anomaly"
BUDGET_MODULE = "budget"
ANOMALY_ARTIFACT = "anomaly_detector.joblib"


def _resolve_pfp_artifact(evaluation: dict) -> str:
    """Resolve the pfp winner artifact filename from evaluation.json."""
    artifact = evaluation.get("winner_artifact")
    if artifact and isinstance(artifact, str) and artifact.endswith(".joblib"):
        return artifact
    winner = evaluation.get("winner")
    if winner:
        return f"{winner}.joblib"
    return "tier1_rule_based.joblib"


logger = logging.getLogger(__name__)


@dataclass
class ModuleModel:
    module: str
    model: Any
    evaluation: dict
    feature_columns: list[str]
    threshold: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        """Serve-time model id from the winner contract.

        Prefers the curated `metadata.json` `model_id` (e.g. ``pfp-tier3_svm``);
        falls back to a derived ``<module>-<winner>`` so stubs and families
        without metadata still report a truthful id.
        """
        mid = (self.metadata or {}).get("model_id")
        if isinstance(mid, str) and mid:
            return mid
        winner = self.evaluation.get("winner")
        if isinstance(winner, str) and winner:
            return f"{self.module}-{winner}"
        return self.module


def _resolve_forecaster_artifact(evaluation: dict, output_dir) -> tuple[str, Any]:
    """Load the forecaster winner artifact selected by evaluation.json.

    `winner_artifact` names the primary serving artifact; PyTorch winners keep
    the `.pth` + `_meta.joblib` pair and are rebuilt into a `_SequenceForecaster`.
    """
    winner = evaluation.get("winner", "")
    artifact = evaluation.get("winner_artifact") or (f"{winner}.joblib" if winner else "")
    if not artifact:
        raise FileNotFoundError(f"no forecaster artifact for winner '{winner}'")

    pth_name = artifact if artifact.endswith(".pth") else None
    pth_path = (
        output_dir / pth_name
        if pth_name
        else (output_dir / f"{winner}.pth" if winner.startswith("tier3_") else None)
    )
    if pth_path is not None and pth_path.exists():
        meta_path = pth_path.with_name(f"{pth_path.stem}_meta.joblib")
        if meta_path.exists():
            import joblib

            meta = joblib.load(str(meta_path))
            state = torch.load(str(pth_path), map_location="cpu", weights_only=True)
            variant = state.get("model_type", winner.replace("tier3_", ""))
            input_size = state.get("input_size", 20)
            hidden_size = state.get("hidden_size", 32)
            seq_length = state.get("seq_length", 3)
            model = _SequenceForecaster(input_size, hidden_size, variant)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            return f"{winner}.pth", {
                "model": model,
                "scaler": meta["scaler"],
                "feature_cols": meta["feature_cols"],
                "seq_length": seq_length,
            }

    return artifact, ModelLoader().load_joblib(FORECASTER_MODULE, artifact)


class ModelRegistry:
    """Loads and exposes all module artifacts at startup (lifespan)."""

    def __init__(self, loader: ModelLoader | None = None):
        self.loader = loader or ModelLoader()
        self.pfp: ModuleModel | None = None
        self.forecaster: ModuleModel | None = None
        self.anomaly: ModuleModel | None = None
        self.budget: ModuleModel | None = None

    def load_all(self) -> None:
        self.pfp = self._load_optional(PFP_MODULE, _resolve_pfp_artifact(self._pfp_evaluation()))
        try:
            self.forecaster = self._load_forecaster()
        except FileNotFoundError as exc:
            logger.warning("forecaster artifacts not found; skipping: %s", exc)
            self.forecaster = None
        try:
            self.anomaly = self._load_anomaly()
        except FileNotFoundError as exc:
            logger.warning("anomaly artifacts not found; skipping: %s", exc)
            self.anomaly = None
        try:
            self.budget = self._load_budget()
        except FileNotFoundError as exc:
            logger.warning("budget metadata not found; skipping: %s", exc)
            self.budget = None

    def _pfp_evaluation(self) -> dict:
        try:
            return self.loader.load_json(PFP_MODULE, "evaluation.json")
        except FileNotFoundError:
            return {}

    def _metadata(self, module: str) -> dict:
        try:
            meta = self.loader.load_json(module, "metadata.json")
        except FileNotFoundError:
            return {}
        return meta if isinstance(meta, dict) else {}

    def _load_optional(self, module: str, artifact: str) -> ModuleModel | None:
        try:
            return self._load_sklearn(module, artifact)
        except FileNotFoundError as exc:
            logger.warning("%s artifacts not found; skipping: %s", module, exc)
            return None

    def _load_sklearn(self, module: str, artifact: str) -> ModuleModel:
        evaluation = self.loader.load_json(module, "evaluation.json")
        model = self.loader.load_joblib(module, artifact)
        feature_columns = evaluation.get("feature_columns") or []
        if not feature_columns and isinstance(model, dict):
            feature_columns = list(model.get("feature_cols", []))
        return ModuleModel(
            module=module,
            model=model,
            evaluation=evaluation,
            feature_columns=feature_columns,
            metadata=self._metadata(module),
        )

    def _load_forecaster(self) -> ModuleModel:
        evaluation = self.loader.load_json(FORECASTER_MODULE, "evaluation.json")
        feature_columns = evaluation.get("feature_columns", [])
        output_dir = self.loader.resolve(FORECASTER_MODULE)
        artifact_name, model = _resolve_forecaster_artifact(evaluation, output_dir)
        return ModuleModel(
            module=FORECASTER_MODULE,
            model=model,
            evaluation=evaluation,
            feature_columns=feature_columns,
            metadata=self._metadata(FORECASTER_MODULE),
        )

    def _load_anomaly(self) -> ModuleModel:
        evaluation = self.loader.load_json(ANOMALY_MODULE, "evaluation.json")
        feature_columns = evaluation.get("feature_columns", [])
        artifact = evaluation.get("winner_artifact")
        if not artifact or not isinstance(artifact, str) or not artifact.endswith(".joblib"):
            artifact = ANOMALY_ARTIFACT
        model = self.loader.load_joblib(ANOMALY_MODULE, artifact)
        threshold = (
            evaluation.get("val_selected_threshold")
            or evaluation.get("final_test_metrics", {}).get("best_threshold")
            or evaluation.get("threshold")
        )
        return ModuleModel(
            module=ANOMALY_MODULE,
            model=model,
            evaluation=evaluation,
            feature_columns=feature_columns,
            threshold=threshold,
            metadata=self._metadata(ANOMALY_MODULE),
        )

    def _load_budget(self) -> ModuleModel:
        evaluation = self.loader.load_json(BUDGET_MODULE, "evaluation.json")
        config = self.loader.load_json(BUDGET_MODULE, "budget_config.json")
        return ModuleModel(
            module=BUDGET_MODULE,
            model=config,
            evaluation=evaluation,
            feature_columns=[],
            metadata=self._metadata(BUDGET_MODULE),
        )

    @property
    def loaded_modules(self) -> list[str]:
        return [
            name
            for name, module in (
                (PFP_MODULE, self.pfp),
                (FORECASTER_MODULE, self.forecaster),
                (ANOMALY_MODULE, self.anomaly),
                (BUDGET_MODULE, self.budget),
            )
            if module is not None
        ]

    @property
    def is_ready(self) -> bool:
        core = (self.forecaster, self.anomaly)
        return all(m is not None for m in core)
