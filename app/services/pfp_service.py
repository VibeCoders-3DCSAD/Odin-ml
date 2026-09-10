from __future__ import annotations

import re

import numpy as np

from app.models.registry import ModuleModel
from app.schemas.common import ModuleStatus
from app.schemas.pfp import PFPClassification, PFPClassifyRequest
from app.services.features import pfp_feature_vector

_TIER_RE = re.compile(r"\btier(\d+)_")


def _tier_from_winner(winner: str) -> int:
    """Parse the tier number from a winner id (``tier3_svm`` -> 3, 0 if unknown)."""
    match = _TIER_RE.search(winner)
    return int(match.group(1)) if match else 0


def _mdd_label(model_label: str) -> str:
    """Map model prediction label to MDD underscore format.

    Model classes use slash and hyphen: ``Stable/Flexible/At-Risk``
    MDD contract uses underscores: ``STABLE_FLEXIBLE_AT_RISK``
    """
    return model_label.replace("/", "_").replace("-", "_").upper()


def _calibrated_scores(classes: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Derive calibrated dimension scores from model predict_proba.

    For the 3 binary PFP dimensions, marginalise the multiclass
    probability over each dimension's positive value:
      - stability_score = Σ P(class) for classes starting with "Stable"
      - weight_score    = Σ P(class) for classes containing "Obligated"
      - tolerance_score = Σ P(class) for classes containing "Tolerant"

    The resulting scores are properly calibrated probabilities derived
    from the model's own probabilistic output, consistent with the
    MDD requirement for calibrated scores.
    """
    class_labels = [str(c) for c in classes]
    stability = sum(p for c, p in zip(class_labels, proba, strict=True) if c.startswith("Stable"))
    weight = sum(p for c, p in zip(class_labels, proba, strict=True) if "Obligated" in c)
    tolerance = sum(p for c, p in zip(class_labels, proba, strict=True) if "Tolerant" in c)
    return {
        "stability": round(stability, 4),
        "weight": round(weight, 4),
        "tolerance": round(tolerance, 4),
    }


def classify_standard(model: ModuleModel, request: PFPClassifyRequest) -> PFPClassification:
    transactions = request.payload.get("historical_transactions") or []
    vector = pfp_feature_vector(transactions, model.feature_columns)

    artifact = model.model
    estimator = artifact.get("model", artifact) if isinstance(artifact, dict) else artifact
    prediction_raw = str(estimator.predict(vector)[0])
    proba = estimator.predict_proba(vector)[0]
    classes = estimator.classes_

    scores = _calibrated_scores(classes, proba)
    winner = str(model.evaluation.get("winner") or model.metadata.get("winner") or "")
    return PFPClassification(
        prediction=_mdd_label(prediction_raw),
        financial_stability_score=scores["stability"],
        financial_weight_score=scores["weight"],
        financial_tolerance_score=scores["tolerance"],
        confidence=round(float(max(proba)), 4),
        status=ModuleStatus.SUCCESS,
        tier_used=_tier_from_winner(winner),
        model_name=winner or "unknown",
    )


def classify_questionnaire(request: PFPClassifyRequest) -> PFPClassification:
    """Deterministic mapping from questionnaire answers to a PFP class.

    Tier 0 fallback (no model inference) for cold-start users, per the
    PFP MDD: QUESTIONNAIRE mode is the system's cold-start answer.
    """
    answers = request.payload.get("questionnaire_answers") or {}
    variability = str(answers.get("income_variability", "")).lower()
    obligation = str(answers.get("obligation_level", "")).lower()
    runway = str(answers.get("emergency_runway", "")).lower()

    stability = 0.2 if "variable" in variability or "irregular" in variability else 0.8
    weight = 0.8 if "high" in obligation else (0.4 if "medium" in obligation else 0.2)
    tolerance = 0.2 if "low" in runway else (0.8 if "high" in runway or "3" in runway else 0.5)

    label_parts = []
    label_parts.append("Variable" if stability < 0.5 else "Stable")
    label_parts.append("Obligated" if weight >= 0.5 else "Flexible")
    label_parts.append("At-Risk" if tolerance < 0.5 else "Tolerant")
    prediction = "/".join(label_parts)

    confidence = round(max(stability, weight, tolerance), 4)
    return PFPClassification(
        prediction=_mdd_label(prediction),
        financial_stability_score=round(stability, 4),
        financial_weight_score=round(weight, 4),
        financial_tolerance_score=round(tolerance, 4),
        confidence=confidence,
        status=ModuleStatus.SUCCESS,
        tier_used=0,
        model_name="questionnaire_rule",
    )


def classify(model: ModuleModel | None, request: PFPClassifyRequest) -> PFPClassification:
    if request.classification_mode.value == "QUESTIONNAIRE":
        return classify_questionnaire(request)
    if model is None:
        raise ValueError("pfp model not loaded; pending training")
    return classify_standard(model, request)
