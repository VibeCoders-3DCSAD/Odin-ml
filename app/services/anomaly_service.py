from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.models.registry import ModuleModel
from app.schemas.anomaly import (
    AnomalousTransaction,
    AnomalyRequest,
    OverspendingTransaction,
    WhitelistEntry,
)
from app.services.features import anomaly_feature_vectors


def _score(module: ModuleModel, X: np.ndarray) -> np.ndarray:
    model = module.model
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            recon = model(torch.tensor(X, dtype=torch.float32)).numpy()
        return np.mean((X - recon) ** 2, axis=1)
    if hasattr(model, "score"):
        return np.asarray(model.score(X), dtype=float)
    if hasattr(model, "decision_function"):
        scores = -model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores
    raise TypeError(f"unsupported anomaly detector type: {type(model)}")


def _normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-8:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo + 1e-8)


def _explanation(txn: dict, score: float, threshold: float) -> tuple[str, list[str]]:
    """Return (reason, feature_contributions) for a single transaction."""
    reason_parts = []
    contributions = []
    amount = float(txn["amount"])
    if score >= threshold:
        reason_parts.append("Reconstruction error above the detection threshold")

    category = str(txn.get("category", "unknown"))
    if category not in {"food", "housing", "transport", "health", "education", "other"}:
        reason_parts.append("Unusual merchant category")
        contributions.append(f"category:{category}")
    if amount > 2000.0:
        reason_parts.append("Transaction amount exceeds typical single-spend size")
        contributions.append(f"amount:{amount:.0f}")

    reason = "; ".join(reason_parts) if reason_parts else "No anomaly signals detected"
    return reason, contributions


def _whitelisted(txn: dict, whitelist: list[WhitelistEntry] | None) -> bool:
    if not whitelist:
        return False
    txn_id = str(txn.get("transaction_id") or "")
    category = str(txn.get("category", "")).lower()
    for entry in whitelist:
        if entry.transaction_id and txn_id and entry.transaction_id == txn_id:
            return True
        if entry.category and category == entry.category.lower():
            return True
    return False


def _overspending_detect(
    transactions: list[dict],
    budget_allocations: list[dict],
) -> list[OverspendingTransaction]:
    """Rule-based overspending detection: category spend vs budget."""
    budget_map = {b["category_id"]: b["budget_amount"] for b in budget_allocations}
    category_totals: dict[str, float] = {}
    for txn in transactions:
        if txn.get("transaction_type") != "expense":
            continue
        cat = txn.get("category", "other")
        category_totals[cat] = category_totals.get(cat, 0.0) + float(txn["amount"])

    results: list[OverspendingTransaction] = []
    for cat, total in category_totals.items():
        budget = budget_map.get(
            cat, budget_map.get(f"essentials_{cat}", budget_map.get(f"discretionary_{cat}"))
        )
        if budget is None:
            continue
        if total > budget:
            for txn in transactions:
                if txn.get("category") == cat and txn.get("transaction_type") == "expense":
                    results.append(
                        OverspendingTransaction(
                            transaction_id=str(txn.get("transaction_id") or ""),
                            budget_excess=round(float(txn["amount"]) * (1 - budget / total), 2),
                            category=cat,
                        )
                    )
    return results


def detect(module: ModuleModel, request: AnomalyRequest) -> list[AnomalousTransaction]:
    """Score each transaction; target = the most recent transaction."""
    transactions = [t.model_dump() for t in request.transactions]

    # Apply whitelist filtering
    txns_for_scoring = [t for t in transactions if not _whitelisted(t, request.whitelist)]

    _, feature_frame = anomaly_feature_vectors(txns_for_scoring)
    if feature_frame.empty:
        raise ValueError("anomaly detection requires baseline transaction history")

    X = feature_frame[module.feature_columns].values.astype(np.float32)
    scores = _normalize(_score(module, X))
    if module.threshold is None:
        raise RuntimeError(
            "anomaly detector has no operating threshold: evaluation.json is missing "
            "val_selected_threshold and no fallback is configured"
        )
    threshold = module.threshold

    results = []
    for i, txn in enumerate(txns_for_scoring):
        score = float(scores[i]) if i < len(scores) else 0.0
        reason, contribs = _explanation(txn, score, threshold)
        results.append(
            AnomalousTransaction(
                transaction_id=str(txn.get("transaction_id") or i),
                anomaly_score=round(score, 4),
                reason=reason,
                feature_contributions=contribs,
            )
        )
    return results
