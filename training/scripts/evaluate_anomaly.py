"""Evaluate the committed anomaly detectors on the held-out TEST split only.

Generalization harness for the anomaly family. Unlike `train_anomaly.py` (which
fits every tier per fold) this script imports the existing fold evidence from
`models/anomaly/evaluation.json` and scores the *committed* artifacts on the
held-out test split to answer one question:

    How well do the fold-selected candidates actually generalize to unseen
    users (+ unseen transactions)?

It assumes Option A (docs/thesis/anomaly-decision-rule-rationale.md):
    - fold level nominates a candidate (stored `fold_nominated_winner`);
    - the served winner must pass PR-AUC >= 1.5×IQR and PR-AUC >= 0.15 on the
      TEST split, else fall back to IQR;
    - the fold-vs-test gap is recorded, not hidden.

What it does:
    1. Re-selects the fold nomination from stored fold PR-AUC means.
    2. Scores the IQR baseline and the persisted hybrid artifact (if present)
       on the held-out TEST split; PR-AUC only (threshold-free).
    3. Applies the Option A gates to the test evidence.
    4. Rewrites `evaluation.json` (winner, winner_reason, decision_rule,
       test_validation) — NO retraining, NO artifact substitution.

Usage:
    python training/scripts/evaluate_anomaly.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ml.models import HybridEnsemble, IQRDetector
from training.scripts.train_anomaly import compute_metrics, extract_features

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "models" / "anomaly"
EVAL_JSON = EVAL_DIR / "evaluation.json"

PR_AUC_MIN_TIMES_IQR = 1.5
PR_AUC_TARGET = 0.15


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(REPO_ROOT / "training" / "datasets" / "anomaly" / f"{name}.parquet")
    return extract_features(df)


def _fold_nomination(summary: dict) -> tuple[str, str]:
    iqr = summary["tier1_iqr"]["pr_auc_mean"]
    excluded = {"baseline"}
    best = max(
        (name for name in summary if name not in excluded),
        key=lambda n: summary[n]["pr_auc_mean"],
    )
    best_pr = summary[best]["pr_auc_mean"]
    imp = (best_pr - iqr) / max(iqr, 1e-8)
    passed = imp >= (PR_AUC_MIN_TIMES_IQR - 1.0) and best_pr >= PR_AUC_TARGET
    reason = (
        f"Fold cross-validation nominated {best} (PR-AUC {best_pr:.4f}, "
        f"improvement over IQR {imp * 100:.1f}%; gates passed: {passed})."
    )
    return best, reason


def _test_pr_auc(model: object, X_test: np.ndarray, y_test: np.ndarray) -> float:
    if isinstance(model, (HybridEnsemble, IQRDetector)):
        scores = model.score(X_test)
    elif hasattr(model, "decision_function"):

        def _score(X):
            return -np.asarray(model.decision_function(X), dtype=float)

        scores = _score(X_test)
    else:
        raise TypeError(f"unsupported scorer: {type(model)}")
    return float(compute_metrics(y_test, scores)["pr_auc"])


def main() -> None:
    if not EVAL_JSON.exists():
        raise FileNotFoundError(f"missing {EVAL_JSON} — run train_anomaly.py first")
    evaluation = json.loads(EVAL_JSON.read_text())
    summary = evaluation["fold_summary"]

    nomination, nomination_reason = _fold_nomination(summary)
    print(f"Fold nomination: {nomination}")

    X_train, _ = _load_split("train")
    X_test, y_test = _load_split("test")
    print(f"TEST split: {X_test.shape} rows, anomaly rate {float(y_test.mean()):.4f}")

    iqr = IQRDetector(iqr_multiplier=1.5).fit(X_train)
    iqr_pr = _test_pr_auc(iqr, X_test, y_test)
    print(f"  IQR test PR-AUC: {iqr_pr:.4f}")

    hybrid = None
    artifact = EVAL_DIR / "anomaly_detector.joblib"
    if artifact.exists():
        try:
            loaded = joblib.load(artifact)
            if isinstance(loaded, HybridEnsemble):
                hybrid = loaded
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  WARNING: artifact unreadable: {exc}")
    if hybrid is None:
        print("  (no persisted HybridEnsemble on disk — candidate validation skipped)")
        validation = {
            "measured": False,
            "reason": "no hybrid artifact available to validate on test",
            "pr_auc": None,
            "iqr_test_pr_auc": round(iqr_pr, 4),
        }
    else:
        hybrid_pr = _test_pr_auc(hybrid, X_test, y_test)
        ratio = hybrid_pr / max(iqr_pr, 1e-8)
        gate = ratio >= PR_AUC_MIN_TIMES_IQR and hybrid_pr >= PR_AUC_TARGET
        validation = {
            "measured": True,
            "pr_auc": round(hybrid_pr, 4),
            "iqr_test_pr_auc": round(iqr_pr, 4),
            "improvement_ratio": round(ratio, 3),
            "pr_auc_improvement_over_iqr_pct": round((ratio - 1.0) * 100.0, 1),
            "improvement_gate_passed": bool(ratio >= PR_AUC_MIN_TIMES_IQR),
            "target_gate_passed": bool(hybrid_pr >= PR_AUC_TARGET),
            "gate_passed": bool(gate),
            "reason": (
                f"hybrid test PR-AUC {hybrid_pr:.4f} vs IQR {iqr_pr:.4f} "
                f"(ratio {ratio:.3f} >= {PR_AUC_MIN_TIMES_IQR}, "
                f"target >= {PR_AUC_TARGET}); passed: {gate}"
            ),
        }
        print(f"  Hybrid test PR-AUC: {hybrid_pr:.4f} (gate passed: {gate})")

    served = "tier1_iqr"
    if not validation.get("measured"):
        winner_reason = (
            "No stored hybrid artifact to validate on held-out test; the "
            "pre-registered Option A rule cannot adopt a candidate that has no "
            "test evidence, so the served winner stays the interpretable IQR "
            "baseline. "
        )
    elif validation.get("gate_passed"):
        served = nomination
        winner_reason = (
            f"Option A passed on held-out test: {validation['reason']} "
        )
    else:
        winner_reason = (
            f"Option A FAILED on held-out test: {validation['reason']} "
            f"Fold nomination ({nomination}) does not generalize to unseen "
            f"users (fold-vs-test gap recorded in test_validation); falling back "
            f"to the interpretable IQR baseline per the pre-registered rule. "
        )
    winner_reason += nomination_reason
    print(f"Served winner: {served}")

    evaluation.update(
        {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "winner": served,
            "winner_reason": winner_reason,
            "fold_nominated_winner": nomination,
            "fold_nominated_reason": nomination_reason,
            "test_validation": validation,
            "decision_rule": {
                "metric": "pr_auc",
                "pr_auc_target": PR_AUC_TARGET,
                "pr_auc_improvement_ratio": PR_AUC_MIN_TIMES_IQR,
                "pr_auc_improvement_over_iqr_pct": validation.get(
                    "pr_auc_improvement_over_iqr_pct"
                ),
                "rule_passed": bool(validation.get("gate_passed", False)),
                "rules_applied_on": "held-out test split (unseen users + unseen txns)",
            },
        }
    )
    EVAL_JSON.write_text(json.dumps(evaluation, indent=2))
    print(f"Updated {EVAL_JSON}")


if __name__ == "__main__":
    main()
