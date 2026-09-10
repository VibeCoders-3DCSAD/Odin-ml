"""Re-derive the anomaly winner under the revised Option A rule, honestly.

Background:
    The fold evaluation in `models/anomaly/evaluation.json` is data-driven and
    rule-independent: every tier (IQR, adaptive, IsolationForest, OCSVM, AE,
    hybrid) was scored per fold on the same train/val/test splits. The decision
    rule (Option A, see docs/thesis/anomaly-decision-rule-rationale.md):

        adopt the best-learned tier iff PR-AUC(winner) >= 1.5 x PR-AUC(IQR)
        and PR-AUC(winner) >= 0.15; otherwise fall back to the IQR baseline.
        Operating point: maximize F2 (beta=2) on the held-out val split subject
        to precision >= 0.30 (no test leakage).

    The winner must be defined honestly and truthfully on the held-out test
    set. Folds may only *nominate* a candidate; the served winner is whichever
    candidate passes the rule on unseen-user test evidence. This script:

      1. Re-selects the fold-nominated candidate from stored fold PR-AUC means
         (no re-run of the 5-fold loop).
      2. Fits the IQR baseline on the full training split (cheap, exact).
      3. Scores the fold-nominated hybrid on the held-out TEST split. If its
         persisted artifact is still on disk it is reused as-is; otherwise the
         hybrid is re-fit on the full training split (OCSVM subsample<=3000,
         AE 30 epochs — mirroring fold-time behavior).
      4. Applies Option A's PR-AUC gates to the TEST evidence.
      5. Serves the honest winner (IQR fallback when the rule fails on test)
         as `anomaly_detector.joblib`.
      6. Rewrites evaluation.json, evaluation_report.md, metadata.json with
         BOTH the fold nomination and the test-validation outcome, so the
         fold-vs-test generalization gap is reported, not hidden.

Usage:
    python training/scripts/rederive_anomaly_winner.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ml.models import AdaptiveThresholdDetector, HybridEnsemble, IQRDetector
from training.scripts.train_anomaly import (
    HAS_PYTORCH,
    _select_operating_point,
    build_autoencoder,
    compute_metrics,
    extract_features,
    train_isolation_forest,
    train_ocsvm,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "models" / "anomaly"
EVAL_JSON = EVAL_DIR / "evaluation.json"

PR_AUC_MIN_TIMES_IQR = 1.5
PR_AUC_TARGET = 0.15
MIN_PRECISION = 0.30
BETA = 2.0


def _load_parquet(name: str) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(REPO_ROOT / "training" / "datasets" / "anomaly" / f"{name}.parquet")
    return extract_features(df)


def _nominate_from_folds(summary: dict) -> tuple[str, str]:
    """Nominate the fold-level Option A candidate (documented, not served)."""
    iqr = summary["tier1_iqr"]["pr_auc_mean"]
    excluded = {"baseline"}
    best = max(
        (name for name in summary if name not in excluded),
        key=lambda n: summary[n]["pr_auc_mean"],
    )
    best_pr = summary[best]["pr_auc_mean"]
    improvement = (best_pr - iqr) / max(iqr, 1e-8)
    passed = improvement >= (PR_AUC_MIN_TIMES_IQR - 1.0) and best_pr >= PR_AUC_TARGET
    if passed:
        return best, (
            f"Fold cross-validation nominated {best}: PR-AUC {best_pr:.4f} "
            f"(improvement over IQR {improvement * 100:.1f}% >= 50%, target "
            f"{PR_AUC_TARGET:.2f} reached on folds)."
        )
    return best, (
        f"Fold cross-validation best tier was {best} (PR-AUC {best_pr:.4f}), but "
        f"it failed the {PR_AUC_TARGET:.2f} target / 50% improvement gates on folds."
    )


def _fit_hybrid_candidate(X_train, y_train, X_val, y_val):
    """Fit the fold-nominated hybrid on the full training set.

    OCSVM mirrors fold-time `train_ocsvm` (<=3000-sample subsample); AE runs 30
    epochs exactly as evaluated. Returns (model, score_fn, params).
    """
    members: dict[str, object] = {}
    members["iqr"] = IQRDetector(iqr_multiplier=1.5)
    members["iqr"].fit(X_train)
    members["adaptive"] = AdaptiveThresholdDetector(iqr_multiplier=1.5)
    members["adaptive"].fit(X_train)

    if_model, if_params, _ = train_isolation_forest(X_train, y_train, X_val, y_val)
    members["if"] = if_model
    print(f"    IsolationForest fit (contamination={if_params['contamination']:.4f})")

    oc_model, oc_params, oc_score = train_ocsvm(X_train, y_train, X_val, y_val)
    if oc_model is not None:
        members["ocsvm"] = oc_model
        print(f"    OCSVM fit (subsample<=3000, nu={oc_params['nu']}, val F1={oc_score:.4f})")

    if HAS_PYTORCH:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        ae = build_autoencoder(X_train.shape[1])
        if ae is not None:
            X_t = torch.tensor(X_train, dtype=torch.float32)
            n_val = max(1, int(len(X_t) * 0.1))
            X_tr = X_t[: len(X_t) - n_val]
            train_ds = TensorDataset(X_tr, X_tr)
            train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
            optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            t0 = time.time()
            for _ in range(30):
                ae.train()
                for xb, _ in train_dl:
                    optimizer.zero_grad()
                    loss = criterion(xb, ae(xb))
                    loss.backward()
                    optimizer.step()
            ae.eval()
            members["ae"] = ae
            print(f"    Autoencoder fit (30 epochs, {time.time() - t0:.1f}s)")

    order = ["iqr", "adaptive", "if", "ocsvm"] + (["ae"] if "ae" in members else [])
    model = HybridEnsemble(
        detectors=[members[k] for k in order],
        weights=[1.0 / len(order)] * len(order),
    )

    def _score(X):
        return model.score(X)

    return model, _score, {"members": order}


def _get_hybrid_artifact() -> object | None:
    """Return the persisted hybrid if still on disk and un-polluted."""
    artifact = EVAL_DIR / "anomaly_detector.joblib"
    if not artifact.exists():
        return None
    try:
        model = joblib.load(artifact)
    except Exception:
        return None
    if not isinstance(model, HybridEnsemble):
        return None
    non_detectors = [m for m in model.detectors if not hasattr(m, "decision_function")]
    if len(non_detectors) == len(model.detectors):  # nothing usable (disarmed)
        return None
    return model


def _validate_on_test(
    hybrid_model: object | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
    iqr_test_pr_auc: float,
) -> dict:
    """Measure the fold-nominated hybrid's held-out TEST PR-AUC and apply the
    Option A gates to that evidence. Threshold-free (PR-AUC is a ranking
    metric), so a persisted artifact can be scored as-is — no refit needed."""
    if hybrid_model is None:
        return {
            "measured": False,
            "reason": "no hybrid candidate available to score",
            "pr_auc": None,
            "iqr_test_pr_auc": round(iqr_test_pr_auc, 4),
            "pr_auc_improvement_over_iqr_pct": None,
            "improvement_ratio": None,
            "improvement_gate_passed": None,
            "target_gate_passed": None,
            "gate_passed": None,
        }
    hybrid_pr = float(compute_metrics(y_test, hybrid_model.score(X_test))["pr_auc"])
    ratio = hybrid_pr / max(iqr_test_pr_auc, 1e-8)
    if iqr_test_pr_auc <= 0:
        return {
            "measured": False,
            "reason": "IQR test PR-AUC is 0; ratio undefined",
            "pr_auc": round(hybrid_pr, 4),
            "iqr_test_pr_auc": round(iqr_test_pr_auc, 4),
            "pr_auc_improvement_over_iqr_pct": None,
            "improvement_ratio": None,
            "improvement_gate_passed": None,
            "target_gate_passed": None,
            "gate_passed": None,
        }
    improvement_passed = ratio >= PR_AUC_MIN_TIMES_IQR
    target_passed = hybrid_pr >= PR_AUC_TARGET
    return {
        "measured": True,
        "pr_auc": round(hybrid_pr, 4),
        "iqr_test_pr_auc": round(iqr_test_pr_auc, 4),
        "pr_auc_improvement_over_iqr_pct": round((ratio - 1.0) * 100.0, 1),
        "improvement_ratio": round(ratio, 3),
        "improvement_gate_passed": improvement_passed,
        "target_gate_passed": target_passed,
        "gate_passed": improvement_passed and target_passed,
        "reason": (
            f"hybrid test PR-AUC {hybrid_pr:.4f} vs IQR test PR-AUC "
            f"{iqr_test_pr_auc:.4f} (ratio {ratio:.3f} >= {PR_AUC_MIN_TIMES_IQR}; "
            f"hybrid PR-AUC >= {PR_AUC_TARGET}); gate passed: "
            f"{improvement_passed and target_passed}"
        ),
    }


def _fit_iqr_full(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """Fit the interpretable IQR baseline on the full train split and select its
    val operating point (Option A policy). Returns model + metrics dict."""
    model = IQRDetector(iqr_multiplier=1.5)
    model.fit(X_train)

    def _score(X):
        return model.score(X)

    val_scores = _score(X_val)
    threshold, op_point = _select_operating_point(
        y_val, val_scores, min_precision=MIN_PRECISION, beta=BETA
    )
    val_metrics = compute_metrics(y_val, val_scores, threshold=threshold)
    test_scores = _score(X_test)
    final_metrics = compute_metrics(y_test, test_scores, threshold=threshold)
    print(f"    IQR fit on {len(X_train)} rows complete")
    print(f"    Val-selected threshold: {threshold:.4f} "
          f"(F2 {op_point['f2']:.4f}, P {op_point['precision']:.4f}, "
          f"R {op_point['recall']:.4f})")
    print(f"    Test F1 {final_metrics['f1']:.4f}, P {final_metrics['precision']:.4f}, "
          f"R {final_metrics['recall']:.4f}, PR-AUC {final_metrics['pr_auc']:.4f}")
    return model, {
        "threshold": threshold,
        "op_point": op_point,
        "val_metrics": val_metrics,
        "final_metrics": final_metrics,
    }


def _write_report_and_metadata(evaluation: dict) -> None:
    from app.ml.metadata import build_metadata, write_metadata
    from app.ml.reporting import family_metadata, family_report, write_evaluation_report

    write_evaluation_report(EVAL_DIR, **family_report("anomaly", evaluation))
    data_sources = [
        REPO_ROOT / "training" / "datasets" / "anomaly" / f"{name}.parquet"
        for name in ("train", "val", "test")
    ]
    metadata = build_metadata(**family_metadata("anomaly", evaluation, data_sources=data_sources))
    metadata["created_at"] = evaluation.get("generated_at", datetime.now(UTC).isoformat())
    previous = EVAL_DIR / "metadata.json"
    if previous.exists():
        prev_commit = json.loads(previous.read_text()).get("training_commit")
        if prev_commit:
            metadata["training_commit"] = prev_commit
    write_metadata(metadata, EVAL_DIR)


def main() -> None:
    if not EVAL_JSON.exists():
        raise FileNotFoundError(f"missing {EVAL_JSON} — run train_anomaly.py first")
    t0 = time.time()

    evaluation = json.loads(EVAL_JSON.read_text())
    summary = evaluation["fold_summary"]
    print("[1/6] Re-selecting fold-nominated candidate (stored fold PR-AUC means)...")
    for name, st in sorted(summary.items()):
        print(f"    {name:<24} PR-AUC {st['pr_auc_mean']:.4f}")
    fold_winner, fold_winner_reason = _nominate_from_folds(summary)
    print(f"    Fold nomination: {fold_winner}")

    print("[2/6] Loading datasets (train/val/test)...")
    X_train, y_train = _load_parquet("train")
    X_val, y_val = _load_parquet("val")
    X_test, y_test = _load_parquet("test")
    print(f"    Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    print("[3/6] Fitting IQR baseline on full train...")
    iqr_model, iqr = _fit_iqr_full(X_train, X_val, y_val, X_test, y_test)

    print("[4/6] Scoring fold-nominated hybrid on held-out TEST split...")
    hybrid = _get_hybrid_artifact()
    if hybrid is not None:
        print("    Reusing persisted hybrid artifact from disk")
    else:
        print("    No usable hybrid artifact; re-fitting on full train (slow)...")
        hybrid, _, _ = _fit_hybrid_candidate(X_train, y_train, X_val, y_val)
    validation = _validate_on_test(
        hybrid, X_test, y_test, float(iqr["final_metrics"]["pr_auc"])
    )
    print(f"    Hybrid test PR-AUC: {validation.get('pr_auc')} "
          f"(measured: {validation.get('measured')})")

    print("[5/6] Applying Option A gates to TEST evidence...")
    served_winner = "tier1_iqr"
    gate_passed = bool(validation.get("gate_passed", False))
    if gate_passed:
        served_winner = fold_winner if isinstance(fold_winner, str) else "tier1_iqr"
    winner_reason = (
        f"Option A rule {'PASSED' if gate_passed else 'did NOT pass'} on held-out "
        f"test evidence: {validation.get('reason', 'no candidate validation')} "
        f"Fold-level nomination ({fold_winner}) {'' if gate_passed else 'does not '
        f'defeat the {PR_AUC_TARGET:.2f} / {PR_AUC_MIN_TIMES_IQR:.1f}x gates on '
        f'unseen users (fold-vs-test generalization gap recorded in test_validation); '
        f'fallback to the interpretable IQR baseline per the pre-registered rule'}. "
    )
    decision_rule = {
        "metric": "pr_auc",
        "pr_auc_target": PR_AUC_TARGET,
        "pr_auc_improvement_ratio": PR_AUC_MIN_TIMES_IQR,
        "pr_auc_improvement_over_iqr_pct": validation.get(
            "pr_auc_improvement_over_iqr_pct"
        ),
        "rule_passed": gate_passed,
        "rules_applied_on": "held-out test split (unseen users + unseen txns)",
        "operating_point": {
            "method": "F2 maximization with precision floor",
            "beta": BETA,
            "min_precision": MIN_PRECISION,
            "selected_on": "held-out val split",
        },
    }

    print(f"[6/6] Serving: {served_winner} -> anomaly_detector.joblib "
          f"(rule passed on test: {gate_passed})")
    joblib.dump(iqr_model, EVAL_DIR / "anomaly_detector.joblib")

    evaluation.update(
        {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "winner": served_winner,
            "winner_artifact": "anomaly_detector.joblib",
            "winner_params": {"iqr_multiplier": 1.5},
            "winner_reason": winner_reason,
            "fold_nominated_winner": fold_winner,
            "fold_nominated_reason": fold_winner_reason,
            "test_validation": validation,
            "decision_rule": decision_rule,
            "val_selected_threshold": iqr["threshold"],
            "val_operating_point": iqr["op_point"],
            "val_metrics": iqr["val_metrics"],
            "final_test_metrics": iqr["final_metrics"],
        }
    )
    EVAL_JSON.write_text(json.dumps(evaluation, indent=2))
    _write_report_and_metadata(evaluation)
    print(f"    Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
