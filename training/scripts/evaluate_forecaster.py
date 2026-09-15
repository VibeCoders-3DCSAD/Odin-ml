"""Evaluate-only generalization harness for the committed forecaster winner.

Phase 2c: measure how ``tier3_sarima`` generalizes to UNSEEN users without
retraining or swapping the served artifact.

The committed fold metrics in ``models/forecaster/evaluation.json`` were
produced by ``run_wfv`` in ``train_forecaster.py``, which concatenated the
train/val/test user splits into one pool (``all_data = pd.concat(...)``).
Unseen-user generalization is therefore NOT captured there.

This harness mirrors the walk-forward protocol exactly (same helper
functions, same temporal folds, same metrics) with two leak-free changes:

1. The pooled SARIMA/ARIMA is fit on the TRAIN split only
   (``training/datasets/forecaster/train.parquet``): test users never
   contribute to the pool used to forecast them.
2. Each unseen test user's pooled forecast is rescaled by that user's own
   trailing expense mean from their own pre-window months (matching the
   production rescale in ``_predict_monthly_total``), falling back to the
   global train mean for cold-start users.

The result is written to ``generalization`` in
``models/forecaster/evaluation.json`` and the committed winner/artifacts are
left untouched.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from training.scripts.train_forecaster import (  # noqa: E402
    aggregate_to_monthly,
    compute_metrics,
    forecast_sarima_pool,
)

_DATA = _REPO / "training" / "datasets" / "forecaster"
_FOLDS = _DATA / "temporal_folds.json"
_EVAL = _REPO / "models" / "forecaster" / "evaluation.json"
_FEATURE_COLS = _DATA / "feature_columns.json"


def _load_folds() -> list[dict[str, Any]]:
    return json.loads(_FOLDS.read_text())


def _load_monthly(name: str, feature_cols: list[str]) -> Any:
    import pandas as pd

    df = pd.read_parquet(_DATA / f"{name}.parquet")
    return aggregate_to_monthly(df, feature_cols)


def _trailing_scaler(
    test_monthly: Any, prior_months: list[int], global_mean: float
) -> dict[str, float]:
    """Per-unseen-user rescale factor from their own pre-window expense mean.

    Mirrors the production rescale (user's own trailing mean with a global
    fallback for cold starts); ``prior_months`` excludes the target months so
    the row being scored never leaks into the scaler.
    """
    window = test_monthly[test_monthly["month"].isin(prior_months)]
    means = window.groupby("user_id")["target_expenses"].mean()
    means = means.where(means > 0, global_mean).fillna(global_mean)
    return dict(means)


def main() -> None:
    print("[1/5] Loading committed evaluation, folds, and feature columns...")
    evaluation = json.loads(_EVAL.read_text())
    folds = _load_folds()
    feature_cols = json.loads(_FEATURE_COLS.read_text())
    committed = evaluation.get("aggregate_metrics", {}).get("tier3_sarima", {})

    print("[2/5] Aggregating train (seen) and test (unseen) splits to monthly...")
    train_monthly = _load_monthly("train", feature_cols)
    test_monthly = _load_monthly("test", feature_cols)
    global_mean = float(train_monthly["target_expenses"].mean())
    print(
        f"    train users={train_monthly['user_id'].nunique()}, "
        f"test users={test_monthly['user_id'].nunique()}"
    )

    print("[3/5] Walk-forward pool forecast fit on TRAIN users only...")
    fold_metrics: list[dict[str, Any]] = []
    naive_metrics_list: list[dict[str, Any]] = []
    for fold in folds:
        train_months = list(fold["train_months"])
        test_months = list(fold["test_months"])
        target_row_months = [m - 1 for m in test_months if m - 1 >= 1]
        prior_months = list(range(1, max(train_months) + 1))

        arima_hist = train_monthly[train_monthly["month"].isin(train_months)]
        train_mean = float(arima_hist["target_expenses"].mean())
        path_map, n_fits = forecast_sarima_pool(arima_hist, target_row_months)

        test_actual = test_monthly[test_monthly["month"].isin(target_row_months)].copy()
        test_actual = test_actual.sort_values(["user_id", "month"])
        if test_actual.empty:
            continue

        scaler = _trailing_scaler(test_monthly, prior_months, global_mean)

        def _rescale(uid: Any, month: int, path_map=path_map, scaler=scaler) -> float:
            level = scaler.get(uid, global_mean)
            return float(path_map.get(month, 1.0)) * level

        y_true = test_actual["target_expenses"].values
        y_pred = np.asarray(
            [
                _rescale(uid, month)
                for uid, month in zip(test_actual["user_id"], test_actual["month"], strict=True)
            ],
            dtype=float,
        )
        prior_rows = test_monthly[test_monthly["month"].isin([m - 2 for m in test_months])]
        prior_map = prior_rows.set_index("user_id")["target_expenses"]
        y_prev = test_actual["user_id"].map(prior_map).values

        metrics = compute_metrics(y_true, y_pred, "tier3_sarima", y_prev=y_prev)
        naive_metrics = compute_metrics(
            y_true,
            np.full(len(y_true), train_mean, dtype=float),
            "naive_baseline",
            y_prev=y_prev,
        )
        fold_metrics.append(
            {
                "fold": fold["fold"],
                "train_months": train_months,
                "test_months": test_months,
                "n_test": int(len(test_actual)),
                "n_pool_fits": n_fits,
                **metrics,
            }
        )
        naive_metrics_list.append(naive_metrics)
        print(
            f"    Fold {fold['fold']}: MAPE={metrics['mape']:.2f}% "
            f"(SARIMA) vs {naive_metrics['mape']:.2f}% (naive), "
            f"SMAPE={metrics['smape']:.2f}%, MDA={metrics['mda']:.4f} "
            f"(unseen users, pool_fits={n_fits})"
        )

    print("[4/5] Aggregating unseen-user metrics and comparing to committed fold means...")
    aggregated: dict[str, Any] = {}
    for key in ("mae", "smape", "mda", "rmse", "mape", "r2"):
        values = [float(f[key]) for f in fold_metrics if np.isfinite(float(f[key]))]
        aggregated[key] = round(float(np.mean(values)), 4) if values else None
        aggregated[f"{key}_std"] = round(float(np.std(values)), 4) if len(values) > 1 else None

    naive_aggregate: dict[str, Any] = {}
    for key in ("mae", "smape", "mda", "rmse", "mape", "r2"):
        values = [float(f[key]) for f in naive_metrics_list if np.isfinite(float(f[key]))]
        naive_aggregate[key] = round(float(np.mean(values)), 4) if values else None
    mape_reduction_pct = (
        1.0 - aggregated["mape"] / naive_aggregate["mape"]
        if aggregated.get("mape") is not None and naive_aggregate.get("mape")
        else None
    )

    deltas: dict[str, Any] = {}
    for key in ("mape", "smape", "mae", "rmse"):
        unseen = aggregated.get(key)
        trained = committed.get(key)
        if unseen is not None and trained is not None:
            deltas[key] = round(float(unseen) - float(trained), 4)

    generalization = {
        "measured": bool(fold_metrics),
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "protocol": (
            "walk-forward (temporal_folds.json) pool forecast; pooled ARIMA/SARIMA "
            "fit on TRAIN-split users only; each unseen test user rescaled by their "
            "own trailing expense mean (global fallback for cold starts); naive = "
            "mean of TRAIN target expenses per fold; matches run_wfv helpers/metrics"
        ),
        "n_folds_evaluated": len(fold_metrics),
        "feature_cols": feature_cols,
        "aggregate_metrics_unseen_users": aggregated,
        "aggregate_naive_unseen_users": naive_aggregate,
        "fold_details": fold_metrics,
        "committed_fold_metrics_seen_and_unseen_users": committed,
        "metric_delta_vs_committed": deltas,
        "mape_reduction_vs_naive_pct": round(mape_reduction_pct * 100.0, 2)
        if mape_reduction_pct is not None
        else None,
        "served_winner_unchanged": True,
        "why_unchanged": (
            "evaluate-only harness per approved plan; winner/artifacts not swapped. "
            "Fold metrics previously concatenated all user splits; this section "
            "records the unseen-user generalization gap for the thesis ledger."
        ),
        "committed_winner": evaluation.get("winner"),
    }

    evaluation["generalization"] = generalization
    _EVAL.write_text(json.dumps(evaluation, indent=2))

    print("[5/5] Writing report...")
    _write_report(evaluation, aggregated, naive_aggregate, mape_reduction_pct)

    print("    Done. generalized_mape=", aggregated.get("mape"))


def _write_report(
    evaluation: dict,
    agg: dict[str, Any],
    naive_agg: dict[str, Any],
    mape_reduction_pct: float | None,
) -> None:
    report = _REPO / "models" / "forecaster" / "evaluation_report.md"
    lines = [
        "## Unseen-User Generalization (evaluate-only, Phase 2c)",
        "",
        f"- **Winner unchanged:** {evaluation.get('winner')} (served artifact untouched)",
        "- **Protocol:** pooled ARIMA/SARIMA fit on TRAIN-split users only; each",
        "  unseen test user rescaled by their own trailing expense mean. Walk-forward",
        "  temporal folds identical to the committed run (temporal_folds.json).",
        "- **Aggregate on unseen users:**",
        f"  - MAPE: {agg.get('mape')}%",
        f"  - SMAPE: {agg.get('smape')}%",
        f"  - MAE: {agg.get('mae')}",
        f"  - RMSE: {agg.get('rmse')}",
        f"  - MDA: {agg.get('mda')}",
        "",
        f"- **Naive baseline on same unseen rows:** MAPE {naive_agg.get('mape')}%",
        f"- **MAPE reduction vs naive (unseen users):** {mape_reduction_pct}%",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
