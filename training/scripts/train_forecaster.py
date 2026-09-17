"""
Training Pipeline for the Spending Forecaster

Trains and compares the lightest tier of regressors for monthly expense
forecasting using temporal walk-forward evaluation.

Scope (top-3-lightest, per docs/models/model-candidate-report.md §5):
  Baseline: naive / trailing-mean (FO-02 cold start)
  Tier 2:   Random Forest regressor (monthly aggregated features)
  Tier 3a:  ARIMA (pooled, user-normalized monthly expense series) — statsmodels
  Tier 3b:  GRU (daily feature sequences) — PyTorch

Heavier candidates (LSTM, BiLSTM, hybrid) are documented as "hold" in the
roster and are not trained in the default scope.

Evaluation:
  - 5-fold expanding window (temporal_folds.json); embargo months excluded from
    training labels but available as feature context
  - Primary metrics: MAE, SMAPE, MDA, RMSE (MDD v2.4)
  - Supplementary: MAPE, R-squared
  - Decision rule: best model must beat naive baseline by 20% MAPE reduction

Usage:
    python training/scripts/train_forecaster.py --input training/datasets/forecaster/ --output models/forecaster/
"""

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys

# Ensure the repo root is importable so `app.ml.*` resolves when this script is
# run directly (python training/scripts/train_forecaster.py). Required for a
# self-contained, replicable training pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ml.metadata import build_metadata, write_metadata
from app.ml.models import _SequenceForecaster
from app.ml.reporting import family_metadata, family_report, write_evaluation_report

HAS_PYTORCH = False
torch = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_PYTORCH = True
except ImportError:
    warnings.warn("torch unavailable — Tier 3 (GRU) will be skipped")

HAS_STATSMODELS = False
ARIMA = None
try:
    from statsmodels.tsa.arima.model import ARIMA

    HAS_STATSMODELS = True
except ImportError:
    warnings.warn("statsmodels unavailable — Tier 3 (ARIMA) will be skipped")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECASTER_FEATURES = [
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month",
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_15d",
    "lag_30d",
    "lag_60d",
    "rolling_mean_7d",
    "rolling_std_7d",
    "rolling_mean_14d",
    "rolling_std_14d",
    "rolling_mean_30d",
    "rolling_std_30d",
    "is_payday",
    "days_to_payday",
    "recency",
    "frequency_30d",
    "monetary_30d",
]

META_COLUMNS = [
    "user_id",
    "date",
    "month",
    "year",
    "month_num",
    "target_expenses",
    "has_transaction",
    "has_target",
]

PRE_REGISTERED_MAPE_REDUCTION = 0.20  # 20% MAPE reduction vs naive

SEQ_LENGTH = 30  # days lookback for LSTM

ARIMA_ORDER = (1, 1, 0)
ARIMA_MIN_HISTORY = 6  # months required from a user before ARIMA applies


@dataclass
class FoldResult:
    fold: int
    train_months: list
    test_months: list
    n_train_samples: int
    n_test_samples: int
    tier_results: dict = field(default_factory=dict)


@dataclass
class TrainingReport:
    timestamp: str = ""
    n_folds: int = 0
    fold_results: list = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)
    naive_mape: float = 0.0
    winner: str = ""
    winner_reason: str = ""


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_forecaster_data(input_dir: str) -> dict:
    input_path = Path(input_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        path = input_path / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Forecaster data not found: {path}")
        splits[name] = pd.read_parquet(path)
    return splits


def load_temporal_folds(folds_path: str) -> list:
    with open(folds_path) as f:
        return json.load(f)


def load_feature_columns(input_dir: str) -> dict:
    path = Path(input_dir) / "feature_columns.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Feature Aggregation
# ---------------------------------------------------------------------------


def aggregate_to_monthly(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Aggregate daily features to monthly level."""
    agg_dict = {}
    for col in feature_cols:
        if col in df.columns:
            agg_dict[col] = "mean"
    agg_dict["target_expenses"] = "first"
    agg_dict["has_target"] = "first"
    agg_dict["has_transaction"] = "mean"

    # Legacy engineered datasets predate the `year` column; recover it from the
    # daily `date` (or assume the single-year 2023 corpus).
    if "year" not in df.columns:
        df = df.assign(
            year=df["date"].dt.year if "date" in df.columns else 2023
        )

    monthly = df.groupby(["user_id", "month", "year"]).agg(agg_dict).reset_index()
    # Absolute month index (grid epoch 2023-01 = 1). Grouping by year keeps two
    # Januaries from different years as separate rows.
    monthly["month_num"] = (monthly["year"] - 2023) * 12 + monthly["month"]
    return monthly


def prepare_monthly_sequences(
    monthly_df: pd.DataFrame,
    feature_cols: list,
    lookback: int = 3,
    filter_months: list | None = None,
) -> tuple:
    """Create sequences for LSTM: past N months of features -> next month target.

    A sample is built from the row in month M (target month M+1): features span
    months M-lookback+1..M and the target is month M+1's total expenses. If
    filter_months is provided, only samples whose TARGET month is in
    filter_months are kept. Returns (X, y, meta, y_prev) where y_prev is the
    actual expenses of month M (used for direction-of-change accuracy).
    """
    X_seq, y_seq, meta, y_prev = [], [], [], []
    for uid in monthly_df["user_id"].unique():
        user_data = monthly_df[monthly_df["user_id"] == uid].sort_values("month_num")
        features = user_data[feature_cols].values
        targets = user_data["target_expenses"].values
        months = user_data["month_num"].values

        for i in range(lookback - 1, len(user_data)):
            if np.isnan(targets[i]):
                continue
            target_month = int(months[i]) + 1
            if filter_months is None or target_month in filter_months:
                X_seq.append(features[i - lookback + 1 : i + 1])
                y_seq.append(targets[i])
                y_prev.append(targets[i - 1])
                meta.append({"user_id": uid, "month": target_month})

    if not X_seq:
        return np.array([]), np.array([]), [], np.array([])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        meta,
        np.array(y_prev, dtype=np.float32),
    )


def prepare_flat_features(
    monthly_df: pd.DataFrame, feature_cols: list, filter_months: list | None = None
) -> tuple:
    """Create flat feature matrix for RF: past N months flattened.

    A sample is built from the row in month M (target month M+1): features span
    months M-lookback+1..M and the target is month M+1's total expenses. If
    filter_months is provided, only samples whose TARGET month is in
    filter_months are kept. Returns (X, y, meta, y_prev).
    """
    lookback = 3
    X_flat, y_flat, meta, y_prev = [], [], [], []
    for uid in monthly_df["user_id"].unique():
        user_data = monthly_df[monthly_df["user_id"] == uid].sort_values("month_num")
        features = user_data[feature_cols].values
        targets = user_data["target_expenses"].values
        months = user_data["month_num"].values

        for i in range(lookback - 1, len(user_data)):
            if np.isnan(targets[i]):
                continue
            target_month = int(months[i]) + 1
            if filter_months is None or target_month in filter_months:
                x = features[i - lookback + 1 : i + 1].flatten()
                X_flat.append(x)
                y_flat.append(targets[i])
                y_prev.append(targets[i - 1])
                meta.append({"user_id": uid, "month": target_month})

    if not X_flat:
        return np.array([]), np.array([]), [], np.array([])
    return (
        np.array(X_flat, dtype=np.float32),
        np.array(y_flat, dtype=np.float32),
        meta,
        np.array(y_prev, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tier_name: str, y_prev: np.ndarray | None = None
) -> dict:
    """Compute regression metrics.

    Primary (MDD v2.4): MAE, SMAPE, MDA, RMSE. Supplementary: MAPE, R-squared.
    MDA (direction-of-change accuracy) compares the sign of the predicted
    change against the sign of the actual change, using y_prev as the prior
    period actual.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.abs(y_true) + np.abs(y_pred)
    smape_vals = np.divide(
        2.0 * np.abs(y_true - y_pred),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom != 0,
    )
    smape = float(np.mean(smape_vals) * 100)

    mda = float("nan")
    if y_prev is not None and len(y_prev) == len(y_true):
        valid = ~np.isnan(y_prev)
        if valid.sum() > 0:
            actual_change = np.sign(y_true[valid] - y_prev[valid])
            pred_change = np.sign(y_pred[valid] - y_prev[valid])
            mda = float(np.mean(actual_change == pred_change))

    # MAPE: exclude zero/near-zero targets to avoid division-by-zero explosion
    # Threshold: 1% of mean target value
    mean_target = np.mean(y_true)
    nonzero_mask = y_true > (mean_target * 0.01)
    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
            * 100
        )
    else:
        mape = float("inf")

    return {
        "mae": round(mae, 4),
        "smape": round(smape, 4),
        "mda": round(mda, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "r2": round(r2, 4),
        "tier": tier_name,
    }


# ---------------------------------------------------------------------------
# Model Training Functions
# ---------------------------------------------------------------------------


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> tuple:
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def _pooled_abs_span(pooled: pd.Series) -> int:
    """Number of distinct absolute months spanned by a pooled series index.

    Measured on the grouped index (absolute month numbers) before reset, so a
    Jan-2024 and a Jan-2025 row remain separate and are NOT collapsed into a
    single "January" observation.
    """
    if pooled.empty:
        return 0
    return int(pooled.index.max() - pooled.index.min() + 1)


def forecast_arima_pool(hist: pd.DataFrame, target_row_months: list) -> tuple[dict, int]:
    """Fit a pooled, user-normalized ARIMA and forecast the test window.

    Each user's monthly expense series is normalized by that user's own
    trailing mean (users sit on a common ~1.0 scale), then averaged across
    users per absolute month into one robust pooled series. A low-order ARIMA
    is fit on that series and forecast forward over `target_row_months` in
    sequence.

    Returns ({month: pooled_scalar}, n_fits). The returned map is the
    scale-normalized pooled forecast path; the caller rescales each user by
    that user's own trailing mean (or the global mean for cold-start users),
    giving full test coverage.
    """
    n_fits = 0
    hist = hist[hist["target_expenses"].notna()].copy()
    if hist.empty:
        return {}, n_fits

    user_mean = hist.groupby("user_id")["target_expenses"].transform("mean").replace(0, np.nan)
    hist["norm"] = hist["target_expenses"] / user_mean

    pooled = hist.groupby("month_num")["norm"].mean().sort_index().reset_index(drop=True)
    if HAS_STATSMODELS and len(pooled) >= ARIMA_MIN_HISTORY:
        model = ARIMA(pooled, order=ARIMA_ORDER).fit(method="burg")
        n_fits = 1
        forecast_vals = model.forecast(len(target_row_months))
    else:
        forecast_vals = np.full(len(target_row_months), pooled.mean())

    target_order = sorted(target_row_months)
    return dict(zip(target_order, np.asarray(forecast_vals, dtype=float), strict=True)), n_fits


def forecast_sarima_pool(hist: pd.DataFrame, target_row_months: list) -> tuple[dict, int]:
    """SARIMA variant of the pooled user-normalized forecast.

    Same pool construction as :func:`forecast_arima_pool` but on absolute
    month indices. Uses a seasonal component only when the pooled series
    spans >= 24 distinct absolute months (a single calendar year spans 12,
    which degrades the seasonal order back to plain ARIMA). Returns
    ({month: scalar}, n_fits).
    """
    n_fits = 0
    hist = hist[hist["target_expenses"].notna()].copy()
    if hist.empty:
        return {}, n_fits

    user_mean = hist.groupby("user_id")["target_expenses"].transform("mean").replace(0, np.nan)
    hist["norm"] = hist["target_expenses"] / user_mean

    pooled = hist.groupby("month_num")["norm"].mean().sort_index()
    abs_span = _pooled_abs_span(pooled)
    pooled = pooled.reset_index(drop=True)
    if HAS_STATSMODELS and len(pooled) >= ARIMA_MIN_HISTORY:
        try:
            if abs_span >= 24:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                model = SARIMAX(pooled, order=ARIMA_ORDER, seasonal_order=(1, 0, 0, 12)).fit(
                    disp=False
                )
                n_fits = 1
            else:
                model = ARIMA(pooled, order=ARIMA_ORDER).fit(method="burg")
                n_fits = 1
            forecast_vals = model.forecast(len(target_row_months))
        except Exception:
            model = ARIMA(pooled, order=ARIMA_ORDER).fit(method="burg")
            forecast_vals = model.forecast(len(target_row_months))
    else:
        forecast_vals = np.full(len(target_row_months), pooled.mean())

    target_order = sorted(target_row_months)
    return dict(zip(target_order, np.asarray(forecast_vals, dtype=float), strict=True)), n_fits


# ---------------------------------------------------------------------------
# PyTorch Sequence Model (GRU)
# ---------------------------------------------------------------------------


def _train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 10,
    verbose: bool = False,
) -> nn.Module:
    """Train a PyTorch model with early stopping on validation loss."""
    device = torch.device("cpu")
    model = model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_t, y_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_sequence_model(input_shape: tuple, model_type: str = "gru") -> Any:
    """Build a sequence forecaster (GRU default; LSTM/BiLSTM supported)."""
    if not HAS_PYTORCH:
        return None
    seq_len, n_feat = input_shape
    return _SequenceForecaster(input_size=n_feat, hidden_size=32, model_type=model_type)


def train_sequence_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "gru",
) -> tuple:
    """Train a sequence forecaster (GRU default; LSTM/BiLSTM supported)."""
    if not HAS_PYTORCH:
        return None, np.zeros(len(y_test))

    model = build_sequence_model((X_train.shape[1], X_train.shape[2]), model_type)
    if model is None:
        return None, np.zeros(len(y_test))

    n_val = max(1, int(len(X_train) * 0.15))
    X_val, y_val = X_train[-n_val:], y_train[-n_val:]
    X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]

    model = _train_pytorch_model(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        epochs=50,
        batch_size=64,
        lr=0.001,
        patience=10,
    )

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_test_t).numpy()

    return model, y_pred


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------


def run_wfv(
    splits: dict,
    folds: list,
    feature_cols: list,
    run_name: str = "",
    run_gru: bool = True,
    run_rf: bool = True,
) -> dict:
    """Run walk-forward validation across all folds.

    Key design: aggregate ALL data to monthly once, then for each fold
    create train/test by filtering on months. Test samples use ALL prior
    months as lookback context (not just test-month data).
    """
    print(f"\n{'=' * 60}")
    print(f"Walk-Forward Validation: {run_name}")
    print(f"{'=' * 60}")

    # Aggregate ALL data to monthly once
    all_data = pd.concat(splits.values(), ignore_index=True)
    all_monthly = aggregate_to_monthly(all_data, feature_cols)
    print(f"Total monthly samples: {len(all_monthly)}")
    print(f"Unique personas: {all_monthly['user_id'].nunique()}")

    all_fold_results = []

    for fold_info in folds:
        fold_num = fold_info["fold"]
        train_months = fold_info["train_months"]
        test_months = fold_info["test_months"]

        print(f"\n--- Fold {fold_num}: Train {train_months} -> Test {test_months} ---")

        # For RF and LSTM: use ALL months up to test month for lookback context
        # Train on train_months only, predict on test_months
        # But the lookback window can use any months <= max(train_months)

        # Training set: months in train_months, with lookback from earlier months
        train_monthly = all_monthly[all_monthly["month_num"].isin(train_months)].copy()

        # Test set: target months in test_months, with lookback from all prior
        # months including embargo months (features only, never training labels)
        embargo_months = fold_info.get("embargo_months", [])
        context_months = sorted(set(train_months + embargo_months + test_months))
        test_context = all_monthly[all_monthly["month_num"].isin(context_months)].copy()

        if train_monthly.empty:
            print(f"  WARNING: Empty train for fold {fold_num}, skipping")
            continue

        print(
            f"  Train monthly: {len(train_monthly)} samples "
            f"({train_monthly['user_id'].nunique()} personas)"
        )
        print(
            f"  Test context: {len(test_context)} samples "
            f"({test_context['user_id'].nunique()} personas)"
        )

        fold_results = {
            "fold": fold_num,
            "train_months": train_months,
            "test_months": test_months,
            "n_train": len(train_monthly),
            "n_test": 0,
            "tier_results": {},
        }

        # --- Naive Baseline ---
        # Test samples target month T (rows in month T-1). Naive predicts the
        # mean of training targets. Prior actual for MDA = month T-2 expenses
        # (from test_context, not shift within test_actual which has 1 row/user).
        target_row_months = [m - 1 for m in test_months if m - 1 >= 1]
        test_actual = test_context[test_context["month_num"].isin(target_row_months)].copy()
        test_actual = test_actual.sort_values(["user_id", "month_num"])
        naive_pred = np.full(len(test_actual), train_monthly["target_expenses"].mean())
        prior_months = [m - 2 for m in test_months if m - 2 >= 1]
        prior_rows = test_context[test_context["month_num"].isin(prior_months)].set_index("user_id")[
            "target_expenses"
        ]
        naive_y_prev = test_actual["user_id"].map(prior_rows).values
        naive_metrics = compute_metrics(
            test_actual["target_expenses"].values,
            naive_pred,
            "naive_baseline",
            y_prev=naive_y_prev,
        )
        fold_results["tier_results"]["naive_baseline"] = naive_metrics
        fold_results["n_test"] = len(test_actual)
        print(
            f"  Naive: MAPE={naive_metrics['mape']:.2f}%, "
            f"SMAPE={naive_metrics['smape']:.2f}%, MDA={naive_metrics['mda']:.4f}"
        )

        # --- Tier 3a (variant): ARIMA (pooled user-normalized; one fit/fold) ---
        arima_hist = all_monthly[all_monthly["month_num"] <= max(train_months)]
        user_means = arima_hist.groupby("user_id")["target_expenses"].mean()
        global_mean = float(arima_hist["target_expenses"].mean())

        def _rescale_pooled(path_map):
            return np.array(
                [
                    path_map.get(int(month), 1.0) * float(user_means.get(uid, global_mean))
                    for uid, month in zip(test_actual["user_id"], test_actual["month_num"], strict=True)
                ]
            )

        arima_path, arima_n_fits = forecast_arima_pool(arima_hist, target_row_months)
        if arima_path:
            arima_pred = _rescale_pooled(arima_path)
            arima_metrics = compute_metrics(
                test_actual["target_expenses"].values,
                arima_pred,
                "tier3_arima",
                y_prev=naive_y_prev,
            )
            fold_results["tier_results"]["tier3_arima"] = arima_metrics
            print(
                f"  ARIMA: MAPE={arima_metrics['mape']:.2f}%, "
                f"SMAPE={arima_metrics['smape']:.2f}%, MDA={arima_metrics['mda']:.4f}, "
                f"R²={arima_metrics['r2']:.4f} (pool fits={arima_n_fits})"
            )
        else:
            print("  ARIMA: Skipped (empty pooled series)")

        # --- Tier 3a (variant): SARIMA (seasonal only when pool >= 24 mo) ---
        sarima_path, sarima_n_fits = forecast_sarima_pool(arima_hist, target_row_months)
        if sarima_path:
            sarima_pred = _rescale_pooled(sarima_path)
            sarima_metrics = compute_metrics(
                test_actual["target_expenses"].values,
                sarima_pred,
                "tier3_sarima",
                y_prev=naive_y_prev,
            )
            fold_results["tier_results"]["tier3_sarima"] = sarima_metrics
            print(
                f"  SARIMA: MAPE={sarima_metrics['mape']:.2f}%, "
                f"SMAPE={sarima_metrics['smape']:.2f}%, MDA={sarima_metrics['mda']:.4f}, "
                f"R²={sarima_metrics['r2']:.4f} (pool fits={sarima_n_fits})"
            )
        else:
            print("  SARIMA: Skipped (empty pooled series)")

        # --- Tier 2: Random Forest ---
        # Train on train months, test on test months (with lookback context)
        if not run_rf:
            print("  RF: Skipped (run_rf=False; CPU-constrained default)")
        else:
            X_train_rf, y_train_rf, _, _ = prepare_flat_features(train_monthly, feature_cols)
            X_test_rf, y_test_rf, meta_rf, y_prev_rf = prepare_flat_features(
                test_context, feature_cols, filter_months=test_months
            )

            if len(X_train_rf) > 0 and len(X_test_rf) > 0:
                scaler_rf = StandardScaler()
                X_train_rf_s = scaler_rf.fit_transform(X_train_rf)
                X_test_rf_s = scaler_rf.transform(X_test_rf)

                rf_model, rf_pred = train_rf(X_train_rf_s, y_train_rf, X_test_rf_s, y_test_rf)
                rf_metrics = compute_metrics(
                    y_test_rf, rf_pred, "tier2_random_forest", y_prev=y_prev_rf
                )
                fold_results["tier_results"]["tier2_random_forest"] = rf_metrics
                print(
                    f"  RF: MAPE={rf_metrics['mape']:.2f}%, "
                    f"SMAPE={rf_metrics['smape']:.2f}%, MDA={rf_metrics['mda']:.4f}, "
                    f"R²={rf_metrics['r2']:.4f}"
                )
            else:
                print(f"  RF: Skipped (train={len(X_train_rf)}, test={len(X_test_rf)})")

        # --- Tier 3: GRU (top-3-lightest scope; LSTM/BiLSTM are "hold") ---
        if HAS_PYTORCH and run_gru:
            X_train_seq, y_train_seq, _, _ = prepare_monthly_sequences(
                train_monthly, feature_cols, lookback=3
            )
            X_test_seq, y_test_seq, meta_seq, y_prev_seq = prepare_monthly_sequences(
                test_context, feature_cols, lookback=3, filter_months=test_months
            )

            if len(X_train_seq) > 0 and len(X_test_seq) > 0:
                scaler_seq = StandardScaler()
                n_train, seq_len, n_feat = X_train_seq.shape
                X_train_flat = X_train_seq.reshape(-1, n_feat)
                X_train_flat_s = scaler_seq.fit_transform(X_train_flat)
                X_train_seq_s = X_train_flat_s.reshape(n_train, seq_len, n_feat)

                n_test = X_test_seq.shape[0]
                X_test_flat = X_test_seq.reshape(-1, n_feat)
                X_test_flat_s = scaler_seq.transform(X_test_flat)
                X_test_seq_s = X_test_flat_s.reshape(n_test, seq_len, n_feat)

                for variant in ["gru"]:
                    tier_name = f"tier3_{variant}"
                    model, pred = train_sequence_variant(
                        X_train_seq_s,
                        y_train_seq,
                        X_test_seq_s,
                        y_test_seq,
                        model_type=variant,
                    )
                    metrics = compute_metrics(y_test_seq, pred, tier_name, y_prev=y_prev_seq)
                    fold_results["tier_results"][tier_name] = metrics
                    print(
                        f"  {variant.upper()}: MAPE={metrics['mape']:.2f}%, "
                        f"SMAPE={metrics['smape']:.2f}%, "
                        f"MDA={metrics['mda']:.4f}, R²={metrics['r2']:.4f}"
                    )
            else:
                print(f"  GRU: Skipped (train={len(X_train_seq)}, test={len(X_test_seq)})")
        elif HAS_PYTORCH and not run_gru:
            print("  GRU: Skipped (--skip-torch; run on a GPU-capable teammate machine)")
        else:
            print("  GRU: Skipped (pytorch not installed)")

        all_fold_results.append(fold_results)

    return {"folds": all_fold_results}


# ---------------------------------------------------------------------------
# Aggregate & Decision Rule
# ---------------------------------------------------------------------------


def aggregate_metrics(fold_results: list) -> dict:
    """Aggregate metrics across folds."""
    tier_names = set()
    for fr in fold_results:
        tier_names.update(fr["tier_results"].keys())

    aggregate = {}
    for tier in tier_names:
        vals = {
            k: [fr["tier_results"][tier][k] for fr in fold_results if tier in fr["tier_results"]]
            for k in ("mae", "smape", "mda", "rmse", "mape", "r2")
        }

        if vals["mape"]:

            def _agg(key, fmt=round):
                arr = np.array(
                    [
                        v
                        for v in vals[key]
                        if v is not None and not (isinstance(v, float) and np.isnan(v))
                    ],
                    dtype=float,
                )
                if len(arr) == 0:
                    return None, None
                return fmt(float(np.mean(arr)), 4), fmt(float(np.std(arr)), 4)

            mae_mean, mae_std = _agg("mae")
            smape_mean, smape_std = _agg("smape")
            mda_mean, mda_std = _agg("mda")
            rmse_mean, rmse_std = _agg("rmse")
            mape_mean, mape_std = _agg("mape")
            r2_mean, r2_std = _agg("r2")

            aggregate[tier] = {
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "smape_mean": smape_mean,
                "smape_std": smape_std,
                "mda_mean": mda_mean,
                "mda_std": mda_std,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "mape_mean": mape_mean,
                "mape_std": mape_std,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "n_folds": len(vals["mape"]),
            }
    return aggregate


def _winner_artifact_name(winner: str) -> str:
    """Primary loadable artifact for the serving registry (winner_artifact)."""
    if winner == "naive_baseline":
        return ""
    if winner in ("tier2_random_forest", "tier3_arima", "tier3_sarima"):
        return f"{winner}.joblib"
    if winner.startswith("tier3_"):
        return f"{winner}.pth"
    return ""


def apply_decision_rule(aggregate: dict, naive_mape: float) -> tuple:
    """Apply the pre-registered MAPE-reduction decision rule."""
    best_tier = None
    best_mape = float("inf")

    for tier, metrics in aggregate.items():
        if tier == "naive_baseline":
            continue
        mape_mean = metrics.get("mape_mean")
        if mape_mean is None or not np.isfinite(mape_mean):
            continue
        if mape_mean < best_mape:
            best_mape = mape_mean
            best_tier = tier

    if best_tier is None or naive_mape <= 0:
        return "naive_baseline", "No learned model available"

    mape_reduction = 1.0 - (best_mape / naive_mape)
    if mape_reduction >= PRE_REGISTERED_MAPE_REDUCTION:
        reason = (
            f"{best_tier} reduces MAPE by {mape_reduction * 100:.1f}% "
            f"(>{PRE_REGISTERED_MAPE_REDUCTION * 100:.0f}% threshold)"
        )
        return best_tier, reason
    else:
        reason = (
            f"Best model {best_tier} reduces MAPE by only "
            f"{mape_reduction * 100:.1f}% (<{PRE_REGISTERED_MAPE_REDUCTION * 100:.0f}% "
            f"threshold). Naive baseline preferred for simplicity."
        )
        return "naive_baseline", reason


# ---------------------------------------------------------------------------
# Model Saving
# ---------------------------------------------------------------------------


def save_models(splits: dict, feature_cols: list, output_dir: Path, winner: str):
    """Re-train winner on full data and save artifacts."""
    print(f"\nRe-training {winner} on full data for final model...")

    all_data = pd.concat(splits.values(), ignore_index=True)
    monthly = aggregate_to_monthly(all_data, feature_cols)

    scaler = StandardScaler()

    if winner == "tier2_random_forest":
        X, y, _, _ = prepare_flat_features(monthly, feature_cols)
        X_s = scaler.fit_transform(X)
        model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_s, y)
        joblib.dump(
            {"model": model, "scaler": scaler, "feature_cols": feature_cols},
            output_dir / "tier2_random_forest.joblib",
        )
        print("  Saved tier2_random_forest.joblib")

    elif winner == "tier3_arima":
        # Refit the pooled ARIMA on the full data for serving. Per-user level
        # scaling at serve time multiplies the normalized pooled forecast by
        # the user's own trailing mean; store `pool_level` (the pooled series
        # mean used as the normalization anchor) for exact rescaling.
        hist = monthly[monthly["target_expenses"].notna()].copy()
        user_mean = hist.groupby("user_id")["target_expenses"].transform("mean").replace(0, np.nan)
        hist["norm"] = hist["target_expenses"] / user_mean
        pooled = hist.groupby("month_num")["norm"].mean().sort_index().reset_index(drop=True)
        pool_level = float(pooled.mean()) if not pooled.empty else 1.0
        profile_level = float(hist["target_expenses"].mean()) if not hist.empty else pool_level
        if HAS_STATSMODELS and len(pooled) >= ARIMA_MIN_HISTORY:
            model = ARIMA(pooled, order=ARIMA_ORDER).fit(method="burg")
        else:
            model = None
        joblib.dump(
            {
                "kind": "arima",
                "model": model,
                "pool_level": pool_level,
                "profile_level": profile_level,
                "feature_cols": feature_cols,
            },
            output_dir / "tier3_arima.joblib",
        )
        print("  Saved tier3_arima.joblib")

    elif winner == "tier3_sarima":
        # SARIMA variant: seasonal order only when the pooled series spans
        # 24+ distinct absolute months; otherwise identical to the plain ARIMA
        # artifact. absolute-span pooling keeps two Januaries from different
        # years as separate observations.
        hist = monthly[monthly["target_expenses"].notna()].copy()
        user_mean = hist.groupby("user_id")["target_expenses"].transform("mean").replace(0, np.nan)
        hist["norm"] = hist["target_expenses"] / user_mean
        pooled = hist.groupby("month_num")["norm"].mean().sort_index()
        abs_span = _pooled_abs_span(pooled)
        pooled = pooled.reset_index(drop=True)
        pool_level = float(pooled.mean()) if not pooled.empty else 1.0
        profile_level = float(hist["target_expenses"].mean()) if not hist.empty else pool_level
        model = None
        if HAS_STATSMODELS and len(pooled) >= ARIMA_MIN_HISTORY:
            try:
                if abs_span >= 24:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX

                    model = SARIMAX(pooled, order=ARIMA_ORDER, seasonal_order=(1, 0, 0, 12)).fit(
                        disp=False
                    )
                else:
                    model = ARIMA(pooled, order=ARIMA_ORDER).fit(method="burg")
            except Exception:
                model = ARIMA(pooled, order=ARIMA_ORDER).fit(method="burg")
        joblib.dump(
            {
                "kind": "sarima",
                "model": model,
                "pool_level": pool_level,
                "profile_level": profile_level,
                "feature_cols": feature_cols,
            },
            output_dir / "tier3_sarima.joblib",
        )
        print("  Saved tier3_sarima.joblib")

    elif winner.startswith("tier3_") and HAS_PYTORCH:
        variant = winner.replace("tier3_", "")
        X, y, _, _ = prepare_monthly_sequences(monthly, feature_cols, lookback=3)
        if len(X) > 0:
            n, seq_len, n_feat = X.shape
            X_flat = X.reshape(-1, n_feat)
            X_flat_s = scaler.fit_transform(X_flat)
            X_s = X_flat_s.reshape(n, seq_len, n_feat)

            model = build_sequence_model((seq_len, n_feat), variant)
            X_t = torch.tensor(X_s, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            n_val = max(1, int(len(X_t) * 0.15))
            model = _train_pytorch_model(
                model,
                X_t[:-n_val].numpy(),
                y_t[:-n_val].numpy(),
                X_t[-n_val:].numpy(),
                y_t[-n_val:].numpy(),
                epochs=50,
                batch_size=32,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_type": variant,
                    "input_size": n_feat,
                    "hidden_size": 32,
                    "seq_length": seq_len,
                },
                output_dir / f"{winner}.pth",
            )
            joblib.dump(
                {"scaler": scaler, "feature_cols": feature_cols},
                output_dir / f"{winner}_meta.joblib",
            )
            print(f"  Saved {winner}.pth + meta.joblib")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_training(config: dict) -> TrainingReport:
    start_time = time.time()
    report = TrainingReport(timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"))

    input_dir = config["input"]
    output_dir = Path(config["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading forecaster data...")
    splits = load_forecaster_data(input_dir)
    feature_meta = load_feature_columns(input_dir)
    feature_cols = feature_meta.get("feature_columns", FORECASTER_FEATURES)

    folds_path = Path(config.get("folds", "training/datasets/processed/temporal_folds.json"))
    folds = load_temporal_folds(str(folds_path))
    report.n_folds = len(folds)

    data_sources = [Path(input_dir) / f"{name}.parquet" for name in ("train", "val", "test")]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Folds: {len(folds)}")

    # Run WFV
    print("\n[2/5] Running walk-forward validation...")
    run_gru = config.get("run_gru", True)
    run_rf = config.get("run_rf", True)
    wfv_results = run_wfv(splits, folds, feature_cols, run_gru=run_gru, run_rf=run_rf)
    report.fold_results = wfv_results["folds"]

    # Aggregate
    print("\n[3/5] Aggregating metrics...")
    report.aggregate_metrics = aggregate_metrics(report.fold_results)

    naive_mape = report.aggregate_metrics.get("naive_baseline", {}).get("mape_mean") or 0
    report.naive_mape = naive_mape

    # Decision rule
    print("\n[4/5] Applying decision rule...")
    winner, reason = apply_decision_rule(report.aggregate_metrics, naive_mape)
    report.winner = winner
    report.winner_reason = reason
    print(f"  Winner: {winner}")
    print(f"  Reason: {reason}")

    # Save models
    print("\n[5/5] Saving models and reports...")
    save_models(splits, feature_cols, output_dir, winner)

    # Save evaluation JSON
    eval_json = {
        "timestamp": report.timestamp,
        "n_folds": report.n_folds,
        "pre_registered_mape_reduction": PRE_REGISTERED_MAPE_REDUCTION,
        "naive_mape": naive_mape,
        "feature_columns": feature_cols,
        "aggregate_metrics": report.aggregate_metrics,
        "winner": report.winner,
        "winner_artifact": _winner_artifact_name(report.winner),
        "winner_reason": report.winner_reason,
        "fold_details": report.fold_results,
    }
    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(eval_json, f, indent=2)

    # Write report
    write_evaluation_report(output_dir, **family_report("forecaster", eval_json))

    # Emit metadata.json (Phase 8 provenance: hash, commit, metrics, rule)
    metadata = build_metadata(**family_metadata("forecaster", eval_json, data_sources=data_sources))
    write_metadata(metadata, output_dir)

    elapsed = time.time() - start_time
    print(f"\nDone. Winner: {winner}, {elapsed:.1f}s")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Pipeline for the Spending Forecaster (ARIMA + RF + GRU)"
    )
    parser.add_argument(
        "--input",
        default="training/datasets/forecaster/",
        help="Input directory with forecaster features",
    )
    parser.add_argument(
        "--output", default="models/forecaster/", help="Output directory for models"
    )
    parser.add_argument(
        "--folds",
        default="training/datasets/processed/temporal_folds.json",
        help="Path to temporal_folds.json",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip CPU-bound PyTorch (GRU) tiers. Use on CPU-limited "
        "machines; run GRU on a GPU-capable teammate machine instead.",
    )
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Skip the CPU-bound Random Forest tier (keep ARIMA + "
        "baseline). Use when ARIMA clearly wins and RF runtime "
        "is prohibitive on a constrained machine.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "input": args.input,
        "output": args.output,
        "folds": args.folds,
        "run_gru": not args.skip_torch,
        "run_rf": not args.skip_rf,
    }
    run_training(config)
