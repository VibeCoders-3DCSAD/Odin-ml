"""
Feature Engineering Pipeline for LSTM Spending Forecaster

Reads raw transactions and monthly summaries from synth/ and produces
daily-level feature matrices for time-series forecasting.

Features (20 total per persona-day):
  - Temporal encoding: day-of-week sin/cos, day-of_month
  - Lag features: 1d, 7d, 14d, 15d, 30d, 60d expense lags
  - Rolling statistics: 7d, 14d, 30d mean and std of expenses
  - Calendar features: is_payday, days_to_payday
  - RFM features: recency, frequency_30d, monetary_30d

Usage:
    python scripts/feature_engineering_forecaster.py \
        --transactions synth/transactions.parquet \
        --summaries synth/monthly_summaries.parquet \
        --splits datasets/processed/split_metadata.json \
        --output datasets/forecaster/

Design principles:
  - Features computed incrementally (no future data leakage)
  - Imputation fit on train only, applied to val/test
  - Persona-level splitting preserved from preprocessor.py
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECASTER_FEATURES = [
    # Temporal encoding (3)
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month",
    # Lag features (6)
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_15d",
    "lag_30d",
    "lag_60d",
    # Rolling statistics (6)
    "rolling_mean_7d",
    "rolling_std_7d",
    "rolling_mean_14d",
    "rolling_std_14d",
    "rolling_mean_30d",
    "rolling_std_30d",
    # Calendar features (2)
    "is_payday",
    "days_to_payday",
    # RFM features (3)
    "recency",
    "frequency_30d",
    "monetary_30d",
]

META_COLUMNS = [
    "user_id",
    "date",
    "month",
    "year",
    "target_expenses",
    "has_transaction",
]

RAW_EXPENSE_COLUMNS = [
    "food_expense",
    "housing_expense",
    "transport_expense",
    "health_expense",
    "education_expense",
    "other_expense",
]


@dataclass
class ForecasterEngineeringConfig:
    input_transactions: str = "synth/transactions.parquet"
    input_summaries: str = "synth/monthly_summaries.parquet"
    input_splits: str = "datasets/processed/split_metadata.json"
    output_dir: str = "datasets/forecaster/"
    seed: int = 42


@dataclass
class ForecasterEngineeringReport:
    timestamp: str = ""
    n_personas: int = 0
    n_features: int = 0
    n_rows: dict = field(default_factory=dict)
    feature_stats: dict = field(default_factory=dict)
    imputation_values: dict = field(default_factory=dict)
    split_sizes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_splits(splits_path: str) -> dict:
    with open(splits_path) as f:
        meta = json.load(f)
    return meta.get("personas", {})


def load_transactions(transactions_path: str) -> pd.DataFrame:
    df = pd.read_parquet(transactions_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_summaries(summaries_path: str) -> pd.DataFrame:
    return pd.read_parquet(summaries_path)


# ---------------------------------------------------------------------------
# Daily Grid Construction
# ---------------------------------------------------------------------------

def build_daily_grid(persona_id: str, year: int = 2023) -> pd.DataFrame:
    """Create a full daily grid for one persona across all 12 months."""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    dates = pd.date_range(start, end, freq="D")
    grid = pd.DataFrame({
        "persona_id": persona_id,
        "date": dates,
        "month": dates.month,
        "year": dates.year,
        # Absolute month index since the grid epoch (2023-01 = 1); keeps two
        # Januaries from different years distinct in pooling/ordering.
        "month_num": (year - 2023) * 12 + dates.month,
        "day_of_week": dates.dayofweek,
        "day_of_month": dates.day,
    })
    return grid


def aggregate_daily_expenses(persona_txns: pd.DataFrame, persona_id: str) -> pd.DataFrame:
    """Aggregate expense transactions to daily level for one persona."""
    mask = persona_txns["transaction_type"] == "expense"
    persona_txns = persona_txns[mask].copy()
    if persona_txns.empty:
        return pd.DataFrame(columns=["date", "daily_expense", "txn_count"])

    daily = persona_txns.groupby("date").agg(
        daily_expense=("amount", "sum"),
        txn_count=("transaction_id", "count"),
    ).reset_index()
    return daily


# ---------------------------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------------------------

def compute_temporal_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week sin/cos encoding and normalized day-of-month."""
    dow = grid["day_of_week"].values
    grid["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    grid["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
    grid["day_of_month"] = grid["day_of_month"].values / 31.0
    return grid


def compute_lag_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Lag features: expense amount at various lookback periods."""
    exp = grid["daily_expense"].values
    n = len(exp)

    for lag_name, lag_days in [("lag_1d", 1), ("lag_7d", 7), ("lag_14d", 14),
                                ("lag_15d", 15), ("lag_30d", 30), ("lag_60d", 60)]:
        lagged = np.full(n, np.nan)
        if n > lag_days:
            lagged[lag_days:] = exp[:n - lag_days]
        grid[lag_name] = lagged
    return grid


def compute_rolling_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std over 7, 14, 30 day windows."""
    exp = grid["daily_expense"]

    for window in [7, 14, 30]:
        roll_mean = exp.rolling(window=window, min_periods=1).mean()
        roll_std = exp.rolling(window=window, min_periods=2).std()
        grid[f"rolling_mean_{window}d"] = roll_mean
        grid[f"rolling_std_{window}d"] = roll_std.fillna(0.0)
    return grid


def compute_calendar_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Payday indicators and days-to-payday."""
    dom = grid["day_of_month"].values * 31.0  # un-normalize
    month = grid["month"].values
    grid["is_payday"] = (((dom >= 15) & (dom <= 16)) | ((dom >= 29) & (dom <= 31))).astype(float)

    # Days to next payday (15th or last day of month)
    days_in_month = np.where(np.isin(month, [4, 6, 9, 11]), 30, np.where(month == 2, 28, 31))
    days_to_payday = np.where(dom <= 15, 15 - dom, np.maximum(0, days_in_month - dom))
    grid["days_to_payday"] = days_to_payday / 30.0  # normalize
    return grid


def compute_rfm_features(grid: pd.DataFrame, txn_dates: np.ndarray,
                          txn_amounts: np.ndarray) -> pd.DataFrame:
    """RFM: recency, frequency_30d, monetary_30d (vectorized)."""
    n = len(grid)
    dates = grid["date"].values

    recency = np.full(n, 365.0)
    freq_30d = np.zeros(n)
    mon_30d = np.zeros(n)

    if len(txn_dates) > 0:
        t = np.asarray(txn_dates, dtype="datetime64[ns]")
        # Recency: days since last transaction on or before each date
        last_idx = np.searchsorted(t, dates, side="right") - 1
        valid = last_idx >= 0
        if valid.any():
            recency[valid] = (dates[valid] - t[last_idx[valid]]).astype("timedelta64[D]").astype(float)

        # Frequency and monetary: count and mean in last 30 days
        left = np.searchsorted(t, dates - np.timedelta64(30, "D"), side="left")
        right = np.searchsorted(t, dates, side="right")
        counts = (right - left).astype(float)
        freq_30d = counts
        cs = np.concatenate([[0.0], np.cumsum(txn_amounts)])
        valid_c = counts > 0
        sums = np.zeros(n)
        sums[valid_c] = cs[right[valid_c]] - cs[left[valid_c]]
        mon_30d = np.where(valid_c, sums / np.where(valid_c, counts, 1.0), 0.0)

    grid["recency"] = recency
    grid["frequency_30d"] = freq_30d
    grid["monetary_30d"] = mon_30d
    return grid


def compute_monthly_target(summaries: pd.DataFrame, persona_id: str) -> pd.Series:
    """Get monthly total expenses as target variable."""
    mask = summaries["persona_id"] == persona_id
    persona_summaries = summaries[mask].sort_values("month")
    target = persona_summaries.set_index("month")["total_expenses"]
    return target


# ---------------------------------------------------------------------------
# Per-Persona Pipeline
# ---------------------------------------------------------------------------

def process_persona(persona_id: str, persona_txns: pd.DataFrame,
                     summaries: pd.DataFrame, train_imputation: Optional[dict] = None,
                     is_train: bool = True) -> pd.DataFrame:
    """Process one persona: build daily grid, compute all features, return DataFrame."""
    # persona_txns is already filtered to this persona; skip if no expense data
    if persona_txns.empty:
        return pd.DataFrame()

    # Build daily grid
    grid = build_daily_grid(persona_id)

    # Aggregate daily expenses
    daily_exp = aggregate_daily_expenses(persona_txns, persona_id)

    # Merge daily expenses into grid
    grid = grid.merge(daily_exp, on="date", how="left")
    grid["daily_expense"] = grid["daily_expense"].fillna(0.0)
    grid["txn_count"] = grid["txn_count"].fillna(0).astype(int)
    grid["has_transaction"] = (grid["txn_count"] > 0).astype(float)

    # Transaction-only data for RFM
    expense_mask = persona_txns["transaction_type"] == "expense"
    txn_dates = persona_txns.loc[expense_mask, "date"].sort_values().values
    txn_amounts = persona_txns.loc[expense_mask, "amount"].values

    # Compute features
    grid = compute_temporal_features(grid)
    grid = compute_lag_features(grid)
    grid = compute_rolling_features(grid)
    grid = compute_calendar_features(grid)

    if len(txn_dates) > 0:
        grid = compute_rfm_features(grid, txn_dates, txn_amounts)
    else:
        grid["recency"] = 365.0
        grid["frequency_30d"] = 0.0
        grid["monetary_30d"] = 0.0

    # Target: next month total expenses (NaN when no next month exists).
    # Keyed by absolute month index (year-aware): Jan-2024 and Jan-2025 stay
    # separate rows in pooling instead of collapsing into one "January".
    persona_summ = summaries.sort_values(["year", "month"]) if not summaries.empty else pd.DataFrame()
    target_map = {}
    if not persona_summ.empty:
        target_map = {
            (int(y) - 2023) * 12 + int(m): total
            for y, m, total in zip(
                persona_summ["year"].values,
                persona_summ["month"].values,
                persona_summ["total_expenses"].values,
                strict=True,
            )
        }
    grid["target_expenses"] = grid["month_num"].map(
        lambda mn: target_map.get(int(mn) + 1, np.nan)
    )

    # Add metadata
    grid["user_id"] = persona_id

    return grid


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def compute_train_imputation(train_dfs: list[pd.DataFrame]) -> dict:
    """Compute imputation values from training data only."""
    all_train = pd.concat(train_dfs, ignore_index=True)
    imputation = {}
    for col in FORECASTER_FEATURES:
        if col in all_train.columns:
            imputation[col] = float(all_train[col].median())
    return imputation


def apply_imputation(df: pd.DataFrame, imputation: dict) -> pd.DataFrame:
    """Apply imputation values to a DataFrame."""
    for col, value in imputation.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: ForecasterEngineeringConfig) -> ForecasterEngineeringReport:
    start_time = time.time()
    report = ForecasterEngineeringReport(timestamp=datetime.now().isoformat())

    print("[1/6] Loading data...")
    transactions = load_transactions(config.input_transactions)
    summaries = load_summaries(config.input_summaries)
    persona_splits = load_splits(config.input_splits)

    train_ids = persona_splits.get("train", [])
    val_ids = persona_splits.get("val", [])
    test_ids = persona_splits.get("test", [])

    report.n_personas = len(train_ids) + len(val_ids) + len(test_ids)
    report.split_sizes = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}

    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Pre-partition transactions/summaries by persona once (avoids full-df scans per persona)
    groups = {pid: g for pid, g in transactions.groupby("persona_id")}
    summary_groups = {pid: g for pid, g in summaries.groupby("persona_id")}
    empty_txn = pd.DataFrame(columns=transactions.columns)

    # Phase 1: Process train split, compute imputation, export, free memory
    print(f"\n[2a] Processing train split ({len(train_ids)} personas)...")
    train_dfs = []
    for i, pid in enumerate(train_ids):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Train persona {i+1}/{len(train_ids)}: {pid} ({time.time()-start_time:.0f}s)")
        persona_txns = groups.get(pid, empty_txn)
        persona_summ = summary_groups.get(pid, pd.DataFrame())
        df = process_persona(pid, persona_txns, persona_summ)
        if not df.empty:
            train_dfs.append(df)

    print("\n[2b] Computing imputation from train split...")
    train_imputation = compute_train_imputation(train_dfs)
    report.imputation_values = {k: round(v, 6) for k, v in train_imputation.items()}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export train split
    print("[2c] Exporting train split...")
    combined = pd.concat(train_dfs, ignore_index=True)
    del train_dfs
    combined = apply_imputation(combined, train_imputation)
    combined["has_target"] = (~combined["target_expenses"].isna()).astype(float)
    output_cols = META_COLUMNS + FORECASTER_FEATURES + ["has_target"]
    combined = combined[[c for c in output_cols if c in combined.columns]]
    report.n_rows["train"] = len(combined)
    combined.to_parquet(output_dir / "train.parquet", index=False)
    print(f"  Exported {len(combined)} train rows")

    # Feature stats from train
    print("[2d] Computing feature statistics...")
    for feat in FORECASTER_FEATURES:
        if feat in combined.columns:
            report.feature_stats[feat] = {
                "mean": round(float(combined[feat].mean()), 4),
                "std": round(float(combined[feat].std()), 4),
                "min": round(float(combined[feat].min()), 4),
                "max": round(float(combined[feat].max()), 4),
            }
    report.n_features = len(FORECASTER_FEATURES)
    del combined

    # Phase 2-3: Process val and test splits one at a time
    for split_name, persona_ids in [("val", val_ids), ("test", test_ids)]:
        print(f"\n[3] Processing {split_name} split ({len(persona_ids)} personas)...")
        dfs = []
        for i, pid in enumerate(persona_ids):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  {split_name} persona {i+1}/{len(persona_ids)}: {pid}")
            persona_txns = groups.get(pid, empty_txn)
            persona_summ = summary_groups.get(pid, pd.DataFrame())
            df = process_persona(pid, persona_txns, persona_summ)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            print(f"  WARNING: No data for {split_name} split")
            continue

        print(f"[4] Exporting {split_name} split...")
        combined = pd.concat(dfs, ignore_index=True)
        del dfs
        combined = apply_imputation(combined, train_imputation)
        combined["has_target"] = (~combined["target_expenses"].isna()).astype(float)
        combined = combined[[c for c in output_cols if c in combined.columns]]
        report.n_rows[split_name] = len(combined)
        combined.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        print(f"  Exported {len(combined)} {split_name} rows")
        del combined

    # Export metadata
    print("\n[5] Exporting metadata...")
    metadata = {
        "timestamp": report.timestamp,
        "n_personas": report.n_personas,
        "n_features": report.n_features,
        "feature_columns": FORECASTER_FEATURES,
        "meta_columns": META_COLUMNS,
        "raw_expense_columns": RAW_EXPENSE_COLUMNS,
        "split_sizes": report.split_sizes,
        "n_rows": report.n_rows,
        "feature_stats": report.feature_stats,
        "imputation_values": report.imputation_values,
    }
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(metadata, f, indent=2)

    pipeline_report = {
        "timestamp": report.timestamp,
        "script": "feature_engineering_forecaster.py",
        "n_personas": report.n_personas,
        "n_features": report.n_features,
        "split_sizes": report.split_sizes,
        "n_rows": report.n_rows,
        "duration_seconds": round(time.time() - start_time, 2),
    }
    with open(output_dir / "pipeline_report.json", "w") as f:
        json.dump(pipeline_report, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone. {report.n_features} features, {sum(report.n_rows.values())} total rows, {elapsed:.1f}s")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature Engineering for LSTM Spending Forecaster"
    )
    parser.add_argument("--transactions", default="synth/transactions.parquet",
                        help="Path to transactions.parquet")
    parser.add_argument("--summaries", default="synth/monthly_summaries.parquet",
                        help="Path to monthly_summaries.parquet")
    parser.add_argument("--splits", default="datasets/processed/split_metadata.json",
                        help="Path to split_metadata.json")
    parser.add_argument("--output", default="datasets/forecaster/",
                        help="Output directory for forecaster features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ForecasterEngineeringConfig(
        input_transactions=args.transactions,
        input_summaries=args.summaries,
        input_splits=args.splits,
        output_dir=args.output,
        seed=args.seed,
    )
    run_pipeline(config)
