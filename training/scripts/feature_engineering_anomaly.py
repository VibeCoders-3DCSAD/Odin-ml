"""
Feature Engineering Pipeline for Anomaly Detector

Vectorized implementation. Reads raw transactions from synth/ and produces
per-transaction feature matrices for anomaly detection.

Features (24 per transaction):
  Baseline (10): mean/std income/expenses, category_dist, txn_frequency_rolling,
                 avg_txn_size, category_entropy, volatility_index, spending_concentration
  Detection (14): amount_deviation, category_deviation, frequency_deviation,
                  income_deviation, expense_deviation,
                  is_novel_category, amount_vs_category_mean, amount_vs_category_std,
                  category_frequency_change, amount_percentile_in_category,
                  days_since_last_txn, is_weekend, amount_zscore_overall, amount_zscore_category

Usage:
    python scripts/feature_engineering_anomaly.py
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPENSE_CATS = ["food", "housing", "transport", "health", "education", "other"]
ALL_CATS = EXPENSE_CATS + ["luxury", "gambling", "investment", "income"]

META_COLUMNS = [
    "user_id", "transaction_id", "year_month", "month", "date", "category",
    "amount", "transaction_type", "is_anomalous", "anomaly_type",
]


def load_data(transactions_path, splits_path):
    txn = pd.read_parquet(transactions_path)
    txn["date"] = pd.to_datetime(txn["date"])
    with open(splits_path) as f:
        splits = json.load(f).get("personas", {})
    return txn, splits


def build_category_history(txn):
    """Precompute per-persona category and income history for baseline computation."""
    history = {}

    for pid, group in txn.groupby("persona_id"):
        expense_mask = group["transaction_type"] == "expense"
        income_mask = group["transaction_type"] == "income"
        exp_group = group[expense_mask].sort_values("date")
        inc_group = group[income_mask].sort_values("date")

        cat_monthly = exp_group.groupby(["year_month", "category"]).size().unstack(fill_value=0)

        cat_amounts = {}
        cat_months = {}
        for cat, cat_group in exp_group.groupby("category"):
            cat_amounts[cat] = cat_group["amount"].values
            cat_months[cat] = cat_group["year_month"].values

        # Income by month
        inc_monthly = inc_group.groupby("year_month")["amount"].sum() if len(inc_group) > 0 else pd.Series(dtype=float)
        inc_amounts = inc_group["amount"].values if len(inc_group) > 0 else np.array([])

        # Transaction count by month
        all_exp = exp_group.copy()
        txn_count_by_month = all_exp.groupby("year_month").size() if len(all_exp) > 0 else pd.Series(dtype=int)

        # Expense totals by month (for monthly-level deviation)
        monthly_expense_totals = all_exp.groupby("year_month")["amount"].sum() if len(all_exp) > 0 else pd.Series(dtype=float)

        history[pid] = {
            "months": sorted(group["year_month"].unique()),
            "cat_monthly_counts": cat_monthly,
            "cat_amounts": cat_amounts,
            "cat_months": cat_months,
            "all_amounts": exp_group["amount"].values,
            "all_months": exp_group["year_month"].values,
            "all_dates": exp_group["date"].values,
            "inc_monthly": inc_monthly,
            "inc_amounts": inc_amounts,
            "txn_count_by_month": txn_count_by_month,
            "monthly_expense_totals": monthly_expense_totals,
        }
    return history


def compute_features_for_persona(pid, txn_row, history, baseline_months=3):
    """Compute features for a single transaction."""
    if pid not in history:
        return {}

    h = history[pid]
    current_month = str(txn_row["year_month"])
    current_cat = txn_row["category"]
    current_amount = float(txn_row["amount"])

    # Baseline months
    current_index = h["months"].index(current_month)
    bl_months = h["months"][max(0, current_index - baseline_months):current_index]

    if not bl_months:
        return _zero_features()

    # Category counts in baseline
    cat_monthly = h["cat_monthly_counts"]
    bl_cats = cat_monthly.loc[cat_monthly.index.isin(bl_months, level=0)]
    if isinstance(bl_cats, pd.Series):
        bl_cats = bl_cats.to_frame()

    # Category proportions
    cat_sums = bl_cats.sum()
    total_exp = cat_sums.sum()
    cat_props = cat_sums / total_exp if total_exp > 0 else cat_sums * 0

    # Baseline amounts
    bl_month_set = set(bl_months)
    bl_mask = np.array([m in bl_month_set for m in h["all_months"]])
    bl_amounts = h["all_amounts"][bl_mask]

    # Overall stats
    mean_expenses = float(bl_amounts.mean()) if len(bl_amounts) > 0 else 0.0
    std_expenses = float(bl_amounts.std()) if len(bl_amounts) > 1 else 0.0

    # Income stats from baseline
    inc_monthly = h["inc_monthly"]
    bl_inc = inc_monthly.loc[inc_monthly.index.isin(bl_months)] if len(inc_monthly) > 0 else pd.Series(dtype=float)
    mean_income = float(bl_inc.mean()) if len(bl_inc) > 0 else 0.0
    std_income = float(bl_inc.std()) if len(bl_inc) > 1 else 0.0

    # Current month income
    current_month_inc = float(inc_monthly.get(current_month, 0)) if current_month in inc_monthly.index else 0.0

    # Frequency deviation: current month txn count vs baseline average
    txn_count = h["txn_count_by_month"]
    bl_counts = txn_count.loc[txn_count.index.isin(bl_months)] if len(txn_count) > 0 else pd.Series(dtype=int)
    mean_freq = float(bl_counts.mean()) if len(bl_counts) > 0 else 0.0
    std_freq = float(bl_counts.std()) if len(bl_counts) > 1 else 0.0
    current_count = float(txn_count.get(current_month, 0)) if current_month in txn_count.index else 0.0
    freq_dev = (current_count - mean_freq) / std_freq if std_freq > 0 else 0.0

    # Income deviation
    inc_dev = (current_month_inc - mean_income) / std_income if std_income > 0 else 0.0

    # Category-specific stats
    cat_amounts = h["cat_amounts"].get(current_cat, np.array([]))
    cat_mons = h["cat_months"].get(current_cat, np.array([]))
    bl_month_set = set(bl_months)
    cat_bl_mask = np.array([m in bl_month_set for m in cat_mons])
    cat_bl_amounts = cat_amounts[cat_bl_mask] if len(cat_amounts) > 0 else np.array([])
    cat_mean = float(cat_bl_amounts.mean()) if len(cat_bl_amounts) > 0 else mean_expenses
    cat_std = float(cat_bl_amounts.std()) if len(cat_bl_amounts) > 1 else std_expenses

    # Detect novel categories: no prior spend in this category before current month
    prior_cat_months = [m for m in cat_mons if m < current_month]
    is_novel = 1.0 if len(prior_cat_months) == 0 else 0.0

    # Amount deviations
    amount_dev = (current_amount - mean_expenses) / std_expenses if std_expenses > 0 else 0.0
    amount_cat_dev = (current_amount - cat_mean) / cat_std if cat_std > 0 else 0.0
    amount_cat_rel = (current_amount - cat_mean) / cat_mean if cat_mean > 0 else 0.0

    # Category deviation
    cat_prop_val = float(cat_props.get(current_cat, 0))
    cat_dev = abs(cat_prop_val - 1.0 / len(ALL_CATS))

    # Category frequency change: current-month category count vs baseline average
    if hasattr(bl_cats, "columns") and current_cat in bl_cats.columns:
        cat_baseline_counts = float(bl_cats[current_cat].mean())
    else:
        cat_baseline_counts = 0.0
    if hasattr(cat_monthly, "columns") and current_cat in cat_monthly.columns:
        current_cat_count = float(
            cat_monthly.loc[current_month, current_cat]) if current_month in cat_monthly.index else 0.0
    else:
        current_cat_count = 0.0
    if cat_baseline_counts > 0:
        cat_freq_change = (current_cat_count - cat_baseline_counts) / cat_baseline_counts
    else:
        cat_freq_change = float(current_cat_count)

    # Spending concentration (Herfindahl index)
    cat_props_arr = cat_props.values.astype(float) if hasattr(cat_props, 'values') else np.array([0.0])
    hhi = float(np.sum(cat_props_arr ** 2))

    # Category entropy
    cat_props_pos = cat_props[cat_props > 0]
    entropy = float(-np.sum(cat_props_pos * np.log2(cat_props_pos))) if len(cat_props_pos) > 0 else 0.0

    # Amount percentile in category
    if len(cat_amounts) > 0:
        percentile = float(np.mean(cat_amounts < current_amount))
    else:
        percentile = 0.5

    # Days since last transaction
    txn_dates = h["all_dates"]
    current_date = pd.to_datetime(txn_row["date"])
    recent_dates = txn_dates[txn_dates < current_date]
    days_since = float((current_date - recent_dates[-1]).days) if len(recent_dates) > 0 else 30.0

    # Is weekend
    is_weekend = 1.0 if current_date.weekday() >= 5 else 0.0

    # Transaction frequency: mean monthly transaction count in baseline
    txn_freq = mean_freq

    # Monthly expense deviation: current month total vs baseline monthly totals
    month_totals = h["monthly_expense_totals"]
    bl_totals = month_totals.loc[month_totals.index.isin(bl_months)] if len(month_totals) > 0 else pd.Series(dtype=float)
    mean_tot = float(bl_totals.mean()) if len(bl_totals) > 0 else 0.0
    std_tot = float(bl_totals.std()) if len(bl_totals) > 1 else 0.0
    current_tot = float(month_totals.get(current_month, 0)) if current_month in month_totals.index else 0.0
    expense_dev = (current_tot - mean_tot) / std_tot if std_tot > 0 else 0.0

    return {
        # Baseline features
        "mean_income_rolling": mean_income,
        "std_income_rolling": std_income,
        "mean_expenses_rolling": mean_expenses,
        "std_expenses_rolling": std_expenses,
        "category_dist": float(cat_props.max()) if len(cat_props) > 0 else 0.0,
        "txn_frequency_rolling": txn_freq,
        "avg_txn_size_rolling": float(bl_amounts.mean()) if len(bl_amounts) > 0 else 0.0,
        "category_entropy": entropy,
        "volatility_index": std_expenses / mean_expenses if mean_expenses > 0 else 0.0,
        "spending_concentration": hhi,
        # Detection features
        "amount_deviation": amount_dev,
        "category_deviation": cat_dev,
        "frequency_deviation": freq_dev,
        "income_deviation": inc_dev,
        "expense_deviation": expense_dev,
        "is_novel_category": is_novel,
        "amount_vs_category_mean": amount_cat_rel,
        "amount_vs_category_std": amount_cat_dev,
        "category_frequency_change": cat_freq_change,
        "amount_percentile_in_category": percentile,
        "days_since_last_txn": days_since,
        "is_weekend": is_weekend,
        "amount_zscore_overall": abs(amount_dev),
        "amount_zscore_category": abs(amount_cat_dev),
    }


def _zero_features():
    """Return zero features for edge cases."""
    return {f: 0.0 for f in [
        "mean_income_rolling", "std_income_rolling",
        "mean_expenses_rolling", "std_expenses_rolling",
        "category_dist", "txn_frequency_rolling", "avg_txn_size_rolling",
        "category_entropy", "volatility_index", "spending_concentration",
        "amount_deviation", "category_deviation", "frequency_deviation",
        "income_deviation", "expense_deviation", "is_novel_category",
        "amount_vs_category_mean", "amount_vs_category_std",
        "category_frequency_change", "amount_percentile_in_category",
        "days_since_last_txn", "is_weekend", "amount_zscore_overall", "amount_zscore_category",
    ]}


ALL_FEATURES = list(_zero_features().keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default="synth/transactions.parquet")
    parser.add_argument("--splits", default="datasets/processed/split_metadata.json")
    parser.add_argument("--output", default="datasets/anomaly/")
    parser.add_argument("--baseline-months", type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()
    print("[1/5] Loading data...")
    txn, splits = load_data(args.transactions, args.splits)

    train_ids = set(splits.get("train", []))
    val_ids = set(splits.get("val", []))
    test_ids = set(splits.get("test", []))
    print(f"  {len(txn)} transactions, {len(train_ids)+len(val_ids)+len(test_ids)} personas")

    print("[2/5] Building category history...")
    history = build_category_history(txn)

    print("[3/5] Computing features per transaction...")
    rows = []
    for idx, (_, txn_row) in enumerate(txn.iterrows()):
        if idx % 100000 == 0 and idx > 0:
            print(f"    {idx:,}/{len(txn):,} transactions ({time.time() - t0:.0f}s)")
        pid = txn_row["persona_id"]
        feats = compute_features_for_persona(pid, txn_row, history, args.baseline_months)
        row = {
            "user_id": pid,
            "transaction_id": txn_row["transaction_id"],
            "year_month": str(txn_row["year_month"]),
            "month": int(txn_row["month"]),
            "date": txn_row["date"],
            "category": txn_row["category"],
            "amount": float(txn_row["amount"]),
            "transaction_type": txn_row["transaction_type"],
            "is_anomalous": bool(txn_row["is_anomalous"]),
            "anomaly_type": str(txn_row.get("anomaly_type", "")),
        }
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} rows, {len(ALL_FEATURES)} features")

    print("[4/5] Standardizing from train only...")
    train_mask = df["user_id"].isin(train_ids)
    train_stats = {}
    for feat in ALL_FEATURES:
        vals = df.loc[train_mask, feat]
        train_stats[feat] = {"mean": float(vals.mean()), "std": float(vals.std())}
        if train_stats[feat]["std"] > 0:
            df[feat] = (df[feat] - train_stats[feat]["mean"]) / train_stats[feat]["std"]

    print("[5/5] Exporting splits...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rows = {}
    anomaly_rates = {}
    for split_name, id_set in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        mask = df["user_id"].isin(id_set)
        split_df = df[mask].copy()
        split_df.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        n_rows[split_name] = len(split_df)
        anomaly_rates[split_name] = round(float(split_df["is_anomalous"].mean()), 4)
        print(f"  {split_name}: {n_rows[split_name]} rows, "
              f"anomaly rate: {anomaly_rates[split_name]*100:.2f}%")

    meta = {
        "timestamp": datetime.now().isoformat(),
        "n_features": len(ALL_FEATURES),
        "feature_columns": ALL_FEATURES,
        "meta_columns": META_COLUMNS,
        "n_rows": n_rows,
        "anomaly_rate": anomaly_rates,
        "standardization_stats": train_stats,
    }
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(ALL_FEATURES)} features, {sum(n_rows.values())} transactions, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
