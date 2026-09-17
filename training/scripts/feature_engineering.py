"""
Feature Engineering Pipeline for Odin ML Models

Takes preprocessed data (raw columns + metadata) from datasets/processed/
and produces enhanced feature matrices in datasets/engineered/.

Provides:
- 17 derived financial behavior features
- Cyclical encoding for temporal features (sin/cos)
- Interaction features
- Categorical encoding (one-hot, label)
- Feature selection (mutual information, RF importance)
- Dimensionality reduction (PCA)

Usage:
    python scripts/feature_engineering.py \
        --input datasets/processed/ \
        --output datasets/engineered/ \
        --encoding cyclical \
        --select-method mutual_info \
        --select-k 15 \
        --pca-variance 0.95

Design principles:
- Features computed incrementally per month (no future data used)
- Feature selection / PCA fit on train only, applied to val/test
- No data leakage across splits
- Backward compatible with preprocessor.py output
"""

import argparse
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class FeatureEngineeringError(Exception):
    pass


FEATURE_COLUMNS = [
    "income_stability_cv",
    "obligation_ratio",
    "savings_rate",
    "debt_to_income",
    "essential_ratio",
    "discretionary_ratio",
    "income_trend",
    "expense_trend",
    "volatility_index",
    "category_entropy",
    "transaction_frequency",
    "avg_transaction_size",
    "income_regularity",
    "expense_regularity",
    "income_expense_gap",
    "essential_income_ratio",
    "savings_income_ratio",
]

RAW_COLUMNS = [
    "total_income",
    "total_expenses",
    "food_expense",
    "housing_expense",
    "transport_expense",
    "health_expense",
    "education_expense",
    "other_expense",
    "savings",
    "debt_payment",
    "transaction_count",
]

METADATA_COLUMNS = [
    "user_id",
    "year_month",
    "month",
    "pfp_label",
    "runway_months",
    "financial_tolerance",
    "is_anomalous",
    "anomaly_type",
]

CYCLICAL_FEATURES = ["month"]

INTERACTION_FEATURES = [
    "income_volatility_interaction",
    "obligation_volatility_interaction",
]

GAUSSIAN_FEATURES = [
    "income_stability_cv", "obligation_ratio", "savings_rate",
    "debt_to_income", "essential_ratio", "discretionary_ratio",
    "income_trend", "expense_trend", "volatility_index",
    "transaction_frequency", "avg_transaction_size",
    "income_expense_gap", "essential_income_ratio", "savings_income_ratio",
]

BOUNDED_FEATURES = [
    "income_regularity", "expense_regularity", "category_entropy",
]

OUTLIER_CAPPED_FEATURES = ["debt_to_income", "savings_rate"]

FEATURE_RANGES = {
    "income_stability_cv": (0, 5),
    "obligation_ratio": (0, 1),
    "savings_rate": (-1, 1),
    "debt_to_income": (0, 10),
    "essential_ratio": (0, 1),
    "discretionary_ratio": (0, 1),
    "income_regularity": (0, 1),
    "expense_regularity": (0, 1),
    "category_entropy": (0, 2),
}

REDUNDANT_PAIRS = [
    ("obligation_ratio", "essential_ratio"),
    ("savings_rate", "savings_income_ratio"),
    ("category_entropy", "expense_regularity"),
]


@dataclass
class EngineeringConfig:
    encoding: str = "cyclical"
    select_method: Optional[str] = None
    select_k: Optional[int] = None
    pca_variance: Optional[float] = None
    pca_components: Optional[int] = None
    drop_redundant: bool = True
    seed: int = 42


@dataclass
class EngineeringReport:
    n_personas: int = 0
    n_original_features: int = 0
    n_engineered_features: int = 0
    n_selected_features: int = 0
    n_pca_components: int = 0
    pca_explained_variance: float = 0.0
    redundant_features_dropped: list = field(default_factory=list)
    selected_feature_names: list = field(default_factory=list)
    feature_importances: dict = field(default_factory=dict)
    multicollinearity_warnings: list = field(default_factory=list)
    scaler_carried_forward: bool = True


def load_processed_data(input_dir: str) -> dict:
    input_path = Path(input_dir)
    splits = {}
    for name in ["train", "val", "test"]:
        path = input_path / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Processed data not found: {path}")
        splits[name] = pd.read_parquet(path)
    return splits


def load_metadata(input_dir: str) -> dict:
    input_path = Path(input_dir)
    metadata = {}
    for name in ["split_metadata.json", "scalers.json", "temporal_folds.json",
                  "feature_columns.json", "pipeline_report.json"]:
        path = input_path / name
        if path.exists():
            with open(path) as f:
                metadata[name.replace(".json", "")] = json.load(f)
    return metadata


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or np.isnan(b):
        return default
    return a / b


def _linear_slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 3:
        return 0.0
    x_v, y_v = x[valid], values[valid]
    slope = np.polyfit(x_v, y_v, 1)[0]
    return float(slope)


def _rolling_std(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return 0.0
    return float(np.std(valid, ddof=1))


def _shannon_entropy(categories: list[str]) -> float:
    if not categories:
        return 0.0
    cat_array = np.array(categories)
    unique, counts = np.unique(cat_array, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_derived_features(
    persona_id: str,
    persona_row: dict,
    summaries: pd.DataFrame,
    anomaly_info: Optional[pd.DataFrame],
) -> list[dict]:
    persona_summaries = summaries[summaries["persona_id"] == persona_id].sort_values("year_month").copy()
    if persona_summaries.empty:
        return []

    persona_anomalies = {}
    if anomaly_info is not None:
        mask = anomaly_info["persona_id"] == persona_id
        for _, row in anomaly_info[mask].iterrows():
            persona_anomalies[row["year_month"]] = {
                "is_anomalous": bool(row["is_anomalous"]),
                "anomaly_type": str(row.get("anomaly_type", "")),
            }

    rows = []
    incomes = []
    expenses = []
    food_exp = []
    housing_exp = []
    transport_exp = []
    health_exp = []
    education_exp = []
    other_exp = []
    savings_list = []
    debt_list = []
    tx_counts = []
    has_income_months = []
    has_expense_months = []

    for _, month_row in persona_summaries.iterrows():
        month = int(month_row["month"])
        year_month = str(month_row["year_month"])
        income = float(month_row["total_income"])
        expense = float(month_row["total_expenses"])
        food = float(month_row.get("food_expense", 0))
        housing = float(month_row.get("housing_expense", 0))
        transport = float(month_row.get("transport_expense", 0))
        health = float(month_row.get("health_expense", 0))
        education = float(month_row.get("education_expense", 0))
        other = float(month_row.get("other_expense", 0))
        saving = float(month_row.get("savings", max(0, income - expense)))
        debt_payment = float(month_row.get("debt_payment", expense * 0.1))
        tx_count = int(month_row.get("transaction_count", 0))

        incomes.append(income)
        expenses.append(expense)
        food_exp.append(food)
        housing_exp.append(housing)
        transport_exp.append(transport)
        health_exp.append(health)
        education_exp.append(education)
        other_exp.append(other)
        savings_list.append(saving)
        debt_list.append(debt_payment)
        tx_counts.append(tx_count)
        has_income_months.append(1 if income > 0 else 0)
        has_expense_months.append(1 if expense > 0 else 0)

        arr_inc = np.array(incomes, dtype=float)
        arr_exp = np.array(expenses, dtype=float)
        arr_food = np.array(food_exp, dtype=float)
        arr_housing = np.array(housing_exp, dtype=float)
        arr_transport = np.array(transport_exp, dtype=float)
        arr_health = np.array(health_exp, dtype=float)
        arr_education = np.array(education_exp, dtype=float)
        arr_other = np.array(other_exp, dtype=float)
        arr_savings = np.array(savings_list, dtype=float)
        arr_debt = np.array(debt_list, dtype=float)
        arr_tx = np.array(tx_counts, dtype=float)

        total_exp = float(arr_exp.sum())
        total_inc = float(arr_inc.sum())
        n_months = len(arr_inc)

        inc_mean = float(np.mean(arr_inc))
        inc_std = float(np.std(arr_inc, ddof=1)) if n_months > 1 else 0.0
        income_cv = inc_std / inc_mean if inc_mean > 0 else 0.0

        essential = float(arr_food.sum() + arr_housing.sum() + arr_transport.sum()
                          + arr_health.sum() + arr_education.sum())
        obligatory = float(arr_debt.sum())
        obligation_ratio = _safe_div(essential + obligatory, total_exp)
        savings_rate = _safe_div(float(arr_savings.sum()), total_inc)
        debt_to_income = _safe_div(debt_payment * n_months, total_inc)
        essential_ratio = _safe_div(essential, total_exp)
        total_discretionary = float(arr_other.sum())
        discretionary_ratio = _safe_div(total_discretionary, total_exp)

        income_trend = _linear_slope(arr_inc)
        expense_trend = _linear_slope(arr_exp)
        volatility_index = _rolling_std(arr_exp)

        all_categories = (
            ["food"] * int(arr_food.sum() > 0) +
            ["housing"] * int(arr_housing.sum() > 0) +
            ["transport"] * int(arr_transport.sum() > 0) +
            ["health"] * int(arr_health.sum() > 0) +
            ["education"] * int(arr_education.sum() > 0) +
            ["other"] * int(arr_other.sum() > 0)
        )
        category_entropy = _shannon_entropy(all_categories) if all_categories else 0.0

        months_active = max(1, n_months)
        transaction_frequency = float(arr_tx.sum()) / months_active
        avg_transaction_size = _safe_div(total_exp, float(arr_tx.sum()), 0.0)

        income_regularity = _safe_div(sum(has_income_months), n_months)
        expense_regularity = _safe_div(sum(has_expense_months), n_months)

        income_expense_gap = total_inc - total_exp
        essential_income_ratio = _safe_div(essential, total_inc)
        savings_income_ratio = _safe_div(float(arr_savings.sum()), total_inc)

        anomaly = persona_anomalies.get(year_month, {"is_anomalous": False, "anomaly_type": ""})

        row = {
            "user_id": persona_id,
            "year_month": year_month,
            "month": month,
            "pfp_label": persona_row.get("pfp_label", ""),
            "is_anomalous": anomaly["is_anomalous"],
            "anomaly_type": anomaly["anomaly_type"],
            "income_stability_cv": round(income_cv, 4),
            "obligation_ratio": round(obligation_ratio, 4),
            "savings_rate": round(savings_rate, 4),
            "debt_to_income": round(debt_to_income, 4),
            "essential_ratio": round(essential_ratio, 4),
            "discretionary_ratio": round(discretionary_ratio, 4),
            "income_trend": round(income_trend, 4),
            "expense_trend": round(expense_trend, 4),
            "volatility_index": round(volatility_index, 4),
            "category_entropy": round(category_entropy, 4),
            "transaction_frequency": round(transaction_frequency, 4),
            "avg_transaction_size": round(avg_transaction_size, 4),
            "income_regularity": round(income_regularity, 4),
            "expense_regularity": round(expense_regularity, 4),
            "income_expense_gap": round(income_expense_gap, 4),
            "essential_income_ratio": round(essential_income_ratio, 4),
            "savings_income_ratio": round(savings_income_ratio, 4),
            "total_income": round(total_inc, 2),
            "total_expenses": round(total_exp, 2),
            "food_expense": round(float(arr_food.sum()), 2),
            "housing_expense": round(float(arr_housing.sum()), 2),
            "transport_expense": round(float(arr_transport.sum()), 2),
            "health_expense": round(float(arr_health.sum()), 2),
            "education_expense": round(float(arr_education.sum()), 2),
            "other_expense": round(float(arr_other.sum()), 2),
            "savings": round(float(arr_savings.sum()), 2),
            "debt_payment": round(debt_payment * n_months, 2),
            "transaction_count": int(arr_tx.sum()),
        }
        rows.append(row)

    return rows


def compute_features_from_row(row: pd.Series) -> dict:
    total_inc = float(row.get("total_income", 0))
    total_exp = float(row.get("total_expenses", 0))
    food = float(row.get("food_expense", 0))
    housing = float(row.get("housing_expense", 0))
    transport = float(row.get("transport_expense", 0))
    health = float(row.get("health_expense", 0))
    education = float(row.get("education_expense", 0))
    other = float(row.get("other_expense", 0))
    saving = float(row.get("savings", 0))
    debt = float(row.get("debt_payment", 0))
    tx_count = float(row.get("transaction_count", 0))
    n_months = max(1, int(row.get("month", 1)))

    essential = food + housing + transport + health + education

    income_cv = 0.0
    inc_mean = total_inc / n_months
    obligation_ratio = _safe_div(essential + debt, total_exp)
    savings_rate = _safe_div(saving, total_inc)
    debt_to_income = _safe_div(debt, total_inc)
    essential_ratio = _safe_div(essential, total_exp)
    discretionary_ratio = _safe_div(other, total_exp)
    income_trend = 0.0
    expense_trend = 0.0
    volatility_index = _safe_div(abs(total_exp - inc_mean), inc_mean, 0.0)

    categories = []
    for cat_name, cat_val in [("food", food), ("housing", housing),
                                ("transport", transport), ("health", health),
                                ("education", education), ("other", other)]:
        if cat_val > 0:
            categories.append(cat_name)
    category_entropy = _shannon_entropy(categories) if categories else 0.0

    transaction_frequency = _safe_div(tx_count, n_months)
    avg_transaction_size = _safe_div(total_exp, tx_count, 0.0)
    income_regularity = 1.0 if total_inc > 0 else 0.0
    expense_regularity = 1.0 if total_exp > 0 else 0.0
    income_expense_gap = total_inc - total_exp
    essential_income_ratio = _safe_div(essential, total_inc)
    savings_income_ratio = _safe_div(saving, total_inc)

    return {
        "income_stability_cv": round(income_cv, 4),
        "obligation_ratio": round(obligation_ratio, 4),
        "savings_rate": round(savings_rate, 4),
        "debt_to_income": round(debt_to_income, 4),
        "essential_ratio": round(essential_ratio, 4),
        "discretionary_ratio": round(discretionary_ratio, 4),
        "income_trend": round(income_trend, 4),
        "expense_trend": round(expense_trend, 4),
        "volatility_index": round(volatility_index, 4),
        "category_entropy": round(category_entropy, 4),
        "transaction_frequency": round(transaction_frequency, 4),
        "avg_transaction_size": round(avg_transaction_size, 4),
        "income_regularity": round(income_regularity, 4),
        "expense_regularity": round(expense_regularity, 4),
        "income_expense_gap": round(income_expense_gap, 4),
        "essential_income_ratio": round(essential_income_ratio, 4),
        "savings_income_ratio": round(savings_income_ratio, 4),
    }


def encode_cyclical(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    df = df.copy()
    vals = df[column].values.astype(float)
    df[f"{column}_sin"] = np.sin(2 * np.pi * vals / period)
    df[f"{column}_cos"] = np.cos(2 * np.pi * vals / period)
    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "income_stability_cv" in df.columns and "total_income" in df.columns:
        df["income_volatility_interaction"] = (
            df["income_stability_cv"] * df["total_income"])
    if "obligation_ratio" in df.columns and "income_stability_cv" in df.columns:
        df["obligation_volatility_interaction"] = (
            df["obligation_ratio"] * df["income_stability_cv"])
    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "one_hot",
    fit_encoders: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = fit_encoders or {}

    for col in columns:
        if col not in df.columns:
            continue

        if method == "label":
            if fit_encoders is None:
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str))
                encoders[col] = enc
            else:
                enc = fit_encoders.get(col)
                if enc:
                    known = set(enc.classes_)
                    mask = df[col].astype(str).isin(known)
                    df.loc[mask, col] = enc.transform(df.loc[mask, col].astype(str))
                    df.loc[~mask, col] = -1

        elif method == "one_hot":
            if fit_encoders is None:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = enc.fit_transform(df[[col]])
                feature_names = [f"{col}_{cat}" for cat in enc.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                encoders[col] = enc
            else:
                enc = fit_encoders.get(col)
                if enc:
                    encoded = enc.transform(df[[col]])
                    feature_names = [f"{col}_{cat}" for cat in enc.categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

    return df, encoders


def detect_redundant_features(df: pd.DataFrame) -> tuple[list[tuple[str, str, float]], list[str]]:
    warnings_list = []
    cols_to_drop = set()

    for col1, col2 in REDUNDANT_PAIRS:
        if col1 in df.columns and col2 in df.columns:
            corr = df[col1].corr(df[col2])
            if abs(corr) > 0.98:
                warnings_list.append((col1, col2, float(corr)))
                cols_to_drop.add(col2)

    return warnings_list, sorted(cols_to_drop)


def drop_redundant_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    drop_warnings, cols_to_drop = detect_redundant_features(df)
    dropped = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=dropped, errors="ignore")
    return df, dropped


def select_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "pfp_label",
    feature_cols: Optional[list[str]] = None,
    method: str = "mutual_info",
    k: int = 15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns
                        if c not in METADATA_COLUMNS + RAW_COLUMNS
                        and c not in ["user_id", "month"]]

    available = [c for c in feature_cols if c in train_df.columns]
    X_train = train_df[available].values
    y_train = train_df[target_col].values

    np.random.seed(seed)

    if method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=min(k, len(available)))
    elif method == "anova":
        selector = SelectKBest(f_classif, k=min(k, len(available)))
    else:
        raise ValueError(f"Unknown selection method: {method}")

    selector.fit(X_train, y_train)
    scores = {available[i]: float(selector.scores_[i]) for i in range(len(available))}
    selected_mask = selector.get_support()
    selected_cols = [available[i] for i in range(len(available)) if selected_mask[i]]

    importances = {col: scores[col] for col in selected_cols}
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    keep_cols = METADATA_COLUMNS + RAW_COLUMNS + selected_cols
    keep_cols = [c for c in keep_cols if c in train_df.columns]

    return (
        train_df[[c for c in keep_cols if c in train_df.columns]],
        val_df[[c for c in keep_cols if c in val_df.columns]],
        test_df[[c for c in keep_cols if c in test_df.columns]],
        selected_cols,
        sorted_importances,
    )


def apply_pca(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA, dict]:
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns
                        if c not in METADATA_COLUMNS + RAW_COLUMNS]

    available = [c for c in feature_cols if c in train_df.columns]
    X_train = train_df[available].values

    n_features = len(available)
    if n_components:
        pca = PCA(n_components=min(n_components, n_features), random_state=seed)
    elif variance_threshold:
        pca = PCA(n_components=variance_threshold, random_state=seed)
    else:
        pca = PCA(n_components=min(n_features, 10), random_state=seed)

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(val_df[available].values)
    X_test_pca = pca.transform(test_df[available].values)

    pca_cols = [f"pca_{i+1}" for i in range(pca.n_components_)]

    pca_train_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=train_df.index)
    pca_val_df = pd.DataFrame(X_val_pca, columns=pca_cols, index=val_df.index)
    pca_test_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=test_df.index)

    meta_cols = [c for c in METADATA_COLUMNS if c in train_df.columns]
    raw_cols = [c for c in RAW_COLUMNS if c in train_df.columns]

    train_out = pd.concat(
        [train_df[meta_cols + raw_cols].reset_index(drop=True), pca_train_df], axis=1)
    val_out = pd.concat(
        [val_df[meta_cols + raw_cols].reset_index(drop=True), pca_val_df], axis=1)
    test_out = pd.concat(
        [test_df[meta_cols + raw_cols].reset_index(drop=True), pca_test_df], axis=1)

    explained_variance = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_.T, index=available, columns=pca_cols)

    pca_info = {
        "n_components": pca.n_components_,
        "explained_variance_ratio": explained_variance.tolist(),
        "cumulative_variance": float(np.cumsum(explained_variance)[-1]),
        "feature_columns": pca_cols,
        "loadings": loadings.to_dict(),
    }

    return train_out, val_out, test_out, pca, pca_info


def impute_features(df: pd.DataFrame, train_median: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    critical_features = ["income_stability_cv", "obligation_ratio"]

    if train_median is None:
        medians = {}
        for col in FEATURE_COLUMNS:
            if col in critical_features:
                medians[col] = float(df[col].median()) if df[col].notna().any() else 0.0
            else:
                medians[col] = 0.0
    else:
        medians = train_median

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            fill = medians.get(col, 0.0)
            df[col] = df[col].fillna(fill)

    return df, medians


def cap_outliers(df: pd.DataFrame, train_percentiles: Optional[dict] = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    if train_percentiles is None:
        percentiles = {}
        for col in OUTLIER_CAPPED_FEATURES:
            if col in df.columns and df[col].notna().any():
                percentiles[col] = float(np.percentile(df[col].dropna(), 99))
            else:
                percentiles[col] = float("inf")
    else:
        percentiles = train_percentiles

    for col in OUTLIER_CAPPED_FEATURES:
        if col in df.columns:
            cap = percentiles.get(col, float("inf"))
            df[col] = df[col].clip(upper=cap)

    return df, percentiles


def validate_feature_ranges(df: pd.DataFrame) -> list[dict]:
    warnings_list = []
    for col, (lo, hi) in FEATURE_RANGES.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        below = (vals < lo).sum()
        above = (vals > hi).sum()
        if below > 0 or above > 0:
            warnings_list.append({
                "feature": col,
                "expected_range": f"[{lo}, {hi}]",
                "below_count": int(below),
                "above_count": int(above),
                "actual_min": float(vals.min()),
                "actual_max": float(vals.max()),
            })
    return warnings_list


def load_monthly_summaries(input_dir: str) -> pd.DataFrame:
    path = Path(input_dir) / "monthly_summaries.parquet"
    if not path.exists():
        raise FileNotFoundError(f"monthly_summaries.parquet not found at {path}")
    df = pd.read_parquet(path)
    required = ["persona_id", "year_month", "month", "year", "total_income", "total_expenses",
                 "transaction_count", "income_stability_cv", "obligation_ratio",
                 "runway_months", "financial_tolerance", "pfp_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FeatureEngineeringError(f"monthly_summaries.parquet missing columns: {missing}")
    if not df["year_month"].astype(str).str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise FeatureEngineeringError("monthly_summaries.parquet has invalid year_month values")
    return df


def load_personas(input_dir: str) -> pd.DataFrame:
    path = Path(input_dir) / "personas.parquet"
    if not path.exists():
        raise FileNotFoundError(f"personas.parquet not found at {path}")
    df = pd.read_parquet(path)
    required = ["persona_id", "pfp_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FeatureEngineeringError(f"personas.parquet missing columns: {missing}")
    return df


def load_anomaly_info(input_dir: str) -> Optional[pd.DataFrame]:
    path = Path(input_dir) / "transactions.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df = df[["persona_id", "year_month", "is_anomalous", "anomaly_type"]]
        agg = df.groupby(["persona_id", "year_month"]).agg(
            is_anomalous=("is_anomalous", "any"),
            anomaly_type=("anomaly_type", lambda x: next((v for v in x if pd.notna(v)), ""))
        ).reset_index()
        return agg
    except Exception:
        return None


def export_results(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    engineered_cols: list[str],
    report: EngineeringReport,
    scalers: Optional[dict] = None,
    temporal_folds: Optional[list] = None,
    split_metadata: Optional[dict] = None,
    output_dir: str = "datasets/engineered/",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out / "val.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)

    feature_info = {
        "feature_columns": engineered_cols,
        "raw_columns": RAW_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "all_columns": [c for c in train_df.columns],
        "gaussian_features": GAUSSIAN_FEATURES,
        "bounded_features": BOUNDED_FEATURES,
        "outlier_capped_features": OUTLIER_CAPPED_FEATURES,
        "feature_ranges": FEATURE_RANGES,
        "redundant_features_dropped": report.redundant_features_dropped,
        "selected_features": report.selected_feature_names,
        "feature_importances": report.feature_importances,
    }
    with open(out / "feature_columns.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    pipeline_report = {
        "timestamp": datetime.now().isoformat(),
        "n_personas": report.n_personas,
        "n_original_features": report.n_original_features,
        "n_engineered_features": report.n_engineered_features,
        "n_selected_features": report.n_selected_features,
        "n_pca_components": report.n_pca_components,
        "pca_explained_variance": report.pca_explained_variance,
        "redundant_features_dropped": report.redundant_features_dropped,
        "selected_feature_names": report.selected_feature_names,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "unique_personas": {
            "train": int(train_df["user_id"].nunique()),
            "val": int(val_df["user_id"].nunique()),
            "test": int(test_df["user_id"].nunique()),
        },
    }
    with open(out / "pipeline_report.json", "w") as f:
        json.dump(pipeline_report, f, indent=2)

    if scalers:
        with open(out / "scalers.json", "w") as f:
            json.dump(scalers, f, indent=2)
    if temporal_folds:
        with open(out / "temporal_folds.json", "w") as f:
            json.dump(temporal_folds, f, indent=2)
    if split_metadata:
        with open(out / "split_metadata.json", "w") as f:
            json.dump(split_metadata, f, indent=2)

    print(f"\nExported to {out}/")
    print(f"  train.parquet: {len(train_df):,} rows, "
          f"{train_df['user_id'].nunique():,} personas")
    print(f"  val.parquet:   {len(val_df):,} rows, "
          f"{val_df['user_id'].nunique():,} personas")
    print(f"  test.parquet:  {len(test_df):,} rows, "
          f"{test_df['user_id'].nunique():,} personas")
    print(f"  Engineered features: {len(engineered_cols)}")


def run_feature_engineering(
    input_dir: str = "datasets/processed/",
    synth_dir: str = "synth/",
    output_dir: str = "datasets/engineered/",
    config: Optional[EngineeringConfig] = None,
) -> None:
    if config is None:
        config = EngineeringConfig()

    print("=" * 60)
    print("Odin ML - Feature Engineering Pipeline")
    print("=" * 60)

    print("\n[1/8] Loading preprocessed data...")
    processed_splits = load_processed_data(input_dir)
    metadata = load_metadata(input_dir)
    print(f"  Loaded train: {len(processed_splits['train']):,} rows")
    print(f"  Loaded val:   {len(processed_splits['val']):,} rows")
    print(f"  Loaded test:  {len(processed_splits['test']):,} rows")

    print("\n[2/8] Loading synthesis data (for incremental feature computation)...")
    try:
        summaries = load_monthly_summaries(synth_dir)
        print(f"  Loaded {len(summaries):,} monthly summaries "
              f"({summaries['persona_id'].nunique():,} personas)")
    except FileNotFoundError:
        print("  Synth data not found - falling back to single-row computation")
        summaries = None

    try:
        personas_df = load_personas(synth_dir)
    except FileNotFoundError:
        personas_df = None

    anomaly_info = load_anomaly_info(synth_dir)

    print("\n[3/8] Computing derived features...")
    if summaries is not None and personas_df is not None:
        personas_lookup = personas_df.set_index("persona_id").to_dict("index")
        split_persona_ids = {}
        if "split_metadata" in metadata and "personas" in metadata["split_metadata"]:
            for sp in ["train", "val", "test"]:
                split_persona_ids[sp] = set(
                    metadata["split_metadata"]["personas"].get(sp, []))

        split_dfs = {}
        for split_name in ["train", "val", "test"]:
            df = processed_splits[split_name]
            pids = split_persona_ids.get(
                split_name, set(df["user_id"].unique()))
            print(f"  Computing for {split_name} ({len(pids)} personas)...")
            all_rows = []
            for pid in pids:
                if pid not in personas_lookup:
                    continue
                rows = compute_derived_features(
                    pid, personas_lookup[pid], summaries, anomaly_info)
                all_rows.extend(rows)
            if all_rows:
                split_dfs[split_name] = pd.DataFrame(all_rows)
            else:
                split_dfs[split_name] = df.copy()
            print(f"    -> {len(split_dfs[split_name]):,} rows")
    else:
        print("  Using preprocessed data directly for feature computation...")
        split_dfs = {}
        for split_name in ["train", "val", "test"]:
            df = processed_splits[split_name].copy()
            feature_rows = []
            for _, row in df.iterrows():
                feature_rows.append(compute_features_from_row(row))
            feat_df = pd.DataFrame(feature_rows)
            meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
            raw_cols = [c for c in RAW_COLUMNS if c in df.columns]
            split_dfs[split_name] = pd.concat(
                [df[meta_cols + raw_cols].reset_index(drop=True), feat_df], axis=1)
            print(f"    -> {len(split_dfs[split_name]):,} rows")

    print("\n[4/8] Imputing + capping outliers (fit on train only)...")
    train_df = split_dfs["train"]
    train_df, train_medians = impute_features(train_df, train_median=None)
    train_df, train_percentiles = cap_outliers(train_df, train_percentiles=None)
    split_dfs["train"] = train_df
    for split_name in ["val", "test"]:
        if split_name in split_dfs:
            split_dfs[split_name], _ = impute_features(
                split_dfs[split_name], train_median=train_medians)
            split_dfs[split_name], _ = cap_outliers(
                split_dfs[split_name], train_percentiles=train_percentiles)

    print("\n[5/8] Encoding temporal features...")
    feature_cols_now = FEATURE_COLUMNS.copy()
    added_cols = []

    if config.encoding == "cyclical":
        for col in CYCLICAL_FEATURES:
            if col in split_dfs["train"].columns:
                period_map = {"month": 12}
                period = period_map.get(col, 12)
                for split_name in split_dfs:
                    split_dfs[split_name] = encode_cyclical(
                        split_dfs[split_name], col, period)
                added_cols.extend([f"{col}_sin", f"{col}_cos"])
                print(f"  Added cyclical encoding for '{col}' (period={period})")

    print("\n[6/8] Computing interaction features...")
    for split_name in split_dfs:
        split_dfs[split_name] = compute_interaction_features(split_dfs[split_name])
    existing_interactions = [
        c for c in INTERACTION_FEATURES if c in split_dfs["train"].columns]
    if existing_interactions:
        added_cols.extend(existing_interactions)
        print(f"  Added interaction features: {existing_interactions}")

    all_engineered = feature_cols_now + added_cols

    print("\n[7/8] Feature selection / dimensionality reduction...")
    report = EngineeringReport()
    report.n_personas = split_dfs["train"]["user_id"].nunique()
    report.n_original_features = len(feature_cols_now)

    if config.drop_redundant:
        redundant_warnings, dropped = drop_redundant_features(split_dfs["train"])
        report.redundant_features_dropped = dropped
        all_engineered = [c for c in all_engineered if c not in dropped]
        for split_name in split_dfs:
            for col in dropped:
                if col in split_dfs[split_name].columns:
                    split_dfs[split_name] = split_dfs[split_name].drop(
                        columns=[col])
        if dropped:
            print(f"  Dropped redundant features: {dropped}")

    if config.select_method and config.select_k:
        train_out, val_out, test_out, selected, importances = select_features(
            split_dfs["train"],
            split_dfs["val"],
            split_dfs["test"],
            target_col="pfp_label",
            feature_cols=all_engineered,
            method=config.select_method,
            k=config.select_k,
            seed=config.seed,
        )
        split_dfs["train"] = train_out
        split_dfs["val"] = val_out
        split_dfs["test"] = test_out
        all_engineered = selected
        report.selected_feature_names = selected
        report.feature_importances = importances
        report.n_selected_features = len(selected)
        print(f"  Selected {len(selected)} features via {config.select_method}")
        for col in selected[:5]:
            print(f"    {col}: {importances[col]:.4f}")

    if config.pca_variance or config.pca_components:
        pca_kwargs = {}
        if config.pca_variance:
            pca_kwargs["variance_threshold"] = config.pca_variance
        if config.pca_components:
            pca_kwargs["n_components"] = config.pca_components

        train_out, val_out, test_out, pca, pca_info = apply_pca(
            split_dfs["train"],
            split_dfs["val"],
            split_dfs["test"],
            feature_cols=all_engineered,
            seed=config.seed,
            **pca_kwargs,
        )
        split_dfs["train"] = train_out
        split_dfs["val"] = val_out
        split_dfs["test"] = test_out
        all_engineered = pca_info["feature_columns"]
        report.n_pca_components = pca_info["n_components"]
        report.pca_explained_variance = pca_info["cumulative_variance"]
        print(f"  PCA: {pca_info['n_components']} components, "
              f"{pca_info['cumulative_variance']:.2%} variance explained")

    report.n_engineered_features = len(all_engineered)

    print("\n[8/8] Exporting engineered feature matrices...")
    scalers = metadata.get("scalers")
    temporal_folds = metadata.get("temporal_folds")
    split_metadata = metadata.get("split_metadata")

    export_results(
        train_df=split_dfs["train"],
        val_df=split_dfs["val"],
        test_df=split_dfs["test"],
        engineered_cols=all_engineered,
        report=report,
        scalers=scalers,
        temporal_folds=temporal_folds,
        split_metadata=split_metadata,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Odin ML - Feature Engineering Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="datasets/processed/",
                        help="Input directory with preprocessed data")
    parser.add_argument("--synth-dir", type=str, default="synth/",
                        help="Synthetic data directory (for incremental computation)")
    parser.add_argument("--output", type=str, default="datasets/engineered/",
                        help="Output directory for engineered features")
    parser.add_argument("--encoding", type=str, default="cyclical",
                        choices=["cyclical", "none"],
                        help="Temporal encoding strategy")
    parser.add_argument("--select-method", type=str, default=None,
                        choices=["mutual_info", "anova", None],
                        help="Feature selection method")
    parser.add_argument("--select-k", type=int, default=None,
                        help="Number of features to select")
    parser.add_argument("--pca-variance", type=float, default=None,
                        help="PCA variance threshold (e.g., 0.95)")
    parser.add_argument("--pca-components", type=int, default=None,
                        help="Number of PCA components")
    parser.add_argument("--no-drop-redundant", action="store_true",
                        help="Skip dropping redundant features")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    config = EngineeringConfig(
        encoding=args.encoding,
        select_method=args.select_method,
        select_k=args.select_k,
        pca_variance=args.pca_variance,
        pca_components=args.pca_components,
        drop_redundant=not args.no_drop_redundant,
        seed=args.seed,
    )

    try:
        run_feature_engineering(
            input_dir=args.input,
            synth_dir=args.synth_dir,
            output_dir=args.output,
            config=config,
        )
    except FeatureEngineeringError as e:
        print(f"Feature engineering error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
