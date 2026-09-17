"""
Data Preprocessing Pipeline for Odin ML Models

Runs the synthetic data pipeline first (persona + transaction generation),
then loads the output, splits personas into train/val/test, and exports
preprocessed raw data ready for feature engineering.

Feature engineering is handled by the separate feature_engineering.py module,
which reads from datasets/processed/ and outputs to datasets/engineered/.

Key design principles:
- Splitting happens BEFORE any feature engineering (no leakage)
- Rolling-origin evaluation with configurable embargo gap
- Train-only fitted parameters exported for downstream use

Usage:
    python scripts/preprocessor.py --input datasets/unprocessed/puf.parquet --output datasets/processed/
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from fies_columns import FIESColumnError, validate_columns
from generate_personas import (
    SyntheticPersona,
    PersonaGenerationError,
    load_fies_data,
    filter_ncr,
    compute_fies_statistics,
    compute_expense_ratios,
    generate_all_personas,
    validate_personas,
    export_personas,
)
from generate_transactions import (
    TransactionGenerationError,
    generate_persona_transactions,
    validate_transactions,
    export_transactions,
)


class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


class DataValidationError(PreprocessingError):
    """Raised when data validation fails."""
    pass


SYNTH_OUTPUT_DIR = "synth/"

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
        raise DataValidationError(f"monthly_summaries.parquet missing columns: {missing}")
    if not df["year_month"].astype(str).str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])").all():
        raise DataValidationError("monthly_summaries.parquet has invalid year_month values")
    return df


def load_personas(input_dir: str) -> pd.DataFrame:
    path = Path(input_dir) / "personas.parquet"
    if not path.exists():
        raise FileNotFoundError(f"personas.parquet not found at {path}")
    df = pd.read_parquet(path)
    required = ["persona_id", "pfp_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"personas.parquet missing columns: {missing}")
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


def validate_data(summaries: pd.DataFrame, personas: pd.DataFrame) -> dict:
    n_personas = summaries["persona_id"].nunique()
    n_months = summaries["year_month"].nunique()
    total_rows = len(summaries)

    report = {
        "n_personas": n_personas,
        "n_months": n_months,
        "total_rows": total_rows,
        "warnings": [],
    }

    if n_months < 2:
        report["warnings"].append(f"Only {n_months} months of data — features will be limited")

    zero_income = (summaries["total_income"] == 0).sum()
    if zero_income > 0:
        report["warnings"].append(f"{zero_income} rows have zero income")

    persona_month_counts = summaries.groupby("persona_id")["year_month"].count()
    incomplete = (persona_month_counts < 6).sum()
    if incomplete > 0:
        report["warnings"].append(f"{incomplete} personas have fewer than 6 months of data")

    return report


def split_personas(
    personas: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    labels = personas["pfp_label"].values
    persona_ids = personas["persona_id"].values

    strat = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(strat.split(persona_ids, labels))

    temp_labels = labels[temp_idx]
    rel_test = test_ratio / (val_ratio + test_ratio)
    strat2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    val_rel_idx, test_rel_idx = next(strat2.split(temp_idx, temp_labels))

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    splits = {
        "train": set(persona_ids[train_idx]),
        "val": set(persona_ids[val_idx]),
        "test": set(persona_ids[test_idx]),
    }

    distributions = {}
    for split_name, idx_set in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_labels = labels[idx_set]
        unique, counts = np.unique(split_labels, return_counts=True)
        distributions[split_name] = {str(k): int(v) for k, v in zip(unique, counts)}

    return {"splits": splits, "distributions": distributions, "n_total": len(persona_ids)}


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or np.isnan(b):
        return default
    return a / b








def generate_temporal_folds(
    periods: list[str],
    min_train_months: int = 6,
    embargo_months: int = 1,
    test_horizon_months: int = 1,
    strategy: str = "expanding",
) -> list[dict]:
    folds = []

    if strategy == "expanding":
        fold_num = 0
        train_end = min_train_months
        while train_end + embargo_months + test_horizon_months <= len(periods):
            fold_num += 1
            emb_start = train_end + 1
            emb_end = train_end + embargo_months
            test_start = emb_end + 1
            test_end = min(test_start + test_horizon_months - 1, len(periods))

            folds.append({
                "fold": fold_num,
                "train_periods": periods[:train_end],
                "embargo_periods": periods[emb_start - 1:emb_end] if embargo_months > 0 else [],
                "test_periods": periods[test_start - 1:test_end],
            })
            train_end += 1

    elif strategy == "rolling":
        fold_num = 0
        train_start = 1
        train_end = min_train_months
        while train_end + embargo_months + test_horizon_months <= len(periods):
            fold_num += 1
            emb_start = train_end + 1
            emb_end = train_end + embargo_months
            test_start = emb_end + 1
            test_end = min(test_start + test_horizon_months - 1, len(periods))

            folds.append({
                "fold": fold_num,
                "train_periods": periods[train_start - 1:train_end],
                "embargo_periods": periods[emb_start - 1:emb_end] if embargo_months > 0 else [],
                "test_periods": periods[test_start - 1:test_end],
            })
            train_start += 1
            train_end += 1

    return folds


def export_results(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_metadata: dict,
    temporal_folds: list[dict],
    validation_report: dict,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out / "val.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)

    with open(out / "split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)

    with open(out / "temporal_folds.json", "w") as f:
        json.dump(temporal_folds, f, indent=2)

    feature_info = {
        "raw_columns": RAW_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "all_columns": [c for c in train_df.columns],
    }
    with open(out / "feature_columns.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    report = {
        "timestamp": datetime.now().isoformat(),
        "validation": validation_report,
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
        json.dump(report, f, indent=2)

    print(f"\nExported to {out}/")
    print(f"  train.parquet: {len(train_df):,} rows, {train_df['user_id'].nunique():,} personas")
    print(f"  val.parquet:   {len(val_df):,} rows, {val_df['user_id'].nunique():,} personas")
    print(f"  test.csv:  {len(test_df):,} rows, {test_df['user_id'].nunique():,} personas")


def _run_synthesis(
    input_path: str,
    output_path: str,
    personas_per_archetype: int,
    num_months: int,
    seed: int,
    skip_fies: bool,
    strict: bool,
) -> None:
    """Generate personas and transactions, writing results to output_path."""
    start_time = time.time()

    # Load FIES data
    fies_stats = None
    expense_ratios = None

    if not skip_fies:
        try:
            df = load_fies_data(input_path)
            report = validate_columns(df, strict=strict)
            print(f"    Found {report['found']}/{report['total_columns']} expected columns")

            if not report["is_valid"] and strict:
                raise PreprocessingError(
                    f"Strict mode: Missing critical columns: {report['critical_missing']}"
                )

            ncr_df = filter_ncr(df)
            fies_stats = compute_fies_statistics(ncr_df)
            expense_ratios = compute_expense_ratios(ncr_df)
        except FileNotFoundError as e:
            print(f"    WARNING: {e} — falling back to default statistics")
        except FIESColumnError as e:
            print(f"    WARNING: {e} — falling back to default statistics")
        except PreprocessingError:
            raise
        except Exception as e:
            print(f"    WARNING: Unexpected error loading FIES data: {e}")

    # Generate personas
    personas = generate_all_personas(
        personas_per_archetype=personas_per_archetype,
        fies_stats=fies_stats,
        expense_ratios=expense_ratios,
        seed=seed,
    )
    validate_personas(personas)
    export_personas(personas, output_path)
    print(f"    Generated {len(personas):,} personas")

    # Generate transactions
    all_transactions = []
    all_summaries = []
    failed = 0

    for i, persona in enumerate(personas):
        if (i + 1) % 1000 == 0:
            print(f"    Processing persona {i + 1:,}/{len(personas):,}...")

        try:
            persona_dict = asdict(persona) if isinstance(persona, SyntheticPersona) else persona
            transactions, summaries = generate_persona_transactions(
                persona_dict, start_year=2023, start_month=1, num_months=num_months, seed=seed,
            )
            all_transactions.extend(transactions)
            all_summaries.extend(summaries)
        except TransactionGenerationError:
            failed += 1
        except Exception:
            failed += 1

    if failed > 0:
        print(f"    WARNING: {failed} personas failed to generate transactions")

    validate_transactions(all_summaries)
    export_transactions(all_transactions, all_summaries, output_path)

    elapsed = time.time() - start_time
    print(f"    Synthesis complete in {elapsed:.1f}s — {len(all_transactions):,} transactions, {len(all_summaries):,} summaries")


def run_preprocessing(
    input_path: str,
    output_dir: str,
    personas_per_archetype: int = 1000,
    num_months: int = 12,
    seed: int = 42,
    skip_fies: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_months: int = 1,
    min_train_months: int = 6,
    test_horizon_days: int = 30,
    wfv_strategy: str = "expanding",
    strict: bool = False,
) -> None:
    print("=" * 60)
    print("Odin ML — Data Preprocessing Pipeline")
    print("=" * 60)

    print("\n[1/8] Running synthetic data pipeline...")
    try:
        _run_synthesis(
            input_path=input_path,
            output_path=SYNTH_OUTPUT_DIR,
            personas_per_archetype=personas_per_archetype,
            num_months=num_months,
            seed=seed,
            skip_fies=skip_fies,
            strict=strict,
        )
    except PreprocessingError:
        raise
    except Exception as e:
        print(f"  Synthesis pipeline failed: {e}")
        raise PreprocessingError(f"Synthesis failed: {e}")

    print("\n[2/8] Loading raw data...")
    summaries = load_monthly_summaries(SYNTH_OUTPUT_DIR)
    personas_df = load_personas(SYNTH_OUTPUT_DIR)
    anomaly_info = load_anomaly_info(SYNTH_OUTPUT_DIR)
    print(f"  Loaded {len(summaries):,} monthly summaries ({summaries['persona_id'].nunique():,} personas)")
    print(f"  Loaded {len(personas_df):,} persona records")
    if anomaly_info is not None:
        print(f"  Loaded anomaly info ({anomaly_info['persona_id'].nunique():,} personas)")

    print("\n[3/8] Validating data...")
    validation_report = validate_data(summaries, personas_df)
    print(f"  Persons: {validation_report['n_personas']}, Months: {validation_report['n_months']}")
    for w in validation_report["warnings"]:
        print(f"  WARNING: {w}")

    print("\n[4/8] Splitting personas (stratified by PFP label)...")
    split_result = split_personas(personas_df, train_ratio, val_ratio, test_ratio, seed)
    split_sizes = {k: len(v) for k, v in split_result["splits"].items()}
    print(f"  Train: {split_sizes['train']}, Val: {split_sizes['val']}, Test: {split_sizes['test']}")
    print(f"  Distributions: {json.dumps(split_result['distributions'], indent=4)}")

    print("\n[5/8] Building per-split feature matrices (raw data only)...")
    split_dfs = {}
    for split_name, persona_ids in split_result["splits"].items():
        print(f"  Building {split_name} ({len(persona_ids)} personas)...")
        mask = summaries["persona_id"].isin(persona_ids)
        persona_merge_cols = [c for c in ["persona_id", "pfp_label", "runway_months", "financial_tolerance"]
                              if c not in summaries.columns or c == "persona_id"]
        split_df = summaries[mask].merge(
            personas_df[persona_merge_cols],
            on="persona_id", how="left"
        )
        split_df = split_df.rename(columns={"persona_id": "user_id"})

        if anomaly_info is not None:
            anomaly_renamed = anomaly_info.rename(columns={"persona_id": "user_id"})
            drop_cols = [c for c in ["is_anomalous", "anomaly_type"] if c in split_df.columns]
            split_df = split_df.drop(columns=drop_cols)
            split_df = split_df.merge(
                anomaly_renamed, on=["user_id", "year_month"], how="left")
            split_df["is_anomalous"] = split_df["is_anomalous"].fillna(False).astype(bool)
            split_df["anomaly_type"] = split_df["anomaly_type"].fillna("")

        meta_cols = [c for c in METADATA_COLUMNS if c in split_df.columns]
        raw_cols = [c for c in RAW_COLUMNS if c in split_df.columns]
        split_dfs[split_name] = split_df[meta_cols + raw_cols].copy()
        print(f"    -> {len(split_dfs[split_name]):,} rows")

    periods = sorted(summaries["year_month"].unique())
    test_horizon_months = max(1, test_horizon_days // 30)

    print("\n[6/8] Generating temporal folds...")
    temporal_folds = generate_temporal_folds(
        periods=periods,
        min_train_months=min_train_months,
        embargo_months=embargo_months,
        test_horizon_months=test_horizon_months,
        strategy=wfv_strategy,
    )
    print(f"  Generated {len(temporal_folds)} temporal folds ({wfv_strategy})")
    for fold in temporal_folds:
        print(f"    Fold {fold['fold']}: Train {fold['train_periods']}, "
              f"Embargo {fold['embargo_periods']}, Test {fold['test_periods']}")

    print("\n[7/8] Exporting preprocessed raw data...")
    split_metadata = {
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "personas": {k: sorted(list(v)) for k, v in split_result["splits"].items()},
        "distributions": split_result["distributions"],
        "split_sizes": split_sizes,
        "embargo_months": embargo_months,
        "min_train_months": min_train_months,
        "test_horizon_days": test_horizon_days,
        "wfv_strategy": wfv_strategy,
    }

    export_results(
        split_dfs["train"],
        split_dfs["val"],
        split_dfs["test"],
        split_metadata,
        temporal_folds,
        validation_report,
        output_dir,
    )

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Odin ML — Data Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="datasets/unprocessed/puf.parquet",
                        help="Path to FIES data file (CSV or Parquet)")
    parser.add_argument("--output", type=str, default="datasets/processed/",
                        help="Output directory for processed data")
    parser.add_argument("--personas-per-archetype", type=int, default=1000,
                        help="Number of personas per archetype")
    parser.add_argument("--months", type=int, default=12,
                        help="Number of months to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-fies", action="store_true",
                        help="Skip FIES data loading (use default statistics)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--embargo-months", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--test-horizon-days", type=int, default=30)
    parser.add_argument("--wfv-strategy", type=str, default="expanding",
                        choices=["expanding", "rolling"])
    parser.add_argument("--strict", action="store_true")

    args = parser.parse_args()

    try:
        run_preprocessing(
            input_path=args.input,
            output_dir=args.output,
            personas_per_archetype=args.personas_per_archetype,
            num_months=args.months,
            seed=args.seed,
            skip_fies=args.skip_fies,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            embargo_months=args.embargo_months,
            min_train_months=args.min_train_months,
            test_horizon_days=args.test_horizon_days,
            wfv_strategy=args.wfv_strategy,
            strict=args.strict,
        )
    except PreprocessingError as e:
        print(f"Preprocessing error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.strict:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
