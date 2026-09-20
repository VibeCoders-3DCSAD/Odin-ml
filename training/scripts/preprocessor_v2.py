"""
Data Preprocessing Pipeline v2 — Synthetic Generation v2

Runs the Synthetic Generation v2 pipeline (FIES-anchored, HFCE-calibrated
persona + transaction generation via `synthesizer_v2.py`), then splits
personas into train/val/test and builds temporal folds — using the exact
same splitting/folding algorithm as v1's `preprocessor.py` (imported
read-only), so v2 data can be evaluated under the same walk-forward
validation contract as v1 (`split_metadata.json`, `temporal_folds.json`).

This module runs **in parallel** to `training/scripts/preprocessor.py` (v1).
It imports read-only helpers from v1's `preprocessor.py` (`split_personas`,
`generate_temporal_folds`, `load_monthly_summaries`, `load_personas`,
`load_anomaly_info`, `validate_data`, `export_results`, `RAW_COLUMNS`,
`METADATA_COLUMNS`) but does not edit, monkeypatch, or rewrite anything in
v1. See `tests/test_synth_v1_untouched.py` for the regression guard.

Output layout mirrors v1's `training/datasets/processed/` exactly, just
under a v2-scoped directory by default:

    training/datasets/processed_v2/
        train.parquet / val.parquet / test.parquet
        split_metadata.json         # includes synth_version = "2.1.0"
        temporal_folds.json
        feature_columns.json
        pipeline_report.json

Usage:
    python training/scripts/preprocessor_v2.py \
        --input training/datasets/unprocessed/puf.parquet \
        --output training/datasets/processed_v2/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add scripts directory to path for sibling imports (mirrors v1 scripts' pattern).
sys.path.insert(0, str(Path(__file__).parent))

from preprocessor import (  # noqa: E402  (v1, read-only reuse)
    METADATA_COLUMNS,
    RAW_COLUMNS,
    PreprocessingError,
    export_results,
    generate_temporal_folds,
    load_anomaly_info,
    load_monthly_summaries,
    load_personas,
    split_personas,
    validate_data,
)
from synthesizer_v2 import PipelineErrorV2, run_pipeline_v2  # noqa: E402
from temporal_disaggregation import SYNTH_VERSION  # noqa: E402

SYNTH_V2_OUTPUT_DIR = "synth_v2/"


class PreprocessingErrorV2(PreprocessingError):
    """Raised when Synthetic Generation v2 preprocessing fails."""

    pass


def run_preprocessing_v2(
    input_path: str,
    output_dir: str,
    synth_output_dir: str = SYNTH_V2_OUTPUT_DIR,
    personas_per_archetype: int = 1000,
    num_months: int = 42,
    seed: int = 42,
    skip_fies: bool = False,
    hfce_path: str | None = None,
    inject_anomalies_flag: bool = True,
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
    print("Odin ML — Data Preprocessing Pipeline v2 (Synthetic Generation v2)")
    print("=" * 60)

    print("\n[1/8] Running Synthetic Generation v2 pipeline...")
    try:
        run_pipeline_v2(
            input_path=input_path,
            output_path=synth_output_dir,
            personas_per_archetype=personas_per_archetype,
            num_months=num_months,
            seed=seed,
            skip_fies=skip_fies,
            strict=strict,
            hfce_path=hfce_path,
            inject_anomalies_flag=inject_anomalies_flag,
        )
    except PipelineErrorV2 as e:
        raise PreprocessingErrorV2(f"Synthetic Generation v2 pipeline failed: {e}") from e

    print("\n[2/8] Loading raw data...")
    summaries = load_monthly_summaries(synth_output_dir)
    personas_df = load_personas(synth_output_dir)
    anomaly_info = load_anomaly_info(synth_output_dir)
    print(
        f"  Loaded {len(summaries):,} monthly summaries "
        f"({summaries['persona_id'].nunique():,} personas)"
    )
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
    print(
        f"  Train: {split_sizes['train']}, Val: {split_sizes['val']}, Test: {split_sizes['test']}"
    )

    print("\n[5/8] Building per-split feature matrices (raw data only)...")
    split_dfs = {}
    for split_name, persona_ids in split_result["splits"].items():
        print(f"  Building {split_name} ({len(persona_ids)} personas)...")
        mask = summaries["persona_id"].isin(persona_ids)
        persona_merge_cols = [
            c
            for c in ["persona_id", "pfp_label", "runway_months", "financial_tolerance"]
            if c not in summaries.columns or c == "persona_id"
        ]
        split_df = summaries[mask].merge(
            personas_df[persona_merge_cols],
            on="persona_id",
            how="left",
        )
        split_df = split_df.rename(columns={"persona_id": "user_id"})

        if anomaly_info is not None:
            anomaly_renamed = anomaly_info.rename(columns={"persona_id": "user_id"})
            drop_cols = [c for c in ["is_anomalous", "anomaly_type"] if c in split_df.columns]
            split_df = split_df.drop(columns=drop_cols)
            split_df = split_df.merge(anomaly_renamed, on=["user_id", "year_month"], how="left")
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
        print(
            f"    Fold {fold['fold']}: Train {fold['train_periods']}, "
            f"Embargo {fold['embargo_periods']}, Test {fold['test_periods']}"
        )

    print("\n[7/8] Exporting preprocessed raw data...")
    split_metadata = {
        "synth_version": SYNTH_VERSION,
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "personas": {k: sorted(v) for k, v in split_result["splits"].items()},
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
    print("Preprocessing v2 complete!")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Odin ML — Data Preprocessing Pipeline v2 (Synthetic Generation v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/unprocessed/puf.parquet",
        help="Path to FIES data file (CSV or Parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/processed_v2/",
        help="Output directory for processed v2 data (splits, folds, reports)",
    )
    parser.add_argument(
        "--synth-output",
        type=str,
        default=SYNTH_V2_OUTPUT_DIR,
        help="Output directory for raw Synthetic Generation v2 artifacts (personas/transactions)",
    )
    parser.add_argument(
        "--hfce",
        type=str,
        default=None,
        help="Path to HFCE quarterly indices config (defaults to "
        "training/config/hfce_quarterly_indices.json)",
    )
    parser.add_argument(
        "--personas-per-archetype", type=int, default=1000, help="Number of personas per archetype"
    )
    parser.add_argument("--months", type=int, default=42, help="Number of months to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--skip-fies", action="store_true", help="Skip FIES data loading (use default statistics)"
    )
    parser.add_argument("--no-anomalies", action="store_true", help="Disable anomaly injection")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--embargo-months", type=int, default=1)
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--test-horizon-days", type=int, default=30)
    parser.add_argument(
        "--wfv-strategy", type=str, default="expanding", choices=["expanding", "rolling"]
    )
    parser.add_argument("--strict", action="store_true")

    args = parser.parse_args()

    try:
        run_preprocessing_v2(
            input_path=args.input,
            output_dir=args.output,
            synth_output_dir=args.synth_output,
            personas_per_archetype=args.personas_per_archetype,
            num_months=args.months,
            seed=args.seed,
            skip_fies=args.skip_fies,
            hfce_path=args.hfce,
            inject_anomalies_flag=not args.no_anomalies,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            embargo_months=args.embargo_months,
            min_train_months=args.min_train_months,
            test_horizon_days=args.test_horizon_days,
            wfv_strategy=args.wfv_strategy,
            strict=args.strict,
        )
    except PreprocessingErrorV2 as e:
        print(f"Preprocessing v2 error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.strict:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
