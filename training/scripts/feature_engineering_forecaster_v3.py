"""Leak-free monthly features for the survey-only v3 forecasting input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from logging_v3 import configure_logging, get_logger

LOGGER = get_logger("feature_engineering")

FEATURE_COLUMNS = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "month_sin", "month_cos"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values(["household_id_v3", "year_month"]).copy()
    groups = output.groupby("household_id_v3", group_keys=False)["total_expenses"]
    for lag in range(1, 4):
        output[f"lag_{lag}"] = groups.shift(lag)
    output["rolling_mean_3"] = groups.transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    month = pd.PeriodIndex(output["year_month"], freq="M").month
    output["month_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * month / 12)
    output["month_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * month / 12)
    output["target_expenses"] = groups.shift(-1)
    return output


def engineer(processed_dir: str | Path, output_dir: str | Path) -> None:
    source, destination = Path(processed_dir), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Building survey-only features from %s", source)
    # Zero is an explicit cold-start value, not a dataset-derived replacement;
    # it keeps feature generation streamable and avoids held-out information.
    imputation = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    for name in ("train", "val", "test"):
        LOGGER.info("Building %s features", name)
        input_file = pq.ParquetFile(source / f"{name}.parquet")
        writer: pq.ParquetWriter | None = None
        carry = pd.DataFrame()
        try:
            for record_batch in input_file.iter_batches(batch_size=100_000):
                frame = pd.concat([carry, record_batch.to_pandas()], ignore_index=True)
                if frame.empty:
                    continue
                last_id = frame.iloc[-1]["household_id_v3"]
                complete, carry = (
                    frame.loc[frame["household_id_v3"] != last_id],
                    frame.loc[frame["household_id_v3"] == last_id],
                )
                if complete.empty:
                    continue
                features = build_features(complete)
                features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].fillna(imputation)
                table = pa.Table.from_pandas(features, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(destination / f"{name}.parquet", table.schema)
                writer.write_table(table)
            if not carry.empty:
                features = build_features(carry)
                features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].fillna(imputation)
                table = pa.Table.from_pandas(features, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(destination / f"{name}.parquet", table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        output_file = destination / f"{name}.parquet"
        if not output_file.exists():
            pq.write_table(pa.Table.from_pylist([], schema=input_file.schema_arrow), output_file)
        LOGGER.info("Wrote %s features to %s", name, output_file)
    provenance = json.loads((source / "split_metadata.json").read_text())
    (destination / "feature_columns.json").write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "imputation": imputation,
                "feature_provenance": "calendar features and strictly prior household months only",
                "source_fingerprint_sha256": provenance.get("source_fingerprint_sha256"),
                "hfce_config_fingerprint_sha256": provenance.get("hfce_config_fingerprint_sha256"),
            },
            indent=2,
        )
    )
    LOGGER.info("Feature engineering complete; wrote metadata to %s", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build leak-free survey-only v3 forecast features."
    )
    parser.add_argument("--input", default="training/datasets/processed_v3")
    parser.add_argument("--output", default="training/datasets/forecaster_v3")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    engineer(args.input, args.output)
