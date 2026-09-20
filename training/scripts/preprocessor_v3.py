"""Household-disjoint preprocessing for survey-only v3 monthly data."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from logging_v3 import get_logger

LOGGER = get_logger("preprocessing")


def split_households(ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    ordered = sorted(
        ids, key=lambda value: __import__("hashlib").sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    n = len(ordered)
    train_end, val_end = round(n * 0.70), round(n * 0.85)
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


def preprocess(monthly_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    source = Path(monthly_path)
    LOGGER.info("Reading monthly summaries from %s", source)
    parquet = pq.ParquetFile(source)
    required = {"household_id_v3", "year_month", "total_expenses"}
    if required - set(parquet.schema.names):
        raise ValueError("monthly summaries lack required v3 columns")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    household_ids: set[str] = set()
    periods: set[str] = set()
    for batch in parquet.iter_batches(
        columns=["household_id_v3", "year_month"], batch_size=100_000
    ):
        frame = batch.to_pandas()
        household_ids.update(frame["household_id_v3"])
        periods.update(frame["year_month"])
    splits = split_households(list(household_ids), seed)
    LOGGER.info(
        "Split %d households into train=%d, val=%d, test=%d",
        len(household_ids),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    membership = {household_id: name for name, ids in splits.items() for household_id in ids}
    writers: dict[str, pq.ParquetWriter] = {}
    row_counts = dict.fromkeys(splits, 0)
    try:
        for batch in parquet.iter_batches(batch_size=100_000):
            frame = batch.to_pandas()
            frame["_split"] = frame["household_id_v3"].map(membership)
            for name in splits:
                partition = frame.loc[frame["_split"] == name].drop(columns="_split")
                if partition.empty:
                    continue
                table = pa.Table.from_pandas(partition, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(destination / f"{name}.parquet", table.schema)
                writers[name].write_table(table)
                row_counts[name] += len(partition)
    finally:
        for writer in writers.values():
            writer.close()
    for name in splits:
        path = destination / f"{name}.parquet"
        if not path.exists():
            pq.write_table(pa.Table.from_pylist([], schema=parquet.schema_arrow), path)
    ordered_periods = sorted(periods)
    folds = [
        {
            "fold": index,
            "train_periods": ordered_periods[:index],
            "test_periods": [ordered_periods[index]],
        }
        for index in range(6, len(ordered_periods) - 1)
    ]
    metadata = {
        "selection_contract_version": "3.0.0",
        "seed": seed,
        "households": splits,
        "timeline": [ordered_periods[0], ordered_periods[-1]],
    }
    (destination / "split_metadata.json").write_text(json.dumps(metadata, indent=2))
    (destination / "temporal_folds.json").write_text(json.dumps(folds, indent=2))
    (destination / "pipeline_report.json").write_text(
        json.dumps(
            {
                "split_rows": row_counts,
                "split_households": {key: len(value) for key, value in splits.items()},
            },
            indent=2,
        )
    )
    LOGGER.info("Preprocessing complete; wrote split artifacts to %s", destination)
    return metadata
