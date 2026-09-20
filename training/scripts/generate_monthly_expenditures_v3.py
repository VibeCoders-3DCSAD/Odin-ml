"""Render HFCE-profiled monthly expenditures from observed annual FIES anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from logging_v3 import get_logger
from temporal_disaggregation import build_year_schedule

LOGGER = get_logger("monthly_generation")

TIMELINE = ("2023-01", "2026-06")
CATEGORIES = ("food", "housing_water", "health", "transport", "education", "other")
HFCE_NAMES = {category: category for category in CATEGORIES}
HFCE_NAMES["housing_water"] = "housing"


def _fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_monthly_expenditures(
    households: pd.DataFrame, hfce_path: str | Path, output_dir: str | Path
) -> None:
    """Allocate observed annual categories in bounded Parquet batches.

    A wide household-month table avoids materializing 41 million long-format
    category records for the full PUF while retaining every category allocation.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Generating monthly allocations for %d households", len(households))
    expenditures_path = destination / "monthly_expenditures.parquet"
    summaries_path = destination / "monthly_summaries.parquet"
    expenditure_writer: pq.ParquetWriter | None = None
    summary_writer: pq.ParquetWriter | None = None
    batch: list[dict[str, object]] = []
    weights_by_year = {
        year: build_year_schedule(dict.fromkeys(HFCE_NAMES.values(), 1.0), year, hfce_path)
        for year in range(2023, 2027)
    }

    def flush() -> None:
        nonlocal expenditure_writer, summary_writer
        if not batch:
            return
        monthly = pd.DataFrame(batch)
        expenditure_table = pa.Table.from_pandas(monthly, preserve_index=False)
        summaries = monthly[["household_id_v3", "year_month", "total_expenses"]]
        summary_table = pa.Table.from_pandas(summaries, preserve_index=False)
        if expenditure_writer is None:
            expenditure_writer = pq.ParquetWriter(expenditures_path, expenditure_table.schema)
            summary_writer = pq.ParquetWriter(summaries_path, summary_table.schema)
        expenditure_writer.write_table(expenditure_table)
        assert summary_writer is not None
        summary_writer.write_table(summary_table)
        LOGGER.info("Wrote monthly allocation batch with %d household-month rows", len(monthly))
        batch.clear()

    for household_index, household in enumerate(households.itertuples(index=False), start=1):
        annual = {
            HFCE_NAMES[category]: float(getattr(household, category)) for category in CATEGORIES
        }
        for year in range(2023, 2027):
            weights = weights_by_year[year]
            for month in next(iter(weights.values())):
                row = {
                    "household_id_v3": household.household_id_v3,
                    "year_month": f"{year}-{month:02d}",
                    "source": "synthetic_monthly_allocation",
                    **{
                        category: annual[HFCE_NAMES[category]]
                        * weights[HFCE_NAMES[category]][month]
                        for category in CATEGORIES
                    },
                }
                row["total_expenses"] = sum(float(row[category]) for category in CATEGORIES)
                batch.append(row)
                if len(batch) >= 10_000:
                    flush()
        if household_index % 10_000 == 0:
            LOGGER.info("Allocated %d of %d households", household_index, len(households))
    flush()
    if expenditure_writer is not None:
        expenditure_writer.close()
    if summary_writer is not None:
        summary_writer.close()
    report = {
        "timeline": TIMELINE,
        "hfce_config_fingerprint_sha256": _fingerprint(hfce_path),
        "observed_fields": ["household_id_v3", *CATEGORIES],
        "synthetic_fields": ["year_month", "amount", "total_expenses"],
        "accounting": "Each available year/category schedule reconciles to the observed annual benchmark.",
    }
    (destination / "monthly_generation_report.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("Monthly allocation complete; wrote artifacts to %s", destination)
