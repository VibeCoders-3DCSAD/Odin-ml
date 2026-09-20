"""Select and validate survey-only 2023 FIES household expenditure anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fies_columns import FIESColumns
from logging_v3 import get_logger

LOGGER = get_logger("selection")

SELECTION_CONTRACT_VERSION = "3.0.0"
CATEGORY_IDS = {
    "food": "FOOD",
    "housing_water": "HOUSING_WATER",
    "health": "HEALTH",
    "transport": "TRANSPORT",
    "education": "EDUCATION",
}
REQUIRED_IDS = ("SEQ_NO", "TOTEX", "HH_SIZE", "W_REGN", *CATEGORY_IDS.values())


class SurveySelectionError(ValueError):
    """Raised when the source cannot satisfy the locked survey contract."""


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_names(frame: pd.DataFrame) -> dict[str, str]:
    fies = FIESColumns(frame, strict=False)
    columns: dict[str, str] = {}
    for field in REQUIRED_IDS:
        info = fies.resolved_columns[field]
        if not info.found:
            raise SurveySelectionError(f"missing required FIES column: {field}")
        columns[field] = info.alternative or info.csv_name
    return columns


def household_id_v3(sequence_number: object) -> str:
    """Return a stable, non-reversible public artifact identifier."""
    value = str(sequence_number).strip().encode("utf-8")
    return f"hhv3_{hashlib.sha256(value).hexdigest()[:24]}"


def select_households(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the documented loss-only cleaning policy to observed FIES values."""
    LOGGER.info("Selecting survey households from %d source rows", len(frame))
    columns = _column_names(frame)
    working = frame.copy()
    report: dict[str, Any] = {"input_rows": len(working), "exclusions": {}}
    for field, column in columns.items():
        working[field] = pd.to_numeric(working[column], errors="coerce")

    duplicate = working["SEQ_NO"].duplicated(keep=False) | working["SEQ_NO"].isna()
    report["exclusions"]["duplicate_or_missing_seq_no"] = int(duplicate.sum())
    working = working.loc[~duplicate].copy()
    required_numeric = ["TOTEX", "HH_SIZE", *CATEGORY_IDS.values()]
    missing = working[required_numeric].isna().any(axis=1)
    report["exclusions"]["missing_required_numeric"] = int(missing.sum())
    working = working.loc[~missing].copy()
    invalid = (working["TOTEX"] <= 0) | (working["HH_SIZE"] <= 0)
    report["exclusions"]["non_positive_total_or_household_size"] = int(invalid.sum())
    working = working.loc[~invalid].copy()
    negative = (working[list(CATEGORY_IDS.values())] < 0).any(axis=1)
    report["exclusions"]["negative_category_expenditure"] = int(negative.sum())
    working = working.loc[~negative].copy()

    category_columns = list(CATEGORY_IDS.values())
    working["other"] = working["TOTEX"] - working[category_columns].sum(axis=1)
    negative_residual = working["other"] < -0.005
    report["exclusions"]["negative_other_residual"] = int(negative_residual.sum())
    working = working.loc[~negative_residual].copy()
    working["other"] = working["other"].clip(lower=0)
    renamed = working.rename(columns={source: target for target, source in CATEGORY_IDS.items()})
    output = renamed[
        ["SEQ_NO", "W_REGN", "HH_SIZE", "TOTEX", *CATEGORY_IDS.keys(), "other"]
    ].rename(columns={"W_REGN": "region", "HH_SIZE": "household_size", "TOTEX": "annual_total"})
    output.insert(0, "household_id_v3", output.pop("SEQ_NO").map(household_id_v3))
    output = output.sort_values("household_id_v3").reset_index(drop=True)
    if not output["household_id_v3"].is_unique:
        raise SurveySelectionError("pseudonymous household IDs are not unique")
    categories = [*CATEGORY_IDS.keys(), "other"]
    if not (output[categories].sum(axis=1).sub(output["annual_total"]).abs() <= 0.005).all():
        raise SurveySelectionError("annual expenditure categories do not reconcile to TOTEX")
    report["selected_rows"] = len(output)
    LOGGER.info("Selected %d eligible households", len(output))
    report["geography_distribution"] = output["region"].value_counts(dropna=False).to_dict()
    report["missingness"] = {column: int(output[column].isna().sum()) for column in output.columns}
    report["category_distribution"] = {
        category: output[category].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        for category in categories
    }
    return output, report


def load_and_select(input_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    LOGGER.info("Loading FIES household summary from %s", path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    households, report = select_households(frame)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    households.to_parquet(destination / "households.parquet", index=False)
    report.update(
        {
            "selection_contract_version": SELECTION_CONTRACT_VERSION,
            "source_file": path.name,
            "source_fingerprint_sha256": file_fingerprint(path),
            "weighted_diagnostics": "RFACT is not used for training or diagnostics in v3.",
        }
    )
    (destination / "survey_selection_report.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    LOGGER.info("Wrote selected households and selection report to %s", destination)
    return households
