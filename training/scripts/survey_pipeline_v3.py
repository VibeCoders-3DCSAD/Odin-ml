"""Command-line orchestration for the additive survey-only v3 data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fies_households_v3 import load_and_select
from generate_monthly_expenditures_v3 import generate_monthly_expenditures
from logging_v3 import configure_logging, get_logger
from preprocessor_v3 import preprocess

LOGGER = get_logger("pipeline")


def run(
    input_path: str, output_root: str, hfce_path: str, processed_dir: str, seed: int = 42
) -> None:
    root = Path(output_root)
    LOGGER.info("Starting survey-only v3 pipeline")
    households = load_and_select(input_path, root)
    generate_monthly_expenditures(households, hfce_path, root)
    preprocess(root / "monthly_summaries.parquet", processed_dir, seed)
    metadata_path = Path(processed_dir) / "split_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    selection = json.loads((root / "survey_selection_report.json").read_text())
    generation = json.loads((root / "monthly_generation_report.json").read_text())
    metadata["source_fingerprint_sha256"] = selection["source_fingerprint_sha256"]
    metadata["hfce_config_fingerprint_sha256"] = generation["hfce_config_fingerprint_sha256"]
    metadata_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Survey-only v3 pipeline complete; selected %d households", len(households))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate survey-only forecast v3 data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="survey_v3")
    parser.add_argument("--processed-output", default="training/datasets/processed_v3")
    parser.add_argument("--hfce", default="training/config/hfce_quarterly_indices.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(args.input, args.output, args.hfce, args.processed_output, args.seed)
