"""Report the v3 release gate without asserting household-level accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from logging_v3 import configure_logging, get_logger

LOGGER = get_logger("release_gate")


def evaluate(output_dir: str | Path) -> dict:
    path = Path(output_dir) / "evaluation.json"
    LOGGER.info("Applying external-validation release gate to %s", path)
    result = json.loads(path.read_text())
    result["release_eligible"] = False
    result["release_reason"] = (
        "Real household transactions or withheld later PSA aggregate validation is required."
    )
    path.write_text(json.dumps(result, indent=2))
    LOGGER.warning("V3 release remains ineligible until external validation is recorded")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record the survey-only v3 release gate.")
    parser.add_argument("--output", default="models/forecaster_v3")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    evaluate(args.output)
