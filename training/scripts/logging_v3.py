"""Shared, privacy-safe logging for the survey-only v3 pipeline."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging once for v3 command-line entry points."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"survey_v3.{name}")
