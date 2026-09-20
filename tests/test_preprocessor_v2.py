"""
Integration tests for training/scripts/preprocessor_v2.py (Synthetic
Generation v2 preprocessing/splitting).

Covers:
    P0 - v1 preprocessor.py untouched guard still green (file + functions exist)
    P1 - Full pipeline smoke test: produces train/val/test + split_metadata +
         temporal_folds, matching v1's output layout
    P2 - split_metadata.json stamps synth_version == "2.1.0"
    P3 - Splitting is deterministic given the same seed
    P4 - Split ratios are respected (stratified by pfp_label, same algorithm as v1)
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "training" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from preprocessor_v2 import run_preprocessing_v2  # noqa: E402

# ---------------------------------------------------------------------------
# P0 — v1 preprocessor.py untouched guard still green
# ---------------------------------------------------------------------------


def test_p0_v1_preprocessor_still_has_split_and_fold_functions() -> None:
    src = (SCRIPTS_DIR / "preprocessor.py").read_text()
    assert "def split_personas(" in src
    assert "def generate_temporal_folds(" in src
    assert "def export_results(" in src
    assert 'SYNTH_OUTPUT_DIR = "synth/"' in src


# ---------------------------------------------------------------------------
# P1 — Full pipeline smoke test
# ---------------------------------------------------------------------------


def _run_smoke_pipeline(tmp_path: Path, seed: int = 42, personas_per_archetype: int = 50) -> Path:
    synth_dir = tmp_path / f"synth_v2_{seed}"
    output_dir = tmp_path / f"processed_v2_{seed}"

    run_preprocessing_v2(
        input_path="unused-because-skip-fies",
        output_dir=str(output_dir),
        synth_output_dir=str(synth_dir),
        personas_per_archetype=personas_per_archetype,
        num_months=6,
        seed=seed,
        skip_fies=True,
        inject_anomalies_flag=False,
    )
    return output_dir


def test_p1_pipeline_produces_v1_shaped_output_layout(tmp_path: Path) -> None:
    output_dir = _run_smoke_pipeline(tmp_path)

    for name in (
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "split_metadata.json",
        "temporal_folds.json",
        "feature_columns.json",
        "pipeline_report.json",
    ):
        assert (output_dir / name).is_file(), f"missing {name}"


def test_p1_pipeline_writes_synth_v2_artifacts(tmp_path: Path) -> None:
    synth_dir = tmp_path / "synth_v2_artifacts"
    output_dir = tmp_path / "processed_v2_artifacts"

    run_preprocessing_v2(
        input_path="unused-because-skip-fies",
        output_dir=str(output_dir),
        synth_output_dir=str(synth_dir),
        personas_per_archetype=50,
        num_months=6,
        seed=42,
        skip_fies=True,
        inject_anomalies_flag=False,
    )

    assert (synth_dir / "personas.parquet").is_file()
    assert (synth_dir / "transactions.parquet").is_file()
    assert (synth_dir / "monthly_summaries.parquet").is_file()
    assert (synth_dir / "synthesis_report.json").is_file()

    report = json.loads((synth_dir / "synthesis_report.json").read_text())
    assert report["synth_version"] == "2.1.0"


# ---------------------------------------------------------------------------
# P2 — split_metadata.json stamps synth_version == "2.1.0"
# ---------------------------------------------------------------------------


def test_p2_split_metadata_stamps_synth_version(tmp_path: Path) -> None:
    output_dir = _run_smoke_pipeline(tmp_path)
    split_metadata = json.loads((output_dir / "split_metadata.json").read_text())
    assert split_metadata["synth_version"] == "2.1.0"


# ---------------------------------------------------------------------------
# P3 — Deterministic given the same seed
# ---------------------------------------------------------------------------


def test_p3_deterministic_split_given_same_seed(tmp_path: Path) -> None:
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"

    for label, output_dir in (("a", output_a), ("b", output_b)):
        run_preprocessing_v2(
            input_path="unused-because-skip-fies",
            output_dir=str(output_dir),
            synth_output_dir=str(tmp_path / f"synth_{label}"),
            personas_per_archetype=50,
            num_months=6,
            seed=7,
            skip_fies=True,
            inject_anomalies_flag=False,
        )

    meta_a = json.loads((output_a / "split_metadata.json").read_text())
    meta_b = json.loads((output_b / "split_metadata.json").read_text())

    assert meta_a["personas"] == meta_b["personas"]
    assert meta_a["split_sizes"] == meta_b["split_sizes"]


# ---------------------------------------------------------------------------
# P4 — Split ratios are respected
# ---------------------------------------------------------------------------


def test_p4_split_ratios_respected(tmp_path: Path) -> None:
    output_dir = _run_smoke_pipeline(tmp_path, personas_per_archetype=100)
    split_metadata = json.loads((output_dir / "split_metadata.json").read_text())

    sizes = split_metadata["split_sizes"]
    total = sizes["train"] + sizes["val"] + sizes["test"]

    assert sizes["train"] / total == pytest.approx(0.70, abs=0.02)
    assert sizes["val"] / total == pytest.approx(0.15, abs=0.02)
    assert sizes["test"] / total == pytest.approx(0.15, abs=0.02)
