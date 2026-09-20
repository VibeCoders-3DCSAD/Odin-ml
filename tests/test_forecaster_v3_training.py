from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from training.scripts.train_forecaster_v3 import HAS_TORCH, train_and_evaluate


def test_v3_evaluation_records_unreleased_synthetic_target_gate(tmp_path: Path):
    source, output = tmp_path / "features", tmp_path / "models"
    source.mkdir()
    (source / "feature_columns.json").write_text(json.dumps({"feature_columns": ["lag_1"]}))
    for name in ("train", "val", "test"):
        pd.DataFrame(
            {
                "household_id_v3": [name] * 3,
                "year_month": ["2023-01", "2023-02", "2023-03"],
                "lag_1": [1.0, 2.0, 3.0],
                "target_expenses": [2.0, 3.0, 4.0],
            }
        ).to_parquet(source / f"{name}.parquet", index=False)

    result = train_and_evaluate(source, output, run_rf=False, run_torch=False)

    assert result["served_forecaster_unchanged"] is True
    assert result["external_validation_gate"] == "not_satisfied"
    assert result["candidate_mae"]["naive"] > 0
    assert result["candidates_skipped"] == {"random_forest": True, "torch": True}
    assert result["resource_limits"]["rf_workers"] == 1


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is not installed")
def test_v3_training_can_run_the_lstm_candidate(tmp_path: Path):
    source, output = tmp_path / "features", tmp_path / "models"
    source.mkdir()
    (source / "feature_columns.json").write_text(json.dumps({"feature_columns": ["lag_1"]}))
    for name in ("train", "val", "test"):
        pd.DataFrame(
            {
                "household_id_v3": [name] * 4,
                "year_month": ["2023-01", "2023-02", "2023-03", "2023-04"],
                "lag_1": [1.0, 2.0, 3.0, 4.0],
                "target_expenses": [2.0, 3.0, 4.0, 5.0],
            }
        ).to_parquet(source / f"{name}.parquet", index=False)

    result = train_and_evaluate(source, output, run_rf=False)

    assert "lstm" in result["candidate_mae"]
    assert result["candidates_skipped"]["torch"] is False


def test_v3_training_rejects_invalid_resource_limits(tmp_path: Path):
    with pytest.raises(ValueError, match="must be positive"):
        train_and_evaluate(tmp_path, tmp_path, sample_modulus=0)
