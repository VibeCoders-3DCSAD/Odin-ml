from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from training.scripts.compare_forecaster_v3_full import compare
from training.scripts.train_forecaster_v3_full import run


def test_full_corpus_naive_candidate_and_comparison(tmp_path: Path):
    source, output = tmp_path / "features", tmp_path / "models"
    source.mkdir()
    (source / "feature_columns.json").write_text(json.dumps({"feature_columns": ["lag_1"]}))
    for name in ("train", "test"):
        pd.DataFrame(
            {
                "household_id_v3": [name] * 3,
                "year_month": ["2023-01", "2023-02", "2023-03"],
                "lag_1": [1.0, 2.0, 3.0],
                "target_expenses": [2.0, 3.0, 4.0],
            }
        ).to_parquet(source / f"{name}.parquet", index=False)

    report = run(
        "naive",
        source,
        output,
        rf_estimators=1,
        rf_workers=1,
        lstm_batch_size=2,
        lstm_epochs=1,
        lstm_patience=1,
    )
    comparison = compare(output)

    assert report["training_scope"] == "full_household_corpus"
    assert report["test_rows"] == 3
    assert comparison["ranking_by_mae"][0]["candidate"] == "naive"


def test_full_corpus_random_forest_streams_feature_columns(tmp_path: Path):
    source, output = tmp_path / "features", tmp_path / "models"
    source.mkdir()
    (source / "feature_columns.json").write_text(
        json.dumps({"feature_columns": ["lag_1", "lag_2"]})
    )
    for name in ("train", "test"):
        pd.DataFrame(
            {
                "household_id_v3": [name] * 4,
                "year_month": ["2023-01", "2023-02", "2023-03", "2023-04"],
                "lag_1": [1.0, 2.0, 3.0, 4.0],
                "lag_2": [0.0, 1.0, 2.0, 3.0],
                "target_expenses": [2.0, 3.0, 4.0, 5.0],
            }
        ).to_parquet(source / f"{name}.parquet", index=False)

    report = run(
        "random_forest",
        source,
        output,
        rf_estimators=1,
        rf_workers=1,
        lstm_batch_size=2,
        lstm_epochs=1,
        lstm_patience=1,
    )

    assert report["candidate"] == "random_forest"
    assert report["mae"] >= 0
