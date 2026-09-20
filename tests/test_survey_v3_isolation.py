from __future__ import annotations

from pathlib import Path

import pandas as pd
from training.scripts.survey_pipeline_v3 import run


def test_v3_pipeline_writes_only_its_explicit_output_directories(tmp_path: Path):
    source = tmp_path / "fies.csv"
    pd.DataFrame(
        {
            "SEQ_NO": [1],
            "TOTEX": [1000],
            "HH_SIZE": [2],
            "W_REGN": [1],
            "FOOD": [200],
            "HOUSING_WATER": [100],
            "HEALTH": [50],
            "TRANSPORT": [100],
            "EDUCATION": [50],
        }
    ).to_csv(source, index=False)
    survey, processed = tmp_path / "survey_v3", tmp_path / "processed_v3"

    run(str(source), str(survey), "training/config/hfce_quarterly_indices.json", str(processed))

    assert {path.name for path in survey.iterdir()} == {
        "households.parquet",
        "survey_selection_report.json",
        "monthly_expenditures.parquet",
        "monthly_summaries.parquet",
        "monthly_generation_report.json",
    }
    assert {path.name for path in processed.iterdir()} == {
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "split_metadata.json",
        "temporal_folds.json",
        "pipeline_report.json",
    }
