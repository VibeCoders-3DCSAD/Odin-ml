from __future__ import annotations

import pandas as pd
from training.scripts.feature_engineering_forecaster_v3 import build_features


def test_features_normalize_against_three_strictly_prior_months():
    frame = pd.DataFrame(
        {
            "household_id_v3": ["hh", "hh", "hh", "hh"],
            "year_month": ["2023-01", "2023-02", "2023-03", "2023-04"],
            "total_expenses": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result = build_features(frame)

    april = result[result["year_month"] == "2023-04"].iloc[0]
    assert april["lag_1"] == 30.0
    assert april["user_scale"] == 20.0
    assert april["lag_1_ratio"] == 1.5
    assert april["lag_2_ratio"] == 1.0
    assert april["lag_3_ratio"] == 0.5
    assert april["rolling_std_3_ratio"] == 0.5
    assert april["target_expenses"] == 40.0
    assert april["target_ratio"] == 2.0
