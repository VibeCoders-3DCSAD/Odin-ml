from __future__ import annotations

import pandas as pd
from training.scripts.feature_engineering_forecaster_v3 import build_features


def test_features_only_use_prior_months_and_target_is_next_month():
    frame = pd.DataFrame(
        {
            "household_id_v3": ["hh", "hh", "hh", "hh"],
            "year_month": ["2023-01", "2023-02", "2023-03", "2023-04"],
            "total_expenses": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result = build_features(frame)

    march = result[result["year_month"] == "2023-03"].iloc[0]
    assert march["lag_1"] == 20.0
    assert march["rolling_mean_3"] == 15.0
    assert march["target_expenses"] == 40.0
