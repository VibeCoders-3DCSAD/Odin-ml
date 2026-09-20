from __future__ import annotations

from pathlib import Path

import pandas as pd
from training.scripts.generate_monthly_expenditures_v3 import generate_monthly_expenditures


def test_monthly_allocation_reconciles_every_available_year(tmp_path: Path):
    households = pd.DataFrame(
        [
            {
                "household_id_v3": "hhv3_example",
                "food": 1200.0,
                "housing_water": 600.0,
                "health": 120.0,
                "transport": 240.0,
                "education": 60.0,
                "other": 300.0,
            }
        ]
    )
    config = Path("training/config/hfce_quarterly_indices.json")
    generate_monthly_expenditures(households, config, tmp_path)
    monthly = pd.read_parquet(tmp_path / "monthly_expenditures.parquet")

    assert monthly["year_month"].min() == "2023-01"
    assert monthly["year_month"].max() == "2026-06"
    for year in (2023, 2024, 2025, 2026):
        annual = monthly[monthly["year_month"].str.startswith(str(year))]
        assert round(annual["food"].sum(), 6) == 1200.0
        assert round(annual["housing_water"].sum(), 6) == 600.0
