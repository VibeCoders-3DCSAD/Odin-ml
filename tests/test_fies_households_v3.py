from __future__ import annotations

import logging

import pandas as pd
import pytest
from training.scripts.fies_households_v3 import SurveySelectionError, select_households


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SEQ_NO": [101, 102, 103],
            "TOTEX": [1000, 1000, 1000],
            "HH_SIZE": [3, 2, 0],
            "W_REGN": [1, 2, 3],
            "FOOD": [200, 200, 200],
            "HOUSING_WATER": [150, 150, 150],
            "HEALTH": [50, 50, 50],
            "TRANSPORT": [100, 100, 100],
            "EDUCATION": [100, 100, 100],
        }
    )


def test_selection_pseudonymizes_and_reconciles_observed_categories():
    households, report = select_households(_source())

    assert len(households) == 2
    assert "SEQ_NO" not in households
    assert households["household_id_v3"].str.startswith("hhv3_").all()
    assert (
        households[["food", "housing_water", "health", "transport", "education", "other"]].sum(
            axis=1
        )
        == households["annual_total"]
    ).all()
    assert report["exclusions"]["non_positive_total_or_household_size"] == 1


def test_selection_rejects_missing_required_column():
    with pytest.raises(SurveySelectionError, match="TOTEX"):
        select_households(_source().drop(columns="TOTEX"))


def test_selection_logs_row_counts_without_source_identifiers(caplog):
    with caplog.at_level(logging.INFO, logger="survey_v3.selection"):
        select_households(_source())

    assert "Selecting survey households from 3 source rows" in caplog.messages
    assert all("101" not in message for message in caplog.messages)
