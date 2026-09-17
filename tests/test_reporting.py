from __future__ import annotations

from app.ml.reporting import family_report


def _tier_metrics(**overrides: float) -> dict:
    metrics = {
        "mae": 1.0,
        "smape": 2.0,
        "mda": 0.5,
        "rmse": 3.0,
        "mape": 4.0,
        "r2": 0.8,
        "macro_f1": 0.6,
        "accuracy": 0.7,
    }
    metrics.update(overrides)
    return metrics


def test_forecaster_report_accepts_v2_period_keys():
    report = family_report(
        "forecaster",
        {
            "n_folds": 1,
            "winner": "tier3_sarima",
            "winner_reason": "ok",
            "naive_mape": 10.0,
            "aggregate_metrics": {
                "tier3_sarima": {"mape_mean": 5.0, "mape_std": 0.1},
            },
            "fold_details": [
                {
                    "fold": 1,
                    "train_periods": ["2023-01", "2023-02"],
                    "test_periods": ["2023-04"],
                    "n_train": 10,
                    "n_test": 2,
                    "tier_results": {"tier3_sarima": _tier_metrics()},
                }
            ],
        },
    )

    assert "Train periods: ['2023-01', '2023-02']" in report["fold_sections"]
    assert "Test periods: ['2023-04']" in report["fold_sections"]


def test_forecaster_report_keeps_v1_month_keys():
    report = family_report(
        "forecaster",
        {
            "n_folds": 1,
            "winner": "tier3_sarima",
            "winner_reason": "ok",
            "fold_details": [
                {
                    "fold": 1,
                    "train_months": [1, 2, 3],
                    "test_months": [5],
                    "n_train": 10,
                    "n_test": 2,
                    "tier_results": {"tier3_sarima": _tier_metrics()},
                }
            ],
        },
    )

    assert "Train months: [1, 2, 3]" in report["fold_sections"]
    assert "Test months: [5]" in report["fold_sections"]


def test_pfp_report_accepts_v2_period_keys():
    report = family_report(
        "pfp",
        {
            "n_folds": 1,
            "winner": "tier3_svm",
            "winner_reason": "ok",
            "fold_details": [
                {
                    "fold": 1,
                    "train_periods": ["2023-01"],
                    "test_periods": ["2023-03"],
                    "n_train_personas": 8,
                    "n_test_personas": 2,
                    "tier_results": {"tier3_svm": _tier_metrics()},
                }
            ],
        },
    )

    assert "Train periods: ['2023-01']" in report["fold_sections"]
    assert "Test periods: ['2023-03']" in report["fold_sections"]
