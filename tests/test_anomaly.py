from __future__ import annotations

from tests.conftest import load_transactions


def test_anomaly_detect(client):
    txns = load_transactions()
    payload = {"user_id": "test-user-1", "transactions": txns}
    resp = client.post("/api/v1/anomaly/detect", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "anomalous_transactions" in body
    assert "overspending_transactions" in body
    assert isinstance(body["anomalous_transactions"], list)
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["status"] in ("SUCCESS", "FALLBACK")
    assert body["model_version"] == "anomaly-tier1_iqr"


def test_anomaly_detect_batch(client):
    txns = load_transactions(n=90)
    payload = {
        "requests": [
            {"user_id": "test-user-1", "transactions": txns[:45]},
            {"user_id": "test-user-2", "transactions": txns[45:]},
        ]
    }
    resp = client.post("/api/v1/anomaly/detect/batch", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert all("anomalous_transactions" in r for r in results)


def test_anomaly_overspending_detects_excess(client):
    txns = load_transactions(n=60)
    low_budget = 1.0
    payload = {
        "user_id": "test-user-3",
        "detection_type": "OVERSPENDING",
        "transactions": txns,
        "budget_allocations": [{"category_id": "food", "budget_amount": low_budget}],
    }
    resp = client.post("/api/v1/anomaly/detect", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert any(o["category"] == "food" for o in body["overspending_transactions"])
    assert body["status"] in ("SUCCESS", "FALLBACK")


def test_anomaly_detect_requires_threshold():
    """Without a registry-supplied threshold, detect() must refuse to score."""
    import numpy as np
    import pytest
    from app.ml.models import IQRDetector
    from app.models.registry import ModuleModel
    from app.schemas.anomaly import AnomalyRequest
    from app.services.anomaly_service import detect

    txns = load_transactions()
    rng = np.random.RandomState(0)
    scorer = IQRDetector(iqr_multiplier=1.5).fit(rng.normal(0.0, 1.0, size=(200, 4)))
    model = ModuleModel(
        module="anomaly",
        model=scorer,
        evaluation={},
        feature_columns=["mean_income_rolling"],
    )
    request = AnomalyRequest(user_id="threshold-test", transactions=txns)
    with pytest.raises(RuntimeError, match="no operating threshold"):
        detect(model, request)


def test_anomaly_detect_scores_are_raw_not_per_request_normalized():
    """detect() must compare RAW model scores against the raw-calibrated
    threshold. Per-request min-max normalization was removed because it rescales
    scores per request and breaks the raw score-vs-threshold unit match."""
    import numpy as np
    from app.models.registry import ModuleModel
    from app.schemas.anomaly import AnomalyRequest
    from app.services.anomaly_service import detect

    class RawScorer:
        """Returns fixed raw scores (0.5..1.0) independent of X shape."""

        def __init__(self, n: int):
            self.values = 0.5 + 0.025 * np.arange(n, dtype=float)

        def score(self, X):
            return self.values[: len(X)]

    txns = load_transactions(n=20)
    model = ModuleModel(
        module="anomaly",
        model=RawScorer(len(txns)),
        evaluation={},
        feature_columns=["mean_income_rolling"],
        threshold=0.90,
    )
    request = AnomalyRequest(user_id="raw-contract-test", transactions=txns)
    anomalous = detect(model, request)

    # raw units: any score >= 0.90 (raw threshold) is flagged. A per-request
    # [0,1] renormalization would squash 0.5..1.0 to 0..1 and shift the cutoff,
    # flagging fewer transactions (2 instead of 4 in this synthetic spread).
    expected_raw = np.asarray([0.5 + 0.025 * i for i in range(len(txns))])
    flagged = [a for a in anomalous if a.anomaly_score >= 0.90]
    assert len(flagged) == int(np.sum(expected_raw >= 0.90))
    for i, a in enumerate(anomalous):
        assert round(float(expected_raw[i]), 4) == a.anomaly_score


def test_adaptive_threshold_detector_scores():
    """AdaptiveThresholdDetector (awarded Tier 2 candidate) is an in-scope,
    app-unpicklable scorer: it must live in app.ml.models and behave like the
    other statistical detectors (fit -> [0, )-bounded score)."""
    import numpy as np
    from app.ml.models import AdaptiveThresholdDetector, IQRDetector

    rng = np.random.RandomState(42)
    X = rng.normal(0.0, 1.0, size=(500, 4))

    adaptive = AdaptiveThresholdDetector(iqr_multiplier=1.5).fit(X)
    scores = adaptive.score(X)
    assert scores.shape == (500,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()

    outlier = X.max(axis=0) * 5.0
    assert adaptive.score(outlier[None, :])[0] > scores.mean()

    baseline = IQRDetector(iqr_multiplier=1.5).fit(X)
    assert np.isfinite(baseline.score(X)).all()
