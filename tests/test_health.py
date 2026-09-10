from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "modules" in body
    assert {"pfp", "forecaster", "anomaly", "budget"} <= set(body["modules"])
    assert body["modules"]["pfp"]["model_version"] == "pfp-tier3_svm"
    assert body["modules"]["forecaster"]["model_version"] == "forecaster-tier3_sarima"
    assert body["modules"]["anomaly"]["model_version"] == "anomaly-tier1_iqr"
    assert body["modules"]["budget"]["model_version"] == "budget-scipy_linprog"
    assert "timestamp" in body["modules"]["pfp"]
