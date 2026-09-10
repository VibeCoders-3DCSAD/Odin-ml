from __future__ import annotations


def _budget_payload():
    return {
        "request_id": "req-budget-1",
        "user_id": "test-user-1",
        "available_funds": 25000.0,
        "period": {"start": "2024-01-01", "end": "2024-01-31"},
        "categories": [
            {
                "category_id": "essentials_food",
                "restriction_level": "PROTECTED",
                "floor": 6000.0,
                "ceiling": 9000.0,
                "priority_weight": 0.4,
                "current_spend": 6800.0,
            },
            {
                "category_id": "essentials_rent",
                "restriction_level": "LOCKED",
                "floor": 8000.0,
                "ceiling": 8000.0,
                "priority_weight": 0.3,
                "current_spend": 8000.0,
            },
            {
                "category_id": "discretionary_leisure",
                "restriction_level": "FREE",
                "floor": 0.0,
                "ceiling": 5000.0,
                "priority_weight": 0.2,
                "current_spend": 2000.0,
            },
            {
                "category_id": "savings",
                "restriction_level": "FREE",
                "floor": 0.0,
                "ceiling": 6000.0,
                "priority_weight": 0.1,
                "current_spend": 0.0,
            },
        ],
        "target_ratios": {
            "essentials_food": 0.28,
            "essentials_rent": 0.32,
            "discretionary_leisure": 0.15,
            "savings": 0.25,
        },
    }


def test_budget_recommend(client):
    resp = client.post("/api/v1/budget/recommend", json=_budget_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] == "req-budget-1"
    rec = body["recommendation"]
    allocations = {a["category_id"]: a["amount"] for a in rec["allocations"]}
    total = sum(allocations.values())
    assert abs(total - 25000.0) < 1.0
    assert allocations["essentials_rent"] == 8000.0  # LOCKED
    assert rec["utilization_rate"] > 0.99
    assert rec["feasibility"] in ("FEASIBLE", "REDUCED", "INFEASIBLE")
    assert body["metadata"]["model_version"] == "budget-scipy_linprog"
    assert body["metadata"]["strategy_used"] == "tier2_lp"


def test_budget_rejects_bad_restriction(client):
    payload = _budget_payload()
    payload["categories"][0]["ceiling"] = 5000.0  # ceiling below floor on PROTECTED
    resp = client.post("/api/v1/budget/recommend", json=payload)
    assert resp.status_code == 422
