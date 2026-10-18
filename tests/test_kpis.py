"""Live KPIs: correct arithmetic, bounded memory, and no complaint text anywhere."""
from fault.kpis import WINDOW, Kpis


def result(rule="model", category="motor_or_drive", escalate=False, confidence=0.9, reason="The only symptom described matches motor_or_drive."):
    return {"rule": rule, "fault_category": category, "escalate": escalate, "confidence": confidence, "reason": reason}


def test_empty_snapshot_is_safe():
    s = Kpis().snapshot()
    assert s["requests"] == 0 and s["latency_ms"]["p95"] == 0.0 and s["within_budget"] is None
    assert s["seconds_since_last_request"] is None and len(s["per_minute"]) == 30
    assert sum(b["count"] for b in s["latency_histogram"]) == 0


def test_counts_rates_and_percentiles():
    k = Kpis()
    for ms in range(1, 101):                                    # 1..100 ms
        k.record(result(), ms, "washing_machine", 120, True)
    k.record(result("C1", "electrical_or_control_board", True, 0.97, "Rule C1: the complaint reports smoke."), 2000, "microwave", 3, False)
    k.record(result("model", "not_heating", True, 0.3, "The only symptom described matches not_heating; rule C2: a gas smell on this product requires escalation."),
             8, None, 4000, True)
    s = k.snapshot()
    assert s["requests"] == 102 and s["escalations"] == 2 and s["window"] == 102
    assert s["escalation"] == {"rate": round(2 / 102, 4), "count": 2, "c1_safety": 1, "c2_gas_smell": 1}
    assert s["latency_ms"]["max"] == 2000 and 45 <= s["latency_ms"]["p50"] <= 55 and s["latency_ms"]["p95"] >= 95
    assert s["within_budget"] == round(101 / 102, 4)
    assert sum(b["count"] for b in s["latency_histogram"]) == 102 and s["latency_histogram"][-1]["count"] == 1
    assert sum(b["count"] for b in s["confidence"]["histogram"]) == 102
    assert s["confidence"]["low_share"] == round(1 / 102, 4)
    assert s["by_product"]["unrecognised"] == 1 and s["by_rule"]["C1"] == 1
    q = s["input_quality"]
    assert q["outside_5_to_2000_characters"] == round(2 / 102, 4) and q["age_unknown_share"] == round(1 / 102, 4)
    assert s["per_minute"][-1]["requests"] == 102 and s["per_minute"][-1]["escalations"] == 2
    assert s["product_by_category"]["washing_machine"]["motor_or_drive"] == 100
    assert sum(sum(row.values()) for row in s["product_by_category"].values()) == 102
    recent = s["recent_escalations"]
    assert len(recent) == 2 and {x["rule"] for x in recent} == {"C1", "C2"}
    assert set(recent[0]) == {"seconds_ago", "product", "category", "rule"}


def test_rejected_and_fallback_counters():
    k = Kpis()
    k.rejected("invalid_json")
    k.rejected("too_large")
    k.record(result("fallback", "other", False, 0.0, "Internal error; safe fallback using the safety rules only."), 1, None, 0, False)
    q = k.snapshot()["input_quality"]
    assert q["rejected_requests"] == 2 and q["fallback_answers"] == 1


def test_memory_is_bounded_and_broken_records_are_ignored():
    k = Kpis()
    for _ in range(WINDOW + 50):
        k.record(result(), 1, "microwave", 10, True)
    k.record({"nonsense": True}, 1, None, 0, False)             # must not raise
    s = k.snapshot()
    assert s["window"] == WINDOW and s["requests"] == WINDOW + 50


def test_per_minute_series_places_old_requests_correctly():
    k = Kpis()
    k.record(result(), 5, "microwave", 10, True)
    later = k.recent[-1].ts + 4 * 60 + 5
    s = k.snapshot(now=later)
    hit = [m for m in s["per_minute"] if m["requests"]]
    assert len(hit) == 1 and hit[0]["minutes_ago"] == 4 and s["throughput"]["per_minute_last_5_min"] == 0.2


def test_api_counts_rejected_requests_and_keeps_text_out(client):
    before = client.get("/metrics").json()["input_quality"]["rejected_requests"]
    client.post("/classify", content=b"{broken", headers={"content-type": "application/json"})
    client.post("/classify", json={"ticket_id": "secret-id-9931", "product_category": "fridge", "complaint_text": "qqzz marker text not cooling"})
    m = client.get("/metrics")
    body = m.json()
    assert body["input_quality"]["rejected_requests"] == before + 1
    assert "qqzz" not in m.text and "secret-id-9931" not in m.text
    assert body["by_product"].get("refrigerator", 0) >= 1
