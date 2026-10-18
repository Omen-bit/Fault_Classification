"""API contract from the brief, lenient input, and the dashboard endpoints."""
import socket

import pytest

from fault.labels import LABELS

BODY = {"ticket_id": "T-1", "product_category": "washing_machine",
        "complaint_text": "drum never turns", "product_age_days": 410}

BRIEF_EXAMPLES = [
    ({"ticket_id": "T-1001", "product_category": "washing_machine", "product_age_days": 410,
      "complaint_text": "machine fills with water then just sits there, drum never turns, display shows E21"},
     "motor_or_drive", False),
    ({"ticket_id": "T-1002", "product_category": "oven_or_cooktop", "product_age_days": 95,
      "complaint_text": "There was a burning smell and a spark from the back when I switched it on, it is off now"},
     "electrical_or_control_board", True),
    ({"ticket_id": "T-1003", "product_category": "refrigerator", "product_age_days": 1,
      "complaint_text": "delivered yesterday, the doors are not level and one does not close properly"},
     "installation_or_setup", False),
]


def test_network_is_blocked():
    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 80), timeout=1)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_classify_contract(client):
    r = client.post("/classify", json=BODY)
    out = r.json()
    assert r.status_code == 200
    assert set(out) == {"ticket_id", "fault_category", "confidence", "escalate", "reason"}
    assert out["ticket_id"] == "T-1"
    assert out["fault_category"] in LABELS
    assert isinstance(out["confidence"], float) and 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["escalate"], bool)
    assert isinstance(out["reason"], str) and out["reason"].strip().endswith(".")


@pytest.mark.parametrize("body,category,escalate", BRIEF_EXAMPLES, ids=["T-1001", "T-1002", "T-1003"])
def test_brief_examples(client, body, category, escalate):
    out = client.post("/classify", json=body).json()
    assert (out["fault_category"], out["escalate"]) == (category, escalate), out


@pytest.mark.parametrize("body", [
    {}, [], "text", 5, None,
    {"ticket_id": "T", "complaint_text": None},
    {"ticket_id": "T", "complaint_text": 12345, "product_category": 7},
    {"ticket_id": "T", "complaint_text": ["a"], "product_age_days": "abc"},
    {"ticket_id": "T", "product_category": "toaster", "complaint_text": "x" * 5000},
    {"ticket_id": "T", "complaint_text": "ok", "product_age_days": -3},
    {"ticket_id": "T", "complaint_text": "फ्रिज ठंडा नहीं कर रहा 🙁", "product_category": "refrigerator"},
    {"ticket_id": "T", "complaint_text": "   ", "product_category": "microwave"}])
def test_odd_input_never_errors(client, body):
    r = client.post("/classify", json=body)
    assert r.status_code == 200 and r.json()["fault_category"] in LABELS


def test_ticket_id_echo(client):
    assert client.post("/classify", json={**BODY, "ticket_id": "ÄB-001 x"}).json()["ticket_id"] == "ÄB-001 x"
    assert client.post("/classify", json={**BODY, "ticket_id": 1001}).json()["ticket_id"] == "1001"


def test_invalid_json_is_400(client):
    r = client.post("/classify", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_age_forms(client):
    text = "The oven is not level after installation and rocks on the floor."
    cat = lambda age: client.post("/classify", json={"ticket_id": "T", "product_category": "oven_or_cooktop",
                                                     "complaint_text": text, "product_age_days": age}).json()
    for young in (0, 30, "12", 12.0):
        assert cat(young)["reason"].startswith("Rule C3"), young
    for not_young in (31, None, -5, "abc", True):
        assert not cat(not_young)["reason"].startswith("Rule C3"), not_young


def test_product_written_differently_still_gets_gas_rule(client):
    body = {"ticket_id": "T", "complaint_text": "Water is not hot. I can smell gas whenever it is on."}
    assert client.post("/classify", json={**body, "product_category": "Water Heater"}).json()["escalate"] is True
    assert client.post("/classify", json={**body, "product_category": "geyser"}).json()["escalate"] is True
    assert client.post("/classify", json={**body, "product_category": "refrigerator"}).json()["escalate"] is False


def test_metrics_shape_and_privacy(client):
    secret = "zzqx-unique-marker the drum will not spin"
    client.post("/classify", json={"ticket_id": "T", "product_category": "washing_machine", "complaint_text": secret})
    m = client.get("/metrics")
    body = m.json()
    assert m.status_code == 200
    assert set(body) >= {"requests", "escalations", "by_rule", "by_category", "latency_ms", "uptime_s", "model"}
    assert body["requests"] >= 1 and set(body["latency_ms"]) >= {"p50", "p95", "p99", "max"}
    assert "zzqx" not in m.text


def test_scorecard_endpoint(client):
    r = client.get("/scorecard")
    assert r.status_code == 200 and isinstance(r.json(), dict)


def test_docs_page_disabled(client):
    assert client.get("/docs").status_code == 404 and client.get("/redoc").status_code == 404


def test_latency_budget(client):
    import time
    texts = ["the drum will not spin. " * k for k in (1, 5, 40)] * 40
    times = []
    for t in texts:
        t0 = time.perf_counter()
        client.post("/classify", json={"ticket_id": "T", "product_category": "washing_machine", "complaint_text": t})
        times.append(time.perf_counter() - t0)
    times.sort()
    # the brief allows 1 s per request; a single slow outlier on a busy machine must not fail the suite
    assert times[len(times) // 2] < 0.1 and times[int(len(times) * 0.95)] < 0.5


def test_health_answers_head_and_page_is_not_cached(client):
    assert client.head("/health").status_code == 200
    assert "no-store" in client.get("/").headers.get("cache-control", "")


def test_version_endpoint_matches_the_page(client):
    import re
    api = client.get("/version").json()["api"]
    page = client.get("/").text
    assert isinstance(api, int) and re.search(r"const PAGE_API = (\d+);", page).group(1) == str(api)


def test_oversized_body_is_refused_and_counted(client):
    before = client.get("/metrics").json()["input_quality"]["rejected_requests"]
    r = client.post("/classify", json={"complaint_text": "a " * 40_000})
    assert r.status_code == 413
    assert client.get("/metrics").json()["input_quality"]["rejected_requests"] == before + 1


def test_largest_allowed_body_is_fast(client):
    import time
    body = {"ticket_id": "T", "complaint_text": "the fan is noisy. " * 3400}        # about 61 KB, just under the cap
    t0 = time.perf_counter()
    r = client.post("/classify", json=body)
    assert r.status_code == 200 and time.perf_counter() - t0 < 1.0
