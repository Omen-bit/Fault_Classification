"""The trace behind "How it works" must describe the real answer, never a different one."""
import csv

import pytest

from fault.decide import classify, explain
from fault.model import Scorer

with open("data/training/tickets.csv", encoding="utf-8", newline="") as f:
    TICKETS = list(csv.DictReader(f))


@pytest.fixture(scope="module")
def scorer():
    return Scorer.load("model")


def test_tracing_never_changes_the_answer(scorer):
    for r in TICKETS:
        args = (r["complaint_text"], r["product_category"], r["product_age_days"] or None)
        plain = classify(scorer, *args)
        traced, trace = explain(scorer, *args)
        assert plain == traced, r["ticket_id"]
        fired = [rung["rule"] for rung in trace["ladder"] if rung["state"] == "fired"]
        assert fired == [plain["rule"]], (r["ticket_id"], fired, plain["rule"])
        rung = next(x for x in trace["ladder"] if x["state"] == "fired")
        assert rung["condition_true"], (r["ticket_id"], rung)
        assert trace["confidence"]["final"] == plain["confidence"]


def test_trace_contents_for_two_symptoms(scorer):
    _, trace = explain(scorer, "Hi team, very loud banging noise. Also the door latch is broken. Thanks.", "washing_machine", 400)
    assert set(trace["symptoms"]["found"]) == {"noise_or_vibration", "door_seal_or_latch"}
    verdicts = {s["text"]: s["verdict"] for s in trace["sentences"]}
    assert verdicts["very loud banging noise"] == "symptom" and verdicts["thanks"] == "no symptom"
    assert all(len(s["top"]) == 3 for s in trace["sentences"])
    states = {x["rule"]: x["state"] for x in trace["ladder"]}
    assert states["C5"] == "fired" and states["C1"] == "no" and states["model"] == "not reached"


def test_trace_shows_typo_repair_and_skips_model_on_safety(scorer):
    result, trace = explain(scorer, "smkoe started coming out of the microwve", "microwave", 300)
    assert result["escalate"] and ["smkoe", "smoke"] in trace["safety"]["typo_repairs"]
    assert trace["model_skipped"] is True and trace["sentences"] == []
    assert trace["ladder"][0]["state"] == "fired"


def test_explain_endpoint_matches_classify(client):
    body = {"ticket_id": "T-9", "product_category": "water_heater", "product_age_days": 800,
            "complaint_text": "Water is not hot enough for a shower. I can smell gas whenever the geyser is on."}
    plain = client.post("/classify", json=body).json()
    before = client.get("/metrics").json()["requests"]
    out = client.post("/explain", json=body)
    assert out.status_code == 200
    data = out.json()
    assert data["response"] == plain and data["trace_available"] is True
    assert data["trace"]["safety"]["c2"] is True and data["trace"]["cleanup"]["product"] == "water_heater"
    assert client.get("/metrics").json()["requests"] == before          # explain is not counted


@pytest.mark.parametrize("body", [{}, [], {"complaint_text": None}, {"complaint_text": 5, "product_category": 9}])
def test_explain_accepts_odd_input(client, body):
    r = client.post("/explain", json=body)
    assert r.status_code == 200 and set(r.json()) == {"response", "trace", "trace_available"}


def test_classify_still_has_exactly_five_fields(client):
    out = client.post("/classify", json={"ticket_id": "T", "product_category": "microwave", "complaint_text": "it sparks"}).json()
    assert set(out) == {"ticket_id", "fault_category", "confidence", "escalate", "reason"}
