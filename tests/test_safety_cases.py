"""Hand-written safety edge cases: plain, typo, capitals, Hinglish and call-centre shorthand.

A missed escalation is the costliest mistake in the brief, so every phrasing in
data/testing/safety_cases.py is checked on its own, through the rules and through the API.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "testing"))
import safety_cases as cases  # noqa: E402

from fault.rules import gas_smell, normalize, safety_terms  # noqa: E402

C1 = [(family, text) for family, texts in cases.POSITIVE.items() for text in texts]


def test_enough_cases():
    assert len(C1) >= 80 and len(cases.GAS) >= 15 and len(cases.NEGATIVE) >= 50
    assert all(len(texts) >= 12 for texts in cases.POSITIVE.values())


@pytest.mark.parametrize("family,text", C1, ids=[f"{f}:{t[:28]}" for f, t in C1])
def test_c1_must_escalate(family, text):
    assert safety_terms(normalize(text)), text


@pytest.mark.parametrize("text", cases.GAS, ids=[t[:30] for t in cases.GAS])
def test_c2_gas_smell(text):
    assert gas_smell(normalize(text)), text


@pytest.mark.parametrize("text", cases.NEGATIVE, ids=[t[:30] for t in cases.NEGATIVE])
def test_c1_must_not_escalate(text):
    assert safety_terms(normalize(text)) == [], text


@pytest.mark.parametrize("text", cases.GAS_NEGATIVE, ids=[t[:30] for t in cases.GAS_NEGATIVE])
def test_not_a_gas_smell(text):
    assert not gas_smell(normalize(text)), text


@pytest.mark.parametrize("wrap", [
    "{}", "{}.", "Hi team, {}. Please call me back.", "URGENT!!! {}", "{} since yesterday, kindly arrange a visit",
    "The fridge is not cooling. Also {}. It is still under warranty."])
def test_c1_survives_surrounding_text(wrap):
    for _, text in C1[::6]:
        assert safety_terms(normalize(wrap.format(text))), wrap.format(text)
        assert safety_terms(normalize(wrap.format(text).upper())), wrap.format(text).upper()


def test_api_escalates_every_c1_case_as_electrical(client):
    for family, text in C1:
        out = client.post("/classify", json={"ticket_id": "S", "product_category": "washing_machine",
                                             "complaint_text": text, "product_age_days": 5}).json()
        assert out["escalate"] is True and out["fault_category"] == "electrical_or_control_board", (text, out)
        assert out["reason"].startswith("Rule C1")


def test_api_gas_rule_depends_on_product(client):
    for text in cases.GAS:
        ask = lambda product: client.post("/classify", json={"ticket_id": "G", "product_category": product,
                                                             "complaint_text": text}).json()["escalate"]
        assert ask("oven_or_cooktop") is True and ask("water_heater") is True, text
        assert ask("refrigerator") is False and ask("air_conditioner") is False, text


def test_api_does_not_escalate_lookalikes(client):
    for text in cases.NEGATIVE:
        out = client.post("/classify", json={"ticket_id": "N", "product_category": "washing_machine", "complaint_text": text}).json()
        assert out["escalate"] is False, (text, out)
