"""Rule precedence, the model's symptom detection, and the crash-safe fallback."""
import pytest

import app as app_module
from fault.decide import classify, decide, fallback
from fault.labels import LABELS
from fault.model import Scorer
from fault.rules import normalize

cal = lambda rule, score: {"C1": 0.97, "C3": 0.95, "C4": 0.9}.get(rule, score)


def run(symptoms=None, c1=(), gas=False, code="", age=None, weak=None, has_text=True):
    return decide(list(c1), gas, code, age, symptoms or {}, weak, cal, has_text=has_text)


def test_c1_beats_everything():
    d = run({"water_leak": 0.99}, c1=["sparks"], code="e21", age=3)
    assert (d["fault_category"], d["escalate"], d["rule"]) == ("electrical_or_control_board", True, "C1")


def test_c2_escalates_without_changing_category():
    d = run({"not_heating": 0.9}, gas=True)
    assert d["fault_category"] == "not_heating" and d["escalate"] and "rule C2" in d["reason"]


def test_c4_code_only_and_code_with_symptom():
    assert run(code="e21")["rule"] == "C4" and run(code="e21")["fault_category"] == "control_panel_or_display"
    assert run({"motor_or_drive": 0.9}, code="e21")["fault_category"] == "motor_or_drive"


def test_c6_non_complaint_has_low_confidence():
    d = run()
    assert (d["fault_category"], d["rule"]) == ("other", "C6") and d["confidence"] < 0.3 and not d["escalate"]


def test_weak_guess_before_other():
    assert run(weak=("no_power", 0.2))["rule"] == "weak"
    assert run(weak=("no_power", 0.05))["fault_category"] == "other"


def test_c3_window_and_priority_over_second_symptom():
    s = {"installation_or_setup": 0.9, "control_panel_or_display": 0.9}
    assert run(s, age=0)["rule"] == "C3" and run(s, age=30)["rule"] == "C3"
    assert run(s, age=31)["fault_category"] == "control_panel_or_display"
    assert run(s, age=None)["fault_category"] == "control_panel_or_display"


def test_c5_picks_earlier_label_and_caps_confidence():
    d = run({"noise_or_vibration": 0.99, "no_power": 0.98})
    assert (d["fault_category"], d["rule"]) == ("no_power", "C5") and d["confidence"] <= 0.70
    assert "no_power and noise_or_vibration" in d["reason"]


def test_single_symptom_and_empty_text():
    d = run({"motor_or_drive": 0.8})
    assert (d["fault_category"], d["rule"], d["confidence"]) == ("motor_or_drive", "model", 0.8)
    assert run(has_text=False)["rule"] == "empty"


def test_nan_confidence_is_safe():
    assert run({"motor_or_drive": float("nan")})["confidence"] == 0.0


@pytest.fixture(scope="module")
def scorer():
    return Scorer.load("model")


@pytest.mark.parametrize("text,want", [
    ("the drum will not spin at all", {"motor_or_drive"}),
    ("food inside is warm, the fridge is not cooling", {"not_cooling"}),
    ("water all over the kitchen floor under the machine", {"water_leak"}),
    ("thank you so much for the quick service", set()),
    ("please call me back. it was fine until the power cut.", set()),
    ("very loud banging noise. also the door latch is broken.", {"noise_or_vibration", "door_seal_or_latch"})])
def test_symptom_detection(scorer, text, want):
    found, _ = scorer.symptoms(normalize(text))
    assert set(found) == want


def test_adding_a_safety_word_flips_escalation(scorer):
    for text, product in [("the drum will not spin at all", "washing_machine"), ("fridge is warm inside", "refrigerator"),
                          ("water all over the floor", "dishwasher")]:
        before = classify(scorer, text, product, 400)
        after = classify(scorer, text + " and I saw smoke", product, 400)
        assert before["escalate"] is False and after["escalate"] is True
        assert after["fault_category"] == "electrical_or_control_board"


def test_case_and_filler_do_not_change_the_category(scorer):
    base = "The drum will not spin at all."
    variants = [base, base.upper(), "Hello team, " + base + " We had guests over this weekend. Please help.", base + " Warranty ends next month."]
    assert {classify(scorer, v, "washing_machine", 400)["fault_category"] for v in variants} == {"motor_or_drive"}


def test_fallback_keeps_escalation():
    d = fallback("Smoke coming from the back", "microwave")
    assert d["escalate"] and d["rule"] == "fallback" and d["fault_category"] == "electrical_or_control_board"
    d2 = fallback(None, None)
    assert (d2["fault_category"], d2["confidence"], d2["escalate"]) == ("other", 0.0, False)


def test_escalation_survives_model_crash(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(app_module.SCORER, "symptoms", boom)
    out = client.post("/classify", json={"ticket_id": "T", "product_category": "water_heater",
                                         "complaint_text": "geyser is leaking and there is a gas smell"}).json()
    assert out["escalate"] is True and out["fault_category"] == "other" and out["confidence"] == 0.0
    assert client.get("/health").status_code == 200


def test_rules_only_mode_still_answers(client, monkeypatch):
    monkeypatch.setattr(app_module, "SCORER", None)
    out = client.post("/classify", json={"ticket_id": "T", "product_category": "microwave", "complaint_text": "sparks from the plug"}).json()
    assert out["escalate"] is True and out["fault_category"] in LABELS


@pytest.mark.parametrize("text,want", [
    # a clear part of a long sentence beats a blurred whole-sentence guess (the brief's first example)
    ("machine fills with water then just sits there, drum never turns, display shows e21", {"motor_or_drive"}),
    # a negated symptom is not a symptom
    ("there is no leak, but the machine does not drain", {"drainage_or_blockage"}),
    # a resolved problem is not a complaint, even though one part of it names a symptom
    ("the fridge is cooling fine now, thank you for fixing it", set()),
    # questions and service complaints carry no fault
    ("how do i clean the filter", set()),
    ("your technician did not come today", set()),
    ("my order number is 48213, when will it be delivered", set())])
def test_fragments_negation_and_non_faults(scorer, text, want):
    found, weak = scorer.symptoms(normalize(text))
    assert set(found) == want


@pytest.mark.parametrize("text", ["How do I clean the filter?", "Your technician did not come today.",
                                  "The fridge is cooling fine now, thank you for fixing it.", "Hello", "I want a refund."])
def test_non_faults_are_other_with_low_confidence(scorer, text):
    out = classify(scorer, text, "refrigerator", 300)
    assert out["fault_category"] == "other" and out["confidence"] < 0.3 and out["escalate"] is False


@pytest.mark.parametrize("text", ["The flame keeps going out on the left burner.", "Burner flame is yellow instead of blue."])
def test_a_normal_burner_flame_does_not_escalate(scorer, text):
    assert classify(scorer, text, "oven_or_cooktop", 300)["escalate"] is False
