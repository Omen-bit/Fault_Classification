"""Safety rules C1 and C2, error codes, and lenient input handling."""
import csv
import time

import pytest

from fault.labels import GAS_PRODUCTS
from fault.rules import coerce_age, error_code, gas_smell, normalize, normalize_product, safety_terms, split_clauses, sub_clauses

with open("data/training/tickets.csv", encoding="utf-8", newline="") as f:
    TICKETS = list(csv.DictReader(f))


def escalates(text, product):
    t = normalize(text)
    return bool(safety_terms(t)) or (gas_smell(t) and product in GAS_PRODUCTS)


def test_every_labelled_escalation_is_caught_with_no_false_alarm():
    missed = [r["ticket_id"] for r in TICKETS if r["escalate"].lower() == "true" and not escalates(r["complaint_text"], r["product_category"])]
    extra = [r["ticket_id"] for r in TICKETS if r["escalate"].lower() == "false" and escalates(r["complaint_text"], r["product_category"])]
    assert sum(r["escalate"].lower() == "true" for r in TICKETS) >= 160
    assert missed == [] and extra == []


@pytest.mark.parametrize("text", [
    "sparks came out from the back", "SPARRKS from the plug", "saw a spark when I switched it on",
    "smoke started coming out", "smkoe coming out of it", "it was smoking badly",
    "there is a burning smell when it runs", "smells like something is burning", "burning plastic smell near the control area",
    "the unit caught fire near the plug", "caught fre near the plug", "small flames at the back",
    "i got an electric shock when i touched it", "got an electric sock from the door", "it shocked me",
    "it trips the main breaker every time", "MCB trips whenever I switch it on", "tips the main breaker every time",
    "the breaker keeps tripping", "dhuan nikal raha hai", "aag lag gayi", "current lagta hai machine se",
    "there is no smoke now but it was scary"])
def test_c1_positive(text):
    assert safety_terms(normalize(text)), text


@pytest.mark.parametrize("text", [
    "water is dripping onto the floor", "it dripped all night", "the shock absorber is worn out",
    "one burner barely gets warm", "burnt food stuck on the tray", "need a spare part for the door",
    "clothes are soaking because it will not empty", "the machine is not starting", "door spring is broken",
    "sparkling clean dishes, thank you", "it was fine until the power cut", "shocking service from your team",
    "the burner will not fire up", "trip to the service centre was wasted", "do you sell extended warranty",
    "electric stove is not heating", "small gap in the door seal", "the compressor is not starting"])
def test_c1_negative(text):
    assert safety_terms(normalize(text)) == [], text


@pytest.mark.parametrize("text", ["gas smell coming from the cooktop", "i can smell gas whenever it is on",
                                  "strong smell of gas near the oven", "gass ssmell coming from it", "there is a gas leak", "gas ki smell aa rahi hai"])
def test_c2_positive(text):
    assert gas_smell(normalize(text)), text


@pytest.mark.parametrize("text", ["the gas range display is blank", "gas burner not lighting", "error i30 on the gas stove", "do you sell gas stoves"])
def test_c2_negative(text):
    assert not gas_smell(normalize(text)), text


def test_safety_word_after_char_2000():
    text = ("the machine is fine but " * 100) + " then i saw smoke"
    assert len(text) > 2000 and "smoke" in safety_terms(normalize(text))


@pytest.mark.parametrize("evil", ["a" * 100_000, "trip " * 20_000, "burn smell " * 9_000, "current " * 12_000, ("x" * 4 + " ") * 20_000],
                         ids=["long-word", "trip", "burn-smell", "current", "short-words"])
def test_no_slow_input(evil):
    t0 = time.perf_counter()
    safety_terms(normalize(evil))
    gas_smell(normalize(evil))
    assert time.perf_counter() - t0 < 1.0


@pytest.mark.parametrize("text,code", [("E21", "e21"), ("Display shows E21", "e21"), ("error code F-02", "f-02"), ("what does le mean on my dishwasher", "le")])
def test_error_code(text, code):
    assert error_code(normalize(text)) == code


def test_no_error_code_in_plain_text():
    assert error_code(normalize("thanks for the quick service")) == ""
    assert error_code(normalize("takes over an hour to reach 180 degrees")) == ""


@pytest.mark.parametrize("raw,want", [
    ("washing_machine", "washing_machine"), ("Washing Machine", "washing_machine"), ("washing-machine", "washing_machine"),
    ("AC", "air_conditioner"), ("fridge", "refrigerator"), ("geyser", "water_heater"), ("oven", "oven_or_cooktop"),
    ("microwave oven", "microwave"), ("toaster", None), (None, None), (7, None), ("", None)])
def test_normalize_product(raw, want):
    assert normalize_product(raw) == want


@pytest.mark.parametrize("raw,want", [(12, 12), ("12", 12), (" 12 ", 12), (12.0, 12), (12.9, 12), (0, 0),
                                      (-5, None), (None, None), ("abc", None), (True, None), ([1], None), ("", None)])
def test_coerce_age(raw, want):
    assert coerce_age(raw) == want


def test_sentence_splitting():
    assert split_clauses(normalize("Hi team, The drum is stuck. Besides this it leaks! Thanks.")) == ["the drum is stuck", "it leaks", "thanks"]
    brief = "machine fills with water then just sits there, drum never turns, display shows e21"
    assert "drum never turns" in sub_clauses(brief) and sub_clauses("drum never turns") == []
