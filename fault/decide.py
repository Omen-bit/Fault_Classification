"""The decision: business rules C1 to C6 in one fixed order.

`classify` is the single entry point used by the API, training and evaluation.
`escalate` is set by rules C1 and C2 only; the model never decides it.
"""
from fault.labels import GAS_PRODUCTS, ORDER
from fault.model import THRESHOLDS
from fault.rules import coerce_age, error_code, gas_smell, install_cue, normalize, normalize_product, safety_terms

ELECTRICAL = "electrical_or_control_board"
INSTALL = "installation_or_setup"


def _clip(value):
    if value != value:                       # NaN
        return 0.0
    return round(min(max(float(value), 0.01), 0.99), 3)


def decide(c1_terms, gas, code, age, symptoms, weak, calibrate, has_text=True, thresholds=THRESHOLDS):
    """Apply the precedence table. Returns the five response fields plus `rule` and `score`."""
    escalate = bool(c1_terms) or bool(gas)
    tail = "; rule C2: a gas smell on this product requires escalation." if gas else "."

    def answer(category, rule, score, reason, confidence=None):
        conf = calibrate(rule, score) if confidence is None else confidence
        if rule == "C5":
            conf = min(conf, thresholds["c5_cap"])
        return {"fault_category": category, "confidence": _clip(conf), "escalate": escalate,
                "reason": reason + tail, "rule": rule, "score": float(score)}

    if c1_terms:
        return answer(ELECTRICAL, "C1", 1.0,
                      f"Rule C1: the complaint reports {c1_terms[0]}, so it is an electrical fault and must be escalated")
    if not has_text:
        return answer("other", "empty", 1.0, "No complaint text was provided, so the category is other", 0.05)

    if not symptoms:
        if code:
            return answer("control_panel_or_display", "C4", 1.0,
                          f"Rule C4: only an error code ({code.upper()}) is reported, with no other symptom", 0.85)
        if weak and weak[1] >= thresholds["t_weak"]:
            return answer(weak[0], "weak", weak[1],
                          f"The closest symptom match is {weak[0]}, with low confidence")
        return answer("other", "C6", 1.0,
                      "Rule C6: the text is not a complaint, so the category is other with low confidence",
                      thresholds["c6_conf"])

    if age is not None and 0 <= age <= 30 and INSTALL in symptoms:
        days = "day" if age == 1 else "days"
        return answer(INSTALL, "C3", symptoms[INSTALL],
                      f"Rule C3: a fit, levelling, connection or first-use problem on a product {age} {days} old is an installation fault")

    ranked = sorted(symptoms, key=ORDER.get)
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        return answer(a, "C5", symptoms[a],
                      f"Rule C5: {a} and {b} are both described, so the earlier Appendix B category is chosen with lowered confidence")
    return answer(ranked[0], "model", symptoms[ranked[0]],
                  f"The only symptom described matches {ranked[0]}")


C5_RATIO = 0.80          # a second symptom counts only if its evidence is at least this share of the strongest
COOLING_PRODUCTS = {"refrigerator", "air_conditioner"}


def product_check(symptoms, weak, product):
    """Remove answers that are physically impossible for the product.

    A refrigerator does not heat, so "wrong temperature" on a fridge is a cooling fault.
    Only refrigerators and air conditioners cool, so a cooling fault elsewhere is dropped.
    An unrecognised product is left alone.
    """
    if product is None:
        return symptoms, weak, []
    out, notes = dict(symptoms), []
    if product == "refrigerator" and "not_heating" in out:
        out["not_cooling"] = max(out.get("not_cooling", 0.0), out.pop("not_heating"))
        notes.append("not_heating read as not_cooling: a refrigerator does not heat")
    if product not in COOLING_PRODUCTS and "not_cooling" in out:
        out.pop("not_cooling")
        notes.append("not_cooling dropped: this product does not cool")
    if weak is not None:
        if product == "refrigerator" and weak[0] == "not_heating":
            weak = ("not_cooling", weak[1])
        elif product not in COOLING_PRODUCTS and weak[0] == "not_cooling":
            weak = None
    return out, weak, notes


def strongest(symptoms, keep, ratio):
    """Keep the symptoms whose evidence is close to the strongest one.

    Rule C5 is about two categories the text supports equally. In a long complaint a stray
    sentence can score as a weak second symptom; that is not equal support, so it is set aside.
    Categories in `keep` are never set aside.
    """
    if len(symptoms) < 2:
        return symptoms, []
    top = max(symptoms.values())
    kept = {c: p for c, p in symptoms.items() if p >= ratio * top or c in keep}
    return kept, sorted(c for c in symptoms if c not in kept)


def classify(scorer, text, product, age, trace=None):
    """Full pipeline for one ticket. `scorer` may be None (rules-only mode).

    If `trace` is a dict it is filled with what each stage saw and decided. Tracing only
    records; it never changes the answer (tests/test_explain.py checks this on every ticket).
    """
    t = normalize(text)
    raw_product, raw_age = product, age
    product = normalize_product(product)
    c1 = safety_terms(t)
    # C2: gas smell on an oven/cooktop or water heater. An unrecognised product also escalates (recall first).
    gas_text = gas_smell(t)
    gas = gas_text and (product in GAS_PRODUCTS or product is None)
    if scorer is None:
        return fallback(text, product)
    detail = [] if trace is not None else None
    symptoms, weak = ({}, None) if (c1 or not t) else scorer.symptoms(t, detail)
    age = coerce_age(age)
    symptoms, weak, remapped = product_check(symptoms, weak, product)
    backstop = age is not None and age <= 30 and INSTALL not in symptoms and not c1 and install_cue(t)
    if backstop:
        symptoms = dict(symptoms, **{INSTALL: 0.6})          # C3 backstop: clear installation wording the model missed
    symptoms, dropped = strongest(symptoms, keep={INSTALL} if (age is not None and age <= 30) else set(),
                                  ratio=scorer.thresholds.get("c5_ratio", C5_RATIO))
    code = error_code(t)
    result = decide(c1, gas, code, age, symptoms, weak, scorer.calibrate,
                    has_text=bool(t), thresholds=scorer.thresholds)
    if trace is not None:
        from fault.rules import safety_trace
        th = scorer.thresholds
        trace.update({
            "cleanup": {"text": t, "characters": len(t), "product_received": None if raw_product is None else str(raw_product)[:60],
                        "product": product, "age_received": None if raw_age is None else str(raw_age)[:20], "age": age},
            "safety": dict(safety_trace(t), c1_terms=c1, gas_smell_in_text=bool(gas_text),
                           gas_product=product in GAS_PRODUCTS or product is None, c2=bool(gas)),
            "sentences": detail,
            "model_skipped": bool(c1) or not t,
            "symptoms": {"found": {k: round(v, 3) for k, v in sorted(symptoms.items(), key=lambda kv: ORDER[kv[0]])},
                         "weak_guess": None if weak is None else [weak[0], round(weak[1], 3)],
                         "error_code": code.upper(), "installation_backstop": bool(backstop),
                         "set_aside_as_weaker": dropped, "changed_for_product": remapped},
            "ladder": ladder(c1, code, age, symptoms, weak, bool(t), th, result["rule"]),
            "confidence": {"rule": result["rule"], "raw_score": round(result["score"], 3),
                           "calibrated": round(float(scorer.calibrate(result["rule"], result["score"])), 3),
                           "final": result["confidence"], "note": CONFIDENCE_NOTES.get(result["rule"], "")},
            "thresholds": th,
        })
    return result


CONFIDENCE_NOTES = {
    "C1": "A safety rule fired, so confidence is the measured accuracy of that rule.",
    "C3": "Measured accuracy of the installation rule on unseen wording.",
    "C4": "Fixed at 0.85: a lone error code is control panel by rule, but the code is not mapped to a specific fault.",
    "C5": "Calibrated, then capped at 0.70 because rule C5 says to lower the confidence.",
    "C6": "Fixed at 0.20 because rule C6 requires confidence below 0.3.",
    "model": "The sentence probability, calibrated on predictions for wording the model had not seen.",
    "weak": "No sentence passed the threshold, so this is a low-confidence best guess.",
    "empty": "No text, so confidence is minimal.",
}


def ladder(c1, code, age, symptoms, weak, has_text, thresholds, fired_rule):
    """The precedence table as a list of rungs for display: each is fired, passed over or not reached."""
    young = age is not None and 0 <= age <= 30
    rungs = [
        ("C1", "Does the text mention sparks, smoke, a burning smell, fire, an electric shock or a tripped supply?", bool(c1)),
        ("empty", "Is there no text at all?", not has_text),
        ("C4", "No symptom found, but an error code is present?", not symptoms and bool(code)),
        ("weak", "No confident symptom, but a weak guess above the low threshold?",
         not symptoms and not code and bool(weak) and weak[1] >= thresholds["t_weak"]),
        ("C6", "Nothing that looks like a symptom, so not a complaint?", not symptoms),
        ("C3", "Is the product 30 days old or less, with a fit, levelling, connection or first-use problem?",
         young and INSTALL in symptoms),
        ("C5", "Are two or more symptoms described?", len(symptoms) >= 2),
        ("model", "Exactly one symptom described?", len(symptoms) == 1),
    ]
    out, done = [], False
    for rule, question, holds in rungs:
        if done:
            state = "not reached"
        elif rule == fired_rule:
            state, done = "fired", True
        else:
            state = "no"
        out.append({"rule": rule, "question": question, "state": state, "condition_true": bool(holds)})
    return out


def explain(scorer, text, product, age):
    """The answer plus a stage-by-stage trace of how it was reached."""
    trace = {}
    result = classify(scorer, text, product, age, trace=trace)
    return result, trace


def fallback(text, product):
    """Rules-only answer used when the model is missing or anything raises. Never raises."""
    try:
        t = normalize(text)
        prod = normalize_product(product)
        c1 = safety_terms(t)
        gas = gas_smell(t) and (prod in GAS_PRODUCTS or prod is None)
    except Exception:
        c1, gas = [], False
    if c1:
        return {"fault_category": ELECTRICAL, "confidence": 0.85, "escalate": True, "rule": "fallback", "score": 1.0,
                "reason": f"Rule C1: the complaint reports {c1[0]}, so it is an electrical fault and must be escalated (safe fallback)."}
    reason = "Internal error; safe fallback using the safety rules only"
    reason += "; rule C2: a gas smell on this product requires escalation." if gas else "."
    return {"fault_category": "other", "confidence": 0.0, "escalate": bool(gas), "rule": "fallback",
            "score": 0.0, "reason": reason}
