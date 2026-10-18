"""Rule C3 backstop: clear installation wording on a young product, even if the model misses it."""
import pytest

from fault.decide import classify
from fault.rules import install_cue, normalize


@pytest.mark.parametrize("text,want", [
    ("installation is still pending", True), ("the machine is not level", True),
    ("nobody has come to install it yet", True), ("it does not fit in the cabinet", True),
    ("using it for the first time and it leaks", True), ("transit bolts were not removed", True),
    ("installer left without connecting it, first use not possible", True),
    ("water level is too high", False), ("noise level is unbearable", False),
    ("since installation it makes a loud noise", False), ("delivered yesterday, not cooling", False),
    ("the door does not close", False), ("thanks for the quick installation", False)])
def test_install_cue(text, want):
    assert install_cue(normalize(text)) is want


class NoSymptoms:
    """A model that finds nothing, to prove the backstop works without it."""
    thresholds = {"t_symptom": 0.3, "t_sub": 0.6, "t_weak": 0.12, "c5_cap": 0.7, "c6_conf": 0.2}

    def symptoms(self, text_norm, detail=None):
        return {}, None

    def calibrate(self, rule, score):
        return score


def test_backstop_fires_only_for_young_products():
    text = "Installation is still pending for the new unit."
    assert classify(NoSymptoms(), text, "air_conditioner", 10)["fault_category"] == "installation_or_setup"
    assert classify(NoSymptoms(), text, "air_conditioner", 10)["reason"].startswith("Rule C3")
    for age in (31, None, "abc"):
        assert not classify(NoSymptoms(), text, "air_conditioner", age)["reason"].startswith("Rule C3")


def test_safety_still_wins_over_backstop():
    out = classify(NoSymptoms(), "Not level after installation and there were sparks.", "washing_machine", 5)
    assert out["fault_category"] == "electrical_or_control_board" and out["escalate"] is True
