"""Keyword features for the sentence classifier."""
import re

import numpy as np

from fault.labels import NONE

# Keywords come from appliance knowledge, not from the CSV sentences, so they still fire on
# wording the model has never seen.
LEXICON = {
    "no_power": r"\b(?:dead|lifeless|no power|not power\w*|stays? off|no lights?|no sign of life|boot|chalu nahi|on nahi|start nahi|band pada|(?:not|n't|never|wont|will not|refuses to) (?:switch|turn|power|start|come|boot)\w*(?: on| up)?|not (?:turning|coming|switching|powering|starting)|power(?:ing)? (?:on|up)|switch(?:ing)? on|turn(?:ing)? on)\b",
    "not_heating": r"\b(?:heat\w*|hot|lukewarm|warm(?:ing)? up|gets? warm|cold water|raw|undercooked|degrees?|temperature|dry|dried|garam|preheat\w*|bak\w+|cook\w*|element|thermostat)\b",
    "not_cooling": r"\b(?:cool\w*|not cold|hardly cold|cold enough|warm inside|warm air|hot air|ice|iced|frost\w*|spoil\w*|sour|rott\w*|going bad|melt\w*|thaw\w*|chill\w*|freez\w+|thanda)\b",
    "water_leak": r"\b(?:leak\w*|drip\w*|dribbl\w*|puddles?|pool\w*|overflow\w*|seep\w*|ooz\w*|trickl\w*|spill\w*|soak\w*|water (?:on|all over|under|beneath|below|around|beside)|flood\w*|wet (?:floor|patch)|tapak\w*)\b",
    "drainage_or_blockage": r"\b(?:drain\w*|block\w*|clog\w*|chok\w*|jammed with|standing water|stagnant|empty\w*|water (?:stays|remains|still|collects|logged)|filter|backs up|nikal nahi)\b",
    "noise_or_vibration": r"\b(?:nois\w*|loud\w*|bang\w*|buzz\w*|hum|hums|humming|rattl\w*|vibrat\w*|squeal\w*|squeak\w*|screech\w*|grind\w*|grating|shak\w*|sounds?|thump\w*|thud|knock\w*|clunk\w*|whin\w*|whistl\w*|rumbl\w*|racket|awaaz|walks|jumps|bounces)\b",
    "door_seal_or_latch": r"\b(?:door|seal|gasket|latch|hinge|lock\w*|unlock|lid|handle|darwaza)\b",
    "control_panel_or_display": r"\b(?:display|button\w*|keys?|keypad|touch ?(?:panel|screen|controls?)|panel|screen|settings?|clock|segments?|digits?|garbled|gibberish|blank|knob|dial|selector|unresponsive|backlight|readout|indicator)\b",
    "electrical_or_control_board": r"\b(?:intermittent\w*|board|pcb|motherboard|module|cuts? (?:in and )?out|cutting out|flicker\w*|wiring|voltage|fluctuat\w*|restart\w*|reboot\w*|resets? (?:itself|mid)|by itself|on its own|randomly|surge|relay)\b",
    "motor_or_drive": r"\b(?:fan|drum|compressor|pump|motor|turntable|tray|plate|spin\w*|rotat\w*|blower|belt|agitator|impeller|spray arms?|wash arms?|turns?|ghoom)\b",
    "installation_or_setup": r"\b(?:install\w*|level\w*|tilted|uneven|fit|fits|fitting\w*|fitter|connect\w*|attached|first (?:use|time)|delivered|set ?(?:ting)? ?(?:it )?up|setup|hoses?|plumb\w*|unbox\w*|demo|mount\w*|bracket|cutout)\b",
    NONE: r"\b(?:thank\w*|great|warranty|price|offers?|buy|bought|manual|showroom|congratulations|click|records|available|five stars|call me|replacement|technician|visit|urgent\w*|asap|please|kindly|regards|hello|invoice|order|refund|dealer|brochure)\b",
}
LEX_RE = [re.compile(LEXICON[k]) for k in LEXICON]


def lexicon_matrix(texts):
    return np.array([[1.0 if rx.search(t.lower()) else 0.0 for rx in LEX_RE] for t in texts])


def lexicon_hits(text):
    """For display: which category keyword lists matched this sentence, and on which words."""
    hits = []
    for key, rx in zip(LEXICON, LEX_RE):
        m = rx.search(text.lower())
        if m and key != NONE:
            hits.append([key, m.group(0)])
    return hits
