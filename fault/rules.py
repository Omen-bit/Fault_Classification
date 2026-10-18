"""Text clean-up and the business-rule detectors (C1, C2, C4 cues).

Nothing in this file is trained and nothing here imports the model, so the safety
escalation keeps working even if the model cannot load.
"""
import re
from functools import lru_cache

from fault.labels import PRODUCTS

BRANDS = r"(?:samsung|lg|whirlpool|bosch|ifb|haier|voltas|kenmore|ge|frigidaire|godrej|panasonic|electrolux|siemens)"
PRODUCT_NOUNS = (r"(?:washer-dryer|top load washer|front loader|washing machine|washer|double door fridge|"
                 r"fridge-freezer|refrigerator|freezer|fridge|built-in dishwasher|dish washer|dishwasher|"
                 r"electric oven|gas range|gas stove|cooktop|hob|oven|split ac|window ac|air conditioner|"
                 r"aircon|ac|instant water heater|storage geyser|water heater|geyser|combination microwave|"
                 r"microwave oven|microwave|machine|appliance|unit|device)")
PROD_RE = re.compile(r"\b(?:the |my |a )?(?:" + BRANDS + r" )?" + PRODUCT_NOUNS + r"\b")
OPENER_RE = re.compile(r"^(?:hi team|hello|hola|namaste|dear sir/madam|complaint:|urgent!|please help\.)[, ]*")
CONNECT_RE = re.compile(r"^(?:besides this|and now|on top of that|also|plus|and|but|then)\s+")
SUB_SPLIT_RE = re.compile(r"\s*,\s*|\s+(?:and|but|then|aur|lekin)\s+")


def normalize(text):
    text = "" if not isinstance(text, str) else text
    text = text.lower().strip()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)         # sooo -> soo
    return re.sub(r"\s+", " ", text)


def split_clauses(text_norm):
    """Sentences with greeting and connector words removed."""
    text_norm = OPENER_RE.sub("", text_norm)
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\s*;\s*|\n+", text_norm):
        s = CONNECT_RE.sub("", s.strip()).strip(" .!?:()")
        if len(s) >= 3:
            out.append(s)
    return out


def sub_clauses(sentence):
    """Comma / 'and' fragments of a long sentence (for tickets written without full stops)."""
    if len(sentence.split()) < 7:
        return []
    parts = [p.strip() for p in SUB_SPLIT_RE.split(sentence)]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts if len(parts) >= 2 else []


def canon(clause):
    """Template form of a sentence: product nouns and numbers masked."""
    c = PROD_RE.sub("PROD", clause)
    c = re.sub(r"\bit\b", "PROD", c)
    return re.sub(r"\d+", "N", c)


def clause_text(clause):
    """Model input: product nouns masked so the model learns symptoms, not appliances."""
    return re.sub(r"\bit\b", "PROD", PROD_RE.sub("PROD", clause))


def osa(a, b):
    """Damerau-Levenshtein (optimal string alignment) distance."""
    if abs(len(a) - len(b)) > 2:
        return 3
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]

# ---- Safety rules C1 and C2 ----

FUZZY_TARGETS = ["fire", "spark", "sparks", "sparking", "smoke", "smoking", "burning", "smell", "smelled",
                 "shock", "flames", "tripped", "tripping", "trips", "breaker", "electric"]
SAFE_WORDS = {"spare", "spoke", "stoke", "shook", "stock", "short", "smile", "small", "shell", "spell", "sell",
              "swell", "still", "frames", "flakes", "blames", "dripped", "dripping", "drips", "gripped",
              "ripped", "tipped", "tipping", "turning", "running", "burner", "burners", "breaks", "broken",
              "bracket", "strips", "tries", "grips", "chips", "ships", "tips", "trims", "electrical",
              "electrics", "shocks", "park", "spar", "sock", "hock", "sparkle", "sparkling", "trimming",
              "trapping", "stripping", "electronic", "electronics", "elective", "fired", "fires"}
SMELL = {"smell", "smells", "smelt", "smelled", "smelling", "odor", "odour", "stink", "stinks", "badbu"}
BURN = {"burn", "burning", "burnt", "burned", "jalne"}
TRIP = {"trip", "trips", "tripped", "tripping"}
SUPPLY = {"supply", "breaker", "mcb", "rccb", "elcb", "rcd", "gfci", "fuse", "mains", "main", "power",
          "electricity", "circuit", "meter"}
STRONG_SUPPLY = {"breaker", "mcb", "rccb", "elcb", "fuse", "mains"}
GAS = {"gas", "lpg", "propane"}
GAS_NEAR = SMELL | {"leak", "leaks", "leaking", "leakage"}


TARGET_LETTERS = {w: set(w) for w in FUZZY_TARGETS}


@lru_cache(maxsize=50000)
def fuzzy_target(t):
    """Safety word that `t` is a likely typo of, or None. Stricter for short words."""
    if len(t) < 4 or t in SAFE_WORDS or t in FUZZY_TARGETS:
        return None
    if len(t) > 9:                            # longer than any safety word plus one slip
        return None
    letters = set(t)
    for w in FUZZY_TARGETS:
        if abs(len(t) - len(w)) > 1 or len(letters & TARGET_LETTERS[w]) < len(TARGET_LETTERS[w]) - 1:
            continue                          # one slip cannot remove two distinct letters
        d = osa(t, w)
        if len(w) <= 4:                       # fire: only swapped or doubled letters (fier, fiire)
            ok = d == 1 and set(t) == set(w) and len(t) >= len(w)
        else:                                 # one slip only: two slips turn "soaking" into "sparking"
            ok = d <= 1
        if ok:
            return w
    return None


def squeeze(t):
    return re.sub(r"(.)\1+", r"\1", t)        # gass -> gas, ssmell -> smel


SQUEEZED = {squeeze(w): w for w in FUZZY_TARGETS + ["gas"]}


def fix_typos(tokens):
    out = []
    for t in tokens:
        hit = fuzzy_target(t)
        if hit is None and t not in SAFE_WORDS and squeeze(t) != t and squeeze(t) in SQUEEZED:
            hit = SQUEEZED[squeeze(t)]         # only doubled letters differ
        out.append(hit or t)
    return out


@lru_cache(maxsize=50000)
def trip_like(t):
    return t not in TRIP and 4 <= len(t) <= 9 and len(set(t) & set("trip")) >= 3 and min(osa(t, w) for w in TRIP) <= 1


def context_fixes(tokens):
    """Typos that are only safe to repair next to a telling neighbour word."""
    out = list(tokens)
    for i, t in enumerate(tokens):
        prev = tokens[i - 1] if i else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t == "sell" and (prev in BURN or prev in GAS or prev in {"strong", "bad", "burnng", "brunng"} or nxt in {"of", "from", "coming", "like"}):
            out[i] = "smell"                                   # "burning sell", "sell of gas"
        elif 5 <= len(prev) <= 7 and prev[0] in "ca" and osa(prev, "caught") <= 1 \
                and t != "fire" and 2 <= len(t) <= 5 and osa(t, "fire") <= 1:
            out[i] = "fire"                                    # "caught fre"
        elif prev in {"electric", "electrical"} and 3 <= len(t) <= 6 and osa(t, "shock") <= 2 and t not in {"stove", "stock"}:
            out[i] = "shock"                                   # "electric shc"
        elif trip_like(t) and any(x in STRONG_SUPPLY for x in tokens[max(0, i - 4):i + 5]):
            out[i] = "trips"                                   # "tips the main breaker"
    return out


HARMLESS_PHRASES = re.compile(r"\bsmoked glass\b|\bshock absorbers?\b")   # contain a safety word, are not a hazard


def safety_tokens(text_norm):
    text_norm = HARMLESS_PHRASES.sub(" ", text_norm)
    return context_fixes(fix_typos(re.findall(r"[a-z]+", text_norm)))


SAFETY_VOCAB = (SMELL | BURN | TRIP | STRONG_SUPPLY | GAS |
                {"spark", "sparks", "sparked", "sparking", "smoke", "smokes", "smoked", "smoking", "smoky", "fire",
                 "flame", "flames", "shock", "shocks", "shocked", "electrocuted", "leak", "leaking", "arcing",
                 "chingari", "dhuan", "dhua", "dhuaan", "aag", "current", "blew", "blown"})


def safety_trace(text_norm):
    """For display only: the words the safety scan read, which typos it repaired, which words matter."""
    raw = re.findall(r"[a-z]+", HARMLESS_PHRASES.sub(" ", text_norm))
    fixed = safety_tokens(text_norm)
    repairs = [[a, b] for a, b in zip(raw, fixed) if a != b]
    words = sorted({w for w in fixed if w in SAFETY_VOCAB})
    return {"words_scanned": len(raw), "typo_repairs": repairs[:20], "watched_words_found": words[:30]}


def near(tokens, set_a, set_b, window):
    """True if a word from set_a is within `window` words of a word from set_b. Linear time."""
    last_a = last_b = None
    for i, t in enumerate(tokens):
        if t in set_a:
            last_a = i
            if last_b is not None and i - last_b <= window:
                return True
        if t in set_b:
            last_b = i
            if last_a is not None and i - last_a <= window:
                return True
    return False


# A burner's flame is normal on a gas appliance, so "flame" alone is not a fire. It counts when
# the wording places it where no flame belongs, or describes it breaking out.
FLAME_PLACES = {"back", "rear", "behind", "plug", "socket", "wire", "wires", "wiring", "cord", "cable", "motor",
                "panel", "board", "control", "compressor", "inside", "bottom", "top", "side", "vent", "vents", "door"}
FLAME_BURNER = {"burner", "burners", "pilot", "hob", "ring", "rings", "stove", "blue", "yellow", "orange",
                "ignite", "ignites", "igniter", "ignition", "lighter", "knob"}


def flame_is_fire(tokens):
    for i, t in enumerate(tokens):
        if t not in ("flame", "flames"):
            continue
        before, around = tokens[max(0, i - 3):i], tokens[max(0, i - 4):i + 6]
        if any(w in ("into", "in", "caught", "shooting", "huge", "big", "open") for w in before[-2:]) and "burner" not in around:
            return True
        if any(w in FLAME_PLACES for w in around) and not any(w in FLAME_BURNER for w in around):
            return True
        if re.search(r"\bflames? (?:came|come|coming|comes|shot|shooting|burst|erupted) (?:out|from|up)\b", " ".join(tokens[i:i + 4])):
            return True
    return False


def safety_terms(text_norm):
    """C1 families mentioned anywhere in the text. Negation does not suppress."""
    tokens = safety_tokens(text_norm)
    joined = " ".join(tokens)
    found = []
    if (re.search(r"\b(?:spark|sparks|sparked|sparking|sparky|chinga+ri|arcing|arced)\b", joined)
            or re.search(r"\b(?:pop|bang|crack) and (?:a )?(?:bright |big |blue )?flash\b|\bflash(?:ed)? (?:of light )?from (?:the )?(?:back|plug|socket|wire|cord|motor|panel|inside)\b", joined)):
        found.append("sparks")
    if re.search(r"\b(?:smoke|smokes|smoked(?! glass)|smoking|smoky|dhuan|dhua|dhuaan)\b", joined):
        found.append("smoke")
    if near(tokens, BURN, SMELL, 4) or re.search(r"\b(?:burn|burning|burnt)\s+(?:plastic|rubber|wire|wiring)\b", joined):
        found.append("a burning smell")
    # visible burn or melt damage on the electrics: the same hazard, seen instead of smelled
    elif (re.search(r"\b(?:burn|burnt|burned|scorch|scorched) marks?\b|\b(?:burnt|burned|scorched|charred) (?:plug|socket|wire|wires|wiring|cable|cord|switch|pin|pins)\b", joined)
            or re.search(r"\b(?:plug|socket|wire|wires|wiring|cable|cord|switch|pin|pins)\b(?: \w+){0,3} (?:melted|melting|melt|charred|scorched|burnt|burned)\b|\b(?:melted|melting|charred) (?:plug|socket|wire|wires|wiring|cable|cord|switch)\b", joined)
            or re.search(r"\b(?:plug|socket|wire|wires|cable|cord|pins?)\b(?: \w+){0,2} (?:turned|gone|went|become|became) (?:brown|black|dark)\b", joined)):
        found.append("burn or melt damage to the plug or wiring")
    # "to fire", "not fire", "fire up" are the verb (a burner igniting), not a fire
    if re.search(r"(?<!\bto )(?<!\bnot )(?<!\bwont )\b(?:fire|ablaze|aag)\b(?! up)", joined) or flame_is_fire(tokens):
        found.append("fire")
    if (re.search(r"\b(?:shock|shocks|shocked|electrocuted|electrocution|zapped|zap|zaps|tingle|tingles|tingling)\b(?! absorber)", joined)
            or re.search(r"\belectric(?:al)? s[a-z]{2,4}k\b", joined)
            or re.search(r"\bcurrent (?:lag|laga|lagta|lagti|aa|aata|maar|mara)\b", joined)):
        found.append("an electric shock")
    if (near(tokens, TRIP, SUPPLY, 5)
            or re.search(r"\bfuse (?:has |is |got )?(?:blown|blew|blows)\b", joined)
            or re.search(r"\b(?:blew|blows|blown|blowing) (?:the |a |its )?fuse\b", joined)
            or re.search(r"\b(?:mcb|elcb|rccb|breaker) (?:gir|gira|girta|girti)\b", joined)      # Hinglish: "MCB gir jata hai"
            # having to reset the breaker means it tripped
            or re.search(r"\b(?:reset|resetting|resets) (?:the |my |a |our )?(?:circuit )?(?:breaker|mcb|fuse)\b|\b(?:breaker|mcb) to (?:switch|turn|get) (?:back on )?the power(?: back on)?\b", joined)
            # the appliance takes the whole house supply down
            or re.search(r"\b(?:whole|entire|full) (?:house|home|flat|building) (?:lights?|power|electricity|supply|current) (?:goes?|went|trips?|tripped|cuts?|shuts?)\b", joined)
            or re.search(r"\b(?:lights?|power|electricity) (?:of|in|for) the (?:whole |entire )?(?:house|home|flat) (?:goes?|went|trips?|tripped|cuts?) (?:off|out)\b", joined)):
        found.append("a tripped supply")
    return found



GAS_FILLING = re.compile(
    r"\b(?:filled|filling|full|fills) (?:up )?(?:with|of) (?:raw |unlit |natural )?(?:gas|propane|lpg)\b"
    r"|\b(?:gas|propane|lpg) (?:is |was |being |gets |got |kept |keeps )?(?:left on|left running|pouring|pours|escaping|escapes|building up|flowing|entering)\b"
    r"|\b(?:raw|unlit|unburned|unburnt) (?:gas|propane)\b")


def gas_smell(text_norm):
    tokens = safety_tokens(text_norm)
    joined = " ".join(tokens)
    return (near(tokens, GAS, GAS_NEAR, 3) or "rotten egg" in joined
            # a leak described without the word "smell": the room filling with gas, or gas left flowing unlit
            or bool(GAS_FILLING.search(joined)))


CODE_WORDS = ["error", "code", "display", "shows", "showing", "getting", "mean", "fault"]
TWO_LETTER_CODES = {"ue", "le", "oe", "de", "he", "pe", "ie", "ce", "fe", "te"}


def error_code(text_norm):
    """Returns the error code in the text, or '' if there is none."""
    tokens = re.findall(r"[a-z0-9-]+", text_norm)
    cue = any(t in CODE_WORDS or (len(t) >= 4 and min(osa(t, w) for w in CODE_WORDS) <= 1) for t in tokens)
    for t in tokens:
        mixed = (re.fullmatch(r"(?=.*\d)(?=.*[a-z])[a-z0-9]{1,4}(?:-\d{1,3})?", t)
                 and not re.fullmatch(r"\d+(?:c|f|kg|l|w|v|hz|mm|cm|am|pm|st|nd|rd|th|x)", t))
        if mixed and (cue or len(tokens) <= 4):
            return t
        if t in TWO_LETTER_CODES and cue and len(tokens) <= 9:
            return t
    return ""


# ---- Rule C3 backstop ----
# Unmistakable fit / levelling / connection / first-use wording. Used only when the product is
# 30 days old or less and the model did not already find an installation symptom.
INSTALL_CUE = re.compile(
    r"\b(?:not (?:been )?(?:installed|levell?ed|connected|fitted|mounted)"
    r"|(?:installation|install|fitting|demo) (?:is )?(?:still )?(?:pending|incomplete|not done|not completed|nahi hua)"
    r"|(?:nobody|no one|technician has not|installer has not) (?:has )?(?:come|came|shown up|turned up)[a-z ]{0,20}install\w*"
    r"|(?:does not|doesn t|doesnt|will not|won t|wont) fit\b"
    r"|(?<!water )(?<!noise )(?<!sound )not level\b|unlevel\w*"
    r"|installed (?:wrongly|incorrectly|wrong)|wrongly installed|transit bolts?|wobbl\w*|rocks on the floor"
    r"|first (?:use|time using)|using it for the first time"
    r"|(?:cannot|can t|cant|unable to) (?:get it |be )?(?:connect\w*|set ?up|install\w*))")



def install_cue(text_norm):
    return bool(INSTALL_CUE.search(re.sub(r"[^a-z0-9 ]", " ", text_norm)))


# ---- Lenient input handling ----
PRODUCT_ALIASES = {p: p for p in PRODUCTS}
PRODUCT_ALIASES.update({
    "washer": "washing_machine", "washer_dryer": "washing_machine", "front_loader": "washing_machine",
    "top_loader": "washing_machine", "fridge": "refrigerator", "freezer": "refrigerator",
    "fridge_freezer": "refrigerator", "dish_washer": "dishwasher", "oven": "oven_or_cooktop",
    "cooktop": "oven_or_cooktop", "stove": "oven_or_cooktop", "hob": "oven_or_cooktop",
    "range": "oven_or_cooktop", "gas_stove": "oven_or_cooktop", "ac": "air_conditioner",
    "a_c": "air_conditioner", "aircon": "air_conditioner", "geyser": "water_heater",
    "heater": "water_heater", "microwave_oven": "microwave"})


def normalize_product(value):
    """Map a product value to one of the seven Appendix A values, or None if unknown."""
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-/]+", "_", value.strip().lower()).strip("_")
    return PRODUCT_ALIASES.get(key)


def coerce_age(value):
    """Days since purchase as a non-negative int, or None when unknown or unusable."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        age = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError, AttributeError):
        return None
    if age != age or age < 0 or age > 100000:
        return None
    return int(age)
