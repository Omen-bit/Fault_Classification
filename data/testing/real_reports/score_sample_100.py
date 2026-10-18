"""Score the model on 100 real incident reports that were labelled by hand BEFORE seeing its answers.

Label per ticket:
  c1   - a C1 word is literally mentioned (sparks, smoke, burning smell, fire, electric shock, tripped supply),
         including hypothetical mentions such as "this is a fire hazard"
  real - the hazard actually happened (not just feared)
  c2   - a gas smell or gas leak on an oven/cooktop or water heater
  cats - acceptable categories when no C1 word is present (several when the taxonomy does not fit cleanly)
"""
import sys
import collections
import pandas as pd

sys.path.insert(0, ".")
from fault.decide import classify
from fault.model import Scorer

E = "electrical_or_control_board"
NC, NH, WL, DR, NV, DS, CP, MD, OT = ("not_cooling", "not_heating", "water_leak", "drainage_or_blockage", "noise_or_vibration",
                                      "door_seal_or_latch", "control_panel_or_display", "motor_or_drive", "other")
# index: (c1, real, c2, acceptable categories if not c1)
L = {
 0: (0, 0, 0, {NC}), 1: (1, 1, 0, None), 2: (1, 1, 0, None), 3: (0, 0, 0, {NH, E, CP, OT}), 4: (1, 1, 0, None),
 5: (0, 0, 0, {DS, OT}), 6: (1, 1, 0, None), 7: (0, 0, 0, {NC}), 8: (0, 0, 0, {NV}), 9: (1, 0, 0, None),
 10: (0, 0, 0, {NC}), 11: (1, 1, 0, None), 12: (0, 0, 1, {OT, CP}), 13: (1, 1, 0, None), 14: (1, 1, 0, None),
 15: (1, 0, 0, None), 16: (0, 0, 0, {OT}), 17: (0, 0, 0, {NC}), 18: (1, 0, 0, None), 19: (0, 0, 0, {DS, OT}),
 20: (0, 0, 0, {DS, OT}), 21: (0, 0, 0, {E, CP, NH, OT}), 22: (0, 0, 0, {WL, DR}), 23: (1, 1, 0, None), 24: (1, 0, 1, None),
 25: (1, 1, 0, None), 26: (0, 0, 0, {MD}), 27: (0, 0, 0, {OT}), 28: (1, 1, 0, None), 29: (1, 1, 0, None),
 30: (0, 0, 0, {WL, E, NH, OT}), 31: (0, 0, 0, {NC}), 32: (1, 0, 0, None), 33: (1, 1, 0, None), 34: (1, 0, 0, None),
 35: (0, 0, 0, {NH}), 36: (0, 0, 1, {OT, CP}), 37: (0, 0, 0, {NH}), 38: (0, 0, 0, {NC}), 39: (0, 0, 0, {NC}),
 40: (0, 0, 0, {WL}), 41: (0, 0, 1, {OT, CP}), 42: (0, 0, 0, {WL}), 43: (1, 1, 0, None), 44: (0, 0, 0, {NC}),
 45: (1, 1, 0, None), 46: (0, 0, 0, {WL}), 47: (1, 1, 0, None), 48: (1, 1, 0, None), 49: (1, 1, 0, None),
 50: (0, 0, 0, {OT}), 51: (1, 1, 0, None), 52: (1, 1, 0, None), 53: (0, 0, 0, {NV}), 54: (0, 0, 0, {NH, OT, E}),
 55: (1, 1, 0, None), 56: (0, 0, 0, {OT}), 57: (0, 0, None, {CP}), 58: (1, 1, 0, None), 59: (1, 1, 0, None),
 60: (0, 0, 0, {DS, OT, NV}), 61: (1, 1, 0, None), 62: (0, 0, 0, {OT, DS}), 63: (1, 1, 0, None), 64: (0, 0, 0, {DS}),
 65: (1, 1, 0, None), 66: (1, 1, 0, None), 67: (0, 0, 0, {CP, E, NH}), 68: (1, 1, 0, None), 69: (0, 0, 0, {NC, E, CP}),
 70: (0, 0, 0, {NC}), 71: (0, 0, 0, {OT}), 72: (1, 0, 0, None), 73: (0, 0, 0, {DS}), 74: (0, 0, 0, {NC}),
 75: (0, 0, 0, {NC}), 76: (0, 0, 0, {NH, E, CP, OT}), 77: (0, 0, 0, {DS, OT}), 78: (1, 1, 0, None), 79: (0, 0, 0, {DS, OT}),
 80: (0, 0, 0, {OT, DR, WL}), 81: (1, 1, 0, None), 82: (1, 0, 0, None), 83: (0, 0, 0, {NC}), 84: (0, 0, 0, {OT}),
 85: (1, 0, 0, None), 86: (0, 0, 0, {NV, MD, OT}), 87: (0, 0, 0, {NV}), 88: (1, 1, 0, None), 89: (0, 0, 0, {NV}),
 90: (0, 0, 0, {CP, E}), 91: (1, 1, 0, None), 92: (1, 1, 0, None), 93: (0, 0, 0, {NC}), 94: (1, 1, 0, None),
 95: (1, 0, 0, None), 96: (0, 0, 0, {CP, E, NH}), 97: (0, 0, 0, {NV, MD, OT}), 98: (1, 1, 0, None), 99: (1, 0, 0, None),
}
s = pd.read_csv(sys.argv[1])
S = Scorer.load("model")
rows = []
for i, r in s.iterrows():
    o = classify(S, r.complaint_text, r.product_category, r.product_age_days)
    c1, real, c2, cats = L[i]
    want_esc = None if c2 is None else bool(c1 or c2)
    ok_cat = (o["fault_category"] == E) if c1 else (o["fault_category"] in cats)
    rows.append(dict(i=i, c1=c1, real=real, c2=c2, want_esc=want_esc, got_esc=o["escalate"], ok_cat=ok_cat, got=o["fault_category"],
                     rule=o["rule"], conf=o["confidence"], want=E if c1 else "/".join(sorted(cats)), text=" ".join(str(r.complaint_text).split())[:110]))
d = pd.DataFrame(rows)
e = d[d.want_esc.notna()]
tp = int((e.want_esc & e.got_esc).sum()); fn = int((e.want_esc & ~e.got_esc).sum()); fp = int((~e.want_esc.astype(bool) & e.got_esc).sum()); tn = len(e) - tp - fn - fp
print(f"tickets {len(d)} | mention a C1 word: {int(d.c1.sum())} | hazard actually happened: {int(d.real.sum())} | gas leak or smell: {int((d.c2 == 1).sum())}")
print(f"ESCALATION (rule as written: any mention)   TP {tp}  FN {fn}  FP {fp}  TN {tn}   recall {tp / (tp + fn):.3f}   precision {tp / max(tp + fp, 1):.3f}")
r_ = d[d.real == 1]
print(f"ESCALATION on hazards that actually happened: caught {int(r_.got_esc.sum())} of {len(r_)}")
print(f"CATEGORY acceptable: {int(d.ok_cat.sum())} of {len(d)} = {d.ok_cat.mean():.3f}")
print(f"  where a C1 word is present: {int(d[d.c1 == 1].ok_cat.sum())} of {int((d.c1 == 1).sum())}   | where none is present: {int(d[d.c1 == 0].ok_cat.sum())} of {int((d.c1 == 0).sum())}")
right, wrong = d[d.ok_cat], d[~d.ok_cat]
print(f"CONFIDENCE mean when right {right.conf.mean():.2f}, when wrong {wrong.conf.mean():.2f} | wrong answers at 0.9 or more: {int((wrong.conf >= 0.9).sum())} of {len(wrong)}")
print("\nMISSED ESCALATIONS")
for _, x in e[e.want_esc.astype(bool) & ~e.got_esc].iterrows():
    print(f"  [{x.i}] c1={x.c1} c2={x.c2} got {x.got} [{x.rule}] :: {x.text}")
print("\nFALSE ESCALATIONS")
for _, x in e[~e.want_esc.astype(bool) & e.got_esc].iterrows():
    print(f"  [{x.i}] got {x.got} [{x.rule}] :: {x.text}")
print("\nWRONG CATEGORY (no C1 word present)")
for _, x in d[(d.c1 == 0) & ~d.ok_cat].iterrows():
    print(f"  [{x.i}] want {x.want} got {x.got} [{x.rule} {x.conf}] :: {x.text}")
print("\nWRONG CATEGORY (C1 word present, so electrical expected)")
for _, x in d[(d.c1 == 1) & ~d.ok_cat].iterrows():
    print(f"  [{x.i}] got {x.got} [{x.rule} {x.conf}] :: {x.text}")
print("\ndecision paths:", dict(collections.Counter(d.rule)))
