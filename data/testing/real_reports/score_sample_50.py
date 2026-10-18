"""Second real sample (50 tickets), labelled by hand BEFORE seeing the model's answers and
never used for tuning. Same label scheme as score_sample_100.py."""
import sys
import pandas as pd
sys.path.insert(0, ".")
from fault.decide import classify
from fault.model import Scorer

E = "electrical_or_control_board"
NC, NH, WL, DR, NV, DS, CP, MD, OT = ("not_cooling", "not_heating", "water_leak", "drainage_or_blockage", "noise_or_vibration",
                                      "door_seal_or_latch", "control_panel_or_display", "motor_or_drive", "other")
L = {0: (0, 0, {OT, DS}), 1: (1, 1, None), 2: (0, 0, {NC}), 3: (1, 1, None), 4: (0, 0, {OT, MD}), 5: (1, 0, None), 6: (0, 0, {OT, MD, NV}),
     7: (0, 0, {E, CP, OT}), 8: (0, 0, {NC, CP}), 9: (1, 1, None), 10: (1, 1, None), 11: (0, 0, {CP, E, DS, OT}), 12: (0, 0, {OT}),
     13: (1, 1, None), 14: (1, 1, None), 15: (0, 0, {WL}), 16: (0, 0, {DS, OT}), 17: (0, 0, {DS, OT}), 18: (0, 0, {E, CP, NH, OT}),
     19: (0, 0, {NV, MD}), 20: (1, 1, None), 21: (1, 0, None), 22: (1, 0, None), 23: (0, 0, {DS, OT}), 24: (0, 0, {DR, WL, MD}),
     25: (0, 0, {DS, OT}), 26: (0, 0, {WL}), 27: (0, 0, {OT, DS}), 28: (1, 1, None), 29: (1, 1, None), 30: (0, 0, {OT, E, NH}),
     31: (1, 1, None), 32: (0, 0, {CP, OT}), 33: (0, 0, {MD, NV, NC, E}), 34: (1, 0, None), 35: (0, 0, {NC, CP}), 36: (1, 1, None),
     37: (1, 1, None), 38: (0, 0, {NH, OT}), 39: (1, 1, None), 40: (0, 0, {OT}), 41: (0, 0, {CP, E}), 42: (0, 0, {OT}), 43: (1, 1, None),
     44: (0, 0, {DS}), 45: (1, 1, None), 46: (0, 0, {OT}), 47: (1, 1, None), 48: (0, 0, {OT, E, NH}), 49: (1, 1, None)}
s = pd.read_csv(sys.argv[1]); S = Scorer.load("model"); rows = []
for i, r in s.iterrows():
    o = classify(S, r.complaint_text, r.product_category, r.product_age_days); c1, real, cats = L[i]
    rows.append(dict(i=i, c1=c1, real=real, esc=o["escalate"], ok=(o["fault_category"] == E) if c1 else (o["fault_category"] in cats),
                     got=o["fault_category"], rule=o["rule"], conf=o["confidence"], want=E if c1 else "/".join(sorted(cats)),
                     text=" ".join(str(r.complaint_text).split())[:105]))
d = pd.DataFrame(rows)
tp = int(((d.c1 == 1) & d.esc).sum()); fn = int(((d.c1 == 1) & ~d.esc).sum()); fp = int(((d.c1 == 0) & d.esc).sum())
print(f"tickets {len(d)} | mention a C1 word {int(d.c1.sum())} | hazard actually happened {int(d.real.sum())}")
print(f"ESCALATION  TP {tp}  FN {fn}  FP {fp}  TN {len(d) - tp - fn - fp}  recall {tp / max(tp + fn, 1):.3f}  precision {tp / max(tp + fp, 1):.3f} | real hazards caught {int(d[d.real == 1].esc.sum())} of {int(d.real.sum())}")
print(f"CATEGORY acceptable {int(d.ok.sum())} of {len(d)} = {d.ok.mean():.3f} | with a C1 word {int(d[d.c1 == 1].ok.sum())} of {int((d.c1 == 1).sum())} | without {int(d[d.c1 == 0].ok.sum())} of {int((d.c1 == 0).sum())}")
w = d[~d.ok]
print(f"CONFIDENCE right {d[d.ok].conf.mean():.2f} wrong {w.conf.mean():.2f} | wrong at 0.9 or more: {int((w.conf >= 0.9).sum())} of {len(w)}")
for _, x in d[(d.c1 == 1) & ~d.esc].iterrows(): print(f"  MISSED ESC [{x.i}] got {x.got} [{x.rule}] :: {x.text}")
for _, x in d[(d.c1 == 0) & d.esc].iterrows(): print(f"  FALSE ESC  [{x.i}] got {x.got} [{x.rule}] :: {x.text}")
for _, x in w.iterrows(): print(f"  WRONG CAT  [{x.i}] want {x.want} got {x.got} [{x.rule} {x.conf}] :: {x.text[:80]}")
