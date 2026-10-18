"""Recover a label for every sentence from ticket-level labels.

The ticket file labels whole complaints. A complaint is one or two symptom sentences plus
filler ("Please call me back."). A sentence template seen under four or more labels is
filler; otherwise it takes the label it appears with, using the `reason` column to know
both labels of a two-symptom ticket.
"""
import collections
import difflib
import os
import re

import pandas as pd

from fault.labels import NONE
from fault.rules import canon, error_code, gas_smell, normalize, safety_terms, split_clauses


def reason_labels(row):
    m = re.search(r"C5: (\w+) and (\w+) are equally", row["reason"])
    if m:
        return {m.group(1), m.group(2)}
    if row["reason"].startswith(("C6", "C4")):
        return set()
    return {row["fault_category"]}


def build_clause_table(df):
    rows = []
    for idx, r in df.iterrows():
        for cl in split_clauses(normalize(r["complaint_text"])):
            rows.append({"ticket": idx, "clause": cl, "canon": canon(cl)})
    cl = pd.DataFrame(rows)

    freq = cl["canon"].value_counts()
    anchors = [t for t, n in freq.items() if n >= 2]
    cache = {}

    def to_template(c):                      # fold typo variants onto the frequent templates
        if c not in cache:
            m = difflib.get_close_matches(c, anchors, n=1, cutoff=0.86) if freq[c] < 2 else [c]
            cache[c] = m[0] if m else c
        return cache[c]

    cl["template"] = cl["canon"].map(to_template)

    single = df["reason"].str.startswith("The only symptom")
    spread = collections.defaultdict(collections.Counter)
    for _, row in cl.iterrows():
        if single[row["ticket"]]:
            spread[row["template"]][df.at[row["ticket"], "fault_category"]] += 1
    tmpl_label = {}
    for t, cnt in spread.items():
        top, n = cnt.most_common(1)[0]
        if len(cnt) >= 4:
            tmpl_label[t] = NONE
        elif n / sum(cnt.values()) >= 0.8 and sum(cnt.values()) >= 2:
            tmpl_label[t] = top

    def fixed_label(c):
        if re.match(r"(?:this started N days ago|model is \S+)$", c):
            return NONE
        if error_code(c.lower()) and len(c.split()) <= 6 and not re.search(r"not|blank|garbled|missing", c):
            return NONE                      # "display shows e21": a code, not a symptom
        return None

    labels = []
    for _, row in cl.iterrows():
        lab = fixed_label(row["canon"]) or tmpl_label.get(row["template"])
        if lab is None:
            if safety_terms(row["clause"]):
                lab = "electrical_or_control_board"
            elif gas_smell(row["clause"]):
                lab = NONE
        labels.append(lab)
    cl["label"] = labels

    non_complaint = df["reason"].str.startswith("C6")
    cl.loc[cl["ticket"].map(non_complaint) & cl["label"].isna(), "label"] = NONE

    for tk, grp in cl.groupby("ticket"):     # leftover sentence: the ticket label nothing else explains
        todo = grp[grp["label"].isna()]
        if len(todo) == 1:
            want = reason_labels(df.loc[tk]) - set(grp["label"].dropna())
            if len(want) == 1:
                cl.loc[todo.index, "label"] = next(iter(want))
    return cl


def load_seeds(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["clause", "label"])
    s = pd.read_csv(path)
    return pd.DataFrame({"clause": s["text"].map(normalize), "label": s["label"]})
