"""Train the sentence classifier and its confidence calibrator.

    python -m fault.train

Reads data/training/tickets.csv and data/training/seed_clauses.csv, writes model/sentence.joblib and
model/model_card.json. Runs in a few seconds on a laptop CPU.
"""
import datetime
import hashlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from fault.decide import classify
from fault.labels import LABELS, NONE
from fault.model import THRESHOLDS, Scorer, build_pipeline
from fault.rules import clause_text
from fault.sentences import build_clause_table, load_seeds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKETS = os.path.join(ROOT, "data", "training", "tickets.csv")
SEEDS = os.path.join(ROOT, "data", "training", "seed_clauses.csv")
MODEL_DIR = os.path.join(ROOT, "model")
SEED = 13


def fit_pipeline(sentence_table, seeds=None):
    train = sentence_table.dropna(subset=["label"])[["clause", "label"]]
    if seeds is not None and len(seeds):
        train = pd.concat([train, seeds], ignore_index=True)
    train = train.drop_duplicates(subset=["clause"])
    pipeline = build_pipeline(seed=SEED)
    pipeline.fit(list(train["clause"].map(clause_text)), list(train["label"]))
    return pipeline, len(train)


def predict_frame(scorer, tickets, texts=None):
    """Classify every ticket; one row per ticket with the response fields, rule, score and ms."""
    texts = tickets["complaint_text"] if texts is None else texts
    rows = []
    for text, r in zip(texts, tickets.itertuples()):
        t0 = time.perf_counter()
        out = classify(scorer, text, r.product_category, r.product_age_days)
        out["ms"] = (time.perf_counter() - t0) * 1000
        rows.append(out)
    return pd.DataFrame(rows, index=tickets.index)


def corrupt(text, rate, rng):
    """Character noise: swap, drop or double a letter with probability `rate` per letter."""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch.isalpha() and rng.rand() < rate:
            op = rng.randint(3)
            if op == 0 and i + 1 < len(text):
                out += [text[i + 1], ch]
                i += 2
                continue
            if op == 1:
                i += 1
                continue
            out += [ch, ch]
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ---- Validation schemes -------------------------------------------------------------------

def run_random_cv(tickets, table, seeds, n_splits=5):
    """A. Same wording in train and test (optimistic)."""
    parts = []
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(tickets, tickets["fault_category"])
    for train_idx, test_idx in folds:
        pipeline, _ = fit_pipeline(table[table["ticket"].isin(tickets.index[train_idx])], seeds)
        parts.append(predict_frame(Scorer(pipeline), tickets.iloc[test_idx]))
    return pd.concat(parts).loc[tickets.index]


def template_folds(table, n_splits=5):
    known = table[table["label"].notna()].drop_duplicates("template").set_index("template")["label"]
    rng = np.random.RandomState(SEED)
    fold = {}
    for _, templates in known.groupby(known).groups.items():
        for pos, t in zip(rng.permutation(len(templates)), templates):
            fold[t] = pos % n_splits
    return fold


def run_template_cv(tickets, table, seeds, n_splits=5, noise=0.0):
    """B. Every symptom and non-complaint wording in a test ticket is unseen in training,
    and a fifth of the filler sentences are unseen too."""
    fold = template_folds(table, n_splits)
    non_complaint = tickets["reason"].str.startswith("C6")
    is_key = table["label"].notna() & ((table["label"] != NONE) | table["ticket"].map(non_complaint))
    rng = np.random.RandomState(SEED)
    parts = []
    for k in range(n_splits):
        held = {t for t, f in fold.items() if f == k}
        test_ids = sorted(set(table[is_key & table["template"].isin(held)]["ticket"]))
        train_table = table[~table["template"].isin(held) & ~table["ticket"].isin(test_ids)]
        pipeline, _ = fit_pipeline(train_table, seeds)
        test = tickets.loc[test_ids]
        texts = [corrupt(t, noise, rng) for t in test["complaint_text"]] if noise else None
        part = predict_frame(Scorer(pipeline), test, texts)
        part["fold"] = k
        parts.append(part)
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="first")]


def run_seed_only(tickets, seeds):
    """C. Trained only on the hand-written sentences, tested on every ticket."""
    pipeline = build_pipeline(seed=SEED)
    pipeline.fit(list(seeds["clause"].map(clause_text)), list(seeds["label"]))
    return predict_frame(Scorer(pipeline), tickets)


# ---- Confidence calibration ---------------------------------------------------------------

def fit_calibrator(pred, correct):
    """Platt scaling per decision path where there is enough data, else the path's accuracy."""
    cal = {}
    for rule, grp in pred.groupby("rule"):
        y = np.asarray(correct[grp.index], dtype=int)
        fitted = None
        if rule in ("model", "weak", "C5", "C3") and len(grp) >= 30 and 0 < y.sum() < len(y):
            lr = LogisticRegression(C=100).fit(grp[["score"]].values, y)
            if lr.coef_[0][0] > 0:                        # a higher score must never mean lower confidence
                fitted = ("platt", float(lr.coef_[0][0]), float(lr.intercept_[0]))
        if fitted:
            cal[rule] = fitted
        else:
            cal[rule] = ("const", float(np.clip((y.sum() + 1) / (len(y) + 2), 0.05, 0.97)))
    return cal


def apply_calibrator(cal, pred, thresholds=THRESHOLDS):
    scorer = Scorer.__new__(Scorer)
    scorer.calibrator = cal
    conf = []
    for rule, score in zip(pred["rule"], pred["score"]):
        c = scorer.calibrate(rule, score)
        if rule == "C6":
            c = thresholds["c6_conf"]
        if rule == "C5":
            c = min(c, thresholds["c5_cap"])
        conf.append(round(float(np.clip(c, 0.01, 0.99)), 3))
    return conf


def crossfit_confidence(pred, correct, extra=None, extra_correct=None):
    """Confidence for scheme B where each fold is calibrated without seeing itself.

    The calibrator for a fold is fitted on the other folds plus `extra` (the seed-only
    predictions), which is how the shipped calibrator is fitted in main().
    """
    conf = pd.Series(0.0, index=pred.index)
    for k in sorted(pred["fold"].unique()):
        rest = pred[pred["fold"] != k].drop(columns="fold")
        rest_ok = correct[rest.index]
        if extra is not None:
            rest = pd.concat([rest, extra], ignore_index=True)
            rest_ok = pd.concat([rest_ok, extra_correct], ignore_index=True)
        else:
            rest, rest_ok = rest.reset_index(drop=True), rest_ok.reset_index(drop=True)
        cal = fit_calibrator(rest, rest_ok)
        conf[pred["fold"] == k] = apply_calibrator(cal, pred[pred["fold"] == k])
    return conf


# ---- Entry point --------------------------------------------------------------------------

def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_data():
    tickets = pd.read_csv(TICKETS)
    seeds = load_seeds(SEEDS)
    return tickets, seeds, build_clause_table(tickets)


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    present = [l for l in LABELS if (np.asarray(y_true) == l).any()]
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def main(model_dir=MODEL_DIR, verbose=True):
    t0 = time.perf_counter()
    tickets, seeds, table = load_data()

    unseen = run_template_cv(tickets, table, seeds)                      # predictions on unseen wording
    correct = tickets.loc[unseen.index, "fault_category"] == unseen["fault_category"]
    # Calibrate on the hardest predictions we have: unseen wording, plus a model that never saw a ticket.
    # The second set has more mistakes, which keeps high confidence from being handed out too easily.
    hard = run_seed_only(tickets, seeds)
    pool = pd.concat([unseen.drop(columns="fold"), hard], ignore_index=True)
    pool_correct = pd.concat([correct, tickets["fault_category"] == hard["fault_category"]], ignore_index=True)
    calibrator = fit_calibrator(pool, pool_correct)

    pipeline, n_sentences = fit_pipeline(table, seeds)                   # final model on everything
    scorer = Scorer(pipeline, calibrator, THRESHOLDS)
    scorer.save(model_dir)

    card = {
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "tickets_sha256": sha256(TICKETS),
        "seeds_sha256": sha256(SEEDS),
        "n_tickets": int(len(tickets)),
        "n_sentences": int(n_sentences),
        "n_templates": int(table[table["label"].notna() & (table["label"] != NONE)]["template"].nunique()),
        "thresholds": THRESHOLDS,
        "calibrator": {k: list(v) for k, v in calibrator.items()},
        "macro_f1_unseen": round(macro_f1(tickets.loc[unseen.index, "fault_category"], unseen["fault_category"]), 4),
        "train_seconds": round(time.perf_counter() - t0, 2),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    with open(os.path.join(model_dir, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    if verbose:
        print(json.dumps({k: card[k] for k in ("n_tickets", "n_sentences", "n_templates", "macro_f1_unseen", "train_seconds")}))
    return card


if __name__ == "__main__":
    main()
