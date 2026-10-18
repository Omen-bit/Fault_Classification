"""Evaluation scorecard and hard gates.

    python -m fault.evaluate

Prints the scorecard, writes model/scorecard.json (aggregate numbers only, read by the
dashboard) and exits non-zero if any hard gate fails.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from fault import train
from fault.labels import LABELS
from fault.model import Scorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLIND = os.path.join(ROOT, "data", "testing", "blind_tickets.csv")
SCORECARD = os.path.join(ROOT, "model", "scorecard.json")
SEED = 13


def macro_f1(y_true, y_pred, labels=LABELS):
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def present_labels(y_true):
    return [l for l in LABELS if (np.asarray(y_true) == l).any()]


def bootstrap_ci(y_true, y_pred, n=1000):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y_true), len(y_true))
        vals.append(macro_f1(y_true[i], y_pred[i], present_labels(y_true[i])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def ece_mce(conf, correct, bins=10):
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece, mce, table = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()),
                          "confidence": round(float(conf[m].mean()), 3), "accuracy": round(float(correct[m].mean()), 3)})
    return float(ece), float(mce), table


def measure(tickets, pred, detail=False):
    """Standard metrics for one validation scheme."""
    d = tickets.loc[pred.index]
    yt, yp = d["fault_category"].values, pred["fault_category"].values
    labels = present_labels(yt)
    et, ep = d["escalate"].astype(bool).values, pred["escalate"].astype(bool).values
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(et, ep, labels=[False, True]).ravel())
    correct = (yt == yp).astype(int)
    conf = pred["confidence"].values
    ece, mce, reliability = ece_mce(conf, correct)
    top = np.argsort(-conf)[: int(0.8 * len(conf))]
    out = {
        "tickets": int(len(d)),
        "labels_present": len(labels),
        "macro_f1": round(macro_f1(yt, yp, labels), 4),
        "macro_f1_ci95": bootstrap_ci(yt, yp),
        "weighted_f1": round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(yt, yp)), 4),
        "mcc": round(float(matthews_corrcoef(yt, yp)), 4),
        "escalation": {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "specificity": round(tn / max(tn + fp, 1), 4),
            "f1": round(float(f1_score(et, ep, zero_division=0)), 4),
            "f2": round(float(fbeta_score(et, ep, beta=2, zero_division=0)), 4),
        },
        "confidence": {
            "ece": round(ece, 4), "mce": round(mce, 4),
            "brier": round(float(brier_score_loss(correct, conf)), 4),
            "auroc": round(float(roc_auc_score(correct, conf)), 4) if 0 < correct.sum() < len(correct) else None,
            "accuracy_top80": round(float(correct[top].mean()), 4),
            "reliability": reliability,
        },
        "latency_ms": {q: round(float(np.percentile(pred["ms"], p)), 2) for q, p in (("p50", 50), ("p95", 95), ("p99", 99))}
                      | {"max": round(float(pred["ms"].max()), 2)},
        "decision_paths": {k: int(v) for k, v in pred["rule"].value_counts().items()},
        "c6_max_confidence": round(float(pred.loc[pred["rule"] == "C6", "confidence"].max()), 3) if (pred["rule"] == "C6").any() else None,
    }
    if detail:
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=LABELS, zero_division=0)
        out["per_class"] = [{"label": l, "precision": round(float(a), 3), "recall": round(float(b), 3),
                             "f1": round(float(c), 3), "support": int(n)} for l, a, b, c, n in zip(LABELS, p, r, f, s)]
        out["confusion_matrix"] = confusion_matrix(yt, yp, labels=LABELS).tolist()
        two = d["reason"].str.startswith("C5").values if "reason" in d else np.zeros(len(d), bool)
        out["two_symptom_accuracy"] = round(float(correct[two].mean()), 4) if two.any() else None
        out["single_symptom_accuracy"] = round(float(correct[~two].mean()), 4)
        mp, mr, _, _ = precision_recall_fscore_support(yt, yp, labels=labels, average="macro", zero_division=0)
        out["macro_precision"], out["macro_recall"] = round(float(mp), 4), round(float(mr), 4)
        cm = np.array(out["confusion_matrix"])
        pairs = sorted(((int(cm[i, j]), LABELS[i], LABELS[j]) for i in range(len(LABELS)) for j in range(len(LABELS))
                        if i != j and cm[i, j]), reverse=True)
        out["top_confusions"] = [{"true": a, "predicted": b, "count": n} for n, a, b in pairs[:8]]

        def slice_by(keys):
            rows = []
            for key in sorted(set(keys), key=str):
                m = np.asarray(keys) == key
                rows.append({"slice": str(key), "tickets": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4),
                             "macro_f1": round(macro_f1(yt[m], yp[m], present_labels(yt[m])), 4)})
            return rows

        length = d["complaint_text"].str.len().values
        out["slices"] = {
            "product": slice_by(d["product_category"].values),
            "decision_path": slice_by(pred["rule"].values),
            "text_length": slice_by(np.where(length < 60, "1. under 60 characters",
                                             np.where(length < 150, "2. 60 to 149 characters", "3. 150 or more characters"))),
            "symptom_count": slice_by(np.where(two, "two symptoms", "one symptom or none")),
        }
        # accuracy when only the most confident share of tickets is answered automatically
        order = np.argsort(-conf, kind="stable")
        out["confidence"]["risk_coverage"] = [{"coverage": c, "accuracy": round(float(correct[order[: max(1, int(c * len(order)))]].mean()), 4)}
                                              for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)]
        if "reason" in d:                    # escalation recall per safety family, read from the labelled reason
            fam = d["reason"].str.extract(r"reports (?:an? )?([a-z ]+?)[.,]")[0]
            fam = fam.where(fam.notna(), np.where(d["reason"].str.contains("gas smell"), "gas smell (C2)", None))
            rows = []
            for name in sorted(x for x in fam.dropna().unique()):
                m = (fam == name).values & et
                if m.any():
                    rows.append({"family": name, "tickets": int(m.sum()), "caught": int((m & ep).sum()),
                                 "recall": round(float((m & ep).sum() / m.sum()), 4)})
            out["escalation"]["by_family"] = rows
    return out


def without_timing(obj):
    """A copy of a scorecard with the run-to-run timing numbers removed."""
    if isinstance(obj, dict):
        return {k: without_timing(v) for k, v in obj.items() if k not in ("latency_ms", "latency_p95_ms")}
    if isinstance(obj, list):
        return [without_timing(v) for v in obj]
    return obj


def same_apart_from_timing(result):
    """True when the stored scorecard already says the same thing, so the file is left alone
    and a re-run does not create a meaningless change in git."""
    try:
        with open(SCORECARD, encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        return False
    return without_timing(json.loads(json.dumps(result))) == without_timing(old)


def gate(value, limit, kind=">="):
    ok = value is not None and (value >= limit if kind == ">=" else value <= limit if kind == "<=" else value == limit)
    return {"pass": bool(ok), "value": value, "limit": f"{kind} {limit}"}


def run(write=True, verbose=True):
    tickets, seeds, table = train.load_data()
    shipped = Scorer.load(train.MODEL_DIR)
    cal = shipped.calibrator

    pb = train.run_template_cv(tickets, table, seeds)
    ok_b = tickets.loc[pb.index, "fault_category"] == pb["fault_category"]
    pc = train.run_seed_only(tickets, seeds)
    ok_c = tickets["fault_category"] == pc["fault_category"]
    pb["confidence"] = train.crossfit_confidence(pb, ok_b, extra=pc, extra_correct=ok_c)
    pa = train.run_random_cv(tickets, table, seeds)
    pa["confidence"] = train.apply_calibrator(cal, pa)
    pc["confidence"] = train.apply_calibrator(cal, pc)

    schemes = {
        "A_seen_wording": dict(measure(tickets, pa, detail=True), title="A. Random stratified 5-fold",
                               note="Same wording in train and test. Optimistic."),
        "B_unseen_wording": dict(measure(tickets, pb, detail=True), title="B. Leave-template-out 5-fold",
                                 note="Symptom, non-complaint and filler wording held out."),
        "C_seed_only": dict(measure(tickets, pc), title="C. Trained on hand-written sentences only",
                            note="No ticket from the file is used for training."),
    }
    if os.path.exists(BLIND):                                # aggregate numbers only; rows are never printed
        blind = pd.read_csv(BLIND)
        pd_ = train.predict_frame(shipped, blind)
        schemes["D_blind"] = dict(measure(blind, pd_), title="D. Blind set", note="Never read during development.")

    noise = {}
    for rate in (0.05, 0.10, 0.15):
        pn = train.run_template_cv(tickets, table, seeds, noise=rate)
        d = tickets.loc[pn.index]
        noise[f"{int(rate * 100)}%"] = {
            "macro_f1": round(macro_f1(d["fault_category"], pn["fault_category"], present_labels(d["fault_category"])), 4),
            "escalation_recall": round(float((d["escalate"] & pn["escalate"]).sum() / max(d["escalate"].sum(), 1)), 4),
            "false_escalations": int((~d["escalate"] & pn["escalate"]).sum())}

    with open(os.path.join(train.MODEL_DIR, "model_card.json"), encoding="utf-8") as f:
        card = json.load(f)
    A, B, C = schemes["A_seen_wording"], schemes["B_unseen_wording"], schemes["C_seed_only"]
    trio = (A, B, C)
    gates = {
        "safety_recall": gate(min(s["escalation"]["recall"] for s in trio), 1.0, "=="),
        "false_escalation": gate(min(s["escalation"]["specificity"] for s in trio), 0.99),
        "c6_confidence": gate(max(s["c6_max_confidence"] or 0 for s in trio), 0.299, "<="),
        "macro_f1_seen": gate(A["macro_f1"], 0.97),
        "macro_f1_unseen": gate(B["macro_f1"], 0.94),
        "macro_f1_seed_only": gate(C["macro_f1"], 0.88),
        "two_symptom_accuracy": gate(A["two_symptom_accuracy"], 0.95),
        "calibration_ece": gate(B["confidence"]["ece"], 0.12, "<="),
        "confidence_auroc": gate(B["confidence"]["auroc"], 0.65),
        "noise_5pct_macro_f1": gate(noise["5%"]["macro_f1"], 0.84),
        "noise_5pct_escalation_recall": gate(noise["5%"]["escalation_recall"], 0.95),
        "latency_p95_ms": gate(A["latency_ms"]["p95"], 100, "<="),
        "train_seconds": gate(card["train_seconds"], 300, "<="),
    }
    result = {"model": {k: card[k] for k in ("created", "n_tickets", "n_sentences", "n_templates", "train_seconds")},
              "labels": LABELS, "schemes": schemes, "noise_on_B": noise, "gates": gates,
              "all_gates_pass": all(g["pass"] for g in gates.values()),
              "generalisation_gap": {"A_minus_B": round(A["macro_f1"] - B["macro_f1"], 4),
                                     "B_minus_C": round(B["macro_f1"] - C["macro_f1"], 4)}}
    if write and not same_apart_from_timing(result):
        with open(SCORECARD, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    if verbose:
        print(f"{'scheme':44s} macroF1  95% CI            acc     MCC     esc.recall  false esc  ECE")
        for s in schemes.values():
            e = s["escalation"]
            print(f"{s['title']:44s} {s['macro_f1']:.4f}  [{s['macro_f1_ci95'][0]:.3f}, {s['macro_f1_ci95'][1]:.3f}]   "
                  f"{s['accuracy']:.4f}  {s['mcc']:.4f}  {e['recall']:.4f}      {e['fp']:<9d}  {s['confidence']['ece']:.3f}")
        print("noise on B:", json.dumps(noise))
        for name, g in gates.items():
            print(f"  {'PASS' if g['pass'] else 'FAIL'}  {name:30s} value {g['value']}  limit {g['limit']}")
        print("ALL GATES PASS" if result["all_gates_pass"] else "GATE FAILURE")
    return result


if __name__ == "__main__":
    sys.exit(0 if run()["all_gates_pass"] else 1)
