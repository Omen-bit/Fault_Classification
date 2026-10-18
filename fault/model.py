"""Sentence classifier: TF-IDF + keyword features + logistic regression.

Each sentence of a complaint is scored as one of 11 symptom categories or "none".
"""
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

from fault.labels import NONE
from fault.lexicon import lexicon_hits, lexicon_matrix
from fault.rules import clause_text, split_clauses, sub_clauses

THRESHOLDS = {
    "t_symptom": 0.30,   # a sentence counts as a symptom above this probability
    "t_sub": 0.60,       # a comma/and fragment counts as an extra symptom above this
    "t_weak": 0.12,      # nothing confident, but still a complaint: low-confidence guess
    "c5_cap": 0.70,      # rule C5 says "lower the confidence"
    "c6_conf": 0.20,     # rule C6 needs confidence below 0.3
    "none_veto": 0.50,   # a sentence this sure to be "no symptom" yields no fragment symptoms and no weak guess
}
MAX_MODEL_CHARS = 4000   # the model reads this much; the rules always read the full text
ARTIFACT = "sentence.joblib"


def build_pipeline(C=10, seed=13):
    features = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        FunctionTransformer(lexicon_matrix, validate=False))
    return make_pipeline(features, LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed))


class Scorer:
    def __init__(self, pipeline, calibrator=None, thresholds=None):
        self.pipeline = pipeline
        self.calibrator = calibrator or {}
        self.thresholds = dict(THRESHOLDS, **(thresholds or {}))
        self.classes = list(pipeline.classes_)
        self.none_at = self.classes.index(NONE)

    @classmethod
    def load(cls, model_dir="model"):
        blob = joblib.load(os.path.join(model_dir, ARTIFACT))
        return cls(blob["pipeline"], blob["calibrator"], blob["thresholds"])

    def save(self, model_dir="model"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "calibrator": self.calibrator, "thresholds": self.thresholds},
                    os.path.join(model_dir, ARTIFACT), compress=3)

    def symptoms(self, text_norm, detail=None):
        """Returns (found, weak).

        found: {category: best probability} for every symptom detected in the text.
        weak:  (category, probability) of the best symptom guess, used when found is empty.
        If `detail` is a list, one record per sentence and fragment is appended to it for display.
        """
        sentences = split_clauses(text_norm[:MAX_MODEL_CHARS])
        if not sentences:
            return {}, None
        fragments, parent = [], []
        for i, sentence in enumerate(sentences):
            for piece in sub_clauses(sentence):
                fragments.append(piece)
                parent.append(i)
        texts = sentences + fragments
        inputs = [clause_text(c) for c in texts]
        proba = self.pipeline.predict_proba(inputs)
        none_p = proba[:, self.none_at]
        th, n_sent = self.thresholds, len(sentences)
        best = proba.argmax(axis=1)
        limit = [th["t_symptom"]] * n_sent + [th["t_sub"]] * len(fragments)
        counted = [best[n] != self.none_at and proba[n, best[n]] >= limit[n] for n in range(len(texts))]
        verdict = ["symptom" if c else ("no symptom" if best[n] == self.none_at else "below threshold") for n, c in enumerate(counted)]

        # 1. A fragment cannot add a symptom when its whole sentence clearly reads as "no symptom":
        #    "the fridge is cooling fine now, thank you" must not become a cooling fault.
        for f in range(len(fragments)):
            n = n_sent + f
            if counted[n] and none_p[parent[f]] >= th["none_veto"]:
                counted[n], verdict[n] = False, "no symptom"
        # 2. A long sentence that mixes several things gets a blurred score. If one of its parts is a
        #    clear symptom, an unsure whole-sentence guess of a different category gives way to the parts.
        for i in range(n_sent):
            parts = {self.classes[best[n_sent + f]] for f in range(len(fragments)) if parent[f] == i and counted[n_sent + f]}
            if parts and counted[i] and proba[i, best[i]] < th["t_sub"] and self.classes[best[i]] not in parts:
                counted[i], verdict[i] = False, "below threshold"

        found, weak = {}, None
        for n, row in enumerate(proba):
            label = self.classes[best[n]]
            if detail is not None:
                top = np.argsort(-row)[:3]
                detail.append({
                    "text": texts[n], "model_input": inputs[n], "kind": "sentence" if n < n_sent else "fragment",
                    "threshold": limit[n], "top": [[str(self.classes[j]), round(float(row[j]), 3)] for j in top],
                    "keywords": lexicon_hits(inputs[n]), "verdict": verdict[n], "category": str(label) if counted[n] else None})
            if counted[n]:
                found[label] = max(found.get(label, 0.0), float(row[best[n]]))
            # weak guess: only from a sentence the model is not sure is "no symptom"
            if n < n_sent and none_p[n] < th["none_veto"]:
                row = row.copy()
                row[self.none_at] = 0.0
                j = int(np.argmax(row))
                if weak is None or row[j] > weak[1]:
                    weak = (self.classes[j], float(row[j]))
        return found, weak

    def calibrate(self, rule, score):
        """Turn a raw score into a confidence, using the calibrator fitted on unseen wording."""
        kind = self.calibrator.get(rule)
        if kind is None:
            return float(score)
        if kind[0] == "platt":
            return float(1.0 / (1.0 + np.exp(-(kind[1] * score + kind[2]))))
        return float(kind[1])
