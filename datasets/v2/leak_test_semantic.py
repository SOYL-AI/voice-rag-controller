"""
leak_test_semantic.py — independent, meaning-level leak test for d2_support.

leak_test_d2.py measures LEXICAL separability (TF-IDF n-grams + length), and
d2_support's items were revised until it passed. A test that items were
revised against can no longer certify them, so this script uses a detector
that played no part in the revision: sentence embeddings from two encoders
(bge-small, the project's retriever, and all-mpnet-base-v2), fed to a
logistic regression.

Reported, all predicting the authored retrieval-needed label:
  * grouped 5-fold CV AUC (folds keep each pair_id group together), 5 repeats;
  * leave-one-domain-out AUC, overall and per held-out domain;
  * per-type AUC: general vs each retrieval-needed type;
  * the same CV AUC on v1's E1 (36) and E3 (50) sets, as the reference point
    for "separable" (no groups there; stratified folds).

    python datasets/v2/leak_test_semantic.py              # writes datasets/v2/d2_support/leak_test_semantic.json

Run with .venv-e2 (sentence-transformers, scikit-learn).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
ENCODERS = ("BAAI/bge-small-en-v1.5", "sentence-transformers/all-mpnet-base-v2")
REPEATS = 5


def item_text(item: dict) -> str:
    history = " ".join(turn.get("content", "") for turn in item.get("history") or [])
    return (history + " " + item["question"]).strip()


def _fit_predict(emb, y, train, test):
    model = LogisticRegressionCV(Cs=10, max_iter=2000).fit(emb[train], y[train])
    return model.predict_proba(emb[test])[:, 1]


def cv_auc(emb, y, groups=None, repeats=REPEATS) -> tuple:
    aucs = []
    for rep in range(repeats):
        pred = np.zeros(len(y))
        if groups is None:
            folds = StratifiedKFold(5, shuffle=True, random_state=rep).split(emb, y)
        else:
            unique = np.unique(groups)
            order = dict(zip(unique, np.random.default_rng(rep).permutation(len(unique))))
            folds = GroupKFold(5).split(emb, y, [order[g] for g in groups])
        for train, test in folds:
            pred[test] = _fit_predict(emb, y, train, test)
        aucs.append(roc_auc_score(y, pred))
    return round(float(np.mean(aucs)), 3), round(float(np.std(aucs)), 3)


def lodo_auc(emb, y, domains) -> tuple:
    pred = np.zeros(len(y))
    for domain in np.unique(domains):
        test = domains == domain
        pred[test] = _fit_predict(emb, y, ~test, test)
    per_domain = {str(d): round(float(roc_auc_score(y[domains == d], pred[domains == d])), 3)
                  for d in np.unique(domains)}
    return round(float(roc_auc_score(y, pred)), 3), per_domain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(REPO / "datasets/v2/d2_support/items.jsonl"))
    ap.add_argument("--out", default=str(REPO / "datasets/v2/d2_support/leak_test_semantic.json"))
    args = ap.parse_args()

    items = [json.loads(line) for line in open(args.items, encoding="utf-8")]
    texts = [item_text(x) for x in items]
    y = np.array([int(bool(x["retrieval_needed_authored"])) for x in items])
    groups = np.array([x["pair_id"] for x in items])
    domains = np.array([x["domain"] for x in items])
    types = np.array([x["item_type"] for x in items])

    e1 = [json.loads(line) for line in open(REPO / "llm/experiment1/dataset.jsonl", encoding="utf-8")]
    e3 = json.load(open(REPO / "simulator/scenarios.json", encoding="utf-8"))

    report = {"items_file": args.items, "n_items": len(items), "encoders": {}}
    for name in ENCODERS:
        encoder = SentenceTransformer(name)
        emb = encoder.encode(texts, normalize_embeddings=True, batch_size=64)
        mean, sd = cv_auc(emb, y, groups)
        lodo, per_domain = lodo_auc(emb, y, domains)
        per_type = {}
        for t in ("org_specific", "counter_prior", "ambiguous"):
            mask = (types == t) | (types == "general")
            per_type[f"general_vs_{t}"] = cv_auc(emb[mask], y[mask], groups[mask])[0]
        ref = {}
        for label, rows in (("v1_e1_36", e1), ("v1_e3_50", e3)):
            ref_emb = encoder.encode([r["question"] for r in rows], normalize_embeddings=True)
            ref[label] = cv_auc(ref_emb, np.array([int(r["retrieval_needed"]) for r in rows]))[0]
        report["encoders"][name] = {
            "grouped_cv_auc_mean": mean, "grouped_cv_auc_sd": sd,
            "lodo_auc": lodo, "lodo_per_domain": per_domain,
            "per_type": per_type, "v1_reference": ref,
        }
        print(name, json.dumps(report["encoders"][name]))

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
