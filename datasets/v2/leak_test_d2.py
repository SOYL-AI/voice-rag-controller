#!/usr/bin/env python3
"""Surface-form leak test for D2 ``d2_support``.

Question: can a bag-of-n-grams classifier predict ``retrieval_needed_authored``
from the caller's wording alone? If it can, a model's uncertainty could be
exploiting surface cues instead of knowledge gaps (the failure the v1 data had).

Model: logistic regression on TF-IDF word 1-2-grams + char_wb 2-5-grams of the
question, plus length features (words, characters, has-history). Vectorisers
are fit inside each fold. Cross-validation is grouped by ``pair_id`` (a whole
pair group is always held out together), so the test cannot win by memorising
one partner's template. The headline number is the out-of-fold ROC AUC pooled
over folds, reported as the worst case (highest AUC) over a small grid of
regularisation strengths and fold seeds. Target: AUC <= 0.65.

Usage::

    PYTHONUTF8=1 python datasets/v2/leak_test_d2.py                    # reads datasets/v2/d2_support/items.jsonl
    PYTHONUTF8=1 python datasets/v2/leak_test_d2.py --raw              # reads raw/*/*/items.jsonl directly
    PYTHONUTF8=1 python datasets/v2/leak_test_d2.py --raw --domains telecom banking --top-features 30

Labels and wording only; no model inference, no use of the split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Keep BLAS single-threaded: the test is small and the host can be short of memory.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DEFAULT_ITEMS = HERE / "d2_support" / "items.jsonl"
DEFAULT_RAW = HERE / "d2_support" / "raw"
DEFAULT_OUT = HERE / "d2_support" / "leak_test.json"
TARGET_AUC = 0.65


def load_items(args) -> list:
    rows = []
    if args.raw:
        for path in sorted(Path(args.raw_dir).glob("*/*/items.jsonl")):
            domain, org = path.parent.parent.name, path.parent.name
            if args.domains and domain not in args.domains:
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        it = json.loads(line)
                        pid = it.get("pair_id")
                        # local pair ids are only unique within an org
                        it["_group"] = f"{domain}/{org}/{pid}" if pid else f"{domain}/{org}/{it.get('id')}"
                        rows.append(it)
    else:
        with Path(args.items).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    it = json.loads(line)
                    if args.domains and it["domain"] not in args.domains:
                        continue
                    it["_group"] = it.get("pair_id") or it["id"]
                    rows.append(it)
    return rows


def text_of(it: dict, with_history: bool) -> str:
    q = it["question"]
    if with_history and it.get("history"):
        q = " ".join(t.get("content", t.get("text", "")) for t in it["history"]) + " || " + q
    return q


class Featuriser:
    def __init__(self, use_word=True, use_char=True, use_len=True):
        self.use_word, self.use_char, self.use_len = use_word, use_char, use_len

    def _len(self, items, texts):
        return np.array([[len(t.split()), len(t), 1.0 if it.get("history") else 0.0]
                         for it, t in zip(items, texts)], dtype=float)

    def fit_transform(self, items, texts):
        mats = []
        if self.use_word:
            self.w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True, min_df=1,
                                     token_pattern=r"(?u)\b\w[\w']*\b", sublinear_tf=True)
            mats.append(self.w.fit_transform(texts))
        if self.use_char:
            self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, min_df=1, sublinear_tf=True)
            mats.append(self.c.fit_transform(texts))
        if self.use_len:
            self.s = StandardScaler()
            mats.append(csr_matrix(self.s.fit_transform(self._len(items, texts))))
        return hstack(mats).tocsr()

    def transform(self, items, texts):
        mats = []
        if self.use_word:
            mats.append(self.w.transform(texts))
        if self.use_char:
            mats.append(self.c.transform(texts))
        if self.use_len:
            mats.append(csr_matrix(self.s.transform(self._len(items, texts))))
        return hstack(mats).tocsr()


def oof_scores(items, y, groups, C, seed, n_folds, with_history, feats):
    texts = [text_of(it, with_history) for it in items]
    # Shuffle group order deterministically so different seeds give different folds.
    rng = np.random.RandomState(seed)
    uniq = sorted(set(groups))
    perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
    gnum = np.array([perm[g] for g in groups])
    scores = np.zeros(len(items))
    fold_aucs = []
    for tr, te in GroupKFold(n_splits=n_folds).split(np.zeros(len(items)), y, gnum):
        f = Featuriser(**feats)
        Xtr = f.fit_transform([items[i] for i in tr], [texts[i] for i in tr])
        Xte = f.transform([items[i] for i in te], [texts[i] for i in te])
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced", solver="liblinear")
        clf.fit(Xtr, y[tr])
        scores[te] = clf.decision_function(Xte)
        if len(set(y[te])) == 2:
            fold_aucs.append(roc_auc_score(y[te], scores[te]))
    return scores, fold_aucs


def group_bootstrap_ci(y, s, groups, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    idx_by_g = defaultdict(list)
    for i, g in enumerate(groups):
        idx_by_g[g].append(i)
    keys = list(idx_by_g)
    aucs = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(keys), len(keys))
        idx = [i for k in pick for i in idx_by_g[keys[k]]]
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def top_features(items, y, with_history, k):
    texts = [text_of(it, with_history) for it in items]
    f = Featuriser(use_len=False)
    X = f.fit_transform(items, texts)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", solver="liblinear").fit(X, y)
    names = [f"w:{n}" for n in f.w.get_feature_names_out()] + [f"c:{n!r}" for n in f.c.get_feature_names_out()]
    coef = clf.coef_[0]
    order = np.argsort(coef)
    return ([(names[i], round(float(coef[i]), 3)) for i in order[::-1][:k]],
            [(names[i], round(float(coef[i]), 3)) for i in order[:k]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=str(DEFAULT_ITEMS))
    ap.add_argument("--raw", action="store_true", help="read raw per-org items instead of the built file")
    ap.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--C", type=float, nargs="*", default=[0.1, 1.0, 10.0])
    ap.add_argument("--top-features", type=int, default=20)
    ap.add_argument("--out", default=None, help=f"write JSON results here (default {DEFAULT_OUT} when not --raw)")
    ap.add_argument("--label", default=None, help="free-text label stored in the JSON (e.g. 'before revision')")
    args = ap.parse_args(argv)
    if args.domains:
        args.domains = [d for part in args.domains for d in part.split(",") if d]

    items = load_items(args)
    items = [it for it in items if it.get("item_type") in ("org_specific", "counter_prior", "ambiguous", "general")]
    if not items:
        print("no items found")
        return 1
    y = np.array([1 if it["retrieval_needed_authored"] else 0 for it in items])
    groups = [it["_group"] for it in items]
    n_folds = min(args.folds, len(set(groups)))
    print(f"items={len(items)} positives={int(y.sum())} groups={len(set(groups))} folds={n_folds}")

    results = {"n_items": len(items), "n_pos": int(y.sum()), "n_groups": len(set(groups)),
               "domains": sorted({it["domain"] for it in items}), "label": args.label,
               "target_auc_max": TARGET_AUC, "configs": []}
    variants = {
        "question:word+char+len": (False, dict(use_word=True, use_char=True, use_len=True)),
        "question:word": (False, dict(use_word=True, use_char=False, use_len=False)),
        "question:char": (False, dict(use_word=False, use_char=True, use_len=False)),
        "question:len_only": (False, dict(use_word=False, use_char=False, use_len=True)),
        "question+history:word+char+len": (True, dict(use_word=True, use_char=True, use_len=True)),
    }
    best = None
    for vname, (wh, feats) in variants.items():
        for C in args.C:
            aucs, all_scores = [], []
            for seed in args.seeds:
                s, fold_aucs = oof_scores(items, y, groups, C, seed, n_folds, wh, feats)
                aucs.append(roc_auc_score(y, s))
                all_scores.append(s)
            mean_auc = float(np.mean(aucs))
            rec = {"variant": vname, "C": C, "auc_pooled_oof_mean_over_seeds": round(mean_auc, 4),
                   "auc_per_seed": [round(a, 4) for a in aucs]}
            results["configs"].append(rec)
            print(f"{vname:34s} C={C:<5g} AUC={mean_auc:.3f}  per-seed={[round(a, 3) for a in aucs]}")
            if vname.startswith("question:word+char+len") and (best is None or mean_auc > best[0]):
                best = (mean_auc, C, np.mean(all_scores, axis=0))

    head_auc, head_C, head_scores = best
    lo, hi = group_bootstrap_ci(y, head_scores, groups)
    results["headline"] = {"variant": "question:word+char+len", "C": head_C, "auc": round(head_auc, 4),
                           "ci95_group_bootstrap": [round(lo, 4), round(hi, 4)],
                           "pass": bool(head_auc <= TARGET_AUC)}
    # Breakdowns of the headline out-of-fold scores
    by_dom, by_type = {}, {}
    for d in sorted({it["domain"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["domain"] == d]
        if len(set(y[idx])) == 2:
            by_dom[d] = round(float(roc_auc_score(y[idx], head_scores[idx])), 4)
    for t in ("org_specific", "counter_prior", "ambiguous"):
        idx = [i for i, it in enumerate(items) if it["item_type"] in (t, "general")]
        by_type[f"{t}_vs_general"] = round(float(roc_auc_score(y[idx], head_scores[idx])), 4)
    worst = max(results["configs"], key=lambda r: r["auc_pooled_oof_mean_over_seeds"])
    results["worst_config_any_variant"] = worst
    results["headline"]["by_domain"] = by_dom
    results["headline"]["by_type"] = by_type
    # Leave-one-domain-out: cues that transfer across domains (what a new domain would inherit).
    doms = sorted({it["domain"] for it in items})
    if len(doms) >= 3:
        texts = [text_of(it, False) for it in items]
        lodo_scores = np.zeros(len(items))
        for d in doms:
            te = np.array([i for i, it in enumerate(items) if it["domain"] == d])
            tr = np.array([i for i, it in enumerate(items) if it["domain"] != d])
            f = Featuriser()
            Xtr = f.fit_transform([items[i] for i in tr], [texts[i] for i in tr])
            Xte = f.transform([items[i] for i in te], [texts[i] for i in te])
            clf = LogisticRegression(C=head_C, max_iter=5000, class_weight="balanced", solver="liblinear").fit(Xtr, y[tr])
            lodo_scores[te] = clf.decision_function(Xte)
        lodo = {"pooled_auc": round(float(roc_auc_score(y, lodo_scores)), 4),
                "by_domain": {d: round(float(roc_auc_score(y[[i for i, it in enumerate(items) if it["domain"] == d]],
                                                           lodo_scores[[i for i, it in enumerate(items) if it["domain"] == d]])), 4)
                              for d in doms}}
        results["leave_one_domain_out"] = lodo
        print(f"leave-one-domain-out (C={head_C:g}): pooled AUC={lodo['pooled_auc']:.3f} by domain={lodo['by_domain']}")
    pos, neg = top_features(items, y, False, args.top_features)
    results["top_features_retrieval_needed"] = pos
    results["top_features_general"] = neg

    print(f"\nHEADLINE (worst case over C, mean over fold seeds): AUC={head_auc:.3f} "
          f"[95% group-bootstrap CI {lo:.3f}-{hi:.3f}] C={head_C:g} -> {'PASS' if head_auc <= TARGET_AUC else 'FAIL'} (target <= {TARGET_AUC})")
    print(f"worst config over all variants: {worst['variant']} C={worst['C']:g} AUC={worst['auc_pooled_oof_mean_over_seeds']:.3f}")
    print(f"by domain: {by_dom}")
    print(f"by type:   {by_type}")
    print("\ntop n-grams -> retrieval needed:", ", ".join(f"{n}({c})" for n, c in pos))
    print("top n-grams -> general:         ", ", ".join(f"{n}({c})" for n, c in neg))

    out = args.out or (None if args.raw else str(DEFAULT_OUT))
    if out:
        Path(out).write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
