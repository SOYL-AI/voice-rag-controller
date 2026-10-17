"""
e3v2_orgfilter.py -- EXPLORATORY / post hoc / not pre-registered (PREREG_V2
note N2): d2_support Phase A with PER-COMPANY retrieval.

The frozen d2_support retriever searches one shared FAISS index over all 24
fictional companies. A deployed support agent searches only its own
company's knowledge base. This module re-runs Phase A retrieval and grounded
answer generation with retrieval restricted to documents whose `org` equals
the item's org, in a SEPARATE output tree, for a sensitivity analysis.

    python -m simulator.e3v2_orgfilter --retrieval-filter org seed      # copy rewrites/answers/skips
    python -m simulator.e3v2_orgfilter --retrieval-filter org <frozen Phase A args...>

* `--retrieval-filter none` (the default) runs the FROZEN Phase A unchanged
  (simulator/e3v2_phase_a.py is not modified; tests/test_v2_orgfilter.py
  checks byte-identical retrieval output in the default mode).
* `--retrieval-filter org` ranks ALL rows of the frozen IndexFlatL2 (exact
  squared-L2, same encoder, same scores) and keeps the best k = 3 distinct
  parents whose `org` is the item's org: identical to an exact per-company
  index. Every company has >= 14 documents, so top-k is always filled with
  own-company documents (asserted).
* Output goes to simulator/e3v2_phase_a/orgfilter/<dataset>/ (refuses the
  confirmatory tree), so the confirmatory store, answers and grades are
  never touched. Retrieval records are keyed by (rewrite, org).
* Controller decisions never read retrieved content (only oracle_v2's
  gold-hit oracle does), so the frozen lazy plan's dispatch points carry
  over; query_variant hashes the retrieved chunk ids, so the cell list is
  re-derived on this tree by the frozen simulator.e3v2_lazy plan (same
  decisions, filtered variants).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1_v2 import common as e1common  # noqa: E402
from simulator import e3v2_phase_a as pa_mod  # noqa: E402
from simulator.e3v2_phase_a import RETRIEVAL_K, PhaseA, dispatch_points, query_variant_for, sha  # noqa: E402

EXPLORATORY_ROOT = _REPO_ROOT / "simulator" / "e3v2_phase_a" / "orgfilter"
CONFIRMATORY_ROOT = pa_mod.DEFAULT_OUT
LABEL = "EXPLORATORY / post hoc / not pre-registered (PREREG_V2 note N2)"


def org_key(query: str, org: str) -> str:
    return sha("retrieval_org", query, org)


def org_filtered_search(index, query_vecs, orgs: Sequence[str], k: int = RETRIEVAL_K) -> List[List[dict]]:
    """Exact per-company search on the frozen flat index: rank every row,
    keep rows of the item's org, best chunk per parent, first k parents."""
    import numpy as np

    from retrieval.v2_index import _l2_normalize

    q = np.asarray(query_vecs, dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if index.meta.get("normalize"):
        q = _l2_normalize(q).astype("float32")
    scores, idx = index.index.search(np.ascontiguousarray(q), index.n_docs)
    out = []
    for srow, irow, org in zip(scores, idx, orgs):
        hits, seen = [], set()
        for s, i in zip(srow, irow):
            if i < 0:
                continue
            d = index.docs[int(i)]
            if d.get("org") != org:
                continue
            parent = d.get("parent_doc_id", d["doc_id"])
            if parent in seen:
                continue
            seen.add(parent)
            hits.append({"doc_id": d["doc_id"], "parent_doc_id": parent, "score": float(s), "row": int(i)})
            if len(hits) == k:
                break
        if len(hits) < k:
            raise ValueError(f"org {org!r} has fewer than {k} documents in the index")
        out.append(hits)
    return out


class OrgFilterPhaseA(PhaseA):
    """PhaseA with retrieval restricted to the item's company."""

    retrieval_filter = "org"

    def _org(self, item_id: str) -> str:
        org = self.by_id[item_id].get("org")
        if not org:
            raise ValueError(f"{item_id} has no org")
        return org

    def _pairs(self, rewrites: Dict[str, str]) -> set:
        pairs = set()
        for traces in self.conditions.values():
            for item_id, tr in traces.items():
                for pt in dispatch_points(tr):
                    rw = rewrites.get(pt["text"])
                    if rw is not None:
                        pairs.add((rw, self._org(item_id)))
        return pairs

    def retrieve(self) -> dict:
        import time

        from retrieval.v2_index import load_index, sentence_transformer_encoder
        rewrites = self.rewrites()
        done = self.done_keys("retrievals")
        todo = sorted(p for p in self._pairs(rewrites) if org_key(*p) not in done)
        if not todo:
            return {"queries": 0, "retrieval_filter": "org"}
        index = load_index(self.args.index_dir)
        encoder = sentence_transformer_encoder(index.meta["model"])
        t0 = time.perf_counter()
        with e1common.JsonlAppender(self.f("retrievals")) as w:
            for start in range(0, len(todo), 256):
                batch = todo[start:start + 256]
                hits = org_filtered_search(index, encoder([q for q, _ in batch]), [o for _, o in batch])
                for (q, org), hs in zip(batch, hits):
                    w.write({"key": org_key(q, org), "query": q, "org": org, "k": RETRIEVAL_K,
                             "retrieval_filter": "org", "hits": hs})
        return {"queries": len(todo), "retrieval_filter": "org", "wall_s": time.perf_counter() - t0,
                "index_meta": {k: index.meta.get(k) for k in ("model", "metric", "normalize", "corpus_sha256",
                                                              "n_docs", "max_words")}}

    def retrieval_rows(self) -> List[dict]:
        rewrites = self.rewrites()
        hits = {(r["query"], r["org"]): r["hits"] for r in e1common.read_jsonl(self.f("retrievals"))} \
            if self.f("retrievals").exists() else {}
        rows = []
        for cond, traces in self.conditions.items():
            for item_id, tr in traces.items():
                org = self._org(item_id)
                for pt in dispatch_points(tr):
                    rw = rewrites.get(pt["text"])
                    hs = hits.get((rw, org)) if rw is not None else None
                    if hs is None:
                        continue
                    ids = [h["doc_id"] for h in hs]
                    rows.append({"kind": "retrieval", "item_id": item_id, "condition": cond,
                                 "partial_index": pt["partial_index"], "query_text": pt["text"],
                                 "rewrite": rw, "retrieved_doc_ids": ids,
                                 "retrieved_parent_ids": [h["parent_doc_id"] for h in hs],
                                 "retrieved_scores": [h["score"] for h in hs],
                                 "query_variant": query_variant_for(ids), "retrieval_filter": "org"})
        return rows


def seed_tree(dataset: str, split: str) -> dict:
    """Copy the inputs that do not depend on retrieval from the confirmatory
    tree: rewrites (same prompts), skip answers (no retrieval) and grounded
    answers (an answer key fixes item, seed, final text and the retrieved
    chunk ids, so a key that recurs under the filter is the same call)."""
    src, dst = CONFIRMATORY_ROOT / dataset, EXPLORATORY_ROOT / dataset
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in (f"rewrites_{split}.jsonl", f"skip_answers_{split}.jsonl", f"answers_{split}.jsonl"):
        if (src / name).exists() and not (dst / name).exists():
            shutil.copyfile(src / name, dst / name)
            out[name] = "copied"
        else:
            out[name] = "present" if (dst / name).exists() else "missing in source"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = "none"
    if "--retrieval-filter" in argv:
        i = argv.index("--retrieval-filter")
        mode = argv[i + 1]
        del argv[i:i + 2]
    if mode not in ("none", "org"):
        raise SystemExit("--retrieval-filter must be none or org")
    if mode == "none":
        return pa_mod.main(argv)  # the frozen behaviour, unchanged
    if argv and argv[0] == "seed":
        ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "d2_support"
        split = argv[argv.index("--split") + 1] if "--split" in argv else "test"
        print(f"[{LABEL}] seeded {EXPLORATORY_ROOT / ds}: {seed_tree(ds, split)}")
        return 0
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1]).resolve()
        if out == CONFIRMATORY_ROOT.resolve():
            raise SystemExit("refusing to write the org-filtered run into the confirmatory Phase A tree")
    else:
        argv += ["--out", str(EXPLORATORY_ROOT)]
    print(f"[{LABEL}] retrieval filter: org; output root {argv[argv.index('--out') + 1]}")
    original = pa_mod.PhaseA
    pa_mod.PhaseA = OrgFilterPhaseA
    try:
        return pa_mod.main(argv)
    finally:
        pa_mod.PhaseA = original


if __name__ == "__main__":
    raise SystemExit(main())
