"""
test_v2_orgfilter.py -- the EXPLORATORY per-company retrieval mode
(simulator/e3v2_orgfilter.py; PREREG_V2 note N2) leaves the frozen
behaviour byte-identical by default, and in `org` mode is exact per-company
search that always fills top-k with own-company documents.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

faiss = pytest.importorskip("faiss")

from retrieval.v2_index import LoadedIndex  # noqa: E402
from simulator import e3v2_orgfilter as of  # noqa: E402
from simulator import e3v2_phase_a as pa_mod  # noqa: E402


def _index(orgs, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((len(orgs), dim)).astype("float32")
    idx = faiss.IndexFlatL2(dim)
    idx.add(vecs)
    docs = [{"doc_id": f"d{i}", "parent_doc_id": f"d{i}", "org": o, "text": f"t{i}", "title": ""}
            for i, o in enumerate(orgs)]
    return LoadedIndex(index=idx, docs=docs, meta={"metric": "l2", "normalize": False}), vecs


def test_default_mode_runs_the_frozen_phase_a_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setattr(pa_mod, "main", lambda argv: seen.setdefault("argv", list(argv)) and 0)
    args = ["--dataset", "d2_support", "--split", "test", "--stages", "retrieve"]
    of.main(list(args))
    assert seen["argv"] == args                      # no flag, no change
    seen.clear()
    of.main(["--retrieval-filter", "none"] + args)
    assert seen["argv"] == args
    assert pa_mod.PhaseA is not of.OrgFilterPhaseA   # the frozen class is never replaced by default


def test_org_mode_restores_the_frozen_class_and_refuses_the_confirmatory_tree(monkeypatch):
    monkeypatch.setattr(pa_mod, "main", lambda argv: 0)
    of.main(["--retrieval-filter", "org", "--dataset", "d2_support", "--split", "test"])
    assert pa_mod.PhaseA is not of.OrgFilterPhaseA
    with pytest.raises(SystemExit):
        of.main(["--retrieval-filter", "org", "--out", str(pa_mod.DEFAULT_OUT)])


def test_single_org_filter_equals_the_frozen_search_byte_for_byte():
    index, vecs = _index(["A"] * 20)
    q = vecs[:5] + 0.1
    frozen = index.search_vectors(q, 3, dedupe_parents=True)
    filtered = of.org_filtered_search(index, q, ["A"] * 5, 3)
    for f, g in zip(frozen, filtered):
        assert [(h["doc_id"], h["parent_doc_id"], h["score"], h["row"]) for h in f] == \
               [(h["doc_id"], h["parent_doc_id"], h["score"], h["row"]) for h in g]


def test_org_filter_is_exact_per_company_search_and_fills_k():
    orgs = ["A", "B", "C"] * 10
    index, vecs = _index(orgs, seed=3)
    q = np.random.default_rng(5).standard_normal((6, 8)).astype("float32")
    want = ["A", "B", "C", "A", "B", "C"]
    hits = of.org_filtered_search(index, q, want, 3)
    for qq, org, hs in zip(q, want, hits):
        assert len(hs) == 3 and all(index.docs[h["row"]]["org"] == org for h in hs)
        rows = [i for i, o in enumerate(orgs) if o == org]
        d = ((vecs[rows] - qq) ** 2).sum(axis=1)
        best = [rows[j] for j in np.argsort(d, kind="stable")[:3]]
        assert [h["row"] for h in hs] == best        # exact per-company ranking
    with pytest.raises(ValueError):
        of.org_filtered_search(index, q[:1], ["Z"], 3)  # an org without k documents
