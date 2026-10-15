"""Fast tests for retrieval profiles, the v2 index, the service, the Modal
script and partial-query risk. No GPU, no network, no model downloads: a
tiny hashing encoder stands in for bge-small."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIM = 32


def fake_encoder(texts):
    out = np.zeros((len(texts), DIM), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().split():
            h = int(hashlib.md5(w.strip(".,?!").encode()).hexdigest(), 16)
            out[i, h % DIM] += 1.0
    return out


@pytest.fixture()
def corpus_file(tmp_path):
    docs = [
        {"doc_id": "pool", "title": "Pool", "text": "The pool opens at six and closes at ten."},
        {"doc_id": "spa", "title": "Spa", "text": "The spa offers massages and facials daily."},
        {"doc_id": "wifi", "title": "Wifi", "text": "Wifi network GrandHorizon guest password at check in."},
        {"doc_id": "parking", "title": "Parking", "text": "Valet parking costs twenty dollars per night."},
        {"doc_id": "long", "title": "Long", "text": " ".join(["breakfast buffet omelette"] * 200)},
    ]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    return p


def test_v2_index_build_load_and_chunking(tmp_path, corpus_file):
    from retrieval.v2_index import build_index, load_index

    meta = build_index(corpus_file, tmp_path / "idx", encoder=fake_encoder, max_words=350)
    li = load_index(tmp_path / "idx")
    for f in ("index.faiss", "docs.jsonl", "meta.json"):
        assert (tmp_path / "idx" / f).exists()
    assert meta["n_docs"] == li.n_docs == li.index.ntotal
    long_chunks = [d for d in li.docs if d.get("parent_doc_id") == "long"]
    assert len(long_chunks) >= 2 and all("#c" in d["doc_id"] for d in long_chunks)
    hits = li.search("pool the pool opens at six and closes at ten", fake_encoder, k=3, dedupe_parents=True)
    assert hits[0]["parent_doc_id"] == "pool"
    assert len({h["parent_doc_id"] for h in hits}) == len(hits)


def test_faiss_retriever_hit_shape(tmp_path, corpus_file):
    from retrieval.profiles import Corpus, FaissRetriever
    from retrieval.v2_index import build_index

    build_index(corpus_file, tmp_path / "idx", encoder=fake_encoder)
    r = FaissRetriever(tmp_path / "idx", encoder=fake_encoder, torch_threads=None)
    hits = r.search("valet parking price", k=3)
    assert len(hits) == 3
    for i, h in enumerate(hits):
        assert {"doc_id", "filename", "text", "score", "rank"} <= set(h)
        assert h["rank"] == i
    assert hits[0]["parent_doc_id"] == "parking"
    assert {"embed_ms", "search_ms", "total_ms"} <= set(r.last_timing)
    assert isinstance(r.corpus, Corpus)


def test_make_retriever_profiles():
    from retrieval.profiles import FaissRerankRetriever, FaissRetriever, RemoteRetriever, make_retriever

    assert isinstance(make_retriever("faiss"), FaissRetriever)
    assert isinstance(make_retriever("faiss_rerank"), FaissRerankRetriever)
    assert isinstance(make_retriever("remote_api", url="http://127.0.0.1:1"), RemoteRetriever)
    with pytest.raises(ValueError):
        make_retriever("nope")


def test_retrieval_signals_from_fixture(tmp_path):
    from retrieval.profiles import retrieval_signals

    q = {"p10": 1.0, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p90": 5.0, "p95": 6.0, "p99": 7.0}
    art = {"schema_version": 1, "canonical": {"faiss": "faiss_cpu4"},
           "profiles": {"faiss_cpu4": {"quantiles_ms": q, "mean_ms": 3.5, "samples_ms": [3.0]}}}
    p = tmp_path / "lp.json"
    p.write_text(json.dumps(art), encoding="utf-8")
    sig = retrieval_signals("faiss", p)
    assert sig == {"latency_profile": "faiss", "expected_latency_ms": 3.0, "latency_quantiles_ms": q}
    assert retrieval_signals("faiss", p, expected_quantile="mean")["expected_latency_ms"] == 3.5
    with pytest.raises(KeyError):
        retrieval_signals("faiss_rerank", p)


class _FakeBackend:
    profile = "faiss"

    def __init__(self):
        self.last_timing = {}

    def search(self, query, k=3):
        self.last_timing = {"total_ms": 1.5}
        return [{"doc_id": "d", "filename": "d", "text": query, "score": 0.1, "rank": 0}][:k]

    def warmup(self, query="x"):
        self.search(query)

    def describe(self):
        return {"profile": self.profile}


def test_service_endpoints(monkeypatch):
    from fastapi.testclient import TestClient
    from retrieval.service import create_app

    monkeypatch.delenv("RETRIEVAL_SERVICE_API_KEY", raising=False)
    with TestClient(create_app(backend=_FakeBackend())) as c:
        assert c.get("/healthz").json()["ready"] is True
        r = c.post("/search", json={"query": "pool hours", "k": 1}).json()
        assert r["profile"] == "faiss" and r["hits"][0]["text"] == "pool hours"
        assert "total_ms" in r["server_ms"] and "handler_ms" in r["server_ms"]
    monkeypatch.setenv("RETRIEVAL_SERVICE_API_KEY", "k1")
    with TestClient(create_app(backend=_FakeBackend())) as c:
        assert c.post("/search", json={"query": "q", "k": 1}).status_code == 401
        assert c.post("/search", json={"query": "q", "k": 1}, headers={"x-api-key": "k1"}).status_code == 200


def test_modal_app_imports_without_modal():
    mod = importlib.import_module("retrieval.modal_app")
    if not mod.MODAL_AVAILABLE:
        assert mod.modal is None
    src = (ROOT / "retrieval" / "modal_app.py").read_text(encoding="utf-8")
    assert "SPEND APPROVAL" in src and "modal.run" not in src


def test_partial_query_risk_rows_and_rho():
    from retrieval.partial_query_risk import analyse, rho_at, score_traces

    class R:
        def search(self, q, k=3):
            ranked = ["pool", "spa", "wifi"] if "pool" in q else ["spa", "wifi", "pool"]
            return [{"parent_doc_id": d} for d in ranked][:k]

    items = {"a": {"id": "a", "gold_doc_ids": ["pool"]}}
    words = "what time does the pool close".split()
    trace = {"item_id": "a", "voice_id": "v",
             "words": [{"w": w, "start_ms": 100 * i, "end_ms": 100 * i + 90} for i, w in enumerate(words)],
             "partials": [{"t_ms": 100 * (i + 1), "text": " ".join(words[: i + 1])} for i in range(6)],
             "asr_final": {"text": "What time does the pool close?"}}
    rows = score_traces(items, [trace], R(), ks=(1,))
    assert [r["hit@1"] for r in rows] == [False, False, False, False, True, True]
    assert all(r["full_hit@1"] for r in rows)
    res = analyse(rows, ks=(1,))
    art = {"analysis": res}
    assert rho_at(art, 1, word_frac=1.0, gate="all_partials") == pytest.approx(0.0)
    assert rho_at(art, 1, word_frac=0.2, gate="all_partials") == pytest.approx(1.0)
