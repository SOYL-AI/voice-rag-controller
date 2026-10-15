"""Retrieval for E1-v2's --with-retrieval path.

The v2 index builder is being written on another branch. E1-v2 codes against
one interface and nothing else:

    retrieval.search.search(query: str, index_dir: str | Path, k: int) -> list[dict]

each hit carrying a document id (`doc_id`, or `id`/`filename`), its `text`,
and a `score` (similarity, higher is better) or `distance` (lower is
better). The module must not load an index at import time (v1's
retrieval/search.py does, with a hard-coded path, which is why it cannot be
used here as-is). `--retriever module:function` points at any other
callable with the same signature.

`lexical` is a dependency-free token-overlap retriever over corpus.jsonl for
the mock smoke test. It is not a retriever of record.
"""

from __future__ import annotations

import importlib
import inspect
import math
import re
from collections import Counter
from pathlib import Path

from llm.experiment1_v2.common import read_jsonl

_TOKEN = re.compile(r"[a-z0-9]+")


def _normalize_hit(hit: dict) -> dict:
    doc_id = hit.get("doc_id") or hit.get("id") or hit.get("filename")
    out = {"doc_id": doc_id, "text": hit.get("text", "")}
    if "score" in hit:
        out["score"], out["score_kind"] = float(hit["score"]), "similarity"
    elif "distance" in hit:
        out["score"], out["score_kind"] = float(hit["distance"]), "distance"
    return out


class CallableRetriever:
    def __init__(self, spec: str, index_dir: Path):
        module_name, _, func_name = spec.partition(":")
        func = getattr(importlib.import_module(module_name), func_name or "search")
        params = inspect.signature(func).parameters
        if "index_dir" not in params:
            raise TypeError(
                f"{spec} does not take index_dir. E1-v2 needs the v2 interface "
                "search(query, index_dir, k); v1's retrieval/search.py "
                "(search(query, k), index fixed at import) is not it."
            )
        self.func = func
        self.index_dir = Path(index_dir)
        self.name = spec

    def search(self, query: str, k: int) -> list:
        return [_normalize_hit(h) for h in self.func(query, index_dir=self.index_dir, k=k)]


class LexicalRetriever:
    """BM25-flavoured token overlap over corpus.jsonl. Plumbing only."""

    name = "lexical-plumbing-only"

    def __init__(self, corpus_path: Path):
        self.docs = read_jsonl(corpus_path, tolerate_partial_tail=False)
        self.tokens = [Counter(_TOKEN.findall((d.get("title", "") + " " + d["text"]).lower()))
                       for d in self.docs]
        df = Counter()
        for counts in self.tokens:
            df.update(counts.keys())
        n = max(1, len(self.docs))
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def search(self, query: str, k: int) -> list:
        q = set(_TOKEN.findall(query.lower()))
        scored = []
        for doc, counts in zip(self.docs, self.tokens):
            score = sum(self.idf.get(t, 0.0) * (1 + math.log(counts[t])) for t in q if counts[t])
            scored.append((score, doc))
        scored.sort(key=lambda pair: -pair[0])
        return [
            {"doc_id": d["doc_id"], "text": d["text"], "score": s, "score_kind": "similarity"}
            for s, d in scored[:k]
        ]


def make_retriever(kind: str, index_dir: Path, corpus_path: Path):
    if kind == "lexical":
        return LexicalRetriever(corpus_path)
    spec = "retrieval.search:search" if kind == "v2" else kind
    return CallableRetriever(spec, index_dir)
