"""Retriever latency profiles behind one interface.

Three profiles (CONTRACTS.md section 6, ``signals["retrieval"]``):

``faiss``
    bge-small-en-v1.5 query embedding + exact FAISS search, in process.
    Same model, same unnormalised vectors and same ``IndexFlatL2`` as the v1
    retriever (retrieval/search.py). ``score`` is the **squared L2 distance**
    FAISS returns: lower is better. For a v2 index built with
    ``--metric ip`` the score is the inner product (cosine for normalised
    vectors): higher is better. ``Retriever.score_kind`` says which.

``faiss_rerank``
    FAISS top-``candidates`` (default 50) followed by a cross-encoder that
    rescores every (query, passage) pair. ``score`` is the raw cross-encoder
    **logit** (no sigmoid): higher is better. ``faiss_score`` keeps the
    first-stage FAISS score. Supported rerankers: ``BAAI/bge-reranker-base``
    and ``cross-encoder/ms-marco-MiniLM-L6-v2`` (the current Hub name of
    ``ms-marco-MiniLM-L-6-v2``; the old name redirects to it).
    ``rerank_dtype="float16"`` runs the reranker in half precision on CUDA.

``remote_api``
    HTTP client for retrieval/service.py (``POST /search``). Hits and
    scores are whatever the server backend returns. ``last_timing`` adds
    ``server_ms`` (compute time measured inside the request handler) and
    ``transport_ms = total_ms - server_ms`` (serialisation, HTTP stack,
    network).

Every hit is ``{"doc_id", "filename", "text", "score", "rank"}`` plus
``parent_doc_id`` (equal to ``doc_id`` for unchunked passages; use it for
hit@k against ``gold_doc_ids``). After each call ``last_timing`` holds the
component timings in milliseconds for that call (``embed_ms``,
``search_ms``, ``rerank_ms``, ``total_ms``; remote adds ``server_ms`` and
``transport_ms``).

Thread count matters. torch defaults to one intra-op thread per logical core
(16 on the development laptop). For a single short query the bge-small
forward pass then oversubscribes the CPU and becomes slow and very
variable; the v1 pipeline ran in that condition. ``make_retriever`` therefore
defaults to ``torch_threads=4``. ``torch.set_num_threads`` is process-global,
so the last retriever constructed wins; pass ``torch_threads=None`` to leave
the process setting untouched (the v1 condition).

Models load lazily on the first ``search`` (or an explicit ``warmup()``);
importing this module does no heavy work and does not need CUDA. A CUDA
device request on a machine without CUDA falls back to CPU with a warning.

The measured latency distributions live in retrieval/latency_profiles_v2.json
(written by retrieval/measure_latency.py); :func:`retrieval_signals` turns
them into the ``signals["retrieval"]`` dict a controller reads.
"""

from __future__ import annotations

import json
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
V1_INDEX_PATH = _HERE / "index.faiss"
V1_CHUNKS_PATH = _HERE / "chunks.json"
DEFAULT_PROFILES_PATH = _HERE / "latency_profiles_v2.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODELS = {
    "bge_reranker_base": "BAAI/bge-reranker-base",
    "minilm_l6": "cross-encoder/ms-marco-MiniLM-L6-v2",
}
DEFAULT_RERANK_MODEL = RERANK_MODELS["minilm_l6"]
DEFAULT_SERVICE_URL = "http://127.0.0.1:8765"
SERVICE_URL_ENV = "RETRIEVAL_SERVICE_URL"
SERVICE_API_KEY_ENV = "RETRIEVAL_SERVICE_API_KEY"

PROFILES = ("faiss", "faiss_rerank", "remote_api")
QUANTILES = ("p10", "p25", "p50", "p75", "p90", "p95", "p99")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def resolve_device(device: str) -> str:
    """Return ``device``, or ``"cpu"`` (with a warning) if CUDA is requested
    but unavailable."""
    if device and device.startswith("cuda"):
        try:
            import torch
            ok = torch.cuda.is_available()
        except Exception:  # torch missing or broken
            ok = False
        if not ok:
            warnings.warn(f"device {device!r} requested but CUDA is unavailable; using cpu")
            return "cpu"
    return device or "cpu"


def apply_torch_threads(n: Optional[int]) -> Optional[int]:
    """Set torch intra-op threads if ``n`` is given; return the effective
    count (or None if torch is unavailable)."""
    try:
        import torch
    except Exception:
        return None
    if n is not None:
        torch.set_num_threads(int(n))
    return torch.get_num_threads()


class Corpus:
    """FAISS index plus the row-aligned passages."""

    def __init__(self, index: Any, docs: list[dict], *, metric: str = "l2",
                 normalize: bool = False, model: str = EMBED_MODEL,
                 source: str = "v1"):
        if index.ntotal != len(docs):
            raise ValueError(f"index has {index.ntotal} vectors but {len(docs)} passages")
        self.index = index
        self.docs = docs
        self.metric = metric
        self.normalize = normalize
        self.model = model
        self.source = source

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @classmethod
    def load(cls, index_dir: Optional[str | Path] = None) -> "Corpus":
        """``index_dir=None`` loads the v1 index (retrieval/index.faiss +
        chunks.json, doc_id = filename); otherwise a v2 directory built by
        retrieval/v2_index.py."""
        import faiss

        if index_dir is None:
            if not V1_INDEX_PATH.exists() or not V1_CHUNKS_PATH.exists():
                raise FileNotFoundError(
                    f"v1 index missing ({V1_INDEX_PATH}, {V1_CHUNKS_PATH}); "
                    "build it with: python retrieval/build_index.py")
            index = faiss.read_index(str(V1_INDEX_PATH))
            with open(V1_CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            docs = [{"doc_id": c["filename"], "parent_doc_id": c["filename"],
                     "filename": c["filename"], "text": c["text"], "passage": c["text"]}
                    for c in chunks]
            return cls(index, docs, metric="l2", normalize=False, model=EMBED_MODEL, source="v1")

        from retrieval.v2_index import embed_text, load_index

        li = load_index(index_dir)
        embed_title = li.meta.get("embed_title", True)
        docs = []
        for row in li.docs:
            docs.append({
                "doc_id": row["doc_id"],
                "parent_doc_id": row.get("parent_doc_id", row["doc_id"]),
                "filename": row["doc_id"],
                "text": row["text"],
                "passage": embed_text(row, embed_title),
            })
        return cls(li.index, docs, metric=li.meta.get("metric", "l2"),
                   normalize=bool(li.meta.get("normalize", False)),
                   model=li.meta.get("model", EMBED_MODEL), source=str(index_dir))


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------

class Retriever:
    """Base class. Subclasses implement ``_search`` and fill ``last_timing``."""

    profile: str = "base"
    score_kind: str = ""

    def __init__(self) -> None:
        self.last_timing: dict[str, float] = {}
        self._lock = threading.Lock()

    def search(self, query: str, k: int = 3) -> list[dict]:
        raise NotImplementedError

    def warmup(self, query: str = "what time is check-in") -> None:
        """Load models and run one throwaway query."""
        self.search(query, k=3)

    def describe(self) -> dict:
        return {"profile": self.profile, "score_kind": self.score_kind}


class FaissRetriever(Retriever):
    """Embed with bge-small and search FAISS. See module docstring for score
    semantics."""

    profile = "faiss"

    def __init__(self, index_dir: Optional[str | Path] = None, *, device: str = "cpu",
                 torch_threads: Optional[int] = 4, corpus: Optional[Corpus] = None,
                 encoder: Optional[Callable[[list], np.ndarray]] = None):
        super().__init__()
        self.index_dir = index_dir
        self.device_requested = device
        self.device = resolve_device(device)
        self.torch_threads_requested = torch_threads
        self.torch_threads: Optional[int] = None
        self._corpus = corpus
        self._encoder = encoder

    # lazy loading ---------------------------------------------------------
    @property
    def corpus(self) -> Corpus:
        if self._corpus is None:
            self._corpus = Corpus.load(self.index_dir)
        return self._corpus

    @property
    def score_kind(self) -> str:  # type: ignore[override]
        return "squared_l2_distance" if self.corpus.metric == "l2" else "inner_product"

    def _get_encoder(self) -> Callable[[list], np.ndarray]:
        if self._encoder is None:
            self.torch_threads = apply_torch_threads(self.torch_threads_requested)
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.corpus.model, device=self.device)

            def encode(texts: list) -> np.ndarray:
                return model.encode(list(texts), convert_to_numpy=True,
                                    show_progress_bar=False)

            self._encoder = encode
        elif self.torch_threads is None:
            self.torch_threads = apply_torch_threads(self.torch_threads_requested)
        return self._encoder

    # search ---------------------------------------------------------------
    def _embed_and_search(self, query: str, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
        enc = self._get_encoder()
        corpus = self.corpus
        t0 = time.perf_counter()
        q = np.asarray(enc([query]), dtype="float32").reshape(1, -1)
        if corpus.normalize:
            norm = float(np.linalg.norm(q))
            if norm > 0:
                q = q / norm
        t1 = time.perf_counter()
        scores, idx = corpus.index.search(np.ascontiguousarray(q), min(n, corpus.n_docs))
        t2 = time.perf_counter()
        return scores[0], idx[0], {"embed_ms": _ms(t0, t1), "search_ms": _ms(t1, t2)}

    def _hit(self, i: int, score: float, rank: int) -> dict:
        d = self.corpus.docs[i]
        return {"doc_id": d["doc_id"], "parent_doc_id": d["parent_doc_id"],
                "filename": d["filename"], "text": d["text"],
                "score": float(score), "rank": rank}

    def search(self, query: str, k: int = 3) -> list[dict]:
        with self._lock:
            t0 = time.perf_counter()
            scores, idx, timing = self._embed_and_search(query, k)
            hits = [self._hit(int(i), s, r) for r, (s, i) in enumerate(zip(scores, idx)) if i >= 0]
            timing["total_ms"] = _ms(t0, time.perf_counter())
            self.last_timing = timing
            return hits

    def describe(self) -> dict:
        return {"profile": self.profile, "device": self.device,
                "torch_threads": self.torch_threads, "model": self.corpus.model,
                "n_docs": self.corpus.n_docs, "metric": self.corpus.metric,
                "score_kind": self.score_kind, "index": self.corpus.source}


def load_cross_encoder(model_name: str, device: str = "cuda", dtype: str = "float32",
                       max_length: int = 512) -> Callable[[list], np.ndarray]:
    """Return ``score(pairs) -> np.ndarray`` of raw cross-encoder logits.

    Scores with the Hugging Face model directly (tokenise, one batched
    forward pass under ``torch.inference_mode``) instead of
    ``sentence_transformers.CrossEncoder.predict``: same logits, without the
    per-call bookkeeping that adds tens of milliseconds for a 50-pair batch.
    Both supported rerankers have a single relevance logit.

    ``dtype="float16"`` halves the weights on CUDA (tensor cores); it is
    ignored on CPU, where half precision is slower.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if dtype == "float16" and device.startswith("cuda"):
        model = model.half()
    elif dtype not in ("float32", "float16"):
        raise ValueError(f"rerank dtype must be float32 or float16, got {dtype!r}")
    model = model.to(device).eval()

    def score(pairs: list) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype="float32")
        feats = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                    truncation=True, max_length=max_length, return_tensors="pt")
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.inference_mode():
            logits = model(**feats).logits
        # .cpu() synchronises with the GPU, so callers time the full forward pass
        return logits[:, 0].float().cpu().numpy().astype("float32")

    return score


class FaissRerankRetriever(FaissRetriever):
    """FAISS top-``candidates`` then cross-encoder rescoring. ``score`` is
    the raw cross-encoder logit (higher is better)."""

    profile = "faiss_rerank"
    score_kind = "cross_encoder_logit"  # type: ignore[assignment]

    def __init__(self, index_dir: Optional[str | Path] = None, *, device: str = "cpu",
                 torch_threads: Optional[int] = 4, rerank_model: str = DEFAULT_RERANK_MODEL,
                 rerank_device: str = "cuda", candidates: int = 50,
                 rerank_dtype: str = "float32",
                 corpus: Optional[Corpus] = None,
                 encoder: Optional[Callable[[list], np.ndarray]] = None,
                 cross_encoder: Optional[Callable[[list], np.ndarray]] = None):
        super().__init__(index_dir, device=device, torch_threads=torch_threads,
                         corpus=corpus, encoder=encoder)
        self.rerank_model = rerank_model
        self.rerank_device = resolve_device(rerank_device)
        # half precision only applies on CUDA (see load_cross_encoder)
        self.rerank_dtype = rerank_dtype if self.rerank_device.startswith("cuda") else "float32"
        self.candidates = int(candidates)
        self._cross = cross_encoder

    def _get_cross(self) -> Callable[[list], np.ndarray]:
        if self._cross is None:
            self._cross = load_cross_encoder(self.rerank_model, self.rerank_device,
                                             self.rerank_dtype)
        return self._cross

    def search(self, query: str, k: int = 3) -> list[dict]:
        with self._lock:
            cross = self._get_cross()
            t0 = time.perf_counter()
            fscores, idx, timing = self._embed_and_search(query, max(self.candidates, k))
            cand = [int(i) for i in idx if i >= 0]
            t1 = time.perf_counter()
            logits = cross([(query, self.corpus.docs[i]["passage"]) for i in cand])
            t2 = time.perf_counter()
            order = np.argsort(-logits, kind="stable")[:k]
            hits = []
            for rank, j in enumerate(order):
                h = self._hit(cand[j], logits[j], rank)
                h["faiss_score"] = float(fscores[j])
                hits.append(h)
            timing["rerank_ms"] = _ms(t1, t2)
            timing["total_ms"] = _ms(t0, time.perf_counter())
            self.last_timing = timing
            return hits

    def describe(self) -> dict:
        d = super().describe()
        d.update({"profile": self.profile, "score_kind": self.score_kind,
                  "rerank_model": self.rerank_model, "rerank_device": self.rerank_device,
                  "rerank_dtype": self.rerank_dtype, "candidates": self.candidates})
        return d


class RemoteRetriever(Retriever):
    """HTTP client for retrieval/service.py. Uses one keep-alive connection.

    The URL comes from the ``url`` argument, else the ``RETRIEVAL_SERVICE_URL``
    environment variable, else a localhost default. If the
    ``RETRIEVAL_SERVICE_API_KEY`` environment variable is set its value is
    sent as the ``x-api-key`` header.
    """

    profile = "remote_api"
    score_kind = "server_defined"

    def __init__(self, url: Optional[str] = None, *, timeout_s: float = 30.0,
                 client: Any = None):
        super().__init__()
        self.url = (url or os.environ.get(SERVICE_URL_ENV) or DEFAULT_SERVICE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self._client = client
        self.server_profile: Optional[str] = None

    def _get_client(self):
        if self._client is None:
            import httpx

            headers = {}
            key = os.environ.get(SERVICE_API_KEY_ENV)
            if key:
                headers["x-api-key"] = key
            self._client = httpx.Client(timeout=self.timeout_s, headers=headers)
        return self._client

    def search(self, query: str, k: int = 3) -> list[dict]:
        client = self._get_client()
        with self._lock:
            t0 = time.perf_counter()
            r = client.post(f"{self.url}/search", json={"query": query, "k": k})
            r.raise_for_status()
            body = r.json()
            t1 = time.perf_counter()
            server = body.get("server_ms", {}) or {}
            total = _ms(t0, t1)
            server_total = float(server.get("handler_ms", server.get("total_ms", 0.0)))
            timing = {"total_ms": total, "server_ms": server_total,
                      "transport_ms": total - server_total}
            for key in ("embed_ms", "search_ms", "rerank_ms"):
                if key in server:
                    timing[f"server_{key}"] = float(server[key])
            self.last_timing = timing
            self.server_profile = body.get("profile")
            return list(body.get("hits", []))

    def health(self) -> dict:
        r = self._get_client().get(f"{self.url}/healthz")
        r.raise_for_status()
        return r.json()

    def warmup(self, query: str = "what time is check-in") -> None:
        self.health()
        self.search(query, k=3)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def describe(self) -> dict:
        return {"profile": self.profile, "server_profile": self.server_profile,
                "score_kind": self.score_kind}


def make_retriever(profile: str, index_dir: Optional[str] = None, device: str = "cpu",
                   torch_threads: Optional[int] = 4, rerank_model: str = DEFAULT_RERANK_MODEL,
                   rerank_device: str = "cuda", candidates: int = 50,
                   url: Optional[str] = None, rerank_dtype: str = "float32") -> Retriever:
    """Construct a retriever for ``profile`` (``faiss`` | ``faiss_rerank`` |
    ``remote_api``). Nothing heavy loads until the first search.

    ``device``/``torch_threads`` apply to the query embedding,
    ``rerank_*``/``candidates`` to ``faiss_rerank`` only, ``url`` to
    ``remote_api`` only.
    """
    if profile == "faiss":
        return FaissRetriever(index_dir, device=device, torch_threads=torch_threads)
    if profile == "faiss_rerank":
        return FaissRerankRetriever(index_dir, device=device, torch_threads=torch_threads,
                                    rerank_model=rerank_model, rerank_device=rerank_device,
                                    candidates=candidates, rerank_dtype=rerank_dtype)
    if profile == "remote_api":
        return RemoteRetriever(url)
    raise ValueError(f"unknown retriever profile {profile!r}; expected one of {PROFILES}")


# --------------------------------------------------------------------------
# Latency artifact -> controller signals
# --------------------------------------------------------------------------

def load_latency_profiles(path: str | Path = DEFAULT_PROFILES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieval_signals(profile: str, path: str | Path = DEFAULT_PROFILES_PATH,
                      expected_quantile: str = "p50") -> dict:
    """Build ``signals["retrieval"]`` (CONTRACTS.md section 6) for
    ``profile``.

    ``profile`` is a canonical name (``faiss``, ``faiss_rerank``,
    ``remote_api``), resolved through the artifact's ``canonical`` map, or
    the name of any measured variant. ``expected_quantile`` is one of
    p10..p99 or ``"mean"``.
    """
    art = load_latency_profiles(path)
    name = art.get("canonical", {}).get(profile, profile)
    entry = art.get("profiles", {}).get(name)
    if entry is None:
        raise KeyError(f"no latency profile {profile!r} in {path}")
    q = {k: float(entry["quantiles_ms"][k]) for k in QUANTILES}
    if expected_quantile == "mean":
        expected = float(entry["mean_ms"])
    elif expected_quantile in q:
        expected = q[expected_quantile]
    else:
        raise ValueError(f"expected_quantile must be one of {QUANTILES} or 'mean'")
    return {"latency_profile": profile, "expected_latency_ms": expected,
            "latency_quantiles_ms": q}
