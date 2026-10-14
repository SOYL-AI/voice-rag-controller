"""HTTP retrieval service (the server side of the ``remote_api`` profile).

Endpoints::

    POST /search   {"query": str, "k": int = 3}
                   -> {"hits": [...], "server_ms": {...}, "profile": str}
    GET  /healthz  -> {"status": "ok", "ready": bool, "backend": {...}}

``server_ms`` holds the backend's component timings for that request
(``embed_ms``, ``search_ms``, ``rerank_ms``, ``total_ms``) plus
``handler_ms``: wall time inside the request handler around the backend
call and hit packing. A client computes transport overhead as its own
round-trip time minus ``handler_ms``.

Backend configuration (CLI flags override environment variables)::

    --backend      RETRIEVAL_BACKEND        faiss | faiss_rerank   (default faiss)
    --index-dir    RETRIEVAL_INDEX_DIR      v2 index dir; unset = v1 hotel index
    --device       RETRIEVAL_DEVICE         embedding device (default cpu)
    --torch-threads RETRIEVAL_TORCH_THREADS torch intra-op threads (default 4; "default" = leave torch's own)
    --rerank-model RETRIEVAL_RERANK_MODEL   cross-encoder name
    --rerank-device RETRIEVAL_RERANK_DEVICE (default cuda, falls back to cpu)
    --candidates   RETRIEVAL_CANDIDATES     FAISS candidates before reranking (default 50)

Optional auth: if the environment variable ``RETRIEVAL_SERVICE_API_KEY`` is
set, requests must carry the same value in the ``x-api-key`` header. No key is
ever stored in code.

Run locally::

    python -m retrieval.service --port 8765 --backend faiss
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from retrieval.profiles import (
    DEFAULT_RERANK_MODEL,
    SERVICE_API_KEY_ENV,
    Retriever,
    make_retriever,
)


class SearchRequest(BaseModel):
    query: str
    k: int = Field(3, ge=1, le=100)


def config_from_env() -> dict:
    threads = os.environ.get("RETRIEVAL_TORCH_THREADS", "4")
    return {
        "backend": os.environ.get("RETRIEVAL_BACKEND", "faiss"),
        "index_dir": os.environ.get("RETRIEVAL_INDEX_DIR") or None,
        "device": os.environ.get("RETRIEVAL_DEVICE", "cpu"),
        "torch_threads": None if threads.lower() in ("", "default", "none") else int(threads),
        "rerank_model": os.environ.get("RETRIEVAL_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        "rerank_device": os.environ.get("RETRIEVAL_RERANK_DEVICE", "cuda"),
        "candidates": int(os.environ.get("RETRIEVAL_CANDIDATES", "50")),
    }


def backend_from_config(cfg: dict) -> Retriever:
    if cfg["backend"] not in ("faiss", "faiss_rerank"):
        raise ValueError(f"service backend must be faiss or faiss_rerank, got {cfg['backend']!r}")
    return make_retriever(cfg["backend"], index_dir=cfg["index_dir"], device=cfg["device"],
                          torch_threads=cfg["torch_threads"], rerank_model=cfg["rerank_model"],
                          rerank_device=cfg["rerank_device"], candidates=cfg["candidates"])


def create_app(backend: Optional[Retriever] = None,
               backend_factory: Optional[Callable[[], Retriever]] = None,
               warmup: bool = True) -> FastAPI:
    """Build the app. Pass a ready ``backend`` (tests), or a
    ``backend_factory``; with neither, the backend comes from the
    environment (:func:`config_from_env`). The backend is created and warmed
    up at startup so the first request does not pay model loading."""
    state: dict = {"backend": backend, "ready": backend is not None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["backend"] is None:
            factory = backend_factory or (lambda: backend_from_config(config_from_env()))
            state["backend"] = factory()
        if warmup and not state["ready"]:
            state["backend"].warmup()
        state["ready"] = True
        yield

    app = FastAPI(title="retrieval-service", lifespan=lifespan)

    def _check_key(x_api_key: Optional[str]) -> None:
        expected = os.environ.get(SERVICE_API_KEY_ENV)
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid or missing x-api-key")

    def _backend() -> Retriever:
        b = state["backend"]
        if b is None:
            raise HTTPException(status_code=503, detail="backend not loaded")
        return b

    @app.get("/healthz")
    def healthz():
        b = state["backend"]
        info = None
        if b is not None and state["ready"]:
            try:
                info = b.describe()
            except Exception as exc:  # describe must never take the probe down
                info = {"error": repr(exc)}
        return {"status": "ok", "ready": bool(state["ready"]), "backend": info}

    # A plain ``def`` handler: FastAPI runs it in its worker thread pool, so a
    # slow backend never blocks the event loop.
    @app.post("/search")
    def search(req: SearchRequest, x_api_key: Optional[str] = Header(default=None)):
        _check_key(x_api_key)
        b = _backend()
        t0 = time.perf_counter()
        hits = b.search(req.query, k=req.k)
        timing = dict(getattr(b, "last_timing", {}) or {})
        timing["handler_ms"] = (time.perf_counter() - t0) * 1000.0
        return JSONResponse({"hits": hits, "server_ms": timing, "profile": b.profile})

    return app


def _main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrieval.service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--backend", choices=["faiss", "faiss_rerank"])
    p.add_argument("--index-dir")
    p.add_argument("--device")
    p.add_argument("--torch-threads", help='int, or "default" to keep torch\'s own setting')
    p.add_argument("--rerank-model")
    p.add_argument("--rerank-device")
    p.add_argument("--candidates", type=int)
    p.add_argument("--log-level", default="warning")
    args = p.parse_args(argv)

    env_map = {"backend": "RETRIEVAL_BACKEND", "index_dir": "RETRIEVAL_INDEX_DIR",
               "device": "RETRIEVAL_DEVICE", "torch_threads": "RETRIEVAL_TORCH_THREADS",
               "rerank_model": "RETRIEVAL_RERANK_MODEL", "rerank_device": "RETRIEVAL_RERANK_DEVICE",
               "candidates": "RETRIEVAL_CANDIDATES"}
    for attr, env in env_map.items():
        val = getattr(args, attr)
        if val is not None:
            os.environ[env] = str(val)

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level,
                access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
