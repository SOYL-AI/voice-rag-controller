"""Modal deployment of retrieval/service.py (the ``remote_api`` profile).

DEPLOYMENT REQUIRES EXPLICIT SPEND APPROVAL. This file is not deployed or run
by any test, script or measurement in this repository. It exists so that a
WAN round-trip measurement can be taken once a deployment is approved.

Deploy (from the repository root, with the ``modal`` package installed and a
Modal account configured)::

    # optional knobs, all read at deploy time:
    #   RETRIEVAL_MODAL_BACKEND    faiss | faiss_rerank          (default faiss)
    #   RETRIEVAL_MODAL_GPU        T4 | L4 | none                (default T4)
    #   RETRIEVAL_MODAL_INDEX_DIR  local v2 index dir to bake in (default: v1 hotel index)
    #   RETRIEVAL_MODAL_RERANK_MODEL cross-encoder name
    #   RETRIEVAL_MODAL_RERANK_DTYPE float32 | float16              (default float32)
    modal deploy retrieval/modal_app.py

Modal prints the endpoint after deploying. Do not commit it. Point the client
at it through the environment, then re-run the remote profile::

    set RETRIEVAL_SERVICE_URL=<printed endpoint>
    python -m retrieval.measure_latency --only remote_api_wan --remote-url %RETRIEVAL_SERVICE_URL%

If the service should require a key, create a Modal secret that defines
``RETRIEVAL_SERVICE_API_KEY`` and attach it with ``RETRIEVAL_MODAL_SECRET=<secret name>``;
the client sends the same environment variable as ``x-api-key``.

Importing this module without ``modal`` installed is safe: everything below
is skipped and ``MODAL_AVAILABLE`` is False.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # pragma: no cover - depends on an optional package
    import modal
except ImportError:  # modal is not a dependency of this repository
    modal = None

MODAL_AVAILABLE = modal is not None

_HERE = Path(__file__).resolve().parent
APP_NAME = "voice-rag-retrieval"
BACKEND = os.environ.get("RETRIEVAL_MODAL_BACKEND", "faiss")
GPU = os.environ.get("RETRIEVAL_MODAL_GPU", "T4")
INDEX_DIR = os.environ.get("RETRIEVAL_MODAL_INDEX_DIR")  # None -> v1 hotel index
RERANK_MODEL = os.environ.get("RETRIEVAL_MODAL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_DTYPE = os.environ.get("RETRIEVAL_MODAL_RERANK_DTYPE", "float32")
SECRET_NAME = os.environ.get("RETRIEVAL_MODAL_SECRET")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
REMOTE_INDEX_DIR = "/index"
REMOTE_PKG_DIR = "/root/retrieval"

# Versions match the host the local latency profiles were measured on.
PIP_PACKAGES = [
    "torch==2.9.1",
    "faiss-cpu==1.15.0",
    "sentence-transformers==5.7.0",
    "fastapi==0.115.0",
    "numpy<2.3",
    "httpx",
]


def _download_models() -> None:
    """Runs at image build time so containers start without downloads."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer(EMBED_MODEL)
    if BACKEND == "faiss_rerank":
        CrossEncoder(RERANK_MODEL)


if MODAL_AVAILABLE:  # pragma: no cover - never executed in tests
    _gpu = None if GPU.lower() in ("", "none", "cpu") else GPU
    _device = "cuda" if _gpu else "cpu"
    _env = {
        "RETRIEVAL_BACKEND": BACKEND,
        "RETRIEVAL_DEVICE": _device,
        "RETRIEVAL_RERANK_DEVICE": _device,
        "RETRIEVAL_RERANK_MODEL": RERANK_MODEL,
        "RETRIEVAL_RERANK_DTYPE": RERANK_DTYPE,
        "RETRIEVAL_TORCH_THREADS": "4",
    }
    if INDEX_DIR:
        _env["RETRIEVAL_INDEX_DIR"] = REMOTE_INDEX_DIR

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(*PIP_PACKAGES)
        .env(_env)
        .run_function(_download_models)
        .add_local_python_source("retrieval")
    )
    if INDEX_DIR:
        image = image.add_local_dir(INDEX_DIR, remote_path=REMOTE_INDEX_DIR)
    else:
        image = (
            image.add_local_file(str(_HERE / "index.faiss"), f"{REMOTE_PKG_DIR}/index.faiss")
            .add_local_file(str(_HERE / "chunks.json"), f"{REMOTE_PKG_DIR}/chunks.json")
        )

    app = modal.App(APP_NAME, image=image)
    _secrets = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []

    @app.function(gpu=_gpu, secrets=_secrets, scaledown_window=300, timeout=600)
    @modal.concurrent(max_inputs=8)
    @modal.asgi_app()
    def web():
        from retrieval.service import create_app

        return create_app()
