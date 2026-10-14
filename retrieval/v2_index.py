"""Build and load a FAISS index over a research-v2 ``corpus.jsonl``.

The corpus schema is docs/research_v2/CONTRACTS.md section 3: one JSON object
per line with ``doc_id``, ``dataset``, ``domain``, ``org``, ``title``,
``text``, ``source`` and ``licence``.

Defaults match the v1 index (retrieval/build_index.py) so v1 and v2 numbers
are comparable:

* embedding model ``BAAI/bge-small-en-v1.5`` (384-d),
* embeddings are **not** normalised,
* ``faiss.IndexFlatL2`` (exact search; scores are squared L2 distances, lower
  is better).

``--metric ip`` (with ``--normalize``, which it implies unless
``--no-normalize`` is given) builds an ``IndexFlatIP`` over unit-norm vectors,
i.e. cosine similarity (higher is better). The choice is written to
``meta.json`` so the query side (retrieval/profiles.py) normalises queries the
same way.

Passages longer than ``--max-words`` (default 350) are split into
consecutive word windows. Each window gets ``doc_id = "<parent>#c<i>"`` and
keeps ``parent_doc_id = "<parent>"`` so hit@k can be computed on the parent
ids that ``gold_doc_ids`` refers to. Unsplit passages have
``doc_id == parent_doc_id``.

Output directory layout::

    index.faiss   FAISS index, one vector per row of docs.jsonl (same order)
    docs.jsonl    {doc_id, parent_doc_id, chunk_index, n_chunks, title, text,
                   dataset, domain, org, source, licence}
    meta.json     {model, dim, n_docs, n_parent_docs, metric, normalize,
                   max_words, embed_title, corpus_sha256, corpus_file,
                   built_at, faiss_version}

CLI::

    python -m retrieval.v2_index build --corpus datasets/v2/d2_support/corpus.jsonl \\
        --out .scratch/index_d2_support
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_WORDS = 350
PASSTHROUGH_FIELDS = ("dataset", "domain", "org", "source", "licence")

# An encoder is anything mapping a list of strings to an (n, dim) float array.
Encoder = Callable[[list], np.ndarray]


# --------------------------------------------------------------------------
# Corpus reading and chunking
# --------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_corpus(path: str | Path) -> list[dict]:
    """Read a corpus.jsonl; validates that doc_id and text are present and
    doc_ids are unique."""
    docs: list[dict] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "doc_id" not in rec or "text" not in rec:
                raise ValueError(f"{path}:{line_no}: record needs doc_id and text")
            doc_id = str(rec["doc_id"])
            if "#c" in doc_id:
                raise ValueError(
                    f"{path}:{line_no}: doc_id {doc_id!r} contains the reserved chunk marker '#c'"
                )
            if doc_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate doc_id {doc_id!r}")
            seen.add(doc_id)
            docs.append(rec)
    return docs


def chunk_words(text: str, max_words: int = DEFAULT_MAX_WORDS) -> list[str]:
    """Split ``text`` into consecutive windows of at most ``max_words`` words.

    Text at or under the limit is returned unchanged (whitespace preserved) as
    a single chunk. Longer text is split on whitespace into near-equal
    windows (so the last chunk is not a tiny remainder), with no overlap.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    n_chunks = -(-len(words) // max_words)  # ceil
    size = -(-len(words) // n_chunks)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def expand_chunks(corpus: Iterable[dict], max_words: int = DEFAULT_MAX_WORDS) -> list[dict]:
    """Turn corpus records into index rows (one per chunk)."""
    rows: list[dict] = []
    for rec in corpus:
        parent = str(rec["doc_id"])
        pieces = chunk_words(rec["text"], max_words)
        for i, piece in enumerate(pieces):
            row = {
                "doc_id": parent if len(pieces) == 1 else f"{parent}#c{i}",
                "parent_doc_id": parent,
                "chunk_index": i,
                "n_chunks": len(pieces),
                "title": rec.get("title") or "",
                "text": piece,
            }
            for k in PASSTHROUGH_FIELDS:
                row[k] = rec.get(k)
            rows.append(row)
    return rows


def embed_text(row: dict, embed_title: bool = True) -> str:
    """Text actually embedded for a row. v1 embedded whole files whose first
    line is the title, so the title is prepended by default for parity."""
    if embed_title and row.get("title"):
        return f"{row['title']}\n\n{row['text']}"
    return row["text"]


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

def sentence_transformer_encoder(
    model_name: str = DEFAULT_MODEL, device: str = "cpu", batch_size: int = 64
) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list) -> np.ndarray:
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 256,
            normalize_embeddings=False,
        )

    return encode


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


# --------------------------------------------------------------------------
# Build / load
# --------------------------------------------------------------------------

def build_index(
    corpus_path: str | Path,
    out_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    metric: str = "l2",
    normalize: Optional[bool] = None,
    max_words: int = DEFAULT_MAX_WORDS,
    embed_title: bool = True,
    device: str = "cpu",
    batch_size: int = 64,
    encoder: Optional[Encoder] = None,
) -> dict:
    """Build an index directory from ``corpus_path``; returns the meta dict.

    ``normalize=None`` means: False for ``metric="l2"`` (v1 parity) and True
    for ``metric="ip"`` (cosine). ``encoder`` may be injected (tests use a
    tiny deterministic fake); otherwise bge-small is loaded on ``device``.
    """
    import faiss

    if metric not in ("l2", "ip"):
        raise ValueError(f"metric must be 'l2' or 'ip', got {metric!r}")
    if normalize is None:
        normalize = metric == "ip"
    if metric == "ip" and not normalize:
        print("warning: metric=ip without normalisation is a raw dot product, not cosine",
              file=sys.stderr)

    corpus_path = Path(corpus_path)
    out_dir = Path(out_dir)
    corpus = read_corpus(corpus_path)
    if not corpus:
        raise ValueError(f"{corpus_path} has no records")
    rows = expand_chunks(corpus, max_words)

    if encoder is None:
        encoder = sentence_transformer_encoder(model_name, device=device, batch_size=batch_size)
    emb = np.asarray(encoder([embed_text(r, embed_title) for r in rows]), dtype="float32")
    if emb.ndim != 2 or emb.shape[0] != len(rows):
        raise ValueError(f"encoder returned shape {emb.shape} for {len(rows)} rows")
    if normalize:
        emb = _l2_normalize(emb).astype("float32")
    dim = int(emb.shape[1])
    index = faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(emb))

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "docs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "model": model_name,
        "dim": dim,
        "n_docs": len(rows),
        "n_parent_docs": len(corpus),
        "metric": metric,
        "normalize": bool(normalize),
        "score": "squared_l2_distance (lower is better)" if metric == "l2"
                 else "inner_product (higher is better)",
        "max_words": max_words,
        "embed_title": embed_title,
        "corpus_sha256": sha256_file(corpus_path),
        "corpus_file": corpus_path.name,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "faiss_version": getattr(faiss, "__version__", None),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


@dataclass
class LoadedIndex:
    index: Any                      # faiss.Index
    docs: list[dict]                # row i <-> vector i
    meta: dict
    path: Optional[Path] = None
    parent_ids: list[str] = field(default_factory=list)

    @property
    def n_docs(self) -> int:
        return len(self.docs)


def load_index(index_dir: str | Path) -> LoadedIndex:
    """Load an index directory written by :func:`build_index`."""
    import faiss

    index_dir = Path(index_dir)
    for name in ("index.faiss", "docs.jsonl", "meta.json"):
        if not (index_dir / name).exists():
            raise FileNotFoundError(
                f"{index_dir / name} is missing; build it with "
                f"`python -m retrieval.v2_index build --corpus <corpus.jsonl> --out {index_dir}`"
            )
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "docs.jsonl", "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    with open(index_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    if index.ntotal != len(docs):
        raise ValueError(f"{index_dir}: index has {index.ntotal} vectors but docs.jsonl has {len(docs)} rows")
    return LoadedIndex(index=index, docs=docs, meta=meta, path=index_dir,
                       parent_ids=[d.get("parent_doc_id", d["doc_id"]) for d in docs])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrieval.v2_index", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build an index directory from a corpus.jsonl")
    b.add_argument("--corpus", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--metric", choices=["l2", "ip"], default="l2",
                   help="l2 = IndexFlatL2 (v1 default); ip = IndexFlatIP")
    norm = b.add_mutually_exclusive_group()
    norm.add_argument("--normalize", dest="normalize", action="store_true", default=None,
                      help="L2-normalise embeddings (default: off for l2, on for ip)")
    norm.add_argument("--no-normalize", dest="normalize", action="store_false")
    b.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    b.add_argument("--no-title", dest="embed_title", action="store_false",
                   help="embed passage text only (default prepends the title, like v1 files)")
    b.add_argument("--device", default="cpu")
    b.add_argument("--batch-size", type=int, default=64)
    i = sub.add_parser("info", help="print meta.json of an index directory")
    i.add_argument("dir")
    args = p.parse_args(argv)

    if args.cmd == "build":
        t0 = time.perf_counter()
        meta = build_index(args.corpus, args.out, model_name=args.model, metric=args.metric,
                           normalize=args.normalize, max_words=args.max_words,
                           embed_title=args.embed_title, device=args.device,
                           batch_size=args.batch_size)
        meta["build_seconds"] = round(time.perf_counter() - t0, 2)
        print(json.dumps(meta, indent=2))
    elif args.cmd == "info":
        print(json.dumps(load_index(args.dir).meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
