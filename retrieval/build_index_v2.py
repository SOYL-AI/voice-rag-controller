"""Build a FAISS index from any research-v2 `corpus.jsonl` and (optionally)
evaluate gold-passage retrieval on one split.

Same retrieval stack as v1 (`retrieval/build_index.py` / `retrieval/search.py`):
BAAI/bge-small-en-v1.5 through sentence-transformers, one vector per corpus
passage, exact `faiss.IndexFlatL2`, no query instruction. bge's
sentence-transformers pipeline ends in a Normalize module, so L2 ranking is
identical to cosine ranking. v2 corpora already hold passages of <= ~350 words;
any longer passage is split into 350-word windows (all windows map back to
the same `doc_id`).

By default each passage is embedded as "<title>\\n<text>" so that passages
that never repeat the article title stay retrievable; `--no-title` embeds the
text alone (the literal v1 behaviour, whose hotel files carry their own
headings).

Outputs (in --out-dir, default `<corpus dir>/index/`, gitignored):
    index.faiss       FAISS IndexFlatL2
    chunks.json       [{"filename": doc_id, "doc_id", "title", "text"}] (same keys search.py reads)
    build_info.json   model, settings, corpus sha256, sizes
    eval_<split>.json retrieval hit@k (only with --eval-items)

Examples:
    set PYTHONUTF8=1
    python retrieval/build_index_v2.py --corpus datasets/v2/d1_open/corpus.jsonl \
        --eval-items datasets/v2/d1_open/items.jsonl --split dev
    python retrieval/build_index_v2.py --corpus datasets/v2/d1_open/corpus.jsonl \
        --eval-items datasets/v2/d1_open/items.jsonl --split dev --eval-only

`--split test` is refused unless docs/research_v2/PREREG_FROZEN exists
(CONTRACTS.md section 7).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_FROZEN = REPO_ROOT / "docs" / "research_v2" / "PREREG_FROZEN"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_WORDS = 350


def load_corpus(path: Path):
    """Read corpus.jsonl or corpus.jsonl.gz (either may be given)."""
    if not path.exists():
        alt = path.with_name(path.name + ".gz") if path.suffix != ".gz" else path.with_suffix("")
        if alt.exists():
            path = alt
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_chunks(corpus, max_words: int):
    chunks = []
    for d in corpus:
        words = d["text"].split()
        if len(words) <= max_words:
            pieces = [d["text"]]
        else:
            pieces = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
        for p in pieces:
            chunks.append({"filename": d["doc_id"], "doc_id": d["doc_id"],
                           "title": d.get("title", ""), "text": p})
    return chunks


def embed_input(chunk, use_title: bool) -> str:
    if use_title and chunk["title"]:
        return f"{chunk['title']}\n{chunk['text']}"
    return chunk["text"]


def content_sha256(corpus) -> str:
    """Hash of (doc_id, title, text) in file order; identical for .jsonl and .jsonl.gz."""
    h = hashlib.sha256()
    for d in corpus:
        h.update(json.dumps([d["doc_id"], d.get("title", ""), d["text"]], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build(args, model):
    import faiss

    corpus = load_corpus(args.corpus)
    chunks = to_chunks(corpus, args.max_words)
    n_split = len(chunks) - len(corpus)
    print(f"Loaded {len(corpus)} passages -> {len(chunks)} chunks "
          f"({n_split} extra from passages over {args.max_words} words).")
    corpus_sha = content_sha256(corpus)
    n_passages = len(corpus)
    del corpus
    # Stream: embed fixed-size blocks and add each to the index immediately, so
    # peak memory is one block of activations plus the (small) flat index.
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    t0 = time.time()
    for start in range(0, len(chunks), args.block_size):
        block = [embed_input(c, not args.no_title) for c in chunks[start:start + args.block_size]]
        emb = model.encode(block, batch_size=args.batch_size, convert_to_numpy=True,
                           show_progress_bar=False).astype("float32")
        index.add(emb)
        done = start + len(block)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  embedded {done}/{len(chunks)} ({rate:.1f}/s, eta {(len(chunks) - done) / rate:.0f}s)",
              flush=True)
    print(f"Embedded {index.ntotal} x {dim} in {time.time() - t0:.0f}s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.out_dir / "index.faiss"))
    with (args.out_dir / "chunks.json").open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False)
    info = {
        "model": MODEL_NAME, "index": "IndexFlatL2", "dim": int(dim),
        "embed_input": "text" if args.no_title else "title\\ntext",
        "max_words": args.max_words, "n_passages": n_passages, "n_chunks": len(chunks),
        "corpus": str(args.corpus), "corpus_content_sha256": corpus_sha,
        "embed_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "build_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"Wrote index with {index.ntotal} vectors to {args.out_dir}")


def evaluate(args, model):
    import faiss

    if args.split == "test" and not PREREG_FROZEN.exists():
        sys.exit("Refusing --split test: docs/research_v2/PREREG_FROZEN does not exist "
                 "(test split is sealed, CONTRACTS.md section 7).")
    index = faiss.read_index(str(args.out_dir / "index.faiss"))
    chunks = json.loads((args.out_dir / "chunks.json").read_text(encoding="utf-8"))
    with args.eval_items.open(encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    items = [it for it in items if it["split"] == args.split]
    if not items:
        sys.exit(f"no items in split {args.split}")
    ks = sorted(args.ks)
    depth = max(50, 10 * max(ks))
    q = model.encode([args.query_prefix + it["question"] for it in items],
                     batch_size=args.batch_size, convert_to_numpy=True).astype("float32")
    _, idx = index.search(q, depth)

    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for it, row in zip(items, idx):
        ranked, seen = [], set()
        for i in row:
            if i < 0:
                continue
            d = chunks[i]["doc_id"]
            if d not in seen:
                seen.add(d)
                ranked.append(d)
        gold = set(it["gold_doc_ids"])
        keys = ["all", it["item_type"]]
        meta = it.get("meta", {})
        if it["item_type"] == "popqa":
            keys += [f"popqa_bin{meta.get('pop_bin')}", f"popqa_rel:{meta.get('relation')}"]
        elif "hotpot_type" in meta:
            keys.append(f"multihop_{meta['hotpot_type']}")
        rec = {}
        for k in ks:
            top = set(ranked[:k])
            rec[f"hit@{k}"] = float(bool(top & gold))
            if it["item_type"] == "popqa" and meta.get("primary_gold_doc_id"):
                rec[f"primary_hit@{k}"] = float(meta["primary_gold_doc_id"] in top)
            if it["item_type"] == "multihop":
                rec[f"all_gold@{k}"] = float(gold <= top)
        first = next((r for r, d in enumerate(ranked, 1) if d in gold), None)
        rec["mrr@50"] = 1.0 / first if first else 0.0
        for key in keys:
            for m, v in rec.items():
                groups[key][m].append(v)

    def order(key):
        return (0 if key == "all" else 1 if "_" not in key and ":" not in key else 2, key)

    result = {"split": args.split, "n_items": len(items), "query": "full question",
              "query_prefix": args.query_prefix, "embed_input": None, "groups": {}}
    info_path = args.out_dir / "build_info.json"
    if info_path.exists():
        result["embed_input"] = json.loads(info_path.read_text(encoding="utf-8"))["embed_input"]
    for key in sorted(groups, key=order):
        metrics = groups[key]
        n = len(metrics["mrr@50"])
        result["groups"][key] = {"n": n, **{m: round(sum(v) / len(v), 4) for m, v in metrics.items()}}
    out = args.out_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    cols = [f"hit@{k}" for k in ks] + ["mrr@50"]
    print(f"\nRetrieval on split={args.split} (n={len(items)}), query = full question")
    print(f"{'group':28s} {'n':>5s} " + " ".join(f"{c:>8s}" for c in cols))
    for key, m in result["groups"].items():
        print(f"{key:28s} {m['n']:5d} " + " ".join(f"{m[c]:8.3f}" for c in cols))
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Build/evaluate a FAISS index for a v2 corpus.jsonl")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <corpus dir>/index")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=2048, help="passages embedded per streamed block")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (shared machine)")
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    ap.add_argument("--no-title", action="store_true", help="embed passage text only (literal v1)")
    ap.add_argument("--eval-items", type=Path, default=None)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--query-prefix", default="", help="optional query instruction (v1 uses none)")
    ap.add_argument("--eval-only", action="store_true", help="skip building; evaluate existing index")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = args.corpus.parent / "index"
    if args.eval_only and not args.eval_items:
        sys.exit("--eval-only needs --eval-items")
    if args.eval_items and args.split == "test" and not PREREG_FROZEN.exists():
        sys.exit("Refusing --split test: docs/research_v2/PREREG_FROZEN does not exist "
                 "(test split is sealed, CONTRACTS.md section 7).")

    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(args.threads)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    if not args.eval_only:
        build(args, model)
    if args.eval_items:
        evaluate(args, model)


if __name__ == "__main__":
    main()
