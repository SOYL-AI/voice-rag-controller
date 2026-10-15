"""
measure_rewrite_latency.py — latency of the LLM query rewrite that a voice
RAG system runs on the live partial transcript before retrieving.

The vector call alone is ~20-70 ms (retrieval/latency_profiles_v2.json), so
it hides almost nothing worth timing; the whole retrieval operation on a
partial is  rewrite (LLM) + embed/search (+ rerank) (+ network). This tool
measures the rewrite part on the served model with the project's real
rewrite prompt and call (retrieval/rewrite.py: same messages, same client
and generation settings as rewrite_query()).

Query pool (what partials look like at dispatch):
    v1          the 50 v1 questions: every query-ready word prefix
                (controller/timing.py is_query_ready) and the full question
    d2_support  DEV items only (CONTRACTS §7; test ids are only loaded to
                exclude them): query-ready prefixes and full questions, with
                the item's conversation history passed to the rewrite as
                "role: content" lines

Load levels: requests are issued by N concurrent workers (default 1, 8, 32),
each with its own client, so per-call latency is measured at that load.
Hard request budget across all levels (default 1500, incl. warm-up).

    python -m retrieval.measure_rewrite_latency --env-file <.env> \\
        --levels 1:150 8:400 32:800 --out-dir analysis/rewrite_latency_v2
    python -m retrieval.measure_rewrite_latency --quality-only --out-dir analysis/rewrite_latency_v2

Outputs: rewrite_latency.json (quantiles per level and source, samples),
rewrites.jsonl (every call: input, rewrite, timings). --quality adds hit@k
of retrieval on the raw partial vs on its rewrite (no requests).
Credentials are read from the env file into the process environment and
never printed or written.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

QS = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
D2_ITEMS = _REPO_ROOT / "datasets" / "v2" / "d2_support" / "items.jsonl"


def quantiles(xs) -> dict:
    a = np.asarray([x for x in xs if x is not None], float)
    if a.size == 0:
        return {k: None for k in QS}
    return {k: round(float(np.quantile(a, q)), 2) for k, q in QS.items()}


def history_lines(history) -> List[str]:
    out = []
    for h in history or []:
        if isinstance(h, dict):
            out.append(f"{h.get('role', 'user')}: {h.get('content', '')}")
        else:
            out.append(str(h))
    return out


def ready_prefixes(text: str) -> List[str]:
    from controller.timing import is_query_ready

    words = text.split()
    out = []
    for n in range(1, len(words)):
        pre = " ".join(words[:n])
        if is_query_ready(pre):
            out.append(pre)
    return out


def build_pool(split: str = "dev") -> List[dict]:
    from harness.v2_split import check_split, filter_split, load_jsonl

    check_split(split)
    pool = []
    scen = json.loads((_REPO_ROOT / "simulator" / "scenarios.json").read_text(encoding="utf-8"))
    for s in scen:
        q = s["question"]
        n = len(q.split())
        for pre in ready_prefixes(q) + [q]:
            pool.append({"source": "v1", "item_id": f"v1-{s['id']:03d}", "text": pre, "history": [],
                         "word_frac": len(pre.split()) / n, "full": pre == q,
                         "gold": [s["document"]], "retrieval_needed": bool(s.get("retrieval_needed"))})
    if D2_ITEMS.exists():
        for it in filter_split(load_jsonl(D2_ITEMS), split):
            q = it["question"]
            n = len(q.split())
            for pre in ready_prefixes(q) + [q]:
                pool.append({"source": "d2_support", "item_id": it["id"], "text": pre,
                             "history": history_lines(it.get("history")),
                             "word_frac": len(pre.split()) / n, "full": pre == q,
                             "gold": it.get("gold_doc_ids") or [],
                             "retrieval_needed": bool(it.get("retrieval_needed_authored"))})
    return pool


def sample_level(pool: List[dict], n: int, seed: int, d2_share: float = 2 / 3) -> List[dict]:
    rng = random.Random(seed)
    v1 = [p for p in pool if p["source"] == "v1"]
    d2 = [p for p in pool if p["source"] == "d2_support"]
    n_d2 = int(round(n * d2_share)) if d2 else 0
    out = [rng.choice(d2) for _ in range(n_d2)] + [rng.choice(v1) for _ in range(n - n_d2)]
    rng.shuffle(out)
    return out


class Budget:
    def __init__(self, limit: int):
        self.limit, self.used, self.lock = limit, 0, threading.Lock()

    def take(self) -> None:
        with self.lock:
            if self.used >= self.limit:
                raise RuntimeError(f"request budget exhausted ({self.used}/{self.limit})")
            self.used += 1


def run_level(entries: List[dict], concurrency: int, budget: Budget, cfg: dict) -> List[dict]:
    from llm.client import VLLMClient
    from retrieval.rewrite import build_messages

    local = threading.local()
    t0_level = time.perf_counter()

    def client():
        if getattr(local, "c", None) is None:
            local.c = VLLMClient(cfg)
        return local.c

    def one(e):
        budget.take()
        msgs = build_messages(e["history"], e["text"])
        t0 = time.perf_counter()
        err = None
        try:
            gen = client().generate(msgs)
            text, ttft, ntok = gen.text.strip(), gen.ttft_ms, len(gen.tokens)
        except Exception as exc:  # recorded, not fatal
            text, ttft, ntok, err = None, None, None, f"{type(exc).__name__}: {exc}"[:300]
        t1 = time.perf_counter()
        return {**{k: e[k] for k in ("source", "item_id", "text", "word_frac", "full", "gold", "retrieval_needed")},
                "n_history": len(e["history"]), "concurrency": concurrency,
                "start_s": round(t0 - t0_level, 3), "total_ms": round((t1 - t0) * 1000.0, 2),
                "ttft_ms": ttft, "n_tokens": ntok, "rewrite": text, "error": err}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(one, entries))


def summarize(rows: List[dict]) -> dict:
    out = {}
    levels = sorted({r["concurrency"] for r in rows})
    for c in levels:
        rs = [r for r in rows if r["concurrency"] == c and r["error"] is None]
        errs = sum(1 for r in rows if r["concurrency"] == c and r["error"] is not None)
        span = max((r["start_s"] + r["total_ms"] / 1000 for r in rs), default=0)
        d = {"n": len(rs), "errors": errs, "throughput_rps": round(len(rs) / span, 2) if span else None,
             "total_ms": quantiles([r["total_ms"] for r in rs]),
             "mean_total_ms": round(float(np.mean([r["total_ms"] for r in rs])), 2) if rs else None,
             "ttft_ms": quantiles([r["ttft_ms"] for r in rs]),
             "n_tokens": quantiles([r["n_tokens"] for r in rs]),
             "samples_ms": [r["total_ms"] for r in rs],
             "by_source": {}}
        for src in sorted({r["source"] for r in rs}):
            ss = [r for r in rs if r["source"] == src]
            d["by_source"][src] = {"n": len(ss), "total_ms": quantiles([r["total_ms"] for r in ss]),
                                   "partial_total_ms": quantiles([r["total_ms"] for r in ss if not r["full"]]),
                                   "full_total_ms": quantiles([r["total_ms"] for r in ss if r["full"]])}
        out[str(c)] = d
    return out


def quality(rows: List[dict], k: int = 3) -> dict:
    """hit@k of retrieval on the raw (partial) text vs on its rewrite, faiss
    profile, v1 index for v1 rows and the d2_support index for d2 rows."""
    from retrieval.profiles import make_retriever

    idx = {"v1": None, "d2_support": str(_REPO_ROOT / "retrieval" / "indexes" / "d2_support")}
    retr = {}
    cache: Dict[tuple, List[str]] = {}

    def parents(src, q):
        key = (src, q)
        if key not in cache:
            if src not in retr:
                retr[src] = make_retriever("faiss", index_dir=idx[src])
            hits = retr[src].search(q, k=k)
            cache[key] = [h.get("parent_doc_id") or h.get("filename") for h in hits]
        return cache[key]

    res = {}
    for src in ("v1", "d2_support"):
        if src == "d2_support" and not Path(idx[src]).exists():
            continue
        rs = [r for r in rows if r["source"] == src and r["rewrite"] and r["retrieval_needed"] and r["gold"]]
        seen, uniq = set(), []
        for r in rs:
            if (r["item_id"], r["text"]) not in seen:
                seen.add((r["item_id"], r["text"]))
                uniq.append(r)
        out = {}
        for part, sel in (("partial", [r for r in uniq if not r["full"]]), ("full", [r for r in uniq if r["full"]])):
            if not sel:
                continue
            raw = [any(g in parents(src, r["text"]) for g in r["gold"]) for r in sel]
            rew = [any(g in parents(src, r["rewrite"]) for g in r["gold"]) for r in sel]
            out[part] = {"n": len(sel), f"hit@{k}_raw": float(np.mean(raw)), f"hit@{k}_rewrite": float(np.mean(rew))}
        res[src] = out
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--levels", nargs="+", default=["1:150", "8:400", "32:800"],
                    help="concurrency:n_requests per load level")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max-requests", type=int, default=1500)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", default="analysis/rewrite_latency_v2")
    ap.add_argument("--quality-only", action="store_true", help="recompute hit@k from rewrites.jsonl (no requests)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quality_only:
        rows = [json.loads(l) for l in open(out_dir / "rewrites.jsonl", encoding="utf-8")]
        art = json.loads((out_dir / "rewrite_latency.json").read_text(encoding="utf-8"))
        art["quality"] = quality(rows)
        (out_dir / "rewrite_latency.json").write_text(json.dumps(art, indent=1), encoding="utf-8")
        print(json.dumps(art["quality"], indent=1))
        return 0

    from harness.measure_llm_tts import load_env_file

    keys = load_env_file(args.env_file)
    print(f"loaded {len(keys)} env keys (values not shown)")
    os.environ["VOICE_RAG_LLM_BACKEND"] = "vllm"
    os.environ.setdefault("VOICE_RAG_MODEL", "mistral7b_v03")
    from llm.config import load_config

    cfg = load_config()
    if cfg["model"] != "mistralai/Mistral-7B-Instruct-v0.3":
        raise SystemExit(f"refusing to run against {cfg['model']}: Mistral-7B-Instruct-v0.3 is the only model")
    levels = [(int(a), int(b)) for a, b in (lv.split(":") for lv in args.levels)]
    planned = args.warmup + sum(n for _, n in levels)
    if planned > args.max_requests:
        raise SystemExit(f"planned {planned} requests > budget {args.max_requests}")
    pool = build_pool(args.split)
    print(f"query pool: {len(pool)} ({sum(p['source'] == 'v1' for p in pool)} v1, "
          f"{sum(p['source'] == 'd2_support' for p in pool)} d2_support dev); planned requests {planned}")
    budget = Budget(args.max_requests)

    warm = run_level(sample_level(pool, args.warmup, args.seed + 99), min(args.warmup, 4), budget, cfg)
    print(f"warm-up: {[round(r['total_ms']) for r in warm]} ms (discarded)")

    rows = []
    for i, (c, n) in enumerate(levels):
        t0 = time.perf_counter()
        rs = run_level(sample_level(pool, n, args.seed + i), c, budget, cfg)
        rows.extend(rs)
        ok = [r["total_ms"] for r in rs if r["error"] is None]
        print(f"concurrency {c:3d}: n={len(ok)} errors={len(rs) - len(ok)} p50={np.median(ok):.0f} "
              f"p90={np.quantile(ok, 0.9):.0f} ms  wall {time.perf_counter() - t0:.0f} s  used {budget.used}")

    with open(out_dir / "rewrites.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    art = {
        "schema_version": 1,
        "model": cfg["model"],
        "prompt": "retrieval/rewrite.py build_messages (history, partial)",
        "generation": {k: cfg.get("generation", {}).get(k) for k in ("max_tokens", "temperature", "top_p")},
        "logprobs_top_k": (cfg.get("logprobs") or {}).get("top_k"),
        "split": args.split,
        "levels": {str(c): n for c, n in levels},
        "requests_used": budget.used,
        "by_concurrency": summarize(rows),
    }
    try:
        art["quality"] = quality(rows)
    except Exception as exc:  # quality is optional; latency is the deliverable
        art["quality_error"] = repr(exc)
    (out_dir / "rewrite_latency.json").write_text(json.dumps(art, indent=1), encoding="utf-8")
    print(f"wrote {out_dir / 'rewrite_latency.json'}; requests used {budget.used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
