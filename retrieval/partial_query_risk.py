"""
partial_query_risk.py — how much retrieval quality is lost by retrieving on
the partial transcript available at dispatch time: rho(t) for the v2
controllers (CONTRACTS §0, channel C3).

For every item in the requested split and every speech trace of that item,
retrieve on each ASR partial (exactly the text a live dispatch would see,
CONTRACTS §4) and on the full transcript (asr_final), and score hit@k against
the item's gold_doc_ids. Reported as a function of

    word_frac     words in the partial / words in the final transcript
    remaining_ms  speech end (last word end) minus the partial's t_ms
    elapsed_frac  t_ms / speech end

with and without the query_ready gate (controller/timing.py), since no
policy dispatches before the gate opens.

rho is the conditional loss  P(partial misses | full transcript hits), fit
as a monotone (isotonic) function of word_frac and of remaining_ms on DEV
items only, and written to a JSON artifact with a grid for controllers:

    rho_by_word_frac[k]     [{"word_frac": w, "rho": ...}, ...]
    rho_by_remaining_ms[k]  [{"remaining_ms": r, "rho": ...}, ...]

Also reported: top-k agreement with the full-transcript query (Jaccard of
the top-k parent-doc sets), which needs no gold labels.

    python -m retrieval.partial_query_risk --items datasets/v2/d2_support/items.jsonl \\
        --traces "datasets/v2/d2_support/speech/traces/*.jsonl" --index-dir <v2 index dir> \\
        --profile faiss --split dev --out analysis/partial_query_risk/d2_support_dev.json

The query is the partial verbatim (no LLM rewrite: a rewrite would need an
endpoint call per partial; --rewrite identity is the only offline option).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

WORD_FRAC_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.999, 1.001]
REMAINING_BINS = [-1e9, 0, 250, 500, 1000, 1500, 2000, 3000, 1e9]


def _parent_ids(hits: List[dict]) -> List[str]:
    out = []
    for h in hits:
        pid = h.get("parent_doc_id") or h.get("doc_id") or h.get("filename")
        if pid not in out:
            out.append(pid)
    return out


def hit_at(parents: List[str], gold: Iterable[str], k: int) -> bool:
    g = set(gold)
    return any(p in g for p in parents[:k])


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


class CachedRetriever:
    def __init__(self, retriever, depth: int):
        self.r = retriever
        self.depth = depth
        self.cache: Dict[str, List[str]] = {}

    def parents(self, query: str) -> List[str]:
        q = query.strip()
        if q not in self.cache:
            self.cache[q] = _parent_ids(self.r.search(q, k=self.depth)) if q else []
        return self.cache[q]


def score_traces(items: Dict[str, dict], traces: List[dict], retriever, ks=(1, 3), query_ready_cfg=None) -> List[dict]:
    """One row per (trace, partial) plus one 'final' row per trace."""
    from controller.timing import is_query_ready

    depth = max(ks) * 4  # chunks of one parent can occupy several ranks
    cr = CachedRetriever(retriever, depth)
    rows = []
    for tr in traces:
        item = items.get(tr["item_id"])
        if item is None:
            continue
        gold = item.get("gold_doc_ids") or []
        final_text = (tr.get("asr_final") or {}).get("text") or item.get("question", "")
        n_final = max(len(final_text.split()), 1)
        words = tr.get("words") or []
        speech_end = float(words[-1]["end_ms"]) if words else float(tr.get("duration_ms") or 0)
        full = cr.parents(final_text)
        full_hits = {k: hit_at(full, gold, k) for k in ks}
        for p in tr.get("partials") or []:
            text = p.get("text", "")
            if not text.strip():
                continue
            par = cr.parents(text)
            rows.append({
                "item_id": tr["item_id"],
                "voice_id": tr.get("voice_id"),
                "t_ms": float(p["t_ms"]),
                "word_frac": min(len(text.split()) / n_final, 1.0),
                "remaining_ms": speech_end - float(p["t_ms"]),
                "elapsed_frac": float(p["t_ms"]) / speech_end if speech_end > 0 else 1.0,
                "query_ready": bool(is_query_ready(text, query_ready_cfg)),
                **{f"hit@{k}": hit_at(par, gold, k) for k in ks},
                **{f"full_hit@{k}": full_hits[k] for k in ks},
                **{f"agree@{k}": jaccard(par[:k], full[:k]) for k in ks},
            })
    return rows


def _bin_summary(rows: List[dict], key: str, edges: List[float], ks) -> List[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        if not sel:
            continue
        d = {"bin": [lo, hi], "n": len(sel)}
        for k in ks:
            hp = np.mean([r[f"hit@{k}"] for r in sel])
            hf = np.mean([r[f"full_hit@{k}"] for r in sel])
            cond = [r for r in sel if r[f"full_hit@{k}"]]
            d[f"hit@{k}"] = float(hp)
            d[f"full_hit@{k}"] = float(hf)
            d[f"rho@{k}"] = float(np.mean([not r[f"hit@{k}"] for r in cond])) if cond else None
            d[f"relative_risk@{k}"] = float(1 - hp / hf) if hf > 0 else None
            d[f"agree@{k}"] = float(np.mean([r[f"agree@{k}"] for r in sel]))
        out.append(d)
    return out


def fit_rho(rows: List[dict], key: str, grid: np.ndarray, k: int, increasing: bool) -> List[dict]:
    """Isotonic P(partial miss | full hit) as a function of `key`."""
    from sklearn.isotonic import IsotonicRegression

    cond = [r for r in rows if r[f"full_hit@{k}"]]
    if len(cond) < 2:
        return []
    x = np.array([r[key] for r in cond], float)
    y = np.array([0.0 if r[f"hit@{k}"] else 1.0 for r in cond])
    iso = IsotonicRegression(increasing=increasing, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)
    name = key
    return [{name: float(g), "rho": float(v)} for g, v in zip(grid, iso.predict(grid))]


def analyse(rows: List[dict], ks=(1, 3)) -> dict:
    res = {"n_partials": len(rows), "n_traces": len({(r["item_id"], r["voice_id"]) for r in rows})}
    for gate in ("all_partials", "query_ready_only"):
        sel = rows if gate == "all_partials" else [r for r in rows if r["query_ready"]]
        res[gate] = {
            "n": len(sel),
            "by_word_frac": _bin_summary(sel, "word_frac", WORD_FRAC_BINS, ks),
            "by_remaining_ms": _bin_summary(sel, "remaining_ms", REMAINING_BINS, ks),
            "rho_by_word_frac": {
                str(k): fit_rho(sel, "word_frac", np.round(np.arange(0, 1.0001, 0.05), 3), k, increasing=False)
                for k in ks
            },
            "rho_by_remaining_ms": {
                str(k): fit_rho(sel, "remaining_ms", np.arange(0, 3001, 100.0), k, increasing=True)
                for k in ks
            },
        }
    return res


def rho_at(artifact: dict, k: int = 3, word_frac: Optional[float] = None, remaining_ms: Optional[float] = None,
           gate: str = "query_ready_only") -> Optional[float]:
    """Interpolate rho from an artifact (controllers' accessor)."""
    sec = artifact["analysis"][gate]
    if word_frac is not None:
        g = sec["rho_by_word_frac"][str(k)]
        return float(np.interp(word_frac, [e["word_frac"] for e in g], [e["rho"] for e in g])) if g else None
    if remaining_ms is not None:
        g = sec["rho_by_remaining_ms"][str(k)]
        return float(np.interp(remaining_ms, [e["remaining_ms"] for e in g], [e["rho"] for e in g])) if g else None
    raise ValueError("give word_frac or remaining_ms")


def main(argv=None) -> int:
    from harness.v2_split import check_split, filter_split, load_jsonl

    ap = argparse.ArgumentParser(description="Partial-transcript retrieval risk rho(t)")
    ap.add_argument("--items", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--index-dir", default=None, help="v2 index dir (default: the v1 index)")
    ap.add_argument("--profile", default="faiss", choices=["faiss", "faiss_rerank"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--all-items-are-dev", action="store_true",
                    help="development data without a v2 split (e.g. the simulated v1 traces)")
    ap.add_argument("--include-not-needed", action="store_true",
                    help="also score items whose retrieval_needed_authored is false")
    ap.add_argument("--rewrite", default="identity", choices=["identity"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    items = load_jsonl(args.items)
    if not args.all_items_are_dev:
        check_split(args.split)
        items = filter_split(items, args.split)
    if not args.include_not_needed:
        items = [it for it in items if it.get("retrieval_needed_authored", True)]
    by_id = {it["id"]: it for it in items}
    traces = []
    for pat in args.traces:
        for p in sorted(glob.glob(pat)):
            traces.extend(load_jsonl(p))
    traces = [t for t in traces if t["item_id"] in by_id]

    from retrieval.profiles import make_retriever

    retriever = make_retriever(args.profile, index_dir=args.index_dir, device=args.device)
    rows = score_traces(by_id, traces, retriever, ks=tuple(args.k))
    simulated = any(t.get("simulated") for t in traces)
    artifact = {
        "schema_version": 1,
        "items": args.items,
        "traces": args.traces,
        "split": "all-dev (development data)" if args.all_items_are_dev else args.split,
        "n_items": len(by_id),
        "index_dir": args.index_dir,
        "profile": args.profile,
        "rewrite": args.rewrite,
        "ks": args.k,
        "simulated_traces": simulated,
        "definition": "rho = P(retrieval on the partial misses gold within k | full transcript hits within k)",
        "analysis": analyse(rows, tuple(args.k)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
    a = artifact["analysis"]["query_ready_only"]
    print(f"{len(rows)} partials over {artifact['analysis']['n_traces']} traces; query-ready subset {a['n']}")
    print("word_frac bin      n   " + "  ".join(f"hit@{k} full@{k} rho@{k}" for k in args.k))
    for b in a["by_word_frac"]:
        cells = "  ".join(f"{b[f'hit@{k}']:.2f}  {b[f'full_hit@{k}']:.2f}   {b[f'rho@{k}'] if b[f'rho@{k}'] is not None else float('nan'):.2f}" for k in args.k)
        print(f"{b['bin'][0]:.2f}-{b['bin'][1]:.2f}  {b['n']:5d}   {cells}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
