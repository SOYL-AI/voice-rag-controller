"""
compose_profiles.py — composite retrieval-operation latency profiles.

What a voice RAG system does on a live partial before it can prompt the LLM
is the whole retrieval OPERATION, not the vector call:

    rewrite_faiss         LLM query rewrite + bge-small/FAISS search
    rewrite_faiss_rerank  LLM query rewrite + FAISS top-50 + cross-encoder
    rewrite_remote        LLM query rewrite + remote retrieval service
                          (loopback until a WAN measurement exists; then the
                          remote_api_wan samples are used automatically)

Each composite's samples are independent Monte Carlo sums of measured
samples: the rewrite at one load level (retrieval/measure_rewrite_latency.py)
plus the bare profile (retrieval/latency_profiles_v2.json). The bare
profiles stay in the artifact, labelled lower_bound=true: they exclude the
rewrite, so they are lower bounds on the hideable latency of a live-partial
dispatch.

    python -m retrieval.compose_profiles --rewrite analysis/rewrite_latency_v2/rewrite_latency.json \\
        --profiles retrieval/latency_profiles_v2.json --level 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

QS = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
BARE = ("faiss", "faiss_rerank", "remote_api")
COMPOSITES = {"rewrite_faiss": "faiss", "rewrite_faiss_rerank": "faiss_rerank", "rewrite_remote": "remote_api"}


def _q(a) -> dict:
    a = np.asarray(a, float)
    return {k: round(float(np.quantile(a, q)), 2) for k, q in QS.items()}


def compose(profiles: dict, rewrite: dict, level: str, n: int = 4000, seed: int = 0,
            extra_levels=()) -> dict:
    rng = np.random.default_rng(seed)
    levels = rewrite["by_concurrency"]
    if level not in levels:
        raise KeyError(f"rewrite level {level!r} not measured; have {sorted(levels)}")
    canon = profiles.setdefault("canonical", {})
    profs = profiles["profiles"]
    for bare in BARE:
        name = canon.get(bare, bare)
        if name in profs:
            profs[name]["lower_bound"] = True
            profs[name].setdefault("notes", "")
            if "excludes the LLM query rewrite" not in profs[name]["notes"]:
                profs[name]["notes"] = (profs[name]["notes"] + " LOWER BOUND: vector call only; excludes "
                                        "the LLM query rewrite a live-partial dispatch runs first.").strip()

    for lv in (level, *extra_levels):
        rw = np.asarray(levels[lv]["samples_ms"], float)
        for comp, bare in COMPOSITES.items():
            src = canon.get(bare, bare)
            if src not in profs:
                continue
            ret = np.asarray(profs[src]["samples_ms"], float)
            s = rng.choice(rw, n) + rng.choice(ret, n)
            name = comp if lv == level else f"{comp}_c{lv}"
            profs[name] = {
                "n": int(n),
                "quantiles_ms": _q(s),
                "mean_ms": round(float(s.mean()), 2),
                "std_ms": round(float(s.std()), 2),
                "samples_ms": [round(float(x), 2) for x in s[:2000]],
                "components_ms": {"rewrite": _q(rw), "retrieval": _q(ret)},
                "composite": {"rewrite_concurrency": int(lv), "retrieval_profile": src,
                              "method": "independent Monte Carlo sum of measured samples"},
                "network": profs[src].get("network"),
                "wan_rtt_ms": profs[src].get("wan_rtt_ms"),
                "lower_bound": False,
                "notes": (f"LLM query rewrite (retrieval/rewrite.py prompt, {rewrite.get('model')}, "
                          f"{lv} concurrent requests) + {src}. Hideable latency of a live-partial "
                          "dispatch with a rewrite."),
            }
            if lv == level:
                canon[comp] = comp
    profiles["composite_note"] = (
        f"rewrite_* profiles = rewrite at concurrency {level} + the canonical bare profile; variants "
        f"*_c<N> use other load levels. Bare profiles are lower bounds (no rewrite).")
    return profiles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rewrite", default="analysis/rewrite_latency_v2/rewrite_latency.json")
    ap.add_argument("--profiles", default="retrieval/latency_profiles_v2.json")
    ap.add_argument("--level", default="8", help="canonical rewrite concurrency level")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args(argv)
    prof = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    rw = json.loads(Path(args.rewrite).read_text(encoding="utf-8"))
    others = tuple(lv for lv in rw["by_concurrency"] if lv != args.level)
    prof = compose(prof, rw, args.level, n=args.n, extra_levels=others)
    Path(args.profiles).write_text(json.dumps(prof, indent=1), encoding="utf-8")
    for name in sorted(prof["profiles"]):
        if name.startswith("rewrite_"):
            q = prof["profiles"][name]["quantiles_ms"]
            print(f"{name:28s} p10={q['p10']:7.0f} p50={q['p50']:7.0f} p90={q['p90']:7.0f} p95={q['p95']:7.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
