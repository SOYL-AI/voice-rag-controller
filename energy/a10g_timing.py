"""Observed LLM timing on the production endpoint (Mistral-7B, vLLM, Modal A10G).

The E2 sweep (750 runs, simulator/e2_results/RUN_MANIFEST.md) and the E3
sweep (1500 runs, skip and retrieve) logged, for
every turn, when the LLM request could start, the first streamed token, and
the final token with its token count. From those logs this script extracts:

* decode time per generated token = (t_final - t_first_token) / (n_tokens - 1)
  (streamed over the WAN, so it is an upper bound on server decode time;
  inter-token arrival is dominated by server decode for a continuous stream);
* time to first token for skip vs retrieve turns. The WAN round trip and
  client overhead are common to both, so the difference estimates the
  server-side prefill time of the ~455 extra passage tokens.

These timings, not a roofline guess, are what energy/llm_estimate.py
multiplies by an A10G power estimate.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.a10g_timing --logs <repo>/logs
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from energy.protocol import mean_ci

OUT = Path(__file__).resolve().parent / "results" / "a10g_timing.json"


def parse_log(path):
    ev = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    turns = {}
    for e in ev:
        turns.setdefault(e.get("turn_id"), []).append(e)
    out = []
    for tid, es in turns.items():
        by = {}
        for e in es:
            by.setdefault(e["event_type"], e)
        if not {"llm_first_token", "llm_final", "controller_action", "user_end"} <= by.keys():
            continue
        action = by["controller_action"]["payload"].get("action")
        action = "skip" if action and action.startswith("skip") else action
        start = by["user_end"]["t_ms"]
        if action == "retrieve":
            if "retrieval_result" not in by:
                continue
            start = max(start, by["retrieval_result"]["t_ms"])
        else:
            start = max(start, by["controller_action"]["t_ms"])
        n = by["llm_final"]["payload"].get("n_tokens")
        out.append({"file": Path(path).name, "turn": tid, "action": action,
                    "ttft_ms": by["llm_first_token"]["t_ms"] - start,
                    "gen_ms": by["llm_final"]["t_ms"] - by["llm_first_token"]["t_ms"],
                    "n_tokens": n})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="directory holding E2 conv_*.jsonl")
    args = ap.parse_args()
    rows = []
    files = (sorted(glob.glob(str(Path(args.logs) / "conv_*.jsonl")))
             + sorted(glob.glob(str(Path(args.logs) / "e3full_*.jsonl"))))
    for f in files:
        try:
            rows.extend(parse_log(f))
        except Exception:
            continue
    rows = [r for r in rows if r["n_tokens"] and r["n_tokens"] > 5 and r["gen_ms"] > 0]
    per_tok = [r["gen_ms"] / (r["n_tokens"] - 1) for r in rows]
    ttft = {a: [r["ttft_ms"] for r in rows if r["action"] == a] for a in ("retrieve", "skip")}
    med = {a: float(np.median(v)) if v else None for a, v in ttft.items()}
    out = {
        "source": "E2 (conv_*) + E3 (e3full_*) logs, Modal vLLM Mistral-7B-Instruct-v0.3 on A10G, Aug 2026; clock res ~15.6 ms",
        "n_turns": len(rows),
        "n_files": len({r["file"] for r in rows}),
        "decode_ms_per_token": mean_ci(per_tok),
        "decode_ms_per_token_median": float(np.median(per_tok)),
        "decode_ms_per_token_p10_p90": [float(np.percentile(per_tok, 10)), float(np.percentile(per_tok, 90))],
        "ttft_ms": {a: {**mean_ci(v), "median": med[a]} for a, v in ttft.items()},
        "ttft_median_diff_retrieve_minus_skip_ms": (med["retrieve"] - med["skip"]
                                                   if med["retrieve"] and med["skip"] else None),
        "n_tokens": {a: mean_ci([r["n_tokens"] for r in rows if r["action"] == a])
                     for a in ("retrieve", "skip")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
