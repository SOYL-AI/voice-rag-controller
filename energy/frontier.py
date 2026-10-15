"""Accuracy-vs-energy operating points from the real E3 sweep.

E3 (simulator/e3_sweep_results/, 5 seeds x 50 scenarios, Mistral-7B on the
Modal A10G, judged answers) reports per decision: correct or not, and the
number of retrievals. Here every decision is re-priced in joules with
energy/model.py, using that decision's OWN token counts:

  prompt tokens  = the scenario's skip / retrieve prompt (token_budget.json)
  gen tokens     = the cached answer actually produced on that path for that
                   (scenario, seed) (simulator/e3_sweep_cache/answers/)
  tts chars      = that answer's length
  asr seconds    = the 3.94 s benchmark utterance (E3 ran the text path, so
                   ASR/VAD are a constant per turn and are included only so
                   the totals are whole-turn figures)
  profile        = faiss (E3 retrieved with bge-small + FAISS)

Output: energy/results/frontier.json with, per (baseline, price point),
mean accuracy, mean retrievals and mean energy per turn (central/low/high)
over seeds x scenarios.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.frontier
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from energy import model as em  # noqa: E402
from energy.tokens import mistral_tokenizer, text_tokens  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
ASR_S = 3.94


def main():
    tok = mistral_tokenizer()
    tb = json.loads((RES / "token_budget.json").read_text())
    prompt = {r["id"]: r for r in tb["per_scenario"]}
    answers = {}
    for f in (REPO / "simulator" / "e3_sweep_cache" / "answers").glob("*.json"):
        a = json.loads(f.read_text(encoding="utf-8"))
        answers[(a["scenario_id"], a["seed"], a["path"])] = (
            text_tokens(tok, a["answer_text"]), len(a["answer_text"].strip()))

    params = em.load_params()
    cache = {}

    def turn(sid, seed, path):
        key = (sid, seed, path)
        if key not in cache:
            n_gen, chars = answers[key]
            p = prompt[sid]
            e = em.turn_energy_j(path, "faiss" if path == "retrieve" else None,
                                 p["prompt_retrieve" if path == "retrieve" else "prompt_skip"],
                                 n_gen, ASR_S, chars, params=params)
            cache[key] = (e["total_j"], e["remote_j"])
        return cache[key]

    groups = defaultdict(list)
    for f in sorted((REPO / "simulator" / "e3_sweep_results").glob("e3_*_seed*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            path = "retrieve" if r["outcome"]["retrievals"] > 0 else "skip"
            pr = r.get("prices") or {}
            key = (r["baseline"], round(pr.get("lambda_l", -1), 6), round(pr.get("lambda_c", -1), 6))
            tot, rem = turn(r["scenario_id"], r["seed"], path)
            groups[key].append((r["outcome"]["answer_correct"], r["outcome"]["retrievals"],
                                tot["central"], tot["low"], tot["high"], rem["central"]))
    pts = []
    for (b, ll, lc), rows in sorted(groups.items()):
        a = np.array(rows, dtype=float)
        pts.append({"baseline": b, "lambda_l": None if ll < 0 else ll,
                    "lambda_c": None if lc < 0 else lc, "n": len(rows),
                    "accuracy": float(a[:, 0].mean()), "retrieval_rate": float(a[:, 1].mean()),
                    "energy_j": {"central": float(a[:, 2].mean()), "low": float(a[:, 3].mean()),
                                 "high": float(a[:, 4].mean())},
                    "remote_llm_j_central": float(a[:, 5].mean())})
    always = [turn(s, d, "retrieve") for (s, d, p) in answers if p == "retrieve"]
    never = [turn(s, d, "skip") for (s, d, p) in answers if p == "skip"]
    out = {"asr_seconds": ASR_S, "profile": "faiss", "params_version": params.get("version"),
           "mean_turn_j_always_retrieve": float(np.mean([t[0]["central"] for t in always])),
           "mean_turn_j_never_retrieve": float(np.mean([t[0]["central"] for t in never])),
           "points": pts}
    (RES / "frontier.json").write_text(json.dumps(out, indent=2))
    print(f"{len(pts)} operating points; always-retrieve {out['mean_turn_j_always_retrieve']:.0f} J/turn,"
          f" never {out['mean_turn_j_never_retrieve']:.0f} J/turn")


if __name__ == "__main__":
    main()
