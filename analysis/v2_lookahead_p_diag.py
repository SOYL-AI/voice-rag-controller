"""
v2_lookahead_p_diag.py -- DIAGNOSTIC (reported, not used for any choice):
the distribution of Ours-v2's lookahead probability p (the next partial
carries new information) at every partial step and at each turn's LAST
partial, on dev, per dataset x condition, using the dev analysis outputs
(signals and frozen-bundle config).

    PYTHONUTF8=1 python -m analysis.v2_lookahead_p_diag

Writes docs/research_v2/v2_lookahead_p_diagnostic.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    RemainingDist,
    load_e3v2_config,
    next_partial_branch,
    partial_spacing_ms,
)
from controller.sequential_v2 import TurnCtx, build_steps  # noqa: E402
from llm.experiment1_v2 import common as e1common  # noqa: E402

PHASE_A = _REPO_ROOT / "simulator" / "e3v2_phase_a"
OUT = _REPO_ROOT / "docs" / "research_v2" / "v2_lookahead_p_diagnostic.json"


def summary(x) -> dict:
    x = np.asarray(x, dtype=float)
    if not x.size:
        return {"n": 0}
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(x.size), "mean": float(x.mean()), "p10": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4]),
            "share_below_0.5": float(np.mean(x < 0.5)), "share_below_0.1": float(np.mean(x < 0.1)),
            "share_above_0.99": float(np.mean(x > 0.99))}


def main() -> int:
    from simulator.e3v2_dev_analysis import slug
    cells = {}
    for ds in ("d1_open", "d2_support"):
        res = e1common.load_json(PHASE_A / ds / "analysis" / "dev_analysis.json")
        for cond in res["conditions"]:
            sl = slug(cond)
            bundle = e1common.load_json(PHASE_A / ds / "analysis" / f"bundle_{sl}.json")
            cfg = load_e3v2_config(overrides=bundle["overrides"])
            all_p, last_p, post_p, n_post = [], [], [], 0
            for r in e1common.read_jsonl(PHASE_A / ds / "analysis" / f"signals_{sl}_dev.jsonl"):
                ctx = TurnCtx(item_id=r["item_id"], signals=r["signals"], condition=r["condition"])
                se = float(r["signals"]["timing"]["speech_end_ms"])
                pl = None
                for o in build_steps(ctx, cfg):
                    if o.is_final or o.remaining_ms is None:
                        continue
                    d = RemainingDist(*o.remaining_quantiles) if o.remaining_quantiles \
                        else RemainingDist.point(o.remaining_ms)
                    p, _ = next_partial_branch(d, partial_spacing_ms(o.partial_interval_ms, cfg), cfg,
                                               o.scheduled_partials_left)
                    all_p.append(p)
                    pl = p
                    if o.partial_index is not None and \
                            float(r["signals"]["timing"]["partials"][o.partial_index]["t_ms"]) > se:
                        post_p.append(p)
                        n_post += 1
                if pl is not None:
                    last_p.append(pl)
            cells[f"{ds}|{cond}"] = {"condition_role": "primary" if cond == res["primary_condition"] else "secondary",
                                     "partial_spacing_ms": cfg["timing"].get("partial_spacing_ms"),
                                     "end_gap_ms": cfg["timing"].get("end_gap_ms"),
                                     "p_at_last_partial": summary(last_p), "p_at_every_partial": summary(all_p),
                                     "p_at_post_speech_partials": summary(post_p)}
    doc = {"_comment": "DIAGNOSTIC, not used for any choice: Ours-v2's lookahead p(next partial carries new "
                       "information) on dev, after the 2026-10-16 implementation correction.",
           "generated_by": "python -m analysis.v2_lookahead_p_diag", "generated_at_utc": e1common.now_utc(),
           "split": "dev", "cells": cells}
    e1common.atomic_write_json(OUT, doc)
    for k, v in cells.items():
        lp, pp = v["p_at_last_partial"], v["p_at_post_speech_partials"]
        print(f"{k}: last partial p mean {lp['mean']:.3f} median {lp['median']:.3f} "
              f"<0.5 {lp['share_below_0.5']:.3f} >0.99 {lp['share_above_0.99']:.3f}; "
              f"post-speech partials n={pp['n']}" + (f" p mean {pp['mean']:.3f}" if pp["n"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
