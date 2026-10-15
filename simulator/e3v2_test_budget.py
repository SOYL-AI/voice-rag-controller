"""
e3v2_test_budget.py — dry-run LLM-call and judge-call counts, and judge cost,
for the E3-v2 TEST split under the PREREG_V2 §2.2/§2.3/§4.8 rules, estimated
WITHOUT touching test content: every count is a dev per-item ratio scaled by
the number of test items (test ids are only counted, per CONTRACTS §7).

Rules applied:
  E1 test scope   self_assessed_need on every text prefix and ASR partial;
                  closed-book generation on the full question only (5 seeds),
                  generated at the pipeline's 256-token cap so the same
                  answers serve as E1's label draws and E3's text-condition
                  skip cells
  E3 Phase A      rewrites for every distinct dispatch text; LAZY: grounded
                  answers and skip answers only for cells some FROZEN
                  controller chooses (the dev ratio is taken from the dev
                  Phase B records of the frozen runs, not the tuning sweep)
  judges          d2_support only (d1_open is alias-graded): primary k = 1
                  (+ the measured re-draw rate), secondary on 30%

    python -m simulator.e3v2_test_budget --out docs/research_v2/v2_test_budget.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1_v2 import common as e1common  # noqa: E402

PA = _REPO_ROOT / "simulator" / "e3v2_phase_a"
SEEDS = 5
INR_PER_USD = 88.0
PRICE = {"primary": (2.0, 10.0), "secondary": (2.5, 10.0)}  # $ per 1M input / output tokens


def n_split(ds: str, split: str) -> int:
    return len(json.load(open(_REPO_ROOT / f"datasets/v2/{ds}/splits.json"))[split])


def lines(path: Path) -> int:
    return sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0


def frozen_cells(ds: str) -> dict:
    """Distinct cells the FROZEN controllers chose on dev (all conditions,
    profiles, prices, architectures): grounded (item, cond, variant) and skip
    (item, cond)."""
    ret, skip = set(), set()
    for f in glob.glob(str(PA / ds / "analysis" / "phase_b" / "*" / "*" / "*_seed11.json")):
        for r in json.load(open(f, encoding="utf-8")):
            if r["used"]:
                ret.add((r["item_id"], r["condition"], r["query_variant"]))
            else:
                skip.add((r["item_id"], r["condition"]))
    return {"retrieve": len(ret), "skip": len(skip)}


def judge_usage(ds: str) -> dict:
    out = {}
    for role, prefix in (("primary", "gpt6"), ("secondary", "gpt4o")):
        pin, pout, n = 0, 0, 0
        for d in glob.glob(str(PA / ds / "judge_cache" / f"{prefix}*")):
            for f in glob.glob(d + "/*.json")[:4000]:
                try:
                    u = (json.load(open(f, encoding="utf-8")).get("usage") or {})
                except Exception:  # noqa: BLE001
                    continue
                if u:
                    pin += u.get("prompt_tokens") or 0
                    pout += u.get("completion_tokens") or 0
                    n += 1
        out[role] = {"in": pin / max(1, n), "out": pout / max(1, n), "n_sampled": n}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "docs" / "research_v2" / "v2_test_budget.json")
    ap.add_argument("--redraw-rate", type=float, default=0.01, help="unparsed primary samples needing a re-draw")
    args = ap.parse_args(argv)

    usage = judge_usage("d2_support")
    cost_call = {role: (usage[role]["in"] * PRICE[role][0] + usage[role]["out"] * PRICE[role][1]) / 1e6
                 for role in PRICE}
    report = {"_comment": __doc__.split("\n\n")[0].strip(), "created_at_utc": e1common.now_utc(),
              "measured_tokens_per_judge_call": usage, "usd_per_judge_call": cost_call, "datasets": {}}
    tot = defaultdict(float)
    for ds in ("d1_open", "d2_support"):
        n_dev, n_test = n_split(ds, "dev"), n_split(ds, "test")
        scale = n_test / n_dev
        sa = lines(PA / ds / "sa" / "self_assessment_dev.jsonl")
        rewrites = lines(PA / ds / "rewrites_dev.jsonl")
        fc = frozen_cells(ds)
        conds = len({json.loads(l)["condition"] for l in open(PA / ds / "conditions_dev.jsonl", encoding="utf-8")})
        llm = {
            "e1_self_assessment": round(sa * scale),
            "e1_closed_book_full_question": n_test * SEEDS,
            "e3_rewrites": round(rewrites * scale),
            "e3_grounded_answers_lazy": round(fc["retrieve"] * scale * SEEDS),
            # text150 skip cells reuse the E1 closed-book answers (same prompt, 256 cap)
            "e3_skip_answers_lazy": round(fc["skip"] * scale * SEEDS * (conds - 1) / conds),
        }
        judged = ds == "d2_support"
        jcalls = {}
        if judged:
            answers = llm["e1_closed_book_full_question"] + llm["e3_grounded_answers_lazy"] + llm["e3_skip_answers_lazy"]
            jcalls = {"primary_k1": round(answers * (1 + args.redraw_rate)), "secondary_30pct": round(0.30 * answers)}
        cost = {role: jcalls.get(k, 0) * cost_call[role] for role, k in (("primary", "primary_k1"),
                                                                           ("secondary", "secondary_30pct"))}
        report["datasets"][ds] = {
            "n_dev": n_dev, "n_test": n_test, "dev_frozen_cells": fc, "llm_calls": llm,
            "llm_calls_total": sum(llm.values()), "judge_calls": jcalls,
            "judge_cost_usd": cost, "judge_cost_inr": {k: v * INR_PER_USD for k, v in cost.items()}}
        tot["llm_calls"] += sum(llm.values())
        tot["primary_inr"] += cost["primary"] * INR_PER_USD
        tot["secondary_inr"] += cost["secondary"] * INR_PER_USD
    report["totals"] = dict(tot)
    e1common.atomic_write_json(args.out, report)
    print(json.dumps({k: report[k] for k in ("usd_per_judge_call", "totals")}, indent=1))
    for ds, d in report["datasets"].items():
        print(ds, d["llm_calls"], d["judge_calls"], {k: round(v) for k, v in d["judge_cost_inr"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
