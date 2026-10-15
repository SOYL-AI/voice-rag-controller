"""
e3v2_lazy.py — LAZY OUTCOME EVALUATION for E3-v2 (PREREG_V2 §4.8).

Controller decisions depend on beliefs, the timing state and the latency
model, never on answer correctness. So the evaluation pipeline is run FIRST
against a recording store (simulator/e3v2_decide.py:CellStore(recording=...)),
which notes every (item, condition, path, query_variant) cell some controller
lands on (all seeds) and answers with a placeholder. That list is written to
disk BEFORE any generation or grading; Phase A then generates and grades
only those cells (simulator/e3v2_phase_a.py --required-cells), and the same
pipeline runs again on the real cells. The result equals grading every cell
(tests/test_v2_lazy.py checks this on a synthetic store).

    # dev: the plan IS the dev analysis in recording mode, so tuning
    # candidates are included; the g+/g- cells it reads become real after
    # the first round, so run plan -> phase_a -> plan again until the second
    # plan adds no cells
    python -m simulator.e3v2_lazy plan --dataset d2_support --split dev

    # test (refused until docs/research_v2/PREREG_FROZEN exists): a frozen
    # bundle per condition (written by the dev analysis) fixes every model,
    # price and tuned parameter; nothing is fitted on test
    python -m simulator.e3v2_lazy plan --dataset d2_support --split test --bundle <bundle.json> ...
    python -m simulator.e3v2_lazy evaluate --dataset d2_support --split test --bundle <bundle.json> \\
        --d-star-speculative <name> --m3 <value>

The test path is the default way to run E3-v2 on test: `plan` is required
before Phase A grades anything on test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import load_e3v2_config  # noqa: E402
from controller.remaining_speech import RemainingSpeechPredictor  # noqa: E402
from controller.sequential_v2 import PRICED_V2, TurnCtx  # noqa: E402
from llm.experiment1_v2 import common as e1common  # noqa: E402
from simulator.e3v2_decide import (  # noqa: E402
    CellStore,
    boundary_contrasts,
    load_latency_model,
    run_architectures,
)

PREREG_FROZEN = _REPO_ROOT / "docs" / "research_v2" / "PREREG_FROZEN"
DEFAULT_PHASE_A = _REPO_ROOT / "simulator" / "e3v2_phase_a"


def check_split(split: str) -> None:
    if split not in ("dev", "test"):
        raise SystemExit("--split must be dev or test")
    if split == "test" and not PREREG_FROZEN.exists():
        raise SystemExit("Refusing --split test: sealed until docs/research_v2/PREREG_FROZEN exists.")


def write_plan(path: Path, cells: set, meta: dict) -> dict:
    rows = sorted([list(c) for c in cells], key=lambda c: tuple(str(x) for x in c))
    by_path = Counter(c[2] for c in rows)
    doc = {"_comment": "LAZY OUTCOME EVALUATION plan (PREREG_V2 §4.8): only these (item, condition, path, "
                       "query_variant) cells, all seeds, are generated and graded. Written before any grading.",
           "created_at_utc": e1common.now_utc(), "git_commit": e1common.git_commit(), **meta,
           "n_cells": len(rows), "by_path": dict(by_path), "cells": rows}
    e1common.atomic_write_json(path, doc)
    return doc


# ---------------------------------------------------------------------------
# Frozen evaluation (test path; also usable on dev to reproduce a bundle)
# ---------------------------------------------------------------------------
def frozen_records(bundle: dict, ds: str, split: str, phase_a: Path, e1_results: Path,
                   store: CellStore, controllers: Sequence[str]) -> List[dict]:
    """Every controller x both architectures x every bundle profile/price
    with the bundle's tuned parameters, on `split`. Nothing is fitted here."""
    from simulator.e3v2_dev_analysis import SEEDS, build_signals, load_inputs

    inp = load_inputs(ds, e1_results, phase_a, split)
    inp["sa_lag_ms"] = float(bundle["sa_lag_ms"])  # the frozen dev estimate
    cond = bundle["condition"]
    if cond not in inp["traces"]:
        raise SystemExit(f"Phase A has no {cond} traces for {ds}/{split}")
    pred = RemainingSpeechPredictor.load(bundle["remaining_speech_predictor"])
    rhat = {}
    for item_id, tr in inp["traces"][cond].items():
        rhat[item_id] = []
        for p in tr["partials"]:
            q = pred.predict(p["text"], float(p["t_ms"]))
            rhat[item_id].append({"t_ms": float(p["t_ms"]), "available_ms": float(p.get("available_ms", p["t_ms"])),
                                  "remaining_ms": float(q["q50"])})
    rows = build_signals(inp, cond, rhat)
    cfg = load_e3v2_config(overrides=bundle["overrides"])
    turns = [TurnCtx(item_id=r["item_id"], signals=r["signals"], condition=cond,
                     transcript_final=r["transcript_final"]) for r in rows]
    labels = {i: bool(v) for i, v in inp["labels"].items() if v is not None}
    gold = {i: list((inp["items"].get(i) or {}).get("gold_doc_ids") or []) for i in inp["traces"][cond]}
    names = [n for n in controllers if n != "oracle_v2" or labels]
    lm = load_latency_model(cfg)
    records: List[dict] = []
    for prof, grid in bundle["price_grid"].items():
        unpriced = [n for n in names if n not in PRICED_V2]
        records += run_architectures(turns, store, unpriced, [], SEEDS, [prof], cfg, lm,
                                     ("commit", "speculative"), split, labels, gold)
        tuned = bundle["tuned"].get(prof) or {}
        for row in grid:
            if not row["feasible"]:
                continue
            pr = {"lambda_l": row["lambda_l"], "lambda_c": row["lambda_c"]}
            t = tuned.get(f"{row['lambda_l']:g}") or {}
            params = {}
            if t.get("b9plus_timing_rule"):
                params["b9plus_v2"] = {"timing_rule": t["b9plus_timing_rule"]}
            if t.get("spec_trigger"):
                params["b10_speculative_v2"] = {"timing": t["spec_trigger"]}
                params["b11_speculative_gated_v2"] = {"timing": t["spec_trigger"]}
            records += run_architectures(turns, store, [n for n in names if n in PRICED_V2], [pr], SEEDS, [prof],
                                         cfg, lm, ("commit", "speculative"), split, labels, gold,
                                         controller_params=params)
    return records


ALL = ("ours_v2", "b9_v2", "b9plus_v2", "b10_speculative_v2", "b11_speculative_gated_v2", "b0_v2",
       "b1_v2_sentence_end", "b1_v2_immediate", "b1_v2_fixed_early_proxy", "b1_v2_forecast",
       "b4_v2", "b5_v2", "b6_v2", "b8_v2", "oracle_v2")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Lazy outcome evaluation for E3-v2 (PREREG_V2 §4.8).")
    ap.add_argument("command", choices=("plan", "evaluate"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--bundle", action="append", default=[], help="frozen bundle(s); required on test")
    ap.add_argument("--phase-a", default=str(DEFAULT_PHASE_A))
    ap.add_argument("--e1-results", default=str(_REPO_ROOT / "llm" / "experiment1_v2" / "results"))
    ap.add_argument("--d-star-speculative", default=None)
    ap.add_argument("--m3", type=float, default=None)
    ap.add_argument("--b9plus-extended", action="store_true", help="plan for the diagnostic B9+ family too")
    args = ap.parse_args(argv)
    check_split(args.split)
    phase_a = Path(args.phase_a) / args.dataset
    out_path = phase_a / (f"required_{args.split}" + ("_b9x" if args.b9plus_extended else "") + ".json")

    if args.command == "plan" and args.split == "dev" and not args.bundle:
        from simulator.e3v2_dev_analysis import run_dataset
        record: set = set()
        run_dataset(argparse.Namespace(phase_a=args.phase_a, e1_results=args.e1_results, primary_condition=None,
                                       g_plus=None, g_minus=None, b9plus_extended=args.b9plus_extended),
                    args.dataset, record=record)
        doc = write_plan(out_path, record, {"dataset": args.dataset, "split": "dev",
                                            "source": "dev analysis in recording mode (tuning candidates included)"})
        print(f"wrote {out_path}: {doc['n_cells']} cells {doc['by_path']}")
        return 0

    if not args.bundle:
        raise SystemExit("--bundle is required (the frozen dev bundle per condition)")
    bundles = [e1common.load_json(Path(b)) for b in args.bundle]
    if args.command == "plan":
        record = set()
        store = CellStore.load(phase_a / f"store_{args.split}.jsonl", recording=record)
        for b in bundles:
            frozen_records(b, args.dataset, args.split, phase_a, Path(args.e1_results), store, ALL)
        doc = write_plan(out_path, record, {"dataset": args.dataset, "split": args.split,
                                            "bundles": args.bundle, "source": "frozen bundles"})
        print(f"wrote {out_path}: {doc['n_cells']} cells {doc['by_path']}")
        return 0

    store = CellStore.load(phase_a / f"store_{args.split}.jsonl")
    out = {}
    for b, path in zip(bundles, args.bundle):
        recs = frozen_records(b, args.dataset, args.split, phase_a, Path(args.e1_results), store, ALL)
        out[b["condition"]] = boundary_contrasts(recs, args.split, args.d_star_speculative, args.m3)
    dest = phase_a / f"contrasts_{args.split}.json"
    e1common.atomic_write_json(dest, {"bundles": args.bundle, "contrasts": out})
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
