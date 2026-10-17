"""
v2_waste_sensitivity_d1.py -- EXPLORATORY sensitivity of Delta[COMMIT]
(Ours-v2[COMMIT] - B9+) on d1_open TEST to the waste coefficient of the
frozen v2 objective. Not a pre-registered analysis; nothing here changes
any confirmatory result.

    PYTHONUTF8=1 VOICE_RAG_MODEL=mistral7b_v03 python -m analysis.v2_waste_sensitivity_d1 \
        --phase-a <repo with the Phase A tree>/simulator/e3v2_phase_a

The frozen objective prices an early commit at lambda_c (1 + w(f)),
w(f) = coef * (1 - f), coef = 0.1 (controller/baselines/_objective_v2.py
waste_at; configs/config.yaml controller.e3v2.waste.coef; the frozen bundles
do not override it). This script re-runs the FROZEN evaluator
(simulator/e3v2_lazy.frozen_records, the code path that produced
contrasts_test.json) for ours_v2 and b9plus_v2 with only the waste
coefficient changed, coef in {0, 0.04, 0.08, 0.16} plus the frozen 0.1:

  * bundle["overrides"]["waste"] = {"coef": c};
  * the bundle's estimates artifact is copied with every policy-constant
    "waste" scaled by c / 0.1 (waste_at is linear in coef, so this is the
    dev mean B9's timing sub-optimizer would have estimated at c);
  * everything else (prices, calibration, belief transitions, tuned B9+
    timing rules, remaining-speech predictor) stays frozen. The tuned rules
    were selected on dev at coef 0.1 and are NOT re-tuned.

Outcomes are looked up in the existing Phase A test store; nothing is
generated or graded. A decision that lands on a cell the lazy plan never
generated is detected (not scored with the placeholder): such items are
dropped from that cell's contrast and counted, so a coef that needs new
answers is reported only on the subset of items that need none.

GATE: at coef 0.1 the per-item Delta[COMMIT] must reproduce
contrasts_test.json exactly (every cell, every item), else the script stops
without writing results.

Statistics: the paired item bootstrap of analysis/v2_test_d1.py (B = 10,000,
seed 20261017, the same item universe and resamples).

Writes docs/research_v2/waste_sensitivity_v2_d1.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import load_e3v2_config  # noqa: E402
from llm.experiment1_v2 import common as e1common  # noqa: E402
from simulator.e3v2_decide import CellStore, _item_utilities, _paired, _price_key  # noqa: E402
from simulator.e3v2_lazy import frozen_records  # noqa: E402

DS = "d1_open"
SPLIT = "test"
FROZEN_COEF = 0.1
COEFS = (0.0, 0.04, 0.08, FROZEN_COEF, 0.16)
CONTROLLERS = ("ours_v2", "b9plus_v2")
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "waste_sensitivity_v2_d1.json"
B = 10_000
SEED = 20261017  # analysis/v2_test_d1.py
PRIMARY = {"condition": "piper-en_US-hfc_male-medium__150wpm__clean", "profile": "rewrite_faiss", "quantile": 0.65}


class MissingAwareStore(CellStore):
    """The frozen CellStore, except that a cell Phase A never generated is
    recorded and answered with the placeholder instead of raising, so the
    affected items can be excluded (never scored)."""

    def __init__(self, records, order_sensitive=True, recording=None):
        super().__init__(records, order_sensitive, None)
        self.missing = set()

    def cell(self, item_id, seed, condition, path, variant):
        key = (item_id, int(seed), condition, path, variant)
        if key not in self.cells:
            self.missing.add(key)
            return dict(self.PLACEHOLDER)
        return self.cells[key]


def scaled_estimates(src: Path, factor: float, dest: Path) -> None:
    est = e1common.load_json(src)
    pc = est.get("policy_constants") or {}
    for prof in (pc.get("profiles") or {}).values():
        for row in prof.values():
            if isinstance(row, dict) and "waste" in row:
                row["waste"] = float(row["waste"]) * factor
    e1common.atomic_write_json(dest, est)


def bundle_at(bundle: dict, coef: float, tmp: Path) -> dict:
    b = copy.deepcopy(bundle)
    if "waste" in (b.get("overrides") or {}):
        raise SystemExit("frozen bundle overrides waste; this script assumes it does not")
    b["overrides"]["waste"] = {"coef": float(coef)}
    est = b["overrides"].get("estimates_artifact")
    if est:
        dest = tmp / f"estimates_{b['condition'].replace('/', '_')}_coef{coef:g}.json"
        scaled_estimates(Path(est), coef / FROZEN_COEF, dest)
        b["overrides"]["estimates_artifact"] = str(dest)
    return b


def commit_deltas(records, missing):
    """{(profile, price): {"delta": {item: d}, "dropped_items": n, "missing_cells": n}} for commit."""
    commit = [r for r in records if r["architecture"] == "commit" and r.get("prices") is not None]
    bad = {}
    for r in commit:
        path = "skip" if r["query_variant"] == "none" else "retrieve"
        key = (r["item_id"], int(r["seed"]), r["condition"], path, r["query_variant"])
        if key in missing:
            bad.setdefault((r["latency_profile"], _price_key(r["prices"])), set()).add(r["item_id"])
    utils = _item_utilities(commit)
    out = {}
    for (arch, prof, pk), bs in utils.items():
        d = _paired(bs["ours_v2"], bs["b9plus_v2"])
        drop = bad.get((prof, pk), set())
        out[(prof, pk)] = {"delta": {i: v for i, v in d.items() if i not in drop},
                           "dropped_items": len(drop & set(d))}
    return out


def decisions(records):
    """(controller, profile, price, item) -> (retrieve, dispatch_partial_index) under commit, seed-invariant."""
    out = {}
    for r in records:
        if r["architecture"] == "commit" and r.get("prices") is not None and r["seed"] == 11:
            out[(r["baseline"], r["latency_profile"], _price_key(r["prices"]), r["item_id"])] = (
                bool(r["decision"]["retrieve"]), r["decision"]["dispatch_partial_index"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--phase-a", required=True, help="the simulator/e3v2_phase_a directory holding d1_open/")
    ap.add_argument("--e1-results", default=str(_REPO_ROOT / "llm" / "experiment1_v2" / "results"))
    ap.add_argument("--coefs", default=",".join(f"{c:g}" for c in COEFS))
    args = ap.parse_args(argv)

    cfg = load_e3v2_config()
    assert float(cfg["waste"]["coef"]) == FROZEN_COEF, "config waste.coef is not the frozen 0.1"
    phase_a = Path(args.phase_a) / DS
    root = Path(args.phase_a).resolve().parent.parent  # bundle paths in contrasts are relative to it
    doc_c = e1common.load_json(phase_a / f"contrasts_{SPLIT}.json")
    frozen = {cond: {(r["latency_profile"], r["price"]): r["per_item"]["delta_commit"] for r in rows}
              for cond, rows in doc_c["contrasts"].items()}
    bundles = []
    for bp in doc_c["bundles"]:
        p = Path(bp)
        bundles.append(e1common.load_json(p if p.is_absolute() else root / p))

    print("loading the Phase A test store ...", flush=True)
    store = MissingAwareStore.load(phase_a / f"store_{SPLIT}.jsonl")
    coefs = [float(c) for c in args.coefs.split(",")]
    if FROZEN_COEF not in coefs:
        coefs.append(FROZEN_COEF)
    coefs = sorted(set(coefs), key=lambda c: (c != FROZEN_COEF, c))  # the gate runs first

    results, dec = {}, {}
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        for coef in coefs:
            for bundle in bundles:
                cond = bundle["condition"]
                store.missing = set()
                recs = frozen_records(bundle_at(bundle, coef, tmp), DS, SPLIT, phase_a, Path(args.e1_results),
                                      store, CONTROLLERS)
                results[(coef, cond)] = commit_deltas(recs, store.missing)
                results[(coef, cond)]["_missing_cells"] = len({k for k in store.missing})
                dec[(coef, cond)] = decisions(recs)
                del recs
                print(f"coef={coef:g} {cond}: missing cells {results[(coef, cond)]['_missing_cells']}", flush=True)
            if coef == FROZEN_COEF:
                for bundle in bundles:
                    cond = bundle["condition"]
                    for key, ref in frozen[cond].items():
                        got = results[(coef, cond)][key]["delta"]
                        if set(got) != set(ref) or any(abs(got[i] - ref[i]) > 1e-12 for i in ref):
                            raise SystemExit(f"GATE FAILED: coef 0.1 does not reproduce contrasts_test.json at "
                                             f"{cond} {key}; no results written")
                print("gate passed: coef 0.1 reproduces contrasts_test.json exactly", flush=True)

    # bootstrap, as analysis/v2_test_d1.py
    universe = sorted({i for rows in doc_c["contrasts"].values() for r in rows
                       for k in ("delta_commit", "delta_speculative") for i in r["per_item"][k]})
    uidx = {i: k for k, i in enumerate(universe)}
    rng = np.random.default_rng(SEED)
    counts = np.zeros((B, len(universe)), dtype=np.float64)
    for b in range(B):
        counts[b] = np.bincount(rng.integers(0, len(universe), size=len(universe)), minlength=len(universe))

    def boot(values):
        x = np.zeros(len(universe))
        m = np.zeros(len(universe))
        for i, v in values.items():
            x[uidx[i]] = v
            m[uidx[i]] = 1.0
        tot = counts @ m
        means = (counts @ x) / np.where(tot > 0, tot, np.nan)
        q = np.nanpercentile(means, [2.5, 97.5])
        return float(np.mean(list(values.values()))), [float(q[0]), float(q[1])]

    cells = []
    for bundle in bundles:
        cond = bundle["condition"]
        for prof, grid in bundle["price_grid"].items():
            for g in grid:
                if not g["feasible"]:
                    continue
                pk = f"{g['lambda_l']:g}|{g['lambda_c']:g}"
                for coef in coefs:
                    r = results[(coef, cond)][(prof, pk)]
                    est, ci = boot(r["delta"])
                    changed = {c: sum(1 for (cc, p, k, i), v in dec[(coef, cond)].items()
                                      if cc == c and p == prof and k == pk
                                      and dec[(FROZEN_COEF, cond)][(cc, p, k, i)] != v)
                               for c in CONTROLLERS}
                    cells.append({"condition": cond, "profile": prof, "belief_quantile": g["quantile"], "price": pk,
                                  "waste_coef": coef, "delta_commit": est, "ci95": ci,
                                  "n_items": len(r["delta"]), "items_dropped_missing_cells": r["dropped_items"],
                                  "decisions_changed_vs_frozen": changed})
    out = {"_comment": "EXPLORATORY sensitivity (not pre-registered). Frozen evaluator re-run with only "
                       "controller.e3v2.waste.coef changed (policy-constant waste scaled accordingly); outcomes "
                       "from the existing Phase A test store, nothing generated. Items whose decision lands on an "
                       "ungenerated cell are dropped from that cell and counted.",
           "generated_by": "python -m analysis.v2_waste_sensitivity_d1", "generated_at_utc": e1common.now_utc(),
           "git_commit": e1common.git_commit(), "dataset": DS, "split": SPLIT, "frozen_coef": FROZEN_COEF,
           "gate": "coef 0.1 reproduced contrasts_test.json per-item Delta[COMMIT] exactly",
           "inputs": {"contrasts_sha256": e1common.sha256_file(phase_a / f"contrasts_{SPLIT}.json"),
                      "store_sha256": e1common.sha256_file(phase_a / f"store_{SPLIT}.jsonl")},
           "bootstrap": {"B": B, "seed": SEED, "n_items_universe": len(universe)},
           "missing_cells": {f"{c:g}|{cond}": results[(c, cond)]["_missing_cells"] for (c, cond) in results},
           "primary_cell": PRIMARY, "cells": cells}
    e1common.atomic_write_json(OUT_JSON, out)
    for c in cells:
        if c["condition"] == PRIMARY["condition"] and c["profile"] == PRIMARY["profile"] \
                and c["belief_quantile"] == PRIMARY["quantile"]:
            print(c)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
