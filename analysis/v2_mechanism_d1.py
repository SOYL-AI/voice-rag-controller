"""
v2_mechanism_d1.py -- EXPLORATORY, POST HOC mechanism analysis: why does
Ours-v2[COMMIT] (the joint controller, "Ours-S" in the paper) lose to B9+
(the tuned decomposed rival) on d1_open TEST in the primary cell?

    PYTHONUTF8=1 VOICE_RAG_MODEL=mistral7b_v03 python -m analysis.v2_mechanism_d1 \
        --phase-a <repo with the Phase A tree>/simulator/e3v2_phase_a

NOT pre-registered. Nothing here changes any confirmatory result. Every
quantity is an association within the frozen test run, not a causal effect.
No model, ASR, retrieval or judge call is made; nothing is generated or graded.

PIPELINE
 1. Re-runs the FROZEN evaluator (simulator/e3v2_lazy.frozen_records, the code
    path that wrote contrasts_test.json) for ours_v2 and b9plus_v2 on the frozen
    bundles; outcomes come from the existing Phase A test store.
    simulator.e3v2_decide.outcome_record is wrapped at run time (the frozen file
    is not edited): the wrapper calls the frozen function unchanged, keeps the
    fields the contrast needs, and for the analysed cells also keeps each
    record's outcome parts and a summary of the commit decision's step trace.
 2. GATE: per-item Delta[COMMIT] = U(Ours-v2[COMMIT]) - U(B9+) recomputed with
    the frozen _item_utilities/_paired must equal contrasts_test.json for every
    (condition, profile, price) cell and every item to 1e-12, else the script
    stops without writing results. A second check requires the three-part
    decomposition below to sum to the frozen per-item Delta to 1e-12.
 3. Statistics: the paired item bootstrap of analysis/v2_test_d1.py (B = 10,000,
    seed 20261017, the same item universe and resamples; d1_open clusters are
    items). d2_support: the pair_id cluster bootstrap of analysis/v2_test_final.py
    (seed 20261018). Percentile 95% CIs. A mean over a subset of items is the
    ratio estimator under the shared resamples (the subset size varies).

ANALYSES (cell = condition x latency profile x price; primary = ASR
piper-en_US-hfc_male-medium__150wpm__clean, rewrite_faiss, 0.65 belief-quantile
price lambda_l = 0.173057, lambda_c = 0.05)
 A. Decomposition of Delta_C: U = Q - lambda_l dTTFA - lambda_c n_ret per record
    (simulator/e3v2_decide.outcome_record), item means over the 5 seeds, so
    Delta_i = dQ_i - lambda_l d(dTTFA)_i - lambda_c d(n_ret)_i exactly. dTTFA is
    TTFA minus the same draw's skip-path TTFA; its exposed-retrieval part is
    reported separately. Retrieval rates, accuracy conditional on retrieving,
    and the 2x2 table of (Ours retrieves, B9+ retrieves) with each block's
    contribution to Delta_C.
 B. Dispatch timing on items where each controller retrieves: dispatch time
    (the step's availability) minus end of speech (last word's end), and the
    fraction of the question's words fully spoken at the dispatched partial's
    audio time (1.0 at the end-of-turn step). B9+'s decision step (where its
    tuned r_below rule decides retrieve/skip; the same step whether or not it
    retrieves) is "when B9+ would have dispatched"; the share of Ours-S
    dispatches made at an earlier step is reported.
 C. Partial quality at dispatch: a dispatched partial is REVISED if its word
    sequence (lower-cased, punctuation stripped; analysis/v2_waste_measured.py)
    is not a prefix of the final ASR transcript's; REVISED-EXCL-LAST ignores its
    last word. A final-transcript dispatch is unrevised by definition. hit@3 =
    any gold document among the 3 retrieved parent ids (Phase A retrieval
    record for that dispatch point); recall@3 = share of gold documents
    retrieved. Reference: the final-transcript retrieval for the same item
    (Phase A records a retrieval for every dispatch point), and, where the
    lazy plan generated it, the final-transcript answer cell. rho_model is the
    controller's modelled quality-loss fraction at dispatch (frozen config),
    not a measurement. Strata of items by Ours-S's decision (early revised /
    early unrevised / end of turn / skip) with the paired Delta and its parts.
 D. The same decomposition on text150 (its own 0.65 price), where Delta_C ~ 0
    and partials are revision-free by construction.
 E. d2_support (EXPLORATORY; judge-graded; Deviation D3): A and B at its
    primary cell (lambda_l = 0.215782) and text150.

Writes docs/research_v2/mechanism_v2.json and docs/research_v2/mechanism_v2.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import simulator.e3v2_decide as e3dec  # noqa: E402
from analysis.v2_waste_measured import is_prefix, words  # noqa: E402
from llm.experiment1_v2 import common as e1common  # noqa: E402
from llm.experiment1_v2.prefixes import spoken_word_fraction  # noqa: E402
from controller.sequential_v2 import _partials, build_steps, end_of_turn_ms, speech_end_ms  # noqa: E402
from simulator.e3v2_decide import CellStore, _item_utilities, _paired, _price_key  # noqa: E402
from simulator.e3v2_dev_analysis import SEEDS  # noqa: E402
from simulator.e3v2_lazy import frozen_records  # noqa: E402

OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "mechanism_v2.json"
OUT_MD = _REPO_ROOT / "docs" / "research_v2" / "mechanism_v2.md"
SPLIT = "test"
OURS, B9P = "ours_v2", "b9plus_v2"
CONTROLLERS = (OURS, B9P)
ASR = "piper-en_US-hfc_male-medium__150wpm__clean"
PROFILE = "rewrite_faiss"
QUANTILE = 0.65
B = 10_000
SEED = {"d1_open": 20261017, "d2_support": 20261018}  # analysis/v2_test_d1.py, analysis/v2_test_final.py
FROZEN_PRIMARY = {"d1_open": "0.173057|0.05", "d2_support": "0.215782|0.05"}
STATUS = "EXPLORATORY, POST HOC (not pre-registered); associations within the frozen test run, not causal effects"
D2_STATUS = ("EXPLORATORY; d2_support outcomes are judge-graded and the pre-registered judge-validation gate "
             "was not passed (Deviation D3)")


# ---------------------------------------------------------------------------
# run-time capture around the frozen outcome_record
# ---------------------------------------------------------------------------
class Capture:
    def __init__(self):
        self.targets = set()          # {(condition, profile, price_key)}
        self.per_seed = defaultdict(list)   # (ctrl, cond, prof, pk, item) -> [per-seed dict]
        self.decisions = {}           # (ctrl, cond, prof, pk, item) -> decision summary
        self.steps = {}               # (item, cond) -> step summary


CAP = Capture()
RAW_CACHE = None  # optional directory for gated per-bundle captures (--raw-cache)
_FROZEN_OUTCOME_RECORD = e3dec.outcome_record


def _step_summary(turn, cfg):
    key = (turn.item_id, turn.condition)
    st = CAP.steps.get(key)
    if st is None:
        steps = build_steps(turn, cfg)
        parts = _partials(turn)
        st = {"n_steps": len(steps), "text": [s.text for s in steps], "pidx": [s.partial_index for s in steps],
              "avail": [s.available_ms for s in steps], "query_ready": [s.query_ready for s in steps],
              "t_ms": [None if s.partial_index is None else float(parts[s.partial_index]["t_ms"]) for s in steps],
              "speech_end_ms": float(speech_end_ms(turn)), "eot_ms": float(end_of_turn_ms(turn, cfg))}
        CAP.steps[key] = st
    return st


def _decision_summary(decision, turn, cfg):
    st = _step_summary(turn, cfg)
    last = decision.trace[-1]
    out = {"retrieve": bool(decision.retrieve), "dispatch_t_ms": decision.dispatch_t_ms,
           "dispatch_step": decision.dispatch_step, "dispatch_partial_index": decision.dispatch_partial_index,
           "query_text": decision.query_text, "timing_label": decision.timing_label,
           "decide_k": int(last.k), "decide_t_ms": int(last.available_ms), "decide_action": last.action,
           "decide_is_final": int(last.k) == st["n_steps"] - 1, "decide_partial_index": st["pidx"][int(last.k)]}
    if decision.retrieve:
        rec = next(t for t in decision.trace if t.k == decision.dispatch_step)
        assert decision.dispatch_t_ms == st["avail"][decision.dispatch_step]
        assert decision.dispatch_partial_index == st["pidx"][decision.dispatch_step]
        out.update({"rho_model": rec.rho, "belief": rec.belief, "waste_model": rec.waste,
                    "u_retrieve": rec.u_retrieve, "v_wait": rec.v_wait, "exposed_model_ms": rec.exposed_ms})
    return out


def _capturing_outcome_record(name, decision, turn, seed, profile, prices, store, latency_model, split,
                              architecture="commit", cfg=None):
    rec = _FROZEN_OUTCOME_RECORD(name, decision, turn, seed, profile, prices, store, latency_model, split,
                                 architecture, cfg)
    if architecture != "commit":
        return None  # the speculative records are not needed here
    if prices is not None:
        key = (turn.condition, profile, _price_key(prices))
        if key in CAP.targets:
            k = (name,) + key + (turn.item_id,)
            CAP.per_seed[k].append({"seed": int(seed), "Q": float(bool(rec["outcome"]["answer_correct"])),
                                    "ttfa_ms": float(rec["outcome"]["ttfa_ms"]),
                                    "skip_ttfa_ms": float(rec["skip_ttfa_ms"]),
                                    "exposed_ms": float(rec["exposed_latency_ms"]),
                                    "n_ret": int(rec["outcome"]["retrievals"]),
                                    "utility": float(rec["realized_utility"]), "query_variant": rec["query_variant"]})
            if k not in CAP.decisions:
                CAP.decisions[k] = _decision_summary(decision, turn, cfg)
    return {"baseline": rec["baseline"], "item_id": rec["item_id"], "architecture": rec["architecture"],
            "latency_profile": rec["latency_profile"], "prices": rec["prices"],
            "realized_utility": rec["realized_utility"]}


e3dec.outcome_record = _capturing_outcome_record  # run-time wrap only; simulator/e3v2_decide.py is unchanged


# ---------------------------------------------------------------------------
# evaluation + gate
# ---------------------------------------------------------------------------
def gold_ids(item):
    g = item.get("gold_doc_ids") or []
    return [str(x) for x in g]


def run_dataset(ds: str, phase_a_root: Path, e1_results: Path) -> dict:
    phase_a = phase_a_root / ds
    root = phase_a_root.resolve().parent.parent  # relative bundle paths in contrasts are relative to it
    doc_c = e1common.load_json(phase_a / f"contrasts_{SPLIT}.json")
    frozen = {cond: {(r["latency_profile"], r["price"]): r["per_item"]["delta_commit"] for r in rows}
              for cond, rows in doc_c["contrasts"].items()}
    bundles = []
    for bp in doc_c["bundles"]:
        p = Path(bp)
        bundles.append(e1common.load_json(p if p.is_absolute() else root / p))
    targets = {}
    for b in bundles:
        g = next(g for g in b["price_grid"][PROFILE] if g["quantile"] == QUANTILE)
        targets[b["condition"]] = (b["condition"], PROFILE, f"{g['lambda_l']:g}|{g['lambda_c']:g}")
    assert targets[ASR][2] == FROZEN_PRIMARY[ds], (ds, targets[ASR])
    items = {it["id"]: it for it in e1common.load_items(e1common.resolve_dataset_paths(ds)["items"])}
    traces = defaultdict(dict)
    for tr in e1common.read_jsonl(phase_a / f"conditions_{SPLIT}.jsonl"):
        if tr["condition"] in targets:
            traces[tr["condition"]][tr["item_id"]] = tr

    print(f"[{ds}] loading the Phase A test store ...", flush=True)
    store = CellStore.load(phase_a / f"store_{SPLIT}.jsonl")
    gate = {"cells_checked": 0, "items_checked": 0, "max_abs_diff": 0.0}
    cells = {}
    for bundle in bundles:
        cond = bundle["condition"]
        CAP.targets = {targets[cond]}
        CAP.per_seed.clear()
        CAP.decisions.clear()
        CAP.steps.clear()
        raw = None if RAW_CACHE is None else RAW_CACHE / f"raw_{ds}_{cond.replace('/', '_')}.pkl"
        if raw is not None and raw.exists():  # development convenience only; written after the gate passed
            with open(raw, "rb") as fh:
                saved = pickle.load(fh)
            CAP.per_seed.update(saved["per_seed"])
            CAP.decisions.update(saved["decisions"])
            CAP.steps.update(saved["steps"])
            for k in gate:
                gate[k] = gate[k] + saved["gate"][k] if k != "max_abs_diff" else max(gate[k], saved["gate"][k])
            print(f"[{ds}] read the gated capture for {cond} from {raw}", flush=True)
        else:
            print(f"[{ds}] frozen evaluator: {cond} ...", flush=True)
            recs = [r for r in frozen_records(bundle, ds, SPLIT, phase_a, e1_results, store, CONTROLLERS)
                    if r is not None]
            utils = _item_utilities(recs)
            del recs
            g1 = {"cells_checked": 0, "items_checked": 0, "max_abs_diff": 0.0}
            for (prof, pk), ref in frozen[cond].items():
                got = _paired(utils[("commit", prof, pk)][OURS], utils[("commit", prof, pk)][B9P])
                if set(got) != set(ref):
                    raise SystemExit(f"GATE FAILED [{ds} {cond} {prof} {pk}]: item sets differ; no results written")
                diff = max(abs(got[i] - ref[i]) for i in ref)
                if diff > 1e-12:
                    raise SystemExit(f"GATE FAILED [{ds} {cond} {prof} {pk}]: max |diff| {diff:.3e}; "
                                     "no results written")
                g1["cells_checked"] += 1
                g1["items_checked"] += len(ref)
                g1["max_abs_diff"] = max(g1["max_abs_diff"], diff)
            print(f"[{ds}] gate passed for {cond}: every cell reproduces contrasts_test.json", flush=True)
            for k in gate:
                gate[k] = gate[k] + g1[k] if k != "max_abs_diff" else max(gate[k], g1[k])
            if raw is not None:
                with open(raw, "wb") as fh:
                    pickle.dump({"per_seed": dict(CAP.per_seed), "decisions": CAP.decisions, "steps": CAP.steps,
                                 "gate": g1}, fh)
        cells[cond] = extract_cell(ds, cond, targets[cond], frozen[cond][targets[cond][1:]], store, items,
                                   traces[cond], bundle)
    del store
    universe = sorted({i for rows in doc_c["contrasts"].values() for r in rows
                       for k in ("delta_commit", "delta_speculative") for i in r["per_item"][k]})
    cluster_of = {i: (it.get("pair_id") or i) for i, it in items.items()}
    return {"dataset": ds, "gate": gate, "cells": cells, "universe": universe, "cluster_of": cluster_of,
            "inputs": {"contrasts_sha256": e1common.sha256_file(phase_a / f"contrasts_{SPLIT}.json"),
                       "store_sha256": e1common.sha256_file(phase_a / f"store_{SPLIT}.jsonl"),
                       "bundles": [str(b) for b in doc_c["bundles"]]}}


def extract_cell(ds, cond, target, frozen_delta, store, items, traces, bundle) -> dict:
    """Per item and controller: seed-mean outcome parts, the decision, and
    retrieval-quality facts of the dispatch point (from the store)."""
    _, prof, pk = target
    # the exact grid prices (the price key is rounded by :g)
    row_ = next(g for g in bundle["price_grid"][prof] if f"{g['lambda_l']:g}|{g['lambda_c']:g}" == pk)
    lam_l, lam_c = float(row_["lambda_l"]), float(row_["lambda_c"])
    out = {"condition": cond, "profile": prof, "price": pk, "lambda_l": lam_l, "lambda_c": lam_c,
           "b9plus_timing_rule": ((bundle["tuned"].get(prof) or {}).get(pk.split("|")[0]) or {}).get(
               "b9plus_timing_rule"),
           "items": {}}
    max_diff = 0.0
    for i in sorted(frozen_delta):
        tr = traces[i]
        st = CAP.steps[(i, cond)]
        final_words = words(st["text"][-1])
        gold = set(gold_ids(items[i]))
        row = {"speech_end_ms": st["speech_end_ms"], "eot_ms": st["eot_ms"]}
        fin = store.retrievals.get((i, cond, None))
        row["final_retrieval"] = None
        if fin is not None:
            ids = fin.get("retrieved_parent_ids") or fin["retrieved_doc_ids"]
            qf = [store.cells.get((i, s, cond, "retrieve", fin["query_variant"])) for s in SEEDS]
            row["final_retrieval"] = {
                "hit3": None if not gold else float(bool(gold & set(ids))),
                "recall3": None if not gold else len(gold & set(ids)) / len(gold),
                "Q_final_cell": (None if any(c is None for c in qf)
                                 else float(np.mean([float(bool(c["answer_correct"])) for c in qf])))}
        for ctrl in CONTROLLERS:
            k = (ctrl, cond, prof, pk, i)
            seeds = CAP.per_seed[k]
            assert [s["seed"] for s in seeds] == list(SEEDS), (k, [s["seed"] for s in seeds])
            d = dict(CAP.decisions[k])
            util = sum(s["utility"] for s in seeds) / len(seeds)  # as _item_utilities
            c = {"U": util, "Q": float(np.mean([s["Q"] for s in seeds])),
                 "dttfa_ms": float(np.mean([s["ttfa_ms"] - s["skip_ttfa_ms"] for s in seeds])),
                 "ttfa_ms": float(np.mean([s["ttfa_ms"] for s in seeds])),
                 "exposed_ms": float(np.mean([s["exposed_ms"] for s in seeds])),
                 "n_ret": float(np.mean([s["n_ret"] for s in seeds])),
                 "query_variant": seeds[0]["query_variant"]}
            assert len({s["query_variant"] for s in seeds}) == 1 and len({s["n_ret"] for s in seeds}) == 1
            c["retrieve"] = d["retrieve"]
            assert bool(c["n_ret"]) == d["retrieve"]
            c["decide_t_ms"] = d["decide_t_ms"]
            c["decide_is_final"] = d["decide_is_final"]
            dk = d["decide_k"]
            c["decide_frac"] = 1.0 if d["decide_is_final"] else frac_heard(tr, st["t_ms"][dk])
            c["decide_rel_ms"] = d["decide_t_ms"] - st["speech_end_ms"]
            if d["retrieve"]:
                pidx = d["dispatch_partial_index"]
                retr = store.retrievals[(i, cond, pidx)]
                assert retr["query_variant"] == c["query_variant"]
                ids = retr.get("retrieved_parent_ids") or retr["retrieved_doc_ids"]
                qw = words(d["query_text"])
                c.update({
                    "dispatch_t_ms": d["dispatch_t_ms"], "dispatch_partial_index": pidx,
                    "dispatch_rel_ms": d["dispatch_t_ms"] - st["speech_end_ms"],
                    "dispatch_frac": 1.0 if pidx is None else frac_heard(tr, st["t_ms"][d["dispatch_step"]]),
                    "early": pidx is not None,
                    "revised": (pidx is not None) and not is_prefix(qw, final_words),
                    "revised_excl_last": (pidx is not None) and not is_prefix(qw[:-1], final_words),
                    "hit3": None if not gold else float(bool(gold & set(ids))),
                    "recall3": None if not gold else len(gold & set(ids)) / len(gold),
                    "rho_model": d.get("rho_model"), "belief": d.get("belief"),
                    "timing_label": d["timing_label"]})
            row[ctrl] = c
        o, b = row[OURS], row[B9P]
        row["delta"] = o["U"] - b["U"]
        row["dQ"] = o["Q"] - b["Q"]
        row["dLat"] = -lam_l * (o["dttfa_ms"] - b["dttfa_ms"]) / 1000.0
        row["dComp"] = -lam_c * (o["n_ret"] - b["n_ret"])
        max_diff = max(max_diff, abs(row["delta"] - frozen_delta[i]),
                       abs(row["dQ"] + row["dLat"] + row["dComp"] - frozen_delta[i]))
        out["items"][i] = row
    if max_diff > 1e-12:
        raise SystemExit(f"CHECK FAILED [{ds} {cond}]: extracted Delta or its decomposition differs from "
                         f"contrasts_test.json by {max_diff:.3e}; no results written")
    out["decomposition_check_max_abs_diff"] = max_diff
    return out


def frac_heard(tr, t_ms):
    """Fraction of the question's words fully spoken by the partial's audio time
    (text150: the prefix's word fraction)."""
    for p in tr["partials"]:
        if float(p["t_ms"]) == float(t_ms) and p.get("word_frac") is not None:
            return float(p["word_frac"])
    f = spoken_word_fraction(tr, t_ms)
    return 0.0 if f is None else float(f)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
class Boot:
    def __init__(self, universe, counts):
        self.universe = universe
        self.idx = {i: k for k, i in enumerate(universe)}
        self.counts = counts

    def _vec(self, values):
        x, m = np.zeros(len(self.universe)), np.zeros(len(self.universe))
        for i, v in values.items():
            x[self.idx[i]] = float(v)
            m[self.idx[i]] = 1.0
        return x, m

    def mean(self, values):
        """Mean over the items in `values` (ratio estimator under resampling)."""
        if not values:
            return None, None
        x, m = self._vec(values)
        tot = self.counts @ m
        return float(np.mean(list(values.values()))), (self.counts @ x) / np.where(tot > 0, tot, np.nan)

    def contrib(self, values, cell_items):
        """sum over `values` / number of items in the cell: a block's share of the cell mean."""
        x, _ = self._vec(values)
        _, mc = self._vec({i: 1.0 for i in cell_items})
        tot = self.counts @ mc
        return float(sum(values.values()) / len(cell_items)), (self.counts @ x) / tot


def summ(est, means, nd=None):
    if est is None:
        return None
    q = np.nanpercentile(means, [2.5, 97.5])
    return {"estimate": est, "ci95": [float(q[0]), float(q[1])], "boot_se": float(np.nanstd(means, ddof=1))}


def diff_summ(a, b):
    (ea, ma), (eb, mb) = a, b
    if ea is None or eb is None:
        return None
    return summ(ea - eb, ma - mb)


def quantiles(vals):
    if not vals:
        return None
    v = np.asarray(vals, dtype=float)
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"n": int(v.size), "mean": float(v.mean()), "p10": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def decomposition(cell, bt: Boot) -> dict:
    it = cell["items"]
    ids = sorted(it)
    out = {"n_items": len(ids)}
    out["delta_C"] = summ(*bt.mean({i: it[i]["delta"] for i in ids}))
    out["parts"] = {
        "accuracy_dQ": summ(*bt.mean({i: it[i]["dQ"] for i in ids})),
        "latency_minus_lambda_l_ddTTFA": summ(*bt.mean({i: it[i]["dLat"] for i in ids})),
        "compute_minus_lambda_c_dn_ret": summ(*bt.mean({i: it[i]["dComp"] for i in ids})),
    }
    out["parts_note"] = ("contributions to Delta_C (Ours-S minus B9+), signed so that they sum to Delta_C: "
                         "dQ - lambda_l * d(TTFA - skip TTFA) [s] - lambda_c * d(n_retrievals)")
    raw = {}
    for ctrl in CONTROLLERS:
        raw[ctrl] = {
            "accuracy": summ(*bt.mean({i: it[i][ctrl]["Q"] for i in ids})),
            "retrieval_rate": summ(*bt.mean({i: float(it[i][ctrl]["retrieve"]) for i in ids})),
            "accuracy_if_retrieves": summ(*bt.mean({i: it[i][ctrl]["Q"] for i in ids if it[i][ctrl]["retrieve"]})),
            "accuracy_if_skips": summ(*bt.mean({i: it[i][ctrl]["Q"] for i in ids if not it[i][ctrl]["retrieve"]})),
            "n_retrieves": sum(1 for i in ids if it[i][ctrl]["retrieve"]),
            "mean_dTTFA_ms": summ(*bt.mean({i: it[i][ctrl]["dttfa_ms"] for i in ids})),
            "mean_TTFA_ms": summ(*bt.mean({i: it[i][ctrl]["ttfa_ms"] for i in ids})),
            "mean_exposed_retrieval_ms": summ(*bt.mean({i: it[i][ctrl]["exposed_ms"] for i in ids})),
            "mean_dTTFA_ms_if_retrieves": summ(*bt.mean({i: it[i][ctrl]["dttfa_ms"] for i in ids
                                                         if it[i][ctrl]["retrieve"]})),
            "mean_exposed_retrieval_ms_if_retrieves": summ(*bt.mean({i: it[i][ctrl]["exposed_ms"] for i in ids
                                                                     if it[i][ctrl]["retrieve"]})),
        }
    out["per_controller"] = raw
    out["differences_ours_minus_b9plus"] = {
        "accuracy": summ(*bt.mean({i: it[i]["dQ"] for i in ids})),
        "retrieval_rate": summ(*bt.mean({i: float(it[i][OURS]["retrieve"]) - float(it[i][B9P]["retrieve"])
                                         for i in ids})),
        "dTTFA_ms": summ(*bt.mean({i: it[i][OURS]["dttfa_ms"] - it[i][B9P]["dttfa_ms"] for i in ids})),
        "exposed_retrieval_ms": summ(*bt.mean({i: it[i][OURS]["exposed_ms"] - it[i][B9P]["exposed_ms"]
                                               for i in ids})),
        "accuracy_if_retrieves_unpaired": diff_summ(
            bt.mean({i: it[i][OURS]["Q"] for i in ids if it[i][OURS]["retrieve"]}),
            bt.mean({i: it[i][B9P]["Q"] for i in ids if it[i][B9P]["retrieve"]})),
    }
    out["latency_split_note"] = ("dTTFA includes the exposed retrieval time and the longer retrieve-path prompt; "
                                 "mean_exposed_retrieval_ms is the exposed part alone")
    # 2x2: (Ours retrieves, B9+ retrieves)
    blocks = {}
    for ro in (True, False):
        for rb in (True, False):
            sub = [i for i in ids if it[i][OURS]["retrieve"] == ro and it[i][B9P]["retrieve"] == rb]
            name = f"ours_{'R' if ro else 'S'}__b9plus_{'R' if rb else 'S'}"
            blk = {"n": len(sub), "share": len(sub) / len(ids)}
            if sub:
                blk["mean_delta"] = summ(*bt.mean({i: it[i]["delta"] for i in sub}))
                blk["contribution_to_delta_C"] = summ(*bt.contrib({i: it[i]["delta"] for i in sub}, ids))
                for part in ("dQ", "dLat", "dComp"):
                    blk[f"contribution_{part}"] = summ(*bt.contrib({i: it[i][part] for i in sub}, ids))
                blk["mean_Q_ours"] = summ(*bt.mean({i: it[i][OURS]["Q"] for i in sub}))
                blk["mean_Q_b9plus"] = summ(*bt.mean({i: it[i][B9P]["Q"] for i in sub}))
                blk["mean_dQ"] = summ(*bt.mean({i: it[i]["dQ"] for i in sub}))
            blocks[name] = blk
    out["two_by_two"] = blocks
    return out


def model_quality(cell, bt: Boot, quality: dict) -> dict:
    """The accuracy difference the FROZEN objective expected at the two dispatches: per item
    q_x = b_x g+ (1 - rho_x) - (1 - b_x) g- if x retrieves (each controller's own belief and modelled rho
    at its dispatch step; _objective_v2.utility_retrieve's quality term), else 0; model dQ = q_ours - q_b9plus.
    Compared with the realised dQ on the same items."""
    it = cell["items"]
    ids = sorted(it)
    gp, gm = float(quality["g_plus"]), float(quality["g_minus"])

    def q(c):
        if not c["retrieve"]:
            return 0.0
        b = min(1.0, max(0.0, float(c["belief"])))
        return b * gp * (1.0 - float(c["rho_model"])) - (1.0 - b) * gm

    model = {i: q(it[i][OURS]) - q(it[i][B9P]) for i in ids}
    out = {"g_plus": gp, "g_minus": gm,
           "note": "model dQ uses the frozen quality term b g+ (1 - rho) - (1 - b) g- (rho scales the g+ term "
                   "only); a diagnostic of what the controllers' objective expected, not a prediction fitted here"}
    both = [i for i in ids if it[i][OURS]["retrieve"] and it[i][B9P]["retrieve"]]
    earlier = [i for i in both if it[i][OURS]["dispatch_t_ms"] < it[i][B9P]["dispatch_t_ms"]]
    for name, sub in (("all_items", ids), ("both_retrieve_ours_earlier", earlier)):
        if not sub:
            continue
        m = bt.mean({i: model[i] for i in sub})
        r = bt.mean({i: it[i]["dQ"] for i in sub})
        out[name] = {"n": len(sub), "model_dQ": summ(*m), "realised_dQ": summ(*r),
                     "realised_minus_model": diff_summ(r, m),
                     "mean_rho_model_ours": float(np.mean([it[i][OURS]["rho_model"] for i in sub
                                                          if it[i][OURS]["retrieve"]])),
                     "mean_rho_model_b9plus": (float(np.mean([it[i][B9P]["rho_model"] for i in sub
                                                              if it[i][B9P]["retrieve"]]))
                                               if any(it[i][B9P]["retrieve"] for i in sub) else None),
                     "mean_belief_ours": float(np.mean([it[i][OURS]["belief"] for i in sub if it[i][OURS]["retrieve"]]))}
    return out


def timing(cell, bt: Boot) -> dict:
    it = cell["items"]
    ids = sorted(it)
    out = {}
    for ctrl in CONTROLLERS:
        r = [i for i in ids if it[i][ctrl]["retrieve"]]
        out[ctrl] = {
            "n_retrievals": len(r),
            "dispatch_minus_speech_end_ms": quantiles([it[i][ctrl]["dispatch_rel_ms"] for i in r]),
            "fraction_of_question_heard": quantiles([it[i][ctrl]["dispatch_frac"] for i in r]),
            "share_before_speech_end": summ(*bt.mean({i: float(it[i][ctrl]["dispatch_rel_ms"] < 0) for i in r})),
            "share_on_early_partial": summ(*bt.mean({i: float(it[i][ctrl]["early"]) for i in r})),
            "share_at_end_of_turn_step": summ(*bt.mean({i: float(not it[i][ctrl]["early"]) for i in r})),
            "mean_dispatch_minus_speech_end_ms": summ(*bt.mean({i: it[i][ctrl]["dispatch_rel_ms"] for i in r})),
            "mean_fraction_heard": summ(*bt.mean({i: it[i][ctrl]["dispatch_frac"] for i in r})),
        }
    out[B9P]["decision_step_all_items"] = {
        "note": "B9+'s tuned r_below rule decides retrieve/skip at one step (first query-ready partial with "
                "R_hat <= r0, else end of turn); this is when it would have dispatched",
        "decide_minus_speech_end_ms": quantiles([it[i][B9P]["decide_rel_ms"] for i in ids]),
        "fraction_heard": quantiles([it[i][B9P]["decide_frac"] for i in ids]),
        "share_at_end_of_turn_step": summ(*bt.mean({i: float(it[i][B9P]["decide_is_final"]) for i in ids})),
    }
    r = [i for i in ids if it[i][OURS]["retrieve"]]
    rel = {i: it[i][OURS]["dispatch_t_ms"] - it[i][B9P]["decide_t_ms"] for i in r}
    out["ours_vs_b9plus_decision_step"] = {
        "n": len(r),
        "share_ours_earlier": summ(*bt.mean({i: float(v < 0) for i, v in rel.items()})),
        "share_same_step": summ(*bt.mean({i: float(v == 0) for i, v in rel.items()})),
        "share_ours_later": summ(*bt.mean({i: float(v > 0) for i, v in rel.items()})),
        "ours_minus_b9plus_ms": quantiles(list(rel.values())),
    }
    both = [i for i in ids if it[i][OURS]["retrieve"] and it[i][B9P]["retrieve"]]
    relb = {i: it[i][OURS]["dispatch_t_ms"] - it[i][B9P]["dispatch_t_ms"] for i in both}
    out["both_retrieve"] = {
        "n": len(both),
        "share_ours_earlier": summ(*bt.mean({i: float(v < 0) for i, v in relb.items()})),
        "share_same_step": summ(*bt.mean({i: float(v == 0) for i, v in relb.items()})),
        "share_ours_later": summ(*bt.mean({i: float(v > 0) for i, v in relb.items()})),
        "share_same_query_variant": summ(*bt.mean({i: float(it[i][OURS]["query_variant"]
                                                            == it[i][B9P]["query_variant"]) for i in both})),
        "ours_minus_b9plus_dispatch_ms": quantiles(list(relb.values())),
    }
    by = {}
    for name, sub in (("ours_earlier", [i for i in both if relb[i] < 0]),
                      ("same_step", [i for i in both if relb[i] == 0]),
                      ("ours_later", [i for i in both if relb[i] > 0])):
        blk = {"n": len(sub)}
        if sub:
            blk["mean_delta"] = summ(*bt.mean({i: it[i]["delta"] for i in sub}))
            blk["contribution_to_delta_C"] = summ(*bt.contrib({i: it[i]["delta"] for i in sub}, ids))
            blk["mean_dQ"] = summ(*bt.mean({i: it[i]["dQ"] for i in sub}))
            blk["mean_latency_part"] = summ(*bt.mean({i: it[i]["dLat"] for i in sub}))
            h = [i for i in sub if it[i][OURS].get("hit3") is not None]
            if h:
                blk["hit3_ours_minus_b9plus"] = summ(*bt.mean({i: it[i][OURS]["hit3"] - it[i][B9P]["hit3"]
                                                              for i in h}))
            blk["mean_fraction_heard_ours"] = float(np.mean([it[i][OURS]["dispatch_frac"] for i in sub]))
            blk["mean_fraction_heard_b9plus"] = float(np.mean([it[i][B9P]["dispatch_frac"] for i in sub]))
        by[name] = blk
    out["both_retrieve"]["by_relative_step"] = by
    return out


STRATA = ("early_revised", "early_unrevised", "end_of_turn", "skip")


def stratum_of(c, key):
    if not c["retrieve"]:
        return "skip"
    if not c["early"]:
        return "end_of_turn"
    return "early_revised" if c[key] else "early_unrevised"


def partial_quality(cell, bt: Boot) -> dict:
    it = cell["items"]
    ids = sorted(it)
    out = {}
    for ctrl in CONTROLLERS:
        r = [i for i in ids if it[i][ctrl]["retrieve"]]
        early = [i for i in r if it[i][ctrl]["early"]]
        blk = {"n_retrievals": len(r), "n_early": len(early)}
        for key in ("revised", "revised_excl_last"):
            blk[f"share_{key}_of_all_retrievals"] = summ(*bt.mean({i: float(it[i][ctrl][key]) for i in r}))
            blk[f"share_{key}_of_early_retrievals"] = summ(*bt.mean({i: float(it[i][ctrl][key]) for i in early}))
        blk["hit3_at_dispatch"] = summ(*bt.mean({i: it[i][ctrl]["hit3"] for i in r if it[i][ctrl]["hit3"] is not None}))
        blk["recall3_at_dispatch"] = summ(*bt.mean({i: it[i][ctrl]["recall3"] for i in r
                                                    if it[i][ctrl]["recall3"] is not None}))
        fr = [i for i in r if it[i]["final_retrieval"] is not None and it[i]["final_retrieval"]["hit3"] is not None
              and it[i][ctrl]["hit3"] is not None]
        blk["hit3_final_transcript_same_items"] = summ(*bt.mean({i: it[i]["final_retrieval"]["hit3"] for i in fr}))
        blk["hit3_dispatch_minus_final_same_items"] = summ(*bt.mean(
            {i: it[i][ctrl]["hit3"] - it[i]["final_retrieval"]["hit3"] for i in fr}))
        fe = [i for i in fr if it[i][ctrl]["early"]]
        blk["early_only"] = {
            "n": len(fe),
            "hit3_at_dispatch": summ(*bt.mean({i: it[i][ctrl]["hit3"] for i in fe})),
            "hit3_final_transcript": summ(*bt.mean({i: it[i]["final_retrieval"]["hit3"] for i in fe})),
            "hit3_dispatch_minus_final": summ(*bt.mean({i: it[i][ctrl]["hit3"] - it[i]["final_retrieval"]["hit3"]
                                                        for i in fe})),
            "mean_rho_model": float(np.mean([it[i][ctrl]["rho_model"] for i in fe])) if fe else None,
        }
        hf = sum(it[i]["final_retrieval"]["hit3"] for i in fe)
        blk["early_only"]["relative_hit3_loss_1_minus_ratio"] = (
            None if not fe or hf == 0 else 1.0 - sum(it[i][ctrl]["hit3"] for i in fe) / hf)
        # final-transcript answer cell where the lazy plan generated it (selection: some controller landed there)
        fq = [i for i in fe if it[i]["final_retrieval"]["Q_final_cell"] is not None]
        blk["early_only"]["final_answer_cell_available"] = len(fq)
        blk["early_only"]["Q_dispatch_minus_Q_final_cell"] = summ(*bt.mean(
            {i: it[i][ctrl]["Q"] - it[i]["final_retrieval"]["Q_final_cell"] for i in fq}))
        out[ctrl] = blk
    # Ours accuracy by partial status
    o = {}
    for key in ("revised", "revised_excl_last"):
        st = {s: [i for i in ids if stratum_of(it[i][OURS], key) == s] for s in STRATA}
        acc = {s: bt.mean({i: it[i][OURS]["Q"] for i in st[s]}) for s in STRATA if s != "skip"}
        hit = {s: bt.mean({i: it[i][OURS]["hit3"] for i in st[s] if it[i][OURS]["hit3"] is not None})
               for s in STRATA if s != "skip"}
        blk = {"ours_accuracy": {s: summ(*acc[s]) for s in acc},
               "ours_hit3": {s: summ(*hit[s]) for s in hit},
               "ours_accuracy_revised_minus_unrevised": diff_summ(acc["early_revised"], acc["early_unrevised"]),
               "ours_hit3_revised_minus_unrevised": diff_summ(hit["early_revised"], hit["early_unrevised"]),
               "strata": {}}
        means = {}
        for s in STRATA:
            sub = st[s]
            row = {"n": len(sub), "share_of_items": len(sub) / len(ids)}
            if sub:
                means[s] = bt.mean({i: it[i]["delta"] for i in sub})
                row["mean_delta"] = summ(*means[s])
                row["contribution_to_delta_C"] = summ(*bt.contrib({i: it[i]["delta"] for i in sub}, ids))
                for part in ("dQ", "dLat", "dComp"):
                    row[f"mean_{part}"] = summ(*bt.mean({i: it[i][part] for i in sub}))
                row["b9plus_retrieval_rate"] = summ(*bt.mean({i: float(it[i][B9P]["retrieve"]) for i in sub}))
                row["mean_Q_ours"] = summ(*bt.mean({i: it[i][OURS]["Q"] for i in sub}))
                row["mean_Q_b9plus"] = summ(*bt.mean({i: it[i][B9P]["Q"] for i in sub}))
                rs = [i for i in sub if it[i][OURS]["retrieve"]]
                rb = [i for i in sub if it[i][B9P]["retrieve"]]
                row["mean_fraction_heard_ours"] = (float(np.mean([it[i][OURS]["dispatch_frac"] for i in rs]))
                                                   if rs else None)
                row["mean_fraction_heard_b9plus"] = (float(np.mean([it[i][B9P]["dispatch_frac"] for i in rb]))
                                                     if rb else None)
                both = [i for i in sub if it[i][OURS]["retrieve"] and it[i][B9P]["retrieve"]
                        and it[i][OURS]["hit3"] is not None]
                row["both_retrieve"] = {
                    "n": len(both),
                    "b9plus_revised_share": summ(*bt.mean({i: float(it[i][B9P][key]) for i in both})),
                    "hit3_ours": summ(*bt.mean({i: it[i][OURS]["hit3"] for i in both})),
                    "hit3_b9plus": summ(*bt.mean({i: it[i][B9P]["hit3"] for i in both})),
                    "hit3_ours_minus_b9plus": summ(*bt.mean({i: it[i][OURS]["hit3"] - it[i][B9P]["hit3"]
                                                             for i in both})),
                    "Q_ours_minus_b9plus": summ(*bt.mean({i: it[i]["dQ"] for i in both})),
                    "mean_delta": summ(*bt.mean({i: it[i]["delta"] for i in both})),
                } if both else {"n": 0}
            blk["strata"][s] = row
        blk["mean_delta_early_revised_minus_early_unrevised"] = (
            diff_summ(means["early_revised"], means["early_unrevised"])
            if "early_revised" in means and "early_unrevised" in means else None)
        rest = [i for i in ids if stratum_of(it[i][OURS], key) != "early_revised"]
        blk["mean_delta_early_revised_minus_all_other_items"] = (
            diff_summ(means["early_revised"], bt.mean({i: it[i]["delta"] for i in rest}))
            if "early_revised" in means else None)
        o[key] = blk
    out["ours_by_partial_status"] = o
    # revised vs unrevised at similar fraction heard (Ours-S early retrievals): revision and earliness co-vary
    bins = ((0.0, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.0001))
    fb = {}
    for key in ("revised", "revised_excl_last"):
        rows = []
        for lo, hi in bins:
            sub = [i for i in ids if it[i][OURS]["retrieve"] and it[i][OURS]["early"]
                   and lo <= it[i][OURS]["dispatch_frac"] < hi and it[i][OURS]["hit3"] is not None]
            rv = [i for i in sub if it[i][OURS][key]]
            un = [i for i in sub if not it[i][OURS][key]]
            rows.append({
                "fraction_heard_bin": [lo, min(hi, 1.0)], "n_revised": len(rv), "n_unrevised": len(un),
                "Q_revised": summ(*bt.mean({i: it[i][OURS]["Q"] for i in rv})),
                "Q_unrevised": summ(*bt.mean({i: it[i][OURS]["Q"] for i in un})),
                "Q_revised_minus_unrevised": diff_summ(bt.mean({i: it[i][OURS]["Q"] for i in rv}),
                                                       bt.mean({i: it[i][OURS]["Q"] for i in un})),
                "hit3_revised_minus_unrevised": diff_summ(bt.mean({i: it[i][OURS]["hit3"] for i in rv}),
                                                          bt.mean({i: it[i][OURS]["hit3"] for i in un})),
                "mean_delta_revised_minus_unrevised": diff_summ(bt.mean({i: it[i]["delta"] for i in rv}),
                                                                bt.mean({i: it[i]["delta"] for i in un})),
            })
        fb[key] = rows
    out["ours_revised_vs_unrevised_by_fraction_heard"] = fb
    return out


def analyse(res: dict, counts: np.ndarray, with_quality: bool, quality: dict) -> dict:
    bt = Boot(res["universe"], counts)
    out = {}
    for cond, cell in res["cells"].items():
        name = "asr_primary" if cond == ASR else cond
        blk = {"condition": cond, "profile": cell["profile"], "price": cell["price"],
               "lambda_l": cell["lambda_l"], "lambda_c": cell["lambda_c"],
               "b9plus_timing_rule": cell["b9plus_timing_rule"],
               "decomposition_check_max_abs_diff": cell["decomposition_check_max_abs_diff"],
               "A_decomposition": decomposition(cell, bt), "B_timing": timing(cell, bt),
               "A2_model_vs_realised_accuracy": model_quality(cell, bt, quality[cond])}
        if with_quality:
            blk["C_partial_quality"] = partial_quality(cell, bt)
        out[name] = blk
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def f(x, nd=4, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(s, nd=4, sign=True, scale=1.0):
    if s is None:
        return "—"
    return (f"{f(s['estimate'] * scale, nd, sign)} [{f(s['ci95'][0] * scale, nd, sign)}, "
            f"{f(s['ci95'][1] * scale, nd, sign)}]")


def md_decomposition(L, a):
    d = a["A_decomposition"]
    p = d["parts"]
    L += ["| | estimate [95% CI] |", "|---|---|",
          f"| Δ_C (Ours-S − B9+) | {ci(d['delta_C'])} |",
          f"| accuracy part: ΔQ | {ci(p['accuracy_dQ'])} |",
          f"| latency part: −λ_l·Δ(dTTFA) | {ci(p['latency_minus_lambda_l_ddTTFA'])} |",
          f"| compute part: −λ_c·Δ(n_ret) | {ci(p['compute_minus_lambda_c_dn_ret'])} |", ""]
    pc = d["per_controller"]
    dd = d["differences_ours_minus_b9plus"]
    L += ["| | Ours-S | B9+ | Ours-S − B9+ |", "|---|---|---|---|",
          f"| retrieval rate | {ci(pc[OURS]['retrieval_rate'], 3, False)} | {ci(pc[B9P]['retrieval_rate'], 3, False)} "
          f"| {ci(dd['retrieval_rate'], 3)} |",
          f"| accuracy (all items) | {ci(pc[OURS]['accuracy'], 3, False)} | {ci(pc[B9P]['accuracy'], 3, False)} "
          f"| {ci(dd['accuracy'], 3)} |",
          f"| accuracy if retrieves | {ci(pc[OURS]['accuracy_if_retrieves'], 3, False)} | "
          f"{ci(pc[B9P]['accuracy_if_retrieves'], 3, False)} | {ci(dd['accuracy_if_retrieves_unpaired'], 3)} (unpaired) |",
          f"| accuracy if skips | {ci(pc[OURS]['accuracy_if_skips'], 3, False)} | "
          f"{ci(pc[B9P]['accuracy_if_skips'], 3, False)} | |",
          f"| mean dTTFA (ms) | {ci(pc[OURS]['mean_dTTFA_ms'], 0, False)} | {ci(pc[B9P]['mean_dTTFA_ms'], 0, False)} "
          f"| {ci(dd['dTTFA_ms'], 0)} |",
          f"| mean exposed retrieval (ms) | {ci(pc[OURS]['mean_exposed_retrieval_ms'], 0, False)} | "
          f"{ci(pc[B9P]['mean_exposed_retrieval_ms'], 0, False)} | {ci(dd['exposed_retrieval_ms'], 0)} |",
          f"| mean dTTFA if retrieves (ms) | {ci(pc[OURS]['mean_dTTFA_ms_if_retrieves'], 0, False)} | "
          f"{ci(pc[B9P]['mean_dTTFA_ms_if_retrieves'], 0, False)} | |", ""]
    L += ["2×2 by decision (R = retrieves, S = skips); contribution = block sum / all items, so the four "
          "contributions add up to Δ_C:", "",
          "| Ours-S / B9+ | n (share) | mean Δ_i [95% CI] | contribution to Δ_C [95% CI] | of which ΔQ | "
          "of which latency | of which compute |", "|---|---|---|---|---|---|---|"]
    for name, blk in d["two_by_two"].items():
        lab = name.replace("ours_", "").replace("__b9plus_", " / ")
        if blk["n"] == 0:
            L.append(f"| {lab} | 0 | — | — | — | — | — |")
            continue
        L.append(f"| {lab} | {blk['n']:,} ({blk['share']:.3f}) | {ci(blk['mean_delta'])} | "
                 f"{ci(blk['contribution_to_delta_C'])} | {f(blk['contribution_dQ']['estimate'])} | "
                 f"{f(blk['contribution_dLat']['estimate'])} | {f(blk['contribution_dComp']['estimate'])} |")
    L.append("")


def md_model(L, a):
    m = a["A2_model_vs_realised_accuracy"]
    L += ["Accuracy difference the frozen objective expected (each controller's own belief b and modelled ρ at its "
          f"dispatch; quality term b·g₊·(1 − ρ) − (1 − b)·g₋ with g₊ = {m['g_plus']:.3f}, g₋ = {m['g_minus']:.3f}) "
          "vs the realised one:", "",
          "| items | n | model ΔQ | realised ΔQ | realised − model | mean modelled ρ Ours / B9+ |",
          "|---|---|---|---|---|---|"]
    for name in ("all_items", "both_retrieve_ours_earlier"):
        r = m.get(name)
        if not r:
            continue
        L.append(f"| {name} | {r['n']:,} | {ci(r['model_dQ'], 3)} | {ci(r['realised_dQ'], 3)} | "
                 f"{ci(r['realised_minus_model'], 3)} | {f(r['mean_rho_model_ours'], 2, False)} / "
                 f"{f(r['mean_rho_model_b9plus'], 2, False)} |")
    L.append("")


def md_timing(L, a):
    t = a["B_timing"]
    L += ["| | Ours-S | B9+ |", "|---|---|---|"]
    q = {c: t[c]["dispatch_minus_speech_end_ms"] for c in CONTROLLERS}
    fr = {c: t[c]["fraction_of_question_heard"] for c in CONTROLLERS}

    def qs(x, nd=0):
        if x is None:
            return "—"
        return (f"{x['p10']:.{nd}f} / {x['p25']:.{nd}f} / {x['median']:.{nd}f} / {x['p75']:.{nd}f} / "
                f"{x['p90']:.{nd}f}")
    L += [f"| retrievals (n) | {t[OURS]['n_retrievals']:,} | {t[B9P]['n_retrievals']:,} |",
          f"| dispatch − end of speech, ms: p10 / p25 / median / p75 / p90 | {qs(q[OURS])} | {qs(q[B9P])} |",
          f"| mean dispatch − end of speech, ms | {ci(t[OURS]['mean_dispatch_minus_speech_end_ms'], 0)} | "
          f"{ci(t[B9P]['mean_dispatch_minus_speech_end_ms'], 0)} |",
          f"| fraction of question heard: p10 / p25 / median / p75 / p90 | {qs(fr[OURS], 2)} | {qs(fr[B9P], 2)} |",
          f"| share dispatched before end of speech | {ci(t[OURS]['share_before_speech_end'], 3, False)} | "
          f"{ci(t[B9P]['share_before_speech_end'], 3, False)} |",
          f"| share on an early partial (vs end-of-turn step) | {ci(t[OURS]['share_on_early_partial'], 3, False)} | "
          f"{ci(t[B9P]['share_on_early_partial'], 3, False)} |", ""]
    v = t["ours_vs_b9plus_decision_step"]
    bd = t[B9P]["decision_step_all_items"]
    L += [f"B9+'s decision step over all items (its tuned rule: `{json.dumps(a['b9plus_timing_rule'])}`): "
          f"median {bd['decide_minus_speech_end_ms']['median']:.0f} ms from end of speech, median fraction heard "
          f"{bd['fraction_heard']['median']:.2f}, share at the end-of-turn step "
          f"{ci(bd['share_at_end_of_turn_step'], 3, False)}.", "",
          f"Of Ours-S's {v['n']:,} dispatches, the share made at an EARLIER step than B9+'s decision step on the same "
          f"item: {ci(v['share_ours_earlier'], 3, False)}; same step {ci(v['share_same_step'], 3, False)}; later "
          f"{ci(v['share_ours_later'], 3, False)} (median Ours − B9+ = {v['ours_minus_b9plus_ms']['median']:.0f} ms).",
          ""]
    b = t["both_retrieve"]
    if b["n"]:
        L += [f"Items where both retrieve (n = {b['n']:,}): Ours-S dispatches earlier on "
              f"{ci(b['share_ours_earlier'], 3, False)}, at the same step on {ci(b['share_same_step'], 3, False)}; "
              f"same retrieved passages (query variant) on {ci(b['share_same_query_variant'], 3, False)}.", "",
              "| both retrieve, by step | n | mean Δ_i [95% CI] | contribution to Δ_C [95% CI] | mean ΔQ | "
              "mean latency part | hit@3 Ours − B9+ | mean fraction heard Ours / B9+ |",
              "|---|---|---|---|---|---|---|---|"]
        for name, blk in b["by_relative_step"].items():
            if blk["n"] == 0:
                continue
            L.append(f"| {name} | {blk['n']:,} | {ci(blk['mean_delta'])} | {ci(blk['contribution_to_delta_C'])} | "
                     f"{f(blk['mean_dQ']['estimate'])} | {f(blk['mean_latency_part']['estimate'])} | "
                     f"{ci(blk.get('hit3_ours_minus_b9plus'), 3)} | {blk['mean_fraction_heard_ours']:.2f} / "
                     f"{blk['mean_fraction_heard_b9plus']:.2f} |")
        L.append("")


def md_quality(L, a):
    c = a["C_partial_quality"]
    L += ["| | Ours-S | B9+ |", "|---|---|---|"]
    for key, lab in (("share_revised_of_all_retrievals", "share of retrievals on a REVISED partial"),
                     ("share_revised_excl_last_of_all_retrievals", "… REVISED-EXCL-LAST"),
                     ("share_revised_of_early_retrievals", "share revised among early-partial retrievals"),
                     ("share_revised_excl_last_of_early_retrievals", "… REVISED-EXCL-LAST"),
                     ("hit3_at_dispatch", "hit@3 at dispatch (all retrievals)"),
                     ("recall3_at_dispatch", "recall@3 at dispatch")):
        L.append(f"| {lab} | {ci(c[OURS][key], 3, False)} | {ci(c[B9P][key], 3, False)} |")
    for ctrl_key, lab in (("hit3_at_dispatch", "early-partial retrievals: hit@3 at dispatch"),
                          ("hit3_final_transcript", "… hit@3 of the final-transcript retrieval, same items"),
                          ("hit3_dispatch_minus_final", "… difference (dispatch − final)")):
        sg = ctrl_key == "hit3_dispatch_minus_final"
        L.append(f"| {lab} | {ci(c[OURS]['early_only'][ctrl_key], 3, sg)} | "
                 f"{ci(c[B9P]['early_only'][ctrl_key], 3, sg)} |")
    L.append(f"| … n early retrievals; relative hit@3 loss 1 − hit(dispatch)/hit(final) | "
             f"{c[OURS]['early_only']['n']:,}; {f(c[OURS]['early_only']['relative_hit3_loss_1_minus_ratio'], 3, False)} | "
             f"{c[B9P]['early_only']['n']:,}; {f(c[B9P]['early_only']['relative_hit3_loss_1_minus_ratio'], 3, False)} |")
    L.append(f"| … modelled ρ at dispatch (frozen config; a model value, not measured) | "
             f"{f(c[OURS]['early_only']['mean_rho_model'], 3, False)} | {f(c[B9P]['early_only']['mean_rho_model'], 3, False)} |")
    L.append(f"| … Q(dispatch) − Q(final-transcript answer cell), items where that cell exists (n) | "
             f"{ci(c[OURS]['early_only']['Q_dispatch_minus_Q_final_cell'], 3)} "
             f"({c[OURS]['early_only']['final_answer_cell_available']:,}) | "
             f"{ci(c[B9P]['early_only']['Q_dispatch_minus_Q_final_cell'], 3)} "
             f"({c[B9P]['early_only']['final_answer_cell_available']:,}) |")
    L.append("")
    for key, lab in (("revised", "REVISED"), ("revised_excl_last", "REVISED-EXCL-LAST")):
        blk = c["ours_by_partial_status"][key]
        L += [f"**Items stratified by Ours-S's decision ({lab} definition).** Contributions add up to Δ_C.", "",
              "| Ours-S stratum | n (share) | mean Δ_i [95% CI] | contribution to Δ_C [95% CI] | mean ΔQ | "
              "mean latency part | mean compute part | B9+ retrieval rate | Ours-S acc. | B9+ acc. | "
              "fraction heard Ours / B9+ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in STRATA:
            r = blk["strata"][s]
            if r["n"] == 0:
                L.append(f"| {s} | 0 | — | — | — | — | — | — | — | — | — |")
                continue
            L.append(f"| {s} | {r['n']:,} ({r['share_of_items']:.3f}) | {ci(r['mean_delta'])} | "
                     f"{ci(r['contribution_to_delta_C'])} | {f(r['mean_dQ']['estimate'])} | "
                     f"{f(r['mean_dLat']['estimate'])} | {f(r['mean_dComp']['estimate'])} | "
                     f"{f(r['b9plus_retrieval_rate']['estimate'], 3, False)} | {f(r['mean_Q_ours']['estimate'], 3, False)} | "
                     f"{f(r['mean_Q_b9plus']['estimate'], 3, False)} | {f(r['mean_fraction_heard_ours'], 2, False)} / "
                     f"{f(r['mean_fraction_heard_b9plus'], 2, False)} |")
        L += ["", f"* mean Δ_i, early_revised − early_unrevised: {ci(blk['mean_delta_early_revised_minus_early_unrevised'])}; "
              f"early_revised − all other items: {ci(blk['mean_delta_early_revised_minus_all_other_items'])}.",
              "* Ours-S accuracy by stratum: " + "; ".join(f"{s} {ci(v, 3, False)}" for s, v in blk["ours_accuracy"].items())
              + f"; revised − unrevised {ci(blk['ours_accuracy_revised_minus_unrevised'], 3)}.",
              "* Ours-S hit@3 by stratum: " + "; ".join(f"{s} {ci(v, 3, False)}" for s, v in blk["ours_hit3"].items())
              + f"; revised − unrevised {ci(blk['ours_hit3_revised_minus_unrevised'], 3)}."]
        L.append("* Items where both retrieve, by Ours-S stratum (paired): " + "; ".join(
            f"{s}: n = {blk['strata'][s]['both_retrieve']['n']:,}, hit@3 Ours−B9+ "
            f"{ci(blk['strata'][s]['both_retrieve'].get('hit3_ours_minus_b9plus'), 3)}, ΔQ "
            f"{ci(blk['strata'][s]['both_retrieve'].get('Q_ours_minus_b9plus'), 3)}"
            for s in ("early_revised", "early_unrevised", "end_of_turn")
            if blk["strata"][s]["n"] and blk["strata"][s]["both_retrieve"]["n"]) + ".")
        L.append("")
    L += ["**Revised vs unrevised at similar fraction heard** (Ours-S early-partial retrievals; revision and "
          "earliness co-vary, so this compares within bins of the fraction of the question heard at dispatch):", "",
          "| definition | fraction heard | n revised / unrevised | Ours-S acc. revised − unrevised | "
          "hit@3 revised − unrevised | mean Δ_i revised − unrevised |", "|---|---|---|---|---|---|"]
    for key, lab in (("revised", "REVISED"), ("revised_excl_last", "EXCL-LAST")):
        for r in c["ours_revised_vs_unrevised_by_fraction_heard"][key]:
            lo, hi = r["fraction_heard_bin"]
            close = "]" if hi >= 1.0 else ")"
            L.append(f"| {lab} | [{lo:.2f}, {hi:.2f}{close} | {r['n_revised']:,} / "
                     f"{r['n_unrevised']:,} | {ci(r['Q_revised_minus_unrevised'], 3)} | "
                     f"{ci(r['hit3_revised_minus_unrevised'], 3)} | {ci(r['mean_delta_revised_minus_unrevised'])} |")
    L.append("")


def write_md(doc, summary_lines):
    d1 = doc["d1_open"]
    L = ["# Why Ours-S loses to B9+ under COMMIT on d1_open (mechanism, exploratory)", "",
         f"**Status: {STATUS}.** Generated by `python -m analysis.v2_mechanism_d1` "
         f"(`docs/research_v2/mechanism_v2.json`). The frozen evaluator (`simulator/e3v2_lazy.frozen_records`) "
         "was re-run offline for Ours-v2[COMMIT] (Ours-S) and B9+ on the frozen bundles; outcomes come from the "
         "existing Phase A test store; nothing was generated or graded and no judge was called.", "",
         f"**Gate passed:** the recomputed per-item Δ[COMMIT] equals `contrasts_test.json` for every cell and item "
         f"(d1_open: {doc['gate']['d1_open']['cells_checked']} cells, {doc['gate']['d1_open']['items_checked']:,} "
         f"item-cells, max |diff| {doc['gate']['d1_open']['max_abs_diff']:.1e}); the three-part decomposition sums "
         f"to the frozen per-item Δ to ≤ 1e-12. Bootstrap: paired item bootstrap of `analysis/v2_test_d1.py` "
         f"(B = {B:,}, seed {SEED['d1_open']}); the primary Δ_C CI reproduces the frozen result "
         f"({doc['bootstrap_check']['reproduced']}).", "",
         "## Summary", ""] + summary_lines + [""]
    a = d1["asr_primary"]
    L += [f"## 1. Decomposition of Δ_C, d1_open primary cell (ASR, {a['profile']}, λ_l = {a['lambda_l']:g}, "
          f"λ_c = {a['lambda_c']:g})", ""]
    md_decomposition(L, a)
    md_model(L, a)
    L += ["## 2. Dispatch timing (items where each controller retrieves)", "",
          "Dispatch time = the step's availability time; end of speech = the last word's end; fraction heard = "
          "share of the question's words fully spoken at the dispatched partial's audio time (1.0 at the "
          "end-of-turn step).", ""]
    md_timing(L, a)
    L += ["## 3. Partial quality at dispatch", "",
          "REVISED = the dispatched partial's words are not a prefix of the final ASR transcript's "
          "(`analysis/v2_waste_measured.py` definitions); REVISED-EXCL-LAST ignores its last word. A final-"
          "transcript dispatch counts as unrevised. hit@3 = any gold document among the 3 retrieved (Phase A "
          "retrieval record of that dispatch point).", ""]
    md_quality(L, a)
    t = d1["text150"]
    L += [f"## 4. Contrast: text150 (revision-free prefixes; {t['profile']}, its 0.65 price λ_l = {t['lambda_l']:g})", ""]
    md_decomposition(L, t)
    md_model(L, t)
    md_timing(L, t)
    d2 = doc.get("d2_support")
    if d2:
        L += [f"## 5. d2_support ({D2_STATUS})", ""]
        for name, blk in d2["cells"].items():
            L += [f"### {name} ({blk['profile']}, λ_l = {blk['lambda_l']:g})", ""]
            md_decomposition(L, blk)
            md_model(L, blk)
            md_timing(L, blk)
    L += ["## What these data do not show", "",
          "* The analysis is post hoc and exploratory; its strata are defined by the controllers' own decisions, "
          "so differences between strata are associations, not effects of revision or timing.",
          "* Accuracy differences on items where both controllers retrieve come from different passages "
          "(different dispatch points), but the answer cells were generated once per passage list and seed; no "
          "counterfactual answer was generated for this analysis.",
          "* The final-transcript answer cell exists only where some controller in the frozen plan landed on it "
          "(lazy evaluation); here it existed for every early retrieval analysed (counts in the table), but it "
          "was generated because of the frozen plan, not for this analysis.",
          "* Items whose partials get revised may be harder for ASR and retrieval in general (B9+ is also less "
          "accurate on them), so accuracy by revision status mixes partial quality with item difficulty; the "
          "paired Δ_i is the cleaner comparison.",
          "* The model-vs-realised comparison uses each controller's own belief at a different step and the frozen "
          "quality term (ρ scales g₊ only); a gap can come from the dev ρ table, the belief calibration or that "
          "functional form, and these data do not tell which.",
          "* d2_support results are judge-graded (Deviation D3: judge-validation gate not passed) and exploratory; "
          "partial-quality analyses (section 3) were run on d1_open only.", ""]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def build_summary(doc):
    """Plain-language summary; every number is read from `doc`."""
    a = doc["d1_open"]["asr_primary"]
    d = a["A_decomposition"]
    p = d["parts"]
    pc = d["per_controller"]
    t = a["B_timing"]
    c = a["C_partial_quality"]
    tx = doc["d1_open"]["text150"]["A_decomposition"]
    return summary_text(doc, a, d, p, pc, t, c, tx)


def summary_text(doc, a, d, p, pc, t, c, tx):
    """Six plain-language sentences; every number is read from the computed results."""
    def cs(x, nd=3):
        return ci(x, nd)

    dd = d["differences_ours_minus_b9plus"]
    rs = d["two_by_two"]["ours_R__b9plus_S"]
    br = t["both_retrieve"]
    e, same = br["by_relative_step"]["ours_earlier"], br["by_relative_step"]["same_step"]
    n_all = d["n_items"]
    rev = c["ours_by_partial_status"]["revised"]
    rel = c["ours_by_partial_status"]["revised_excl_last"]
    bins = c["ours_revised_vs_unrevised_by_fraction_heard"]

    def n_excl(rows):
        return sum(1 for r in rows if r["mean_delta_revised_minus_unrevised"] is not None
                   and (r["mean_delta_revised_minus_unrevised"]["ci95"][1] < 0
                        or r["mean_delta_revised_minus_unrevised"]["ci95"][0] > 0))
    m = a["A2_model_vs_realised_accuracy"]["both_retrieve_ours_earlier"]
    t150 = doc["d1_open"]["text150"]
    tb = t150["B_timing"]["both_retrieve"]
    return [
        f"In the primary d1_open cell, Ours-S's deficit against B9+ (Δ_C {cs(d['delta_C'], 4)}) is an accuracy "
        f"deficit: the accuracy part is {cs(p['accuracy_dQ'], 4)}, partly offset by a latency gain of "
        f"{cs(p['latency_minus_lambda_l_ddTTFA'], 4)} (its answers start {-dd['dTTFA_ms']['estimate']:.0f} ms sooner "
        f"on average) and a compute cost of {cs(p['compute_minus_lambda_c_dn_ret'], 4)} (it retrieves on "
        f"{pc[OURS]['retrieval_rate']['estimate']:.1%} of items vs {pc[B9P]['retrieval_rate']['estimate']:.1%}).",
        f"Practically all of the deficit sits on the {e['n']:,} items ({e['n'] / n_all:.0%}) where both controllers "
        f"retrieve but Ours-S dispatches at an earlier partial (contribution {cs(e['contribution_to_delta_C'], 4)}); "
        f"where both dispatch at the same step ({same['n']:,} items) the outcomes are identical, and the items on "
        f"which only Ours-S retrieves contribute {cs(rs['contribution_to_delta_C'], 4)}.",
        f"On those items Ours-S's query had heard on average {e['mean_fraction_heard_ours']:.0%} of the question "
        f"against {e['mean_fraction_heard_b9plus']:.0%} for B9+'s, found a gold passage in the top 3 less often "
        f"(hit@3 {cs(e['hit3_ours_minus_b9plus'])}) and answered correctly less often (accuracy "
        f"{e['mean_dQ']['estimate']:+.3f}), which outweighed its latency gain there "
        f"({e['mean_latency_part']['estimate']:+.3f} utility per item).",
        f"Ours-S also dispatched more often on partials that the ASR later revised "
        f"({c[OURS]['share_revised_of_all_retrievals']['estimate']:.1%} vs "
        f"{c[B9P]['share_revised_of_all_retrievals']['estimate']:.1%} of retrievals; "
        f"{c[OURS]['share_revised_excl_last_of_all_retrievals']['estimate']:.1%} vs "
        f"{c[B9P]['share_revised_excl_last_of_all_retrievals']['estimate']:.1%} ignoring the last word), and the "
        f"deficit is concentrated on those items (mean Δ_i {rev['strata']['early_revised']['mean_delta']['estimate']:+.3f} "
        f"vs {rev['strata']['early_unrevised']['mean_delta']['estimate']:+.3f} on unrevised early partials, difference "
        f"{cs(rev['mean_delta_early_revised_minus_early_unrevised'])}); but revision goes together with dispatching "
        f"early: ignoring the last word the difference shrinks to "
        f"{cs(rel['mean_delta_early_revised_minus_early_unrevised'])}, and within bins of the fraction of the question "
        f"heard it excludes zero in only {n_excl(bins['revised'])} of {len(bins['revised'])} bins "
        f"({n_excl(bins['revised_excl_last'])} of {len(bins['revised_excl_last'])} ignoring the last word), so these "
        f"data do not separate a revised query from a merely incomplete one.",
        f"The frozen objective anticipated only part of this cost: on the earlier-dispatch items its own quality "
        f"term (each controller's belief and dev-measured ρ at dispatch, mean ρ {m['mean_rho_model_ours']:.2f} vs "
        f"{m['mean_rho_model_b9plus']:.2f}) implied an accuracy difference of {cs(m['model_dQ'])}, against a realised "
        f"{cs(m['realised_dQ'])} (difference {cs(m['realised_minus_model'])}).",
        f"On text150, whose prefixes are never revised, the two controllers dispatch at the same step on "
        f"{tb['share_same_step']['estimate']:.1%} of the items where both retrieve and Δ_C is "
        f"{cs(t150['A_decomposition']['delta_C'], 4)} (Ours-S's extra retrievals buy accuracy "
        f"{t150['A_decomposition']['parts']['accuracy_dQ']['estimate']:+.4f}, offset by latency "
        f"{t150['A_decomposition']['parts']['latency_minus_lambda_l_ddTTFA']['estimate']:+.4f} and compute "
        f"{t150['A_decomposition']['parts']['compute_minus_lambda_c_dn_ret']['estimate']:+.4f}); all of this is a "
        f"post hoc description of one frozen test run, and it does not show that early dispatch or revision causes "
        f"the loss.",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--phase-a", required=True, help="the simulator/e3v2_phase_a directory holding d1_open/ and d2_support/")
    ap.add_argument("--e1-results", default=str(_REPO_ROOT / "llm" / "experiment1_v2" / "results"))
    ap.add_argument("--skip-d2", action="store_true")
    ap.add_argument("--cache", default=None, help="optional pickle of the extracted per-item data (written after "
                                                  "the gate passes; read instead of re-running the evaluator)")
    ap.add_argument("--raw-cache", default=None, help="optional directory: per-bundle captures are written there "
                                                      "after their gate passes and read back on a re-run")
    args = ap.parse_args(argv)
    global RAW_CACHE
    if args.raw_cache:
        RAW_CACHE = Path(args.raw_cache)
        RAW_CACHE.mkdir(parents=True, exist_ok=True)
    phase_a_root = Path(args.phase_a)
    datasets = ["d1_open"] + ([] if args.skip_d2 else ["d2_support"])
    cache = Path(args.cache) if args.cache else None
    if cache is not None and cache.exists():
        with open(cache, "rb") as fh:
            res = pickle.load(fh)
        print(f"read extracted data from {cache} (gate passed when it was written)")
    else:
        res = {ds: run_dataset(ds, phase_a_root, Path(args.e1_results)) for ds in datasets}
        if cache is not None:
            with open(cache, "wb") as fh:
                pickle.dump(res, fh)
    from analysis.v2_test_d1 import bootstrap_counts
    from analysis.v2_test_final import cluster_counts
    doc = {"_comment": STATUS + ". Ours-S = ours_v2 under COMMIT (the joint controller); B9+ = b9plus_v2 (the "
                       "tuned decomposed rival, D*[COMMIT]). See the module docstring for definitions.",
           "status": STATUS, "generated_by": "python -m analysis.v2_mechanism_d1",
           "generated_at_utc": e1common.now_utc(), "git_commit": e1common.git_commit(),
           "split": SPLIT, "primary_cell": {"condition": ASR, "profile": PROFILE, "belief_quantile": QUANTILE,
                                            "price": FROZEN_PRIMARY},
           "gate": {ds: {**r["gate"], "rule": "per-item Delta[COMMIT] == contrasts_test.json to 1e-12, every cell"}
                    for ds, r in res.items()},
           "inputs": {ds: r["inputs"] for ds, r in res.items()},
           "bootstrap": {"B": B, "seeds": SEED, "unit": {"d1_open": "item", "d2_support": "pair_id cluster"},
                         "source": "analysis/v2_test_d1.py bootstrap_counts (d1_open); analysis/v2_test_final.py "
                                   "cluster_counts (d2_support)"}}
    for ds in datasets:
        r = res[ds]
        if ds == "d1_open":
            assert len(r["universe"]) == 3734
            counts = bootstrap_counts(len(r["universe"]))
        else:
            counts = cluster_counts(r["universe"], r["cluster_of"], SEED[ds])
        quality = {}
        for bp in r["inputs"]["bundles"]:
            bpath = Path(bp)
            bdl = e1common.load_json(bpath if bpath.is_absolute() else phase_a_root.resolve().parent.parent / bpath)
            quality[bdl["condition"]] = bdl["overrides"]["quality"]
        blk = analyse(r, counts, with_quality=(ds == "d1_open"), quality=quality)
        doc[ds] = blk if ds == "d1_open" else {"status": D2_STATUS, "cells": blk}
        del counts
    # bootstrap identity check against the frozen final results
    fin = e1common.load_json(_REPO_ROOT / "docs" / "research_v2" / "v2_test_results_final.json")
    ref = fin["per_dataset"]["d1_open"]["primary"]["delta_commit"]
    got = doc["d1_open"]["asr_primary"]["A_decomposition"]["delta_C"]
    doc["bootstrap_check"] = {"frozen_d1_primary_delta_commit": {"estimate": ref["estimate"], "ci95": ref["ci95"]},
                              "here": {"estimate": got["estimate"], "ci95": got["ci95"]},
                              "reproduced": bool(abs(ref["estimate"] - got["estimate"]) < 1e-12
                                                 and all(abs(x - y) < 1e-12 for x, y in zip(ref["ci95"], got["ci95"])))}
    if "d2_support" in doc:
        ref2 = fin["per_dataset"]["d2_support"]["primary"]["delta_commit"]
        got2 = doc["d2_support"]["cells"]["asr_primary"]["A_decomposition"]["delta_C"]
        doc["bootstrap_check"]["d2_support"] = {
            "frozen": {"estimate": ref2["estimate"], "ci95": ref2["ci95"]},
            "here": {"estimate": got2["estimate"], "ci95": got2["ci95"]},
            "reproduced": bool(abs(ref2["estimate"] - got2["estimate"]) < 1e-12
                               and all(abs(x - y) < 1e-12 for x, y in zip(ref2["ci95"], got2["ci95"])))}
    summary = build_summary(doc)
    doc["summary"] = summary
    e1common.atomic_write_json(OUT_JSON, doc)
    write_md(doc, summary)
    print(json.dumps(doc["bootstrap_check"], indent=1))
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
