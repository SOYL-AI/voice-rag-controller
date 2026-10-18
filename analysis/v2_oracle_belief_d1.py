"""
v2_oracle_belief_d1.py -- EXPLORATORY, POST HOC ablation on d1_open TEST:
does joint control (Ours-v2, "Ours-S" in the paper) beat the decomposed
rival B9+ when the belief is good? Not a pre-registered analysis; nothing
here changes any confirmatory result (PREREG_V2 is untouched).

    PYTHONUTF8=1 VOICE_RAG_MODEL=mistral7b_v03 python -m analysis.v2_oracle_belief_d1 \
        --phase-a <repo with the Phase A tree>/simulator/e3v2_phase_a [--plan]

WHAT IS RE-RUN. Ours-v2 and B9+ (b9plus_v2 with its FROZEN dev-tuned timing
rule) under both architectures (COMMIT; SPECULATIVE with the frozen
D*[SPEC] = b9plus_v2 for d1), on the frozen primary cell (primary ASR
condition, profile rewrite_faiss, belief quantile 0.65: lambda_l =
0.17305735434823866, lambda_c = 0.05) and, for contrast, on text150 at its own
frozen 0.65-quantile rewrite_faiss price. Every frozen input (bundle overrides:
g+/g-, rho table, timing constants, prices, remaining-speech predictor,
latency model, B9+ timing rule and policy constants, seeds, common random
numbers) is kept. Only the BELIEF changes, in four variants:

  frozen   the frozen inputs (the gate, below)
  oracle   b_k = the empirical need label y (E1-v2: closed-book wrong in >= 3
           of 5 draws, alias-graded) at EVERY step, from the first partial
  sharpen  b_k = p0 + f_k (y - p0): from the dev prior p0 (dev base rate of the
           empirical need label) towards the label as speech is heard; f_k =
           the CAUSAL fraction-heard estimate e/(e + R_hat) the controllers
           already compute (estimated_frac; 1 at the end of turn), never the
           reference word fraction. NOT calibrated at intermediate steps: b_k
           is a deterministic function of y, so P(y = 1 | b_k) is 0 or 1
  reveal   (added after `sharpen` turned out to be miscalibrated; same spirit,
           calibrated) b_k = p0 until f_k >= u_i, then y; u_i ~ U(0, 1) per item
           from a fixed hash of the item id, independent of y, so
           P(y = 1 | b_k = p0) = p0 and the belief is a martingale that
           sharpens (reveals the label) as speech is heard; b = y at the end

How the belief is replaced (no frozen file is edited): each turn's
signals["model"] is replaced by one prefix entry available at t = 0 and one
end-of-turn entry, both carrying y, and the controllers' calibration object
is replaced by OracleCalibration (identity on y for `oracle`; p0 + f (y - p0)
for `sharpen`; y if f >= u_i else p0 for `reveal`, the entry then carrying
2 y + u_i; f being the frac belief_from_obs passes, i.e. the causal estimate
since the frozen config has calibration.frac_source = predicted).
Everything else that reads the belief reads it through belief_from_obs, so
the two controllers see the same b_k at every step.

What is derived from the belief and refitted (dev split only): Ours-v2's
belief-transition (lookahead) model. The frozen model describes the real
belief's movement and would be incoherent for the oracle beliefs. For each
oracle variant it is refitted with the frozen fitter
(_objective_v2.fit_belief_transitions: 10 bins, phase edges 0/.34/.67/1,
alpha 1) on DEV sequences of that variant's belief, built from the frozen dev
signal rows of both conditions (analysis/signals_*_dev.jsonl, the rows the
frozen fit used, with their out-of-fold remaining-speech predictions) exactly
as the controller sees them: one point (f_k, b_k) per decision step of
build_steps, post-speech re-transcription partials dropped (as the frozen
belief_sequences does), ending at the end-of-turn belief. Phase = the causal
f_k (the frozen fit used the reference word fraction for the phase; the
controller looks phases up with the causal estimate, so the refit uses the
latter). For `oracle` the fit is the identity on {0, 1} (the belief never
moves), for `sharpen` it encodes the deterministic drift towards the label,
for `reveal` the per-phase probability that the label is revealed.
Nothing else is refitted: B9+'s frozen timing rule and policy constants
are timing-only (r_below / r_threshold rules never read the prior belief),
and prices, g+/g- and rho are not belief inputs.

GATE: the `frozen` variant of this re-implementation must reproduce
contrasts_test.json's per-item Delta[COMMIT] and Delta[SPECULATIVE] at both
cells to 1e-12, else the script stops without writing results.

LAZY CELLS. A decision that lands on a (item, seed, condition, path,
query_variant) cell the frozen lazy plan never generated is detected. --plan
writes the list of such cells (all seeds) to
simulator/e3v2_oracle/d1_open/required_test_oracle.json and stops; the cells
are then generated with the frozen Phase A code path into that directory
(never into the frozen store) and passed back with --extra-store. The full
run refuses to score a missing cell.

Statistics: the paired item bootstrap of analysis/v2_test_d1.py (B = 10,000,
seed 20261017, the same item universe and resamples). The utility
decomposition is U = accuracy - lambda_l (TTFA - skip TTFA) - lambda_c
n_retrievals, per item averaged over the five seeds.

Writes docs/research_v2/oracle_belief_v2.json and oracle_belief_v2.md
(--write-md rewrites the .md from the JSON alone).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    BeliefCalibration,
    belief_from_obs,
    estimated_frac,
    fit_belief_transitions,
    load_e3v2_config,
)
from controller.baselines.b9plus_v2 import B9PlusV2  # noqa: E402
from controller.baselines.ours_v2 import OursV2  # noqa: E402
from controller.remaining_speech import RemainingSpeechPredictor  # noqa: E402
from controller.sequential_v2 import TurnCtx, build_steps  # noqa: E402
from llm.experiment1_v2 import common as e1common  # noqa: E402
from simulator.e3v2_decide import CellStore, _price_key, load_latency_model, run_phase_b  # noqa: E402

DS = "d1_open"
SPLIT = "test"
ASR = "piper-en_US-hfc_male-medium__150wpm__clean"
PROFILE = "rewrite_faiss"
QUANTILE = 0.65
PRIMARY = {"condition": ASR, "profile": PROFILE, "quantile": QUANTILE,
           "lambda_l": 0.17305735434823866, "lambda_c": 0.05}
CONTROLLERS = ("ours_v2", "b9plus_v2")
D_STAR = {"commit": "b9plus_v2", "speculative": "b9plus_v2"}  # frozen for d1
ARCHS = ("commit", "speculative")
VARIANTS = ("frozen", "oracle", "sharpen", "reveal")
ORACLE_VARIANTS = VARIANTS[1:]
ORACLE_FIELD = "oracle_need"
B = 10_000
SEED = 20261017  # analysis/v2_test_d1.py
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "oracle_belief_v2.json"
OUT_MD = _REPO_ROOT / "docs" / "research_v2" / "oracle_belief_v2.md"
NEW_DIR = _REPO_ROOT / "simulator" / "e3v2_oracle" / DS
REQUIRED = NEW_DIR / "required_test_oracle.json"
LABEL = "EXPLORATORY post hoc ablation (not pre-registered; not part of the pre-registered test)"


# ---------------------------------------------------------------------------
# Belief replacement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OracleCalibration(BeliefCalibration):
    """b = y (mode "oracle"), p0 + f (y - p0) (mode "sharpen") or, mode
    "reveal", y if f >= u else p0 (the entry carries x = 2 y + u, u in [0, 1));
    y = the entry's ORACLE_FIELD, f = the frac belief_from_obs passes (the
    causal e/(e+R_hat); 1 at the end of turn)."""

    mode: str = "oracle"
    prior: float = 0.5

    def apply(self, x: float, frac: float = 1.0) -> float:
        y = float(x)
        if self.mode == "oracle":
            return y
        if self.mode == "sharpen":
            f = min(1.0, max(0.0, float(frac)))
            return self.prior + f * (y - self.prior)
        if self.mode == "reveal":
            label = 1.0 if y >= 1.5 else 0.0
            u = y - 2.0 * label
            return label if float(frac) >= u else self.prior
        raise ValueError(self.mode)


def oracle_calibration(mode: str, prior: float) -> OracleCalibration:
    return OracleCalibration(signal=ORACLE_FIELD, kind="oracle", intercept=0.0, slope=1.0, frac_slope=0.0,
                             interaction=0.0, calibrated=False, source=f"EXPLORATORY oracle belief ({mode})",
                             mode=mode, prior=prior)


def reveal_u(item_id: str) -> float:
    """u_i ~ U(0, 1), a fixed hash of the item id (independent of the label)."""
    return int(hashlib.sha256(f"oracle-reveal|{item_id}".encode("utf-8")).hexdigest()[:13], 16) / 16 ** 13


def oracle_x(variant: str, y: bool, item_id: str) -> float:
    return 2.0 * float(bool(y)) + reveal_u(item_id) if variant == "reveal" else float(bool(y))


def oracle_signals(signals: dict, x: float) -> dict:
    """The turn's signals with the belief inputs replaced by the oracle value
    x: one prefix entry available from t = 0 (so every step has it) and the
    end-of-turn entry (read by build_steps' final step)."""
    s = dict(signals)
    s["model"] = {"available": True,
                  "uncertainty_by_prefix": [{"t_ms": 0.0, "available_ms": 0.0, "kind": "partial",
                                             ORACLE_FIELD: float(x)}],
                  "uncertainty": {"kind": "full", ORACLE_FIELD: float(x)}}
    return s


def dev_sequences(rows: List[dict], labels: Dict[str, bool], cfg: dict, cal: OracleCalibration) -> List[list]:
    """(f_k, b_k) per build_steps decision step, as the controller computes
    them; post-speech partials dropped (frozen belief_sequences rule)."""
    seqs = []
    for r in rows:
        y = labels.get(r["item_id"])
        if y is None:
            continue
        ctx = TurnCtx(item_id=r["item_id"], signals=oracle_signals(r["signals"], oracle_x(cal.mode, y, r["item_id"])),
                      condition=r["condition"],
                      transcript_final=r.get("transcript_final", ""))
        timing = r["signals"]["timing"]
        se = float(timing.get("speech_end_ms") or math.inf)
        parts = sorted([p for p in timing["partials"] if p.get("t_ms") is not None], key=lambda p: float(p["t_ms"]))
        seq = []
        for obs in build_steps(ctx, cfg):
            if not obs.is_final and float(parts[obs.partial_index]["t_ms"]) > se:
                continue
            b, _ = belief_from_obs(obs, cfg, cal)
            f = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
            seq.append((1.0 if obs.is_final else (0.0 if f is None else f), b))
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# The frozen evaluation, re-implemented for two controllers and one price
# ---------------------------------------------------------------------------
class MissingAwareStore(CellStore):
    """The frozen CellStore; a cell Phase A never generated is recorded (and
    answered with the placeholder, which is never scored: the run refuses)."""

    def __init__(self, records, order_sensitive=True, recording=None):
        super().__init__(records, order_sensitive, None)
        self.missing = set()

    def cell(self, item_id, seed, condition, path, variant):
        key = (item_id, int(seed), condition, path, variant)
        if key not in self.cells:
            self.missing.add(key)
            return dict(self.PLACEHOLDER)
        return self.cells[key]


def load_store(frozen: Path, extra: List[Path]) -> MissingAwareStore:
    records = e1common.read_jsonl(frozen, tolerate_partial_tail=False)
    keys = {(r["item_id"], int(r["seed"]), r["condition"], r["path"], r.get("query_variant", "none"))
            for r in records if r.get("kind") == "cell"}
    n_frozen_cells = len(keys)  # unique cells (two dispatch points with one passage list share a cell row)
    seen: Dict[tuple, bool] = {}
    for p in extra:
        for r in e1common.read_jsonl(p, tolerate_partial_tail=False):
            if r.get("kind") != "cell":
                continue  # retrievals / skip prompts come from the frozen store only
            k = (r["item_id"], int(r["seed"]), r["condition"], r["path"], r.get("query_variant", "none"))
            if k in keys:
                raise SystemExit(f"extra store {p} re-defines frozen cell {k}")
            # two dispatch points with one passage list give two identical rows (as in the frozen store)
            if k in seen and seen[k] != bool(r["answer_correct"]):
                raise SystemExit(f"extra store {p} has conflicting rows for {k}")
            seen[k] = bool(r["answer_correct"])
            records.append(r)
    store = MissingAwareStore(records)
    store.n_frozen_cells = n_frozen_cells
    store.n_extra_cells = len(store.cells) - n_frozen_cells
    return store


def generation_accounting(extra: List[Path]) -> dict:
    """What the new-cell Phase A run(s) cost, from their own manifest and
    answer records (the Phase A copy holds the frozen answers too; the new
    ones are those written by the runs in this directory's manifest)."""
    out = {"n_cells": 0, "n_answers_needed_all_seeds": 0, "n_answers_generated": 0, "answer_wall_s": 0.0,
           "output_tokens": 0, "prompt_chars": 0, "finish_reason": {}, "served_model": {}, "runs": []}
    for p in extra:
        d = p.parent
        req = e1common.load_json(d / "required_test_oracle.json")
        out["n_cells"] += req["n_cells"]
        out["n_answers_needed_all_seeds"] += req["n_answers_all_seeds"]
        man = e1common.load_json(d / f"manifest_{SPLIT}.json")
        windows = []
        for run in man["runs"]:
            st = (run.get("stages") or {}).get("answer") or {}
            out["n_answers_generated"] += int(st.get("done", 0))
            out["answer_wall_s"] += float(st.get("wall_s", 0.0))
            windows.append((run["started_at_utc"], run["finished_at_utc"]))
            out["runs"].append({"started_at_utc": run["started_at_utc"], "finished_at_utc": run["finished_at_utc"],
                                "git_commit": run.get("git_commit"), "stages": run.get("stages"),
                                "args": {k: run["args"].get(k) for k in ("stages", "concurrency", "required_cells")}})
        for r in e1common.read_jsonl(d / f"answers_{SPLIT}.jsonl"):
            if any(a <= r["created_at_utc"] <= b for a, b in windows):
                out["output_tokens"] += int(r["n_tokens"])
                out["prompt_chars"] += int(r["prompt_chars"])
                out["finish_reason"][r["finish_reason"]] = out["finish_reason"].get(r["finish_reason"], 0) + 1
                out["served_model"][r["served_model"]] = out["served_model"].get(r["served_model"], 0) + 1
    out["n_answers_reused_identical_key"] = out["n_answers_needed_all_seeds"] - out["n_answers_generated"]
    out["prompt_tokens_approx"] = round(out["prompt_chars"] / 4.103)  # Phase A's chars-per-token factor
    out["grading"] = "alias match only (d1_open scorer of record); no LLM judge or Azure call"
    out["cost_note"] = ("GPU cost was not metered: GPU time = this wall time x the number of serving containers, "
                        "which Phase A does not record (64 requests in flight).")
    return out


def condition_rows(bundle: dict, inp: dict) -> List[dict]:
    """frozen_records' signal construction (simulator/e3v2_lazy.py), verbatim."""
    from simulator.e3v2_dev_analysis import build_signals

    inp = dict(inp)
    inp["sa_lag_ms"] = float(bundle["sa_lag_ms"])
    cond = bundle["condition"]
    pred = RemainingSpeechPredictor.load(bundle["remaining_speech_predictor"])
    rhat = {}
    for item_id, tr in inp["traces"][cond].items():
        rhat[item_id] = []
        for p in tr["partials"]:
            q = pred.predict(p["text"], float(p["t_ms"]))
            rhat[item_id].append({"t_ms": float(p["t_ms"]), "available_ms": float(p.get("available_ms", p["t_ms"])),
                                  "remaining_ms": float(q["q50"]), "remaining_q10_ms": float(q["q10"]),
                                  "remaining_q90_ms": float(q["q90"])})
    return build_signals(inp, cond, rhat)


def cell_price(bundle: dict) -> dict:
    row = next(g for g in bundle["price_grid"][PROFILE] if g["quantile"] == QUANTILE and g["feasible"])
    return {"lambda_l": row["lambda_l"], "lambda_c": row["lambda_c"]}


def run_variant(variant: str, bundle: dict, rows: List[dict], labels: Dict[str, bool], gold: dict,
                store: MissingAwareStore, cal: Optional[OracleCalibration], transitions) -> List[dict]:
    from simulator.e3v2_dev_analysis import SEEDS

    cond = bundle["condition"]
    cfg = load_e3v2_config(overrides=bundle["overrides"])
    pr = cell_price(bundle)
    tuned = (bundle["tuned"].get(PROFILE) or {}).get(f"{pr['lambda_l']:g}") or {}
    b9p_params = {"timing_rule": tuned["b9plus_timing_rule"]} if tuned.get("b9plus_timing_rule") else {}
    if variant == "frozen":
        turns = [TurnCtx(item_id=r["item_id"], signals=r["signals"], condition=cond,
                         transcript_final=r["transcript_final"]) for r in rows]
    else:
        turns = [TurnCtx(item_id=r["item_id"],
                         signals=oracle_signals(r["signals"], oracle_x(variant, labels[r["item_id"]], r["item_id"])),
                         condition=cond, transcript_final=r["transcript_final"]) for r in rows]
    lm = load_latency_model(cfg)
    records: List[dict] = []
    for arch in ARCHS:
        arch_cfg = {**cfg, "architecture": arch}
        ours = OursV2(cfg=arch_cfg, transitions=None if variant == "frozen" else transitions)
        b9p = B9PlusV2(cfg=arch_cfg, **b9p_params)
        if variant != "frozen":
            ours.calibration = cal
            b9p.calibration = cal
        ours.name, b9p.name = "ours_v2", "b9plus_v2"
        records += run_phase_b(turns, store, {"ours_v2": ours, "b9plus_v2": b9p}, [pr], SEEDS, [PROFILE],
                               arch_cfg, lm, SPLIT, labels, gold)
    return records


# ---------------------------------------------------------------------------
# Per-item quantities
# ---------------------------------------------------------------------------
def per_item(records: List[dict]) -> dict:
    """(arch, controller) -> {"u","acc","lat","comp": {item: seed-mean}, "dec": {item: decision}}."""
    acc: dict = {}
    for r in records:
        lam_l, lam_c = float(r["prices"]["lambda_l"]), float(r["prices"]["lambda_c"])
        a = float(bool(r["outcome"]["answer_correct"]))
        lat = lam_l * (float(r["outcome"]["ttfa_ms"]) - float(r["skip_ttfa_ms"])) / 1000.0
        comp = lam_c * float(r["outcome"]["retrievals"])
        if abs((a - lat - comp) - float(r["realized_utility"])) > 1e-12:
            raise SystemExit("utility decomposition does not add up")
        d = acc.setdefault((r["architecture"], r["baseline"]), {})
        for k, v in (("u", float(r["realized_utility"])), ("acc", a), ("lat", lat), ("comp", comp),
                     ("ttfa_minus_skip_ms", float(r["outcome"]["ttfa_ms"]) - float(r["skip_ttfa_ms"])),
                     ("exposed_ms", float(r["exposed_latency_ms"]))):
            d.setdefault(k, {}).setdefault(r["item_id"], []).append(v)
        dec = (bool(r["decision"]["retrieve"]), r["decision"]["dispatch_partial_index"],
               r["decision"]["used"], r["dispatch_t_ms"])
        prev = d.setdefault("dec", {}).get(r["item_id"])
        if prev is not None and prev != dec:
            raise SystemExit("decision varies with the seed")
        d["dec"][r["item_id"]] = dec
    out = {}
    for key, d in acc.items():
        out[key] = {k: ({i: float(np.mean(v)) for i, v in d[k].items()} if k != "dec" else d[k]) for k in d}
    return out


def deltas(pi: dict) -> dict:
    out = {}
    for arch in ARCHS:
        a, b = pi[(arch, "ours_v2")]["u"], pi[(arch, D_STAR[arch])]["u"]
        out[arch] = {i: a[i] - b[i] for i in sorted(set(a) & set(b))}
    out["interaction"] = {i: out["commit"][i] - out["speculative"][i]
                          for i in sorted(set(out["commit"]) & set(out["speculative"]))}
    return out


# ---------------------------------------------------------------------------
# Bootstrap (analysis/v2_test_d1.py)
# ---------------------------------------------------------------------------
class Boot:
    def __init__(self, universe: List[str]):
        self.universe = universe
        self.uidx = {i: k for k, i in enumerate(universe)}
        rng = np.random.default_rng(SEED)
        n = len(universe)
        self.counts = np.zeros((B, n), dtype=np.float64)
        for b in range(B):
            self.counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)

    def __call__(self, values: Dict[str, float]) -> dict:
        x = np.zeros(len(self.universe))
        m = np.zeros(len(self.universe))
        for i, v in values.items():
            x[self.uidx[i]] = float(v)
            m[self.uidx[i]] = 1.0
        tot = self.counts @ m
        means = (self.counts @ x) / np.where(tot > 0, tot, np.nan)
        q = np.nanpercentile(means, [2.5, 97.5])
        return {"estimate": float(np.mean(list(values.values()))), "ci95": [float(q[0]), float(q[1])],
                "boot_se": float(np.nanstd(means, ddof=1)), "n_items": len(values),
                "p_one_sided_le_0": float(np.nanmean(means <= 0)), "p_one_sided_ge_0": float(np.nanmean(means >= 0))}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def controller_summary(d: dict, labels: Dict[str, bool]) -> dict:
    dec = d["dec"]
    n = len(dec)
    disp = [i for i, x in dec.items() if x[0]]
    used = [i for i, x in dec.items() if x[0] and x[2] is not False]
    early = [i for i in disp if dec[i][1] is not None]
    need = [i for i in dec if labels.get(i)]
    noneed = [i for i in dec if not labels.get(i)]
    m = lambda key: float(np.mean(list(d[key].values())))  # noqa: E731
    return {
        "n_items": n,
        "utility": m("u"), "accuracy": m("acc"), "latency_term": m("lat"), "compute_term": m("comp"),
        "mean_ttfa_minus_skip_ms": m("ttfa_minus_skip_ms"), "mean_exposed_retrieval_ms": m("exposed_ms"),
        "dispatch_rate": len(disp) / n,
        "uses_passages_rate": len(used) / n,
        "discarded_rate": (len(disp) - len(used)) / n,
        "early_dispatch_rate": len(early) / n,
        "early_share_of_dispatches": (len(early) / len(disp)) if disp else None,
        "uses_passages_rate_need": (sum(1 for i in used if labels.get(i)) / len(need)) if need else None,
        "uses_passages_rate_no_need": (sum(1 for i in used if not labels.get(i)) / len(noneed)) if noneed else None,
        "dispatch_rate_need": (sum(1 for i in disp if labels.get(i)) / len(need)) if need else None,
        "dispatch_rate_no_need": (sum(1 for i in disp if not labels.get(i)) / len(noneed)) if noneed else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--phase-a", help="the frozen simulator/e3v2_phase_a directory holding d1_open/")
    ap.add_argument("--e1-results", default=str(_REPO_ROOT / "llm" / "experiment1_v2" / "results"))
    ap.add_argument("--extra-store", action="append", default=[],
                    help="store JSONL(s) of newly generated cells (simulator/e3v2_oracle/d1_open/...)")
    ap.add_argument("--plan", action="store_true", help="write the missing-cell list and stop")
    ap.add_argument("--write-md", action="store_true", help="only (re)write the .md from the JSON")
    args = ap.parse_args(argv)
    if args.write_md:
        write_md(e1common.load_json(OUT_JSON))
        return 0
    if not args.phase_a:
        raise SystemExit("--phase-a is required")

    from simulator.e3v2_dev_analysis import load_inputs

    phase_a = Path(args.phase_a) / DS
    root = Path(args.phase_a).resolve().parent.parent  # bundle paths in contrasts are relative to it
    doc_c = e1common.load_json(phase_a / f"contrasts_{SPLIT}.json")
    bundles = []
    for bp in doc_c["bundles"]:
        p = Path(bp)
        bundles.append(e1common.load_json(p if p.is_absolute() else root / p))
    bundles.sort(key=lambda b: b["condition"] != ASR)  # primary first
    frozen = {}
    for b in bundles:
        pk = _price_key(cell_price(b))
        row = next(r for r in doc_c["contrasts"][b["condition"]] if r["latency_profile"] == PROFILE and r["price"] == pk)
        frozen[b["condition"]] = row

    # ---- labels, dev prior, refitted transitions (dev only) ---------------
    e1dir = Path(args.e1_results) / DS / "mistral7b_v03"
    lab = {s: {i: v.get("empirical_need") for i, v in e1common.load_json(e1dir / f"labels_{s}.json")["items"].items()}
           for s in ("dev", "test")}
    if any(v is None for s in lab for v in lab[s].values()):
        raise SystemExit("undetermined empirical-need labels: decide how to treat them before running")
    dev_labels = {i: bool(v) for i, v in lab["dev"].items()}
    test_labels = {i: bool(v) for i, v in lab["test"].items()}
    p0 = float(np.mean(list(dev_labels.values())))
    est = e1common.load_json(Path(bundles[0]["overrides"]["estimates_artifact"]))
    cals = {v: oracle_calibration(v, p0) for v in ORACLE_VARIANTS}
    dev_rows = {}
    for b in bundles:
        slug = Path(b["remaining_speech_predictor"]).stem.replace("remaining_speech_", "")
        path = Path(b["remaining_speech_predictor"]).parent / f"signals_{slug}_dev.jsonl"
        dev_rows[b["condition"]] = (e1common.read_jsonl(path, tolerate_partial_tail=False), path)
    transitions, trans_meta = {}, {}
    for v in ORACLE_VARIANTS:
        seqs = []
        for b in bundles:
            cfg_b = load_e3v2_config(overrides=b["overrides"])
            seqs += dev_sequences(dev_rows[b["condition"]][0], dev_labels, cfg_b, cals[v])
        transitions[v] = fit_belief_transitions(seqs, n_bins=10, provenance={
            "kind": "fitted", "generated_by": "python -m analysis.v2_oracle_belief_d1 (EXPLORATORY)",
            "dataset": DS, "split": "dev", "belief": v, "n_sequences": len(seqs)})
        moved = sum(1 for s in seqs if len({round(x[1], 12) for x in s}) > 1)
        trans_meta[v] = {"split": "dev", "n_sequences": len(seqs), "n_sequences_where_belief_moves": moved,
                         "n_points": sum(len(s) for s in seqs),
                         "sources": {c: {"path": str(p), "sha256": e1common.sha256_file(p)}
                                     for c, (_, p) in dev_rows.items()},
                         "fitter": "controller.baselines._objective_v2.fit_belief_transitions(n_bins=10, "
                                   "phase_edges=(0, .34, .67, 1), alpha=1)",
                         "model": transitions[v].to_json()}
        print(f"refit transitions [{v}] on dev: {len(seqs)} sequences, belief moves in {moved}", flush=True)

    # ---- the test evaluation ---------------------------------------------
    print("loading test inputs and the Phase A test store ...", flush=True)
    inp = load_inputs(DS, Path(args.e1_results), phase_a, SPLIT)
    store = load_store(phase_a / f"store_{SPLIT}.jsonl", [Path(p) for p in args.extra_store])
    print(f"store: {store.n_frozen_cells} frozen cells + {store.n_extra_cells} new cells", flush=True)
    results, missing_by = {}, {}
    for b in bundles:
        cond = b["condition"]
        rows = condition_rows(b, inp)
        gold = {i: list((inp["items"].get(i) or {}).get("gold_doc_ids") or []) for i in inp["traces"][cond]}
        for v in VARIANTS:
            store.missing = set()
            recs = run_variant(v, b, rows, test_labels, gold, store, cals.get(v), transitions.get(v))
            missing_by[(v, cond)] = set(store.missing)
            print(f"{v:8s} {cond[:24]:24s}: {len(recs)} records, missing cells {len(store.missing)}", flush=True)
            if v == "frozen":
                if store.missing:
                    raise SystemExit("GATE FAILED: the frozen variant lands on ungenerated cells")
                pi = per_item(recs)
                dl = deltas(pi)
                ref = frozen[cond]["per_item"]
                for mine, theirs in (("commit", "delta_commit"), ("speculative", "delta_speculative")):
                    got, want = dl[mine], ref[theirs]
                    if set(got) != set(want) or any(abs(got[i] - want[i]) > 1e-12 for i in want):
                        raise SystemExit(f"GATE FAILED at {cond}: per-item {theirs} not reproduced; no results written")
                print(f"gate passed at {cond}: per-item Delta[COMMIT] and Delta[SPEC] reproduced to 1e-12", flush=True)
            results[(v, cond)] = recs

    all_missing = set().union(*missing_by.values())
    if args.plan or all_missing:
        cells = sorted({(i, c, p, q) for (i, s, c, p, q) in all_missing})
        NEW_DIR.mkdir(parents=True, exist_ok=True)
        e1common.atomic_write_json(REQUIRED, {
            "_comment": "EXPLORATORY oracle-belief ablation (analysis/v2_oracle_belief_d1.py): cells the oracle "
                        "variants land on that the frozen lazy plan never generated (all seeds). Generate with the "
                        "frozen Phase A code path into this directory; never into the frozen store.",
            "created_at_utc": e1common.now_utc(), "git_commit": e1common.git_commit(), "dataset": DS,
            "split": SPLIT, "n_cells": len(cells), "n_answers_all_seeds": len(cells) * 5,
            "by_variant_condition": {f"{v}|{c}": len({(i, cc, p, q) for (i, s, cc, p, q) in m})
                                     for (v, c), m in missing_by.items()},
            "by_path": {p: sum(1 for x in cells if x[2] == p) for p in {x[2] for x in cells}},
            "cells": [list(x) for x in cells]})
        print(f"wrote {REQUIRED}: {len(cells)} cells ({len(cells) * 5} answers over 5 seeds)")
        if all_missing and not args.plan:
            raise SystemExit("missing cells: generate them (see the required file) and re-run with --extra-store")
        return 0

    # ---- statistics ---------------------------------------------------------
    universe = sorted({i for rows in doc_c["contrasts"].values() for r in rows
                       for k in ("delta_commit", "delta_speculative") for i in r["per_item"][k]})
    boot = Boot(universe)
    cells_out = []
    pis = {}
    for b in bundles:
        cond = b["condition"]
        for v in VARIANTS:
            pis[(v, cond)] = per_item(results[(v, cond)])
    for b in bundles:
        cond = b["condition"]
        pr = cell_price(b)
        tuned = (b["tuned"].get(PROFILE) or {}).get(f"{pr['lambda_l']:g}") or {}
        for v in VARIANTS:
            pi = pis[(v, cond)]
            dl = deltas(pi)
            comp = {}
            for arch in ARCHS:
                ours, riv = pi[(arch, "ours_v2")], pi[(arch, D_STAR[arch])]
                comp[arch] = {k: boot({i: ours[k][i] - riv[k][i] for i in ours[k]}) for k in ("acc", "lat", "comp")}
            gains = {}
            if v != "frozen":
                base = pis[("frozen", cond)]
                for arch in ARCHS:
                    for c in CONTROLLERS:
                        a, z = pi[(arch, c)]["u"], base[(arch, c)]["u"]
                        gains[f"{arch}|{c}"] = boot({i: a[i] - z[i] for i in a})
            changed = {}
            if v != "frozen":
                base = pis[("frozen", cond)]
                for arch in ARCHS:
                    for c in CONTROLLERS:
                        d0, d1 = base[(arch, c)]["dec"], pi[(arch, c)]["dec"]
                        changed[f"{arch}|{c}"] = sum(1 for i in d1 if d1[i] != d0[i])
            changed_vs_oracle = {}
            if v not in ("frozen", "oracle"):
                base = pis[("oracle", cond)]
                for arch in ARCHS:
                    for c in CONTROLLERS:
                        d0, d1 = base[(arch, c)]["dec"], pi[(arch, c)]["dec"]
                        changed_vs_oracle[f"{arch}|{c}"] = sum(1 for i in d1 if d1[i] != d0[i])
            cells_out.append({
                "condition": cond, "condition_role": "primary" if cond == ASR else "secondary (text150)",
                "profile": PROFILE, "belief_quantile": QUANTILE, "price": _price_key(pr),
                "b9plus_timing_rule": tuned.get("b9plus_timing_rule"), "belief_variant": v,
                "delta_commit": boot(dl["commit"]), "delta_speculative": boot(dl["speculative"]),
                "interaction": boot(dl["interaction"]),
                "delta_decomposition": comp,
                "controllers": {f"{arch}|{c}": controller_summary(pi[(arch, c)], test_labels)
                                for arch in ARCHS for c in CONTROLLERS},
                "utility_change_vs_frozen_belief": gains or None,
                "decisions_changed_vs_frozen_belief": changed or None,
                "decisions_changed_vs_oracle_belief": changed_vs_oracle or None,
                "frozen_contrasts_reference": None if v != "frozen" else {
                    "delta_commit": frozen[cond]["delta"]["commit"],
                    "delta_speculative": frozen[cond]["delta"]["speculative"],
                    "interaction": frozen[cond]["interaction"]},
            })
    gen = generation_accounting([Path(p) for p in args.extra_store]) if args.extra_store else None
    out = {
        "_comment": LABEL + ". Ours-v2 and B9+ re-run on d1_open test with only the belief inputs replaced "
                            "(and Ours-v2's belief-transition model refitted on dev for the oracle beliefs); "
                            "alias-graded outcomes; no LLM judge.",
        "status": LABEL,
        "generated_by": "python -m analysis.v2_oracle_belief_d1", "generated_at_utc": e1common.now_utc(),
        "git_commit": e1common.git_commit(), "dataset": DS, "split": SPLIT,
        "gate": "the frozen-belief variant reproduced contrasts_test.json per-item Delta[COMMIT] and "
                "Delta[SPECULATIVE] at both cells to 1e-12",
        "primary_cell": PRIMARY, "d_star": D_STAR,
        "belief_variants": {
            "frozen": "the frozen calibrated self-assessment belief (gate)",
            "oracle": "b_k = empirical need label y (0/1) at every step from the first partial",
            "sharpen": "b_k = p0 + f_k (y - p0), f_k = causal e/(e+R_hat) (1 at end of turn), p0 = dev base rate",
            "reveal": "b_k = p0 until f_k >= u_i, then y (u_i ~ U(0,1) from a hash of the item id): calibrated, "
                      "sharpening; added by the analyst after `sharpen` proved miscalibrated at intermediate steps",
        },
        "dev_prior_p0": p0, "dev_prior_source": "mean of the E1-v2 dev empirical-need labels (d1_open, alias)",
        "dev_mean_calibrated_end_belief_frozen": (est.get("policy_constants") or {}).get("prior_belief"),
        "test_need_rate": float(np.mean(list(test_labels.values()))),
        "refitted": {"what": "Ours-v2 belief-transition (lookahead) model only, per oracle variant",
                     "phase_variable": "causal f_k (the frozen fit used the reference word fraction)",
                     **{v: {k: x for k, x in trans_meta[v].items() if k != "model"} for v in trans_meta},
                     "models": {v: trans_meta[v]["model"] for v in trans_meta}},
        "not_refitted": "B9+ timing rule (frozen dev-tuned, belief-independent r_below / r_threshold rule), "
                        "B9+ policy constants, prices, g+/g-, rho table, remaining-speech predictor, latency model",
        "inputs": {"contrasts_sha256": e1common.sha256_file(phase_a / f"contrasts_{SPLIT}.json"),
                   "store_sha256": e1common.sha256_file(phase_a / f"store_{SPLIT}.jsonl"),
                   "extra_stores": {p: e1common.sha256_file(Path(p)) for p in args.extra_store},
                   "labels_test_sha256": e1common.sha256_file(e1dir / "labels_test.json"),
                   "labels_dev_sha256": e1common.sha256_file(e1dir / "labels_dev.json")},
        "new_cells": {"n_cells_used_from_new_store": store.n_extra_cells,
                      "required_file": str(REQUIRED.relative_to(_REPO_ROOT)) if args.extra_store else None,
                      "required_file_sha256": e1common.sha256_file(REQUIRED) if args.extra_store else None,
                      "generation": gen},
        "bootstrap": {"B": B, "seed": SEED, "unit": "item", "n_items_universe": len(universe),
                      "same_resamples_as": "analysis/v2_test_d1.py"},
        "objective_diagnostics": {b["condition"]: objective_diagnostics(b) for b in bundles},
        "cells": cells_out,
    }
    out["summary_md"] = make_summary(out)
    out["caveats_md"] = CAVEATS
    e1common.atomic_write_json(OUT_JSON, out)
    write_md(out)
    for c in cells_out:
        print(c["condition"][:20], c["belief_variant"], "dC", c["delta_commit"]["estimate"], c["delta_commit"]["ci95"],
              "dS", c["delta_speculative"]["estimate"], c["delta_speculative"]["ci95"],
              "I", c["interaction"]["estimate"], c["interaction"]["ci95"])
    print(f"wrote {OUT_JSON}")
    return 0


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def objective_diagnostics(bundle: dict) -> dict:
    """Frozen-objective terms at the cell's price: U(retrieve | b) for b = 0
    and b = 1 for a fully hidden dispatch (exposed = 0, priced latency = pi)
    at word fraction f, and U(b = 0) at the end of turn (exposed = E[L])."""
    from controller.baselines._objective_v2 import (latency_dist_for, priced_latency_ms, quality_model,
                                                    read_prices_v2, rho_at, utility_retrieve, waste_at)
    cfg = load_e3v2_config(overrides=bundle["overrides"])
    prices, quality = read_prices_v2(cell_price(bundle), cfg), quality_model(cfg)
    lat = latency_dist_for({"latency_profile": PROFILE}, cfg)
    hidden = priced_latency_ms(lat, 0.0)
    out = {"g_plus": quality.g_plus, "g_minus": quality.g_minus, "lambda_l": prices.lambda_l,
           "lambda_c": prices.lambda_c, "pi_ms": lat.nonhideable_ms, "mean_latency_ms": lat.mean_ms,
           "u_b0_hidden": {}, "u_b1_hidden": {}, "rho": {}}
    for f in (0.5, 0.7, 0.9):
        w = waste_at(f, False, cfg)
        out["u_b0_hidden"][str(f)] = utility_retrieve(0.0, rho_at(f, True, cfg), hidden, w, prices, quality)
        out["u_b1_hidden"][str(f)] = utility_retrieve(1.0, rho_at(f, True, cfg), hidden, w, prices, quality)
        out["rho"][str(f)] = rho_at(f, True, cfg)
    out["u_b0_end_of_turn"] = utility_retrieve(0.0, 0.0, priced_latency_ms(lat, lat.mean_ms), 0.0, prices, quality)
    return out


CAVEATS = [
    "Exploratory and post hoc: designed after the pre-registered d1_open analysis; it is not part of PREREG_V2 and "
    "changes none of its tests. The 'calibrated reveal' variant was added by the analyst after seeing that the "
    "requested sharpening belief is miscalibrated mid-utterance.",
    "One dataset (d1_open), one latency profile and price per condition (the frozen primary cell and text150's own "
    "0.65-quantile price), alias grading only (no LLM judge); items are the bootstrap clusters.",
    "Only the belief and Ours-v2's lookahead transitions change. B9+ keeps the timing rule tuned on dev under the "
    "REAL belief (r_below on ASR, r_threshold on text150); a rule re-tuned under the oracle belief could do better, "
    "so the oracle contrasts may favour Ours-v2. g+/g−, ρ, prices and the remaining-speech predictor are the frozen "
    "dev fits; text150 uses the g+/g− fitted on the primary ASR condition, which is what drives its negative oracle "
    "result.",
    "The oracle is the empirical need label (closed-book wrong on the text question in ≥ 3 of 5 draws); it is "
    "hindsight information no deployable controller has, and it is not the per-turn outcome (ASR-condition answers "
    "with or without passages), so 'perfect' refers to the label, not to the realised gain of retrieval.",
    "Newly generated cells (Mistral-7B, same Phase A code path) were measured on 2026-10-18, the frozen ones on "
    "2026-10-16, on the same endpoint and model; vLLM sampling with fixed seeds is reproducible per request but not guaranteed "
    "bit-identical across server batches, so these cells are comparable to, not identical with, cells the frozen "
    "run would have produced.",
]


def _cell(doc: dict, cond: str, variant: str) -> dict:
    return next(c for c in doc["cells"] if c["condition"] == cond and c["belief_variant"] == variant)


def make_summary(doc: dict) -> str:
    """The plain-language summary, every number read from the results."""
    def ci(b):
        return f"{b['estimate']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"
    txt = sorted(c for c in {c["condition"] for c in doc["cells"]} if c != ASR)
    fr, orc, sh, rv = (_cell(doc, ASR, v) for v in VARIANTS)
    sh_nn = sh["controllers"]["commit|ours_v2"]["uses_passages_rate_no_need"]
    parts = [
        "This ablation is exploratory and post hoc, not part of the pre-registered test: Ours-v2 and B9+ are re-run "
        "on the d1_open test items with only the belief replaced.",
        f"With the empirical need label as the belief (oracle), the primary-cell contrast changes sign: "
        f"Δ[COMMIT] = Δ[SPEC] = {ci(orc['delta_commit'])}, against {ci(fr['delta_commit'])} with the real belief; "
        f"both controllers then retrieve exactly the needed items and the gain comes from dispatch timing "
        f"(Δ accuracy {orc['delta_decomposition']['commit']['acc']['estimate']:+.4f}, Δ latency term "
        f"{orc['delta_decomposition']['commit']['lat']['estimate']:+.4f}), with zero interaction because no "
        f"information arrives during the turn.",
        f"With a calibrated belief that reveals the label at a random point of the utterance (a variant added by "
        f"the analyst), joint control still wins (Δ[COMMIT] {ci(rv['delta_commit'])}, Δ[SPEC] "
        f"{ci(rv['delta_speculative'])}), and the interaction is {ci(rv['interaction'])}, the opposite sign to the "
        f"P2 prediction.",
        f"With the requested sharpening belief p0 + f (y − p0), which is informative but not calibrated "
        f"mid-utterance, Ours-v2[COMMIT] commits early on intermediate beliefs and retrieves for "
        f"{100 * sh_nn:.1f}% of the not-needed items (Δ[COMMIT] {ci(sh['delta_commit'])}), while Ours-v2[SPEC] "
        f"decides exactly as under the oracle (Δ[SPEC] {ci(sh['delta_speculative'])}).",
    ]
    if txt:
        t = txt[0]
        to, tr = _cell(doc, t, "oracle"), _cell(doc, t, "reveal")
        od = doc["objective_diagnostics"][t]
        parts.append(
            f"On text150 the oracle and reveal beliefs make Ours-v2 worse (Δ[COMMIT] {ci(to['delta_commit'])} and "
            f"{ci(tr['delta_commit'])}): with the dev-fitted g− = {od['g_minus']:+.3f}, the frozen objective values a "
            f"fully hidden retrieval at f = 0.7 for an item known not to need it at {od['u_b0_hidden']['0.7']:+.4f}, "
            f"so Ours-v2 takes it ({100 * to['controllers']['commit|ours_v2']['uses_passages_rate_no_need']:.1f}% of "
            f"such items under the oracle) and pays in accuracy and latency.")
    parts.append(
        "Taken together, the ablation suggests that the pre-registered negative result reflects the belief (and "
        "objective terms such as g−) more than the joint decision structure, but it does not show that a realistic "
        "belief reaches the regime where joint control pays: it is post hoc, on one dataset and price cell, "
        "alias-graded, with B9+'s timing rule tuned under the real belief, and the gain it finds is small.")
    return " ".join(parts)


def _f(x, nd=4):
    return "—" if x is None else f"{x:+.{nd}f}"


def _ci(b, nd=4):
    return f"{_f(b['estimate'], nd)} [{_f(b['ci95'][0], nd)}, {_f(b['ci95'][1], nd)}]"


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


VARIANT_LABEL = {"frozen": "frozen belief (gate; = pre-registered test)", "oracle": "oracle: b = y at every step",
                 "sharpen": "sharpening oracle: b = p0 + f (y − p0)",
                 "reveal": "calibrated reveal: b = p0, then y once f ≥ u"}


def write_md(doc: dict) -> None:
    cells = doc["cells"]
    prim = [c for c in cells if c["condition"] == ASR]
    sec = [c for c in cells if c["condition"] != ASR]
    p = doc["primary_cell"]
    L = ["# Oracle-belief ablation, d1_open test (EXPLORATORY)", "",
         f"**Status: {doc['status']}.** Post hoc: designed and run after the pre-registered d1_open test was "
         "analysed; PREREG_V2 does not mention it and none of its tests or verdicts change. Alias-graded outcomes; "
         "no LLM judge. Source: `docs/research_v2/oracle_belief_v2.json` (`analysis/v2_oracle_belief_d1.py`).", "",
         "## Question and design", "",
         "Does the joint controller (Ours-v2, \"Ours-S\" in the paper) beat the decomposed rival B9+ when the belief "
         "it plans with is good? Ours-v2 and B9+ (frozen dev-tuned timing rule) are re-run on the d1_open test items "
         "under both architectures, with D*[SPEC] = B9+ as frozen for d1, changing ONLY the belief b_k (at every "
         "partial and at the end of turn) and what is derived from it:", "",
         "* **oracle**: b_k = y, the E1-v2 empirical need label (closed-book wrong in ≥ 3 of 5 draws, alias), "
         "from the first partial on. The belief never moves, so waiting buys no information (no C2).",
         f"* **sharpening oracle**: b_k = p0 + f_k (y − p0), p0 = {doc['dev_prior_p0']:.4f} (dev base rate of y), "
         "f_k = the causal fraction-heard estimate e/(e + R̂) the controllers already use (1 at the end of turn; "
         "never the reference word fraction). The belief moves from the prior towards the label as speech is heard. "
         "It is NOT calibrated at intermediate steps: b_k is a deterministic function of y, so P(y = 1 | b_k) is 0 "
         "or 1 while b_k is in between; a controller that reads b_k as a probability misjudges it (a not-needed "
         "item at b_k = 0.15 is treated as a 15% chance of need).",
         "* **calibrated reveal** (added by the analyst after the sharpening variant proved miscalibrated; not in the "
         "original request): b_k = p0 until f_k ≥ u_i, then y, with u_i ~ U(0, 1) a fixed hash of the item id "
         "(independent of y). P(y = 1 | b_k = p0) = p0, so the belief is calibrated at every step and sharpens "
         "(reveals the label) as speech is heard; b = y at the end of turn.",
         "* Refitted (dev split only): Ours-v2's belief-transition (lookahead) model, per variant, with the frozen "
         "fitter on dev sequences of that variant's belief as the controller computes it "
         f"({doc['refitted']['oracle']['n_sequences']:,} dev turns over both conditions; phase = causal f; the oracle "
         "fit is the identity on {0, 1}, the sharpening fit encodes the drift towards the label, the reveal fit the "
         "per-phase reveal probability). Not refitted: "
         f"{doc['not_refitted']}.",
         "* Gate: with the frozen belief this re-implementation reproduces the per-item Δ[COMMIT] and Δ[SPEC] of "
         "the frozen test contrasts to 1e-12 at both cells (first row of each table).",
         f"* Statistics: paired item bootstrap, B = {doc['bootstrap']['B']:,}, seed {doc['bootstrap']['seed']}, the "
         "resamples of `analysis/v2_test_d1.py`; percentile 95% CIs. U = accuracy − λ_l (TTFA − skip TTFA) − "
         "λ_c · retrievals per item (mean of 5 seeds); Δ = U[Ours-v2] − U[B9+].", "",
         f"## Primary cell (primary ASR, rewrite_faiss, belief quantile 0.65, λ_l = {p['lambda_l']}, "
         f"λ_c = {p['lambda_c']})", "",
         "| belief | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction Δ[C] − Δ[S] [95% CI] | n items |",
         "|---|---|---|---|---|"]
    for c in prim:
        L.append(f"| {VARIANT_LABEL[c['belief_variant']]} | {_ci(c['delta_commit'])} | {_ci(c['delta_speculative'])} | "
                 f"{_ci(c['interaction'])} | {c['delta_commit']['n_items']:,} |")
    L += ["", "### Utility decomposition and retrieval behaviour (primary cell)", "",
          "Means over items. Latency and compute terms are the amounts SUBTRACTED from accuracy "
          "(U = accuracy − latency term − compute term). 'uses' = answer generated with passages; 'early' = share "
          "of dispatches made on a partial rather than the final transcript; 'uses | need' and 'uses | no need' "
          "split by the label y.", "",
          "| belief | arch | controller | U | accuracy | latency term | compute term | dispatch | uses | early | "
          "uses \\| need | uses \\| no need | TTFA − skip (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in prim:
        for key, s in c["controllers"].items():
            arch, ctrl = key.split("|")
            L.append(f"| {c['belief_variant']} | {arch} | {ctrl} | {s['utility']:.4f} | {s['accuracy']:.4f} | "
                     f"{s['latency_term']:.4f} | {s['compute_term']:.4f} | {_pct(s['dispatch_rate'])} | "
                     f"{_pct(s['uses_passages_rate'])} | {_pct(s['early_share_of_dispatches'])} | "
                     f"{_pct(s['uses_passages_rate_need'])} | {_pct(s['uses_passages_rate_no_need'])} | "
                     f"{s['mean_ttfa_minus_skip_ms']:.0f} |")
    L += ["", "Δ decomposed (Ours-v2 − B9+; Δ = Δaccuracy − Δlatency term − Δcompute term):", "",
          "| belief | arch | Δ accuracy [95% CI] | Δ latency term [95% CI] | Δ compute term [95% CI] |",
          "|---|---|---|---|---|"]
    for c in prim:
        for arch in ARCHS:
            d = c["delta_decomposition"][arch]
            L.append(f"| {c['belief_variant']} | {arch} | {_ci(d['acc'])} | {_ci(d['lat'])} | {_ci(d['comp'])} |")
    L += ["", "Effect of the belief on each controller (U[variant] − U[frozen belief], same items, paired):", "",
          "| belief | arch | controller | ΔU vs frozen belief [95% CI] | decisions ≠ frozen belief | "
          "decisions ≠ oracle |", "|---|---|---|---|---|---|"]
    for c in prim:
        if not c["utility_change_vs_frozen_belief"]:
            continue
        for key, g in c["utility_change_vs_frozen_belief"].items():
            arch, ctrl = key.split("|")
            vo = (c.get("decisions_changed_vs_oracle_belief") or {}).get(key)
            L.append(f"| {c['belief_variant']} | {arch} | {ctrl} | {_ci(g)} | "
                     f"{c['decisions_changed_vs_frozen_belief'][key]:,} | {'—' if vo is None else f'{vo:,}'} |")
    if sec:
        s0 = sec[0]
        L += ["", f"## text150 (contrast; rewrite_faiss, its own 0.65-quantile price {s0['price']}, B9+ rule "
                  f"{s0['b9plus_timing_rule'].get('kind') if s0['b9plus_timing_rule'] else None})", "",
              "| belief | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] | n items |",
              "|---|---|---|---|---|"]
        for c in sec:
            L.append(f"| {VARIANT_LABEL[c['belief_variant']]} | {_ci(c['delta_commit'])} | "
                     f"{_ci(c['delta_speculative'])} | {_ci(c['interaction'])} | {c['delta_commit']['n_items']:,} |")
        L += ["", "| belief | arch | controller | U | accuracy | latency term | compute term | uses | early | "
                  "uses \\| need | uses \\| no need |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in sec:
            for key, s in c["controllers"].items():
                arch, ctrl = key.split("|")
                L.append(f"| {c['belief_variant']} | {arch} | {ctrl} | {s['utility']:.4f} | {s['accuracy']:.4f} | "
                         f"{s['latency_term']:.4f} | {s['compute_term']:.4f} | {_pct(s['uses_passages_rate'])} | "
                         f"{_pct(s['early_share_of_dispatches'])} | {_pct(s['uses_passages_rate_need'])} | "
                         f"{_pct(s['uses_passages_rate_no_need'])} |")
    nc = doc["new_cells"]
    gen = nc.get("generation") or {}
    L += ["", "## New generation", ""]
    if nc["n_cells_used_from_new_store"]:
        L.append(f"* The oracle variants land on {gen['n_cells']:,} (item, condition, retrieve, query variant) cells "
                 f"the frozen lazy plan had not generated ({gen['n_answers_needed_all_seeds']:,} answers over the 5 "
                 f"seeds). They were generated with the frozen Phase A code path (`simulator/e3v2_phase_a.py "
                 f"--required-cells`, same prompts, seeds, retrieval index and passages, temperature/top-p/max-tokens) "
                 f"into a copy under `simulator/e3v2_oracle/d1_open/` (the frozen store is untouched) and graded by "
                 f"alias match only: {gen['n_answers_generated']:,} new Mistral-7B-Instruct-v0.3 answers "
                 f"({gen['n_answers_reused_identical_key']:,} more had an identical answer key in the frozen run and "
                 f"were reused), {gen['output_tokens']:,} output tokens, ~{gen['prompt_tokens_approx']:,} prompt "
                 f"tokens, answer-stage wall time {gen['answer_wall_s'] / 60:.1f} min on the existing Modal vLLM "
                 f"endpoint; no LLM judge or Azure call. {gen['cost_note']}")
        L.append(f"* {nc['n_cells_used_from_new_store']:,} new (item, seed) cells were loaded from the new store; "
                 "latency draws use the frozen common random numbers (seeded per item, seed, condition, profile).")
    else:
        L.append("* None: every decision landed on a cell the frozen lazy plan had generated.")
    L += ["", "## Frozen-objective terms behind the decisions", "",
          "Model utility of a retrieval whose call is fully hidden (exposed = 0, so the priced latency is π alone) "
          "for an item with belief b = 0 and b = 1, at estimated word fraction f (frozen g+, g−, λ, π, the "
          "condition's dev ρ table, waste), and U(b = 0) at the end of turn (exposed = E[L]). With the dev-fitted "
          "g− < 0 (retrieval modelled as helping items that do not need it), U(b = 0) for a hidden dispatch is "
          "within ±0.005 of zero, so a confident 'not needed' belief does not by itself rule out a well-hidden "
          "retrieval, and on text150 it is slightly positive.", "",
          "| condition | g+ | g− | λ_l | π (ms) | ρ at f = 0.5 / 0.7 / 0.9 | U(b=0) hidden, f = 0.5 / 0.7 / 0.9 | "
          "U(b=1) hidden, f = 0.5 / 0.7 / 0.9 | U(b=0) at end of turn |", "|---|---|---|---|---|---|---|---|---|"]
    for cond, od in doc["objective_diagnostics"].items():
        L.append(f"| {'primary ASR' if cond == ASR else cond} | {od['g_plus']:.4f} | {od['g_minus']:+.4f} | "
                 f"{od['lambda_l']:.4f} | {od['pi_ms']:.0f} | "
                 + " / ".join(f"{od['rho'][k]:.3f}" for k in ("0.5", "0.7", "0.9")) + " | "
                 + " / ".join(f"{od['u_b0_hidden'][k]:+.4f}" for k in ("0.5", "0.7", "0.9")) + " | "
                 + " / ".join(f"{od['u_b1_hidden'][k]:+.4f}" for k in ("0.5", "0.7", "0.9")) + " | "
                 f"{od['u_b0_end_of_turn']:+.4f} |")
    L += ["", "## Summary", ""] + [doc.get("summary_md") or "(pending)"] + [""]
    L += ["## Caveats", ""] + [f"* {c}" for c in doc.get("caveats_md") or []] + [""]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
