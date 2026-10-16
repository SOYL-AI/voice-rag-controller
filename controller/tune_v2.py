"""
tune_v2.py — the EQUAL-BUDGET tuning harness for the v2 baselines (the
fairness rule of E3_BASELINE_INTERFACE.md / handbook 29.3), DEV SPLIT ONLY.

    python -m controller.tune_v2 --signals <dev_signals.jsonl> \
        --cells <phase_a_cells.jsonl> --items datasets/v2/d2_support/items.jsonl \
        --labels <e1v2_labels.json> --profiles faiss,remote_api \
        --out controller/baselines/e3v2_estimates.json

What is tuned, and with what budget:

  UNPRICED (one threshold each, the SAME number of grid evaluations each):
    b4 theta      grid = `budget` quantiles of the dev min-token-probability
    b5 threshold  grid = `budget` quantiles of the dev classifier score (the
                  naive-Bayes Adaptive-RAG classifier is trained on dev texts
                  with the E1-v2 empirical labels first)
    b6 threshold  grid = `budget` quantiles of dev max_entropy_nats
  Each grid value is scored by mean dev answer accuracy (Phase A cells, all
  seeds); ties go to the lower retrieval rate. "Maximise dev accuracy at its
  own compute": no compute penalty is imposed on an unpriced baseline.

    b8  trigger = the best of b4/b5/b6 by tuned dev accuracy (no extra grid);
        timing  = tuned IN ISOLATION on the timing objective: the policy with
        the most dev hidden latency among those whose always-retrieve dev
        accuracy is within `equal_accuracy_tolerance` of sentence_end's
        (E2's equal-accuracy condition).

  PRICED (ours_v2, b9_v2, b9plus_v2): nothing is tuned per price point. Only
  SHARED estimates are produced, used identically by all three:
    policy_constants  per latency profile and timing policy: dev-mean
                      realised hidden/exposed latency, rho, waste (B9/B9+/B8)
    rho_table         partial-query risk fitted on dev (1 - hit(f)/hit(final)
                      on needed items, isotonic in f) — applies to every
                      controller via load_e3v2_config()
    empirical_quality dev g+/g- measured from Phase A cells — LOGGED for
                      cross-checking E1-v2's fit; config's quality block
                      stays the source of truth.

The artifact is consumed via controller.e3v2.estimates_artifact (or
simulator/e3v2_decide.py --estimates). Every grid, every score and every
chosen value is recorded in it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    POLICIES_V2,
    estimate_policy_constants,
    load_e3v2_config,
)
from controller.baselines.unpriced_v2 import (  # noqa: E402
    classifier_score,
    max_entropy,
    min_token_prob,
    train_adaptive_rag_classifier,
)
from controller.sequential_v2 import TurnCtx, build_steps, get_controller_v2  # noqa: E402
from simulator.e3v2_decide import (  # noqa: E402
    CellStore,
    load_items,
    load_latency_model,
    load_turns,
    run_phase_b,
    split_ids,
    with_profile,
)


def quantile_grid(values: Sequence[float], budget: int) -> List[float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return []
    if budget <= 1:
        return [vals[len(vals) // 2]]
    grid = []
    for i in range(budget):
        pos = (len(vals) - 1) * i / (budget - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        grid.append(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))
    out = []
    for g in grid:
        if not out or abs(g - out[-1]) > 1e-12:
            out.append(g)
    return out


def accuracy_and_rate(records: List[dict]) -> Dict[str, float]:
    n = max(1, len(records))
    return {"accuracy": sum(bool(r["outcome"]["answer_correct"]) for r in records) / n,
            "retrieval_rate": sum(int(r["outcome"]["retrievals"]) for r in records) / n}


def _eval(name: str, params: dict, turns, store, seeds, profile, cfg, lm) -> Dict[str, float]:
    ctrl = get_controller_v2(name, cfg=cfg, **params)
    recs = run_phase_b(turns, store, {name: ctrl}, [None], seeds, [profile], cfg, lm)
    return accuracy_and_rate(recs)


def tune_threshold(name: str, param: str, grid: List[float], turns, store, seeds, profile, cfg, lm,
                   extra: Optional[dict] = None) -> dict:
    rows = []
    for value in grid:
        params = {**(extra or {}), param: value}
        rows.append({"value": value, **_eval(name, params, turns, store, seeds, profile, cfg, lm)})
    if not rows:
        return {"param": param, "grid": [], "scores": [], "chosen": {}, "n_evaluated": 0,
                "note": "no dev signal values; left at config default"}
    best = max(range(len(rows)), key=lambda i: (rows[i]["accuracy"], -rows[i]["retrieval_rate"], -i))
    chosen = {**(extra or {}), param: rows[best]["value"]}
    return {"param": param, "grid": grid, "scores": rows, "chosen": chosen,
            "chosen_accuracy": rows[best]["accuracy"], "n_evaluated": len(rows)}


def _pav_decreasing(ys: List[float], ws: List[float]) -> List[float]:
    """Pool-adjacent-violators for a non-increasing fit."""
    blocks = []  # [value, weight, count]
    for y, w in zip(ys, ws):
        blocks.append([y, w, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            tw = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / tw if tw > 0 else (v1 + v2) / 2, tw, c1 + c2])
    out = []
    for v, _, c in blocks:
        out.extend([v] * c)
    return out


def fit_rho_table(turns: Sequence[TurnCtx], store: CellStore, gold: Dict[str, List[str]],
                  labels: Optional[Dict[str, bool]], n_bins: int = 5) -> Optional[list]:
    """rho(f) = 1 - hit(f)/hit(final) on dev items that need retrieval, f the
    partial's reference word fraction (hindsight is fine on dev), binned and
    made non-increasing in f. None when there is nothing to fit on."""
    if not labels or not gold:
        return None
    hits = [[0.0, 0.0] for _ in range(n_bins)]
    final = [0.0, 0.0]
    for turn in turns:
        if not labels.get(turn.item_id) or not gold.get(turn.item_id):
            continue
        g = set(gold[turn.item_id])
        n_final = max(1, len((turn.transcript_final or "").split()))
        parts = ((turn.signals or {}).get("timing") or {}).get("partials") or []
        for k, part in enumerate(sorted(parts, key=lambda p: float(p["t_ms"]))):
            key = (turn.item_id, turn.condition, k)
            if key not in store.retrievals:
                continue
            frac = min(1.0, len(str(part.get("text", "")).split()) / n_final)
            b = min(n_bins - 1, int(frac * n_bins))
            rec = store.retrievals[key]
            hits[b][0] += 1.0 if g & set(rec.get("retrieved_parent_ids") or rec["retrieved_doc_ids"]) else 0.0
            hits[b][1] += 1.0
        fkey = (turn.item_id, turn.condition, None)
        if fkey in store.retrievals:
            frec = store.retrievals[fkey]
            final[0] += 1.0 if g & set(frec.get("retrieved_parent_ids") or frec["retrieved_doc_ids"]) else 0.0
            final[1] += 1.0
    if final[1] == 0 or final[0] == 0:
        return None
    hit_final = final[0] / final[1]
    xs, ys, ws = [], [], []
    for i, (h, n) in enumerate(hits):
        if n == 0:
            continue
        xs.append((i + 0.5) / n_bins)
        ys.append(min(1.0, max(0.0, 1.0 - (h / n) / hit_final)))
        ws.append(n)
    if not xs:
        return None
    ys = _pav_decreasing(ys, ws)
    return [[x, y] for x, y in zip(xs, ys)] + [[1.0, 0.0]]


def empirical_quality(turns, store: CellStore, labels, seeds) -> Optional[dict]:
    if not labels:
        return None
    gains = {True: [], False: []}
    for turn in turns:
        need = labels.get(turn.item_id)
        if need is None:
            continue
        fkey = (turn.item_id, turn.condition, None)
        if fkey not in store.retrievals:
            continue
        qv = store.retrievals[fkey]["query_variant"]
        for s in seeds:
            try:
                ret = store.cell(turn.item_id, s, turn.condition, "retrieve", qv)["answer_correct"]
                skip = store.cell(turn.item_id, s, turn.condition, "skip", "none")["answer_correct"]
            except KeyError:
                continue
            gains[bool(need)].append(float(ret) - float(skip))
    out = {}
    if gains[True]:
        out["g_plus"] = sum(gains[True]) / len(gains[True])
    if gains[False]:
        out["g_minus"] = -sum(gains[False]) / len(gains[False])
    out["n_needed"], out["n_not_needed"] = len(gains[True]), len(gains[False])
    return out


def _mean_utility(records: List[dict]) -> float:
    vals = [r["realized_utility"] for r in records if r.get("realized_utility") is not None]
    return sum(vals) / len(vals) if vals else float("-inf")


def r_below_grid(turns: Sequence[TurnCtx], profile: str, cfg: dict, n_grid: int = 21) -> List[dict]:
    """
    The pre-registered r_below grid (PREREG_V2 §4.5, added 2026-10-16):
    r0 at n_grid quantiles of R_hat over every query-ready partial of the dev
    turns, and for each r0 the dev means at the rule's FIRING step (the first
    query-ready partial with R_hat <= r0; the end of turn if none): realised
    exposed retrieval latency (B9+'s price), and remaining / elapsed speech
    (SPEC-theta's gate scalars).
    """
    import numpy as np

    from controller.baselines._objective_v2 import latency_dist_for
    from controller.sequential_v2 import end_of_turn_ms

    ccfg = {**cfg, "architecture": "commit"}
    all_r, per_turn = [], []
    for t in turns:
        ctx = with_profile(t, profile, ccfg)
        lat = latency_dist_for(ctx.signals["retrieval"], ccfg)
        end = end_of_turn_ms(ctx, ccfg)
        rs = [(float(o.remaining_ms), max(0.0, end - o.available_ms), float(o.speech_elapsed_ms))
              for o in build_steps(ctx, ccfg)
              if not o.is_final and o.query_ready and o.remaining_ms is not None]
        all_r += [r for r, _, _ in rs]
        per_turn.append((rs, lat))
    out = []
    if not all_r:
        return out
    for r0 in np.quantile(np.asarray(all_r), np.linspace(0.0, 1.0, n_grid)):
        ex, rem, ela, fired = [], [], [], 0
        for rs, lat in per_turn:
            hit = next(((r, w, e) for r, w, e in rs if r <= r0), None)
            if hit is None:
                ex.append(lat.mean_ms)
                continue
            fired += 1
            ex.append(lat.expected_exposed_ms(hit[1]))
            rem.append(hit[0])
            ela.append(hit[2])
        out.append({"r0": float(r0), "late_exposed_ms": float(np.mean(ex)),
                    "remaining_ms": float(np.mean(rem)) if rem else None,
                    "elapsed_ms": float(np.mean(ela)) if ela else 0.0,
                    "fire_fraction": fired / max(1, len(per_turn))})
    return out


def tune_b9plus_timing(turns: Sequence[TurnCtx], store: CellStore, seeds: Sequence[int], profile: str,
                       prices: dict, cfg: dict, lm, n_grid: int = 21) -> dict:
    """
    PREREG_V2 §4.5: B9+'s timing rule, tuned on dev at one (profile, price)
    by mean realised utility over the pre-registered family
      early | late | r_threshold (dispatch at the first query-ready partial
      iff R_hat > r0) | r_below (dispatch at the first query-ready partial
      with R_hat <= r0; added 2026-10-16),
    each r0 on an n_grid-point quantile grid. A branch that dispatches
    before the end of turn is priced at the dev-mean realised exposure of the
    turns it applies to.
    """
    import numpy as np

    from controller.baselines._objective_v2 import latency_dist_for, policy_fires
    from controller.sequential_v2 import end_of_turn_ms

    ccfg = {**cfg, "architecture": "commit"}
    r_hat, exposed = [], []
    for t in turns:
        ctx = with_profile(t, profile, ccfg)
        lat = latency_dist_for(ctx.signals["retrieval"], ccfg)
        end = end_of_turn_ms(ctx, ccfg)
        obs = next(o for o in build_steps(ctx, ccfg) if policy_fires("immediate", o, ccfg, lat))
        if obs.is_final or obs.remaining_ms is None:
            continue
        r_hat.append(float(obs.remaining_ms))
        exposed.append(lat.expected_exposed_ms(max(0.0, end - obs.available_ms)))
    r_hat, exposed = np.asarray(r_hat), np.asarray(exposed)
    candidates = [{"kind": "early"}, {"kind": "late"}]
    if r_hat.size:
        for r0 in np.quantile(r_hat, np.linspace(0.0, 1.0, n_grid)):
            sel = r_hat > r0
            if sel.any():
                candidates.append({"kind": "r_threshold", "r0": float(r0),
                                   "early_exposed_ms": float(exposed[sel].mean()),
                                   "early_fraction": float(sel.mean())})
    for row in r_below_grid(turns, profile, cfg, n_grid):
        candidates.append({"kind": "r_below", "r0": row["r0"], "late_exposed_ms": row["late_exposed_ms"]})
    scores = []
    for rule in candidates:
        ctrl = get_controller_v2("b9plus_v2", cfg=ccfg, timing_rule=rule)
        recs = run_phase_b(turns, store, {"b9plus_v2": ctrl}, [prices], seeds, [profile], ccfg, lm)
        scores.append(_mean_utility(recs))
    best = max(range(len(candidates)), key=lambda i: (scores[i], -i))
    return {"chosen": candidates[best], "dev_utility": scores[best], "n_evaluated": len(candidates),
            "candidates": [{**c, "dev_utility": s} for c, s in zip(candidates, scores)]}


def tune_spec_trigger(turns: Sequence[TurnCtx], store: CellStore, seeds: Sequence[int], profile: str,
                      prices: dict, cfg: dict, lm, triggers: Sequence = ("immediate", "forecast"),
                      n_grid: int = 21) -> dict:
    """PREREG_V2 §4.5: the SHARED SPEC / SPEC-theta trigger, tuned on dev
    over {immediate, forecast} + the r_below grid (added 2026-10-16): the
    trigger under which the better of the two has the higher mean realised
    utility (D*[SPEC] is the better of them). For an r_below trigger SPEC-
    theta's gate scalars are its dev means at the firing step."""
    scfg = {**cfg, "architecture": "speculative"}
    cands = [(t, None) for t in triggers]
    for row in r_below_grid(turns, profile, cfg, n_grid):
        cands.append(({"kind": "r_below", "r0": row["r0"]},
                      {profile: {"remaining_ms": row["remaining_ms"], "elapsed_ms": row["elapsed_ms"]}}))
    rows = []
    for trig, consts in cands:
        ctrls = {"b10_speculative_v2": get_controller_v2("b10_speculative_v2", cfg=scfg, timing=trig),
                 "b11_speculative_gated_v2": get_controller_v2("b11_speculative_gated_v2", cfg=scfg, timing=trig,
                                                               trigger_constants=consts)}
        recs = run_phase_b(turns, store, ctrls, [prices], seeds, [profile], scfg, lm)
        per = {n: _mean_utility([r for r in recs if r["baseline"] == n]) for n in ctrls}
        rows.append({"trigger": trig, "trigger_constants": consts, "dev_utility": per, "best": max(per.values())})
    best = max(range(len(rows)), key=lambda i: (rows[i]["best"], -i))
    return {"chosen": rows[best]["trigger"], "trigger_constants": rows[best]["trigger_constants"],
            "n_evaluated": 2 * len(rows), "candidates": rows}


def tune_all(turns: Sequence[TurnCtx], store: CellStore, seeds: Sequence[int], profiles: Sequence[str],
             cfg: dict, labels: Optional[Dict[str, bool]], gold: Dict[str, List[str]],
             budget: Optional[int] = None, latency_model=None, plan_all_b8: bool = False) -> dict:
    budget = int(budget if budget is not None else cfg["tuning"]["budget"])
    lm = latency_model if latency_model is not None else load_latency_model(cfg)
    acc_profile = profiles[0]  # accuracy does not depend on the latency profile

    constants = estimate_policy_constants(
        [with_profile(t, p, cfg) for t in turns for p in profiles], cfg)

    finals = [build_steps(t, cfg)[-1] for t in turns]
    unpriced = {}

    p_tok = [min_token_prob(o.belief_entry)[0] for o in finals]
    unpriced["b4"] = tune_threshold("b4_v2", "theta", quantile_grid(p_tok, budget),
                                    turns, store, seeds, acc_profile, cfg, lm)
    h = [max_entropy(o.belief_entry) for o in finals]
    unpriced["b6"] = tune_threshold("b6_v2", "threshold", quantile_grid(h, budget),
                                    turns, store, seeds, acc_profile, cfg, lm)
    if labels:
        texts = [o.text for t, o in zip(turns, finals) if t.item_id in labels]
        ys = [bool(labels[t.item_id]) for t in turns if t.item_id in labels]
        model = train_adaptive_rag_classifier(texts, ys)
        scores = [classifier_score(model, o.text) for o in finals]
        unpriced["b5"] = tune_threshold("b5_v2", "threshold", quantile_grid(scores, budget),
                                        turns, store, seeds, acc_profile, cfg, lm, extra={"model": model})
        unpriced["b5"]["train"] = {"n": len(ys), "n_positive": sum(ys)}
    else:
        unpriced["b5"] = {"param": "threshold", "grid": [], "scores": [], "chosen": {}, "n_evaluated": 0,
                          "note": "no E1-v2 labels: classifier untrained, v1 keyword fallback"}

    best_trigger = max(("b4", "b5", "b6"),
                       key=lambda k: unpriced[k].get("chosen_accuracy", -1.0))
    timing_scores = {}
    for policy in POLICIES_V2:
        acc = _eval(f"b1_v2_{policy}", {}, turns, store, seeds, acc_profile, cfg, lm)
        hidden = [constants["profiles"][p][policy]["hidden_ms"] for p in constants["profiles"]]
        timing_scores[policy] = {**acc, "mean_hidden_ms": sum(hidden) / max(1, len(hidden))}
    if plan_all_b8:
        # Lazy-evaluation PLANNING only (store records the cells it is asked
        # for): B8's final combination is chosen from outcomes, so every
        # (trigger, grid threshold, timing) combination is decided once here
        # and its cells are recorded.
        for trig, param in (("b4", "theta"), ("b5", "threshold"), ("b6", "threshold")):
            extra = {"model": unpriced["b5"]["chosen"].get("model")} if trig == "b5" else {}
            for value in unpriced[trig].get("grid") or []:
                for policy in POLICIES_V2:
                    _eval("b8_v2", {"trigger": trig, "timing": policy,
                                    "trigger_params": {**extra, param: value}},
                          turns, store, seeds, acc_profile, cfg, lm)
    tol = float(cfg["tuning"].get("equal_accuracy_tolerance", 0.02))
    floor = timing_scores["sentence_end"]["accuracy"] - tol
    eligible = [p for p in POLICIES_V2 if timing_scores[p]["accuracy"] >= floor]
    b8_timing = max(eligible, key=lambda p: timing_scores[p]["mean_hidden_ms"])
    unpriced["b8"] = {"chosen": {"trigger": best_trigger, "timing": b8_timing},
                      "timing_scores": timing_scores, "equal_accuracy_tolerance": tol,
                      "note": "trigger = best tuned of b4/b5/b6; timing tuned in isolation"}

    return {
        "_comment": "GENERATED by python -m controller.tune_v2 on the DEV split. Do not hand-edit.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "budget": budget,
        "seeds": list(seeds),
        "profiles": list(profiles),
        "n_turns": len(turns),
        "policy_constants": constants,
        "rho_table": fit_rho_table(turns, store, gold, labels),
        "empirical_quality": empirical_quality(turns, store, labels, seeds),
        "unpriced": unpriced,
        "priced": {"note": "ours_v2 / b9_v2 / b9plus_v2 have no per-price tunables; they share "
                           "policy_constants, rho_table and the config's quality/calibration."},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Equal-budget dev tuning for the v2 baselines.")
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--traces-dir", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--profiles", default="faiss")
    ap.add_argument("--seeds", default="11,23,37,53,71")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "controller" / "baselines" / "e3v2_estimates.json")
    args = ap.parse_args(argv)

    cfg = load_e3v2_config(overrides={"estimates_artifact": None})
    items = load_items(args.items)
    keep = split_ids(items, args.splits, "dev")  # tuning never reads test
    if not keep:
        raise SystemExit("no dev items found")
    turns = load_turns(args.signals, args.traces_dir, keep)
    store = CellStore.load(args.cells, bool(cfg["phase_b"]["query_variant_order_sensitive"]))
    labels = None
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as fh:
            labels = {str(k): bool(v) for k, v in json.load(fh).items() if str(k) in keep}
    gold = {i: list(it.get("gold_doc_ids") or []) for i, it in items.items() if i in keep}
    artifact = tune_all(turns, store, [int(s) for s in args.seeds.split(",")], args.profiles.split(","),
                        cfg, labels, gold, args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
