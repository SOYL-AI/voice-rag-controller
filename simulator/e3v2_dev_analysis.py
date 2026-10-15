"""
e3v2_dev_analysis.py — everything between Phase A and the freeze, on DEV only:

  1. calibrate the operational belief (self_assessed_need, frozen prompt
     sa-v1-frozen) on dev: prefix-aware logistic
         P(need) = sigmoid(a + b*logit(SA) + c*f + d*logit(SA)*f)
     against E1-v2's empirical need label, f = spoken word fraction;
  2. remaining-speech predictions R_hat out-of-fold (5 folds by item) with
     controller/remaining_speech.py's "qr" model, per condition;
  3. Phase B signals per (item, condition): trace timing, R_hat entries
     (usable from the partial's available_ms), the belief per partial
     (usable from available_ms + the measured unqueued self-assessment
     latency), the full-transcript belief at the end-of-turn decision;
  4. the belief-transition model (controller/baselines/_objective_v2.py
     fit_belief_transitions) from those dev sequences;
  5. g+ / g- from Phase A cells (final-transcript dispatch vs skip, by
     empirical need);
  6. equal-budget dev tuning (controller/tune_v2.py): policy constants per
     latency profile, rho table, unpriced thresholds;
  7. the PREREG §4.3 price grid per profile: lambda_l such that the late
     B9+ cut equals the dev end-of-turn belief quantiles {0.5,0.65,0.8,0.9},
     lambda_c = 0.05 (as analysis/v2_power.py):
         lambda_l = (b_q (g+ + g-) - g- - lambda_c) / ((E[L] + pi) / 1000);
  8. Phase B (simulator/e3v2_decide.py) for every controller x both
     architectures x the composite latency profiles x the grid;
  9. the PREREG §3 contrasts: Delta[COMMIT], Delta[SPEC], their interaction,
     m3 = 1/2 max(Delta[COMMIT], 0), D*[arch], with paired cluster-bootstrap
     standard errors (items; pair_id clusters for d2_support).

It picks nothing: no primary profile/price is selected and nothing is frozen.

    python -m simulator.e3v2_dev_analysis --dataset d2_support \
        --e1-results <repo>/llm/experiment1_v2/results
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    fit_belief_transitions,
    latency_dist_from_quantiles,
    load_e3v2_config,
)
from controller.remaining_speech import RemainingSpeechPredictor, speech_end_ms  # noqa: E402
from controller.sequential_v2 import TurnCtx, end_of_turn_ms  # noqa: E402
from controller.tune_v2 import tune_all, tune_b9plus_timing, tune_spec_trigger  # noqa: E402
from controller.sequential_v2 import PRICED_V2  # noqa: E402
from llm.experiment1_v2 import common as e1common  # noqa: E402
from llm.experiment1_v2.prefixes import normalize_text, spoken_word_fraction  # noqa: E402
from simulator.e3v2_decide import (  # noqa: E402
    CellStore,
    boundary_contrasts,
    load_latency_model,
    run_architectures,
    write_records,
)

MODEL_KEY = "mistral7b_v03"
SIGNAL = "self_assessed_need"
SIGNAL_X = "self_assessed_need_logit"
SEEDS = (11, 23, 37, 53, 71)
PROFILES = ("rewrite_faiss", "rewrite_faiss_rerank", "rewrite_remote")
QUANTILES = (0.5, 0.65, 0.8, 0.9)
LAMBDA_C = 0.05
ALL_CONTROLLERS = ("ours_v2", "b9_v2", "b9plus_v2", "b10_speculative_v2", "b11_speculative_gated_v2",
                   "b0_v2", "b1_v2_sentence_end", "b1_v2_immediate", "b1_v2_fixed_early_proxy",
                   "b1_v2_forecast", "b4_v2", "b5_v2", "b6_v2", "b8_v2", "oracle_v2")


def slug(cond: str) -> str:
    """Short, stable directory name for a condition (Windows MAX_PATH)."""
    return cond if len(cond) <= 12 else "asr_" + e1common.sha256_text(cond)[:8]


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(1 - eps, max(eps, float(p)))
    return math.log(p / (1 - p))


def fold_of(item_id: str, k: int = 5) -> int:
    return int(e1common.sha256_text("e3v2-dev-fold", item_id)[:8], 16) % k


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_inputs(ds: str, e1_results: Path, phase_a: Path, split: str = "dev"):
    """Inputs for one split. The self-assessment LATENCY is always the dev
    measurement (a frozen shared estimate); labels are optional off dev (they
    only feed the oracle)."""
    e1dir = e1_results / ds / MODEL_KEY
    sa = {}
    # Phase A's own self-assessment file (E1-v2's records plus the same frozen
    # probe run on the ASR partials E1-v2 did not cover) when present.
    sa_path = phase_a / "sa" / f"self_assessment_{split}.jsonl"
    if not sa_path.exists():
        sa_path = e1dir / f"self_assessment_{split}.jsonl"
    for r in e1common.read_jsonl(sa_path):
        sa[(r["item_id"], normalize_text(r["prefix_text"]))] = float(r[SIGNAL])
    lat = [float(r["ttft_ms"]) for r in e1common.read_jsonl(e1dir / "self_assessment_latency_dev.jsonl")
           if r.get("ttft_ms") is not None and not r.get("timing_synthetic")]
    sa_lag = float(np.median(lat)) if lat else 0.0
    labels_path = e1dir / f"labels_{split}.json"
    labels_doc = e1common.load_json(labels_path) if labels_path.exists() else {"items": {}, "meta": {}}
    labels = {i: v.get("empirical_need") for i, v in labels_doc["items"].items()}
    ent = defaultdict(lambda: defaultdict(list))
    for r in e1common.read_jsonl(e1dir / f"signals_{split}.jsonl"):
        s = r.get("signals") or {}
        key = (r["item_id"], normalize_text((r.get("prefix") or {}).get("text", "")))
        for name in ("mean_entropy_nats", "max_entropy_nats"):
            if s.get(name) is not None:
                ent[key][name].append(float(s[name]))
        if s.get("negative_min_logprob") is not None:
            ent[key]["min_token_prob"].append(math.exp(-float(s["negative_min_logprob"])))
    entropy = {k: {n: float(np.mean(v)) for n, v in d.items()} for k, d in ent.items()}
    traces = defaultdict(dict)
    for tr in e1common.read_jsonl(phase_a / f"conditions_{split}.jsonl"):
        traces[tr["condition"]][tr["item_id"]] = tr
    items = {}
    paths = e1common.resolve_dataset_paths(ds)
    for it in e1common.load_items(paths["items"]):
        items[it["id"]] = it
    return {"sa": sa, "sa_lag_ms": sa_lag, "labels": labels, "entropy": entropy, "traces": dict(traces),
            "items": items, "labels_meta": labels_doc.get("meta", {}), "e1dir": e1dir}


def prefix_frac(tr: dict, k: Optional[int]) -> float:
    if k is None:
        return 1.0
    p = tr["partials"][k]
    if p.get("word_frac") is not None:
        return float(p["word_frac"])
    f = spoken_word_fraction(tr, p["t_ms"])
    return 0.0 if f is None else float(f)


# ---------------------------------------------------------------------------
# 1. Calibration
# ---------------------------------------------------------------------------
def fit_calibration(inp: dict, ds: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y, groups, fr = [], [], [], []
    for cond, traces in inp["traces"].items():
        for item_id, tr in traces.items():
            lab = inp["labels"].get(item_id)
            if lab is None:
                continue
            pts = [(k, normalize_text(p["text"])) for k, p in enumerate(tr["partials"])]
            pts.append((None, normalize_text(tr["asr_final"]["text"])))
            for k, text in pts:
                v = inp["sa"].get((item_id, text))
                if v is None:
                    continue
                f = prefix_frac(tr, k)
                x = logit(v)
                X.append([x, f, x * f])
                y.append(int(bool(lab)))
                groups.append(item_id)
                fr.append(f)
    X, y, fr = np.asarray(X), np.asarray(y), np.asarray(fr)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    # grouped 5-fold out-of-fold AUC (by item)
    oof = np.zeros(len(y))
    folds = np.array([fold_of(g) for g in groups])
    for k in range(5):
        tr_, te = folds != k, folds == k
        if not te.any():
            continue
        if len(set(y[tr_])) < 2:  # degenerate fold (tiny data only)
            oof[te] = p[te]
            continue
        oof[te] = LogisticRegression(C=1e6, max_iter=1000).fit(X[tr_], y[tr_]).predict_proba(X[te])[:, 1]
    by_frac = {}
    for lo, hi in ((0, 0.4), (0.4, 0.8), (0.8, 0.999), (0.999, 1.01)):
        m = (fr >= lo) & (fr < hi)
        if m.sum() > 10 and len(set(y[m])) == 2:
            by_frac[f"[{lo},{hi})"] = {"n": int(m.sum()), "auc_oof": float(roc_auc_score(y[m], oof[m]))}
    intercept = float(model.intercept_[0])
    slope, frac_slope, inter = (float(c) for c in model.coef_[0])
    return {"calibration": {
        "signal": SIGNAL_X, "kind": "logistic", "intercept": intercept, "slope": slope,
        "frac_slope": frac_slope, "interaction": inter, "calibrated": slope > 0,
        "source": (f"Fitted on {ds} dev ({len(set(groups))} items, {len(y)} prefix records over "
                   f"{len(inp['traces'])} conditions): P(empirical need) = sigmoid(a + b*logit(SA) + c*f + "
                   f"d*logit(SA)*f), SA = self_assessed_need (sa-v1-frozen), f = spoken word fraction"),
        "auc": float(roc_auc_score(y, p))},
        "diagnostics": {"auc_in_sample": float(roc_auc_score(y, p)), "auc_oof_grouped5": float(roc_auc_score(y, oof)),
                        "auc_oof_by_frac": by_frac, "n_records": int(len(y)), "n_items": len(set(groups)),
                        "base_rate": float(y.mean())}}


# ---------------------------------------------------------------------------
# 2-3. R_hat out of fold, and signals
# ---------------------------------------------------------------------------
def oof_remaining(traces: Dict[str, dict]) -> Dict[str, List[dict]]:
    out = {}
    trs = list(traces.values())
    for k in range(5):
        train = [t for t in trs if fold_of(t["item_id"]) != k]
        test = [t for t in trs if fold_of(t["item_id"]) == k]
        if not test:
            continue
        pred = RemainingSpeechPredictor.fit(train, model="qr")
        for t in test:
            entries = []
            for p in t["partials"]:
                q = pred.predict(p["text"], float(p["t_ms"]))
                entries.append({"t_ms": float(p["t_ms"]), "available_ms": float(p.get("available_ms", p["t_ms"])),
                                "remaining_ms": float(q["q50"]), "remaining_q10_ms": float(q["q10"]),
                                "remaining_q90_ms": float(q["q90"])})
            out[t["item_id"]] = entries
    return out


def build_signals(inp: dict, cond: str, rhat: Dict[str, List[dict]]) -> List[dict]:
    rows = []
    for item_id, tr in inp["traces"][cond].items():
        se = speech_end_ms(tr)
        vad = tr.get("vad_user_end_ms")
        if vad is None:  # VAD endpoint missed: the final transcript's time stands in
            vad = (tr.get("asr_final") or {}).get("t_ms", se)
        timing = {"speech_onset_ms": 0, "speech_end_ms": se, "user_end_ms": float(vad),
                  "partials": [{"t_ms": p["t_ms"], "available_ms": p.get("available_ms", p["t_ms"]),
                                "text": p["text"]} for p in tr["partials"]],
                  "asr_final": tr["asr_final"], "partial_interval_ms": tr.get("partial_interval_ms"),
                  "predicted_remaining_speech_ms": rhat.get(item_id, []),
                  "endpoint_reason": tr.get("endpoint_reason")}
        by_prefix = []
        for k, p in enumerate(tr["partials"]):
            text = normalize_text(p["text"])
            v = inp["sa"].get((item_id, text))
            if v is None:
                continue
            e = {"t_ms": float(p["t_ms"]),
                 "available_ms": float(p.get("available_ms", p["t_ms"])) + inp["sa_lag_ms"],
                 "kind": "partial", "word_frac": prefix_frac(tr, k), SIGNAL: v, SIGNAL_X: logit(v)}
            e.update(inp["entropy"].get((item_id, text), {}))
            by_prefix.append(e)
        final_text = normalize_text(tr["asr_final"]["text"])
        v = inp["sa"].get((item_id, final_text))
        ctx = TurnCtx(item_id=item_id, signals={"timing": timing})
        eot = end_of_turn_ms(ctx, {"timing": {"eot_detection_ms": 0}})
        if v is not None:
            # the end-of-turn decision waits for the final transcript's belief
            # (same for every controller; its call latency is not charged)
            e = {"t_ms": float(eot), "available_ms": float(eot), "kind": "full", "word_frac": 1.0,
                 SIGNAL: v, SIGNAL_X: logit(v)}
            question = normalize_text(inp["items"][item_id]["question"])
            e.update(inp["entropy"].get((item_id, final_text)) or inp["entropy"].get((item_id, question), {}))
            by_prefix.append(e)
        rows.append({"item_id": item_id, "condition": cond, "transcript_final": final_text,
                     "signals": {"model": {"available": True, "uncertainty_by_prefix": by_prefix}, "timing": timing},
                     "end_gap_ms": float(eot) - se})
    return rows


# ---------------------------------------------------------------------------
# 4-5. Transitions, g+/g-
# ---------------------------------------------------------------------------
def belief_sequences(rows: List[dict], cal: dict) -> List[list]:
    c = cal["calibration"]

    def b(e):
        x, f = e[SIGNAL_X], float(e.get("word_frac") or 0.0)
        z = c["intercept"] + c["slope"] * x + c["frac_slope"] * f + c["interaction"] * x * f
        return 1 / (1 + math.exp(-max(-40, min(40, z))))

    seqs = []
    for r in rows:
        es = sorted(r["signals"]["model"]["uncertainty_by_prefix"], key=lambda e: (e["kind"] == "full", e["t_ms"]))
        seq = [(float(e.get("word_frac") or 0.0) if e["kind"] != "full" else 1.0, b(e)) for e in es]
        if len(seq) >= 2:
            seqs.append(seq)
    return seqs


def g_estimates(store: CellStore, labels: dict, cond: str, items: Sequence[str]) -> dict:
    gains = {True: [], False: []}
    for item in items:
        need = labels.get(item)
        rec = store.retrievals.get((item, cond, None))
        if need is None or rec is None:
            continue
        diffs = []
        for s in SEEDS:
            try:
                r = store.cell(item, s, cond, "retrieve", rec["query_variant"])["answer_correct"]
                k = store.cell(item, s, cond, "skip", "none")["answer_correct"]
            except KeyError:
                continue
            diffs.append(float(r) - float(k))
        if diffs:
            gains[bool(need)].append(float(np.mean(diffs)))
    return {"g_plus": float(np.mean(gains[True])) if gains[True] else None,
            "g_minus": float(-np.mean(gains[False])) if gains[False] else None,
            "n_needed": len(gains[True]), "n_not_needed": len(gains[False]),
            "source": f"Phase A cells, condition {cond}: final-transcript dispatch vs skip, per item mean over seeds"}


# ---------------------------------------------------------------------------
# 7. Price grid
# ---------------------------------------------------------------------------
def price_grid(final_beliefs: Sequence[float], g: dict, cfg: dict) -> Dict[str, list]:
    gp, gm = g["g_plus"], g["g_minus"]
    pi = float(cfg["nonhideable_increment_ms"])
    out = {}
    for prof in PROFILES:
        lat = latency_dist_from_quantiles(cfg["latency_profiles"][prof], int(cfg["latency_points"]), prof)
        x_s = (lat.mean_ms + pi) / 1000.0
        rows = []
        for q in QUANTILES:
            b = float(np.quantile(final_beliefs, q))
            lam = (b * (gp + gm) - gm - LAMBDA_C) / x_s
            rows.append({"quantile": q, "b_cut": b, "lambda_l": lam, "lambda_c": LAMBDA_C,
                         "x_late_ms": lat.mean_ms + pi, "feasible": lam > 0})
        out[prof] = rows
    return out


# ---------------------------------------------------------------------------
# 9. Bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap_se(per_item: Dict[str, float], cluster_of: Dict[str, str], B: int = 2000,
                         seed: int = 20261016) -> Optional[float]:
    if len(per_item) < 3:
        return None
    clusters = defaultdict(list)
    for i, v in per_item.items():
        clusters[cluster_of.get(i) or i].append(v)
    keys = list(clusters)
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        vals = []
        for k in (rng.choice(keys) for _ in keys):
            vals.extend(clusters[k])
        means.append(sum(vals) / len(vals))
    return float(np.std(means, ddof=1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_dataset(args, ds: str, record: Optional[set] = None) -> dict:
    """The dev analysis. With `record` (a set) it is the LAZY-EVALUATION
    PLAN: the identical pipeline runs against a store that records every
    cell it is asked for and answers missing ones with a placeholder; all
    outputs go to analysis_plan/ and no artifact is overwritten."""
    t0 = time.perf_counter()
    phase_a = Path(args.phase_a) / ds
    plan = record is not None
    out = phase_a / ("analysis_plan" if plan else "analysis")
    out.mkdir(parents=True, exist_ok=True)
    inp = load_inputs(ds, Path(args.e1_results), phase_a)
    store = CellStore.load(phase_a / "store_dev.jsonl", recording=record)
    dev_items = sorted({i for traces in inp["traces"].values() for i in traces})
    print(f"[{ds}] {len(dev_items)} dev items, conditions {list(inp['traces'])}, SA lag {inp['sa_lag_ms']:.0f} ms")

    cal = fit_calibration(inp, ds)
    art_dir = out if plan else _REPO_ROOT / "controller" / "baselines"
    cal_path = art_dir / f"e3v2_calibration_{ds}_{MODEL_KEY}_{SIGNAL_X}.json"
    e1common.atomic_write_json(cal_path, {**cal, "generated_by": "python -m simulator.e3v2_dev_analysis",
                                          "fitted_at_utc": e1common.now_utc(), "split": "dev"})
    print(f"[{ds}] calibration: {cal['calibration']} oof AUC {cal['diagnostics']['auc_oof_grouped5']:.3f}")

    signals_by_cond, seqs = {}, []
    for cond, traces in inp["traces"].items():
        rows = build_signals(inp, cond, oof_remaining(traces))
        # the remaining-speech predictor a frozen (test) run applies: fitted
        # on ALL dev traces of this condition
        RemainingSpeechPredictor.fit(list(traces.values()), model="qr").save(
            out / f"remaining_speech_{slug(cond)}.json")
        signals_by_cond[cond] = rows
        seqs += belief_sequences(rows, cal)
        with open(out / f"signals_{slug(cond)}_dev.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    trans = fit_belief_transitions(seqs, n_bins=10, provenance={
        "kind": "fitted", "generated_by": "python -m simulator.e3v2_dev_analysis", "dataset": ds,
        "split": "dev", "n_sequences": len(seqs), "calibration": str(cal_path.name),
        "fitted_at_utc": e1common.now_utc()})
    trans_path = art_dir / f"belief_transitions_v2_{ds}.json"
    e1common.atomic_write_json(trans_path, trans.to_json())

    primary = args.primary_condition or next(c for c in inp["traces"] if c != "text150") \
        if any(c != "text150" for c in inp["traces"]) else "text150"
    g = g_estimates(store, inp["labels"], primary, dev_items)
    if args.g_plus is not None or args.g_minus is not None:  # plumbing tests only
        g = {**g, "g_plus": args.g_plus, "g_minus": args.g_minus, "source": "CLI OVERRIDE (not data)"}
    g_text = g_estimates(store, inp["labels"], "text150", dev_items) if "text150" in inp["traces"] else None
    print(f"[{ds}] g on {primary}: {g}")

    results = {"dataset": ds, "primary_condition": primary, "calibration": cal, "g": g, "g_text150": g_text,
               "sa_lag_ms": inp["sa_lag_ms"], "conditions": {}}
    labels = {i: bool(v) for i, v in inp["labels"].items() if v is not None and i in set(dev_items)}
    gold = {i: list(inp["items"][i].get("gold_doc_ids") or []) for i in dev_items}
    cluster_of = {i: (inp["items"][i].get("pair_id") or i) for i in dev_items}

    for cond, rows in signals_by_cond.items():
        end_gap = float(np.median([r["end_gap_ms"] for r in rows]))
        overrides = {
            "quality": {"g_plus": g["g_plus"], "g_minus": g["g_minus"]},
            "calibration": {"inline": cal["calibration"], "signal": SIGNAL_X},
            "ours": {"belief_transitions_artifact": str(trans_path)},
            "timing": {"eot_detection_ms": 0, "end_gap_ms": end_gap},
            "estimates_artifact": None,
        }
        cfg = load_e3v2_config(overrides=overrides)
        turns = [TurnCtx(item_id=r["item_id"], signals=r["signals"], condition=cond,
                         transcript_final=r["transcript_final"]) for r in rows]
        lm = load_latency_model(cfg)
        est = tune_all(turns, store, SEEDS, list(PROFILES), cfg, labels, gold, budget=21, latency_model=lm,
                       plan_all_b8=plan)
        est_path = out / f"estimates_{slug(cond)}_dev.json"
        e1common.atomic_write_json(est_path, est)
        cfg = load_e3v2_config(overrides={**overrides, "estimates_artifact": str(est_path)})
        # end-of-turn calibrated beliefs (for the grid)
        c = cal["calibration"]
        finals = []
        for r in rows:
            full = [e for e in r["signals"]["model"]["uncertainty_by_prefix"] if e["kind"] == "full"]
            if full:
                x = full[0][SIGNAL_X]  # f = 1 at the end of turn
                z = c["intercept"] + c["frac_slope"] + (c["slope"] + c["interaction"]) * x
                finals.append(1 / (1 + math.exp(-max(-40, min(40, z)))))
        grid = price_grid(finals, g, cfg)
        cond_res = {"slug": slug(cond), "end_gap_ms_median": end_gap, "n_turns": len(turns), "price_grid": grid,
                    "estimates": str(est_path), "profiles": {}}
        for prof in PROFILES:
            prices = [{"lambda_l": row["lambda_l"], "lambda_c": LAMBDA_C} for row in grid[prof] if row["feasible"]]
            if not prices:
                cond_res["profiles"][prof] = {"note": "no feasible price"}
                continue
            # PREREG §4.5 per-price tuning on dev: B9+'s timing rule and the
            # shared SPEC / SPEC-theta trigger; priced controllers then run one
            # price at a time with their tuned parameters, unpriced ones once.
            unpriced = [n for n in ALL_CONTROLLERS if n not in PRICED_V2]
            priced = [n for n in ALL_CONTROLLERS if n in PRICED_V2]
            recs = run_architectures(turns, store, unpriced, [], SEEDS, [prof], cfg, lm,
                                     ("commit", "speculative"), "dev", labels, gold)
            tuning = {}
            for pr in prices:
                b9p = tune_b9plus_timing(turns, store, SEEDS, prof, pr, cfg, lm)
                trig = tune_spec_trigger(turns, store, SEEDS, prof, pr, cfg, lm)
                tuning[f"{pr['lambda_l']:g}"] = {"b9plus_timing_rule": b9p, "spec_trigger": trig}
                params = {"b9plus_v2": {"timing_rule": b9p["chosen"]},
                          "b10_speculative_v2": {"timing": trig["chosen"]},
                          "b11_speculative_gated_v2": {"timing": trig["chosen"]}}
                recs += run_architectures(turns, store, priced, [pr], SEEDS, [prof], cfg, lm,
                                          ("commit", "speculative"), "dev", labels, gold, controller_params=params)
            write_records(recs, out / "phase_b" / slug(cond) / prof)
            contrasts = boundary_contrasts(recs, "dev")
            prof_rows = []
            for crow in contrasts:
                per = crow["per_item"]
                ses = {k: cluster_bootstrap_se(per[k2], cluster_of)
                       for k, k2 in (("commit", "delta_commit"), ("speculative", "delta_speculative"),
                                     ("interaction", "interaction"))}
                q = next((r["quantile"] for r in grid[prof] if f"{r['lambda_l']:g}|{LAMBDA_C:g}" == crow["price"]), None)
                prof_rows.append({
                    "belief_quantile": q, "price": crow["price"], "d_star": crow["d_star"],
                    "d_star_speculative_dev_utilities": crow.get("d_star_speculative_dev_utilities"),
                    "delta_commit": crow["delta"]["commit"], "se_commit": ses["commit"],
                    "delta_speculative": crow["delta"]["speculative"], "se_speculative": ses["speculative"],
                    "interaction": crow["interaction"], "se_interaction": ses["interaction"],
                    "z_commit": (crow["delta"]["commit"] / ses["commit"]) if ses["commit"] else None,
                    "m3": crow["m3"], "q_hat": crow["q_hat"], "w_bar": crow["w_bar"],
                    "m_c_qhat_diagnostic": crow.get("m_c_qhat"), "n_items": crow["n_items"],
                })
            # mean realised utility per controller x architecture at each price (context)
            util = defaultdict(list)
            for r in recs:
                if r.get("realized_utility") is not None:
                    util[(r["architecture"], r["baseline"], f"{r['prices']['lambda_l']:g}")].append(r["realized_utility"])
            acc = defaultdict(list)
            for r in recs:
                acc[(r["architecture"], r["baseline"], None if r["prices"] is None else f"{r['prices']['lambda_l']:g}")]\
                    .append((r["outcome"]["answer_correct"], r["outcome"]["retrievals"], r["outcome"]["ttfa_ms"]))
            cond_res["profiles"][prof] = {
                "contrasts": prof_rows,
                "tuning": {lam: {"b9plus_timing_rule": t["b9plus_timing_rule"]["chosen"],
                                 "b9plus_n_evaluated": t["b9plus_timing_rule"]["n_evaluated"],
                                 "b9plus_candidates": t["b9plus_timing_rule"]["candidates"],
                                 "spec_trigger": t["spec_trigger"]["chosen"],
                                 "spec_candidates": t["spec_trigger"]["candidates"]}
                           for lam, t in tuning.items()},
                "mean_utility": {f"{a}|{b}|{l}": float(np.mean(v)) for (a, b, l), v in util.items()},
                "operating_points": {f"{a}|{b}|{l}": {"accuracy": float(np.mean([x[0] for x in v])),
                                                      "retrieval_rate": float(np.mean([x[1] for x in v])),
                                                      "p95_ttfa_ms": float(np.percentile([x[2] for x in v], 95))}
                                     for (a, b, l), v in acc.items()},
            }
            print(f"[{ds}/{cond}/{prof}] " + "; ".join(
                f"q={r['belief_quantile']}: dC={r['delta_commit']:+.4f}(se {r['se_commit'] or 0:.4f}) "
                f"dS={r['delta_speculative']:+.4f} I={r['interaction']:+.4f} D*S={r['d_star']['speculative']}"
                for r in prof_rows))
        results["conditions"][cond] = cond_res
        # The frozen bundle a test run would apply (nothing here is chosen:
        # the primary profile/price, D*[SPEC] and m3 are fixed at freeze).
        bundle = {"dataset": ds, "condition": cond, "split_fitted": "dev",
                  "overrides": {**overrides, "estimates_artifact": str(est_path)},
                  "price_grid": grid, "sa_lag_ms": inp["sa_lag_ms"],
                  "remaining_speech_predictor": str(out / f"remaining_speech_{slug(cond)}.json"),
                  "tuned": {prof: (cond_res["profiles"].get(prof) or {}).get("tuning", {}) for prof in PROFILES}}
        e1common.atomic_write_json(out / f"bundle_{slug(cond)}.json", bundle)
    results["wall_s"] = time.perf_counter() - t0
    e1common.atomic_write_json(out / "dev_analysis.json", results)
    return results


def write_predictions(phase_a: Path, datasets: Sequence[str], out_path: Path) -> dict:
    """docs/research_v2/v2_predictions.json from the per-dataset dev analyses.
    Every quantity PREREG_V2 §3 names, per dataset x condition x composite
    profile x belief-quantile price; plus an item-weighted pool of the
    datasets' PRIMARY conditions per (profile, belief quantile) — lambda_l
    differs by dataset at the same quantile, so the pool is by quantile."""
    cells, inputs = [], {}
    for ds in datasets:
        res = e1common.load_json(Path(phase_a) / ds / "analysis" / "dev_analysis.json")
        inputs[ds] = {"primary_condition": res["primary_condition"], "g": res["g"], "g_text150": res["g_text150"],
                      "calibration": res["calibration"], "sa_lag_ms": res["sa_lag_ms"],
                      "calibration_artifact": f"controller/baselines/e3v2_calibration_{ds}_{MODEL_KEY}_{SIGNAL_X}.json",
                      "belief_transitions_artifact": f"controller/baselines/belief_transitions_v2_{ds}.json"}
        for cond, cr in res["conditions"].items():
            for prof, pr in cr["profiles"].items():
                grid = {r["quantile"]: r for r in cr["price_grid"][prof]}
                for row in pr.get("contrasts", []):
                    g = grid.get(row["belief_quantile"], {})
                    cells.append({"dataset": ds, "condition": cond,
                                  "condition_role": "primary" if cond == res["primary_condition"] else "secondary",
                                  "profile": prof, "belief_quantile": row["belief_quantile"],
                                  "b_cut": g.get("b_cut"), "lambda_l": g.get("lambda_l"), "lambda_c": LAMBDA_C,
                                  **{k: row[k] for k in ("d_star", "delta_commit", "se_commit", "delta_speculative",
                                                         "se_speculative", "interaction", "se_interaction", "z_commit",
                                                         "m3", "q_hat", "w_bar", "n_items",
                                                         "d_star_speculative_dev_utilities")}})
                for q, g in grid.items():
                    if not g["feasible"]:
                        cells.append({"dataset": ds, "condition": cond, "profile": prof, "belief_quantile": q,
                                      "b_cut": g["b_cut"], "lambda_l": g["lambda_l"],
                                      "note": "infeasible price (lambda_l <= 0 at this belief quantile)"})
    pooled = []
    prim = [c for c in cells if c.get("condition_role") == "primary" and "delta_commit" in c]
    for prof in PROFILES:
        for q in QUANTILES:
            rows = [c for c in prim if c["profile"] == prof and c["belief_quantile"] == q]
            if len(rows) < 2:
                continue
            w = [c["n_items"]["paired"] for c in rows]
            W = float(sum(w))

            def pool(key, se_key):
                vals = [c[key] for c in rows]
                ses = [c[se_key] for c in rows]
                if any(v is None for v in vals):
                    return None, None
                m = sum(wi * v for wi, v in zip(w, vals)) / W
                se = None if any(x is None for x in ses) else math.sqrt(sum((wi / W) ** 2 * x ** 2
                                                                            for wi, x in zip(w, ses)))
                return m, se

            dc, sc = pool("delta_commit", "se_commit")
            ds_, ss = pool("delta_speculative", "se_speculative")
            it, si = pool("interaction", "se_interaction")
            pooled.append({"profile": prof, "belief_quantile": q, "datasets": [c["dataset"] for c in rows],
                           "n_items": int(W), "delta_commit": dc, "se_commit": sc, "delta_speculative": ds_,
                           "se_speculative": ss, "interaction": it, "se_interaction": si,
                           "m3": None if dc is None else 0.5 * max(dc, 0.0),
                           "z_commit": (dc / sc) if (dc is not None and sc) else None,
                           "lambda_l_by_dataset": {c["dataset"]: c["lambda_l"] for c in rows},
                           "d_star_by_dataset": {c["dataset"]: c["d_star"] for c in rows}})
    doc = {
        "_comment": ("DEV-ONLY predicted gaps for PREREG_V2 §3, computed by python -m simulator.e3v2_dev_analysis "
                     "from Phase A cells (real Mistral-7B generations, real grading) and Phase B on dev. Nothing is "
                     "selected or frozen here: the primary (profile, price), D*[SPEC] and m3 are fixed at freeze."),
        "generated_at_utc": e1common.now_utc(), "git_commit": e1common.git_commit(), "split": "dev",
        "definitions": {
            "delta_commit": "mean_i [U_i(Ours-v2[COMMIT]) - U_i(B9+)]",
            "delta_speculative": "mean_i [U_i(Ours-v2[SPEC]) - U_i(D*[SPEC])], D*[SPEC] = dev-best of B9+, SPEC, SPEC-theta",
            "interaction": "mean_i [Delta_i[COMMIT] - Delta_i[SPEC]] on the same items",
            "m3": "1/2 max(Delta[COMMIT], 0) on dev (PREREG_V2 §3 P3 margin)",
            "U_i": "Q - lambda_l (TTFA - skip TTFA)/1000 - lambda_c n_retrievals, mean over 5 seeds",
            "se": "paired cluster bootstrap over items (pair_id clusters for d2_support), B=2000",
            "price_grid": "lambda_l = (b_q (g+ + g-) - g- - lambda_c) / ((E[L] + pi)/1000), b_q = dev end-of-turn belief quantile",
        },
        "inputs": inputs, "cells": cells, "pooled_primary": pooled,
    }
    e1common.atomic_write_json(out_path, doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="d2_support | d1_open | both comma-separated")
    ap.add_argument("--e1-results", default=str(_REPO_ROOT / "llm" / "experiment1_v2" / "results"))
    ap.add_argument("--phase-a", default=str(_REPO_ROOT / "simulator" / "e3v2_phase_a"))
    ap.add_argument("--primary-condition", default=None)
    ap.add_argument("--write-predictions", action="store_true",
                    help="only aggregate existing dev analyses into docs/research_v2/v2_predictions.json")
    ap.add_argument("--g-plus", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--g-minus", type=float, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.write_predictions:
        write_predictions(Path(args.phase_a), args.dataset.split(","),
                          _REPO_ROOT / "docs" / "research_v2" / "v2_predictions.json")
        return 0
    for ds in args.dataset.split(","):
        run_dataset(args, ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
