"""
v1_theory_check.py -- verify the research-v2 theory (docs/research_v2/THEORY.md)
against the committed v1 E3 sweep, by computation rather than assertion.

    PYTHONUTF8=1 python -m analysis.v1_theory_check

Reads only committed artifacts:
    configs/e3_price_grid.json                 the 26-point lambda_l grid
    simulator/e3_scenario_signals.json         frozen per-scenario beliefs
    simulator/scenarios.json                   retrieval_needed labels
    simulator/e3_sweep_results/e3_{ours,b9}_seed*.json   Phase B records
    simulator/e3_sweep_cache/answers/*.json    per (scenario, seed, path) correctness
    simulator/e3_sweep_cache/cells/*.json      per (scenario, seed, path, timing) TTFA

and the controllers themselves (controller/baselines/{ours,b9_decomposed,
_objective}.py), so every cut is the code's exact cut, not a re-derivation.

Writes:
    docs/research_v2/v1_theory_check.json
    docs/research_v2/v1_theory_check.png

Checks (numbered as in THEORY.md section 7):
  7.1  Prop. 1 on the real grid: Ours(lambda_i) vs B9(lambda_{i-1}), decision level.
  7.2  Prop. 1 off the grid: Ours(lambda) vs B9(lambda'(lambda)) on a dense
       log-spaced price set, with the exact map and with the approximate
       map lambda*(L-o*)/L.
  7.3  The grid recurrence: lambda_i/lambda_{i-1} and the fixed-point form.
  7.4  Band at every grid price with divergence_band()'s exact cuts; scenarios
       in band (predicted) vs observed Ours/B9 disagreements at the same price.
  7.5  Prop. 2: model-predicted utility gap vs bound vs realised gap; predicted
       vs observed accuracy gap (v1 objective and corrected objective).
  7.6  B9+ (retrieval priced at the expected exposed latency of its own timing
       optimizer's choice) vs Ours: decision-level identity at every price.
  7.7  Retrieval effect by need label (g+, g-) and the corrected threshold b*.
  7.8  Calibration of the belief on the 50 scenarios (why the realised in-band
       gap can have the wrong sign).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baseline import Ctx  # noqa: E402
from controller.baselines import _objective as obj  # noqa: E402
from controller.baselines.b9_decomposed import (  # noqa: E402
    DecomposedWithPrices,
    _optimize_timing,
)
from controller.baselines.ours import UnifiedController  # noqa: E402
import controller.baselines.b9_decomposed as _b9_mod  # noqa: E402
import controller.baselines.ours as _ours_mod  # noqa: E402

# The controllers re-read configs/config.yaml and the E1 calibration artifact
# on every decide(). That is correct but slow for ~10^5 decisions, so the
# (unchanged) config and calibration are loaded once and served from memory.
# This changes no value: it is the same dict and the same calibration record.
_CFG = obj.load_e3_config()
_CAL = obj.load_voi_calibration(_CFG)
obj.load_voi_calibration = lambda cfg=None: _CAL
for _m in (obj, _ours_mod, _b9_mod):
    _m.load_e3_config = lambda path=None: _CFG

SEEDS = (11, 23, 37, 53, 71)
B_BOOT = 10_000
RNG_SEED = 20261014
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "v1_theory_check.json"
OUT_PNG = _REPO_ROOT / "docs" / "research_v2" / "v1_theory_check.png"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ctx_for(entry: dict, prices: dict) -> Ctx:
    """Identical construction to simulator/e3_full_sweep.py's ctx_for()."""
    return Ctx(
        transcript=entry["question"],
        signals={
            "model": {"available": True, "uncertainty": entry["model_uncertainty"]},
            "tracker": entry["tracker"],
        },
        elapsed_ms=0,
        user_end_known=True,
        turn_id=1,
        conversation_id=f"v1_theory_check_{entry['scenario_id']:03d}",
        prices=prices,
    )


def load_all():
    grid = _read(_REPO_ROOT / "configs" / "e3_price_grid.json")["grid"]
    lam = [float(x) for x in grid["lambda_l_grid"]]
    lam_c = float(grid["lambda_c"])
    signals = _read(_REPO_ROOT / "simulator" / "e3_scenario_signals.json")["signals"]
    entries = {e["scenario_id"]: e for e in signals}
    scenarios = _read(_REPO_ROOT / "simulator" / "scenarios.json")
    need = {s["id"]: bool(s["retrieval_needed"]) for s in scenarios}
    sids = sorted(need)

    results = {}
    for name in ("ours", "b9"):
        for seed in SEEDS:
            recs = _read(_REPO_ROOT / "simulator" / "e3_sweep_results" / f"e3_{name}_seed{seed}.json")
            for r in recs:
                i = lam.index(float(r["prices"]["lambda_l"]))
                results[(name, seed, i, r["scenario_id"])] = r

    cache = _REPO_ROOT / "simulator" / "e3_sweep_cache"
    Y = {}  # (sid, seed, path) -> 0/1
    for sid in sids:
        for seed in SEEDS:
            for path in ("skip", "retrieve"):
                a = _read(cache / "answers" / f"s{sid:03d}_seed{seed}_{path}.json")
                Y[(sid, seed, path)] = int(bool(a["answer_correct"]))
    T = {}  # (sid, seed, path, timing) -> ttfa_ms
    for sid in sids:
        for seed in SEEDS:
            for path in ("skip", "retrieve"):
                for timing in obj.TIMING_ORDER:
                    c = _read(cache / "cells" / f"s{sid:03d}_seed{seed}_{path}_{timing}.json")
                    T[(sid, seed, path, timing)] = float(c["ttfa_ms"])
    return lam, lam_c, entries, need, sids, results, Y, T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def boot_mean_ci(x: np.ndarray, rng, B=B_BOOT, axis_items=0):
    """Percentile bootstrap CI of the mean over items (rows)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def b9plus_decide(entry, prices, cfg):
    """
    B9+ -- the decomposed controller the reviewer describes: its retrieval
    sub-optimizer prices latency at the EXPECTED EXPOSED latency of the timing
    its own (independent) timing sub-optimizer selects, instead of the full
    schedule. The timing sub-optimizer is B9's, unchanged. The retrieval
    sub-optimizer sees one scalar (the expected exposed latency of the default
    timing) and never sees a per-turn timing state, so it is still decomposed.
    """
    ctx = ctx_for(entry, prices)
    p = obj.read_prices(ctx, cfg)
    t_hat, _ = _optimize_timing(ctx, p, cfg)
    voi = obj.estimate_voi(ctx, cfg)
    u = obj.scalarized_utility(
        voi * (1.0 - obj.partial_query_risk(t_hat, cfg)),
        obj.exposed_latency_s(t_hat, cfg),
        obj.retrieval_compute_units(t_hat, cfg),
        p,
    )
    return (u > 0.0), t_hat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    cfg = obj.load_e3_config()
    lam, lam_c, entries, need, sids, results, Y, T = load_all()
    n_items = len(sids)
    K = len(lam)

    L = obj.retrieval_latency_s(cfg)
    timings = list(obj.TIMING_ORDER)
    o = {t: obj.overlap_s(t, cfg) for t in timings}
    w = {t: obj.waste_rate(t, cfg) for t in timings}
    rho = {t: obj.partial_query_risk(t, cfg) for t in timings}
    t_star = max(timings, key=lambda t: o[t])
    o_star, w_star = o[t_star], w[t_star]

    # Beliefs exactly as the controllers compute them.
    b = {sid: obj.estimate_voi(ctx_for(entries[sid], {}), cfg) for sid in sids}
    b_arr = np.array([b[s] for s in sids])
    need_arr = np.array([need[s] for s in sids], dtype=float)

    ours, b9 = UnifiedController(), DecomposedWithPrices()

    memo = {}

    def decide(ctrl, sid, lam_l):
        key = (type(ctrl).__name__, sid, float(lam_l))
        if key not in memo:
            d = ctrl.decide(ctx_for(entries[sid], {"lambda_l": float(lam_l), "lambda_c": lam_c}))
            memo[key] = (bool(d.retrieve), d.timing)
        return memo[key]

    out = {
        "constants": {
            "L_s": L, "overlap_s": o, "waste_rate": w, "partial_query_risk": rho,
            "t_star": t_star, "o_star_s": o_star, "lambda_c": lam_c,
            "n_scenarios": n_items, "n_needed": int(need_arr.sum()), "seeds": list(SEEDS),
            "grid_lambda_l": lam,
        }
    }

    # ---- 0. sweep records reproduce the controllers; decisions seed-invariant
    rec_mismatch = 0
    seed_variance = 0
    for name, ctrl in (("ours", ours), ("b9", b9)):
        for i in range(K):
            for sid in sids:
                live = decide(ctrl, sid, lam[i])
                logged = {(results[(name, s, i, sid)]["decision"]["retrieve"],
                           results[(name, s, i, sid)]["decision"]["timing"]) for s in SEEDS}
                if len(logged) != 1:
                    seed_variance += 1
                if live not in logged:
                    rec_mismatch += 1
    out["sanity"] = {
        "logged_decisions_not_reproduced_by_live_controllers": rec_mismatch,
        "decisions_varying_across_seeds": seed_variance,
        "n_decisions_checked": 2 * K * n_items,
    }

    # ---- 7.1 Prop 1 on the grid: Ours(i) vs B9(i-1)
    agree_r = agree_t = n_ret = 0
    per_price_shift = []
    for i in range(1, K):
        a = 0
        for sid in sids:
            ro, to = decide(ours, sid, lam[i])
            rb, tb = decide(b9, sid, lam[i - 1])
            if ro == rb:
                a += 1
                agree_r += 1
                if ro:
                    n_ret += 1
                    agree_t += int(to == tb)
        per_price_shift.append({"i": i, "agree": a, "of": n_items})

    # outcome-level frontier identity Ours(i) vs B9(i-1), per seed
    def op_point(name, seed, i, skip_timing_canonical=False):
        acc, ttfa, comp = [], [], []
        for sid in sids:
            r = results[(name, seed, i, sid)]
            acc.append(r["outcome"]["answer_correct"])
            comp.append(r["outcome"]["retrievals"])
            if skip_timing_canonical and not r["decision"]["retrieve"]:
                ttfa.append(T[(sid, seed, "skip", "sentence_end")])
            else:
                ttfa.append(r["outcome"]["ttfa_ms"])
        ttfa = np.array(ttfa)
        return (float(np.mean(acc)), float(np.percentile(ttfa, 95)),
                float(np.mean(ttfa)), float(np.mean(comp)))

    frontier_identical = {"accuracy_and_compute": 0, "p95_ttfa_as_logged": 0,
                          "p95_ttfa_skip_canonical": 0, "of": (K - 1) * len(SEEDS)}
    max_p95_diff_logged = 0.0
    for seed in SEEDS:
        for i in range(1, K):
            po = op_point("ours", seed, i)
            pb = op_point("b9", seed, i - 1)
            if po[0] == pb[0] and po[3] == pb[3]:
                frontier_identical["accuracy_and_compute"] += 1
            if abs(po[1] - pb[1]) < 1e-9:
                frontier_identical["p95_ttfa_as_logged"] += 1
            max_p95_diff_logged = max(max_p95_diff_logged, abs(po[1] - pb[1]))
            pc = op_point("ours", seed, i, True)
            pbc = op_point("b9", seed, i - 1, True)
            if abs(pc[1] - pbc[1]) < 1e-9 and abs(pc[2] - pbc[2]) < 1e-9:
                frontier_identical["p95_ttfa_skip_canonical"] += 1
    frontier_identical["max_abs_p95_ttfa_diff_ms_as_logged"] = max_p95_diff_logged

    # skip-cell TTFA noise: same skip decision, different timing label
    skip_noise = [T[(s, sd, "skip", "fixed_early_proxy")] - T[(s, sd, "skip", "sentence_end")]
                  for s in sids for sd in SEEDS]
    out["prop1_on_grid"] = {
        "comparisons": (K - 1) * n_items,
        "retrieve_agreement": agree_r,
        "timing_agreement_given_both_retrieve": agree_t,
        "both_retrieve": n_ret,
        "per_price": per_price_shift,
        "operating_points_Ours_i_vs_B9_im1": frontier_identical,
        "skip_cell_ttfa_fixed_early_minus_sentence_end_ms": {
            "mean": float(np.mean(skip_noise)), "sd": float(np.std(skip_noise, ddof=1)),
            "note": "B9 logs its timing-optimizer choice on skipped turns, Ours logs "
                    "sentence_end; the sweep reads a separately measured skip cell per "
                    "timing label, so identical retrieve decisions can carry different "
                    "measured TTFA. This is run-to-run noise in a no-retrieval cell."},
    }

    # ---- 7.2 Prop 1 off the grid (dense prices, exact and approximate map)
    dense = np.geomspace(0.005, 3.0, 200)
    exact_agree = approx_agree = total = 0
    approx_fail_prices = []
    for lam_l in dense:
        band = obj.divergence_band(obj.Prices(lam_l, lam_c), cfg)
        lam_exact = (band.ours_cut - lam_c * obj.retrieval_compute_units("sentence_end", cfg)) / L
        lam_approx = lam_l * (L - o_star) / L
        fails = 0
        for sid in sids:
            ro, _ = decide(ours, sid, float(lam_l))
            if lam_exact >= 0:
                rb, _ = decide(b9, sid, float(lam_exact))
                exact_agree += int(ro == rb)
            ra, _ = decide(b9, sid, float(lam_approx))
            approx_agree += int(ro == ra)
            fails += int(ro != ra)
            total += 1
        if fails:
            approx_fail_prices.append({"lambda_l": float(lam_l), "disagreements": fails})
    # minimum price above which the exact map is non-negative (Ours cut >= B9's floor)
    out["prop1_dense"] = {
        "n_prices": len(dense), "price_range": [float(dense[0]), float(dense[-1])],
        "decisions": total,
        "exact_map_agreement": exact_agree,
        "approx_map_agreement": approx_agree,
        "approx_map_disagreement_prices": approx_fail_prices[:20],
        "n_prices_where_approx_map_fails": len(approx_fail_prices),
        "exact_map": "lambda' = [lambda*(L-o*) + lambda_c*w*]/L  (= lambda*(L-o*)/L + lambda_c*w*/L)",
    }

    # ---- 7.3 grid recurrence
    lam_fix = lam_c * (w_star - w["sentence_end"]) / o_star
    ratios = [lam[i] / lam[i - 1] for i in range(1, K)]
    fixed_ratios = [(lam[i] - lam_fix) / (lam[i - 1] - lam_fix) for i in range(1, K)]
    out["grid_recurrence"] = {
        "L_over_L_minus_o_star": L / (L - o_star),
        "fixed_point_lambda_deg": lam_fix,
        "raw_ratio_min": min(ratios), "raw_ratio_max": max(ratios),
        "raw_ratios": ratios,
        "fixed_point_ratio_min": min(fixed_ratios), "fixed_point_ratio_max": max(fixed_ratios),
        "ours_cut_i_equals_b9_cut_im1_max_abs_err": max(
            abs(obj.divergence_band(obj.Prices(lam[i], lam_c), cfg).ours_cut
                - obj.divergence_band(obj.Prices(lam[i - 1], lam_c), cfg).b9_cut)
            for i in range(1, K)),
    }

    # ---- 7.4 band per price, predicted vs observed disagreement
    band_rows = []
    for i in range(K):
        band = obj.divergence_band(obj.Prices(lam[i], lam_c), cfg)
        in_band = [sid for sid in sids if band.contains(b[sid])]
        dis = [sid for sid in sids if decide(ours, sid, lam[i])[0] != decide(b9, sid, lam[i])[0]]
        band_rows.append({
            "i": i, "lambda_l": lam[i], "ours_cut": band.ours_cut, "b9_cut": band.b9_cut,
            "width": band.width, "timing": band.timing,
            "n_in_band": len(in_band), "frac_in_band": len(in_band) / n_items,
            "n_observed_disagreements": len(dis),
            "disagreements_subset_of_band": set(dis) <= set(in_band),
            "disagreements_equal_band": set(dis) == set(in_band),
            "in_band_ids": in_band,
            "n_ours_retrieve": sum(decide(ours, s, lam[i])[0] for s in sids),
            "n_b9_retrieve": sum(decide(b9, s, lam[i])[0] for s in sids),
        })
    out["band_per_price"] = band_rows
    out["band_summary"] = {
        "prices_with_nonempty_band": sum(r["n_in_band"] > 0 for r in band_rows),
        "total_in_band_scenario_prices": sum(r["n_in_band"] for r in band_rows),
        "mean_frac_in_band": float(np.mean([r["frac_in_band"] for r in band_rows])),
        "all_disagreements_equal_band": all(r["disagreements_equal_band"] for r in band_rows),
        "distinct_scenarios_ever_in_band": len({s for r in band_rows for s in r["in_band_ids"]}),
    }

    # ---- 7.7 retrieval effect by need label
    eff = np.array([[Y[(s, sd, "retrieve")] - Y[(s, sd, "skip")] for sd in SEEDS] for s in sids], float)
    eff_item = eff.mean(axis=1)
    g_plus = boot_mean_ci(eff_item[need_arr == 1], rng)
    g_minus_raw = boot_mean_ci(eff_item[need_arr == 0], rng)
    gp, gm = g_plus[0], -g_minus_raw[0]
    acc_skip = np.array([[Y[(s, sd, "skip")] for sd in SEEDS] for s in sids], float).mean(1)
    acc_ret = np.array([[Y[(s, sd, "retrieve")] for sd in SEEDS] for s in sids], float).mean(1)
    b_star0 = gm / (gp + gm)
    out["retrieval_effect"] = {
        "g_plus_needed": {"mean": g_plus[0], "ci95": g_plus[1:]},
        "g_minus_not_needed": {"mean": gm, "ci95_of_effect": g_minus_raw[1:],
                               "note": "g- is reported positive; the effect itself is -g-"},
        "acc_skip_needed": float(acc_skip[need_arr == 1].mean()),
        "acc_retrieve_needed": float(acc_ret[need_arr == 1].mean()),
        "acc_skip_not_needed": float(acc_skip[need_arr == 0].mean()),
        "acc_retrieve_not_needed": float(acc_ret[need_arr == 0].mean()),
        "b_star_at_zero_cost": b_star0,
        "frac_v1_beliefs_within_0.15_of_b_star": float(np.mean(np.abs(b_arr - b_star0) <= 0.15)),
        "frac_v1_beliefs_below_b_star": float(np.mean(b_arr < b_star0)),
    }

    # ---- 7.8 calibration of b on the 50 scenarios
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (b_arr >= lo) & (b_arr < hi)
        if m.sum():
            calib.append({"bin": [lo, min(hi, 1.0)], "n": int(m.sum()),
                          "mean_b": float(b_arr[m].mean()), "need_rate": float(need_arr[m].mean()),
                          "mean_retrieval_effect": float(eff_item[m].mean())})
    out["calibration"] = {
        "brier": float(np.mean((b_arr - need_arr) ** 2)),
        "mean_b": float(b_arr.mean()), "need_rate": float(need_arr.mean()),
        "bins": calib,
        "b_needed_min": float(b_arr[need_arr == 1].min()),
        "b_not_needed_max": float(b_arr[need_arr == 0].max()),
        "n_not_needed_with_b_gt_0.5": int(((b_arr > 0.5) & (need_arr == 0)).sum()),
    }

    # ---- 7.5a measured latency cost of retrieval in the E3 cells
    def mean_diff(f):
        v = np.array([f(s, sd) for s in sids for sd in SEEDS]) / 1000.0
        return float(v.mean()), float(v.std(ddof=1))
    meas = {}
    for t in timings:
        meas[t] = {
            "ttfa_ret_t_minus_skip_sentence_end_s": mean_diff(
                lambda s, sd: T[(s, sd, "retrieve", t)] - T[(s, sd, "skip", "sentence_end")]),
            "ttfa_ret_t_minus_skip_t_s": mean_diff(
                lambda s, sd: T[(s, sd, "retrieve", t)] - T[(s, sd, "skip", t)]),
        }
    meas["realised_overlap_saving_s(ret@sentence_end - ret@fixed_early)"] = mean_diff(
        lambda s, sd: T[(s, sd, "retrieve", "sentence_end")] - T[(s, sd, "retrieve", "fixed_early_proxy")])
    meas["model_exposed_latency_s"] = {t: obj.exposed_latency_s(t, cfg) for t in timings}
    out["measured_latency"] = meas
    dT_star = meas[t_star]["ttfa_ret_t_minus_skip_sentence_end_s"][0]
    n_ours_ret_total = sum(1 for i in range(K) for s in sids if decide(ours, s, lam[i])[0])

    # ---- 7.5 Prop 2: predicted vs observed gap per price
    prop2 = []
    for i, row in enumerate(band_rows):
        band = obj.divergence_band(obj.Prices(lam[i], lam_c), cfg)
        m = np.array([s in row["in_band_ids"] for s in sids])
        p_band = m.mean()
        c_O, c_B = band.ours_cut, band.b9_cut
        pred_u_v1 = float(np.sum((b_arr - c_O) * m) / n_items)          # E[(b - c_O) 1{band}]
        bound_tight = float((c_B - c_O) * p_band)
        bound_loose = float(lam[i] * o_star * p_band)
        pred_acc_v1 = float(np.sum(b_arr * m) / n_items)               # v1 model: quality gain = b
        pred_acc_corr = float(np.sum((b_arr * gp - (1 - b_arr) * gm) * m) / n_items)
        pred_acc_oracle = float(np.sum((need_arr * gp - (1 - need_arr) * gm) * m) / n_items)
        # realised-cost prediction: corrected quality model + measured TTFA cost
        pred_u_meas = float(np.sum((b_arr * gp - (1 - b_arr) * gm - lam[i] * dT_star - lam_c) * m) / n_items)
        pred_u_meas_oracle = float(np.sum((eff_item - lam[i] * dT_star - lam_c) * m) / n_items)
        # observed: per item, mean over seeds of Ours - B9 at the same price
        d_acc = np.zeros(n_items)
        d_u = np.zeros(n_items)
        d_u_canon = np.zeros(n_items)
        for k, sid in enumerate(sids):
            da = du = duc = 0.0
            for sd in SEEDS:
                ro = results[("ours", sd, i, sid)]
                rb = results[("b9", sd, i, sid)]
                a = ro["outcome"]["answer_correct"] - rb["outcome"]["answer_correct"]
                dt = (ro["outcome"]["ttfa_ms"] - rb["outcome"]["ttfa_ms"]) / 1000.0
                dc = ro["outcome"]["retrievals"] - rb["outcome"]["retrievals"]
                # skip cells normalised to one timing label
                def tt(r):
                    if r["decision"]["retrieve"]:
                        return r["outcome"]["ttfa_ms"] / 1000.0
                    return T[(sid, sd, "skip", "sentence_end")] / 1000.0
                dtc = tt(ro) - tt(rb)
                da += a
                du += a - lam[i] * dt - lam_c * dc
                duc += a - lam[i] * dtc - lam_c * dc
            d_acc[k], d_u[k], d_u_canon[k] = da / 5, du / 5, duc / 5
        obs_acc = boot_mean_ci(d_acc, rng)
        obs_u = boot_mean_ci(d_u_canon, rng)
        prop2.append({
            "i": i, "lambda_l": lam[i], "p_band": float(p_band),
            "pred_utility_gap_model": pred_u_v1,
            "bound_(cB-cO)P": bound_tight, "bound_lambda_o_star_P": bound_loose,
            "pred_acc_gap_v1_model": pred_acc_v1,
            "pred_acc_gap_corrected_model": pred_acc_corr,
            "pred_acc_gap_oracle_labels": pred_acc_oracle,
            "obs_acc_gap": obs_acc[0], "obs_acc_gap_ci95": obs_acc[1:],
            "obs_utility_gap_skipcanonical": obs_u[0], "obs_utility_gap_ci95": obs_u[1:],
            "obs_utility_gap_as_logged": float(d_u.mean()),
            "pred_utility_gap_corrected_measured_cost": pred_u_meas,
            "pred_utility_gap_item_effects_measured_cost": pred_u_meas_oracle,
        })
    out["prop2"] = prop2
    nz = [r for r in prop2 if r["p_band"] > 0]
    out["prop2_summary"] = {
        "prices_with_band_mass": len(nz),
        "max_pred_utility_gap_model": max(r["pred_utility_gap_model"] for r in prop2),
        "max_bound_lambda_o_star_P": max(r["bound_lambda_o_star_P"] for r in prop2),
        "bound_holds_everywhere_for_model_gap": all(
            r["pred_utility_gap_model"] <= r["bound_(cB-cO)P"] + 1e-12 <= r["bound_lambda_o_star_P"] + 1e-12
            for r in prop2),
        "corr_pred_corrected_vs_obs_acc": float(np.corrcoef(
            [r["pred_acc_gap_corrected_model"] for r in nz], [r["obs_acc_gap"] for r in nz])[0, 1]) if len(nz) > 2 else None,
        "corr_oracle_vs_obs_acc": float(np.corrcoef(
            [r["pred_acc_gap_oracle_labels"] for r in nz], [r["obs_acc_gap"] for r in nz])[0, 1]) if len(nz) > 2 else None,
        "mean_obs_acc_gap_over_band_prices": float(np.mean([r["obs_acc_gap"] for r in nz])),
        "mean_pred_acc_gap_v1_over_band_prices": float(np.mean([r["pred_acc_gap_v1_model"] for r in nz])),
        "mean_pred_acc_gap_corrected_over_band_prices": float(np.mean([r["pred_acc_gap_corrected_model"] for r in nz])),
        "obs_acc_ci_excludes_zero_at": [r["i"] for r in nz if r["obs_acc_gap_ci95"][0] > 0 or r["obs_acc_gap_ci95"][1] < 0],
        "utility_pred_measured_vs_obs": {
            "pearson": float(np.corrcoef([r["pred_utility_gap_corrected_measured_cost"] for r in nz],
                                         [r["obs_utility_gap_skipcanonical"] for r in nz])[0, 1]),
            "mean_abs_err": float(np.mean([abs(r["pred_utility_gap_corrected_measured_cost"]
                                               - r["obs_utility_gap_skipcanonical"]) for r in nz])),
            "n_obs_inside_ci_of_pred": int(sum(r["obs_utility_gap_ci95"][0] <= r["pred_utility_gap_corrected_measured_cost"]
                                               <= r["obs_utility_gap_ci95"][1] for r in nz)),
            "n_prices": len(nz),
            "sign_agreement": int(sum(np.sign(r["pred_utility_gap_corrected_measured_cost"])
                                      == np.sign(r["obs_utility_gap_skipcanonical"]) for r in nz)),
        },
        "utility_pred_v1model_sign_vs_obs_sign_agreement": int(sum(
            np.sign(r["pred_utility_gap_model"]) == np.sign(r["obs_utility_gap_skipcanonical"]) for r in nz)),
    }

    # ---- 7.6 B9+ reproduces Ours
    same = 0
    same_t = 0
    tot = 0
    for i in range(K):
        for sid in sids:
            rp, tp = b9plus_decide(entries[sid], {"lambda_l": lam[i], "lambda_c": lam_c}, cfg)
            ro, to = decide(ours, sid, lam[i])
            same += int(rp == ro)
            if rp and ro:
                same_t += int(tp == to)
            tot += 1
    # dense prices as well
    same_d = tot_d = 0
    for lam_l in dense:
        for sid in sids:
            rp, _ = b9plus_decide(entries[sid], {"lambda_l": float(lam_l), "lambda_c": lam_c}, cfg)
            ro, _ = decide(ours, sid, float(lam_l))
            same_d += int(rp == ro)
            tot_d += 1
    out["b9plus"] = {
        "grid_decisions": tot, "grid_retrieve_agreement": same,
        "grid_timing_agreement_given_retrieve": same_t,
        "grid_both_retrieve": n_ours_ret_total,
        "dense_decisions": tot_d, "dense_retrieve_agreement": same_d,
        "definition": "retrieve iff voi*(1-rho(t_hat)) - lambda_l*(L - o(t_hat)) - lambda_c*(1+w(t_hat)) > 0, "
                      "t_hat = B9's own timing sub-optimizer choice (independent of the retrieve decision)",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, b_arr, need_arr, eff_item, gp, gm)
    print_summary(out)
    return 0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(out, b_arr, need_arr, eff_item, gp, gm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6})
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.6))
    ax = axs.ravel()

    # (a) band tiling over the belief distribution
    a = ax[0]
    rows = out["band_per_price"]
    for r in rows:
        y = math.log10(r["lambda_l"])
        a.plot([r["ours_cut"], r["b9_cut"]], [y, y], color=INK2, lw=2, solid_capstyle="round")
    yb = math.log10(rows[0]["lambda_l"]) - 0.12
    a.scatter(b_arr[need_arr == 1], np.full((need_arr == 1).sum(), yb), s=14, color=BLUE,
              label="needed", zorder=3, edgecolor="white", linewidth=0.5)
    a.scatter(b_arr[need_arr == 0], np.full((need_arr == 0).sum(), yb - 0.07), s=14, color=ORANGE,
              label="not needed", zorder=3, edgecolor="white", linewidth=0.5)
    a.set_xlabel("belief b (calibrated P(need))")
    a.set_ylabel(r"log$_{10}\,\lambda_l$")
    a.set_title("(a) divergence band $(c_O, c_{B9}]$ per grid price", fontsize=8, color=INK)
    a.legend(frameon=False, loc="upper left", fontsize=7)
    a.grid(axis="x", color=GRID, lw=0.5)

    # (b) predicted vs observed accuracy gap Ours - B9 per price
    a = ax[1]
    p2 = out["prop2"]
    x = np.arange(len(p2))
    obs = np.array([r["obs_acc_gap"] for r in p2])
    lo = np.array([r["obs_acc_gap_ci95"][0] for r in p2])
    hi = np.array([r["obs_acc_gap_ci95"][1] for r in p2])
    a.axhline(0, color=INK2, lw=0.6)
    a.fill_between(x, lo, hi, color=INK2, alpha=0.12, lw=0, label="observed 95% CI")
    a.plot(x, obs, color=INK, lw=1.5, marker="o", ms=3, label="observed")
    a.plot(x, [r["pred_acc_gap_v1_model"] for r in p2], color=BLUE, lw=1.5, ls="--",
           label="predicted, v1 objective")
    a.plot(x, [r["pred_acc_gap_corrected_model"] for r in p2], color=ORANGE, lw=1.5,
           label=r"predicted, corrected ($g^+,g^-$)")
    a.plot(x, [r["pred_acc_gap_oracle_labels"] for r in p2], color=AQUA, lw=1.5, ls=":",
           label="predicted, true labels")
    a.set_xlabel("grid price index i")
    a.set_ylabel("accuracy gap, Ours - B9 (same price)")
    a.set_title("(b) Prop. 2: predicted vs observed gap", fontsize=8, color=INK)
    a.legend(frameon=False, fontsize=6.5, loc="upper left")
    a.grid(axis="y", color=GRID, lw=0.5)

    # (c) retrieval effect vs belief
    a = ax[2]
    a.axhline(0, color=INK2, lw=0.6)
    a.scatter(b_arr[need_arr == 1], eff_item[need_arr == 1], s=14, color=BLUE, label="needed",
              edgecolor="white", linewidth=0.5, zorder=3)
    a.scatter(b_arr[need_arr == 0], eff_item[need_arr == 0], s=14, color=ORANGE, label="not needed",
              edgecolor="white", linewidth=0.5, zorder=3)
    bb = np.linspace(0, 1, 50)
    a.plot(bb, bb * gp - (1 - bb) * gm, color=INK, lw=1.5, label=r"$b g^+ - (1-b) g^-$")
    bstar = gm / (gp + gm)
    a.axvline(bstar, color=INK2, lw=0.8, ls="--")
    a.text(bstar + 0.01, -0.95, f"b*={bstar:.2f}", color=INK2, fontsize=7)
    a.set_xlabel("belief b")
    a.set_ylabel("acc(retrieve) - acc(skip), mean of 5 seeds")
    a.set_title("(c) retrieval effect by belief and need", fontsize=8, color=INK)
    a.legend(frameon=False, fontsize=6.5, loc="upper left")
    a.set_ylim(-1.05, 1.05)
    a.grid(axis="y", color=GRID, lw=0.5)

    # (d) realised utility gap: predicted (corrected quality + measured TTFA cost) vs observed
    a = ax[3]
    nz = [r for r in p2 if r["p_band"] > 0]
    px = np.array([r["pred_utility_gap_corrected_measured_cost"] for r in nz])
    py = np.array([r["obs_utility_gap_skipcanonical"] for r in nz])
    plo = np.array([r["obs_utility_gap_ci95"][0] for r in nz])
    phi = np.array([r["obs_utility_gap_ci95"][1] for r in nz])
    lim = [min(px.min(), plo.min()) - 0.05, max(px.max(), phi.max(), 0) + 0.05]
    a.plot(lim, lim, color=INK2, lw=0.8, ls="--", label="y = x")
    a.errorbar(px, py, yerr=[py - plo, phi - py], fmt="o", ms=4, color=BLUE, ecolor=BLUE,
               elinewidth=1, capsize=0, label="band prices (95% CI)")
    a.scatter([r["pred_utility_gap_model"] for r in nz], py, s=14, marker="x", color=ORANGE,
              label="v1 controller model (pred.)", zorder=3)
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("predicted realised utility gap")
    a.set_ylabel("observed realised utility gap, Ours - B9")
    a.set_title("(d) utility gap: corrected model + measured cost", fontsize=8, color=INK)
    a.legend(frameon=False, fontsize=6.5, loc="upper left")
    a.grid(color=GRID, lw=0.5)

    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)


def print_summary(out):
    p1 = out["prop1_on_grid"]
    print(f"Prop1 grid shift: retrieve agree {p1['retrieve_agreement']}/{p1['comparisons']}, "
          f"timing agree {p1['timing_agreement_given_both_retrieve']}/{p1['both_retrieve']}")
    print("operating points:", p1["operating_points_Ours_i_vs_B9_im1"])
    d = out["prop1_dense"]
    print(f"Prop1 dense: exact {d['exact_map_agreement']}/{d['decisions']}, "
          f"approx {d['approx_map_agreement']}/{d['decisions']} "
          f"({d['n_prices_where_approx_map_fails']} prices fail)")
    g = out["grid_recurrence"]
    print(f"grid ratios raw [{g['raw_ratio_min']:.4f},{g['raw_ratio_max']:.4f}], "
          f"fixed-point [{g['fixed_point_ratio_min']:.6f},{g['fixed_point_ratio_max']:.6f}], "
          f"L/(L-o*)={g['L_over_L_minus_o_star']:.6f}, lam_deg={g['fixed_point_lambda_deg']:.6f}")
    print("band summary:", out["band_summary"])
    print("sanity:", out["sanity"])
    print("retrieval effect:", {k: v for k, v in out["retrieval_effect"].items()})
    print("calibration:", out["calibration"])
    print("prop2 summary:", out["prop2_summary"])
    for r in out["prop2"]:
        if r["p_band"] > 0:
            print(f"  i={r['i']:2d} lam={r['lambda_l']:.4f} P={r['p_band']:.2f} "
                  f"predU={r['pred_utility_gap_model']:+.4f} bound={r['bound_lambda_o_star_P']:.4f} "
                  f"pred_acc v1={r['pred_acc_gap_v1_model']:+.3f} corr={r['pred_acc_gap_corrected_model']:+.3f} "
                  f"oracle={r['pred_acc_gap_oracle_labels']:+.3f} obs={r['obs_acc_gap']:+.3f} "
                  f"[{r['obs_acc_gap_ci95'][0]:+.3f},{r['obs_acc_gap_ci95'][1]:+.3f}] "
                  f"obsU={r['obs_utility_gap_skipcanonical']:+.4f} predUmeas={r['pred_utility_gap_corrected_measured_cost']:+.4f} "
                  f"predUitem={r['pred_utility_gap_item_effects_measured_cost']:+.4f}")
    print("measured latency:", out["measured_latency"])
    print("b9plus:", out["b9plus"])


if __name__ == "__main__":
    raise SystemExit(main())
