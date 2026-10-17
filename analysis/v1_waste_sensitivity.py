"""
v1_waste_sensitivity.py -- sensitivity of the v1 E3 results to the waste
rate w (EXPLORATORY; not part of any pre-registered test).

    PYTHONUTF8=1 python -m analysis.v1_waste_sensitivity

In v1, w is controller.e3.waste_rate, a per-timing PLACEHOLDER (never
measured): sentence_end 0, immediate 0.02, fixed_early_proxy 0.08
(controller/baselines/_objective.py waste_rate()). It enters the retrieval
compute charge lambda_c * (1 + w(t)) of Ours and B9+, and B9's timing
sub-optimizer (lambda_c * w(t)); B9's retrieval sub-optimizer uses
w(sentence_end) = 0.

Scenarios: the logged placeholder (reproduction check), and w in
{0, 0.04, 0.08, 0.16} applied to BOTH early timings (immediate and
fixed_early_proxy), with w(sentence_end) = 0 (a waste only exists for an
early launch). Nothing else changes: the same 26-point price grid, the same
frozen beliefs, and the controllers' own code (read-only; only the config
dict they read is swapped).

No generation is needed: Phase A measured EVERY (scenario, seed,
skip|retrieve, timing) cell (simulator/e3_sweep_cache), so any decision any
w produces is a lookup. The script checks that, at the placeholder, the
reconstructed records equal the logged sweep records.

Per scenario it reports
  * decision agreement: Ours vs B9 and Ours vs B9+ at the same price
    (retrieve; timing given both retrieve), Ours(lambda_i) vs B9(lambda_{i-1})
    (THEORY Prop. 1 on the grid), and decisions changed vs the logged run;
  * the frontier: pooled (5 seeds x 50 scenarios) accuracy / p95 TTFA /
    retrievals per price for Ours, B9, B9+, the per-price accuracy gap
    Ours - B9, and the pre-registered dominance test of
    analysis/e3_frontier.py (Ours vs B9; Ours vs B9+), unchanged.

Writes docs/research_v2/waste_sensitivity_v1.json.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# v1_theory_check patches the controllers' config loaders at import; it is
# imported for its loaders only, and the patch is replaced below.
from analysis import v1_theory_check as tc  # noqa: E402
from analysis import e3_frontier as ef  # noqa: E402
from controller.baselines import _objective as obj  # noqa: E402
from controller.baselines.b9_decomposed import DecomposedWithPrices  # noqa: E402
from controller.baselines.ours import UnifiedController  # noqa: E402
import controller.baselines.b9_decomposed as _b9_mod  # noqa: E402
import controller.baselines.ours as _ours_mod  # noqa: E402

OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "waste_sensitivity_v1.json"
SEEDS = tc.SEEDS
W_GRID = (0.0, 0.04, 0.08, 0.16)

_BASE_CFG = copy.deepcopy(tc._CFG)
_CUR = {"cfg": _BASE_CFG}
for _m in (obj, _ours_mod, _b9_mod):
    _m.load_e3_config = lambda path=None: _CUR["cfg"]


def scenarios() -> list:
    out = [{"name": "v1_placeholder (logged)", "w": None,
            "waste_rate": dict(_BASE_CFG["waste_rate"])}]
    for w in W_GRID:
        out.append({"name": f"w={w:g}", "w": w,
                    "waste_rate": {"sentence_end": 0.0, "immediate": w, "fixed_early_proxy": w}})
    return out


def run_scenario(sc, lam, lam_c, entries, sids, results, Y, T, tmpdir: Path) -> dict:
    cfg = copy.deepcopy(_BASE_CFG)
    cfg["waste_rate"] = dict(sc["waste_rate"])
    _CUR["cfg"] = cfg
    ours, b9 = UnifiedController(), DecomposedWithPrices()
    K = len(lam)

    dec = {"ours": {}, "b9": {}, "b9plus": {}}
    for i in range(K):
        prices = {"lambda_l": float(lam[i]), "lambda_c": lam_c}
        for sid in sids:
            d = ours.decide(tc.ctx_for(entries[sid], prices))
            dec["ours"][(i, sid)] = (bool(d.retrieve), d.timing)
            d = b9.decide(tc.ctx_for(entries[sid], prices))
            dec["b9"][(i, sid)] = (bool(d.retrieve), d.timing)
            dec["b9plus"][(i, sid)] = tuple(tc.b9plus_decide(entries[sid], prices, cfg))

    # decision agreement
    def agree(a, b, pairs):
        r = t = both = 0
        for ka, kb in pairs:
            ra, ta = dec[a][ka]
            rb, tb = dec[b][kb]
            r += int(ra == rb)
            if ra and rb:
                both += 1
                t += int(ta == tb)
        return {"retrieve_agreement": r, "of": len(pairs), "both_retrieve": both,
                "timing_agreement_given_both_retrieve": t}

    same = [((i, s), (i, s)) for i in range(K) for s in sids]
    shift = [((i, s), (i - 1, s)) for i in range(1, K) for s in sids]
    agreement = {
        "ours_vs_b9_same_price": agree("ours", "b9", same),
        "ours_vs_b9plus_same_price": agree("ours", "b9plus", same),
        "ours_i_vs_b9_im1 (Prop. 1 on the grid)": agree("ours", "b9", shift),
        "n_retrieve": {k: sum(v[0] for v in dec[k].values()) for k in dec},
        "timing_counts_given_retrieve": {
            k: {t: sum(1 for r, tt in dec[k].values() if r and tt == t) for t in obj.TIMING_ORDER}
            for k in dec},
    }
    changed = {}
    for name in ("ours", "b9"):
        n = n_ret = 0
        for i in range(K):
            for sid in sids:
                lr = results[(name, SEEDS[0], i, sid)]["decision"]
                n += int((bool(lr["retrieve"]), lr["timing"]) != dec[name][(i, sid)])
                n_ret += int(bool(lr["retrieve"]) != dec[name][(i, sid)][0])
        changed[name] = {"decisions_changed_vs_logged": n, "retrieve_flips_vs_logged": n_ret,
                         "timing_label_only_changes": n - n_ret, "of": K * len(sids)}
    agreement["vs_logged_run"] = changed

    # outcome records from the measured cells (no generation)
    records = []
    mismatch = {"ours": 0, "b9": 0}
    for name in ("ours", "b9", "b9plus"):
        for i in range(K):
            for sid in sids:
                r, t = dec[name][(i, sid)]
                path = "retrieve" if r else "skip"
                for seed in SEEDS:
                    rec = {"baseline": name, "seed": seed, "scenario_id": sid,
                           "prices": {"lambda_l": float(lam[i]), "lambda_c": lam_c},
                           "outcome": {"answer_correct": bool(Y[(sid, seed, path)]),
                                       "ttfa_ms": float(T[(sid, seed, path, t)]),
                                       "retrievals": 1 if r else 0}}
                    records.append(rec)
                    if sc["w"] is None and name in mismatch:
                        lo = results[(name, seed, i, sid)]["outcome"]
                        if (bool(lo["answer_correct"]) != rec["outcome"]["answer_correct"]
                                or abs(float(lo["ttfa_ms"]) - rec["outcome"]["ttfa_ms"]) > 1e-9
                                or int(lo["retrievals"]) != rec["outcome"]["retrievals"]):
                            mismatch[name] += 1

    # pooled operating points (5 seeds x 50 scenarios per price), as paper_figures_v1.op_point
    by = {}
    for rec in records:
        by.setdefault((rec["baseline"], rec["prices"]["lambda_l"]), []).append(rec)
    pooled = {}
    for name in ("ours", "b9", "b9plus"):
        rows = []
        for i in range(K):
            rs = by[(name, float(lam[i]))]
            rows.append({"i": i, "lambda_l": float(lam[i]),
                         "acc": float(np.mean([x["outcome"]["answer_correct"] for x in rs])),
                         "p95_ms": float(np.percentile([x["outcome"]["ttfa_ms"] for x in rs], 95)),
                         "compute": float(np.mean([x["outcome"]["retrievals"] for x in rs]))})
        pooled[name] = rows
    gap = []
    for i in range(K):
        o, b = pooled["ours"][i], pooled["b9"][i]
        gap.append({"i": i, "acc_ours_minus_b9": o["acc"] - b["acc"],
                    "p95_ours_minus_b9_ms": o["p95_ms"] - b["p95_ms"],
                    "compute_ours_minus_b9": o["compute"] - b["compute"]})

    # the pre-registered dominance test, unchanged (e3_frontier), into a temp dir
    agg = ef.aggregate_frontier(records)
    dom = {}
    for rival in ("b9", "b9plus"):
        out = ef.dominance_test(agg, output_dir=str(tmpdir), ours="ours", rival=rival)
        dom[f"ours_vs_{rival}"] = {
            "seeds_with_strict_dominance": out["seeds_with_strict_dominance"],
            "overall_criterion_met": out["overall_criterion_met"],
            "dominated_points_per_seed": {str(p["seed"]): p["dominated_operating_points"] for p in out["per_seed"]},
        }

    accs = np.array([g["acc_ours_minus_b9"] for g in gap])
    frontier = {
        "pooled_operating_points": pooled,
        "per_price_gap_ours_minus_b9": gap,
        "summary": {
            "n_prices": K,
            "prices_with_nonzero_acc_gap": int(np.sum(np.abs(accs) > 1e-12)),
            "mean_acc_gap_over_prices": float(accs.mean()),
            "min_acc_gap": float(accs.min()), "max_acc_gap": float(accs.max()),
            "prices_with_identical_operating_point": int(sum(
                abs(g["acc_ours_minus_b9"]) < 1e-12 and abs(g["p95_ours_minus_b9_ms"]) < 1e-9
                and abs(g["compute_ours_minus_b9"]) < 1e-12 for g in gap)),
            "ours_acc_range": [min(r["acc"] for r in pooled["ours"]), max(r["acc"] for r in pooled["ours"])],
            "b9_acc_range": [min(r["acc"] for r in pooled["b9"]), max(r["acc"] for r in pooled["b9"])],
        },
        "dominance": dom,
    }
    res = {"scenario": sc["name"], "w": sc["w"], "waste_rate": sc["waste_rate"],
           "agreement": agreement, "frontier": frontier}
    if sc["w"] is None:
        res["reproduction_check"] = {
            "records_not_matching_logged_sweep": mismatch,
            "n_records_checked_per_controller": K * len(sids) * len(SEEDS)}
    return res


def main() -> int:
    lam, lam_c, entries, need, sids, results, Y, T = tc.load_all()
    out = {
        "_comment": "EXPLORATORY sensitivity of the v1 E3 results to the waste-rate placeholder w. "
                    "Generated by python -m analysis.v1_waste_sensitivity from committed v1 artifacts only "
                    "(configs/e3_price_grid.json, simulator/e3_scenario_signals.json, simulator/scenarios.json, "
                    "simulator/e3_sweep_results/, simulator/e3_sweep_cache/). w(sentence_end)=0; w applied to both "
                    "early timings (immediate, fixed_early_proxy). Price grid, beliefs and controller code unchanged.",
        "grid_lambda_l": lam, "lambda_c": lam_c, "n_scenarios": len(sids), "seeds": list(SEEDS),
        "scenarios": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for sc in scenarios():
            out["scenarios"].append(run_scenario(sc, lam, lam_c, entries, sids, results, Y, T, Path(tmp)))
    _CUR["cfg"] = _BASE_CFG
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for s in out["scenarios"]:
        a, f = s["agreement"], s["frontier"]
        print(s["scenario"], "| Ours=B9", a["ours_vs_b9_same_price"]["retrieve_agreement"], "/",
              a["ours_vs_b9_same_price"]["of"], "| Ours=B9+", a["ours_vs_b9plus_same_price"]["retrieve_agreement"],
              "| Prop1", a["ours_i_vs_b9_im1 (Prop. 1 on the grid)"]["retrieve_agreement"], "/",
              a["ours_i_vs_b9_im1 (Prop. 1 on the grid)"]["of"], "| changed", a["vs_logged_run"],
              "| dom", {k: v["seeds_with_strict_dominance"] for k, v in f["dominance"].items()},
              "| acc gap mean", round(f["summary"]["mean_acc_gap_over_prices"], 4),
              s.get("reproduction_check", ""))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
