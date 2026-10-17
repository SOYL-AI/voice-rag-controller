"""
v2_test_final.py -- the CONFIRMATORY v2 test analysis (PREREG_V2 §3.2-§3.4,
§4.1-§4.2, §5), pooled over d1_open and d2_support. Run only after d2_support
grading is complete (§9 steps 8-9).

    PYTHONUTF8=1 python -m analysis.v2_test_final

Inputs (frozen code outputs):
  * simulator/e3v2_phase_a/<ds>/contrasts_test.json from
        python -m simulator.e3v2_lazy evaluate --dataset <ds> --split test
            --bundle <asr> --bundle <text150> --d-star-speculative <D*[SPEC]> --m3 0.016
    (D*[COMMIT] = B9+; D*[SPEC] = b9plus_v2 for d1_open, b10_speculative_v2 for d2_support)
  * the frozen bundles (price grid -> belief quantile)
  * docs/research_v2/v2_predictions.json (dev predictions, 48 cells)
  * <E1 results>/<ds>/analysis_test/analysis_test.json (llm.experiment1_v2.analyze --split test)

Statistics:
  * paired cluster bootstrap, B = 10,000: items (d1_open) / pair_id clusters
    (d2_support), resampled WITHIN dataset (stratified), the same resamples
    for every contrast and cell; pooled estimate = sum_d (n_test,d / N) mean_d
    with n_test = 3,734 : 498; percentile CIs.
  * P1 (H0 delta_C <= 0), P2 (H0 delta_C - delta_S <= 0), P3 TOST at
    m3 = 0.016, H-REV (H0 delta_C >= 0): one-sided bootstrap p-values,
    alpha = 0.025; IUT = P1 and P2 and both P3 sides; Holm over
    {P1, P2, P3 (larger side), H-REV} at 0.025.
  * Secondary family: per dataset at the primary cell, delta_C, delta_S and
    the interaction, two-sided p = 2 min(P*(<=0), P*(>=0)), Holm at 0.05.
  * Predicted vs observed (48 cells, delta_C): Spearman r_s >= 0.5 with
    one-sided 95% bootstrap lower bound > 0; inverse-variance-weighted slope
    (weights 1 / bootstrap var of each test cell) with a 95% bootstrap CI
    containing 1 and excluding 0. agree / directional / disagree.
  * H-E1v2 gate per dataset (success AUC >= 0.70; kill iff AUC < 0.60 on both).
A p-value of 0 means no resample crossed the null (p < 1/B).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1_v2 import common as e1common  # noqa: E402

PA = _REPO_ROOT / "simulator" / "e3v2_phase_a"
E1 = Path(r"C:\Users\ryang\OneDrive\Desktop\voice-rag\voice-rag-controller\llm\experiment1_v2\results")
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_final.json"
OUT_MD = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_final.md"
B, SEED, M3 = 10_000, 20261017, 0.016
N_TEST = {"d1_open": 3734, "d2_support": 498}
ASR = "piper-en_US-hfc_male-medium__150wpm__clean"
PRIMARY = {"condition": ASR, "profile": "rewrite_faiss", "quantile": 0.65,
           "lambda_l": {"d1_open": 0.17305735434823866, "d2_support": 0.2157823714003309}, "lambda_c": 0.05}
D_STAR_SPEC = {"d1_open": "b9plus_v2", "d2_support": "b10_speculative_v2"}
KINDS = ("delta_commit", "delta_speculative", "interaction")


# ---------------------------------------------------------------------------
# bootstrap machinery
# ---------------------------------------------------------------------------
def cluster_counts(universe: List[str], cluster_of: Dict[str, str], seed: int, n_boot: int = B) -> np.ndarray:
    """(n_boot, n_items) multiplicity of each item under cluster resampling."""
    clusters = sorted({cluster_of.get(i, i) for i in universe})
    cidx = {c: k for k, c in enumerate(clusters)}
    item_c = np.array([cidx[cluster_of.get(i, i)] for i in universe])
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, len(universe)))
    for b in range(n_boot):
        cc = np.bincount(rng.integers(0, len(clusters), size=len(clusters)), minlength=len(clusters))
        counts[b] = cc[item_c]
    return counts


class Dataset:
    def __init__(self, name: str, contrasts: dict, bundles: dict, cluster_of: dict, seed: int, n_boot: int = B):
        self.name = name
        self.rows = {}
        for cond, rows in contrasts.items():
            grid = bundles[cond]["price_grid"]
            for r in rows:
                q = next((g["quantile"] for g in grid[r["latency_profile"]]
                          if f"{g['lambda_l']:g}|{g['lambda_c']:g}" == r["price"]), None)
                self.rows[(cond, r["latency_profile"], q)] = r
        self.universe = sorted({i for r in self.rows.values() for k in KINDS for i in r["per_item"][k]})
        self.uidx = {i: k for k, i in enumerate(self.universe)}
        self.counts = cluster_counts(self.universe, cluster_of, seed, n_boot)
        self.n_clusters = len({cluster_of.get(i, i) for i in self.universe})

    def boot(self, cell, kind):
        vals = self.rows[cell]["per_item"][kind]
        x, m = np.zeros(len(self.universe)), np.zeros(len(self.universe))
        for i, v in vals.items():
            x[self.uidx[i]], m[self.uidx[i]] = float(v), 1.0
        tot = self.counts @ m
        return float(np.mean(list(vals.values()))), (self.counts @ x) / np.where(tot > 0, tot, np.nan), len(vals)


def summary(est, means) -> dict:
    q = np.nanpercentile(means, [2.5, 97.5])
    return {"estimate": float(est), "ci95": [float(q[0]), float(q[1])], "boot_se": float(np.nanstd(means, ddof=1))}


def one_sided_tests(c, s, i) -> dict:
    """c, s, i: (estimate, boot means) for delta_C, delta_S, interaction."""
    (ce, cb), (se_, sb), (ie, ib) = c, s, i
    cq, sq, iq = (np.nanpercentile(x, [2.5, 97.5]) for x in (cb, sb, ib))
    return {
        "P1": {"H0": "delta_C <= 0", "p": float(np.nanmean(cb <= 0)), "reject": bool(cq[0] > 0)},
        "P2": {"H0": "delta_C - delta_S <= 0", "p": float(np.nanmean(ib <= 0)), "reject": bool(iq[0] > 0)},
        "P3": {"m3": M3, "p_upper": float(np.nanmean(sb >= M3)), "p_lower": float(np.nanmean(sb <= -M3)),
               "reject_upper": bool(sq[1] < M3), "reject_lower": bool(sq[0] > -M3),
               "p": float(max(np.nanmean(sb >= M3), np.nanmean(sb <= -M3))),
               "pass": bool(-M3 < sq[0] and sq[1] < M3)},
        "H-REV": {"H0": "delta_C >= 0", "p": float(np.nanmean(cb >= 0)), "reject": bool(cq[1] < 0)},
    }


def holm(pvals: Dict[str, float], alpha: float) -> dict:
    order = sorted(pvals, key=lambda k: pvals[k])
    out, stop = {}, False
    for r, k in enumerate(order):
        thr = alpha / (len(order) - r)
        rej = (not stop) and pvals[k] <= thr
        stop = stop or not rej
        out[k] = {"p": pvals[k], "threshold": thr, "reject": rej}
    return out


def two_sided_p(means) -> float:
    return float(min(1.0, 2 * min(np.nanmean(means <= 0), np.nanmean(means >= 0))))


def spearman(x, y) -> float:
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def wls_slope(x, y, w) -> float:
    xw, yw = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xw) * (y - yw)) / np.sum(w * (x - xw) ** 2))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def analyse(ds_objs: Dict[str, Dataset], predictions: dict) -> dict:
    N = sum(N_TEST[d] for d in ds_objs)
    w = {d: N_TEST[d] / N for d in ds_objs}
    out = {"weights": w, "per_dataset": {}, "pooled": {}, "cells": [], "pooled_cells": []}
    prim = {d: (PRIMARY["condition"], PRIMARY["profile"], PRIMARY["quantile"]) for d in ds_objs}
    boots = {}
    for d, o in ds_objs.items():
        assert o.rows[prim[d]]["price"] == f"{PRIMARY['lambda_l'][d]:g}|{PRIMARY['lambda_c']:g}", d
        assert o.rows[prim[d]]["d_star"]["speculative"] == D_STAR_SPEC[d], d
        res = {}
        for k in KINDS:
            est, means, n = o.boot(prim[d], k)
            boots[(d, k)] = (est, means)
            res[k] = {**summary(est, means), "n_items": n, "p_two_sided": two_sided_p(means)}
        res["one_sided_tests_per_dataset_descriptive"] = one_sided_tests(*(boots[(d, k)] for k in KINDS))
        out["per_dataset"][d] = {"n_items": len(o.universe), "n_clusters": o.n_clusters, "primary": res}
    # pooled primary
    pooled = {}
    for k in KINDS:
        est = sum(w[d] * boots[(d, k)][0] for d in ds_objs)
        means = sum(w[d] * boots[(d, k)][1] for d in ds_objs)
        pooled[k] = (est, means)
        out["pooled"][k] = summary(est, means)
    tests = one_sided_tests(*(pooled[k] for k in KINDS))
    out["pooled"]["confirmatory_tests"] = tests
    out["pooled"]["IUT_boundary_claim"] = bool(tests["P1"]["reject"] and tests["P2"]["reject"] and tests["P3"]["pass"])
    out["pooled"]["holm_individual_claims"] = holm({"P1": tests["P1"]["p"], "P2": tests["P2"]["p"],
                                                    "P3": tests["P3"]["p"], "H-REV": tests["H-REV"]["p"]}, 0.025)
    # secondary family: per dataset delta_C, delta_S, interaction, two-sided, Holm 0.05
    sec = {f"{d}|{k}": out["per_dataset"][d]["primary"][k]["p_two_sided"] for d in ds_objs for k in KINDS}
    out["secondary_family"] = holm(sec, 0.05)
    # descriptive: every cell per dataset, and pooled by (condition, profile, quantile)
    cell_means = {}
    for d, o in ds_objs.items():
        for cell in sorted(o.rows, key=lambda c: (c[0], c[1], c[2] or 0)):
            row = {"dataset": d, "condition": cell[0], "profile": cell[1], "belief_quantile": cell[2],
                   "price": o.rows[cell]["price"], "d_star": o.rows[cell]["d_star"]}
            for k in KINDS:
                est, means, n = o.boot(cell, k)
                cell_means[(d,) + cell + (k,)] = (est, means)
                row[k] = {**summary(est, means), "n_items": n}
            out["cells"].append(row)
    shared = set.intersection(*(set(o.rows) for o in ds_objs.values()))
    for cell in sorted(shared, key=lambda c: (c[0], c[1], c[2] or 0)):
        row = {"condition": cell[0], "profile": cell[1], "belief_quantile": cell[2]}
        for k in KINDS:
            est = sum(w[d] * cell_means[(d,) + cell + (k,)][0] for d in ds_objs)
            means = sum(w[d] * cell_means[(d,) + cell + (k,)][1] for d in ds_objs)
            row[k] = summary(est, means)
        out["pooled_cells"].append(row)
    # predicted vs observed (48 cells, delta_C)
    pv = []
    for c in predictions.get("cells", []):
        key = (c.get("dataset"), c.get("condition"), c.get("profile"), c.get("belief_quantile"), "delta_commit")
        if c.get("delta_commit") is None or key not in cell_means:
            continue
        pv.append((c["delta_commit"], cell_means[key]))
    if len(pv) >= 3:
        x = np.array([p[0] for p in pv])
        y = np.array([p[1][0] for p in pv])
        yb = np.vstack([p[1][1] for p in pv])            # (cells, B)
        var = np.nanvar(yb, axis=1, ddof=1)
        wts = 1.0 / np.where(var > 0, var, np.nan)
        ok = ~np.isnan(wts)
        rs = spearman(x, y)
        rs_b = np.array([spearman(x, yb[:, b]) for b in range(yb.shape[1])])
        sl = wls_slope(x[ok], y[ok], wts[ok])
        sl_b = np.array([wls_slope(x[ok], yb[ok, b], wts[ok]) for b in range(yb.shape[1])])
        lo_rs = float(np.nanpercentile(rs_b, 5))
        sl_ci = [float(v) for v in np.nanpercentile(sl_b, [2.5, 97.5])]
        a = bool(rs >= 0.5 and lo_rs > 0)
        b_ok = bool(sl_ci[0] <= 1 <= sl_ci[1] and not (sl_ci[0] <= 0 <= sl_ci[1]))
        out["predicted_vs_observed"] = {
            "n_cells": len(pv), "spearman": rs, "spearman_one_sided_95_lower": lo_rs, "criterion_a": a,
            "ivw_slope": sl, "ivw_slope_ci95": sl_ci, "criterion_b": b_ok,
            "verdict": "agree" if (a and b_ok) else ("directional" if a else "disagree")}
    return out


def e1_gate(datasets) -> dict:
    out = {}
    for d in datasets:
        p = E1 / d / "analysis_test" / "analysis_test.json"
        if not p.exists():
            out[d] = {"missing": str(p)}
            continue
        m = e1common.load_json(p)["models"]["mistral7b_v03"]
        full = ((m.get("auc_full") or {}).get("self_assessed_need") or {}).get("empirical")
        curve = {str(r["word_frac"]): r["empirical"]["auc"]
                 for r in (m.get("prefix_curve_text") or {}).get("self_assessed_need", []) if r.get("empirical")}
        out[d] = {"auc_full": full, "success_ge_0.70": None if not full else bool(full["auc"] >= 0.70),
                  "below_0.60": None if not full else bool(full["auc"] < 0.60), "prefix_curve": curve,
                  "c2_available": None if "0.5" not in curve else bool(curve["0.5"] >= 0.65)}
    vals = [v for v in out.values() if v.get("below_0.60") is not None]
    out["kill_auc_below_0.60_on_both"] = None if len(vals) < 2 else all(v["below_0.60"] for v in vals)
    return out


def load_dataset(d: str, seed: int, n_boot: int = B) -> Dataset:
    doc = e1common.load_json(PA / d / "contrasts_test.json")
    bundles = {}
    for p in doc["bundles"]:
        b = e1common.load_json(Path(p))
        bundles[b["condition"]] = b
    items = e1common.load_items(e1common.resolve_dataset_paths(d)["items"])
    cluster_of = {it["id"]: (it.get("pair_id") or it["id"]) for it in items}
    return Dataset(d, doc["contrasts"], bundles, cluster_of, seed, n_boot)


def main() -> int:
    dsets = {"d1_open": load_dataset("d1_open", SEED), "d2_support": load_dataset("d2_support", SEED + 1)}
    preds = e1common.load_json(_REPO_ROOT / "docs" / "research_v2" / "v2_predictions.json")
    res = analyse(dsets, preds)
    doc = {"_comment": "CONFIRMATORY v2 test results (PREREG_V2 §3.2-§3.4), pooled over d1_open and d2_support.",
           "generated_by": "python -m analysis.v2_test_final", "generated_at_utc": e1common.now_utc(),
           "git_commit": e1common.git_commit(), "split": "test",
           "frozen": {"prereg_tag": "prereg-v2-frozen", "frozen_code": "ddde67f", "deviations": ["D1", "D2"],
                      "notes": ["N1", "N2 (exploratory; not used here)", "N3"], "d_star_speculative": D_STAR_SPEC,
                      "d_star_commit": "b9plus_v2", "m3": M3, "primary_cell": PRIMARY},
           "inputs": {d: {"contrasts_sha256": e1common.sha256_file(PA / d / "contrasts_test.json"),
                          "required_cells_sha256": e1common.sha256_file(PA / d / "required_test.json")}
                      for d in dsets},
           "bootstrap": {"B": B, "seeds": {"d1_open": SEED, "d2_support": SEED + 1},
                         "unit": {"d1_open": "item", "d2_support": "pair_id cluster"},
                         "stratified_by_dataset": True, "same_resamples_for_every_contrast": True},
           **res, "e1_gate": e1_gate(dsets)}
    e1common.atomic_write_json(OUT_JSON, doc)
    write_md(doc)
    print(json.dumps({"pooled": {k: doc["pooled"][k] for k in KINDS}, "tests": doc["pooled"]["confirmatory_tests"],
                      "IUT": doc["pooled"]["IUT_boundary_claim"]}, indent=1))
    return 0


def f4(x):
    return f"{x:+.4f}"


def pf(p):
    return f"< {1 / B:.0e}" if p == 0 else f"{p:.4f}"


def write_md(doc):
    P = doc["pooled"]
    t = P["confirmatory_tests"]
    L = ["# v2 test results (confirmatory, pooled over d1_open and d2_support)", "",
         "Frozen code `ddde67f` + D1 (PREREG_V2 Deviations; D2 gate supplement; notes N1, N3). Primary cell: "
         "rewrite_faiss, 0.65 belief-quantile price, primary ASR condition. D\\*[COMMIT] = B9+; D\\*[SPEC] = B9+ "
         "(d1_open), SPEC (d2_support); m₃ = 0.016. Paired cluster bootstrap, B = 10,000, stratified by dataset, "
         "pooled weights 3,734 : 498. Source: `docs/research_v2/v2_test_results_final.json`.", "",
         "## Headline (primary cell)", "", "| | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |",
         "|---|---|---|---|"]
    for d, r in doc["per_dataset"].items():
        p = r["primary"]
        L.append(f"| {d} | " + " | ".join(f"{f4(p[k]['estimate'])} [{f4(p[k]['ci95'][0])}, {f4(p[k]['ci95'][1])}]"
                                         for k in KINDS) + " |")
    L.append("| **pooled** | " + " | ".join(f"**{f4(P[k]['estimate'])}** [{f4(P[k]['ci95'][0])}, {f4(P[k]['ci95'][1])}]"
                                           for k in KINDS) + " |")
    h = P["holm_individual_claims"]
    L += ["", "## Confirmatory tests (pooled, one-sided α = 0.025)", "",
          "| test | p | rejects | Holm (α = 0.025) |", "|---|---|---|---|",
          f"| P1: Δ[C] > 0 | {pf(t['P1']['p'])} | {t['P1']['reject']} | {h['P1']['reject']} |",
          f"| P2: Δ[C] − Δ[S] > 0 | {pf(t['P2']['p'])} | {t['P2']['reject']} | {h['P2']['reject']} |",
          f"| P3: TOST ±0.016 | upper {pf(t['P3']['p_upper'])} / lower {pf(t['P3']['p_lower'])} | {t['P3']['pass']} | "
          f"{h['P3']['reject']} |",
          f"| H-REV: Δ[C] < 0 | {pf(t['H-REV']['p'])} | {t['H-REV']['reject']} | {h['H-REV']['reject']} |", "",
          f"**Boundary claim (IUT P1 ∧ P2 ∧ P3): {P['IUT_boundary_claim']}.**", "",
          "## Secondary family (per dataset, two-sided, Holm α = 0.05)", "", "| test | p | rejects |", "|---|---|---|"]
    for k, v in doc["secondary_family"].items():
        L.append(f"| {k} | {pf(v['p'])} | {v['reject']} |")
    pvo = doc.get("predicted_vs_observed")
    if pvo:
        L += ["", f"## Predicted vs observed (Δ[COMMIT], {pvo['n_cells']} cells)", "",
              f"Spearman {pvo['spearman']:.3f} (one-sided 95% lower {pvo['spearman_one_sided_95_lower']:.3f}); "
              f"IVW slope {pvo['ivw_slope']:.3f} [{pvo['ivw_slope_ci95'][0]:.3f}, {pvo['ivw_slope_ci95'][1]:.3f}]. "
              f"Verdict: **{pvo['verdict']}**."]
    g = doc["e1_gate"]
    L += ["", "## H-E1v2 gate (test)", ""]
    for d in ("d1_open", "d2_support"):
        v = g.get(d, {})
        if v.get("auc_full"):
            L.append(f"* {d}: AUC {v['auc_full']['auc']:.3f} [{v['auc_full']['ci95'][0]:.3f}, "
                     f"{v['auc_full']['ci95'][1]:.3f}]; success {v['success_ge_0.70']}; C2 available {v['c2_available']}.")
    L.append(f"* Kill (AUC < 0.60 on both): {g.get('kill_auc_below_0.60_on_both')}.")
    L += ["", "## All cells (descriptive)", "", "| dataset | condition | profile | q | Δ[COMMIT] | Δ[SPEC] | interaction |",
          "|---|---|---|---|---|---|---|"]
    for c in doc["cells"]:
        cond = "ASR" if c["condition"] == ASR else c["condition"]
        L.append(f"| {c['dataset']} | {cond} | {c['profile']} | {c['belief_quantile']} | " + " | ".join(
            f"{f4(c[k]['estimate'])} [{f4(c[k]['ci95'][0])}, {f4(c[k]['ci95'][1])}]" for k in KINDS) + " |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
