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
        t = one_sided_tests(*(boots[(d, k)] for k in KINDS))
        res["one_sided_tests"] = t
        res["IUT_boundary_claim"] = bool(t["P1"]["reject"] and t["P2"]["reject"] and t["P3"]["pass"])
        res["holm_individual_claims"] = holm({"P1": t["P1"]["p"], "P2": t["P2"]["p"], "P3": t["P3"]["p"],
                                              "H-REV": t["H-REV"]["p"]}, 0.025)
        res["secondary_family_this_dataset"] = holm({k: res[k]["p_two_sided"] for k in KINDS}, 0.05)
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
    # predicted vs observed (48 cells, delta_C), and per dataset (24 cells each)
    pvo = pred_vs_obs(predictions, cell_means, set(ds_objs))
    if pvo:
        out["predicted_vs_observed"] = pvo
    out["predicted_vs_observed_by_dataset"] = {d: pred_vs_obs(predictions, cell_means, {d}) for d in ds_objs}
    return out


def pred_vs_obs(predictions: dict, cell_means: dict, datasets: set):
    pv = []
    for c in predictions.get("cells", []):
        key = (c.get("dataset"), c.get("condition"), c.get("profile"), c.get("belief_quantile"), "delta_commit")
        if c.get("dataset") not in datasets or c.get("delta_commit") is None or key not in cell_means:
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
        return {"datasets": sorted(datasets),
                "n_cells": len(pv), "spearman": rs, "spearman_one_sided_95_lower": lo_rs, "criterion_a": a,
                "ivw_slope": sl, "ivw_slope_ci95": sl_ci, "criterion_b": b_ok,
                "verdict": "agree" if (a and b_ok) else ("directional" if a else "disagree")}
    return None


CONFIRMATORY = "confirmatory"


def apply_status(doc: dict, d2_reason) -> dict:
    """Tag every verdict. d2_reason None: all confirmatory, the pooled test is
    primary (PREREG_V2 as frozen). Otherwise (proposed D3, option B): d1_open's
    alias-graded results stay CONFIRMATORY and become the primary results;
    everything that uses d2_support's judge-graded outcomes (d2_support
    per-dataset results, every pooled result, the 6-test secondary family and
    the 48-cell predicted-vs-observed check, the two-dataset E1 kill rule,
    the judge checks) is tagged EXPLORATORY with the reason."""
    expl = None if d2_reason is None else f"exploratory ({d2_reason})"
    st_d2 = CONFIRMATORY if expl is None else expl
    doc["d2_status_reason"] = d2_reason
    for d, r in doc["per_dataset"].items():
        r["status"] = CONFIRMATORY if d == "d1_open" else st_d2
    doc["pooled"]["status"] = st_d2
    doc["secondary_family_status"] = st_d2
    if doc.get("predicted_vs_observed"):
        doc["predicted_vs_observed"]["status"] = st_d2
    for d, v in (doc.get("predicted_vs_observed_by_dataset") or {}).items():
        if v:
            v["status"] = CONFIRMATORY if d == "d1_open" else st_d2
    g = doc.get("e1_gate") or {}
    for d in ("d1_open", "d2_support"):
        if isinstance(g.get(d), dict):
            g[d]["status"] = CONFIRMATORY if d == "d1_open" else st_d2
    g["kill_rule_status"] = st_d2
    if isinstance(doc.get("judge_checks"), dict):
        doc["judge_checks"]["status"] = st_d2
    for c in doc.get("cells", []):
        c["status"] = "descriptive" + ("" if expl is None or c["dataset"] == "d1_open" else f"; {expl}")
    for c in doc.get("pooled_cells", []):
        c["status"] = "descriptive" + ("" if expl is None else f"; {expl}")
    doc["primary_results"] = ("pooled confirmatory tests (PREREG_V2 §3.2-§3.3)" if expl is None else
                              "d1_open per-dataset tests (alias-graded; CONFIRMATORY); d2_support and pooled "
                              f"results are {expl}")
    return doc


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


def judge_checks() -> dict:
    """(1) The 111-type cross-key check (PI 2026-10-17): (item, answer) texts graded
    both by label.py at k = 1 (judges_test_k1.json key) and by Phase A's
    grade_primary_k1 (k = 3 configured key, sample 0); agreement of the two
    primary verdicts. (2) Secondary judge on the E1 label draws: label.py ran
    gpt-4o on ALL draws (a superset); the pre-registered result is the
    stratified 30% subsample (dataset x item_type x primary verdict, seed
    20261016, simulator/e3v2_phase_a.py:secondary_subsample); both reported.
    (3) Phase A secondary agreement on its 30% subsample."""
    from simulator.e3v2_phase_a import secondary_subsample
    out = {}
    lab_path = E1 / "d2_support" / "mistral7b_v03" / "labels_test.json"
    pa = PA / "d2_support"
    if not lab_path.exists() or not (pa / "grades_k1_test.jsonl").exists():
        return {"status": "pending d2_support grading"}
    items = {it["id"]: it for it in e1common.load_items(e1common.resolve_dataset_paths("d2_support")["items"])}
    draws = e1common.load_json(lab_path)["draws"]

    def verdict(d, role):
        j = (d.get("judges") or {}).get(role) or {}
        return j.get("answer_correct") if j.get("parsed_ok") else None

    lab = {}
    for d in draws:
        if d.get("condition", "closed_book") == "closed_book":
            v = verdict(d, "primary")
            if v is not None:
                lab[(d["item_id"], (d.get("answer") or "").strip())] = bool(v)
    text = {}
    for name in ("answers_test.jsonl", "skip_answers_test.jsonl"):
        for r in e1common.read_jsonl(pa / name):
            text[r["key"]] = (r["item_id"], (r.get("text") or "").strip())
    phase = {}
    for r in e1common.read_jsonl(pa / "grades_k1_test.jsonl"):
        if r.get("primary") is not None and r["key"] in text:
            phase[text[r["key"]]] = bool(r["primary"])
    both = sorted(set(lab) & set(phase))
    out["cross_key_primary_agreement"] = {
        "n_shared_item_answer_texts": len(both),
        "agreement": None if not both else float(np.mean([lab[k] == phase[k] for k in both])),
        "note": "label.py (judges_test_k1.json key) vs Phase A grade_primary_k1 (k = 3 configured key, sample 0)"}
    rows = [{"key": f"{d['item_id']}|{d['draw_seed']}", "item_id": d["item_id"], "primary": verdict(d, "primary"),
             "secondary": verdict(d, "secondary")} for d in draws if d.get("condition", "closed_book") == "closed_book"]
    sub = secondary_subsample([r for r in rows if r["primary"] is not None], items)

    def agree(rs):
        rs = [r for r in rs if r["primary"] is not None and r["secondary"] is not None]
        return {"n": len(rs), "agreement": None if not rs else float(np.mean([r["primary"] == r["secondary"]
                                                                               for r in rs]))}
    out["e1_labels_secondary"] = {"preregistered_stratified_30pct": agree([r for r in rows if r["key"] in sub]),
                                  "all_draws_superset": agree(rows)}
    sec_path = pa / "grades_secondary_test.jsonl"
    if sec_path.exists():
        prim = {r["key"]: r.get("primary") for r in e1common.read_jsonl(pa / "grades_k1_test.jsonl")}
        pr = [{"primary": prim.get(r["key"]), "secondary": r.get("secondary")} for r in e1common.read_jsonl(sec_path)]
        out["phase_a_secondary_30pct"] = agree(pr)
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


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="v2 test analysis (PREREG_V2 §3.2-§3.4)")
    ap.add_argument("--d2-exploratory-reason", default=None,
                    help="tag every d2_support-dependent and pooled verdict as 'exploratory (<reason>)' and keep "
                         "the d1_open (alias-graded) results confirmatory and primary; e.g. "
                         "'D3: kappa_human < 0.70'. Default: as frozen (all confirmatory, pooled primary).")
    args = ap.parse_args(argv)
    dsets = {"d1_open": load_dataset("d1_open", SEED), "d2_support": load_dataset("d2_support", SEED + 1)}
    preds = e1common.load_json(_REPO_ROOT / "docs" / "research_v2" / "v2_predictions.json")
    res = analyse(dsets, preds)
    doc = {"_comment": ("v2 test results (PREREG_V2 §3.2-§3.4), d1_open and d2_support, pooled. Every verdict "
                        "carries a 'status'."),
           "generated_by": "python -m analysis.v2_test_final" + (
               "" if args.d2_exploratory_reason is None else f" --d2-exploratory-reason '{args.d2_exploratory_reason}'"),
           "generated_at_utc": e1common.now_utc(), "git_commit": e1common.git_commit(), "split": "test",
           "frozen": {"prereg_tag": "prereg-v2-frozen", "frozen_code": "ddde67f", "deviations": ["D1", "D2"],
                      "notes": ["N1", "N2 (exploratory; not used here)", "N3"], "d_star_speculative": D_STAR_SPEC,
                      "d_star_commit": "b9plus_v2", "m3": M3, "primary_cell": PRIMARY},
           "inputs": {d: {"contrasts_sha256": e1common.sha256_file(PA / d / "contrasts_test.json"),
                          "required_cells_sha256": e1common.sha256_file(PA / d / "required_test.json")}
                      for d in dsets},
           "bootstrap": {"B": B, "seeds": {"d1_open": SEED, "d2_support": SEED + 1},
                         "unit": {"d1_open": "item", "d2_support": "pair_id cluster"},
                         "stratified_by_dataset": True, "same_resamples_for_every_contrast": True},
           **res, "e1_gate": e1_gate(dsets), "judge_checks": judge_checks()}
    apply_status(doc, args.d2_exploratory_reason)
    e1common.atomic_write_json(OUT_JSON, doc)
    write_md(doc)
    print(json.dumps({"primary_results": doc["primary_results"],
                      "d1_open": doc["per_dataset"]["d1_open"]["primary"]["one_sided_tests"],
                      "pooled": {k: doc["pooled"][k] for k in KINDS}, "pooled_status": doc["pooled"]["status"]},
                     indent=1)[:4000])
    return 0


def f4(x):
    return f"{x:+.4f}"


def pf(p):
    return f"< {1 / B:.0e}" if p == 0 else f"{p:.4f}"


def _tests_table(t, h) -> list:
    return ["| test | p | rejects | Holm (α = 0.025) |", "|---|---|---|---|",
            f"| P1: Δ[C] > 0 | {pf(t['P1']['p'])} | {t['P1']['reject']} | {h['P1']['reject']} |",
            f"| P2: Δ[C] − Δ[S] > 0 | {pf(t['P2']['p'])} | {t['P2']['reject']} | {h['P2']['reject']} |",
            f"| P3: TOST ±0.016 | upper {pf(t['P3']['p_upper'])} / lower {pf(t['P3']['p_lower'])} | "
            f"{t['P3']['pass']} | {h['P3']['reject']} |",
            f"| H-REV: Δ[C] < 0 | {pf(t['H-REV']['p'])} | {t['H-REV']['reject']} | {h['H-REV']['reject']} |"]


def _est_row(name, p) -> str:
    return f"| {name} | " + " | ".join(f"{f4(p[k]['estimate'])} [{f4(p[k]['ci95'][0])}, {f4(p[k]['ci95'][1])}]"
                                       for k in KINDS) + " |"


def _pvo_line(pvo) -> str:
    return (f"{pvo['n_cells']} cells: Spearman {pvo['spearman']:.3f} (one-sided 95% lower "
            f"{pvo['spearman_one_sided_95_lower']:.3f}); IVW slope {pvo['ivw_slope']:.3f} "
            f"[{pvo['ivw_slope_ci95'][0]:.3f}, {pvo['ivw_slope_ci95'][1]:.3f}]. Verdict: **{pvo['verdict']}**.")


def write_md(doc):
    P = doc["pooled"]
    reason = doc.get("d2_status_reason")
    st = lambda s: f"*[{s.upper() if s.startswith('exploratory') else s}]*"  # noqa: E731
    d1 = doc["per_dataset"]["d1_open"]["primary"]
    d2 = doc["per_dataset"]["d2_support"]["primary"]
    hdr = ["Frozen code `ddde67f` + D1 (PREREG_V2 Deviations; D2 gate supplement; notes N1, N3). Primary cell: "
           "rewrite_faiss, 0.65 belief-quantile price, primary ASR condition. D\\*[COMMIT] = B9+; D\\*[SPEC] = B9+ "
           "(d1_open), SPEC (d2_support); m₃ = 0.016. Paired cluster bootstrap, B = 10,000, stratified by dataset, "
           "pooled weights 3,734 : 498. Source: `docs/research_v2/v2_test_results_final.json`.", ""]
    if reason is None:
        L = ["# v2 test results (confirmatory, pooled over d1_open and d2_support)", ""] + hdr
    else:
        L = ["# v2 test results", "",
             f"**Status: d1_open (alias-graded) results are CONFIRMATORY and primary. Every result that uses "
             f"d2_support's judge-graded outcomes — d2_support itself, every pooled result, the 6-test secondary family, "
             f"the 48-cell predicted-vs-observed check and the two-dataset E1 kill rule — is EXPLORATORY ({reason}).**",
             ""] + hdr
        L += [f"## Confirmatory: d1_open, primary cell {st('confirmatory')}", "",
              "| | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |", "|---|---|---|---|",
              _est_row("d1_open", d1), ""] + _tests_table(d1["one_sided_tests"], d1["holm_individual_claims"]) + [
              "", f"**Boundary claim (IUT P1 ∧ P2 ∧ P3), d1_open: {d1['IUT_boundary_claim']}.**", "",
              "Secondary (d1_open; two-sided, Holm α = 0.05 over its 3 tests): " + "; ".join(
                  f"{k} p {pf(v['p'])} rejects {v['reject']}" for k, v in d1["secondary_family_this_dataset"].items())]
        pv1 = (doc.get("predicted_vs_observed_by_dataset") or {}).get("d1_open")
        if pv1:
            L += ["", f"Predicted vs observed (d1_open): {_pvo_line(pv1)}"]
        g1 = (doc.get("e1_gate") or {}).get("d1_open") or {}
        if g1.get("auc_full"):
            L += ["", f"H-E1v2 (d1_open): AUC {g1['auc_full']['auc']:.3f} [{g1['auc_full']['ci95'][0]:.3f}, "
                      f"{g1['auc_full']['ci95'][1]:.3f}]; success {g1['success_ge_0.70']}; C2 available "
                      f"{g1['c2_available']}."]
        L += ["", f"# EXPLORATORY ({reason}): d2_support and pooled", ""]
    L += [f"## Headline (primary cell) {st(P['status'])}", "",
          "| | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |", "|---|---|---|---|",
          _est_row(f"d1_open {st(doc['per_dataset']['d1_open']['status'])}", d1),
          _est_row(f"d2_support {st(doc['per_dataset']['d2_support']['status'])}", d2),
          _est_row(f"**pooled** {st(P['status'])}", P), "",
          f"## Pooled tests (one-sided α = 0.025) {st(P['status'])}", ""] + \
        _tests_table(P["confirmatory_tests"], P["holm_individual_claims"]) + [
          "", f"**Boundary claim (IUT P1 ∧ P2 ∧ P3), pooled: {P['IUT_boundary_claim']}** {st(P['status'])}.", "",
          f"## d2_support per-dataset tests {st(doc['per_dataset']['d2_support']['status'])}", ""] + \
        _tests_table(d2["one_sided_tests"], d2["holm_individual_claims"]) + [
          "", f"## Secondary family (per dataset, two-sided, Holm α = 0.05 over 6) {st(doc['secondary_family_status'])}",
          "", "| test | p | rejects |", "|---|---|---|"]
    for k, v in doc["secondary_family"].items():
        L.append(f"| {k} | {pf(v['p'])} | {v['reject']} |")
    pvo = doc.get("predicted_vs_observed")
    if pvo:
        L += ["", f"## Predicted vs observed (Δ[COMMIT], both datasets) {st(pvo['status'])}", "", _pvo_line(pvo)]
    g = doc["e1_gate"]
    L += ["", "## H-E1v2 gate (test)", ""]
    for d in ("d1_open", "d2_support"):
        v = g.get(d, {})
        if v.get("auc_full"):
            L.append(f"* {d} {st(v['status'])}: AUC {v['auc_full']['auc']:.3f} [{v['auc_full']['ci95'][0]:.3f}, "
                     f"{v['auc_full']['ci95'][1]:.3f}]; success {v['success_ge_0.70']}; C2 available {v['c2_available']}.")
    L.append(f"* Kill (AUC < 0.60 on both) {st(g['kill_rule_status'])}: {g.get('kill_auc_below_0.60_on_both')}.")
    jc = doc.get("judge_checks") or {}
    if "cross_key_primary_agreement" in jc:
        c = jc["cross_key_primary_agreement"]
        s = jc["e1_labels_secondary"]
        L += ["", f"## Judge checks (d2_support) {st(jc['status'])}", "",
              f"* Cross-key consistency: {c['n_shared_item_answer_texts']} (item, answer) texts graded by both label.py "
              f"(k = 1 key) and Phase A (k = 3 key, sample 0); primary agreement {c['agreement']}.",
              f"* E1 label draws, primary vs secondary: pre-registered stratified 30% subsample "
              f"{s['preregistered_stratified_30pct']}; all draws (superset run) {s['all_draws_superset']}."]
        if "phase_a_secondary_30pct" in jc:
            L.append(f"* Phase A answers, primary vs secondary (30% subsample): {jc['phase_a_secondary_30pct']}.")
    L += ["", "## All cells (descriptive)", "",
          "| dataset | condition | profile | q | Δ[COMMIT] | Δ[SPEC] | interaction | status |",
          "|---|---|---|---|---|---|---|---|"]
    for c in doc["cells"]:
        cond = "ASR" if c["condition"] == ASR else c["condition"]
        L.append(f"| {c['dataset']} | {cond} | {c['profile']} | {c['belief_quantile']} | " + " | ".join(
            f"{f4(c[k]['estimate'])} [{f4(c[k]['ci95'][0])}, {f4(c[k]['ci95'][1])}]" for k in KINDS)
            + f" | {c['status']} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
