"""
v2_test_d1.py -- d1_open-ONLY test analysis (PREREG_V2 §3, §4.1, §5), run
after test generation and alias grading. NOT the confirmatory pooled test:
the pooled verdicts wait for d2_support grading after the judge gate.

    PYTHONUTF8=1 python -m analysis.v2_test_d1

Inputs (all produced by the frozen code):
  * simulator/e3v2_phase_a/d1_open/contrasts_test.json
        python -m simulator.e3v2_lazy evaluate --dataset d1_open --split test
            --bundle <asr bundle> --bundle <text150 bundle>
            --d-star-speculative b9plus_v2 --m3 0.016
    per (condition, latency profile, price) per-item paired differences
    (Ours-v2[arch] - D*[arch]; D*[COMMIT] = D*[SPECULATIVE] = B9+ for d1).
  * the frozen dev bundles (price grid -> belief quantile, dev g)
  * <E1 results>/d1_open/analysis_test/analysis_test.json
        python -m llm.experiment1_v2.analyze --dataset d1_open --split test
    (AUC on the full question, prefix and ASR curves, C2 flag)
  * the dev-fitted operational calibration
    controller/baselines/e3v2_calibration_d1_open_mistral7b_v03_self_assessed_need_logit.json
    applied to the test self-assessment records (ECE, 10 bins; boundary mass).

Statistics (PREREG_V2 §4.1, §5): paired bootstrap over items (d1's
clusters are items), B = 10,000, the SAME resamples for every contrast;
percentile CIs; one-sided bootstrap p-values:
  P1     H0: delta_C <= 0          p = P*(Dbar_C <= 0)     reject iff q2.5 > 0
  H-REV  H0: delta_C >= 0          p = P*(Dbar_C >= 0)     reject iff q97.5 < 0
  P2     H0: delta_C - delta_S <= 0 (items in both)        reject iff q2.5 > 0
  P3     TOST at m3 = 0.016: upper H0 delta_S >= m3 (p = P*(Dbar_S >= m3)),
         lower H0 delta_S <= -m3 (p = P*(Dbar_S <= -m3)); passes iff
         [q2.5, q97.5] lies inside (-m3, m3); p(P3) = the larger side.
A p-value of 0 means no resample crossed the null (p < 1/B).
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

from llm.experiment1_v2 import common as e1common  # noqa: E402
from llm.experiment1_v2.prefixes import normalize_text  # noqa: E402
from llm.experiment1_v2.stats import ece as ece_fn  # noqa: E402

DS = "d1_open"
PA = _REPO_ROOT / "simulator" / "e3v2_phase_a" / DS
E1 = Path(r"C:\Users\ryang\OneDrive\Desktop\voice-rag\voice-rag-controller\llm\experiment1_v2\results") / DS
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_d1.json"
OUT_MD = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_d1.md"
B = 10_000
SEED = 20261017
M3 = 0.016
PRIMARY = {"condition": "piper-en_US-hfc_male-medium__150wpm__clean", "profile": "rewrite_faiss",
           "quantile": 0.65, "lambda_l": 0.17305735434823866, "lambda_c": 0.05}
LABEL = "d1_open only; pooled confirmatory results pending d2 grading after the judge gate"
CAL = _REPO_ROOT / "controller" / "baselines" / "e3v2_calibration_d1_open_mistral7b_v03_self_assessed_need_logit.json"


def bootstrap_counts(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    counts = np.zeros((B, n), dtype=np.float64)
    for b in range(B):
        counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
    return counts


def boot_means(counts, universe_idx, values: dict) -> tuple:
    """Bootstrap means of a per-item contrast under the shared resamples."""
    x = np.zeros(counts.shape[1])
    mask = np.zeros(counts.shape[1])
    for i, v in values.items():
        x[universe_idx[i]] = float(v)
        mask[universe_idx[i]] = 1.0
    tot = counts @ mask
    means = (counts @ x) / np.where(tot > 0, tot, np.nan)
    return float(np.mean(list(values.values()))), means


def summarize(est, means, null_low=None, null_high=None) -> dict:
    q = np.nanpercentile(means, [2.5, 97.5])
    out = {"estimate": est, "ci95": [float(q[0]), float(q[1])], "boot_se": float(np.nanstd(means, ddof=1))}
    return out


def contrast_block(counts, uidx, per_item) -> dict:
    dc_est, dc = boot_means(counts, uidx, per_item["delta_commit"])
    ds_est, ds = boot_means(counts, uidx, per_item["delta_speculative"])
    it_est, it = boot_means(counts, uidx, per_item["interaction"])
    c, s, i = summarize(dc_est, dc), summarize(ds_est, ds), summarize(it_est, it)
    p1 = float(np.nanmean(dc <= 0))
    hrev = float(np.nanmean(dc >= 0))
    p2 = float(np.nanmean(it <= 0))
    p3_up = float(np.nanmean(ds >= M3))
    p3_lo = float(np.nanmean(ds <= -M3))
    return {
        "n_items": {"commit": len(per_item["delta_commit"]), "speculative": len(per_item["delta_speculative"]),
                    "paired": len(per_item["interaction"])},
        "delta_commit": c, "delta_speculative": s, "interaction": i,
        "tests_d1_only": {
            "P1": {"H0": "delta_C <= 0", "p_one_sided": p1, "reject": bool(c["ci95"][0] > 0)},
            "H-REV": {"H0": "delta_C >= 0", "p_one_sided": hrev, "reject": bool(c["ci95"][1] < 0)},
            "P2": {"H0": "delta_C - delta_S <= 0", "p_one_sided": p2, "reject": bool(i["ci95"][0] > 0)},
            "P3_TOST": {"m3": M3, "p_upper": p3_up, "p_lower": p3_lo, "p": max(p3_up, p3_lo),
                        "reject_upper": bool(s["ci95"][1] < M3), "reject_lower": bool(s["ci95"][0] > -M3),
                        "pass": bool(-M3 < s["ci95"][0] and s["ci95"][1] < M3)},
        },
    }


def holm(pvals: dict, alpha: float = 0.025) -> dict:
    order = sorted(pvals, key=lambda k: pvals[k])
    out, stop = {}, False
    for r, k in enumerate(order):
        thr = alpha / (len(order) - r)
        rej = (not stop) and pvals[k] <= thr
        stop = stop or not rej
        out[k] = {"p": pvals[k], "holm_threshold": thr, "reject": rej}
    return out


def e1_block() -> dict:
    """Primary AUC / curves / C2 from the frozen analyze.py; ECE and boundary
    mass of the DEV-fitted operational calibration applied to test."""
    out = {}
    a_path = E1 / "analysis_test" / "analysis_test.json"
    if a_path.exists():
        out["analyze_py"] = {"path": str(a_path), "sha256": e1common.sha256_file(a_path)}
        out["analyze_py_model"] = e1common.load_json(a_path)["models"]["mistral7b_v03"]
    cal = e1common.load_json(CAL)["calibration"]
    labels = {i: v.get("empirical_need") for i, v in
              e1common.load_json(E1 / "mistral7b_v03" / "labels_test.json")["items"].items()}
    items = {it["id"]: it for it in e1common.load_items(e1common.resolve_dataset_paths(DS)["items"])}
    sa = {}
    for r in e1common.read_jsonl(E1 / "mistral7b_v03" / "self_assessment_test.jsonl"):
        sa[(r["item_id"], normalize_text(r["prefix_text"]))] = float(r["self_assessed_need"])

    def belief(v, f):
        p = min(1 - 1e-4, max(1e-4, v))
        x = math.log(p / (1 - p))
        z = cal["intercept"] + cal["slope"] * x + cal["frac_slope"] * f + cal["interaction"] * x * f
        return 1 / (1 + math.exp(-max(-40, min(40, z))))

    # end-of-turn beliefs on the full TEXT question and on the primary ASR final transcript
    asr_final = {}
    for line in open(PA / "conditions_test.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        if tr["condition"] == PRIMARY["condition"]:
            asr_final[tr["item_id"]] = normalize_text(tr["asr_final"]["text"])
    g = e1common.load_json(PA / "analysis" / "bundle_asr_3e1c1ada.json")["overrides"]["quality"]
    b_star = g["g_minus"] / (g["g_plus"] + g["g_minus"])
    blocks = {}
    for name, text_of in (("full_text_question", lambda i: normalize_text(items[i]["question"])),
                          ("primary_asr_final", lambda i: asr_final.get(i))):
        ys, ps = [], []
        for i, y in labels.items():
            if y is None:
                continue
            t = text_of(i)
            v = sa.get((i, t)) if t else None
            if v is None:
                continue
            ys.append(int(bool(y)))
            ps.append(belief(v, 1.0))
        ys, ps = np.array(ys), np.array(ps)
        blocks[name] = {
            "n_items": int(ys.size), "base_rate": float(ys.mean()), "mean_belief": float(ps.mean()),
            "ece_10_bins_dev_calibration": ece_fn(ys, ps, 10),
            "boundary_mass": {"b_star": b_star, "half_width": 0.15,
                              "fraction": float(np.mean(np.abs(ps - b_star) <= 0.15)),
                              "prediction": ">= 0.15", "dev_g": g},
        }
    out["dev_calibration_on_test"] = blocks

    # gate criterion, text prefix curve (+ paired bootstrap of AUC(1.0) - AUC(0.3)), C2 flag
    m = out.get("analyze_py_model") or {}
    full = ((m.get("auc_full") or {}).get("self_assessed_need") or {}).get("empirical")
    curve = {str(r["word_frac"]): r["empirical"] for r in (m.get("prefix_curve_text") or {}).get("self_assessed_need", [])}
    from llm.experiment1_v2.prefixes import text_prefixes
    y_of = {i: int(bool(v)) for i, v in labels.items() if v is not None}
    ids = sorted(y_of)
    texts = {}
    for i in ids:
        for pre in text_prefixes(items[i]["question"], [0.3, 0.5, 0.7]):
            f = 1.0 if pre.get("kind") == "full" else float(pre.get("word_frac"))
            texts[(i, f)] = normalize_text(pre["text"])
    def sa_at(i, f):
        return sa.get((i, texts.get((i, f))))
    y = np.array([y_of[i] for i in ids])
    s03 = np.array([sa_at(i, 0.3) if sa_at(i, 0.3) is not None else np.nan for i in ids])
    s10 = np.array([sa_at(i, 1.0) if sa_at(i, 1.0) is not None else np.nan for i in ids])
    ok = ~np.isnan(s03) & ~np.isnan(s10)
    diff = paired_auc_diff(y[ok], s10[ok], s03[ok])
    out["gate_and_curves"] = {
        "primary_auc_full_question": full,
        "gate_d1_only": {"success_auc_ge_0.70": None if not full else bool(full["auc"] >= 0.70),
                         "below_kill_0.60": None if not full else bool(full["auc"] < 0.60),
                         "note": "the kill rule needs AUC < 0.60 on BOTH datasets; d2_support pending"},
        "text_prefix_curve": curve,
        "auc_1.0_minus_auc_0.3": diff,
        "c2_available_auc_0.5_ge_0.65": None if "0.5" not in curve else bool(curve["0.5"]["auc"] >= 0.65),
        "asr_curve_primary_condition": asr_curve(y_of, sa),
    }
    return out


def auc(y, s) -> float:
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = y.size - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_auc_diff(y, s_hi, s_lo, n_boot: int = B) -> dict:
    rng = np.random.default_rng(SEED + 1)
    est = auc(y, s_hi) - auc(y, s_lo)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, y.size, size=y.size)
        if 0 < y[k].sum() < y.size:
            d.append(auc(y[k], s_hi[k]) - auc(y[k], s_lo[k]))
    d = np.array(d)
    q = np.percentile(d, [2.5, 97.5])
    return {"estimate": est, "ci95": [float(q[0]), float(q[1])], "p_one_sided_H0_le_0": float(np.mean(d <= 0)),
            "n_items": int(y.size), "B": int(d.size)}


def asr_curve(y_of: dict, sa: dict) -> dict:
    """SA AUC at the latest primary-ASR partial with spoken word fraction <= f,
    and on the ASR final transcript (computed here: the frozen analyze.py curves
    need ASR signal records, which the test E1 run does not produce; note N1)."""
    from simulator.e3v2_dev_analysis import prefix_frac
    out = {}
    trs = {}
    for line in open(PA / "conditions_test.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        if tr["condition"] == PRIMARY["condition"]:
            trs[tr["item_id"]] = tr
    for f in (0.3, 0.5, 0.7, 1.0):
        ys, ss = [], []
        for i, tr in trs.items():
            if i not in y_of:
                continue
            if f == 1.0:
                v = sa.get((i, normalize_text(tr["asr_final"]["text"])))
            else:
                v = None
                for k, p in enumerate(tr["partials"]):
                    if not normalize_text(p.get("text", "")):
                        continue
                    if prefix_frac(tr, k) <= f:
                        v = sa.get((i, normalize_text(p["text"])), v)
            if v is not None:
                ys.append(y_of[i])
                ss.append(v)
        ys, ss = np.array(ys), np.array(ss)
        out[str(f)] = {"n_items": int(ys.size), "auc": auc(ys, ss) if 0 < ys.sum() < ys.size else None}
    return out


def main() -> int:
    doc_c = e1common.load_json(PA / "contrasts_test.json")
    bundles = {}
    for bpath in doc_c["bundles"]:
        b = e1common.load_json(Path(bpath))
        bundles[b["condition"]] = b
    # universe of items (d1 clusters = items) and the shared resamples
    universe = sorted({i for rows in doc_c["contrasts"].values() for r in rows
                       for k in ("delta_commit", "delta_speculative") for i in r["per_item"][k]})
    uidx = {i: k for k, i in enumerate(universe)}
    counts = bootstrap_counts(len(universe))
    cells = []
    primary = None
    for cond, rows in doc_c["contrasts"].items():
        grid = bundles[cond]["price_grid"]
        for r in rows:
            prof = r["latency_profile"]
            q = next((g["quantile"] for g in grid[prof] if f"{g['lambda_l']:g}|{g['lambda_c']:g}" == r["price"]),
                     None)
            blk = contrast_block(counts, uidx, r["per_item"])
            cell = {"condition": cond, "condition_role": "primary" if cond == PRIMARY["condition"] else "secondary",
                    "profile": prof, "belief_quantile": q, "price": r["price"], "d_star": r["d_star"],
                    "m3": r["m3"], "q_hat": r.get("q_hat"), "w_bar": r.get("w_bar"), **blk}
            cells.append(cell)
            if cond == PRIMARY["condition"] and prof == PRIMARY["profile"] and q == PRIMARY["quantile"]:
                primary = cell
    assert primary is not None, "primary cell missing"
    assert primary["price"] == f"{PRIMARY['lambda_l']:g}|{PRIMARY['lambda_c']:g}"
    t = primary["tests_d1_only"]
    pv = {"P1": t["P1"]["p_one_sided"], "P2": t["P2"]["p_one_sided"], "P3": t["P3_TOST"]["p"],
          "H-REV": t["H-REV"]["p_one_sided"]}
    primary["holm_d1_only"] = holm(pv)
    primary["boundary_claim_IUT_d1_only"] = bool(t["P1"]["reject"] and t["P2"]["reject"] and t["P3_TOST"]["pass"])
    e1 = e1_block()
    doc = {
        "_comment": LABEL + ". Descriptive per-dataset statistics; the pre-registered confirmatory tests are pooled "
                            "over d1_open and d2_support (PREREG_V2 §3.2, §4.1).",
        "status": LABEL,
        "generated_by": "python -m analysis.v2_test_d1", "generated_at_utc": e1common.now_utc(),
        "git_commit": e1common.git_commit(), "split": "test", "dataset": DS,
        "frozen": {"prereg_tag": "prereg-v2-frozen", "frozen_code": "ddde67f", "deviations": ["D1 (43ca350)"],
                   "d_star": {"commit": "b9plus_v2", "speculative": "b9plus_v2"}, "m3": M3,
                   "primary_cell": PRIMARY},
        "inputs": {"contrasts": str(PA / "contrasts_test.json"),
                   "contrasts_sha256": e1common.sha256_file(PA / "contrasts_test.json"),
                   "bundles": {c: {"path": str(p), "sha256": e1common.sha256_file(Path(p))}
                               for c, p in zip(bundles, doc_c["bundles"])},
                   "required_cells_sha256": e1common.sha256_file(PA / "required_test.json")},
        "bootstrap": {"B": B, "seed": SEED, "unit": "item (d1_open clusters are items)",
                      "same_resamples_for_every_contrast": True, "n_items_universe": len(universe)},
        "e1": e1, "primary": primary, "cells": cells,
    }
    e1common.atomic_write_json(OUT_JSON, doc)
    write_md(doc)
    print(json.dumps({k: primary[k] for k in ("delta_commit", "delta_speculative", "interaction")}, indent=1))
    print(json.dumps(primary["tests_d1_only"], indent=1))
    return 0


def fmt(x, nd=4):
    return "—" if x is None else f"{x:+.{nd}f}"


def pfmt(p):
    return f"< {1 / B:.0e}" if p == 0 else f"{p:.4f}"


def write_md(doc):
    p = doc["primary"]
    t = p["tests_d1_only"]
    e1 = doc["e1"]
    L = [f"# v2 test results — d1_open only", "",
         f"**Status: {LABEL}.** Every p-value below is per-dataset and labelled *d1-only, not the confirmatory "
         f"pooled test*. Frozen code `ddde67f` plus deviation D1; D\\*[COMMIT] = D\\*[SPECULATIVE] = B9+; "
         f"m₃ = {M3}. Paired bootstrap over items, B = {B:,}, the same resamples for every contrast. "
         f"Source: `docs/research_v2/v2_test_results_d1.json` (`analysis/v2_test_d1.py`).", "",
         "## Primary cell (rewrite_faiss, belief quantile 0.65, λ_l = 0.17305735434823866, λ_c = 0.05, primary ASR)", "",
         "| contrast | estimate | 95% CI | n items |", "|---|---|---|---|"]
    for k, lab, n in (("delta_commit", "Δ[COMMIT]", "commit"), ("delta_speculative", "Δ[SPECULATIVE]", "speculative"),
                      ("interaction", "interaction Δ[C] − Δ[S]", "paired")):
        c = p[k]
        L.append(f"| {lab} | {fmt(c['estimate'])} | [{fmt(c['ci95'][0])}, {fmt(c['ci95'][1])}] | {p['n_items'][n]:,} |")
    L += ["", "**One-sided tests (d1-only, not the confirmatory pooled test):**", "",
          "| test | H0 | p (bootstrap) | rejects at α = 0.025 | Holm (d1-only) |", "|---|---|---|---|---|",
          f"| P1 | Δ[C] ≤ 0 | {pfmt(t['P1']['p_one_sided'])} | {t['P1']['reject']} | {p['holm_d1_only']['P1']['reject']} |",
          f"| P2 | Δ[C] − Δ[S] ≤ 0 | {pfmt(t['P2']['p_one_sided'])} | {t['P2']['reject']} | {p['holm_d1_only']['P2']['reject']} |",
          f"| P3 (TOST, ±{M3}) | upper Δ[S] ≥ m₃ / lower Δ[S] ≤ −m₃ | {pfmt(t['P3_TOST']['p_upper'])} / "
          f"{pfmt(t['P3_TOST']['p_lower'])} | {t['P3_TOST']['pass']} (upper {t['P3_TOST']['reject_upper']}, "
          f"lower {t['P3_TOST']['reject_lower']}) | {p['holm_d1_only']['P3']['reject']} |",
          f"| H-REV | Δ[C] ≥ 0 | {pfmt(t['H-REV']['p_one_sided'])} | {t['H-REV']['reject']} | "
          f"{p['holm_d1_only']['H-REV']['reject']} |", "",
          f"Boundary claim (P1 ∧ P2 ∧ P3), d1-only: **{p['boundary_claim_IUT_d1_only']}**.", ""]
    sec = next((c for c in doc["cells"] if c["condition_role"] == "secondary" and c["profile"] == "rewrite_faiss"
                and c["belief_quantile"] == 0.65), None)
    if sec:
        L += [f"**Secondary condition text150** (rewrite_faiss, its own 0.65-quantile price {sec['price']}): "
              f"Δ[COMMIT] {fmt(sec['delta_commit']['estimate'])} [{fmt(sec['delta_commit']['ci95'][0])}, "
              f"{fmt(sec['delta_commit']['ci95'][1])}], Δ[SPEC] {fmt(sec['delta_speculative']['estimate'])} "
              f"[{fmt(sec['delta_speculative']['ci95'][0])}, {fmt(sec['delta_speculative']['ci95'][1])}], "
              f"interaction {fmt(sec['interaction']['estimate'])}.", ""]
    L += ["## All cells (descriptive)", "",
          "| condition | profile | q | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |",
          "|---|---|---|---|---|---|"]
    for c in sorted(doc["cells"], key=lambda c: (c["condition_role"], c["profile"], c["belief_quantile"] or 0)):
        role = "ASR (primary)" if c["condition_role"] == "primary" else "text150"
        L.append(f"| {role} | {c['profile']} | {c['belief_quantile']} | "
                 + " | ".join(f"{fmt(c[k]['estimate'])} [{fmt(c[k]['ci95'][0])}, {fmt(c[k]['ci95'][1])}]"
                              for k in ("delta_commit", "delta_speculative", "interaction")) + " |")
    L += ["", "## E1-v2 on test (d1_open only)", ""]
    m = e1.get("analyze_py_model") or {}
    L.append(f"* Frozen `analyze.py` output: `{(e1.get('analyze_py') or {}).get('path')}`. The gate criterion "
             "(success AUC ≥ 0.70; kill AUC < 0.60 on both datasets) is reported for d1 only.")
    for name, blk in e1["dev_calibration_on_test"].items():
        L.append(f"* Dev-fitted calibration on test, {name}: ECE (10 bins) {blk['ece_10_bins_dev_calibration']:.4f}; "
                 f"base rate {blk['base_rate']:.3f}, mean belief {blk['mean_belief']:.3f}; boundary mass "
                 f"{blk['boundary_mass']['fraction']:.3f} within ±0.15 of b* = {blk['boundary_mass']['b_star']:.3f} "
                 "(prediction ≥ 0.15).")
    g = e1.get("gate_and_curves") or {}
    full = g.get("primary_auc_full_question")
    if full:
        L.append(f"* **Primary (gate) AUC**, raw self_assessed_need on the full question vs the empirical label: "
                 f"**{full['auc']:.3f}** [{full['ci95'][0]:.3f}, {full['ci95'][1]:.3f}] (n = {full['n_items']:,}, "
                 f"{full['n_pos']:,} needed). Success (≥ 0.70): {g['gate_d1_only']['success_auc_ge_0.70']}; below "
                 f"0.60: {g['gate_d1_only']['below_kill_0.60']} (the kill rule needs both datasets).")
    if g.get("text_prefix_curve"):
        L.append("* Text prefix curve (AUC at word_frac 0.3 / 0.5 / 0.7 / 1.0): "
                 + " / ".join(f"{g['text_prefix_curve'][k]['auc']:.3f}" for k in ("0.3", "0.5", "0.7", "1.0")
                              if k in g["text_prefix_curve"])
                 + f"; AUC(1.0) − AUC(0.3) = {g['auc_1.0_minus_auc_0.3']['estimate']:+.3f} "
                   f"[{g['auc_1.0_minus_auc_0.3']['ci95'][0]:+.3f}, {g['auc_1.0_minus_auc_0.3']['ci95'][1]:+.3f}] "
                   f"(paired bootstrap, B = {g['auc_1.0_minus_auc_0.3']['B']:,}; one-sided p "
                   f"{pfmt(g['auc_1.0_minus_auc_0.3']['p_one_sided_H0_le_0'])}).")
        L.append(f"* C2 available (AUC(0.5) ≥ 0.65): **{g['c2_available_auc_0.5_ge_0.65']}**.")
    if g.get("asr_curve_primary_condition"):
        a = g["asr_curve_primary_condition"]
        L.append("* Primary-ASR curve (latest partial with spoken word fraction ≤ f; f = 1.0 is the ASR final "
                 "transcript; computed by `analysis/v2_test_d1.py` because the test E1 run has no ASR signal records, "
                 "note N1): " + " / ".join(f"f={k}: {v['auc']:.3f} (n={v['n_items']:,})" for k, v in a.items()) + ".")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
