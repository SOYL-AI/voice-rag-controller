"""
paper_figures_v1.py -- every v1 figure and every v1 number quoted in paper_v2/,
regenerated from committed repository data with one command:

    PYTHONUTF8=1 python -m analysis.paper_figures_v1

Inputs (all committed, all REAL runs):
    simulator/e3_sweep_results/e3_<baseline>_seed<seed>.json   E3 Phase B records
    simulator/e3_sweep_cache/answers/*.json                    E3 judged answers
    simulator/e3_scenario_signals.json                         frozen belief signals
    simulator/scenarios.json                                   E3 scenarios + labels
    configs/e3_price_grid.json                                 E3 lambda_l grid
    controller/baselines/*                                     the controllers' own cuts
    llm/experiment1/dataset.jsonl                              E1 gate questions
    logs/conv_*.jsonl                                          E2 raw event logs

Outputs:
    paper_v2/figures/fig_e3_acc_vs_p95.pdf        accuracy vs p95 TTFA (Ours, B9, B0, B6, B8)
    paper_v2/figures/fig_e3_acc_vs_compute.pdf    accuracy vs retrievals per turn
    paper_v2/figures/fig_e3_price_shift.pdf       Ours(lambda_i) = B9(lambda_{i-1})
    paper_v2/figures/fig_e3_band_occupancy.pdf    divergence-band occupancy vs price
    paper_v2/figures/fig_e3_per_seed.pdf          per-seed panels (supplementary)
    paper_v2/figures/fig_e2_latency_breakdown.pdf E2 stage medians relative to user end
    analysis/paper_v1_numbers.json                every number the paper quotes, with
                                                  the computation that produced it

The synthetic sample in e3_sample_result/ is NEVER read. load_e3_records()
refuses any path under it.

Statistical unit. E3 decisions are a deterministic function of frozen,
per-scenario belief signals, so they do not vary across seeds; the seed
varies only the sampled answer (temperature 0.2) and the measured TTFA of
that cell. Seeds are therefore repeated measurements of the same 50
scenarios, not independent replicates. Every confidence interval here is a
cluster bootstrap that resamples SCENARIOS and carries all five seeds of a
drawn scenario together.

Python: system 3.12 is fine (matplotlib, numpy, yaml). Nothing here imports
pyarrow, faiss or faster-whisper.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from controller.baselines._objective import (  # noqa: E402
    Prices,
    divergence_band,
    load_e3_config,
)

E3_RESULTS = REPO / "simulator" / "e3_sweep_results"
E3_ANSWERS = REPO / "simulator" / "e3_sweep_cache" / "answers"
SIGNALS = REPO / "simulator" / "e3_scenario_signals.json"
SCENARIOS = REPO / "simulator" / "scenarios.json"
GRID = REPO / "configs" / "e3_price_grid.json"
E1_DATASET = REPO / "llm" / "experiment1" / "dataset.jsonl"
E1_RESULT = REPO / "llm" / "experiment1" / "results" / "experiment_1_modal_mistral7b_20260811_151212.json"
CALIBRATION = REPO / "controller" / "baselines" / "e1_calibration_mean_entropy.json"
E2_LOGS = REPO / "logs"
SYNTHETIC = (REPO / "e3_sample_result").resolve()

SEEDS = (11, 23, 37, 53, 71)
BOOT_ITER = 2000
BOOT_SEED = 12345

# ---------------------------------------------------------------------------
# Figure style: IEEE single column (3.5 in), fonts at or above 8 pt at print
# size, no titles, vector output with embedded TrueType fonts.
# Colour: two categorical slots (validated all-pairs for CVD and normal
# vision) for the two priced frontiers; unpriced baselines are neutral ink
# with distinct marker shapes and direct labels, so identity never rests on
# colour alone and the figures survive greyscale printing.
# ---------------------------------------------------------------------------
C_OURS = "#2a78d6"
C_B9 = "#eb6834"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#d9d8d4"
STYLE = {
    "ours": dict(color=C_OURS, marker="o", ls="-", label="Ours (joint)"),
    "b9": dict(color=C_B9, marker="s", ls="--", label="B9 (decomposed, priced)"),
    "b0": dict(color=C_INK, marker="X", label="B0 never"),
    "b6": dict(color=C_INK2, marker="^", label="B6 DRAGIN"),
    "b8": dict(color=C_INK2, marker="D", label="B8 indep.-best"),
}
COL_W = 3.5


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": C_INK2,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.titlesize": 9,
    })


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _refuse_synthetic(path: Path):
    p = Path(path).resolve()
    if p == SYNTHETIC or SYNTHETIC in p.parents:
        raise SystemExit(
            f"refusing to read {p}: e3_sample_result/ is SYNTHETIC test data "
            "(see its RUN_MANIFEST.md) and must never feed a paper figure"
        )


def load_e3_records(results_dir: Path = E3_RESULTS):
    _refuse_synthetic(results_dir)
    recs = []
    for f in sorted(Path(results_dir).glob("e3_*_seed*.json")):
        _refuse_synthetic(f)
        recs.extend(json.loads(f.read_text(encoding="utf-8")))
    return recs


def load_grid():
    g = json.loads(GRID.read_text(encoding="utf-8"))
    return [float(x) for x in g["grid"]["lambda_l_grid"]], float(g["grid"]["lambda_c"]), g


def price_index(grid, lam):
    for i, g in enumerate(grid):
        if math.isclose(g, lam, rel_tol=1e-12, abs_tol=1e-15):
            return i
    raise KeyError(lam)


def organise(recs, grid):
    """table[baseline][price_idx or None][seed][scenario] = record"""
    t = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in recs:
        pi = None if r["prices"] is None else price_index(grid, r["prices"]["lambda_l"])
        t[r["baseline"]][pi][r["seed"]][r["scenario_id"]] = r
    return t


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def pct(values, p):
    """Linear-interpolation percentile; identical to analysis/e3_frontier.percentile
    and to numpy's default, so p95 values match e3_dominance.json exactly."""
    return float(np.percentile(np.asarray(values, dtype=float), p))


def op_point(cell):
    """cell: {seed: {scenario: rec}} -> pooled (acc, p95, compute) over all (scenario, seed)."""
    acc, ttfa, comp = [], [], []
    for seed in cell:
        for rec in cell[seed].values():
            acc.append(1.0 if rec["outcome"]["answer_correct"] else 0.0)
            ttfa.append(rec["outcome"]["ttfa_ms"])
            comp.append(rec["outcome"]["retrievals"])
    return float(np.mean(acc)), pct(ttfa, 95), float(np.mean(comp))


def op_point_seed(cell_seed):
    recs = list(cell_seed.values())
    acc = np.mean([1.0 if r["outcome"]["answer_correct"] else 0.0 for r in recs])
    p95 = pct([r["outcome"]["ttfa_ms"] for r in recs], 95)
    comp = np.mean([r["outcome"]["retrievals"] for r in recs])
    return float(acc), p95, float(comp)


def cluster_boot(cell, rng_seed=BOOT_SEED, iters=BOOT_ITER):
    """95% CI for (acc, p95, compute), resampling scenarios with all their seeds."""
    scen = sorted(next(iter(cell.values())).keys())
    seeds = sorted(cell.keys())
    A = np.array([[1.0 if cell[s][sc]["outcome"]["answer_correct"] else 0.0 for s in seeds] for sc in scen])
    T = np.array([[cell[s][sc]["outcome"]["ttfa_ms"] for s in seeds] for sc in scen])
    C = np.array([[cell[s][sc]["outcome"]["retrievals"] for s in seeds] for sc in scen])
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(scen), size=(iters, len(scen)))
    acc = A[idx].mean(axis=(1, 2))
    comp = C[idx].mean(axis=(1, 2))
    p95 = np.array([np.percentile(T[i].ravel(), 95) for i in idx])
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))  # noqa: E731
    return {"acc": q(acc), "p95": q(p95), "compute": q(comp)}


def paired_acc_diff(cell_a, cell_b, rng_seed=BOOT_SEED, iters=BOOT_ITER):
    """Mean accuracy difference a-b and its scenario-cluster bootstrap CI."""
    scen = sorted(next(iter(cell_a.values())).keys())
    seeds = sorted(cell_a.keys())
    D = np.array([[
        (1.0 if cell_a[s][sc]["outcome"]["answer_correct"] else 0.0)
        - (1.0 if cell_b[s][sc]["outcome"]["answer_correct"] else 0.0)
        for s in seeds] for sc in scen])
    per_sc = D.mean(axis=1)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(scen), size=(iters, len(scen)))
    boots = per_sc[idx].mean(axis=1)
    return float(per_sc.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def auc(scores, labels):
    """Mann-Whitney AUC with ties counted as 1/2."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    return tot / (len(pos) * len(neg))


def ece(probs, labels, n_bins=5):
    """Expected calibration error, equal-width bins on [0, 1] (Guo et al. 2017)."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    bins = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        m = (probs >= lo) & ((probs < hi) if k < n_bins - 1 else (probs <= hi))
        if m.sum() == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0})
            continue
        conf, freq = probs[m].mean(), labels[m].mean()
        total += m.sum() / len(probs) * abs(conf - freq)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "mean_pred": float(conf), "need_rate": float(freq)})
    return float(total), bins


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# E3 computations
# ---------------------------------------------------------------------------
def e3_numbers(table, grid, lam_c):
    out = {}
    n_prices = len(grid)

    # Pooled operating points + CIs for every priced point and every unpriced baseline.
    pooled = {}
    for b in ("ours", "b9"):
        pooled[b] = []
        for i in range(n_prices):
            cell = table[b][i]
            a, p, c = op_point(cell)
            ci = cluster_boot(cell)
            pooled[b].append({"i": i, "lambda_l": grid[i], "acc": a, "p95_ms": p, "compute": c, "ci": ci})
    for b in ("b0", "b4", "b6", "b8"):
        cell = table[b][None]
        a, p, c = op_point(cell)
        pooled[b] = {"acc": a, "p95_ms": p, "compute": c, "ci": cluster_boot(cell)}
    out["pooled_operating_points"] = pooled

    # Per-seed operating points.
    per_seed = {}
    for b in ("ours", "b9"):
        per_seed[b] = {s: [op_point_seed(table[b][i][s]) for i in range(n_prices)] for s in SEEDS}
    for b in ("b0", "b4", "b6", "b8"):
        per_seed[b] = {s: op_point_seed(table[b][None][s]) for s in SEEDS}
    out["per_seed"] = per_seed

    # Low-price coincidence of Ours and B9 (lambda index 0).
    low = {s: {"ours": per_seed["ours"][s][0], "b9": per_seed["b9"][s][0]} for s in SEEDS}
    out["lowest_price"] = {"lambda_l": grid[0], "per_seed": low}

    # Ranges of unpriced baselines across seeds.
    def rng_of(b, k):
        vals = [per_seed[b][s][k] for s in SEEDS]
        return [min(vals), max(vals)]
    out["unpriced_ranges"] = {b: {"acc": rng_of(b, 0), "p95_ms": rng_of(b, 1), "compute": rng_of(b, 2)}
                              for b in ("b0", "b4", "b6", "b8")}

    # B4 identical to B0, record by record.
    same = all(
        table["b4"][None][s][sc]["outcome"] == table["b0"][None][s][sc]["outcome"]
        and table["b4"][None][s][sc]["decision"]["retrieve"] == table["b0"][None][s][sc]["decision"]["retrieve"]
        for s in SEEDS for sc in table["b0"][None][s])
    b4_retrieves = sum(r["decision"]["retrieve"] for s in SEEDS for r in table["b4"][None][s].values())
    out["b4_identical_to_b0"] = {"identical_outcomes_and_decisions": same, "b4_retrievals_total": int(b4_retrieves)}

    # Decisions are seed-invariant?
    inv = True
    for b in table:
        for pi in table[b]:
            for sc in table[b][pi][SEEDS[0]]:
                d = {(table[b][pi][s][sc]["decision"]["retrieve"], table[b][pi][s][sc]["decision"]["timing"]) for s in SEEDS}
                if len(d) > 1:
                    inv = False
    out["decisions_seed_invariant"] = inv

    # B9 at the top price.
    top = n_prices - 1
    out["b9_top_price"] = {"lambda_l": grid[top],
                           "retrievals_of_50": int(sum(r["decision"]["retrieve"] for r in table["b9"][top][SEEDS[0]].values()))}
    out["ours_top_price_retrievals_of_50"] = int(sum(r["decision"]["retrieve"] for r in table["ours"][top][SEEDS[0]].values()))

    # Price-shift equivalence: Ours(i) vs B9(i-1).
    shift = []
    for i in range(1, n_prices):
        dec_agree = sum(
            table["ours"][i][s][sc]["decision"]["retrieve"] == table["b9"][i - 1][s][sc]["decision"]["retrieve"]
            for s in SEEDS for sc in table["ours"][i][s])
        both = [(s, sc) for s in SEEDS for sc in table["ours"][i][s]
                if table["ours"][i][s][sc]["decision"]["retrieve"] and table["b9"][i - 1][s][sc]["decision"]["retrieve"]]
        tim_agree = sum(table["ours"][i][s][sc]["decision"]["timing"] == table["b9"][i - 1][s][sc]["decision"]["timing"] for s, sc in both)
        per_seed_eq = []
        for s in SEEDS:
            o, b = per_seed["ours"][s][i], per_seed["b9"][s][i - 1]
            per_seed_eq.append({"seed": s, "acc_equal": o[0] == b[0], "compute_equal": o[2] == b[2],
                                "p95_diff_ms": o[1] - b[1]})
        shift.append({"i": i, "decision_agree": int(dec_agree), "n": 250,
                      "timing_agree": int(tim_agree), "n_both_retrieve": len(both), "per_seed": per_seed_eq})
    out["price_shift"] = shift
    out["price_shift_summary"] = {
        "decision_agree_total": sum(x["decision_agree"] for x in shift),
        "decision_total": 250 * len(shift),
        "acc_equal_seed_points": sum(e["acc_equal"] for x in shift for e in x["per_seed"]),
        "compute_equal_seed_points": sum(e["compute_equal"] for x in shift for e in x["per_seed"]),
        "p95_equal_seed_points": sum(abs(e["p95_diff_ms"]) < 1e-9 for x in shift for e in x["per_seed"]),
        "seed_points": 5 * len(shift),
        "max_abs_p95_diff_ms": max(abs(e["p95_diff_ms"]) for x in shift for e in x["per_seed"]),
    }

    # Grid recurrence: raw ratios, and ratio of distances to the fixed point.
    cfg = load_e3_config()
    L = float(cfg["retrieval_latency_ms"]) / 1000.0
    o_star = float(cfg["overlap_ms"]["fixed_early_proxy"]) / 1000.0
    w_star = float(cfg["waste_rate"]["fixed_early_proxy"])
    w_se = float(cfg["waste_rate"]["sentence_end"])
    lam_deg = lam_c * (w_star - w_se) / o_star
    raw_ratios = [grid[i] / grid[i - 1] for i in range(1, n_prices)]
    fp_ratios = [(grid[i] - lam_deg) / (grid[i - 1] - lam_deg) for i in range(1, n_prices)]
    out["grid"] = {
        "n": n_prices, "lambda_c": lam_c, "min": grid[0], "max": grid[-1],
        "L_s": L, "o_star_s": o_star, "exposed_s": L - o_star, "L_over_exposed": L / (L - o_star),
        "raw_ratio_min": min(raw_ratios), "raw_ratio_max": max(raw_ratios),
        "lambda_deg": lam_deg, "fixed_point_ratio_min": min(fp_ratios), "fixed_point_ratio_max": max(fp_ratios),
    }

    # Divergence band at each price, from the controllers' own divergence_band().
    voi = {}
    for r in table["ours"][0][SEEDS[0]].values():
        m = re.search(r"voi=([0-9.]+)", r["decision"]["reason"])
        voi[r["scenario_id"]] = float(m.group(1))
    grid_json = json.loads(GRID.read_text(encoding="utf-8"))
    voi_exact = {v["scenario_id"]: float(v["voi"]) for v in grid_json["voi_per_scenario"]}
    assert all(abs(voi_exact[k] - voi[k]) < 6e-4 for k in voi), "logged VoI disagrees with grid artifact"
    band = []
    for i, lam in enumerate(grid):
        db = divergence_band(Prices(lambda_l=lam, lambda_c=lam_c), cfg)
        in_band = sorted(k for k, v in voi_exact.items() if db.contains(v))
        disagree = sorted(sc for sc in table["ours"][i][SEEDS[0]]
                          if table["ours"][i][SEEDS[0]][sc]["decision"]["retrieve"] != table["b9"][i][SEEDS[0]][sc]["decision"]["retrieve"])
        d, ci = paired_acc_diff(table["ours"][i], table["b9"][i])
        ours_ret = sum(r["decision"]["retrieve"] for r in table["ours"][i][SEEDS[0]].values())
        b9_ret = sum(r["decision"]["retrieve"] for r in table["b9"][i][SEEDS[0]].values())
        band.append({
            "i": i, "lambda_l": lam, "ours_cut": db.ours_cut, "b9_cut": db.b9_cut,
            "width": db.width, "lambda_l_times_o_star": lam * o_star,
            "n_in_band": len(in_band), "frac_in_band": len(in_band) / len(voi_exact),
            "in_band_ids": in_band, "n_disagree": len(disagree),
            "disagree_equals_band": disagree == in_band,
            "acc_gap_ours_minus_b9": d, "acc_gap_ci": ci,
            "ours_retrieve_frac": ours_ret / 50, "b9_retrieve_frac": b9_ret / 50,
        })
    out["band"] = band
    out["band_summary"] = {
        "prices_with_nonempty_band": sum(1 for b in band if b["n_in_band"] > 0),
        "scenario_prices_in_band": sum(b["n_in_band"] for b in band),
        "decisions_total": 50 * n_prices,
        "disagree_equals_band_all_prices": all(b["disagree_equals_band"] for b in band),
        "prices_where_gap_ci_excludes_0": [b["i"] for b in band if b["acc_gap_ci"][0] > 0 or b["acc_gap_ci"][1] < 0],
    }
    return out


def retrieval_effect():
    """Per-scenario retrieve vs skip accuracy, split by the retrieval_needed label."""
    scen = {s["id"]: s for s in json.loads(SCENARIOS.read_text(encoding="utf-8"))}
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(E3_ANSWERS.glob("s*_seed*_*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        need = scen[a["scenario_id"]]["retrieval_needed"]
        acc[need][a["path"]].append(1.0 if a["answer_correct"] else 0.0)
    res = {}
    for need in (True, False):
        r, s = float(np.mean(acc[need]["retrieve"])), float(np.mean(acc[need]["skip"]))
        res["needed" if need else "not_needed"] = {
            "n_answers_per_path": len(acc[need]["retrieve"]),
            "skip_acc": s, "retrieve_acc": r, "retrieve_minus_skip": r - s}
    # Scenario-cluster bootstrap on the two effects.
    per = defaultdict(dict)
    for f in sorted(E3_ANSWERS.glob("s*_seed*_*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        per[a["scenario_id"]].setdefault(a["path"], []).append(1.0 if a["answer_correct"] else 0.0)
    for need, key in ((True, "needed"), (False, "not_needed")):
        ids = sorted(k for k in per if scen[k]["retrieval_needed"] == need)
        eff = np.array([np.mean(per[k]["retrieve"]) - np.mean(per[k]["skip"]) for k in ids])
        rng = np.random.default_rng(BOOT_SEED)
        boots = eff[rng.integers(0, len(ids), size=(BOOT_ITER, len(ids)))].mean(axis=1)
        res[key]["n_scenarios"] = len(ids)
        res[key]["effect_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    res["n_answers_total"] = sum(len(v) for d in acc.values() for v in d.values())
    return res


def signal_quality():
    """E3 discrimination (mean vs max entropy), calibration (ECE), overlap with E1."""
    sig = json.loads(SIGNALS.read_text(encoding="utf-8"))["signals"]
    labels = [bool(s["retrieval_needed"]) for s in sig]
    mean_h = [s["model_uncertainty"]["mean_entropy_nats"] for s in sig]
    max_h = [s["model_uncertainty"]["max_entropy_nats"] for s in sig]
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    cal = cal.get("calibration", cal)
    slope, intercept = float(cal["slope"]), float(cal["intercept"])
    p_raw = [sigmoid(slope * x + intercept) for x in mean_h]
    # The operational belief is estimate_voi()'s output: P floored at voi.floor
    # (0.05). ECE/Brier are reported on that belief, and on the raw P for reference.
    floor = float(load_e3_config()["voi"]["floor"])
    p = [max(floor, x) for x in p_raw]
    e5, bins5 = ece(p, labels, 5)
    e10, _ = ece(p, labels, 10)
    e5_raw, _ = ece(p_raw, labels, 5)
    brier = float(np.mean([(pi - yi) ** 2 for pi, yi in zip(p, labels)]))

    e1_q = set()
    for line in E1_DATASET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e1_q.add(json.loads(line)["question"].strip().lower())
    e3_q = {s["question"].strip().lower() for s in sig}
    shared = sorted(e3_q & e1_q)

    e1 = json.loads(E1_RESULT.read_text(encoding="utf-8"))
    return {
        "n_scenarios": len(sig), "n_needed": sum(labels),
        "e3_auc_mean_entropy": auc(mean_h, labels),
        "e3_auc_max_entropy": auc(max_h, labels),
        "belief": f"max({floor}, sigmoid(slope*mean_entropy_nats + intercept)) = estimate_voi()",
        "e3_ece_5bins": e5, "e3_ece_10bins": e10, "e3_ece_5bins_unfloored": e5_raw,
        "e3_brier": brier, "e3_reliability_5bins": bins5,
        "calibration": {"slope": slope, "intercept": intercept, "fit_on": "E1 180 samples (in-sample)",
                        "auc_in_sample": cal.get("auc")},
        "e3_unique_questions": len(e3_q), "e1_unique_questions": len(e1_q),
        "e3_e1_shared_questions": len(shared), "shared": shared,
        "e1_auc": {k: {"mean": v["mean_auc"], "min": v["min_auc"], "max": v["max_auc"]}
                   for k, v in e1["aggregate_auc"].items() if k in ("mean_entropy_nats", "max_entropy_nats")},
        "e1_n": e1["dataset"]["examples"],
    }


# ---------------------------------------------------------------------------
# E2 computations (raw logs)
# ---------------------------------------------------------------------------
def e2_numbers():
    from harness.e2_bootstrap import (
        collapse_to_scenario_medians, load_all_ttfa, paired_bootstrap, p95_with_cluster_bootstrap)
    raw, _, _ = load_all_ttfa(E2_LOGS)
    med, inter = collapse_to_scenario_medians(raw)
    rng = np.random.default_rng(42)  # harness/e2_bootstrap.py default seed
    out = {"policies": {}, "paired_vs_sentence_end": {}}
    for pol in ("sentence_end", "immediate", "fixed_early_proxy"):
        allv = np.concatenate([np.asarray(raw[pol][s]) for s in inter])
        out["policies"][pol] = {"n_runs": int(allv.size), "p50_ms": float(np.percentile(allv, 50)),
                                "p95_ms": float(np.percentile(allv, 95))}
    for pol in ("immediate", "fixed_early_proxy"):
        diffs = np.array([med[pol][s] - med["sentence_end"][s] for s in inter])
        m, lo, hi, _ = paired_bootstrap(diffs, rng, 10000)
        out["paired_vs_sentence_end"][pol] = {
            "statistic": "mean over scenarios of (median-over-seeds TTFA difference)",
            "mean_ms": m, "ci95_ms": [lo, hi], "median_of_paired_diffs_ms": float(np.median(diffs))}
    for pol in ("sentence_end", "immediate", "fixed_early_proxy"):
        p, lo, hi = p95_with_cluster_bootstrap(raw[pol], inter, rng, 10000)
        out["policies"][pol]["p95_cluster_ci_ms"] = [lo, hi]
    out["bootstrap_note"] = ("rng = numpy default_rng(42) consumed in the order of "
                             "harness/e2_bootstrap.py main(); paired CIs reproduce that script's output")

    # Stage timeline relative to user_end (median over all runs per policy).
    stages = ("retrieval_launched", "retrieval_result", "llm_first_token", "llm_final", "tts_first_audio")
    tl = defaultdict(lambda: defaultdict(list))
    ntok = defaultdict(list)
    runs = defaultdict(list)
    for f in sorted(E2_LOGS.glob("conv_*.jsonl")):
        pol = next((p for p in ("fixed_early_proxy", "sentence_end", "immediate") if f.name.startswith(f"conv_{p}_s")), None)
        if pol is None:
            continue
        ev = defaultdict(list)
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                ev[e["event_type"]].append(e)
        ue = ev["user_end"][-1]["t_ms"]
        for st in stages:
            if ev[st]:
                tl[pol][st].append(ev[st][-1]["t_ms"] - ue)
        ntok[pol].append(ev["llm_final"][-1]["payload"].get("n_tokens"))
        runs[pol].append((
            ev["tts_first_audio"][-1]["t_ms"] - ue,
            ev["llm_final"][-1]["t_ms"] - ev["llm_first_token"][-1]["t_ms"],
            ev["llm_first_token"][-1]["t_ms"] - ue,
            float(ev["llm_final"][-1]["payload"].get("n_tokens") or float("nan")),
            float(ev["retrieval_launched"][-1]["payload"].get("launch_offset_ms_vs_user_end", float("nan"))),
            ev["retrieval_result"][-1]["t_ms"] - ue,
        ))
    # Tail composition: what distinguishes the slowest 5% of runs per policy.
    tail = {}
    for pol, v in runs.items():
        a = np.array(v, dtype=float)
        thr = np.percentile(a[:, 0], 95)
        top = a[a[:, 0] >= thr]
        tail[pol] = {
            "p95_ms": float(thr), "n_tail_runs": int(len(top)),
            "median_generation_ms_all": float(np.median(a[:, 1])),
            "median_generation_ms_tail": float(np.median(top[:, 1])),
            "median_user_end_to_first_token_ms_all": float(np.median(a[:, 2])),
            "median_user_end_to_first_token_ms_tail": float(np.median(top[:, 2])),
            "median_answer_tokens_all": float(np.nanmedian(a[:, 3])),
            "median_answer_tokens_tail": float(np.nanmedian(top[:, 3])),
            "median_user_end_to_retrieval_result_ms_all": float(np.median(a[:, 5])),
            "median_user_end_to_retrieval_result_ms_tail": float(np.median(top[:, 5])),
            "median_retrieval_result_to_first_token_ms_all": float(np.median(a[:, 2] - a[:, 5])),
            "median_retrieval_result_to_first_token_ms_tail": float(np.median(top[:, 2] - top[:, 5])),
            "pearson_ttfa_generation_ms": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
            "runs_launched_over_1000ms_after_user_end": int(np.sum(a[:, 4] > 1000)),
        }
    out["tail_analysis"] = tail
    out["stage_medians_ms_vs_user_end"] = {
        pol: {st: float(np.median(v)) for st, v in d.items()} for pol, d in tl.items()}
    out["median_answer_tokens"] = {pol: float(np.median([x for x in v if x is not None])) for pol, v in ntok.items()}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
PNG_PREVIEW_DIR = None


def _save(fig, path):
    """Vector PDF for the paper; optional PNG preview for visual checks."""
    fig.savefig(path)
    if PNG_PREVIEW_DIR is not None:
        fig.savefig(Path(PNG_PREVIEW_DIR) / (Path(path).stem + ".png"), dpi=200)
    plt.close(fig)

def _ribbon(ax, xs, lo, hi, color, hatch=None):
    """Accuracy CI ribbon following the frontier in price order (polygon, so a
    non-monotone x path is still drawn correctly). The second ribbon is
    hatched rather than filled so overlap stays readable in greyscale."""
    px = list(xs) + list(xs)[::-1]
    py = list(hi) + list(lo)[::-1]
    if hatch:
        ax.fill(px, py, facecolor="none", edgecolor=color, alpha=0.45, hatch=hatch, lw=0, zorder=1)
    else:
        ax.fill(px, py, color=color, alpha=0.16, lw=0, zorder=1)


def fig_frontier(pooled, xkey, xlabel, xscale, path, xci_key):
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for b in ("b9", "ours"):
        pts = pooled[b]
        xs = [p[xkey] * xscale for p in pts]
        ys = [p["acc"] for p in pts]
        _ribbon(ax, xs, [p["ci"]["acc"][0] for p in pts], [p["ci"]["acc"][1] for p in pts], STYLE[b]["color"],
                hatch="////" if b == "b9" else None)
        ax.plot(xs, ys, color=STYLE[b]["color"], ls=STYLE[b]["ls"], marker=STYLE[b]["marker"],
                markersize=3.2 if b == "ours" else 4.2, mfc="white" if b == "b9" else STYLE[b]["color"],
                mew=0.9, label=STYLE[b]["label"], zorder=3 if b == "ours" else 2)
    for b in ("b0", "b6", "b8"):
        p = pooled[b]
        x = p[xkey] * xscale
        xe = [[x - p["ci"][xci_key][0] * xscale], [p["ci"][xci_key][1] * xscale - x]]
        ye = [[p["acc"] - p["ci"]["acc"][0]], [p["ci"]["acc"][1] - p["acc"]]]
        ax.errorbar(x, p["acc"], xerr=xe, yerr=ye, fmt=STYLE[b]["marker"], color=STYLE[b]["color"],
                    ms=5.5, mfc="white", mew=1.1, elinewidth=0.7, capsize=1.5, zorder=4, label=STYLE[b]["label"])
    top = pooled["b9"][-1]
    ax.annotate("B9 at top price:\nno retrievals",(top[xkey] * xscale, top["acc"]),
                xytext=(10, -26), textcoords="offset points", fontsize=7, color=C_INK2,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_INK2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Answer accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=2.2, borderaxespad=0.2)
    _save(fig, path)


def fig_per_seed(per_seed, path):
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.9), sharey=True, sharex=True)
    for ax, s in zip(axes, SEEDS):
        for b in ("b9", "ours"):
            pts = per_seed[b][s]
            ax.plot([p[1] / 1000 for p in pts], [p[0] for p in pts], color=STYLE[b]["color"],
                    ls=STYLE[b]["ls"], marker=STYLE[b]["marker"], ms=2.5, lw=1.0,
                    mfc="white" if b == "b9" else STYLE[b]["color"], label=STYLE[b]["label"])
        for b in ("b0", "b6", "b8"):
            a, p95, _ = per_seed[b][s]
            ax.plot(p95 / 1000, a, STYLE[b]["marker"], color=STYLE[b]["color"], ms=4.5, mfc="white",
                    mew=1.0, label=STYLE[b]["label"])
        ax.text(0.04, 0.95, f"seed {s}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("p95 TTFA (s)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.12))
    _save(fig, path)


def fig_price_shift(e3, grid, path):
    """(a) fraction of turns retrieved vs lambda_l; B9 re-plotted one grid step
    to the right lands exactly on Ours. (b) Ours(i) - B9(i-1) per seed."""
    band = e3["band"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.12})
    ours = [b["ours_retrieve_frac"] for b in band]
    b9 = [b["b9_retrieve_frac"] for b in band]
    ax1.step(grid, ours, where="post", color=C_OURS, lw=1.6, label=r"Ours$(\lambda_i)$")
    ax1.step(grid, b9, where="post", color=C_B9, lw=1.4, ls="--", label=r"B9$(\lambda_i)$")
    ax1.plot(grid[1:], b9[:-1], ls="none", marker="s", ms=4.2, mfc="white", mec=C_B9, mew=0.9,
             label=r"B9$(\lambda_{i-1})$ plotted at $\lambda_i$")
    ax1.plot(grid, ours, ls="none", marker="o", ms=2.2, color=C_OURS)
    ax1.set_ylabel("Turns retrieved\n(fraction of 50)")
    ax1.set_ylim(-0.03, 1.03)
    ax1.legend(loc="lower left", frameon=False)
    xs, p95_d = [], []
    for x in e3["price_shift"]:
        for e in x["per_seed"]:
            xs.append(grid[x["i"]])
            p95_d.append(e["p95_diff_ms"])
    accd = [e3["per_seed"]["ours"][s][i][0] - e3["per_seed"]["b9"][s][i - 1][0]
            for s in SEEDS for i in range(1, len(grid))]
    compd = [e3["per_seed"]["ours"][s][i][2] - e3["per_seed"]["b9"][s][i - 1][2]
             for s in SEEDS for i in range(1, len(grid))]
    n_same = sum(1 for a_, c_ in zip(accd, compd) if a_ == 0 and c_ == 0)
    ax2.axhline(0, color=C_INK2, lw=0.6)
    ax2.plot(xs, p95_d, "^", ms=3.8, color=C_B9, mfc="white", mew=0.9,
             label="p95 TTFA difference (ms), one marker per seed")
    ax2.text(0.02, 0.06, f"accuracy and retrievals/turn identical in {n_same}/{len(accd)}\nseed-price comparisons",
             transform=ax2.transAxes, fontsize=7, va="bottom", color=C_INK)
    ax2.set_ylabel("Ours$(\\lambda_i)$ $-$\nB9$(\\lambda_{i-1})$, ms")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Latency price $\lambda_l$ (accuracy per second, log scale)")
    lim = max(10.0, max(abs(v) for v in p95_d) * 1.3)
    ax2.set_ylim(-lim, lim * 0.6)
    ax2.legend(loc="upper left", frameon=False, fontsize=7)
    _save(fig, path)


def fig_band(e3, grid, path):
    band = e3["band"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12})
    frac = [b["frac_in_band"] for b in band]
    dis = [b["n_disagree"] / 50 for b in band]
    ax1.vlines(grid, 0, frac, color=C_OURS, lw=2.2, alpha=0.55, label=r"predicted: $P(b\in\mathcal{B}_\lambda)$")
    ax1.plot(grid, dis, "o", ms=3.4, mfc="white", mec=C_INK, mew=0.9, label="observed: Ours $\\neq$ B9 decisions")
    ax1.set_ylabel("Fraction of turns")
    ax1.set_ylim(0, max(frac) * 1.15 + 0.02)
    ax1.legend(loc="upper left", frameon=False)
    gap = [b["acc_gap_ours_minus_b9"] for b in band]
    lo = [b["acc_gap_ci"][0] for b in band]
    hi = [b["acc_gap_ci"][1] for b in band]
    ax2.fill_between(grid, lo, hi, color=C_OURS, alpha=0.16, lw=0, step=None)
    ax2.plot(grid, gap, "o-", ms=2.8, lw=1.0, color=C_OURS, label="Ours $-$ B9 accuracy (95% CI)")
    ax2.axhline(0, color=C_INK2, lw=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Latency price $\lambda_l$ (accuracy per second, log scale)")
    ax2.set_ylabel("Accuracy gap\n(same price)")
    ax2.legend(loc="upper left", frameon=False)
    _save(fig, path)


def fig_latency_breakdown(e2, path):
    order = ("sentence_end", "immediate", "fixed_early_proxy")
    names = {"sentence_end": "sentence_end", "immediate": "immediate", "fixed_early_proxy": "fixed_early_proxy"}
    st = e2["stage_medians_ms_vs_user_end"]
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    segs = (("retrieval_launched", "retrieval_result", "retrieval", "#2a78d6"),
            ("retrieval_result", "llm_first_token", "prompt + first token", "#9dbfe9"),
            ("llm_first_token", "llm_final", "full answer generation", "#eb6834"),
            ("llm_final", "tts_first_audio", "TTS start", C_INK))
    for y, pol in enumerate(order):
        d = st[pol]
        for a, b, lab, col in segs:
            ax.barh(y, (d[b] - d[a]) / 1000, left=d[a] / 1000, height=0.55, color=col,
                    edgecolor="white", lw=0.8, label=lab if y == 0 else None)
        ax.text(d["tts_first_audio"] / 1000 + 0.06, y, f"{d['tts_first_audio']/1000:.2f} s",
                va="center", fontsize=7.5)
    ax.axvline(0, color=C_INK, lw=0.8)
    ax.text(0.03, 2.42, "user end", fontsize=7, va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[p] for p in order], fontsize=7.5)
    ax.set_ylim(-0.5, 2.85)
    ax.set_xlabel("Median time relative to user end of speech (s)")
    ax.set_xlim(-0.6, 4.9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False, fontsize=7)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fig-dir", default=str(REPO / "paper_v2" / "figures"))
    ap.add_argument("--numbers", default=str(REPO / "analysis" / "paper_v1_numbers.json"))
    ap.add_argument("--skip-e2", action="store_true", help="skip the E2 log parse (slow on OneDrive)")
    ap.add_argument("--png-preview", default=None, help="also write PNG previews to this directory")
    args = ap.parse_args(argv)
    global PNG_PREVIEW_DIR
    if args.png_preview:
        Path(args.png_preview).mkdir(parents=True, exist_ok=True)
        PNG_PREVIEW_DIR = args.png_preview
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    grid, lam_c, _ = load_grid()
    recs = load_e3_records()
    table = organise(recs, grid)
    e3 = e3_numbers(table, grid, lam_c)
    numbers = {
        "_comment": "GENERATED by `python -m analysis.paper_figures_v1`. Do not hand-edit. "
                    "Real v1 data only; e3_sample_result/ (synthetic) is never read.",
        "e3": e3,
        "e3_retrieval_effect": retrieval_effect(),
        "e3_signal_quality": signal_quality(),
    }
    fig_frontier(e3["pooled_operating_points"], "p95_ms", "p95 time to first audio (s)", 1e-3,
                 fig_dir / "fig_e3_acc_vs_p95.pdf", "p95")
    fig_frontier(e3["pooled_operating_points"], "compute", "Retrievals per turn", 1.0,
                 fig_dir / "fig_e3_acc_vs_compute.pdf", "compute")
    fig_per_seed(e3["per_seed"], fig_dir / "fig_e3_per_seed.pdf")
    fig_price_shift(e3, grid, fig_dir / "fig_e3_price_shift.pdf")
    fig_band(e3, grid, fig_dir / "fig_e3_band_occupancy.pdf")
    if not args.skip_e2:
        e2 = e2_numbers()
        numbers["e2"] = e2
        fig_latency_breakdown(e2, fig_dir / "fig_e2_latency_breakdown.pdf")

    Path(args.numbers).write_text(json.dumps(_jsonable(numbers), indent=1), encoding="utf-8")
    s = numbers
    print("E3 lowest price (seed 11) Ours/B9:", s["e3"]["lowest_price"]["per_seed"][11])
    print("price-shift summary:", s["e3"]["price_shift_summary"])
    print("band summary:", s["e3"]["band_summary"])
    print("B9 top price:", s["e3"]["b9_top_price"])
    print("unpriced ranges:", json.dumps(s["e3"]["unpriced_ranges"]))
    print("B4==B0:", s["e3"]["b4_identical_to_b0"], "seed-invariant decisions:", s["e3"]["decisions_seed_invariant"])
    print("retrieval effect:", json.dumps(s["e3_retrieval_effect"]))
    sq = dict(s["e3_signal_quality"])
    sq.pop("e3_reliability_5bins")
    print("signal quality:", json.dumps(sq))
    print("grid:", s["e3"]["grid"])
    if "e2" in s:
        print("E2:", json.dumps({k: v for k, v in s["e2"].items() if k != "bootstrap_note"}, indent=1))
    print(f"wrote figures to {fig_dir} and numbers to {args.numbers}")


if __name__ == "__main__":
    main()
