"""
v2_power_dev.py -- power of the PREREG_V2 boundary test (P1, P2, P3 and their
intersection-union conjunction) from DEV-MEASURED inputs.

    PYTHONUTF8=1 python -m analysis.v2_power_dev

Writes docs/research_v2/v2_power_dev.json. It replaces the parametric inputs
of analysis/v2_power.py (synthetic belief mixtures, two decision epochs, v1
quality effects, placeholder latency profiles) with the dev Phase B itself:
the per-item, seed-averaged paired utility differences that the
pre-registered test statistics average. Those differences already embody
every dev-measured input -- the calibrated belief distribution, the fitted
belief transitions, g+/g-, the composite latency profiles with the
non-hideable TTFA increment, the measured ASR timing, and the tuned rivals --
so nothing is assumed beyond "test items are exchangeable with dev items".

Per (latency profile, belief-quantile price), on the PRIMARY condition of
each dataset (ASR):

  columns   P1:  D_C                    (Ours-v2[COMMIT] - B9+)
            P2:  D_C - D_S              (interaction, paired items)
            P3:  m3 - D_S and D_S + m3  (TOST; m3 = 1/2 max(Delta_C,dev, 0))
            with D_S = Ours-v2[SPEC] - D*[SPEC], D* the dev-best of
            {B9+, SPEC, SPEC-theta} (as boundary_contrasts chooses it).
  test N    d1_open 3,734 + d2_support 498 = 4,232 pooled items.
  pooled    test mean = sum_d (n_d/N) mean_d; its covariance
            sum_d (n_d/N^2) S_d, S_d the per-item covariance of dataset d
            with CLUSTER structure (d2_support pair_id) kept: S_d =
            sum_c (T_c - m_c mu)(T_c - m_c mu)^T / n_dev,d (ratio estimator).
  power     each column's mean must be > 0 one-sided at alpha = 0.025;
            the conjunction rejects iff all four do (normal approximation,
            50,000 common normal draws), as analysis/v2_power.py.
  check     at the cell with the highest conjunction power, test sets of the
            planned size are drawn by resampling dev clusters and tested with
            the Wald rule (200 sets) to check the normal approximation.

The dev effects are plug-in estimates, so the power is a dev-predicted power
(PREREG_V2 §4.3), not a guarantee: each column mean's dev standard error is
reported next to it.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1_v2 import common as e1common  # noqa: E402
from simulator.e3v2_decide import boundary_contrasts  # noqa: E402

PHASE_A = _REPO_ROOT / "simulator" / "e3v2_phase_a"
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "v2_power_dev.json"
N_TEST = {"d1_open": 3734, "d2_support": 498}
ALPHA, POWER = 0.025, 0.80
Z_ALPHA = 1.959963984540054
Z_POWER = 0.8416212335729143
RNG_SEED = 20261016
_STD = np.random.default_rng(RNG_SEED).standard_normal((50_000, 4))
PROFILES = ("rewrite_faiss", "rewrite_faiss_rerank", "rewrite_remote")
COLS = ("P1: D_C", "P2: D_C - D_S", "P3 upper: m3 - D_S", "P3 lower: D_S + m3")


def slug(cond: str) -> str:
    from simulator.e3v2_dev_analysis import slug as _slug
    return _slug(cond)


def load_cells(ds: str, cond: str, prof: str):
    """{quantile: {"commit": {item: D}, "spec": {...}, "d_star": ..., "price": ...}}"""
    res = e1common.load_json(PHASE_A / ds / "analysis" / "dev_analysis.json")
    grid = res["conditions"][cond]["price_grid"][prof]
    q_of = {f"{r['lambda_l']:g}|{r['lambda_c']:g}" if "lambda_c" in r else f"{r['lambda_l']:g}|0.05": r["quantile"]
            for r in grid if r["feasible"]}
    recs = []
    for f in sorted(glob.glob(str(PHASE_A / ds / "analysis" / "phase_b" / slug(cond) / prof / "e3_v2_*.json"))):
        with open(f, encoding="utf-8") as fh:
            recs += json.load(fh)
    out = {}
    for row in boundary_contrasts(recs, "dev"):
        q = q_of.get(row["price"])
        if q is None:
            continue
        out[q] = {"commit": row["per_item"]["delta_commit"], "spec": row["per_item"]["delta_speculative"],
                  "d_star": row["d_star"], "price": row["price"]}
    return out


def cluster_cov(Z: np.ndarray, clusters: list) -> tuple:
    """(mu, S): S such that Var(mean over n items) ~ S / n, clusters kept."""
    mu = Z.mean(axis=0)
    idx = defaultdict(list)
    for k, c in enumerate(clusters):
        idx[c].append(k)
    T = np.array([Z[v].sum(axis=0) - len(v) * mu for v in idx.values()])
    return mu, (T.T @ T) / Z.shape[0]


def conj_power(mu: np.ndarray, cov: np.ndarray) -> tuple:
    L = np.linalg.cholesky(cov + 1e-14 * np.eye(cov.shape[0]))
    sims = mu + _STD[:, :mu.size] @ L.T
    sd = np.sqrt(np.diag(cov))
    rej = sims - Z_ALPHA * sd > 0
    return float(rej.all(axis=1).mean()), [float(x) for x in rej.mean(axis=0)]


def n_for_80(mu_by_d: dict, S_by_d: dict, share: dict) -> float:
    """Pooled test N (same d1:d2 ratio) for 80% conjunction power."""
    mu = sum(share[d] * mu_by_d[d] for d in share)
    if np.any(mu <= 0):
        return math.inf

    def p(N):
        cov = sum(share[d] ** 2 * S_by_d[d] / (share[d] * N) for d in share)
        return conj_power(mu, cov)[0]

    lo, hi = 10, 50_000_000
    if p(hi) < POWER:
        return math.inf
    while hi / lo > 1.005:
        mid = int(math.sqrt(lo * hi))
        lo, hi = (lo, mid) if p(mid) >= POWER else (mid, hi)
    return hi


def main() -> int:
    items = {}
    primary = {}
    for ds in N_TEST:
        res = e1common.load_json(PHASE_A / ds / "analysis" / "dev_analysis.json")
        primary[ds] = res["primary_condition"]
        for it in e1common.load_items(e1common.resolve_dataset_paths(ds)["items"]):
            items[it["id"]] = it.get("pair_id") or it["id"]
    N = sum(N_TEST.values())
    share = {d: n / N for d, n in N_TEST.items()}
    rows_out, best = [], None
    for prof in PROFILES:
        cells = {ds: load_cells(ds, primary[ds], prof) for ds in N_TEST}
        for q in sorted(set.intersection(*[set(c) for c in cells.values()])):
            per_d, mu_d, S_d, Zs = {}, {}, {}, {}
            dc_pool = sum(share[d] * float(np.mean(list(cells[d][q]["commit"].values()))) for d in N_TEST)
            m3 = 0.5 * max(dc_pool, 0.0)
            for d in N_TEST:
                c = cells[d][q]
                common = sorted(set(c["commit"]) & set(c["spec"]))
                DC = np.array([c["commit"][i] for i in common])
                DS = np.array([c["spec"][i] for i in common])
                Z = np.column_stack([DC, DC - DS, m3 - DS, DS + m3])
                mu, S = cluster_cov(Z, [items.get(i, i) for i in common])
                mu_d[d], S_d[d], Zs[d] = mu, S, (Z, [items.get(i, i) for i in common])
                se = np.sqrt(np.diag(S) / len(common))
                per_d[d] = {"n_dev_items": len(common), "price": c["price"], "d_star": c["d_star"],
                            "delta_commit": float(DC.mean()), "delta_spec": float(DS.mean()),
                            "interaction": float((DC - DS).mean()),
                            "column_means": [float(x) for x in mu], "column_dev_se": [float(x) for x in se],
                            "column_sd_per_item_clustered": [float(x) for x in np.sqrt(np.diag(S))]}
            mu = sum(share[d] * mu_d[d] for d in N_TEST)
            cov = sum(N_TEST[d] * S_d[d] for d in N_TEST) / N ** 2
            p_conj, p_ind = conj_power(mu, cov)
            # each dataset alone at its own test size
            alone = {}
            for d in N_TEST:
                pc, pi = conj_power(mu_d[d], S_d[d] / N_TEST[d])
                alone[d] = {"conjunction": pc, "individual": dict(zip(COLS, pi))}
            n80 = n_for_80(mu_d, S_d, share)
            row = {"profile": prof, "belief_quantile": q, "m3_pooled_dev": m3,
                   "pooled_test_means": dict(zip(COLS, [float(x) for x in mu])),
                   "pooled_test_se": dict(zip(COLS, [float(x) for x in np.sqrt(np.diag(cov))])),
                   # smallest true mean each of P1 / P2 detects with 80% power at the
                   # planned N (one-sided alpha 0.025), with the dev covariance
                   "mde_80pct_at_planned_N": {COLS[0]: float((Z_ALPHA + Z_POWER) * math.sqrt(cov[0, 0])),
                                              COLS[1]: float((Z_ALPHA + Z_POWER) * math.sqrt(cov[1, 1]))},
                   "p3_tost_half_width_needed": float((Z_ALPHA + Z_POWER) * math.sqrt(cov[3, 3])),
                   "power_at_planned_N": {"N_test": N, "conjunction": p_conj, "individual": dict(zip(COLS, p_ind))},
                   "N_test_80pct_conjunction_same_ratio": n80,
                   "per_dataset": per_d, "per_dataset_alone_at_own_N": alone}
            rows_out.append(row)
            if best is None or p_conj > best[0]:
                best = (p_conj, row, Zs, mu_d)
    # direct check of the normal approximation at the best cell
    check = None
    if best is not None:
        brng = np.random.default_rng(RNG_SEED + 1)
        p_conj, row, Zs, _ = best
        hits, reps = 0, 200
        for _ in range(reps):
            parts = []
            for d, (Z, cl) in Zs.items():
                byc = defaultdict(list)
                for k, c in enumerate(cl):
                    byc[c].append(k)
                keys = list(byc)
                pick = []
                while len(pick) < N_TEST[d]:
                    pick += byc[keys[brng.integers(len(keys))]]
                parts.append((Z[pick[:N_TEST[d]]], N_TEST[d]))
            m = sum(n / N * x.mean(axis=0) for x, n in parts)
            v = sum((n / N) ** 2 * x.var(axis=0, ddof=1) / n for x, n in parts)
            hits += int(np.all(m - Z_ALPHA * np.sqrt(v) > 0))
        check = {"cell": f"{row['profile']} q={row['belief_quantile']}", "reps": reps,
                 "wald_on_resampled_test_sets": hits / reps, "normal_approx": p_conj,
                 "mc_se": math.sqrt(max(p_conj * (1 - p_conj), 0.01) / reps)}
    doc = {
        "_comment": "Dev-predicted power of the PREREG_V2 boundary test (P1, P2, P3 TOST, conjunction) at the "
                    "planned test N, per composite latency profile and belief-quantile price, on the primary "
                    "(ASR) condition pooled over d1_open and d2_support. Inputs are the dev Phase B per-item "
                    "paired utility differences (analysis/v2_power_dev.py); nothing is selected or frozen here.",
        "generated_by": "python -m analysis.v2_power_dev", "generated_at_utc": e1common.now_utc(),
        "split": "dev", "alpha_one_sided": ALPHA, "target_power": POWER, "n_test": N_TEST, "test_shares": share,
        "primary_conditions": primary, "columns": COLS,
        "m3_rule": "1/2 max(pooled dev Delta[COMMIT], 0), share-weighted over datasets at the test N",
        "cells": rows_out, "normal_approximation_check": check,
    }
    e1common.atomic_write_json(OUT_JSON, doc)
    for r in rows_out:
        pm = r["pooled_test_means"]
        print(f"{r['profile']:22s} q={r['belief_quantile']}: dC={pm[COLS[0]]:+.4f} I={pm[COLS[1]]:+.4f} "
              f"m3={r['m3_pooled_dev']:.4f} dS={pm[COLS[3]] - r['m3_pooled_dev']:+.4f} | power conj "
              f"{r['power_at_planned_N']['conjunction']:.3f} ind "
              + " ".join(f"{v:.2f}" for v in r["power_at_planned_N"]["individual"].values())
              + f" | N80 {r['N_test_80pct_conjunction_same_ratio']} | MDE P1 "
              f"{r['mde_80pct_at_planned_N'][COLS[0]]:.4f} P2 {r['mde_80pct_at_planned_N'][COLS[1]]:.4f}")
    print("check:", check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
