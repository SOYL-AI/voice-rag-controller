"""
v2_power.py -- power analysis for the research-v2 pre-registration addendum
(docs/research_v2/PREREG_V2.md, section "Power analysis").

    PYTHONUTF8=1 python -m analysis.v2_power

Uses only json + numpy. Writes docs/research_v2/v2_power.json.

WHAT IS SIMULATED (and what is assumed -- every assumption is a named
parameter in ASSUMPTIONS below and is echoed into the output JSON):

  Item i has a calibrated belief b_i ~ a synthetic, dataset-specific belief
  distribution, and need_i ~ Bernoulli(b_i) (calibration is assumed; see
  THEORY.md Prop. 2 for what miscalibration does).

  Answer quality is NOT synthetic: each simulated item borrows the five-seed
  (skip, retrieve) correctness vectors of a randomly drawn v1 E3 scenario with
  the same need label (simulator/e3_sweep_cache/answers). This preserves the
  real item-level heterogeneity of the retrieval effect (g+ = 0.72,
  g- = 0.59 in v1) and the real within-item seed noise, which together
  dominate the variance of the paired utility difference.

  Turn timing (channel C1): remaining speech after the earliest meaningful
  partial, R_i = max(0, frac_remaining * duration_i - asr_lag), duration
  log-normal per dataset. Retrieval adds a hideable call latency L_i
  (log-normal per latency profile) and a non-hideable prefill penalty
  (longer prompt). Exposed latency X_i = prefill + max(0, L_i - R_i).

  Controllers at price (lambda_l, lambda_c), corrected objective (THEORY.md S6):
    Ours-v2  retrieve iff b g+ - (1-b) g- - lambda_l E[X | R_hat_i] - lambda_c > 0,
             with R_hat_i = R_i * exp(eps), eps ~ N(0, pred_sigma)  (imperfect
             remaining-speech predictor; the latency distribution is known,
             its realisation is not).
    B9+      retrieve iff b g+ - (1-b) g- - lambda_l E[X] - lambda_c > 0
             (best decomposed controller when b is independent of R; Prop. 3).
    B9       as B9+ but pricing the full un-overlapped latency prefill + E[L].

  Paired per-item difference at one price:
    D_i = (a_Ours,i - a_B,i) * (Q_i - lambda_l X_i - lambda_c),
    Q_i = mean over seeds of (correct_retrieve - correct_skip).
  The primary statistic averages D_i over the pre-specified price set
  (cuts placed at fixed quantiles of the dev belief distribution).

  Power: delta = E[D], sigma = SD(D) from a 400k-item population;
  N_req = ((z_{1-alpha} + z_{power}) sigma / delta)^2 for the one-sided test
  of the item mean; MDE(N) = (z_{1-alpha} + z_{power}) sigma / sqrt(N).
  A direct bootstrap-test power check at the planned N verifies the normal
  approximation for one cell.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "v2_power.json"
SEEDS = (11, 23, 37, 53, 71)
RNG_SEED = 20261015

Z = {0.025: 1.959963984540054, 0.0125: 2.241402727604947, 0.00833: 2.394, 0.05: 1.6448536269514722,
     0.8: 0.8416212335729143, 0.9: 1.2815515655446004}

ASSUMPTIONS = {
    "planned_items_total": {"d1_open": 1200, "d2_support": 720},
    "split": {"dev": 0.30, "test": 0.70},
    "alpha_one_sided_primary": 0.025,
    "power": 0.80,
    "lambda_c": 0.05,
    "belief_cut_quantiles": [0.5, 0.65, 0.8, 0.9],
    "belief_distributions": {
        "d1_open": {"mixture": [[0.5, 2.0, 2.0], [0.25, 5.0, 1.5], [0.25, 1.5, 5.0]],
                    "why": "popularity-stratified: closed-book confidence varies smoothly with popularity, so mass spans the middle"},
        "d2_support": {"mixture": [[0.5, 6.0, 2.0], [0.5, 2.0, 6.0]],
                       "why": "matched org-specific/general pairs: bimodal but overlapping"},
        "v1_like": {"resample_v1": True,
                    "why": "the 50 v1 E3 beliefs, saturated; shows the v1 regime"},
    },
    "speech": {
        "d1_open": {"duration_median_s": 2.8, "duration_sigma_log": 0.30},
        "d2_support": {"duration_median_s": 4.5, "duration_sigma_log": 0.35},
        "frac_remaining_at_first_meaningful_partial": 0.6,
        "asr_lag_s": 0.3,
    },
    "latency_profiles": {
        "faiss": {"call_median_s": 0.672, "call_sigma_log": 0.45},
        "faiss_rerank": {"call_median_s": 1.3, "call_sigma_log": 0.45},
        "remote_api": {"call_median_s": 2.2, "call_sigma_log": 0.50},
    },
    "prefill_penalty_s": 0.9,
    "remaining_speech_predictor_sigma_log": 0.30,
    "quality_effect_scale": 1.0,
    "c2_prefix_noise_logit_sd": 1.5,
    "notes": [
        "c2_prefix_noise_logit_sd sets how much less informative the prefix belief is than the "
        "final one; it is reported as the implied prefix-vs-final AUC and must be replaced by the "
        "E1-v2 dev prefix curve before freezing.",
        "call_median for faiss is the v1 measured median retrieval duration (672 ms).",
        "prefill_penalty is the v1 measured TTFA increment of retrieval at sentence_end (1.566 s) "
        "minus the 672 ms call: the part of retrieval's latency that overlap cannot hide.",
        "faiss_rerank / remote_api medians, speech durations and the predictor error are PLACEHOLDERS "
        "to be replaced by dev measurements before freezing (PREREG_V2 section 5.4).",
    ],
}


def load_v1_outcomes():
    cache = _REPO_ROOT / "simulator" / "e3_sweep_cache" / "answers"
    scen = json.load(open(_REPO_ROOT / "simulator" / "scenarios.json", encoding="utf-8"))
    need = {s["id"]: bool(s["retrieval_needed"]) for s in scen}
    Q = {True: [], False: []}
    for sid, nd in sorted(need.items()):
        r = []
        for sd in SEEDS:
            ys = json.load(open(cache / f"s{sid:03d}_seed{sd}_skip.json", encoding="utf-8"))["answer_correct"]
            yr = json.load(open(cache / f"s{sid:03d}_seed{sd}_retrieve.json", encoding="utf-8"))["answer_correct"]
            r.append(int(bool(yr)) - int(bool(ys)))
        Q[nd].append(np.mean(r))
    grid = json.load(open(_REPO_ROOT / "configs" / "e3_price_grid.json", encoding="utf-8"))
    v1_b = np.array([x["voi"] for x in grid["voi_per_scenario"]])
    return {k: np.array(v) for k, v in Q.items()}, v1_b


def sample_beliefs(rng, spec, n, v1_b):
    if spec.get("resample_v1"):
        return rng.choice(v1_b, size=n, replace=True)
    comp = np.array([m[0] for m in spec["mixture"]])
    k = rng.choice(len(comp), size=n, p=comp / comp.sum())
    a = np.array([m[1] for m in spec["mixture"]])[k]
    bb = np.array([m[2] for m in spec["mixture"]])[k]
    return rng.beta(a, bb)


def expected_exposed(r_hat, med, sig, prefill, rng):
    """E[prefill + max(0, L - r)] for L ~ LogNormal(log med, sig), vectorised via a lookup table."""
    Ls = np.exp(np.log(med) + sig * rng.standard_normal(20000))
    grid = np.linspace(0, 12, 1201)
    tab = np.array([np.maximum(0.0, Ls - g).mean() for g in grid])
    return prefill + np.interp(r_hat, grid, tab), prefill + float(Ls.mean())


def simulate_cell(rng, dataset, profile, Q, v1_b, n=400_000, A=ASSUMPTIONS, pred_sigma=None,
                  q_scale=None):
    pred_sigma = A["remaining_speech_predictor_sigma_log"] if pred_sigma is None else pred_sigma
    q_scale = A["quality_effect_scale"] if q_scale is None else q_scale
    bspec = A["belief_distributions"][dataset if dataset in A["belief_distributions"] else "d1_open"]
    b = sample_beliefs(rng, bspec, n, v1_b)
    need = rng.random(n) < b
    # quality effect: borrow v1 items of the same need label
    q = np.where(need, rng.choice(Q[True], n), rng.choice(Q[False], n)) * q_scale
    gp, gm = float(Q[True].mean() * q_scale), float(-Q[False].mean() * q_scale)

    sp_key = "d2_support" if dataset == "d2_support" else "d1_open"
    sp = A["speech"][sp_key]
    dur = np.exp(np.log(sp["duration_median_s"]) + sp["duration_sigma_log"] * rng.standard_normal(n))
    R = np.maximum(0.0, A["speech"]["frac_remaining_at_first_meaningful_partial"] * dur - A["speech"]["asr_lag_s"])
    prof = A["latency_profiles"][profile]
    Lcall = np.exp(np.log(prof["call_median_s"]) + prof["call_sigma_log"] * rng.standard_normal(n))
    X = A["prefill_penalty_s"] + np.maximum(0.0, Lcall - R)                    # realised exposed latency
    R_hat = R * np.exp(pred_sigma * rng.standard_normal(n))
    X_hat, X_full = expected_exposed(R_hat, prof["call_median_s"], prof["call_sigma_log"],
                                     A["prefill_penalty_s"], rng)
    X_bar = float(X.mean())                                                    # B9+'s scalar (dev estimate)
    lam_c = A["lambda_c"]
    A_b = b * (gp + gm) - gm                                                   # quality part of U(retrieve)

    rows = []
    D_sum = np.zeros(n)
    D9_sum = np.zeros(n)
    for qtl in A["belief_cut_quantiles"]:
        b_cut = float(np.quantile(b, qtl))
        lam = (b_cut * (gp + gm) - gm - lam_c) / X_bar                        # B9+ cut at this belief quantile
        if lam <= 0:
            rows.append({"quantile": qtl, "b_cut": b_cut, "lambda_l": lam, "skipped": "cut below zero-price threshold"})
            continue
        a_O = A_b - lam * X_hat - lam_c > 0
        a_P = A_b - lam * X_bar - lam_c > 0
        a_9 = A_b - lam * X_full - lam_c > 0
        u_ret = q - lam * X - lam_c
        D = (a_O.astype(float) - a_P) * u_ret
        D9 = (a_O.astype(float) - a_9) * u_ret
        D_sum += D
        D9_sum += D9
        # model-predicted gain (Prop. 3, expected-utility form, with the controller's own model)
        # Prop. 3 second-order approximation: G ~= f(b*) lam^2 Var(X_hat) / (2 (g+ + g-))
        h = 0.02
        f_bstar = float(np.mean(np.abs(b - b_cut) < h) / (2 * h))
        G_approx = f_bstar * lam ** 2 * float(X_hat.var()) / (2 * (gp + gm))
        # model-expected gain with the controller's own X_hat (no outcome noise)
        u_model = A_b - lam_c
        G_model = float(np.mean(np.maximum(0, u_model - lam * X_hat))
                        - np.mean(np.maximum(0, u_model - lam * X_bar)))
        rows.append({
            "quantile": qtl, "b_cut": b_cut, "lambda_l": float(lam),
            "G_second_order_approx": G_approx, "G_model_expected": G_model,
            "f_bstar": f_bstar, "var_X_hat": float(X_hat.var()),
            "frac_decisions_differ_vs_B9plus": float((a_O != a_P).mean()),
            "delta_vs_B9plus": float(D.mean()), "sd_vs_B9plus": float(D.std(ddof=1)),
            "delta_vs_B9": float(D9.mean()), "sd_vs_B9": float(D9.std(ddof=1)),
            "retrieve_rate_ours": float(a_O.mean()), "retrieve_rate_b9plus": float(a_P.mean()),
        })
    k = sum(1 for r in rows if "skipped" not in r)
    Dbar = D_sum / max(k, 1)
    D9bar = D9_sum / max(k, 1)
    # oracle (perfect predictor) gain for reference: the Prop.3 Jensen gap with true X
    return {
        "dataset": dataset, "profile": profile, "g_plus": gp, "g_minus": gm,
        "mean_R_s": float(R.mean()), "mean_X_s": X_bar, "sd_X_s": float(X.std()),
        "frac_call_fully_hidden": float((Lcall <= R).mean()),
        "per_price": rows,
        "primary_delta": float(Dbar.mean()), "primary_sd": float(Dbar.std(ddof=1)),
        "vs_B9_delta": float(D9bar.mean()), "vs_B9_sd": float(D9bar.std(ddof=1)),
        "_Dbar": Dbar,
    }


def _bin_mean(key, val, nbins=400):
    """E[val | key] by quantile bins of key (population regression, used for
    calibrated prefix beliefs and continuation values)."""
    edges = np.quantile(key, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, nbins - 1)
    sums = np.bincount(idx, weights=val, minlength=nbins)
    cnts = np.bincount(idx, minlength=nbins)
    return (sums / np.maximum(cnts, 1))[idx]


def simulate_c2(rng, dataset, profile, Q, v1_b, n=400_000, A=ASSUMPTIONS, prefix_noise=None):
    """
    Channels C1 + C2 with two decision epochs: the first meaningful partial
    (prefix belief b1, remaining speech R) and end of turn (final belief bK,
    no overlap). COMMIT architecture: a dispatched retrieval is used.

    b1 = E[bK | s1], s1 = logit(bK) + N(0, prefix_noise^2): a calibrated,
    martingale-consistent coarsening of the final belief (THEORY.md Prop. 4).

    Joint (Ours-v2): dispatch at the prefix iff its value beats the
    continuation value E[max(0, U_K) | s1] (one-step optimal stopping, exact
    here because there are two epochs); otherwise decide at end of turn.
    Decomposed rivals (retrieve decision uses only the belief available at a
    dispatch time fixed by a timing rule that never sees the belief):
      early     -- always decide at the prefix with b1;
      late      -- always decide at the end with bK;
      R-rule    -- decide at the prefix iff R_hat > r0 (timing state only),
                   else at the end; r0 chosen to maximise the rival's own
                   expected utility (an oracle choice that favours the rival).
    Also reported: SPEC, the speculative architecture (always dispatch at the
    prefix, decide at the end whether to USE the result; an unused retrieval
    costs compute only). SPEC is decomposed.
    """
    pn = A["c2_prefix_noise_logit_sd"] if prefix_noise is None else prefix_noise
    bspec = A["belief_distributions"][dataset]
    bK = np.clip(sample_beliefs(rng, bspec, n, v1_b), 1e-4, 1 - 1e-4)
    need = rng.random(n) < bK
    q = np.where(need, rng.choice(Q[True], n), rng.choice(Q[False], n))
    gp, gm = float(Q[True].mean()), float(-Q[False].mean())
    s1 = np.log(bK / (1 - bK)) + pn * rng.standard_normal(n)
    b1 = _bin_mean(s1, bK)

    sp = A["speech"]["d2_support" if dataset == "d2_support" else "d1_open"]
    dur = np.exp(np.log(sp["duration_median_s"]) + sp["duration_sigma_log"] * rng.standard_normal(n))
    R = np.maximum(0.0, A["speech"]["frac_remaining_at_first_meaningful_partial"] * dur - A["speech"]["asr_lag_s"])
    prof = A["latency_profiles"][profile]
    Lcall = np.exp(np.log(prof["call_median_s"]) + prof["call_sigma_log"] * rng.standard_normal(n))
    pre = A["prefill_penalty_s"]
    X1 = pre + np.maximum(0.0, Lcall - R)
    XK = pre + Lcall
    R_hat = R * np.exp(A["remaining_speech_predictor_sigma_log"] * rng.standard_normal(n))
    X1_hat, XK_hat = expected_exposed(R_hat, prof["call_median_s"], prof["call_sigma_log"], pre, rng)
    lam_c = A["lambda_c"]
    Ab1 = b1 * (gp + gm) - gm
    AbK = bK * (gp + gm) - gm

    out = []
    for qtl in A["belief_cut_quantiles"]:
        b_cut = float(np.quantile(bK, qtl))
        lam = (b_cut * (gp + gm) - gm - lam_c) / float(XK.mean())   # late-B9+ cut at this quantile
        if lam <= 0:
            continue
        U_early_real = q - lam * X1 - lam_c
        U_late_real = q - lam * XK - lam_c
        # joint
        D1 = Ab1 - lam * X1_hat - lam_c
        cont = _bin_mean(s1, np.maximum(0.0, AbK - lam * XK_hat - lam_c))
        go_early = D1 > np.maximum(cont, 0.0)
        late_ret = (~go_early) & (AbK - lam * XK_hat - lam_c > 0)
        U_joint = np.where(go_early, U_early_real, np.where(late_ret, U_late_real, 0.0))
        # decomposed rivals
        E_X1 = float(X1.mean())
        U_early = np.where(Ab1 - lam * E_X1 - lam_c > 0, U_early_real, 0.0)
        U_late = np.where(AbK - lam * float(XK.mean()) - lam_c > 0, U_late_real, 0.0)
        best_r = None
        for r0 in np.quantile(R_hat, np.linspace(0.0, 1.0, 21)):
            early_t = R_hat > r0
            ex = float(X1[early_t].mean()) if early_t.any() else E_X1
            U_r = np.where(early_t, np.where(Ab1 - lam * ex - lam_c > 0, U_early_real, 0.0), U_late)
            if best_r is None or U_r.mean() > best_r[1].mean():
                best_r = (float(r0), U_r)
        rivals = {"early": U_early, "late": U_late, "R_rule": best_r[1]}
        best_name = max(rivals, key=lambda k: rivals[k].mean())
        U_best = rivals[best_name]
        # speculative architecture
        use = AbK - lam * X1_hat > 0
        U_spec = -lam_c + np.where(use, q - lam * X1, 0.0)
        D = U_joint - U_best
        out.append({
            "quantile": qtl, "lambda_l": float(lam),
            "frac_joint_early": float(go_early.mean()),
            "E_U_joint": float(U_joint.mean()),
            "E_U_rivals": {k: float(v.mean()) for k, v in rivals.items()},
            "best_rival": best_name, "R_rule_r0": best_r[0],
            "delta_vs_best_decomposed": float(D.mean()), "sd": float(D.std(ddof=1)),
            "N_required": n_required(float(D.mean()), float(D.std(ddof=1)),
                                     A["alpha_one_sided_primary"], A["power"]),
            "E_U_spec": float(U_spec.mean()),
            "delta_joint_minus_spec": float((U_joint - U_spec).mean()),
        })
    # prefix informativeness, for mapping to an E1-v2 prefix AUC
    pos, neg = b1[need], b1[~need]
    m = min(len(pos), len(neg), 20000)
    auc1 = float(np.mean(rng.choice(pos, m)[:, None] > rng.choice(neg, m)[None, :200]))
    posK, negK = bK[need], bK[~need]
    aucK = float(np.mean(rng.choice(posK, m)[:, None] > rng.choice(negK, m)[None, :200]))
    return {"dataset": dataset, "profile": profile, "prefix_noise_logit_sd": pn,
            "auc_prefix_belief": auc1, "auc_final_belief": aucK, "per_price": out}


def n_required(delta, sd, alpha, power):
    if delta <= 0:
        return math.inf
    return math.ceil(((Z[alpha] + Z[power]) * sd / delta) ** 2)


def mde(sd, n, alpha, power):
    return (Z[alpha] + Z[power]) * sd / math.sqrt(n)


def bootstrap_power(rng, Dpop, n, alpha=0.025, reps=400, B=2000):
    """Direct check: fraction of simulated test sets whose one-sided percentile
    bootstrap lower bound exceeds 0."""
    hits = 0
    for _ in range(reps):
        x = rng.choice(Dpop, size=n, replace=True)
        idx = rng.integers(0, n, size=(B, n))
        lo = np.percentile(x[idx].mean(axis=1), 100 * alpha)
        hits += int(lo > 0)
    return hits / reps


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    Q, v1_b = load_v1_outcomes()
    A = ASSUMPTIONS
    alpha, power = A["alpha_one_sided_primary"], A["power"]
    n_test = {d: int(round(A["planned_items_total"][d] * A["split"]["test"])) for d in A["planned_items_total"]}

    cells = []
    for dataset in ("d1_open", "d2_support", "v1_like"):
        for profile in A["latency_profiles"]:
            c = simulate_cell(rng, dataset, profile, Q, v1_b)
            nt = n_test.get(dataset, 50)
            c["n_test_planned"] = nt
            c["N_required_vs_B9plus"] = n_required(c["primary_delta"], c["primary_sd"], alpha, power)
            c["N_required_vs_B9"] = n_required(c["vs_B9_delta"], c["vs_B9_sd"], alpha, power)
            c["MDE_at_planned_N"] = mde(c["primary_sd"], nt, alpha, power)
            c["power_at_planned_N_normal"] = float(
                0.5 * (1 + math.erf(((c["primary_delta"] / (c["primary_sd"] / math.sqrt(nt))) - Z[alpha]) / math.sqrt(2)))
            ) if c["primary_sd"] > 0 else None
            cells.append(c)

    # pooled primary: remote_api, both datasets, item-weighted
    pool = [c for c in cells if c["profile"] == "remote_api" and c["dataset"] in ("d1_open", "d2_support")]
    w = np.array([c["n_test_planned"] for c in pool], float)
    w /= w.sum()
    pooled_delta = float(sum(wi * c["primary_delta"] for wi, c in zip(w, pool)))
    # variance of an item drawn from the mixture
    m2 = sum(wi * (c["primary_sd"] ** 2 + c["primary_delta"] ** 2) for wi, c in zip(w, pool))
    pooled_sd = float(math.sqrt(m2 - pooled_delta ** 2))
    n_pool = sum(c["n_test_planned"] for c in pool)
    pooled = {
        "cell": "remote_api x {d1_open, d2_support}, item-weighted",
        "delta": pooled_delta, "sd": pooled_sd, "n_test_planned": n_pool,
        "N_required": n_required(pooled_delta, pooled_sd, alpha, power),
        "MDE_at_planned_N": mde(pooled_sd, n_pool, alpha, power),
    }
    # direct bootstrap check of the normal approximation on the pooled primary
    Dpool = np.concatenate([rng.choice(c["_Dbar"], size=int(200_000 * wi)) for wi, c in zip(w, pool)])
    pooled["bootstrap_power_at_planned_N"] = bootstrap_power(rng, Dpool, n_pool)
    pooled["normal_power_at_planned_N"] = float(
        0.5 * (1 + math.erf(((pooled_delta / (pooled_sd / math.sqrt(n_pool))) - Z[alpha]) / math.sqrt(2))))

    # sensitivity: predictor error, quality-effect scale (remote_api, d2_support)
    sens = []
    for ps in (0.0, 0.15, 0.30, 0.60):
        for qs in (1.0, 0.5):
            c = simulate_cell(rng, "d2_support", "remote_api", Q, v1_b, n=200_000, pred_sigma=ps, q_scale=qs)
            sens.append({"pred_sigma_log": ps, "quality_effect_scale": qs,
                         "delta": c["primary_delta"], "sd": c["primary_sd"],
                         "N_required": n_required(c["primary_delta"], c["primary_sd"], alpha, power)})

    c2 = []
    for dataset in ("d1_open", "d2_support"):
        for profile in A["latency_profiles"]:
            c2.append(simulate_c2(rng, dataset, profile, Q, v1_b, n=200_000))
    c2_noise = [simulate_c2(rng, "d2_support", "remote_api", Q, v1_b, n=200_000, prefix_noise=pn)
                for pn in (0.75, 1.5, 3.0)]

    # Dev-selected single price per cell (max delta/sd) -- the pre-registered
    # primary-price rule -- then N and MDE per dataset and pooled per profile.
    c2_summary = []
    for c in c2:
        best = max(c["per_price"], key=lambda r: r["delta_vs_best_decomposed"] / max(r["sd"], 1e-12))
        nt = n_test.get(c["dataset"])
        c2_summary.append({
            "dataset": c["dataset"], "profile": c["profile"], "selected_quantile": best["quantile"],
            "delta": best["delta_vs_best_decomposed"], "sd": best["sd"],
            "N_test_required": best["N_required"],
            "N_total_required_at_30_70": math.ceil(best["N_required"] / A["split"]["test"]),
            "n_test_planned": nt, "MDE_at_planned_N": mde(best["sd"], nt, alpha, power),
            "power_at_planned_N": float(0.5 * (1 + math.erf(
                (best["delta_vs_best_decomposed"] / (best["sd"] / math.sqrt(nt)) - Z[alpha]) / math.sqrt(2)))),
        })
    c2_pooled = []
    for profile in A["latency_profiles"]:
        rows = [r for r in c2_summary if r["profile"] == profile]
        ww = np.array([r["n_test_planned"] for r in rows], float)
        ww /= ww.sum()
        dlt = float(sum(wi * r["delta"] for wi, r in zip(ww, rows)))
        m2 = sum(wi * (r["sd"] ** 2 + r["delta"] ** 2) for wi, r in zip(ww, rows))
        sdp = float(math.sqrt(m2 - dlt ** 2))
        npl = sum(r["n_test_planned"] for r in rows)
        nreq = n_required(dlt, sdp, alpha, power)
        c2_pooled.append({"profile": profile, "delta": dlt, "sd": sdp, "n_test_planned": npl,
                          "N_test_required": nreq,
                          "N_total_required_at_30_70": math.ceil(nreq / A["split"]["test"]),
                          "MDE_at_planned_N": mde(sdp, npl, alpha, power)})

    for c in cells:
        c.pop("_Dbar", None)
    out = {"assumptions": A, "n_test_planned": n_test, "alpha_one_sided": alpha, "power": power,
           "cells_C1": cells, "pooled_primary_C1": pooled, "sensitivity_C1_d2_remote_api": sens,
           "cells_C1C2_commit": c2, "C2_prefix_noise_sensitivity_d2_remote_api": c2_noise,
           "C1C2_selected_price_summary": c2_summary, "C1C2_pooled_by_profile": c2_pooled,
           "v1_quality_effects": {"g_plus": float(Q[True].mean()), "g_minus": float(-Q[False].mean()),
                                  "sd_item_effect_needed": float(Q[True].std(ddof=1)),
                                  "sd_item_effect_not_needed": float(Q[False].std(ddof=1))}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda x: None if x == math.inf else x)

    for c in cells:
        print(f"{c['dataset']:10s} {c['profile']:12s} hidden={c['frac_call_fully_hidden']:.2f} "
              f"X={c['mean_X_s']:.2f}±{c['sd_X_s']:.2f} delta={c['primary_delta']:+.4f} sd={c['primary_sd']:.4f} "
              f"Nreq={c['N_required_vs_B9plus']} nt={c['n_test_planned']} MDE={c['MDE_at_planned_N']:.4f} "
              f"pow={c['power_at_planned_N_normal']} | vsB9 delta={c['vs_B9_delta']:+.4f} N={c['N_required_vs_B9']}")
        for r in c["per_price"]:
            if "skipped" not in r:
                print(f"     q={r['quantile']} lam={r['lambda_l']:.3f} differ={r['frac_decisions_differ_vs_B9plus']:.3f} "
                      f"d={r['delta_vs_B9plus']:+.4f} sd={r['sd_vs_B9plus']:.3f} "
                      f"G2={r['G_second_order_approx']:.5f} Gmodel={r['G_model_expected']:.5f} "
                      f"N1={n_required(r['delta_vs_B9plus'], r['sd_vs_B9plus'], alpha, power)}")
    print("pooled:", pooled)
    for s in sens:
        print("sens:", s)
    for c in c2 + c2_noise:
        print(f"C2 {c['dataset']:10s} {c['profile']:12s} pn={c['prefix_noise_logit_sd']} "
              f"AUC1={c['auc_prefix_belief']:.3f} AUCK={c['auc_final_belief']:.3f}")
        for r in c["per_price"]:
            print(f"     q={r['quantile']} lam={r['lambda_l']:.3f} early={r['frac_joint_early']:.2f} "
                  f"U_joint={r['E_U_joint']:+.4f} rivals={ {k: round(v, 4) for k, v in r['E_U_rivals'].items()} } "
                  f"best={r['best_rival']} d={r['delta_vs_best_decomposed']:+.4f} sd={r['sd']:.3f} N={r['N_required']} "
                  f"spec={r['E_U_spec']:+.4f} joint-spec={r['delta_joint_minus_spec']:+.4f}")
    for r in c2_summary:
        print("C1C2 selected:", r)
    for r in c2_pooled:
        print("C1C2 pooled:", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
