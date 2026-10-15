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
    "waste_early": 0.08,
    "waste_end": 0.0,
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


def _cond_mean_plus(key, values, thresholds, nbins=400):
    """E[(V - a_i)^+ | key-bin of i] for item-specific thresholds a_i, where V
    ranges over the bin's members. Sorted prefix sums, O(n log n) per bin."""
    edges = np.quantile(key, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(edges, key, side="right") - 1, 0, nbins - 1)
    out = np.empty_like(values, dtype=float)
    order = np.argsort(idx, kind="stable")
    bounds = np.searchsorted(idx[order], np.arange(nbins + 1))
    for k in range(nbins):
        members = order[bounds[k]:bounds[k + 1]]
        if members.size == 0:
            continue
        v = np.sort(values[members])
        suffix = np.concatenate([np.cumsum(v[::-1])[::-1], [0.0]])
        pos = np.searchsorted(v, thresholds[members], side="right")
        cnt = v.size - pos
        out[members] = (suffix[pos] - thresholds[members] * cnt) / v.size
    return out


def simulate_boundary(rng, dataset, profile, Q, v1_b, n=200_000, A=ASSUMPTIONS):
    """
    The PREREG_V2 boundary test: the same items under COMMIT and SPECULATIVE,
    two decision epochs (first meaningful partial, end of turn), channels
    C1+C2 active. Compute per dispatch is lambda_c*(1+w) with the v1 waste
    placeholders (early 0.08, end 0), applied identically to every controller.

    COMMIT
      Ours-v2[C]  dispatch-and-commit at epoch 1 iff A(b1) - lam*X1_hat - c1
                  exceeds the continuation E[(A(bK) - lam*XK_hat - cK)^+ | F1];
                  otherwise decide at end of turn.
      D*[C]       best (by population mean, i.e. tuned with oracle knowledge)
                  of the B9+ family {early, late, R-threshold}.
    SPECULATIVE
      SPEC        dispatch at epoch 1 always; use iff Y = A(bK) - lam*X1_hat > 0.
      Ours-v2[S]  dispatch at epoch 1 iff the option value
                  E[(A(bK) - lam*X1_hat)^+ | F1] - c1 exceeds the value of
                  deferring, E[(A(bK) - lam*XK_hat - cK)^+ | F1]; use rule as SPEC;
                  a deferred turn dispatches-and-uses at end iff worthwhile.
      D*[S]       best of {B9+ family best, SPEC}.
    Returns per-price summaries and per-item paired differences.
    """
    pn = A["c2_prefix_noise_logit_sd"]
    bK = np.clip(sample_beliefs(rng, A["belief_distributions"][dataset], n, v1_b), 1e-4, 1 - 1e-4)
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
    w1, wK = A["waste_early"], A["waste_end"]
    c1, cK = lam_c * (1 + w1), lam_c * (1 + wK)
    Ab1 = b1 * (gp + gm) - gm
    AbK = bK * (gp + gm) - gm

    rows, items = [], {}
    for qtl in A["belief_cut_quantiles"]:
        b_cut = float(np.quantile(bK, qtl))
        lam = (b_cut * (gp + gm) - gm - cK) / float(XK.mean())
        if lam <= 0:
            continue
        U_early = q - lam * X1 - c1          # realised, committed early dispatch
        U_late = q - lam * XK - cK           # realised, end-of-turn dispatch
        late_go = AbK - lam * XK_hat - cK > 0
        defer_val = _bin_mean(s1, np.maximum(0.0, AbK - lam * XK_hat - cK))
        # --- COMMIT
        go_C = (Ab1 - lam * X1_hat - c1) > defer_val
        U_OC = np.where(go_C, U_early, np.where(late_go, U_late, 0.0))
        E_X1 = float(X1.mean())
        U_bE = np.where(Ab1 - lam * E_X1 - c1 > 0, U_early, 0.0)
        U_bL = np.where(AbK - lam * float(XK.mean()) - cK > 0, U_late, 0.0)
        best_r = None
        for r0 in np.quantile(R_hat, np.linspace(0.0, 1.0, 21)):
            et = R_hat > r0
            ex = float(X1[et].mean()) if et.any() else E_X1
            U_r = np.where(et, np.where(Ab1 - lam * ex - c1 > 0, U_early, 0.0), U_bL)
            if best_r is None or U_r.mean() > best_r.mean():
                best_r = U_r
        fam = {"early": U_bE, "late": U_bL, "R_rule": best_r}
        b9p_name = max(fam, key=lambda k: fam[k].mean())
        U_DC = fam[b9p_name]
        # --- SPECULATIVE
        Y = AbK - lam * X1_hat                       # model use value of an early retrieval
        use = Y > 0
        U_spec_on = -c1 + np.where(use, q - lam * X1, 0.0)   # realised, dispatched early
        U_SPEC = U_spec_on
        option = _cond_mean_plus(s1, AbK, lam * X1_hat) - c1
        go_S = option > defer_val
        U_OS = np.where(go_S, U_spec_on, np.where(late_go, U_late, 0.0))
        # SPEC-theta: belief-gated speculation, DECOMPOSED. Identical to
        # Ours-v2[S] except that every exposure is the population scalar
        # (X1_bar, XK_bar) instead of the turn-level X1_hat: the prefetch and
        # use decisions read the belief and fixed prices only.
        X1_bar, XK_bar = float(X1_hat.mean()), float(XK_hat)
        use_t = AbK - lam * X1_bar > 0
        late_t = AbK - lam * XK_bar - cK > 0
        opt_t = _bin_mean(s1, np.maximum(AbK - lam * X1_bar, 0.0)) - c1
        def_t = _bin_mean(s1, np.maximum(AbK - lam * XK_bar - cK, 0.0))
        go_t = opt_t > def_t
        U_SPECT = np.where(go_t, -c1 + np.where(use_t, q - lam * X1, 0.0),
                           np.where(late_t, U_late, 0.0))
        cands = {"SPEC": U_SPEC, "SPEC_theta": U_SPECT, "B9+:" + b9p_name: U_DC}
        ds_name = max(cands, key=lambda k: cands[k].mean())
        U_DS = cands[ds_name]
        # the PI's D*[S] (best of B9+ and plain SPEC), kept for comparison
        pi_name = "SPEC" if U_SPEC.mean() >= U_DC.mean() else "B9+:" + b9p_name
        D_S_pi = U_OS - cands[pi_name]
        D_C = U_OC - U_DC
        D_S = U_OS - U_DS
        # rigorous first-order bound vs SPEC-theta (perturbation argument, THEORY Prop. 4'')
        m_C1_bound = float(lam * np.mean(np.abs(X1_hat - X1_bar)
                                         * ((go_S & use).astype(float) + (go_t & use_t))))
        # margins (model quantities, as they would be computed on dev)
        q_hat = float((~use).mean())
        m_pi = c1 * q_hat
        prophet = np.where(Y <= 0, c1, np.maximum(np.maximum(c1 - Y, 0.0),
                                                    np.maximum(c1 - cK - lam * (XK_hat - X1_hat), 0.0)))
        m_prophet = float(prophet.mean())
        pi_defer = float((~go_S).mean())
        m_defer = c1 * pi_defer
        M = np.maximum(Y, 0) - np.maximum(AbK - lam * XK_hat - cK, 0)
        G_exact_model = float(np.mean((~go_S) * (c1 - _bin_mean(s1, M))))
        G_vs_spec_real = float((U_OS - U_SPEC).mean())
        rows.append({
            "quantile": qtl, "lambda_l": float(lam), "c1": c1, "cK": cK,
            "D_star_commit": "B9+:" + b9p_name, "D_star_spec": ds_name,
            "delta_commit": float(D_C.mean()), "sd_commit": float(D_C.std(ddof=1)),
            "delta_spec": float(D_S.mean()), "sd_spec": float(D_S.std(ddof=1)),
            "delta_interaction": float((D_C - D_S).mean()), "sd_interaction": float((D_C - D_S).std(ddof=1)),
            "corr_DC_DS": float(np.corrcoef(D_C, D_S)[0, 1]),
            "ours_spec_minus_SPEC_realised": G_vs_spec_real,
            "ours_spec_minus_SPEC_model_exact": G_exact_model,
            "frac_joint_spec_defers": pi_defer,
            "q_hat_SPEC_unused": q_hat,
            "frac_SPEC_used_but_below_compute": float(((Y > 0) & (Y < c1)).mean()),
            "m_PI": m_pi, "m_prophet": m_prophet, "m_defer": m_defer, "m_C1_bound": m_C1_bound,
            "D_star_spec_PI_definition": pi_name,
            "delta_spec_PI_definition": float(D_S_pi.mean()),
            "delta_interaction_PI_definition": float((D_C - D_S_pi).mean()),
            "ours_spec_minus_SPEC_theta": float((U_OS - U_SPECT).mean()),
            "SPEC_theta_prefetch_rate": float(go_t.mean()),
            "E_U": {"ours_commit": float(U_OC.mean()), "ours_spec": float(U_OS.mean()),
                    "SPEC": float(U_SPEC.mean()), "SPEC_theta": float(U_SPECT.mean()),
                    "B9plus": float(U_DC.mean())},
        })
        items[qtl] = (D_C, D_S, D_S_pi)
    return {"dataset": dataset, "profile": profile, "per_price": rows, "_items": items}


_STD_NORMAL = np.random.default_rng(12345).standard_normal((50_000, 4))


def _moments(z):
    mu = z.mean(axis=0)
    S = np.cov(z, rowvar=False)
    # the TOST pair (m - D, D + m) is perfectly anti-correlated in D; jitter keeps Cholesky defined
    Lc = np.linalg.cholesky(S + 1e-12 * np.eye(S.shape[0]))
    return mu, S, Lc


def conjunction_power(z, N, alpha, rng=None, moments=None):
    """Power of the intersection-union test: each column of z is a per-item
    statistic whose mean must be shown > 0 one-sided at alpha. The vector of
    sample means is approximately N(mu, Sigma/N); 50,000 fixed normal draws
    (common random numbers, so N-search is monotone)."""
    mu, S, Lc = moments if moments is not None else _moments(z)
    k = mu.size
    sims = mu + (_STD_NORMAL[:, :k] @ Lc.T) / math.sqrt(N)
    sd = np.sqrt(np.diag(S) / N)
    rej = sims - Z[alpha] * sd > 0
    return float(rej.all(axis=1).mean()), [float(x) for x in rej.mean(axis=0)]


def n_for_conjunction(z, alpha, target=0.8, rng=None):
    mom = _moments(z)
    if np.any(mom[0] <= 0):
        return math.inf
    lo, hi = 10, 10_000_000
    if conjunction_power(z, hi, alpha, moments=mom)[0] < target:
        return math.inf
    while hi / lo > 1.01:
        mid = int(math.sqrt(lo * hi))
        if conjunction_power(z, mid, alpha, moments=mom)[0] >= target:
            hi = mid
        else:
            lo = mid
    return hi


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

    boundary = boundary_test_power(Q, v1_b, n_test, alpha, power)

    for c in cells:
        c.pop("_Dbar", None)
    out = {"assumptions": A, "n_test_planned": n_test, "alpha_one_sided": alpha, "power": power,
           "boundary_test": boundary,
           "cells_C1": cells, "pooled_primary_C1": pooled, "sensitivity_C1_d2_remote_api": sens,
           "cells_C1C2_commit": c2, "C2_prefix_noise_sensitivity_d2_remote_api": c2_noise,
           "C1C2_selected_price_summary": c2_summary, "C1C2_pooled_by_profile": c2_pooled,
           "MDE_table": {
               "N_test": [1344, 2000, 3000, 4000, 6000, 8000],
               "C1C2_pooled": {r["profile"]: [mde(r["sd"], n, alpha, power) for n in (1344, 2000, 3000, 4000, 6000, 8000)]
                               for r in c2_pooled},
               "C1_pooled_remote_api": [mde(pooled["sd"], n, alpha, power) for n in (1344, 2000, 3000, 4000, 6000, 8000)],
               "multiplier_z_alpha_plus_z_power": Z[alpha] + Z[power],
           },
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
    for prof, rows in boundary["pooled"].items():
        for r in rows:
            print(f"BOUNDARY {prof:12s} q={r['quantile']} dC={r['delta_commit']:+.4f} "
                  f"dS={r['delta_spec']:+.4f} dS_PI={r['delta_spec_PI']:+.4f} | "
                  f"mPI={r['m_PI']:.4f} mProph={r['m_prophet']:.4f} mC1={r['m_C1_bound']:.4f} "
                  f"mHalf={r['m_half']:.4f} | N: " + ", ".join(f"{k}={v['N_test_80pct']}" for k, v in r["designs"].items()))
    print("BOUNDARY selected:", json.dumps(boundary["selected"], indent=0))
    return 0


def boundary_test_power(Q, v1_b, n_test, alpha, power, n=100_000):
    """
    Conjunction (intersection-union) power for the PREREG_V2 boundary test,
    pooled over d1_open and d2_support (item-weighted by planned test size),
    per latency profile and price. Uses a separate random stream so the
    earlier sections of this script are unchanged.

    Designs (each column is a per-item statistic whose mean must be shown > 0
    one-sided at alpha; the conjunction rejects iff every column does):
      PI      P1: D_C;  P2: D_C - D_S_PI;  P3: m_PI - D_S_PI
              with D*[S] = best of {B9+, SPEC} and m_PI = lambda_c (1+w) q_hat.
      PI_prophet  same, with the valid prophet bound as the margin.
      corrected family: D*[S] = best of {B9+, SPEC, SPEC-theta};
              P1: D_C;  P2: D_C - D_S;  P3 (TOST, two one-sided tests):
              m_half - D_S and D_S + m_half, m_half = delta_C / 2.
      corrected_nonsup   as corrected, P3 one-sided only (m_half - D_S).
      corrected_C1bound  as corrected_nonsup with the rigorous Prop. 4''
              margin m_C1 instead of m_half.
      recommended        = corrected (TOST with m3 = delta_C / 2); the design PREREG_V2 adopts.
      recommended_C1cap  TOST with m3 = min(m_half, m_C1) (valid only for two
              epochs and timing-invariant rho; THEORY Prop. 4'').
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    A = ASSUMPTIONS
    share = {d: n_test[d] / sum(n_test.values()) for d in n_test}
    sims = {(d, p): simulate_boundary(rng, d, p, Q, v1_b, n=n)
            for d in ("d1_open", "d2_support") for p in A["latency_profiles"]}
    pooled = {}
    for p in A["latency_profiles"]:
        rows = []
        for qtl in A["belief_cut_quantiles"]:
            parts = []
            meta = {}
            for d in ("d1_open", "d2_support"):
                s = sims[(d, p)]
                if qtl not in s["_items"]:
                    break
                D_C, D_S, D_S_pi = s["_items"][qtl]
                k = int(n * share[d])
                sel = rng.choice(D_C.size, size=k, replace=False)
                parts.append(np.column_stack([D_C[sel], D_S[sel], D_S_pi[sel]]))
                meta[d] = next(r for r in s["per_price"] if r["quantile"] == qtl)
            if len(parts) < 2:
                continue
            X = np.vstack(parts)
            D_C, D_S, D_S_pi = X[:, 0], X[:, 1], X[:, 2]
            w = {d: share[d] for d in meta}
            m_pi = sum(w[d] * meta[d]["m_PI"] for d in meta)
            m_pr = sum(w[d] * meta[d]["m_prophet"] for d in meta)
            m_c1 = sum(w[d] * meta[d]["m_C1_bound"] for d in meta)
            dC = float(D_C.mean())
            m_half = max(dC, 0.0) / 2
            m_rec = min(m_half, m_c1)
            designs = {
                "recommended": np.column_stack([D_C, D_C - D_S, m_half - D_S, D_S + m_half]),
                "recommended_C1cap": np.column_stack([D_C, D_C - D_S, m_rec - D_S, D_S + m_rec]),
                "PI": np.column_stack([D_C, D_C - D_S_pi, m_pi - D_S_pi]),
                "PI_prophet": np.column_stack([D_C, D_C - D_S_pi, m_pr - D_S_pi]),
                "corrected_nonsup": np.column_stack([D_C, D_C - D_S, m_half - D_S]),
                "corrected_C1bound": np.column_stack([D_C, D_C - D_S, m_c1 - D_S]),
            }
            if p == "faiss" and qtl == 0.65:
                check_z = designs["recommended"]
            res = {}
            for name, z in designs.items():
                nreq = n_for_conjunction(z, alpha, power, rng=np.random.default_rng(7))
                p_at = {N: conjunction_power(z, N, alpha, rng=np.random.default_rng(7))
                        for N in (1344, 4000)}
                res[name] = {
                    "column_means": [float(x) for x in z.mean(axis=0)],
                    "column_sds": [float(x) for x in z.std(axis=0, ddof=1)],
                    "N_test_80pct": nreq,
                    "N_total_80pct_at_30_70": (math.ceil(nreq / A["split"]["test"]) if nreq != math.inf else math.inf),
                    "power_at_1344": {"conjunction": p_at[1344][0], "individual": p_at[1344][1]},
                    "power_at_4000": {"conjunction": p_at[4000][0], "individual": p_at[4000][1]},
                }
            rows.append({
                "quantile": qtl,
                "lambda_l_by_dataset": {d: meta[d]["lambda_l"] for d in meta},
                "D_star_spec_by_dataset": {d: meta[d]["D_star_spec"] for d in meta},
                "D_star_spec_PI_by_dataset": {d: meta[d]["D_star_spec_PI_definition"] for d in meta},
                "delta_commit": dC, "sd_commit": float(D_C.std(ddof=1)),
                "delta_spec": float(D_S.mean()), "sd_spec": float(D_S.std(ddof=1)),
                "delta_spec_PI": float(D_S_pi.mean()), "sd_spec_PI": float(D_S_pi.std(ddof=1)),
                "delta_interaction": float((D_C - D_S).mean()),
                "delta_interaction_PI": float((D_C - D_S_pi).mean()),
                "m_PI": m_pi, "m_prophet": m_pr, "m_C1_bound": m_c1, "m_half": m_half, "m_recommended": m_rec,
                "PI_margin_valid_as_bound_vs_SPEC": bool(all(
                    meta[d]["ours_spec_minus_SPEC_model_exact"] <= meta[d]["m_PI"] + 1e-12 for d in meta)),
                "designs": res,
            })
        pooled[p] = rows
    # price selection: (i) the PI's rule (max delta_C / sd_C); (ii) min conjunction N (corrected design)
    selected = {}
    for p, rows in pooled.items():
        r_pi = max(rows, key=lambda r: r["delta_commit"] / max(r["sd_commit"], 1e-12))
        r_n = min(rows, key=lambda r: r["designs"]["recommended"]["N_test_80pct"])
        selected[p] = {
            "rule_max_dC_over_sd": {"quantile": r_pi["quantile"],
                                    **{k: v["N_test_80pct"] for k, v in r_pi["designs"].items()}},
            "rule_min_conjunction_N_recommended": {"quantile": r_n["quantile"],
                                                 **{k: v["N_test_80pct"] for k, v in r_n["designs"].items()}},
        }
    import gc
    for s_ in sims.values():
        s_.pop("_items", None)
    gc.collect()
    # direct check of the normal approximation: simulate test sets of the
    # required size and run the actual paired percentile bootstrap on each
    n_chk = pooled["faiss"][[r["quantile"] for r in pooled["faiss"]].index(0.65)]["designs"]["recommended"]["N_test_80pct"]
    brng = np.random.default_rng(RNG_SEED + 2)
    hits, reps, B = 0, 200, 1000
    hits_wald = 0
    for _ in range(reps):
        x = check_z[brng.integers(0, check_z.shape[0], size=n_chk)]
        hits_wald += int(np.all(x.mean(axis=0) - Z[alpha] * x.std(axis=0, ddof=1) / math.sqrt(n_chk) > 0))
        means = np.vstack([x[brng.integers(0, n_chk, size=(10, n_chk))].mean(axis=1)
                           for _ in range(B // 10)])
        lo = np.percentile(means, 100 * alpha, axis=0)
        hits += int(np.all(lo > 0))
    boot_check = {"cell": "faiss, q=0.65, recommended design", "N_test": n_chk,
                  "reps": reps, "B": B, "bootstrap_conjunction_power": hits / reps,
                  "wald_same_datasets_conjunction_power": hits_wald / reps,
                  "mc_se": math.sqrt(0.8 * 0.2 / reps),
                  "normal_approx_conjunction_power": conjunction_power(check_z, n_chk, alpha)[0]}
    per_dataset = {f"{d}|{p}": s["per_price"] for (d, p), s in sims.items()}
    return {"pooled": pooled, "selected": selected, "per_dataset": per_dataset,
            "test_shares": share, "bootstrap_check": boot_check}


if __name__ == "__main__":
    raise SystemExit(main())
