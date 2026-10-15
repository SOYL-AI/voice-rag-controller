"""
test_v2_theory.py — the v2 theory, checked on synthetic beliefs.

  (i)   Proposition 1. Under A1-A3 (constant overlap o*, rho = 0, belief
        constant across prefixes, g- = 0) Ours-v2 at lambda makes exactly
        B9-v2's decisions at lambda * (L - o*) / L. (And B9+ at lambda, which
        prices L - o*, coincides with Ours at lambda.)
  (ii)  Proposition 2. Under A1-A3 the expected-utility gap
        EU(Ours) - EU(B9) at one price lies in [0, lambda_l * o* * P(b in band)].
  (iii) Proposition 3. With turn-dependent remaining speech (C1) — and,
        separately, with an informative belief process (C2) — Ours-v2's
        expected utility is >= B9+'s at every price, strictly when the band
        has mass.
  (iv)  Causality. Perturbing everything after time tau leaves every step
        before tau, and any dispatch before tau, unchanged.

Expected utility is the shared v2 objective evaluated at each decision's
dispatch step (controller.sequential_v2.model_utility); on synthetic data
the remaining-speech prediction equals the truth, so this is the true
expected utility.
"""

import copy
import itertools
import math
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective import Prices
from controller.baselines._objective_v2 import (
    BeliefTransitions,
    belief_cut,
    divergence_band_v2,
    estimate_policy_constants,
    fit_belief_transitions,
    latency_dist_for,
    load_belief_calibration,
    load_e3v2_config,
    quality_model,
    step_terms,
)
from controller.sequential_v2 import TurnCtx, build_steps, get_controller_v2, model_utility
from simulator.e3v2_synthetic import synthetic_cfg_overrides, synthetic_turn

L_MS = 672.0
O_STAR = 188.0
DEGENERATE = {"latency_profile": "synthetic", "latency_quantiles_ms": {"p50": L_MS}}
SPREAD = {"latency_profile": "synthetic", "latency_quantiles_ms": {"p10": 300, "p50": 672, "p90": 1500}}
LAMBDA_L_GRID = [0.05, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0, 3.0]
LAMBDA_C = 0.05


def a_world_cfg(g_minus=0.0):
    """A1: rho = 0. A2 comes from the turns (constant R = o*). A3 from the
    beliefs. Waste 0 so the compute terms of the two sides coincide."""
    return load_e3v2_config(overrides=synthetic_cfg_overrides(
        rho={"kind": "power", "scale": 0.0, "power": 2.0, "not_ready": 1.0, "table": None},
        waste={"coef": 0.0},
        quality={"g_plus": 0.72, "g_minus": g_minus},
    ))


def a_world_turn(i, b, prices):
    n = 5
    return synthetic_turn(f"a-{i}", [b] * n, b, interval_ms=300.0, retrieval=DEGENERATE,
                          prices=prices, remaining_pred=[O_STAR] * n)


def a_world_constants():
    early = {"hidden_ms": O_STAR, "exposed_ms": L_MS - O_STAR, "rho": 0.0, "waste": 0.0}
    return {"prior_belief": 0.5, "profiles": {"synthetic": {
        "sentence_end": {"hidden_ms": 0.0, "exposed_ms": L_MS, "rho": 0.0, "waste": 0.0},
        "immediate": dict(early), "fixed_early_proxy": dict(early), "forecast": dict(early)}}}


def with_prices(turn, prices):
    return TurnCtx(item_id=turn.item_id, signals=turn.signals, prices=prices,
                   condition=turn.condition, transcript_final=turn.transcript_final)


# ---------------------------------------------------------------------------
# (i) Proposition 1
# ---------------------------------------------------------------------------
def test_prop1_ours_at_lambda_equals_b9_at_scaled_lambda():
    cfg = a_world_cfg(g_minus=0.0)
    ours = get_controller_v2("ours_v2", cfg=cfg)
    b9 = get_controller_v2("b9_v2", cfg=cfg, policy_constants=a_world_constants())
    b9plus = get_controller_v2("b9plus_v2", cfg=cfg, policy_constants=a_world_constants())
    rng = random.Random(1)
    beliefs = [rng.random() for _ in range(300)]
    scale = (L_MS - O_STAR) / L_MS
    n_retrieve = 0
    for lam in LAMBDA_L_GRID:
        p_ours = {"lambda_l": lam, "lambda_c": LAMBDA_C}
        p_b9 = {"lambda_l": lam * scale, "lambda_c": LAMBDA_C}
        for i, b in enumerate(beliefs):
            turn = a_world_turn(i, b, p_ours)
            d_ours = ours.decide_turn(turn)
            d_b9 = b9.decide_turn(with_prices(turn, p_b9))
            d_b9p = b9plus.decide_turn(turn)
            assert d_ours.retrieve == d_b9.retrieve, (lam, b)
            # The reviewer's point: under A1-A3, B9+ IS Ours.
            assert d_ours.retrieve == d_b9p.retrieve, (lam, b)
            n_retrieve += d_ours.retrieve
    assert 0 < n_retrieve < len(beliefs) * len(LAMBDA_L_GRID)  # non-vacuous


# ---------------------------------------------------------------------------
# (ii) Proposition 2
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("g_minus", [0.0, 0.59])
def test_prop2_gap_bounded_by_band_mass(g_minus):
    cfg = a_world_cfg(g_minus=g_minus)
    quality = quality_model(cfg)
    ours = get_controller_v2("ours_v2", cfg=cfg)
    b9 = get_controller_v2("b9_v2", cfg=cfg, policy_constants=a_world_constants())
    rng = random.Random(2)
    beliefs = [rng.random() for _ in range(400)]
    positive_gaps = 0
    for lam in LAMBDA_L_GRID:
        prices = {"lambda_l": lam, "lambda_c": LAMBDA_C}
        band = divergence_band_v2(Prices(lam, LAMBDA_C), quality, L_MS - O_STAR, L_MS)
        eu_ours = eu_b9 = 0.0
        for i, b in enumerate(beliefs):
            turn = a_world_turn(i, b, prices)
            eu_ours += model_utility(ours.decide_turn(turn), turn, cfg)
            eu_b9 += model_utility(b9.decide_turn(turn), turn, cfg)
        gap = (eu_ours - eu_b9) / len(beliefs)
        mass = sum(band.contains(b) for b in beliefs) / len(beliefs)
        bound = lam * (O_STAR / 1000.0) * mass
        assert -1e-12 <= gap <= bound + 1e-12, (lam, gap, bound)
        positive_gaps += gap > 1e-6
    assert positive_gaps >= 3


# ---------------------------------------------------------------------------
# (iii) Proposition 3 — C1: turn-dependent remaining speech
# ---------------------------------------------------------------------------
def c1_turns(n=160, seed=3, interval=300.0):
    rng = random.Random(seed)
    turns = []
    for i in range(n):
        end = rng.uniform(700.0, 4200.0)
        n_partials = max(1, math.ceil(end / interval) - 1)
        b = rng.random()
        turns.append(synthetic_turn(f"c1-{i}", [b] * n_partials, b, interval_ms=interval,
                                    user_end_ms=end, retrieval=SPREAD))
    return turns


def _band_mass_c1(turns, decisions_b9p, b9p, cfg, prices):
    """Fraction of turns whose belief lies strictly between Ours's cut (the
    lowest belief at which SOME dispatch of this turn is positive) and B9+'s
    cut at its trigger with its constant exposure price."""
    cal = load_belief_calibration(cfg)
    quality = quality_model(cfg)
    inside = 0
    for turn, dec in zip(turns, decisions_b9p):
        lat = latency_dist_for(turn.signals["retrieval"], cfg)
        steps = build_steps(turn, cfg)
        terms = [step_terms(o, cfg, cal, lat) for o in steps]
        ours_cut = min(belief_cut(t.rho, t.exposed_ms, t.waste, prices, quality) for t in terms)
        trig = terms[dec.trace[-1].k]
        # priced_latency_ms is B9+'s constant for this (profile, price): the
        # same for every turn here, so reading it after the loop is exact.
        rival_cut = belief_cut(trig.rho, b9p.priced_latency_ms, trig.waste, prices, quality)
        lo, hi = sorted((ours_cut, rival_cut))
        inside += lo < trig.belief < hi
    return inside / len(turns)


def test_prop3_c1_ours_dominates_b9plus():
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(quality={"g_plus": 0.72, "g_minus": 0.59}))
    turns = c1_turns()
    constants = estimate_policy_constants(turns, cfg)
    ours = get_controller_v2("ours_v2", cfg=cfg, lookahead_steps=40)
    b9p = get_controller_v2("b9plus_v2", cfg=cfg, policy_constants=constants)
    b9 = get_controller_v2("b9_v2", cfg=cfg, policy_constants=constants)
    strict = 0
    for lam in LAMBDA_L_GRID:
        prices = {"lambda_l": lam, "lambda_c": LAMBDA_C}
        priced = [with_prices(t, prices) for t in turns]
        d_ours = [ours.decide_turn(t) for t in priced]
        d_b9p = [b9p.decide_turn(t) for t in priced]
        d_b9 = [b9.decide_turn(t) for t in priced]
        eu = lambda ds: sum(model_utility(d, t, cfg) for d, t in zip(ds, priced)) / len(priced)
        eu_ours, eu_b9p, eu_b9 = eu(d_ours), eu(d_b9p), eu(d_b9)
        assert eu_ours >= eu_b9p - 1e-12, (lam, eu_ours, eu_b9p)
        assert eu_ours >= eu_b9 - 1e-12, (lam, eu_ours, eu_b9)
        mass = _band_mass_c1(priced, d_b9p, b9p, cfg, Prices(lam, LAMBDA_C))
        if mass > 0:
            assert eu_ours > eu_b9p, (lam, mass)
            strict += 1
    assert strict >= 3  # the band is populated at several prices: non-vacuous


# ---------------------------------------------------------------------------
# (iii) Proposition 3 — C2: informative belief process, exact expectation
# ---------------------------------------------------------------------------
VALUES = [0.1, 0.35, 0.65, 0.9]


def c2_transitions():
    nxt, fin = {}, {}
    for i in range(4):
        k = [0.0] * 4
        k[i] = 0.5
        for j in (i - 1, i + 1):
            if 0 <= j < 4:
                k[j] = 0.25
        s = sum(k)
        nxt[str(i)] = {"probs": [x / s for x in k], "values": list(VALUES), "n": 100}
    sharpen = [[0.8, 0.1, 0.05, 0.05], [0.45, 0.25, 0.15, 0.15],
               [0.15, 0.15, 0.25, 0.45], [0.05, 0.05, 0.1, 0.8]]
    for i in range(4):
        fin[str(i)] = {"probs": sharpen[i], "values": list(VALUES), "n": 100}
    return BeliefTransitions([0, 0.25, 0.5, 0.75, 1.0], [0.0, 1.0], {"0": nxt}, {"0": fin},
                             {"kind": "synthetic"})


def c2_paths(trans):
    """Every belief path of a 3-partial turn (+ end) under `trans`, weighted."""
    out = []
    for b0 in VALUES:
        for (p1, b1) in trans.next_dist(b0, 0.0):
            for (p2, b2) in trans.next_dist(b1, 0.0):
                for (pf, bf) in trans.final_dist(b2, 0.0):
                    out.append((0.25 * p1 * p2 * pf, [b0, b1, b2], bf))
    return out


def test_prop3_c2_ours_dominates_with_information():
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(quality={"g_plus": 0.72, "g_minus": 0.59}))
    trans = c2_transitions()
    paths = c2_paths(trans)
    assert abs(sum(w for w, _, _ in paths) - 1.0) < 1e-9
    turns = [synthetic_turn(f"c2-{i}", bs, bf, interval_ms=300.0, user_end_ms=1200.0, retrieval=SPREAD)
             for i, (_, bs, bf) in enumerate(paths)]
    constants = estimate_policy_constants(turns, cfg)
    ours = get_controller_v2("ours_v2", cfg=cfg, transitions=trans, lookahead_steps=6)
    rivals = {"b9plus_v2": get_controller_v2("b9plus_v2", cfg=cfg, policy_constants=constants),
              "b9_v2": get_controller_v2("b9_v2", cfg=cfg, policy_constants=constants)}
    for pol in ("sentence_end", "immediate", "fixed_early_proxy", "forecast"):
        rivals[f"b1_v2_{pol}"] = get_controller_v2(f"b1_v2_{pol}", cfg=cfg)
    max_gap = 0.0
    for lam in LAMBDA_L_GRID:
        prices = {"lambda_l": lam, "lambda_c": LAMBDA_C}
        priced = [with_prices(t, prices) for t in turns]
        weights = [w for w, _, _ in paths]

        def eu(ctrl):
            return sum(w * model_utility(ctrl.decide_turn(t), t, cfg) for w, t in zip(weights, priced))

        eu_ours = eu(ours)
        for name, ctrl in rivals.items():
            eu_r = eu(ctrl)
            assert eu_ours >= eu_r - 1e-12, (lam, name, eu_ours, eu_r)
            if name == "b9plus_v2":
                max_gap = max(max_gap, eu_ours - eu_r)
    assert max_gap > 1e-3


def test_ours_waits_for_information_when_it_pays():
    """C2 in one turn: an ambiguous early belief whose resolution is
    informative makes waiting worth more than retrieving now."""
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(quality={"g_plus": 0.72, "g_minus": 0.59}))
    trans = c2_transitions()
    ours_info = get_controller_v2("ours_v2", cfg=cfg, transitions=trans, lookahead_steps=6)
    ours_flat = get_controller_v2("ours_v2", cfg=cfg, transitions=BeliefTransitions.identity(4),
                                  lookahead_steps=6)
    turn = synthetic_turn("wait", [0.65, 0.65, 0.65], 0.65, interval_ms=300.0, user_end_ms=1200.0,
                          retrieval=SPREAD, prices={"lambda_l": 0.2, "lambda_c": LAMBDA_C})
    first_info = ours_info.decide_turn(turn).trace[0]
    first_flat = ours_flat.decide_turn(turn).trace[0]
    assert first_info.v_wait > first_flat.v_wait


# ---------------------------------------------------------------------------
# (iv) Causality
# ---------------------------------------------------------------------------
def _perturb_after(turn: TurnCtx, tau: float, rng: random.Random) -> TurnCtx:
    sig = copy.deepcopy(turn.signals)
    timing = sig["timing"]
    for p in timing["partials"]:
        if p["t_ms"] > tau:
            p["text"] = " ".join(rng.choice(["zebra", "quantum", "invoice", "the", "router"])
                                 for _ in range(rng.randint(1, 9)))
    for e in sig["model"]["uncertainty_by_prefix"]:
        if e["t_ms"] > tau:
            e["p_needed"] = 1.0 - e["p_needed"]
            if "max_entropy_nats" in e:
                e["max_entropy_nats"] = 3.0 - e["max_entropy_nats"]
            if "min_token_prob" in e:
                e["min_token_prob"] = 1.0 - e["min_token_prob"]
    for r in timing["predicted_remaining_speech_ms"]:
        if r["t_ms"] > tau:
            r["remaining_ms"] = r["remaining_ms"] * 0.3 + 50
    timing["user_end_ms"] = timing["user_end_ms"] + 500
    timing["asr_final"] = {"t_ms": timing["asr_final"]["t_ms"] + 500, "text": "something else entirely"}
    timing["partials"].append({"t_ms": timing["user_end_ms"] - 100, "text": "late extra partial words here"})
    return TurnCtx(item_id=turn.item_id, signals=sig, prices=turn.prices, condition=turn.condition,
                   transcript_final="something else entirely")


def test_causality_future_perturbation_does_not_change_the_past():
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(quality={"g_plus": 0.72, "g_minus": 0.59}))
    rng = random.Random(5)
    seqs = [[(k / 8, min(1, max(0, 0.5 + rng.gauss(0, 0.3)))) for k in range(9)] for _ in range(200)]
    trans = fit_belief_transitions(seqs, n_bins=6)
    base_turns = []
    for i in range(12):
        bs = [min(1, max(0, rng.random())) for _ in range(8)]
        extra = [{"max_entropy_nats": rng.uniform(0, 2), "min_token_prob": rng.random()} for _ in range(8)]
        base_turns.append(synthetic_turn(f"cz-{i}", bs, rng.random(), interval_ms=300.0, retrieval=SPREAD,
                                         entry_extra=extra,
                                         final_extra={"max_entropy_nats": 1.0, "min_token_prob": 0.4}))
    constants = estimate_policy_constants(base_turns, cfg)
    names = ["ours_v2", "b9_v2", "b9plus_v2", "b0_v2", "b4_v2", "b5_v2", "b6_v2", "b8_v2",
             "b1_v2_immediate", "b1_v2_fixed_early_proxy", "b1_v2_forecast", "b1_v2_sentence_end"]
    ctrls = {}
    for n in names:
        kw = {}
        if n == "ours_v2":
            kw = {"transitions": trans, "lookahead_steps": 2}
        elif n in ("b9_v2", "b9plus_v2"):
            kw = {"policy_constants": constants}
        elif n == "b8_v2":
            kw = {"trigger": "b6", "timing": "immediate"}
        ctrls[n] = get_controller_v2(n, cfg=cfg, **kw)
    # speculative architecture: the joint speculative controller and SPEC
    spec_cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(
        quality={"g_plus": 0.72, "g_minus": 0.59}, architecture="speculative"))
    ctrls["ours_v2[spec]"] = get_controller_v2("ours_v2", cfg=spec_cfg, transitions=trans,
                                               lookahead_steps=2)
    ctrls["b10_speculative_v2"] = get_controller_v2("b10_speculative_v2", cfg=spec_cfg)

    checked_dispatches = 0
    for turn in base_turns:
        for tau in (650.0, 1250.0, 1850.0):
            for lam in (0.05, 0.5, 2.0):
                prices = {"lambda_l": lam, "lambda_c": LAMBDA_C}
                orig = with_prices(turn, prices)
                pert = _perturb_after(orig, tau, random.Random(int(tau) + int(lam * 100)))
                s0 = [o for o in build_steps(orig, cfg) if o.available_ms <= tau]
                s1 = [o for o in build_steps(pert, cfg) if o.available_ms <= tau]
                assert s0 == s1
                assert not any(o.is_final for o in s0)
                for name, ctrl in ctrls.items():
                    d0, d1 = ctrl.decide_turn(orig), ctrl.decide_turn(pert)
                    t0 = [t for t in d0.trace if t.available_ms <= tau]
                    t1 = [t for t in d1.trace if t.available_ms <= tau]
                    assert t0 == t1, (name, tau, lam)
                    if d0.retrieve and d0.dispatch_t_ms <= tau:
                        checked_dispatches += 1
                        assert (d1.dispatch_t_ms, d1.query_text, d1.dispatch_partial_index) == \
                               (d0.dispatch_t_ms, d0.query_text, d0.dispatch_partial_index), name
                    elif d0.trace[-1].available_ms <= tau and not d0.retrieve:
                        assert not d1.retrieve, name  # an early skip is final
                    else:
                        assert not (d1.retrieve and d1.dispatch_t_ms <= tau), name
    assert checked_dispatches > 20  # many pre-tau dispatches were actually exercised
