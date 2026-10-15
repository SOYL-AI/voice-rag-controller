"""
test_v2_speculative.py — the non-hideable TTFA cost pi, the
commit/speculative architecture switch, the SPEC rival (b10_speculative_v2)
and THEORY.md Proposition 4', checked on synthetic data.

Prop. 4' (rho = 0, remaining speech non-increasing): with lambda_c = 0 SPEC
is optimal over all policies, so EU(SPEC) >= EU(joint-commit); with
lambda_c > 0 the joint optimum (hence joint-commit) beats SPEC by at most
lambda_c (1 + w_1) per turn. Expected utilities are EXACT expectations over
every path of a martingale belief chain (a Bayes-consistent belief, as the
proposition assumes).
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.e3_frontier import aggregate_frontier, dominance_test, validate_v2_records
from controller.baselines._objective_v2 import (
    BeliefTransitions,
    estimate_policy_constants,
    latency_dist_for,
    load_e3v2_config,
    waste_at,
)
from controller.sequential_v2 import TurnCtx, get_controller_v2, model_utility
from harness.latency_model_v2_stub import LatencyModel
from simulator.e3v2_decide import CellStore, run_architectures, run_phase_b
from simulator.e3v2_synthetic import synthetic_cfg_overrides, synthetic_dataset, synthetic_turn

SPREAD = {"latency_profile": "synthetic", "latency_quantiles_ms": {"p10": 300, "p50": 672, "p90": 1500}}
LAMBDA_L_GRID = [0.05, 0.2, 0.5, 1.0, 2.0, 3.0]
M = 8  # martingale grid: beliefs j/M


def cfg_with(**kw):
    return load_e3v2_config(overrides=synthetic_cfg_overrides(**kw))


def with_prices(turn, prices):
    return TurnCtx(item_id=turn.item_id, signals=turn.signals, prices=prices,
                   condition=turn.condition, transcript_final=turn.transcript_final)


# ---------------------------------------------------------------------------
# Proposition 4'
# ---------------------------------------------------------------------------
def martingale_chain() -> BeliefTransitions:
    """Symmetric +-1/M random walk on {0, 1/M, ..., 1}, absorbing at 0 and 1:
    E[b' | b] = b exactly. One bin per grid value."""
    edges = [0.0] + [(2 * j + 1) / (2 * M) for j in range(M)] + [1.0]
    table = {}
    for j in range(M + 1):
        probs = [0.0] * (M + 1)
        if j in (0, M):
            probs[j] = 1.0
        else:
            probs[j - 1] = probs[j + 1] = 0.5
        table[str(j)] = {"probs": probs, "values": [i / M for i in range(M + 1)], "n": 1}
    return BeliefTransitions(edges, [0.0, 1.0], {"0": table}, {"0": dict(table)}, {"kind": "martingale"})


def martingale_paths(chain):
    out = []
    starts = [j / M for j in range(1, M)]
    for b0 in starts:
        for p1, b1 in chain.next_dist(b0, 0.0):
            for p2, b2 in chain.next_dist(b1, 0.0):
                for pf, bf in chain.final_dist(b2, 0.0):
                    out.append((p1 * p2 * pf / len(starts), [b0, b1, b2], bf))
    return out


def prop4_setup(lambda_c):
    cfg = cfg_with(architecture="speculative",
                   rho={"kind": "power", "scale": 0.0, "power": 2.0, "not_ready": 1.0, "table": None},
                   quality={"g_plus": 0.72, "g_minus": 0.59},
                   nonhideable_increment_ms=400.0)
    chain = martingale_chain()
    paths = martingale_paths(chain)
    assert abs(sum(w for w, _, _ in paths) - 1.0) < 1e-12
    turns = [synthetic_turn(f"p4-{i}", bs, bf, interval_ms=300.0, user_end_ms=1200.0, retrieval=SPREAD)
             for i, (_, bs, bf) in enumerate(paths)]
    constants = estimate_policy_constants(turns, cfg)
    ctrls = {
        "ours_v2": get_controller_v2("ours_v2", cfg=cfg, transitions=chain, lookahead_steps=6),
        "b10_speculative_v2": get_controller_v2("b10_speculative_v2", cfg=cfg, timing="immediate"),
        "b9plus_v2": get_controller_v2("b9plus_v2", cfg=cfg, policy_constants=constants),
        "b9_v2": get_controller_v2("b9_v2", cfg=cfg, policy_constants=constants),
    }
    for pol in ("sentence_end", "immediate", "forecast"):
        ctrls[f"b1_v2_{pol}"] = get_controller_v2(f"b1_v2_{pol}", cfg=cfg)
    return cfg, paths, turns, ctrls


def expected_utilities(cfg, paths, turns, ctrls, lam_l, lam_c):
    prices = {"lambda_l": lam_l, "lambda_c": lam_c}
    priced = [with_prices(t, prices) for t in turns]
    return {name: sum(w * model_utility(c.decide_turn(t), t, cfg) for (w, _, _), t in zip(paths, priced))
            for name, c in ctrls.items()}


def test_prop4prime_spec_is_optimal_when_compute_is_free():
    cfg, paths, turns, ctrls = prop4_setup(0.0)
    strict = 0
    for lam in LAMBDA_L_GRID:
        eu = expected_utilities(cfg, paths, turns, ctrls, lam, 0.0)
        for name, value in eu.items():
            assert eu["b10_speculative_v2"] >= value - 1e-12, (lam, name, eu)
        strict += eu["b10_speculative_v2"] > eu["ours_v2"] + 1e-6
    assert strict >= 2  # SPEC strictly beats joint-commit at several prices


@pytest.mark.parametrize("lam_c", [0.05, 0.2])
def test_prop4prime_joint_gain_bounded_by_spec_compute(lam_c):
    cfg, paths, turns, ctrls = prop4_setup(lam_c)
    # w_1: waste of a dispatch at the first epoch (f = 300 / 1200)
    w1 = waste_at(0.25, False, cfg)
    bound = lam_c * (1.0 + w1)
    for lam in LAMBDA_L_GRID:
        eu = expected_utilities(cfg, paths, turns, ctrls, lam, lam_c)
        for name, value in eu.items():
            assert value - eu["b10_speculative_v2"] <= bound + 1e-12, (lam, name, value, eu, bound)


def test_spec_dispatches_unconditionally_and_decides_use_on_final_belief():
    cfg = cfg_with(architecture="speculative", nonhideable_increment_ms=400.0)
    b10 = get_controller_v2("b10_speculative_v2", cfg=cfg)
    prices = {"lambda_l": 0.5, "lambda_c": 0.05}
    lo = synthetic_turn("lo", [0.9, 0.9, 0.9], 0.02, retrieval=SPREAD, prices=prices)
    hi = synthetic_turn("hi", [0.02, 0.02, 0.02], 0.98, retrieval=SPREAD, prices=prices)
    d_lo, d_hi = b10.decide_turn(lo), b10.decide_turn(hi)
    # same dispatch whatever the belief: the dispatch decision is timing-only
    assert d_lo.dispatch_t_ms == d_hi.dispatch_t_ms == 300
    assert d_lo.retrieve and d_hi.retrieve
    assert d_lo.used is False and d_lo.cancelled_dispatches == 1 and not d_lo.uses_passages
    assert d_hi.used is True and d_hi.uses_passages
    assert model_utility(d_lo, lo, cfg) == pytest.approx(-0.05 * (1 + waste_at(0.25, False, cfg)))


def test_speculation_is_refused_under_commit():
    cfg = cfg_with(architecture="commit")
    b10 = get_controller_v2("b10_speculative_v2", cfg=cfg)
    turn = synthetic_turn("c", [0.5, 0.5], 0.5, retrieval=SPREAD)
    with pytest.raises(RuntimeError):
        b10.decide_turn(turn)
    with pytest.raises(ValueError):
        load_e3v2_config(overrides={"architecture": "maybe"})


# ---------------------------------------------------------------------------
# Non-hideable cost: applied identically to every priced controller
# ---------------------------------------------------------------------------
def _priced_utilities(pi):
    cfg = cfg_with(architecture="speculative", nonhideable_increment_ms=pi)
    turn = synthetic_turn("pi", [0.55] * 6, 0.6, retrieval=SPREAD,
                          prices={"lambda_l": 0.8, "lambda_c": 0.05})
    constants = estimate_policy_constants([turn], cfg)
    out = {}
    for name in ("ours_v2", "b9_v2", "b9plus_v2", "b10_speculative_v2"):
        kw = {"policy_constants": constants} if name in ("b9_v2", "b9plus_v2") else {}
        dec = get_controller_v2(name, cfg=cfg, **kw).decide_turn(turn)
        # the utility each controller evaluated at the point where it priced
        # latency: Ours at step 0, B9/B9+ at the trigger, SPEC at the use step
        rec = dec.trace[0] if name == "ours_v2" else [t for t in dec.trace if t.u_retrieve is not None][-1]
        out[name] = (rec.k, rec.u_retrieve)
    return out


def test_nonhideable_cost_shifts_every_priced_controller_identically():
    base, more = _priced_utilities(0.0), _priced_utilities(500.0)
    for name in base:
        assert base[name][0] == more[name][0], name  # same decision point
        assert base[name][1] - more[name][1] == pytest.approx(0.8 * 0.5), name  # lambda_l * 0.5 s


def test_nonhideable_from_signals_overrides_config_and_reaches_ttfa():
    cfg = cfg_with(nonhideable_increment_ms=900.0)
    assert latency_dist_for({"latency_profile": "faiss"}, cfg).nonhideable_ms == 900.0
    assert latency_dist_for({"latency_profile": "faiss", "nonhideable_increment_ms": 250}, cfg).nonhideable_ms == 250.0
    import random
    lm = LatencyModel(cfg=cfg)
    ret = lm.sample_ttfa("retrieve", 0, 5000, "faiss", 600, random.Random(1))
    skip = lm.sample_ttfa("skip", None, 5000, "faiss", 120, random.Random(1))
    # dispatch 5 s before the end of speech hides the whole call: only pi remains
    assert ret["ttfa_ms"] - skip["ttfa_ms"] == pytest.approx(900.0)


def test_prop1_with_nonhideable_cost():
    """A1-A3 with pi > 0: Ours at lambda == B9 at lambda (L+pi-o*)/(L+pi),
    and == B9+ at lambda."""
    L, o, pi = 672.0, 188.0, 400.0
    cfg = cfg_with(rho={"kind": "power", "scale": 0.0, "power": 2.0, "not_ready": 1.0, "table": None},
                   waste={"coef": 0.0}, quality={"g_plus": 0.72, "g_minus": 0.0},
                   nonhideable_increment_ms=pi)
    early = {"hidden_ms": o, "exposed_ms": L - o, "rho": 0.0, "waste": 0.0}
    constants = {"prior_belief": 0.5, "profiles": {"synthetic": {
        "sentence_end": {"hidden_ms": 0.0, "exposed_ms": L, "rho": 0.0, "waste": 0.0},
        "immediate": dict(early), "fixed_early_proxy": dict(early), "forecast": dict(early)}}}
    ours = get_controller_v2("ours_v2", cfg=cfg)
    b9 = get_controller_v2("b9_v2", cfg=cfg, policy_constants=constants)
    b9p = get_controller_v2("b9plus_v2", cfg=cfg, policy_constants=constants)
    deg = {"latency_profile": "synthetic", "latency_quantiles_ms": {"p50": L}}
    scale = (L + pi - o) / (L + pi)
    import random
    rng = random.Random(3)
    for lam in (0.1, 0.4, 1.0, 2.0):
        for i in range(120):
            b = rng.random()
            t = synthetic_turn(f"a{i}", [b] * 5, b, retrieval=deg, remaining_pred=[o] * 5,
                               prices={"lambda_l": lam, "lambda_c": 0.05})
            d = ours.decide_turn(t).retrieve
            assert d == b9.decide_turn(with_prices(t, {"lambda_l": lam * scale, "lambda_c": 0.05})).retrieve
            assert d == b9p.decide_turn(t).retrieve


# ---------------------------------------------------------------------------
# Evaluator: per-architecture reporting
# ---------------------------------------------------------------------------
def test_evaluator_reports_per_architecture(tmp_path):
    turns, cells, labels, items = synthetic_dataset(16, seed=2)
    cfg = cfg_with(nonhideable_increment_ms=600.0)
    store = CellStore(cells)
    names = ["ours_v2", "b9plus_v2", "b10_speculative_v2", "b0_v2"]
    prices = [{"lambda_l": 0.2, "lambda_c": 0.05}, {"lambda_l": 0.6, "lambda_c": 0.05},
              {"lambda_l": 1.2, "lambda_c": 0.05}]
    recs = run_architectures(list(turns.values()), store, names, prices, [11, 23], ["faiss"], cfg,
                             LatencyModel(cfg=cfg), ("commit", "speculative"))
    validate_v2_records(recs)
    by_arch = {}
    for r in recs:
        by_arch.setdefault(r["architecture"], set()).add(r["baseline"])
    assert by_arch["commit"] == {"ours_v2", "b9plus_v2", "b0_v2"}
    assert by_arch["speculative"] == set(names)
    discarded = [r for r in recs if r["used"] is False]
    assert discarded and all(r["architecture"] == "speculative" and r["outcome"]["retrievals"] == 1
                             and r["exposed_latency_ms"] == 0 and r["query_variant"] == "none"
                             and r["decision"]["discarded_query_variant"].startswith("qv-")
                             for r in discarded)
    # commit controllers decide identically under both architectures
    key = lambda r: (r["baseline"], r["item_id"], r["seed"], str(r["prices"]), r["dispatch_t_ms"])
    commit = {key(r) for r in recs if r["architecture"] == "commit"}
    spec = {key(r) for r in recs if r["architecture"] == "speculative" and r["baseline"] != "b10_speculative_v2"}
    assert commit == spec
    bad = dict(discarded[0], architecture="commit")
    with pytest.raises(ValueError):
        validate_v2_records([bad])
    with pytest.raises(ValueError):  # B10 cannot be evaluated under commit
        run_phase_b(list(turns.values()), store,
                    {"b10_speculative_v2": get_controller_v2("b10_speculative_v2", cfg=cfg)},
                    prices, [11], ["faiss"], cfg, LatencyModel(cfg=cfg))

    # parameterised dominance test on the speculative stratum
    spec_rows = [r for r in recs if r["architecture"] == "speculative"]
    aggregated = aggregate_frontier(spec_rows)
    out = dominance_test(aggregated, output_dir=tmp_path, ours="ours_v2", rival="b10_speculative_v2")
    assert out["compared"] == {"ours": "ours_v2", "rival": "b10_speculative_v2"}
    saved = json.loads((tmp_path / "e3_dominance_ours_v2_vs_b10_speculative_v2.json").read_text())
    assert saved["per_seed"] and saved["per_seed"][0]["matched_operating_points"] == len(prices)
    assert "b10_speculative_v2" in saved["per_seed"][0]["operating_points"][0]
