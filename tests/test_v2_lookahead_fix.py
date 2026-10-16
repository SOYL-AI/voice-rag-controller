"""
test_v2_lookahead_fix.py — the one-time correction of Ours-v2's lookahead
(2026-10-16). The probability that another partial becomes available before
the end-of-turn decision comes from the remaining-speech predictor's
quantiles, p = P(R + gap > spacing), and the value of waiting is the mixture
p * (next-partial value) + (1 - p) * (end-of-turn decision value). The same
mixture is used by Ours-v2[SPECULATIVE].
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import RemainingDist, estimated_frac, load_e3v2_config
from controller.sequential_v2 import TurnStatic, get_controller_v2
from simulator.e3v2_synthetic import synthetic_cfg_overrides
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v2_speculative import martingale_chain  # noqa: E402

RET = {"latency_profile": "synthetic", "latency_quantiles_ms": {"p10": 300, "p50": 672, "p90": 1500}}
INTERVAL, B, E = 400.0, 0.5, 1200.0


def make(arch="commit", gap=800.0):
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(
        architecture=arch, quality={"g_plus": 0.6, "g_minus": 0.2}, nonhideable_increment_ms=300.0,
        timing={"eot_detection_ms": 0, "end_gap_ms": gap}))
    ours = get_controller_v2("ours_v2", cfg=cfg, transitions=martingale_chain(), lookahead_steps=1)
    ours.begin_turn(TurnStatic(prices={"lambda_l": 0.4, "lambda_c": 0.05}, retrieval=RET))
    return ours


def test_no_further_partial_is_certain():
    # no end gap and every quantile below the 400 ms spacing: p = 0, so the
    # value of waiting is exactly the end-of-turn decision value
    ours = make(gap=0.0)
    v = ours.v_wait(B, 100.0, E, INTERVAL, 1, (50.0, 100.0, 150.0))
    assert v == pytest.approx(ours._v_end(B, estimated_frac(E, 100.0, False)))


def test_further_partial_is_certain_recovers_old_behaviour():
    ours = make()
    rq = (2000.0, 2000.0, 2000.0)  # degenerate and far above the spacing: p = 1
    v_new = ours.v_wait(B, 2000.0, E, INTERVAL, 1, rq)
    # the pre-fix recursion: next partial at R - interval, elapsed + interval
    f0 = estimated_frac(E, 2000.0, False)
    r1, e1 = 2000.0 - INTERVAL, E + INTERVAL
    f1 = estimated_frac(e1, r1, False)
    v_old = sum(p * max(ours._u_partial(b1, r1, e1), ours._v_end(b1, f1))
                for p, b1 in ours.transitions.next_dist(B, f0))
    assert v_new == pytest.approx(v_old)
    ours._memo = {}  # a point prediction (no quantiles) gives the same
    assert ours.v_wait(B, 2000.0, E, INTERVAL, 1, None) == pytest.approx(v_old)


@pytest.mark.parametrize("arch", ["commit", "speculative"])
def test_mixed_case_is_the_mixture(arch):
    ours = make(arch, gap=0.0)  # p = P(R > 400)
    rq = (100.0, 500.0, 900.0)
    dist = RemainingDist(*rq)
    p = dist.p_greater(INTERVAL)
    assert 0.0 < p < 1.0
    r_next = dist.conditional_median_above(INTERVAL) - INTERVAL
    f0 = estimated_frac(E, 500.0, False)
    e1 = E + INTERVAL
    f1 = estimated_frac(e1, max(0.0, r_next), False)
    if arch == "commit":
        v = ours.v_wait(B, 500.0, E, INTERVAL, 1, rq)
        nxt = sum(q * max(ours._u_partial(b1, r_next, e1), ours._v_end(b1, f1))
                  for q, b1 in ours.transitions.next_dist(B, f0))
        end = ours._v_end(B, f0)
    else:
        v = ours.v_wait_spec(B, 500.0, E, INTERVAL, 1, rq)
        nxt = sum(q * max(ours._option_value(b1, r_next, e1, INTERVAL)[0],
                          ours._no_dispatch_value(b1, r_next, e1, INTERVAL))
                  for q, b1 in ours.transitions.next_dist(B, f0))
        end = sum(q * max(ours._u_final(b2), 0.0) for q, b2 in ours.transitions.final_dist(B, f0))
    assert v == pytest.approx(p * nxt + (1.0 - p) * end)


def test_gap_makes_a_late_partial_possible():
    # with an 800 ms end gap, a partial 400 ms away still arrives before the
    # end-of-turn decision even when only 100 ms of speech remain
    ours = make(gap=800.0)
    branches, spacing = ours._branches(100.0, INTERVAL, (50.0, 100.0, 150.0))
    assert spacing == INTERVAL and branches == [(1.0, pytest.approx(100.0 - INTERVAL))]


def test_remaining_dist():
    d = RemainingDist(100.0, 500.0, 900.0)
    assert d.cdf(100.0) == pytest.approx(0.1) and d.cdf(500.0) == pytest.approx(0.5)
    assert d.cdf(900.0) == pytest.approx(0.9) and d.cdf(-1e9) == 0.0 and d.cdf(1e9) == 1.0
    assert d.quantile(0.5) == pytest.approx(500.0)
    assert d.p_greater(500.0) == pytest.approx(0.5)
    pt = RemainingDist.point(300.0)
    assert pt.p_greater(299.0) == 1.0 and pt.p_greater(300.0) == 0.0
