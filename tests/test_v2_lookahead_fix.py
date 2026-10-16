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


def test_gap_does_not_enter_p():
    # implementation correction (2026-10-16): p = P(remaining SPEECH > spacing);
    # the end gap only enters the exposure. 100 ms of speech left, spacing
    # 400 ms: no information-carrying partial can follow, whatever the gap.
    for gap in (0.0, 800.0, 5000.0):
        ours = make(gap=gap)
        branches, spacing = ours._branches(100.0, INTERVAL, (50.0, 100.0, 150.0))
        assert spacing == INTERVAL and branches == [(1.0, None)]


def _asr_like_turn():
    """Speech ends at 2700 ms; partials every 300 ms keep coming after it
    (post-speech re-transcriptions at 3000 and 3300 ms); the predictor's
    quantiles are the truth +- 150 ms; end of turn at 3800 ms."""
    from controller.sequential_v2 import TurnCtx
    speech_end, end = 2700.0, 3800.0
    partials, preds, by_prefix = [], [], []
    for k in range(11):
        t = 300.0 * (k + 1)
        r = speech_end - t
        partials.append({"t_ms": t, "available_ms": t, "text": "what is the refund policy for orders"})
        preds.append({"t_ms": t, "available_ms": t, "remaining_ms": r, "remaining_q10_ms": r - 150.0,
                      "remaining_q90_ms": r + 150.0})
        by_prefix.append({"t_ms": t, "word_frac": min(1.0, t / speech_end), "p_needed": 0.5})
    by_prefix.append({"t_ms": end, "word_frac": 1.0, "kind": "full", "p_needed": 0.5})
    timing = {"speech_onset_ms": 0, "speech_end_ms": speech_end, "user_end_ms": end,
              "partials": partials, "asr_final": {"t_ms": end, "available_ms": end, "text": "x"},
              "partial_interval_ms": 300.0, "predicted_remaining_speech_ms": preds}
    return TurnCtx(item_id="asr", signals={"timing": timing,
                                           "model": {"available": True, "uncertainty_by_prefix": by_prefix}})


def test_p_below_half_at_the_last_pre_end_partial():
    from controller.sequential_v2 import build_steps
    ours = make(gap=1100.0)
    cfg = ours.cfg
    steps = [o for o in build_steps(_asr_like_turn(), cfg) if not o.is_final]
    ps = {}
    for o in steps:
        branches, _ = ours._branches(o.remaining_ms, o.partial_interval_ms, o.remaining_quantiles,
                                     o.scheduled_partials_left)
        ps[o.available_ms] = sum(pr for pr, r in branches if r is not None)
    assert ps[2700] < 0.5  # the last partial cut before speech ends (R = 0)
    assert ps[2400] <= 0.5 + 1e-9  # R = 300 ms = one spacing: at most a coin flip
    assert ps[1200] > 0.95  # mid-utterance: more speech is near-certain
    assert ps[3000] < 0.01 and ps[3300] < 0.01  # post-speech: no new information
    # the aged quantiles are not clipped: they keep falling after speech end
    post = next(o for o in steps if o.available_ms == 3300)
    assert post.remaining_quantiles[1] == pytest.approx(-600.0)


def test_post_speech_partials_are_identity_transitions():
    # in the transition fit a post-speech partial is merged into the step
    # before it (identity); in the lookahead p = 0 from it, so waiting is
    # worth exactly the end-of-turn decision value from the CURRENT belief
    import math as _m

    from simulator.e3v2_dev_analysis import SIGNAL_X, belief_sequences
    cal = {"calibration": {"intercept": 0.0, "slope": 1.0, "frac_slope": 0.0, "interaction": 0.0}}
    by_prefix = [{"t_ms": t, "kind": "partial", "word_frac": f, SIGNAL_X: x}
                 for t, f, x in ((500, 0.3, -1.0), (1000, 0.8, 0.5), (1600, 1.0, 2.0), (1900, 1.0, 3.0))]
    by_prefix.append({"t_ms": 2500, "kind": "full", "word_frac": 1.0, SIGNAL_X: 1.0})
    row = {"signals": {"timing": {"speech_end_ms": 1200.0}, "model": {"uncertainty_by_prefix": by_prefix}}}
    seq = belief_sequences([row], cal)[0]
    sig = lambda x: 1 / (1 + _m.exp(-x))  # noqa: E731
    assert [round(b, 9) for _, b in seq] == [round(sig(x), 9) for x in (-1.0, 0.5, 1.0)]
    ours = make(gap=1100.0)
    rq = (-700.0, -600.0, -500.0)  # predicted end of speech already passed
    v = ours.v_wait(B, -600.0, E, INTERVAL, 3, rq)
    assert v == pytest.approx(ours._v_end(B, estimated_frac(E, -600.0, False)))


@pytest.mark.parametrize("arch", ["commit", "speculative"])
def test_text150_last_prefix_has_p_zero(arch):
    from controller.sequential_v2 import TurnCtx, build_steps
    ours = make(arch, gap=800.0)
    partials = [{"t_ms": t, "text": "what is the refund policy"} for t in (900.0, 1500.0, 2100.0)]
    preds = [{"t_ms": t, "remaining_ms": 3000.0 - t, "remaining_q10_ms": 2500.0 - t,
              "remaining_q90_ms": 3500.0 - t} for t in (900.0, 1500.0, 2100.0)]
    timing = {"speech_onset_ms": 0, "speech_end_ms": 3000.0, "user_end_ms": 3800.0, "partials": partials,
              "asr_final": {"t_ms": 3800.0, "text": "x"}, "partial_interval_ms": 600.0,
              "predicted_remaining_speech_ms": preds, "endpoint_reason": "text_timeline",
              "partial_schedule": "fixed"}
    steps = [o for o in build_steps(TurnCtx(item_id="t", signals={"timing": timing}), ours.cfg) if not o.is_final]
    assert [o.scheduled_partials_left for o in steps] == [2, 1, 0]
    first, last = steps[0], steps[-1]
    b_first, _ = ours._branches(first.remaining_ms, 600.0, first.remaining_quantiles, first.scheduled_partials_left)
    b_last, _ = ours._branches(last.remaining_ms, 600.0, last.remaining_quantiles, last.scheduled_partials_left)
    assert [pr for pr, _ in b_first] == [1.0] and b_last == [(1.0, None)]
    # 900 ms of speech remain at the last prefix, yet the next step is the end of turn
    if arch == "commit":
        v = ours.v_wait(B, last.remaining_ms, 2100.0, 600.0, 3, last.remaining_quantiles, 0)
        assert v == pytest.approx(ours._v_end(B, estimated_frac(2100.0, last.remaining_ms, False)))


def test_remaining_dist():
    d = RemainingDist(100.0, 500.0, 900.0)
    assert d.cdf(100.0) == pytest.approx(0.1) and d.cdf(500.0) == pytest.approx(0.5)
    assert d.cdf(900.0) == pytest.approx(0.9) and d.cdf(-1e9) == 0.0 and d.cdf(1e9) == 1.0
    assert d.quantile(0.5) == pytest.approx(500.0)
    assert d.p_greater(500.0) == pytest.approx(0.5)
    pt = RemainingDist.point(300.0)
    assert pt.p_greater(299.0) == 1.0 and pt.p_greater(300.0) == 0.0
