"""
ours.py — Experiment 3's UNIFIED controller ("Ours"), the system the
headline experiment is about.

THE CLAIM THIS FILE IMPLEMENTS (H3, per paper/preregistration.md's
Experiment 3 and E3_BASELINE_INTERFACE.md): a controller that reasons about
the retrieve decision and the timing decision JOINTLY, under one objective
and one shared state, beats one that optimizes them separately against the
same prices (B9, controller/baselines/b9_decomposed.py).

Both baselines have the same objective, the same price point, the same
value-of-information estimate, and the same latency/compute constants —
all of them imported unchanged from controller/baselines/_objective.py,
precisely so that neither side gets an incidental advantage. The ONLY
difference is in this file versus that one: how the two decisions are
composed.

---------------------------------------------------------------------------
WHAT "JOINT" MEANS HERE, CONCRETELY
---------------------------------------------------------------------------
Ours does not decide "should I retrieve?" and then, separately, "when?".
It enumerates the full cross product of both decisions as CANDIDATE
ACTIONS, scores every candidate under one objective, and takes the argmax:

    candidates = {skip} U {retrieve} x {sentence_end, immediate,
                                        fixed_early_proxy}

    U(retrieve, t) = voi * (1 - partial_query_risk(t))          <- quality
                     - lambda_l * exposed_latency_s(t)          <- latency
                     - lambda_c * (1 + waste_rate(t))           <- compute
    U(skip)        = 0

The shared state is the candidate itself. Both terms that involve timing
sit inside ONE expression that also involves the retrieve decision, so
neither can be evaluated without the other. That's the coupling, and it
runs through two independent channels:

  CHANNEL 1 (latency): exposed_latency_s(t) = retrieval_latency - overlap(t).
    Retrieval launched during speech is partly hidden from the user, so
    retrieval's latency COST depends on the timing chosen alongside it.
    Measured overlap: sentence_end 0 ms, immediate 50 ms,
    fixed_early_proxy 188 ms (see _objective.py).

  CHANNEL 2 (quality): voi * (1 - partial_query_risk(t)).
    Retrieval launched early runs on a partial transcript, so retrieval's
    quality VALUE also depends on the timing chosen alongside it. Without
    this channel the controller would trivially always pick the
    highest-overlap policy and "joint reasoning" would be a one-line
    max(); with it, the timing choice is a real trade-off that the
    retrieve decision has a stake in.

---------------------------------------------------------------------------
THE PREDICTION THAT DISTINGUISHES OURS FROM B9
---------------------------------------------------------------------------
B9's retrieval optimizer prices retrieval at its FULL latency, because it
has no access to the timing decision (that's what decomposition means).
So the two retrieve when:

    B9   retrieves iff  voi > lambda_l * 0.672 + lambda_c
    Ours retrieves iff  voi > lambda_l * 0.484 + lambda_c    (via
                              fixed_early_proxy: 0.672 - 0.188 overlap,
                              ignoring the small risk/waste terms)

which leaves a band

    lambda_l * 0.484 + lambda_c  <  voi  <  lambda_l * 0.672 + lambda_c

where B9 skips a retrieval that Ours correctly takes early — net-positive
value that decomposition cannot see. The band's width is lambda_l * 0.188
(exactly the measured overlap), so the effect is PRICE-DEPENDENT by
construction: wide at high lambda_l, negligible at low lambda_l. That is
why E3 sweeps a price grid rather than reporting one number, and it makes
the claim falsifiable — if measured speech overlap collapses, so does the
predicted advantage. Run this module directly (see __main__) to print the
band at any price point.

---------------------------------------------------------------------------
MEMORY (Experiment 4)
---------------------------------------------------------------------------
Not implemented — see _objective.py's decide_memory(). The structure is
open for it: JointCandidate already carries a memory_op field, and adding
memory means extending the candidate enumeration in _enumerate_candidates()
with the memory dimension, leaving the scoring and argmax below untouched.
That is deliberate; E4 being conditional shouldn't mean rewriting the
controller whose result the paper leads with.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baseline import BaselineController, Ctx, Decision, register
from controller.baselines._objective import (
    Prices,
    decide_memory,
    exposed_latency_s,
    load_e3_config,
    partial_query_risk,
    read_prices,
    retrieval_compute_units,
    scalarized_utility,
    voi_estimate,
)

# The timing policies Ours chooses among when it retrieves — controller/
# timing.py's VALID_POLICIES, ordered cheapest-to-earliest so that ties in
# utility resolve toward the more conservative (later, lower-risk) policy.
# Ties are not hypothetical: at lambda_l = 0 the latency term vanishes and
# every timing scores identically on that channel, and picking the most
# aggressive policy on a coin flip would misrepresent what the controller
# actually believes at that price.
_TIMING_CANDIDATES = ("sentence_end", "immediate", "fixed_early_proxy")


@dataclass(frozen=True)
class JointCandidate:
    """
    One complete joint action Ours scores as a unit.

    The point of scoring a CANDIDATE rather than a sequence of independent
    choices is that a candidate's cost and value are computed from all of
    its fields together — see score_candidate(). This is the data
    structure the joint-vs-decomposed distinction actually lives in.

    memory_op is present and always None today (Experiment 4 is not
    implemented — see _objective.decide_memory). It's here so the
    candidate space grows by one dimension rather than by a redesign.
    """

    retrieve: bool
    timing: str
    memory_op: Optional[str] = None


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate plus its utility and the term-by-term breakdown that
    produced it — the breakdown is kept because "why did Ours pick this"
    is what the Decision.reason string has to answer for anyone auditing a
    sweep log."""

    candidate: JointCandidate
    utility: float
    quality: float
    latency_s: float
    compute_units: float


def _enumerate_candidates() -> List[JointCandidate]:
    """
    The joint action space: skip, plus retrieve x every timing policy.

    Skip appears once, with timing="sentence_end". Timing is genuinely
    moot when nothing is retrieved, and sentence_end is Decision's honest
    "no early-timing behavior" value (same reasoning as B0's, see
    controller/baselines/b0_never_retrieve.py) — enumerating three
    identical skip candidates would just add ties to break.

    Experiment 4 extends this function and nothing else: the memory
    dimension multiplies into the cross product here, and score_candidate()
    gains a memory term.
    """
    candidates = [JointCandidate(retrieve=False, timing="sentence_end")]
    candidates.extend(
        JointCandidate(retrieve=True, timing=timing) for timing in _TIMING_CANDIDATES
    )
    return candidates


def score_candidate(
    candidate: JointCandidate, voi: float, prices: Prices, cfg: dict
) -> ScoredCandidate:
    """
    Score one joint candidate under the shared objective.

    THIS FUNCTION IS THE HYPOTHESIS. Both timing-dependent terms are
    computed here, in one place, from the same candidate that carries the
    retrieve decision — so retrieval's latency cost and retrieval's
    quality value are BOTH conditioned on the timing chosen with it:

        quality = voi * (1 - partial_query_risk(timing))
        latency = exposed_latency_s(timing)
                = retrieval_latency - overlap(timing)
        compute = 1 + waste_rate(timing)

    Compare b9_decomposed.py's _optimize_retrieval(), which computes its
    latency term as the full retrieval_latency_s() with no timing argument
    available to it at all.

    Skip is scored at exactly zero on all three terms: no quality gained,
    no retrieval latency incurred, no retrieval compute spent. Zero is the
    reference point the retrieve candidates must beat, which is what makes
    the price point meaningful — raise lambda_l or lambda_c enough and
    every retrieve candidate falls below skip.
    """
    if not candidate.retrieve:
        return ScoredCandidate(
            candidate=candidate,
            utility=0.0,
            quality=0.0,
            latency_s=0.0,
            compute_units=0.0,
        )

    quality = voi * (1.0 - partial_query_risk(candidate.timing, cfg))
    latency_s = exposed_latency_s(candidate.timing, cfg)
    compute_units = retrieval_compute_units(candidate.timing, cfg)
    return ScoredCandidate(
        candidate=candidate,
        utility=scalarized_utility(quality, latency_s, compute_units, prices),
        quality=quality,
        latency_s=latency_s,
        compute_units=compute_units,
    )


def rank_candidates(ctx: Ctx, cfg: Optional[dict] = None) -> List[ScoredCandidate]:
    """
    Every joint candidate, scored and sorted best-first. Exposed
    separately from decide() so a sweep-analysis or a unit test can
    inspect the whole joint surface — e.g. how far the chosen candidate
    beat the runner-up, or at what price the argmax flips — without
    re-deriving the scoring.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    prices = read_prices(ctx, cfg)
    voi = voi_estimate(ctx, cfg)
    scored = [score_candidate(c, voi, prices, cfg) for c in _enumerate_candidates()]
    # Stable sort on utility alone: _enumerate_candidates() already emits
    # skip first and then timings cheapest-to-earliest, so an exact tie
    # keeps that conservative order (see _TIMING_CANDIDATES).
    return sorted(scored, key=lambda s: s.utility, reverse=True)


@register("ours")
class UnifiedController(BaselineController):
    """
    Ours: one decide() that picks (retrieve, timing) together as the argmax
    of the shared scalarized objective over the joint candidate space.

    Stateless across turns — the "shared state" the hypothesis is about is
    shared between the two DECISIONS within a turn (they're scored as one
    candidate), not accumulated across turns. That keeps decide() a pure
    function of ctx as controller/baseline.py requires, and keeps the
    comparison against B9 clean: B9 is stateless in exactly the same way,
    so the contrast is joint-vs-decomposed and not stateful-vs-stateless.
    """

    def decide(self, ctx: Ctx) -> Decision:
        cfg = load_e3_config()
        ranked = rank_candidates(ctx, cfg)
        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        prices = read_prices(ctx, cfg)
        voi = voi_estimate(ctx, cfg)

        # The reason string is what someone auditing a sweep log reads to
        # reconstruct this decision, so it carries the inputs (voi, the
        # price point), the chosen candidate's utility, and the margin over
        # the runner-up. The margin matters specifically for defending the
        # B9 contrast: a decision Ours won by 0.001 is not evidence of
        # anything, and this makes that visible in the log rather than
        # requiring a re-run to discover.
        action = "retrieve" if best.candidate.retrieve else "skip"
        margin = (best.utility - runner_up.utility) if runner_up else float("nan")
        reason = (
            f"Ours(joint): {action}@{best.candidate.timing} "
            f"U={best.utility:+.3f} (voi={voi:.3f}, "
            f"q={best.quality:.3f}, lat={best.latency_s:.3f}s, "
            f"comp={best.compute_units:.2f}) "
            f"beats {_describe(runner_up)} by {margin:+.3f} "
            f"@ lambda_l={prices.lambda_l:g}, lambda_c={prices.lambda_c:g}"
        )

        return Decision(
            retrieve=best.candidate.retrieve,
            timing=best.candidate.timing,
            reason=reason,
            # Experiment 4's hook — returns None today. See
            # _objective.decide_memory() for why it's wired now.
            memory_op=decide_memory(ctx),
        )


def _describe(scored: Optional[ScoredCandidate]) -> str:
    """Short label for a candidate in the reason string."""
    if scored is None:
        return "(none)"
    if not scored.candidate.retrieve:
        return "skip"
    return f"retrieve@{scored.candidate.timing}"


if __name__ == "__main__":
    # Standalone check, same spirit as controller/decide.py's and B0's
    # __main__ blocks — and additionally prints the Ours-vs-B9 divergence
    # band described in this module's docstring, since that band is the
    # quantitative form of the hypothesis and is worth being able to quote
    # without running the full sweep.
    from controller.baselines._objective import overlap_s, retrieval_latency_s

    cfg = load_e3_config()
    baseline = UnifiedController()

    def make_ctx(max_entropy_nats, unknown_fraction, prices):
        return Ctx(
            transcript="what time does the pool close",
            signals={
                "model": {
                    "available": True,
                    "uncertainty": {"max_entropy_nats": max_entropy_nats},
                },
                "tracker": {"unknown_fraction": unknown_fraction},
            },
            elapsed_ms=0,
            user_end_known=False,
            turn_id=1,
            conversation_id="conv_ours_selftest",
            prices=prices,
        )

    print("=" * 72)
    print("Ours — joint candidate scoring")
    print("=" * 72)
    for label, entropy, unknown in [
        ("high uncertainty", 2.2, 0.75),
        ("mid uncertainty ", 1.1, 0.50),
        ("low uncertainty ", 0.2, 0.00),
    ]:
        ctx = make_ctx(entropy, unknown, {"lambda_l": 1.0, "lambda_c": 0.05})
        print(f"\n{label} (voi={voi_estimate(ctx, cfg):.3f}):")
        for scored in rank_candidates(ctx, cfg):
            print(f"   {_describe(scored):<28} U={scored.utility:+.4f}")
        print(f"   -> {baseline.decide(ctx).reason}")

    print()
    print("=" * 72)
    print("Ours-vs-B9 divergence band (see module docstring)")
    print("=" * 72)
    full_s = retrieval_latency_s(cfg)
    best_timing = max(_TIMING_CANDIDATES, key=lambda t: overlap_s(t, cfg))
    hidden_s = overlap_s(best_timing, cfg)
    print(f"full retrieval latency : {full_s:.3f}s")
    print(f"best-overlap timing    : {best_timing} (hides {hidden_s:.3f}s)")
    for lambda_l in (0.1, 0.5, 1.0, 2.0, 5.0):
        lambda_c = 0.05
        b9_cut = lambda_l * full_s + lambda_c
        ours_cut = lambda_l * (full_s - hidden_s) + lambda_c
        print(
            f"  lambda_l={lambda_l:<4g} lambda_c={lambda_c:g} -> "
            f"B9 retrieves iff voi>{b9_cut:.3f}, "
            f"Ours iff voi>{ours_cut:.3f}  "
            f"[band width {b9_cut - ours_cut:.3f}]"
        )
