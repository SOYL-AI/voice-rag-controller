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
neither can be evaluated without the other. That's the coupling. The
objective provides two channels for it to run through, and — this is
measured, not assumed — only one of them is active in this setup:

  CHANNEL 1 (latency): exposed_latency_s(t) = retrieval_latency - overlap(t).
    ACTIVE. Retrieval launched during speech is partly hidden from the
    user, so retrieval's latency COST depends on the timing chosen
    alongside it. Measured overlap: sentence_end 0 ms, immediate 50 ms,
    fixed_early_proxy 188 ms (see _objective.py).

  CHANNEL 2 (quality): voi * (1 - partial_query_risk(t)).
    EMPIRICALLY INACTIVE. partial_query_risk is MEASURED at 0 for all
    three policies — E2's per-policy hit@3 is identical (250/250 each),
    and identical per-instance at k=1 too, so this is not a saturation
    artifact. See _objective.partial_query_risk() for the fit, the
    reproduction command, and the two properties of E2's setup responsible
    (a frozen precomputed query, plus timing.py's query_ready gate). The
    factor is therefore identically 1 here, and the term is retained
    parameterized because it would be nonzero on harder retrieval or on a
    live partial-transcript path.

This is a clean isolation, not a limitation. Because the quality channel
contributes nothing, any Ours-vs-B9 gap the sweep measures is attributable
to the latency channel and to nothing else — there is no quality effect
available to confound it.

---------------------------------------------------------------------------
*** THE PAPER CLAIM (H3) — the defense, written where the code is ***
---------------------------------------------------------------------------
Coupling matters even when the timing sub-decision is trivial, because
knowing timing is cheap shifts the retrieve/skip threshold. Ours accepts
borderline retrievals a decomposed controller, pricing latency in
isolation, declines — the band

    lambda_l * (retrieval_latency - overlap)  <  voi  <  lambda_l * retrieval_latency

The anticipated objection, and the answer:

  OBJECTION. "With risk measured at 0, picking a timing is trivial — both
  controllers just take the highest-overlap policy. So B9 chooses the same
  timing as Ours, and there is nothing left for joint reasoning to do."

  ANSWER. The premise is right and the conclusion does not follow. B9 does
  pick the same timing (see b9_decomposed.py: its timing optimizer is
  GOOD, which is what makes B9 a strong rival). The effect does not live
  in WHICH timing gets picked. It lives in WHETHER TO RETRIEVE AT ALL.
  B9's retrieval optimizer prices retrieval at the full 0.672s because it
  structurally cannot see the timing decision, so it declines retrievals
  whose true exposed cost is 0.484s. Its own timing optimizer then goes on
  to select the very policy that would have hidden that 0.188s — a saving
  already forfeited by a decision taken separately and earlier. The two
  sub-decisions are individually optimal and jointly incoherent, and the
  incoherence is a skipped retrieval, not a mistimed one.

  So the trivial timing sub-decision does not weaken H3; it sharpens it.
  It removes the alternative explanation ("Ours just times better") and
  leaves exactly one mechanism standing.

---------------------------------------------------------------------------
THE PREDICTION THAT DISTINGUISHES OURS FROM B9
---------------------------------------------------------------------------
B9's retrieval optimizer prices retrieval at its FULL latency, because it
has no access to the timing decision (that's what decomposition means).
So the two retrieve when:

    B9   retrieves iff  voi > lambda_l * 0.672 + lambda_c * 1.00
    Ours retrieves iff  voi > lambda_l * 0.484 + lambda_c * 1.08   (via
                              fixed_early_proxy: 0.672 - 0.188 overlap;
                              risk is 0 so it drops out exactly, and the
                              compute terms differ only by waste_rate)

which leaves a band

    lambda_l * 0.484 + 1.08 lambda_c  <  voi  <=  lambda_l * 0.672 + lambda_c

where B9 skips a retrieval that Ours correctly takes early — net-positive
value that decomposition cannot see. As lambda_c -> 0 this is exactly the
headline form in the PAPER CLAIM block above, of width lambda_l * 0.188
(exactly the measured overlap), so the effect is PRICE-DEPENDENT by
construction: wide at high lambda_l, vanishing at lambda_l = 0. That is
why E3 sweeps a price grid rather than reporting one number, and it makes
the claim falsifiable — if measured speech overlap collapses, so does the
predicted advantage.

_objective.divergence_band() computes this band exactly (it carries the
risk and waste terms rather than dropping them), and decide() below tags
every decision that lands inside it with a greppable DIVERGENCE_BAND
marker, so the retrieve/skip divergence is READ off a sweep log rather
than reconstructed from it. Run this module directly (see __main__) to
print the band at a range of price points.

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
    divergence_band,
    exposed_latency_s,
    load_e3_config,
    partial_query_risk,
    read_prices,
    retrieval_compute_units,
    retrieval_latency_s,
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
            f" | {_band_note(best, voi, prices, cfg)}"
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


def _band_note(
    best: ScoredCandidate, voi: float, prices: Prices, cfg: dict
) -> str:
    """
    The retrieve/skip-boundary annotation appended to every reason string.

    This exists so E3's central claim is READ off a sweep log rather than
    reverse-engineered from one. It always reports the exposed-vs-full
    latency split and the band bounds at this price point, and when the
    decision actually falls in the band it emits the token

        DIVERGENCE_BAND[ours-takes]

    which b9_decomposed.py mirrors as DIVERGENCE_BAND[b9-declines] on the
    same voi. So `grep DIVERGENCE_BAND` over a sweep's logs returns exactly
    the turns where joint reasoning changed the outcome, from both sides.

    Purely diagnostic: computed after the argmax, and nothing here is fed
    back into the decision (see _objective.divergence_band()).
    """
    band = divergence_band(prices, cfg)

    if not best.candidate.retrieve:
        if voi > band.b9_cut:
            # REVERSE divergence: B9 retrieves where Ours skips. Only
            # reachable when band.width <= 0, i.e. when the latency saving
            # no longer covers the extra waste_rate Ours pays for an early
            # launch (lambda_l at or near 0, or lambda_c large). Logged
            # under the same greppable token, with the direction named, so
            # a sweep that produces these does not look like a clean sweep
            # of one-sided wins when it isn't.
            return (
                f"{band.describe()} | DIVERGENCE_BAND[b9-takes-ours-skips]: "
                f"voi={voi:.3f} > B9 cut {band.b9_cut:.3f} but <= Ours cut "
                f"{band.ours_cut:.3f} — band inverted (width {band.width:+.3f}), "
                f"early launch's waste outprices its hidden latency"
            )
        # Below both cuts: neither retrieves, nothing divides the two here.
        # Logged anyway so a skip is distinguishable from a turn the
        # annotation simply didn't run on.
        return (
            f"exposed_lat=0.000s (skipped; would have been "
            f"{band.exposed_latency_s:.3f}s @{band.timing} vs B9's full "
            f"{band.full_latency_s:.3f}s) | {band.describe()} | "
            f"band:below (voi={voi:.3f}, B9 skips too)"
        )

    hidden_s = retrieval_latency_s(cfg) - best.latency_s
    split = (
        f"exposed_lat={best.latency_s:.3f}s "
        f"(full {retrieval_latency_s(cfg):.3f}s - hidden {hidden_s:.3f}s)"
    )

    if band.contains(voi):
        return (
            f"{split} | {band.describe()} | "
            f"DIVERGENCE_BAND[ours-takes]: voi={voi:.3f} is in "
            f"({band.ours_cut:.3f},{band.b9_cut:.3f}] -> B9 prices this "
            f"retrieval at the full {band.full_latency_s:.3f}s in isolation "
            f"and SKIPS it; Ours takes it @{best.candidate.timing} at an "
            f"exposed {band.exposed_latency_s:.3f}s"
        )
    return (
        f"{split} | {band.describe()} | "
        f"band:above (voi={voi:.3f} > {band.b9_cut:.3f}, B9 retrieves too)"
    )


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
    print("Ours-vs-B9 RETRIEVE/SKIP DIVERGENCE BAND — the H3 claim")
    print("=" * 72)
    print("Coupling matters even though the timing sub-decision is trivial")
    print("(partial_query_risk measured = 0): knowing timing is cheap shifts")
    print("the retrieve/skip threshold. Everything inside the band is a")
    print("retrieval Ours takes and B9 declines.")
    print()

    full_s = retrieval_latency_s(cfg)
    best_timing = max(_TIMING_CANDIDATES, key=lambda t: overlap_s(t, cfg))
    print(f"full retrieval latency : {full_s:.3f}s   (measured, logs/)")
    print(
        f"best-overlap timing    : {best_timing} "
        f"(hides {overlap_s(best_timing, cfg):.3f}s, measured)"
    )
    print(
        "quality channel        : partial_query_risk = "
        + ", ".join(f"{t}={partial_query_risk(t, cfg):.2f}" for t in _TIMING_CANDIDATES)
        + "  -> INACTIVE"
    )
    print()
    print(
        f"  {'lambda_l':>8} {'lambda_c':>8} | {'Ours cut':>9} {'B9 cut':>9} "
        f"{'width':>8} | timing"
    )
    print("  " + "-" * 62)
    for lambda_l in (0.1, 0.5, 1.0, 2.0, 5.0):
        lambda_c = 0.05
        band = divergence_band(Prices(lambda_l=lambda_l, lambda_c=lambda_c), cfg)
        print(
            f"  {lambda_l:>8g} {lambda_c:>8g} | {band.ours_cut:>9.3f} "
            f"{band.b9_cut:>9.3f} {band.width:>+8.3f} | {band.timing}"
        )
    print()
    print("  read: at this price, any turn with Ours-cut < voi <= B9-cut is one")
    print("  Ours retrieves and B9 skips. Those turns are tagged")
    print("  DIVERGENCE_BAND[ours-takes] in Ours's logs and")
    print("  DIVERGENCE_BAND[b9-declines] in B9's — grep either to find them.")

    # Demonstrate the tag firing on a voi placed inside the band, so the
    # self-test shows the actual log line and not just the arithmetic.
    print()
    print("=" * 72)
    print("The tag, fired on a voi placed inside the band")
    print("=" * 72)
    demo_prices = Prices(lambda_l=1.0, lambda_c=0.05)
    demo_band = divergence_band(demo_prices, cfg)
    target_voi = (demo_band.ours_cut + demo_band.b9_cut) / 2.0
    # voi_estimate() with only a tracker signal returns unknown_fraction
    # directly (clipped, floored), so this hits `target_voi` exactly.
    band_ctx = Ctx(
        transcript="what time does the pool close",
        signals={"tracker": {"unknown_fraction": target_voi}},
        elapsed_ms=0,
        user_end_known=False,
        turn_id=1,
        conversation_id="conv_ours_band_demo",
        prices={"lambda_l": demo_prices.lambda_l, "lambda_c": demo_prices.lambda_c},
    )
    print(f"voi = {voi_estimate(band_ctx, cfg):.3f} (midpoint of the band)")
    print(baseline.decide(band_ctx).reason)
