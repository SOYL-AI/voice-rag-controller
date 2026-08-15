"""
b9_decomposed.py — Experiment 3's B9 baseline: DECOMPOSED-WITH-PRICES.

Per E3_BASELINE_INTERFACE.md: "each component (trigger, timing, memory)
optimizes the objective separately against the shared prices — no shared
state between them. This is the strongest rival: coordinated by price, but
not jointly reasoned."

B9 is the rival the headline result is measured against, so the single most
important property of this file is that B9 is NOT A STRAWMAN. It gets, by
construction, everything Ours gets except joint reasoning:

  - the same scalarized objective        (_objective.scalarized_utility)
  - the same price point                 (_objective.read_prices, ctx.prices)
  - the same value-of-information estimate (_objective.voi_estimate)
  - the same measured latency/compute constants
  - a genuine optimizer on BOTH decisions, not a fixed rule on either

All of those are imported unchanged from controller/baselines/_objective.py
— the same module ours.py imports — rather than reimplemented here, so
there is no room for the two to drift apart on anything except the one
axis under test. E3_BASELINE_INTERFACE.md's fairness rule (handbook 29.3)
is explicit that a strawman rival invalidates the whole headline.

---------------------------------------------------------------------------
WHAT "DECOMPOSED" MEANS HERE, CONCRETELY
---------------------------------------------------------------------------
Two sub-optimizers, run independently, whose outputs are concatenated:

  _optimize_retrieval(ctx, prices, cfg) -> (retrieve, why)
  _optimize_timing(ctx, prices, cfg)    -> (timing, why)

The separation is enforced STRUCTURALLY, not by convention, and that's
deliberate — "we promise we didn't peek" is not a defensible claim in a
paper. Each is a module-level pure function taking only (ctx, prices, cfg).
Neither receives the other's return value, neither can call the other, and
decide() below does nothing but call both and assemble a Decision from the
results. There is no object, no shared mutable state, and no argument by
which one sub-decision could inform the other. Contrast ours.py's
score_candidate(), which takes a single JointCandidate carrying BOTH
decisions and computes terms that depend on both at once.

The consequence — and the reason B9 loses where it loses:

  _optimize_retrieval prices retrieval at its FULL latency
  (_objective.retrieval_latency_s(), 672 ms measured) because it has no
  timing to condition on. It cannot know that a retrieval launched early
  would have ~188 ms of that hidden inside the user's speech, because
  knowing that would require exactly the shared state B9 is defined not to
  have. So it systematically OVERPRICES retrieval, and skips retrievals
  whose true net value under a well-chosen timing is positive.

  _optimize_timing values earliness at the shared prices — but under an
  ASSUMED retrieval (its unconditional prior), not the actual retrieve
  decision, which it cannot see. So when it happens to choose a good early
  timing, that timing's latency saving was already forfeited: the retrieval
  optimizer had, separately and earlier, priced the retrieval as though no
  such saving existed. The two decisions are individually sensible and
  jointly incoherent. That incoherence IS the hypothesis.

Note that B9's timing optimizer being GOOD is what makes B9 strong. It is
not handicapped: at most price points it selects the same timing Ours does.
B9's loss comes from the retrieve/skip decision made without it, not from
picking a worse timing.

---------------------------------------------------------------------------
MEMORY (Experiment 4)
---------------------------------------------------------------------------
Not implemented — _objective.decide_memory() returns None. When it becomes
real, it slots in as a THIRD independent sub-optimizer alongside the two
below (that's what decomposition means for memory too), while Ours gains a
memory dimension inside its single joint argmax. The structural contrast
extends to three decisions without either file being restructured.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baseline import BaselineController, Ctx, Decision, register
from controller.baselines._objective import (
    Prices,
    decide_memory,
    load_e3_config,
    overlap_s,
    partial_query_risk,
    read_prices,
    retrieval_compute_units,
    retrieval_latency_s,
    scalarized_utility,
    voi_estimate,
    waste_rate,
)

# B9's timing sub-problem ranges over the same policies Ours does — an
# unequal action space would be a second difference between the two and
# would confound the comparison.
_TIMING_CANDIDATES = ("sentence_end", "immediate", "fixed_early_proxy")


# ---------------------------------------------------------------------------
# Sub-optimizer 1 — retrieve or not, priced at a FIXED latency schedule
# ---------------------------------------------------------------------------
def _optimize_retrieval(
    ctx: Ctx, prices: Prices, cfg: dict
) -> Tuple[bool, str]:
    """
    Decide retrieve/skip alone, under the shared objective and the shared
    prices — with retrieval's latency priced at the FIXED full schedule.

    Signature note (this is the load-bearing part): this function takes no
    timing argument, and there is nothing in ctx from which the timing
    decision could be recovered. That is not an oversight to be fixed —
    it's the definition of the baseline. B9's components share prices, not
    state, so this optimizer's only available price for a retrieval is its
    unconditional, full-schedule cost.

        U(retrieve) = voi
                      - lambda_l * retrieval_latency_s()   <- FULL 672 ms
                      - lambda_c * (1 + waste_rate(sentence_end))
        U(skip)     = 0

    Compare ours.py's score_candidate(), whose latency term is
    exposed_latency_s(timing) = retrieval_latency - overlap(timing). The
    difference between those two latency terms is the entire mechanism of
    the predicted E3 effect.

    The compute term uses sentence_end's waste rate (i.e. zero waste): with
    no timing decision in view, the honest unconditional assumption is the
    non-speculative launch, since a wasted early launch is a cost that only
    exists once someone has chosen to launch early. Assuming a higher waste
    rate here would make B9 skip MORE, which would flatter Ours — so the
    conservative choice for a fair comparison is the one that makes B9
    retrieve more readily, which is what this is.
    """
    voi = voi_estimate(ctx, cfg)
    latency_s = retrieval_latency_s(cfg)
    compute_units = retrieval_compute_units("sentence_end", cfg)
    utility = scalarized_utility(voi, latency_s, compute_units, prices)

    retrieve = utility > 0.0
    why = (
        f"retrieval-opt: U={utility:+.3f} "
        f"(voi={voi:.3f} - {prices.lambda_l:g}*{latency_s:.3f}s "
        f"- {prices.lambda_c:g}*{compute_units:.2f}) "
        f"{'>' if retrieve else '<='} 0 [full-latency schedule, no timing]"
    )
    return retrieve, why


# ---------------------------------------------------------------------------
# Sub-optimizer 2 — timing, chosen independently under the same prices
# ---------------------------------------------------------------------------
def _optimize_timing(ctx: Ctx, prices: Prices, cfg: dict) -> Tuple[str, str]:
    """
    Pick the timing policy alone, under the shared objective and the same
    prices — with no knowledge of whether a retrieval will actually happen.

    Signature note, again load-bearing: no retrieve argument, and no access
    to _optimize_retrieval's result. This optimizer therefore evaluates
    timing under an ASSUMED retrieval — its unconditional prior that a
    retrieval occurs — which is the strongest honest thing it can do:

        U(t) = - voi * partial_query_risk(t)      <- quality risked
               + lambda_l * overlap_s(t)          <- latency hidden
               - lambda_c * waste_rate(t)         <- compute wasted

    This is the same three-term trade-off ours.py scores, restricted to the
    terms that vary with timing. The constant terms (voi itself, the full
    retrieval latency, the base compute unit) are identical across every
    timing candidate and so drop out of an argmax over timing — dropping
    them is correct optimization, not a handicap, which is exactly why B9
    usually picks the SAME timing Ours does.

    What it cannot do is notice that its latency saving, lambda_l *
    overlap_s(t), is precisely the quantity _optimize_retrieval already
    assumed away when it priced retrieval at full latency. Reconciling
    those two facts requires evaluating both decisions in one expression,
    which is ours.py.

    Ties resolve toward the earlier entry in _TIMING_CANDIDATES (i.e. the
    more conservative policy) — max() returns the first maximal element,
    and the tuple is ordered cheapest-to-earliest. Matters at lambda_l = 0,
    where the latency channel vanishes entirely.
    """
    voi = voi_estimate(ctx, cfg)

    def timing_utility(timing: str) -> float:
        return (
            -voi * partial_query_risk(timing, cfg)
            + prices.lambda_l * overlap_s(timing, cfg)
            - prices.lambda_c * waste_rate(timing, cfg)
        )

    best = max(_TIMING_CANDIDATES, key=timing_utility)
    why = (
        f"timing-opt: {best} U={timing_utility(best):+.3f} "
        f"(hides {overlap_s(best, cfg):.3f}s, "
        f"risk={partial_query_risk(best, cfg):.2f}, "
        f"waste={waste_rate(best, cfg):.2f}) [assumed retrieval, "
        f"no retrieve decision]"
    )
    return best, why


@register("b9")
class DecomposedWithPrices(BaselineController):
    """
    B9: decomposed-with-prices. Runs the two independent sub-optimizers
    above and concatenates their answers.

    decide() deliberately contains no logic of its own beyond assembly. Any
    line here that combined the two results — adjusting the timing because
    of the retrieve decision, or vice versa — would silently convert B9
    into a joint controller and destroy the experiment. If you are editing
    this method, that is the invariant to preserve.
    """

    def decide(self, ctx: Ctx) -> Decision:
        cfg = load_e3_config()
        prices = read_prices(ctx, cfg)

        # Two independent calls. Neither result is passed to the other;
        # neither function can observe the other's output. See this
        # module's docstring on why the separation is structural.
        retrieve, retrieval_why = _optimize_retrieval(ctx, prices, cfg)
        timing, timing_why = _optimize_timing(ctx, prices, cfg)

        reason = (
            f"B9(decomposed): {retrieval_why} | {timing_why} "
            f"@ lambda_l={prices.lambda_l:g}, lambda_c={prices.lambda_c:g}"
        )

        return Decision(
            retrieve=retrieve,
            # The timing sub-optimizer's answer is reported as-is even when
            # the retrieval sub-optimizer said skip. Overriding it to
            # sentence_end here ("timing is moot if we don't retrieve")
            # would be a coordination between the two decisions — small,
            # but exactly the kind B9 is defined not to have. Logging what
            # each component actually decided also makes the decomposition
            # auditable in the sweep logs.
            timing=timing,
            reason=reason,
            # Experiment 4's third sub-optimizer slots in here; inert today.
            memory_op=decide_memory(ctx),
        )


if __name__ == "__main__":
    # Standalone check, and a direct side-by-side against Ours at one price
    # point — the contrast this file exists to create, printable without
    # running the full sweep.
    from controller.baselines.ours import UnifiedController

    cfg = load_e3_config()
    b9 = DecomposedWithPrices()
    ours = UnifiedController()

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
            conversation_id="conv_b9_selftest",
            prices=prices,
        )

    price_point = {"lambda_l": 1.0, "lambda_c": 0.05}
    print("=" * 78)
    print(f"B9 vs Ours @ {price_point}")
    print("=" * 78)
    print(f"{'voi-ish input':<22} {'B9':<26} {'Ours':<26} agree?")
    for entropy, unknown in [
        (2.4, 0.9), (2.0, 0.75), (1.6, 0.6), (1.4, 0.5),
        (1.2, 0.45), (1.0, 0.35), (0.6, 0.2), (0.2, 0.0),
    ]:
        ctx = make_ctx(entropy, unknown, price_point)
        voi = voi_estimate(ctx, cfg)
        d9 = b9.decide(ctx)
        do = ours.decide(ctx)
        f9 = f"{'retrieve' if d9.retrieve else 'skip':<9}@{d9.timing}"
        fo = f"{'retrieve' if do.retrieve else 'skip':<9}@{do.timing}"
        agree = "yes" if d9.retrieve == do.retrieve else "*** DIVERGES ***"
        print(f"voi={voi:.3f} (H={entropy:<4g}) {f9:<26} {fo:<26} {agree}")
