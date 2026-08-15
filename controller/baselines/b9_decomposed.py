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
WHY B9 STILL LOSES WHEN THE TIMING SUB-DECISION IS TRIVIAL
---------------------------------------------------------------------------
partial_query_risk is MEASURED at 0 for every policy (see
_objective.partial_query_risk(): E2's per-policy hit@3 is identical,
250/250 each, and identical per-instance at k=1 too). So the quality
coupling channel is empirically inactive, and _optimize_timing() below
reduces to "take the most overlap unless lambda_c's waste term outprices
it" — a nearly trivial argmax that B9 gets RIGHT, picking the same timing
Ours picks at every sane price.

That does not rescue B9, and the reason is the whole point of E3. The
coupling that matters here is not WHICH timing but WHETHER TO RETRIEVE:

    _optimize_retrieval prices the retrieval at 0.672s.
    _optimize_timing then selects fixed_early_proxy, which would have
    hidden 0.188s of it.

The saving is real, it is available, and B9's own timing component
identifies it — one function call after the retrieval component already
declined the retrieval on the assumption that no such saving existed.
Every VoI in

    lambda_l * 0.484 + 1.08 lambda_c  <  voi  <=  lambda_l * 0.672 + lambda_c

is a retrieval with positive net value that B9 skips and Ours takes. So a
trivial timing sub-decision SHARPENS the E3 claim rather than weakening
it: it rules out "Ours simply times better" and leaves the retrieve/skip
threshold shift as the only mechanism in play.

decide() below tags exactly those turns with a greppable
DIVERGENCE_BAND[b9-declines] marker — the mirror of ours.py's
DIVERGENCE_BAND[ours-takes] on the same VoI — so the divergence is read
directly off a sweep log from both sides. That annotation is computed
strictly AFTER both sub-optimizers have returned and feeds nothing; see
decide() and _objective.divergence_band() on why that ordering is
load-bearing rather than stylistic.

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
    divergence_band,
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


def _band_note(
    retrieve: bool, timing: str, ctx: Ctx, prices: Prices, cfg: dict
) -> str:
    """
    B9's side of the retrieve/skip-boundary annotation — the mirror of
    ours.py's _band_note().

    DIAGNOSTIC ONLY, and called only after both sub-optimizers have
    returned. See decide()'s comment block for why that matters here
    specifically: this function reads timing-dependent quantities that B9's
    retrieval optimizer must not see.

    What it makes visible, which is the thing E3 would otherwise force a
    reader to reverse-engineer: B9 priced this retrieval at the FULL
    retrieval latency, while its own timing component selected a policy
    that would have hidden part of it. When that gap is what flipped the
    decision to skip, the line carries

        DIVERGENCE_BAND[b9-declines]

    matching DIVERGENCE_BAND[ours-takes] in ours.py's log on the same VoI,
    so one grep over a sweep recovers both sides of every divergence.
    """
    band = divergence_band(prices, cfg)
    voi = voi_estimate(ctx, cfg)
    full_s = retrieval_latency_s(cfg)
    hidden_s = overlap_s(timing, cfg)

    # What B9's retrieval optimizer actually charged — always the full
    # schedule, by construction. Logged next to the overlap its own timing
    # optimizer went on to select, because the difference between these two
    # numbers IS B9's incoherence, and it should be readable without
    # arithmetic.
    priced = (
        f"priced_lat={full_s:.3f}s (full schedule; its own timing-opt chose "
        f"{timing}, which hides {hidden_s:.3f}s — not credited to the "
        f"retrieve decision)"
    )

    if not retrieve and band.contains(voi):
        return (
            f"{priced} | {band.describe()} | "
            f"DIVERGENCE_BAND[b9-declines]: voi={voi:.3f} is in "
            f"({band.ours_cut:.3f},{band.b9_cut:.3f}] -> DECLINED here, but a "
            f"joint controller retrieves @{band.timing} at an exposed "
            f"{band.exposed_latency_s:.3f}s. The {hidden_s:.3f}s this run's "
            f"own timing-opt would have hidden was already forfeited by the "
            f"retrieve decision that priced it away"
        )
    if retrieve and voi > band.ours_cut:
        return f"{priced} | {band.describe()} | band:above (voi={voi:.3f}, Ours retrieves too)"
    if not retrieve:
        return f"{priced} | {band.describe()} | band:below (voi={voi:.3f}, Ours skips too)"
    # Retrieving at or below Ours's cut — the inverted-band case
    # (width <= 0), where Ours's early-launch waste outprices its hidden
    # latency. Named explicitly rather than left as an unlabeled "else",
    # since a sweep producing these is not a sweep of one-sided wins.
    return (
        f"{priced} | {band.describe()} | "
        f"DIVERGENCE_BAND[b9-takes-ours-skips]: voi={voi:.3f} <= Ours cut "
        f"{band.ours_cut:.3f} — band inverted (width {band.width:+.3f})"
    )


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

        # ------------------------------------------------------------------
        # DIAGNOSTIC ONLY — everything below this line is logging.
        #
        # Both sub-optimizers have already returned; `retrieve` and `timing`
        # are fixed and are not revised by anything here. That ordering is
        # load-bearing, not stylistic: _band_note() calls
        # _objective.divergence_band(), which reads overlap_s() — exactly
        # the timing-dependent quantity B9's retrieval optimizer is defined
        # not to have. Reading it BEFORE the retrieve decision, or letting
        # any of it reach that decision, would convert B9 into a joint
        # controller and destroy the experiment. It annotates the log; it
        # decides nothing.
        # ------------------------------------------------------------------
        reason = (
            f"B9(decomposed): {retrieval_why} | {timing_why} "
            f"@ lambda_l={prices.lambda_l:g}, lambda_c={prices.lambda_c:g}"
            f" | {_band_note(retrieve, timing, ctx, prices, cfg)}"
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

    # Print both sides' full reason strings for one divergent turn — the
    # log lines a sweep actually produces, so the DIVERGENCE_BAND tags are
    # visible here and not only in a results file.
    band = divergence_band(Prices(**price_point), cfg)
    print()
    print("=" * 78)
    print(f"Both log lines on one in-band turn — {band.describe()}")
    print("=" * 78)
    band_ctx = Ctx(
        transcript="what time does the pool close",
        # tracker-only signal, so voi_estimate() returns this value exactly
        signals={"tracker": {"unknown_fraction": (band.ours_cut + band.b9_cut) / 2.0}},
        elapsed_ms=0,
        user_end_known=False,
        turn_id=1,
        conversation_id="conv_b9_band_demo",
        prices=price_point,
    )
    print(f"\nvoi = {voi_estimate(band_ctx, cfg):.3f}\n")
    print(f"OURS: {ours.decide(band_ctx).reason}\n")
    print(f"B9  : {b9.decide(band_ctx).reason}")
