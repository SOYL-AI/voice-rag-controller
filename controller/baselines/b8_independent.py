"""
b8_independent.py — Experiment 3's B8 baseline: INDEPENDENT-BEST.

paper/preregistration.md's Experiment 3 calls this rival A: "three
independent best controllers sharing nothing". B8 is the best available
retrieval trigger and a separately-tuned timing policy, each tuned on its
own, bolted together — the strong-but-uncoordinated rival that sits between
the simple heuristics (B0-B7) and the priced controllers (B9, Ours).

THE DEFINING PROPERTY: B8 DOES NOT PRICE. It never reads ctx.prices, and it
imports nothing from controller/baselines/_objective.py that involves
lambda_l or lambda_c. Per E3_BASELINE_INTERFACE.md, "B0-B7 do NOT price
(they're fixed heuristics) — they appear as single points, not frontiers,
which is the intended contrast", and B8 shares that property: it is one
point on the accuracy-vs-latency plane regardless of the price grid, while
B9 and Ours each trace a curve across it. If you find yourself wanting to
read a price here, you are building B9, not B8.

This makes B8 the control for a specific alternative explanation: maybe
Ours wins simply because it has good components, not because it couples
them. B8 has good components and no coupling at all. B9 has good components
and price coupling without shared state. Ours has good components, price
coupling, AND shared state. The three together isolate which ingredient is
doing the work — which is why B8 has to be genuinely well-tuned rather
than a throwaway.

---------------------------------------------------------------------------
THE TWO INDEPENDENT COMPONENTS
---------------------------------------------------------------------------
1. Retrieval trigger — a threshold on the model's own uncertainty.

   *** PLUG POINT: replace with the best of B4-B6 once Lakshya's triggers
   *** land (retrieval/README.md lists them: B4 FLARE confidence, B5, B6
   *** DRAGIN entropy). B8 is by definition "the best independent trigger",
   *** so it should wrap whichever of those scores highest on the dev
   *** split, not keep its own private rule. See _retrieval_trigger() —
   *** swapping it is a one-function change and touches nothing else here.

   Placeholder until then: a threshold on max_entropy_nats, the E1 entropy
   family, read from the frozen per-scenario snapshot
   (simulator/e3_signals.py).

   STATUS OF THIS SIGNAL CHOICE: vindicated, as of the real E1 gate run
   (llm/experiment1/results/experiment_1_modal_mistral7b_20260811_151212.json,
   Mistral-7B on Modal, verdict PASS). Its best_signal is max_entropy_nats
   at mean AUC 0.978 — the signal this trigger already read, chosen before
   that run existed as a reasonable member of the entropy family E1 was
   designed to test. It is now the signal a real gate selected, and an AUC
   may be cited for it.

   The THRESHOLD is a different question and is still un-swept. 1.2 nats
   predates the run; the fitted calibration puts P(retrieval needed) = 0.5
   at 1.360 nats (controller/baselines/e1_calibration.json), which is the
   obvious candidate for the dev-split sweep below to start from or land
   on. Note B8 must keep reading a raw threshold rather than importing
   _objective's calibrated VoI: B8 is the no-price-coupling rival, and
   borrowing the priced controllers' estimator would blur the distinction
   it exists to draw.

2. Timing policy — one fixed, separately-tuned choice.

   fixed_early_proxy, which hides the most retrieval latency inside speech
   of the three E2 policies (mean overlap 188 ms vs immediate's 50 ms vs
   sentence_end's 0 ms, measured over 750 retrievals in logs/). Tuned in
   isolation on the timing objective alone — "which policy hides the most
   latency" — with no reference to the retrieve decision, to prices, or to
   the retrieval trigger above. That isolation is the point.

Both components' thresholds still need the dev-split sweep every baseline
owes under E3_BASELINE_INTERFACE.md's fairness rule (handbook 29.3). The
values in configs/config.yaml's controller.e3.b8 are where that sweep
starts, not its result.

---------------------------------------------------------------------------
MEMORY (Experiment 4)
---------------------------------------------------------------------------
Not implemented — _objective.decide_memory() returns None. For B8 it would
become a third independently-tuned component, consistent with "sharing
nothing".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baseline import BaselineController, Ctx, Decision, register

# NOTE what is and isn't imported: load_e3_config for the tunables and
# decide_memory for the inert E4 hook, and nothing else from _objective.
# No read_prices, no scalarized_utility, no exposed_latency_s — B8 is the
# no-pricing rival, and keeping the import list this short is the cheapest
# way to keep that property auditable.
from controller.baselines._objective import decide_memory, load_e3_config


def _retrieval_trigger(ctx: Ctx, cfg: dict) -> Tuple[bool, str]:
    """
    B8's independently-tuned retrieval trigger: retrieve when the model's
    own uncertainty on this question exceeds a threshold.

    *** THIS IS THE PLUG POINT. *** Replace the body with the best of
    B4-B6 (retrieval/README.md) once those land — B8 is defined as the
    best independent trigger, so it should delegate to whichever scores
    highest on the dev split rather than keeping this placeholder rule.
    The contract to preserve is only the signature: (ctx, cfg) ->
    (retrieve, why). Nothing else in this file needs to change.

    Reads signals["model"]["uncertainty"]["max_entropy_nats"] from the
    frozen snapshot, with .get() throughout per Ctx's contract. See this
    module's docstring for the honest status of that signal choice — it is
    not backed by a real E1 gate run yet.

    Fallbacks, in order, when max_entropy_nats isn't present:
      - signals["tracker"]["unknown_fraction"] vs its own threshold;
      - the raw tracker_slots rule (intent unknown -> retrieve), which is
        what controller/decide.py does and is always available because
        pipeline/spine.py always puts tracker_slots in Ctx.signals.
    A baseline that hard-required one signal would simply crash on a call
    site that hasn't wired it in, which Ctx's docstring explicitly warns
    against.
    """
    b8_cfg = cfg["b8"]
    signals = ctx.signals or {}

    uncertainty = (signals.get("model") or {}).get("uncertainty") or {}
    max_entropy = uncertainty.get("max_entropy_nats")
    if max_entropy is not None:
        threshold = float(b8_cfg["max_entropy_nats_threshold"])
        retrieve = float(max_entropy) >= threshold
        return retrieve, (
            f"max_entropy_nats={float(max_entropy):.3f} "
            f"{'>=' if retrieve else '<'} {threshold:g}"
        )

    tracker = signals.get("tracker") or {}
    unknown_fraction = tracker.get("unknown_fraction")
    if unknown_fraction is not None:
        threshold = float(b8_cfg["unknown_fraction_threshold"])
        retrieve = float(unknown_fraction) >= threshold
        return retrieve, (
            f"no entropy signal; tracker unknown_fraction="
            f"{float(unknown_fraction):.3f} "
            f"{'>=' if retrieve else '<'} {threshold:g}"
        )

    slots = signals.get("tracker_slots") or {}
    intent_unknown = slots.get("intent", "unknown") == "unknown"
    return intent_unknown, (
        f"no entropy/tracker signal; intent "
        f"{'unknown -> retrieve' if intent_unknown else 'known -> skip'}"
    )


def _tuned_timing(cfg: dict) -> str:
    """
    B8's separately-tuned timing policy — a single fixed choice, taking no
    ctx and no prices.

    The empty-ish signature is the point: this component was tuned on the
    timing objective in isolation (which policy hides the most latency —
    fixed_early_proxy, at a measured mean 188 ms of speech overlap) and it
    does not adapt per turn, per price, or per retrieve decision. That is
    what "independent best, sharing nothing" means. Contrast
    b9_decomposed.py's _optimize_timing(), which at least reprices per
    price point, and ours.py's joint scoring, which repicks per turn
    alongside the retrieve decision.
    """
    return cfg["b8"]["timing"]


@register("b8")
class IndependentBest(BaselineController):
    """
    B8: best independent retrieval trigger + separately tuned timing, no
    price coupling and no shared state between the two.

    decide() reads ctx.signals and nothing else. It never touches
    ctx.prices — see this module's docstring.
    """

    def decide(self, ctx: Ctx) -> Decision:
        cfg = load_e3_config()

        retrieve, trigger_why = _retrieval_trigger(ctx, cfg)
        timing = _tuned_timing(cfg)

        reason = (
            f"B8(independent): trigger[{trigger_why}] | "
            f"timing[{timing}, tuned separately] | no pricing"
        )

        return Decision(
            retrieve=retrieve,
            timing=timing,
            reason=reason,
            memory_op=decide_memory(ctx),
        )


if __name__ == "__main__":
    baseline = IndependentBest()

    def make_ctx(signals):
        return Ctx(
            transcript="what time does the pool close",
            signals=signals,
            elapsed_ms=0,
            user_end_known=False,
            turn_id=1,
            conversation_id="conv_b8_selftest",
            # Prices deliberately supplied and deliberately ignored — this
            # is the check that B8 really is price-blind. Its decision must
            # be identical with and without them.
            prices={"lambda_l": 99.0, "lambda_c": 99.0},
        )

    cases = [
        ("high entropy", {"model": {"uncertainty": {"max_entropy_nats": 2.2}}}),
        ("low entropy", {"model": {"uncertainty": {"max_entropy_nats": 0.3}}}),
        ("tracker only", {"tracker": {"unknown_fraction": 0.8}}),
        ("slots only", {"tracker_slots": {"intent": "unknown", "room": "unknown"}}),
        ("no signals", {}),
    ]
    for label, signals in cases:
        decision = baseline.decide(make_ctx(signals))
        print(f"{label:<14} -> retrieve={decision.retrieve!s:<5} "
              f"timing={decision.timing:<18} {decision.reason}")

    # Price-blindness check: same signals, wildly different prices, same answer.
    signals = {"model": {"uncertainty": {"max_entropy_nats": 1.5}}}
    priced = baseline.decide(make_ctx(signals))
    unpriced_ctx = Ctx(
        transcript="what time does the pool close",
        signals=signals,
        elapsed_ms=0,
        user_end_known=False,
        turn_id=1,
        conversation_id="conv_b8_selftest",
        prices={},
    )
    unpriced = baseline.decide(unpriced_ctx)
    same = (priced.retrieve, priced.timing) == (unpriced.retrieve, unpriced.timing)
    print(f"\nprice-blind: {'OK' if same else 'FAIL — B8 reacted to prices!'}")
