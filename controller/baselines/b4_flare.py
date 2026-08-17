"""
b0_never_retrieve.py — Experiment 3's B0 baseline: never retrieve.

*** THIS FILE IS THE REFERENCE TEMPLATE. Copy this structure for your own
*** baseline (B1-B9 / Ours). Everything below marked "COPY THIS" is the
*** part that's the same shape for every baseline; only the body of
*** decide() and the module-level constants above it are yours to change.

B0 is the simplest possible rival: it always skips retrieval, no matter
what the transcript or belief signals say. It exists as (a) a floor —
every other baseline needs to actually beat "do nothing" to be worth
building — and (b) the fully-worked example of the interface in
controller/baseline.py, since a real decide() with actual logic (e.g. "B1:
threshold on the tracker's unknown-slot fraction, tuned separately from
Ours") would bury the parts of the contract that are boilerplate.

Read controller/baseline.py's module docstring first if you haven't — it
defines Ctx, Decision, and BaselineController, all of which this file only
uses, never redefines.
"""

from __future__ import annotations

import sys
from pathlib import Path

# COPY THIS: same sys.path fixup controller/baseline.py, controller/
# decide.py, and controller/timing.py all use, so `python -m
# controller.baselines.b0_never_retrieve` and a plain module import both
# work regardless of what added the repo root to sys.path first.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# COPY THIS: pull in the shared contract, nothing else. A baseline should
# not need to import pipeline/, eventlog/, or another baseline's module —
# everything it's allowed to see arrives via the Ctx that decide() is
# called with.
from controller.baseline import BaselineController, Ctx, Decision, register
from controller.baselines._objective import load_e3_config



# COPY THIS: @register("<your baseline's name>") — the string every other
# component (pipeline/spine.py's baseline= arg, simulator/e3_sweep.py's
# --baseline flag) will use to select this class. Must be unique across
# every baseline module — see controller/baseline.py's register().
@register("b4")
class FlareBaseline(BaselineController):
    """
    
    B4: FLARE confidence baseline.

    Retrieves when the model's uncertainty exceeds a configurable threshold.
    Uses the precomputed belief signals available in `ctx.signals` to decide
    whether retrieval should be triggered. The threshold will be tuned during
    the Experiment 3 sweep rather than being hardcoded.

    """

    # COPY THIS: decide(self, ctx: Ctx) -> Decision — the exact signature
    # BaselineController declares as abstract. Nothing else on this class
    # is required; log_decision()/decide_and_log() are inherited as-is.
    def decide(self, ctx: Ctx) -> Decision:
        uncertainty = (
            ctx.signals.get("model", {})
            .get("uncertainty", {})
        )       
        uncertainty_score = uncertainty.get(
            "one_minus_mean_confidence",
            0.0,
        )
        cfg = load_e3_config()
        threshold = cfg["b4"]["threshold"]
        retrieve = uncertainty_score >= threshold

        reason = (
            "uncertainty above threshold"
            if retrieve
            else "uncertainty below threshold"
        )

        return Decision(
            retrieve=retrieve,
            timing="sentence_end",
            reason=reason,
        )


if __name__ == "__main__":
    # Small standalone sanity check, same spirit as controller/decide.py's
    # __main__ block: prove decide() and the logging helper both work
    # without needing the full pipeline running. Not a substitute for
    # simulator/e3_sweep.py, which exercises this through the real
    # pipeline end to end.
    baseline = FlareBaseline()
    ctx = Ctx(
        transcript="what time does the pool close",
        signals={},
        elapsed_ms=0,
        user_end_known=True,
        turn_id=1,
        conversation_id="conv_b4_selftest",
        prices={},
    )
    decision = baseline.decide(ctx)
    print(f"decide(ctx) -> {decision}")
    payload = baseline.log_decision(decision, ctx, log_writer=None)
    print(f"logged payload shape -> {payload}")
