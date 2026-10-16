"""
b11_speculative_gated_v2.py — SPEC-theta: belief-gated speculative prefetch,
the strongest DECOMPOSED rival under the speculative architecture
(THEORY.md §6.2, Proposition 4''). Requires
controller.e3v2.architecture = speculative.

Why it exists. Under speculation the joint controller (ours_v2 with
architecture: speculative) beats SPEC (b10) mainly by NOT prefetching on
turns whose belief says the passages will almost surely go unused, which
saves lambda_c (1 + w) on those turns. Deciding that needs only the belief
and fixed prices, so a decomposed controller can do it too. Without this
rival the speculative contrast of the boundary test would measure that
saving, not joint control (THEORY §6.2: the interaction is then predicted
NEGATIVE).

Two decisions, neither of which reads the turn's timing state:

  1. WHEN: SPEC's independently chosen trigger (controller.e3v2.b11.timing,
     default = b10's). Timing only.

  2. WHETHER to prefetch at the trigger: belief only, with dev-mean scalars.
     Prefetch iff

         E[ max(U_use(b_end; W_bar, rho_bar), 0) | b ] - lambda_c (1 + w_bar)
             >= E[ max(U_final(b_end), 0) | b ]          and the left side > 0,

     where W_bar, rho_bar, w_bar come from the dev-mean remaining and
     elapsed speech at the trigger (policy constants, the same artifact
     B9+ prices from), NOT from this turn. The expectations use the same
     belief-transition model as ours_v2. This is ours_v2's speculative
     option value evaluated at the scalars, i.e. exactly the gate that is
     optimal when exposure is replaced by its dev mean (Prop. 4'').

  3. USE at the end of turn: b10's rule (final belief, realised window).
     Without a prefetch it decides like any end-of-turn controller.

If the policy constants are unfitted (no remaining_ms for the trigger), the
gate credits no hiding (window 0), which is conservative and still
belief-only; the reason string says so.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    POLICIES_V2,
    estimated_frac,
    policy_fires,
    trigger_fires,
    trigger_label,
    resolve_policy_constants,
    rho_at,
    step_terms,
    waste_at,
)
from controller.baselines.ours_v2 import OursV2  # noqa: E402
from controller.sequential_v2 import (  # noqa: E402
    SPECULATE,
    WAIT,
    StepObs,
    StepTrace,
    TurnStatic,
)


class B11SpeculativeGatedV2(OursV2):
    priced = True

    def __init__(self, cfg: Optional[dict] = None, timing=None,
                 policy_constants: Optional[dict] = None, trigger_constants: Optional[dict] = None, **params):
        super().__init__(cfg=cfg, **params)
        # Buildable under any config (as b10 is); it refuses to DECIDE unless the
        # architecture is speculative, and e3v2_decide never runs it under commit.
        b11 = self.cfg.get("b11") or {}
        self.policy = timing or b11.get("timing") or self.cfg["b10"]["timing"]
        if not (self.policy in POLICIES_V2 or (isinstance(self.policy, dict)
                                               and self.policy.get("kind") == "r_below")):
            raise ValueError(f"b11 timing must be one of {POLICIES_V2} or an r_below trigger, got {self.policy!r}")
        # for an r_below trigger: {profile: {"remaining_ms", "elapsed_ms"}} dev
        # means at its firing step (controller/tune_v2.py:tune_spec_trigger)
        self.trigger_constants = trigger_constants or {}
        self.constants, self.constants_source = resolve_policy_constants(self.cfg, policy_constants)

    def begin_turn(self, static: TurnStatic) -> None:
        if not self.speculative:
            raise ValueError("b11_speculative_gated_v2 needs controller.e3v2.architecture: speculative")
        super().begin_turn(static)
        prof = (self.constants.get("profiles") or {}).get(self.lat.profile) or {}
        row = (prof.get(self.policy) if isinstance(self.policy, str)
               else self.trigger_constants.get(self.lat.profile)) or {}
        rem = row.get("remaining_ms")
        self.scalar_remaining = None if rem is None else float(rem)
        self.scalar_elapsed = float(row.get("elapsed_ms") or 0.0)

    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        if memory.get("spec") is not None:
            # passages in flight / use-or-discard: b10's rule (shared code path)
            return self._step_speculative(obs, memory)
        if memory.get("gated_off"):
            if not obs.is_final:
                return WAIT, StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT,
                                       remaining_ms=obs.remaining_ms, note="gate declined; waiting for end")
            return self._step_commit(obs, memory)
        if not trigger_fires(self.policy, obs, self.cfg, self.lat):
            return WAIT, StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT,
                                   remaining_ms=obs.remaining_ms, note=f"waiting for {trigger_label(self.policy)} trigger")
        if obs.is_final:
            return self._step_commit(obs, memory)

        # The gate: belief only, every timing quantity a dev scalar.
        terms = step_terms(obs, self.cfg, self.calibration, self.lat)
        b = terms.belief
        interval = obs.partial_interval_ms
        s_now, _, _ = self._option_value(b, self.scalar_remaining, self.scalar_elapsed, interval)
        v_no = self._no_dispatch_value(b, self.scalar_remaining, self.scalar_elapsed, interval)
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT, belief=b,
                           remaining_ms=obs.remaining_ms, u_retrieve=s_now, v_wait=v_no)
        scalars = (f"W_bar from remaining={self.scalar_remaining}ms elapsed={self.scalar_elapsed}ms "
                   f"[{self.constants_source}]")
        if s_now > 0 and s_now >= v_no:
            # the dispatch itself happens now, so its realised rho / waste are this step's
            frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
            memory["spec"] = {"t_ms": obs.available_ms,
                              "rho": rho_at(frac, obs.query_ready, self.cfg),
                              "waste": waste_at(frac, False, self.cfg)}
            record.rho, record.waste = memory["spec"]["rho"], memory["spec"]["waste"]
            record.note = (f"SPEC-theta gate @{self.policy}: b={b:.3f} S={s_now:+.3f} >= "
                           f"V_no_prefetch={v_no:+.3f} -> {SPECULATE} | {scalars}")
            return SPECULATE, record
        memory["gated_off"] = True
        record.note = (f"SPEC-theta gate @{self.policy}: b={b:.3f} S={s_now:+.3f} < "
                       f"V_no_prefetch={v_no:+.3f} -> no prefetch | {scalars}")
        return WAIT, record

    def timing_label(self, obs: StepObs, memory: dict) -> str:
        return trigger_label(self.policy)
