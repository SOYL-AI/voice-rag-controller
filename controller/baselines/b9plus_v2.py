"""
b9plus_v2.py — B9+: the strongest decomposed rival.

Identical to b9_v2.B9V2 (same timing sub-optimizer, same trigger, same
retrieval sub-optimizer, same g+/g-/rho/waste/calibration) except for the
latency the retrieval sub-optimizer prices: not the full E[L] but the
EXPECTED EXPOSED latency of the policy the timing sub-optimizer chose, a
constant estimated on dev:

    priced_latency = Ebar_p + pi,  Ebar_p = mean over dev turns of
                     E_L[max(0, L - W_true)] at policy p's trigger,
                     W_true = end_of_turn - dispatch

This is the controller a reviewer argued would equal the joint one in v1 —
and under A1-A3 it does (constant overlap makes Ebar_p = L - o*, the very
price Ours uses). What it cannot do is price THIS turn's exposure, which
under C1 varies with the turn's remaining speech, nor trade waiting for
information (C2) against overlap. That gap — and only that gap — is what
ours_v2 is tested on against it.

TUNED TIMING RULE (PREREG_V2 §4.5). With `timing_rule` the timing
sub-optimizer is replaced by a rule tuned on dev, one of

    {"kind": "early"}                 decide at the first query-ready partial,
                                      priced at the dev-mean exposure there
    {"kind": "late"}                  decide at the end of turn, priced E[L]
    {"kind": "r_threshold", "r0": ms, "early_exposed_ms": ms}
                                      at the first query-ready partial: if the
                                      predicted remaining speech R_hat > r0,
                                      decide NOW priced at early_exposed_ms (the
                                      dev-mean realised exposure of the turns
                                      this rule sends early); else wait and
                                      decide at the end of turn priced E[L]

    {"kind": "r_below", "r0": ms, "late_exposed_ms": ms}
                                      (in the PREREG_V2 §4.5 family since
                                      2026-10-16, pre-freeze; PI decision):
                                      decide at the first query-ready partial whose
                                      R_hat <= r0 (late in the utterance, where a
                                      dispatch is still hidden by the end-of-turn
                                      gap), priced at the dev-mean realised exposure
                                      of that rule; at the end of turn if none

(+ pi in every case). The rule reads the timing state (R_hat) only to pick
WHEN; the retrieve decision still prices a dev scalar and never this turn's
exposure, so B9+ stays in the decomposed class Pi_D (THEORY §1).
controller/tune_v2.py:tune_b9plus_timing() selects the rule per price.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    belief_from_obs,
    estimated_frac,
    priced_latency_ms,
    rho_at,
    waste_at,
)
from controller.baselines.b9_v2 import B9V2, _optimize_retrieval  # noqa: E402
from controller.sequential_v2 import RETRIEVE, SKIP, WAIT, StepObs, StepTrace, TurnStatic  # noqa: E402

TIMING_RULE_KINDS = ("early", "late", "r_threshold", "r_below")


class B9PlusV2(B9V2):
    latency_pricing = "policy_expected_exposed"

    def __init__(self, cfg: Optional[dict] = None, timing_rule: Optional[dict] = None, **params):
        super().__init__(cfg=cfg, **params)
        if timing_rule is not None and timing_rule.get("kind") not in TIMING_RULE_KINDS:
            raise ValueError(f"timing_rule kind must be one of {TIMING_RULE_KINDS}")
        self.timing_rule = timing_rule

    def begin_turn(self, static: TurnStatic) -> None:
        super().begin_turn(static)
        rule = self.timing_rule
        if rule is None:
            return
        if rule["kind"] == "early":
            self.policy, self.timing_why = "immediate", "timing-rule[early]"
        elif rule["kind"] == "late":
            self.policy, self.timing_why = "sentence_end", "timing-rule[late]"
        else:
            self.policy = "immediate"
            self.timing_why = f"timing-rule[R_hat > {float(rule['r0']):.0f} ms -> early, else late]"
        self.priced_latency_ms = self._priced_latency_ms()

    def _priced_latency_ms(self) -> float:
        row = self.prof.get(self.policy) or {}
        exposed = self.lat.mean_ms if row.get("exposed_ms") is None else float(row["exposed_ms"])
        # dev-mean exposure of the retrieval call + pi (identical for all)
        return priced_latency_ms(self.lat, exposed)

    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        rule = self.timing_rule
        if rule is not None and rule["kind"] == "r_below":
            return self._step_r_below(obs, rule)
        if rule is None or rule["kind"] != "r_threshold":
            return super().step(obs, memory)
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT, remaining_ms=obs.remaining_ms)
        if memory.get("late") and not obs.is_final:
            record.note = "timing-rule: late branch, waiting for end of turn"
            return WAIT, record
        if not obs.is_final and not obs.query_ready:
            record.note = "timing-rule: waiting for the first query-ready partial"
            return WAIT, record
        if not obs.is_final and not (obs.remaining_ms is not None and obs.remaining_ms > float(rule["r0"])):
            memory["late"] = True  # the rule chose late at its one decision point
            record.note = f"timing-rule: R_hat={obs.remaining_ms} <= r0 -> late"
            return WAIT, record
        if obs.is_final:
            priced = priced_latency_ms(self.lat, self.lat.mean_ms)
        else:
            priced = priced_latency_ms(self.lat, float(rule["early_exposed_ms"]))
        belief, source = belief_from_obs(obs, self.cfg, self.calibration)
        frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
        rho, waste = rho_at(frac, obs.query_ready, self.cfg), waste_at(frac, obs.is_final, self.cfg)
        retrieve, u, why = _optimize_retrieval(belief, rho, waste, priced, self.prices, self.quality)
        record.belief, record.rho, record.waste, record.u_retrieve, record.exposed_ms = belief, rho, waste, u, priced
        record.note = f"{self.timing_why} | {why} [{source}] branch={'late' if obs.is_final else 'early'}"
        memory["branch"] = "late" if obs.is_final else "early"
        return (RETRIEVE if retrieve else SKIP), record

    def _step_r_below(self, obs: StepObs, rule: dict) -> Tuple[str, StepTrace]:
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT, remaining_ms=obs.remaining_ms)
        if not obs.is_final and (not obs.query_ready or obs.remaining_ms is None
                                 or obs.remaining_ms > float(rule["r0"])):
            record.note = f"timing-rule[r_below]: waiting for R_hat <= {float(rule['r0']):.0f} ms"
            return WAIT, record
        exposed = self.lat.mean_ms if obs.is_final else float(rule["late_exposed_ms"])
        priced = priced_latency_ms(self.lat, exposed)
        belief, source = belief_from_obs(obs, self.cfg, self.calibration)
        frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
        rho, waste = rho_at(frac, obs.query_ready, self.cfg), waste_at(frac, obs.is_final, self.cfg)
        retrieve, u, why = _optimize_retrieval(belief, rho, waste, priced, self.prices, self.quality)
        record.belief, record.rho, record.waste, record.u_retrieve, record.exposed_ms = belief, rho, waste, u, priced
        record.note = f"timing-rule[r_below r0={float(rule['r0']):.0f}] | {why} [{source}]"
        return (RETRIEVE if retrieve else SKIP), record

    def timing_label(self, obs: StepObs, memory: dict) -> str:
        if self.timing_rule is not None and self.timing_rule["kind"] == "r_below":
            return "sentence_end" if obs.is_final else "r_below_partial"
        if self.timing_rule is not None and self.timing_rule["kind"] == "r_threshold":
            return "sentence_end" if obs.is_final else "r_threshold_early"
        return self.policy
