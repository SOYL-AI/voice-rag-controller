"""
ours_v2.py — the JOINT v2 controller: retrieve/skip and dispatch time chosen
together, as an optimal-stopping problem over the turn's partial stream.

At each step k the controller compares

    U_ret(k)  = b_k g+ (1 - rho_k) - (1 - b_k) g- - lambda_l E[max(0, L - W_k)]
                - lambda_c (1 + waste_k)                         (retrieve now)
    V_wait(k) = E_{b_{k+1}, R_{k+1}} [ max( U_ret(k+1), C(k+1) ) ]  (wait)

and retrieves now iff U_ret(k) > 0 and U_ret(k) >= V_wait(k); otherwise it
waits. At the end-of-turn step (W = 0, rho = 0, waste = 0) it retrieves iff
U_ret > 0, else skips. All terms come from _objective_v2 — exactly the ones
b9_v2 / b9plus_v2 use; only the decision structure differs.

THE LOOKAHEAD MODEL (n-step, n = controller.e3v2.ours.lookahead_steps):

  * Belief: b_{k+1} ~ BeliefTransitions.next_dist(b_k, phase(f_k)), the
    empirical conditional distribution of the next-prefix calibrated belief
    given the current one, binned, fitted on dev E1-v2 signal records by
    controller/fit_belief_transitions.py. The jump to the end-of-turn
    belief uses final_dist(b, phase). Unfitted -> identity (A3: waiting
    buys no information).
  * Remaining speech: deterministic R_{k+1} = R_k - Delta, Delta the ASR
    partial interval; elapsed speech e_{k+1} = e_k + Delta. If R_k <= Delta
    the next step is the end of turn.
  * The next partial is assumed query-ready (its rho is the word-fraction
    rho); a not-yet-ready partial now has rho = rho.not_ready.
  * Continuation C(k+1): with depth left, V_wait(k+1) (recursive); with the
    depth exhausted, the value of the feasible policy "wait until the end,
    then retrieve iff U_final > 0" = E[max(U_final(b_end), 0) | b_{k+1}].
    So every truncation is a LOWER bound on the true continuation value;
    with n >= the number of remaining partials the recursion is the exact
    finite-horizon dynamic program under the model.

Early SKIP is not taken by default (allow_early_skip=False): skip is worth 0
and waiting is worth >= 0 under the model, so committing to skip early can
only forfeit later information.

SPECULATIVE DISPATCH (documented hook, not implemented): dispatching early
and cancelling if the end-of-turn belief turns the retrieval negative would
add an action "speculate" with value E[max(U_hidden(b_end), -lambda_c*1)].
Scoring it needs Phase A to cache the speculative passage set separately
from the final one, and the cancellation cost is exactly waste. The flag
controller.e3v2.ours.speculative raises at construction rather than being
silently ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    BeliefTransitions,
    divergence_band_v2,
    estimated_frac,
    expected_exposed_ms,
    latency_dist_for,
    load_belief_calibration,
    load_belief_transitions,
    quality_model,
    read_prices_v2,
    rho_at,
    step_terms,
    utility_retrieve,
    waste_at,
)
from controller.sequential_v2 import (  # noqa: E402
    RETRIEVE,
    SKIP,
    WAIT,
    SequentialController,
    StepObs,
    StepTrace,
    TurnStatic,
)


class OursV2(SequentialController):
    priced = True

    def __init__(self, cfg: Optional[dict] = None, transitions: Optional[BeliefTransitions] = None,
                 lookahead_steps: Optional[int] = None, **params):
        super().__init__(cfg=cfg, **params)
        ocfg = self.cfg["ours"]
        if ocfg.get("speculative"):
            raise NotImplementedError(
                "ours_v2 speculative dispatch is a documented hook only (see module docstring)")
        self.calibration = load_belief_calibration(self.cfg)
        self.transitions = transitions if transitions is not None else load_belief_transitions(self.cfg)
        self.lookahead = int(lookahead_steps if lookahead_steps is not None else ocfg["lookahead_steps"])
        if self.lookahead < 1:
            raise ValueError("lookahead_steps must be >= 1")
        self.allow_early_skip = bool(ocfg.get("allow_early_skip", False))
        self.quality = quality_model(self.cfg)

    # -- per-turn static state -------------------------------------------
    def begin_turn(self, static: TurnStatic) -> None:
        self.prices = read_prices_v2(static.prices, self.cfg)
        self.lat = latency_dist_for(static.retrieval, self.cfg)
        self._memo = {}

    # -- the objective at a hypothetical future step ------------------------
    def _u_final(self, b: float) -> float:
        return utility_retrieve(b, 0.0, expected_exposed_ms(self.lat, 0.0, True, self.cfg), 0.0,
                                self.prices, self.quality)

    def _u_partial(self, b: float, remaining: float, elapsed: float) -> float:
        frac = estimated_frac(elapsed, remaining, False)
        return utility_retrieve(
            b, rho_at(frac, True, self.cfg),
            expected_exposed_ms(self.lat, remaining, False, self.cfg),
            waste_at(frac, False, self.cfg), self.prices, self.quality)

    def _v_end(self, b: float, frac: Optional[float]) -> float:
        """Value of waiting until the end of turn and deciding then."""
        return sum(p * max(self._u_final(b2), 0.0) for p, b2 in self.transitions.final_dist(b, frac))

    def v_wait(self, b: float, remaining: Optional[float], elapsed: float, interval: float, depth: int) -> float:
        """n-step lookahead value of waiting from a step with belief b."""
        key = (round(b, 12), None if remaining is None else round(remaining, 6),
               round(elapsed, 6), round(interval, 6), depth)
        if key in self._memo:
            return self._memo[key]
        frac = estimated_frac(elapsed, remaining, False)
        if remaining is None or remaining <= interval:
            value = self._v_end(b, frac)
        else:
            r1, e1 = remaining - interval, elapsed + interval
            f1 = estimated_frac(e1, r1, False)
            value = 0.0
            for p, b1 in self.transitions.next_dist(b, frac):
                cont = (self.v_wait(b1, r1, e1, interval, depth - 1) if depth > 1
                        else self._v_end(b1, f1))
                value += p * max(self._u_partial(b1, r1, e1), cont)
        self._memo[key] = value
        return value

    # -- the step --------------------------------------------------------------
    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        terms = step_terms(obs, self.cfg, self.calibration, self.lat)
        u = utility_retrieve(terms.belief, terms.rho, terms.exposed_ms, terms.waste,
                             self.prices, self.quality)
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT,
                           belief=terms.belief, remaining_ms=obs.remaining_ms,
                           exposed_ms=terms.exposed_ms, rho=terms.rho, waste=terms.waste,
                           u_retrieve=u)
        if obs.is_final:
            action = RETRIEVE if u > 0 else SKIP
            record.v_wait = 0.0
            record.note = self._note(obs, terms, u, 0.0, action)
            return action, record

        v = self.v_wait(terms.belief, obs.remaining_ms, obs.speech_elapsed_ms,
                        obs.partial_interval_ms, self.lookahead)
        record.v_wait = v
        if u > 0 and u >= v:
            action = RETRIEVE
        elif self.allow_early_skip and u <= 0 and v <= 0:
            action = SKIP
        else:
            action = WAIT
        record.note = self._note(obs, terms, u, v, action)
        return action, record

    def timing_label(self, obs: StepObs, memory: dict) -> str:
        return "sentence_end" if obs.is_final else "joint_partial"

    def _note(self, obs, terms, u, v, action) -> str:
        note = (f"b={terms.belief:.3f}({terms.belief_source}) rho={terms.rho:.2f} "
                f"exp={terms.exposed_ms:.0f}ms w={terms.waste:.3f} U={u:+.3f} V_wait={v:+.3f} "
                f"n={self.lookahead} -> {action}")
        if action == RETRIEVE:
            # Diagnostic only: would a full-latency pricer (B9) take this?
            band = divergence_band_v2(self.prices, self.quality, terms.exposed_ms, self.lat.mean_ms,
                                      terms.rho, terms.waste, terms.waste)
            if band.contains(terms.belief) and band.width > 0:
                note += f" | DIVERGENCE_V2[ours-takes] {band.describe()}"
        return note
