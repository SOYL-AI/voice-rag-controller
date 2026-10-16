"""
ours_v2.py — the JOINT v2 controller: retrieve/skip and dispatch time chosen
together, as an optimal-stopping problem over the turn's partial stream.

At each step k the controller compares

    U_ret(k)  = b_k g+ (1 - rho_k) - (1 - b_k) g- - lambda_l (E[max(0, L - W_k)] + pi)
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
  * Next decision point (CORRECTED 2026-10-16; implementation corrected
    once more the same day, PI decision). The next partial carries NEW
    information with probability p = P(R > Delta), R = remaining SPEECH under
    the predictor's q10/q50/q90 aged to this step's availability and NOT
    clipped (so p -> 0 after the predicted end of speech), Delta = the
    dev-measured partial-to-partial availability spacing (ASR lag
    included). Post-speech re-transcription partials carry no new
    information (identity transition), so the other branch is the end-of-
    turn decision. The end gap enters only the exposure term. With a
    deterministic prefix schedule (text150) p = 1 while a prefix is still to
    come and 0 after the last. The continuation is the mixture
    p * (next-partial value at R' = median(R | R > Delta) - Delta)
    + (1 - p) * (end-of-turn decision value). Deeper hypothetical steps use
    the point R'. History: the first version assumed the next partial
    certain whenever the point R_hat exceeded Delta; the first correction
    used p = P(R + gap > Delta) with R clipped at 0, which is ~1 at every
    dev step (gap >= Delta), so its end-of-turn branch never fired.
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

ARCHITECTURE (controller.e3v2.architecture) selects which joint controller
this is — PREREG_V2 §3's Ours-v2[arch]:

  commit       Ours-v2[COMMIT], described above: a dispatch is used.

  speculative  Ours-v2[SPECULATIVE]: the same optimal-stopping structure,
               but dispatching at k buys the OPTION of having the passages
               ready at the end of turn, where they are used iff that pays:

      S(k)      = E[ max(U_use(b_end; W_k, rho_k), 0) | b_k ] - lambda_c (1 + w_k)
      U_use(b)  = b g+ (1 - rho_k) - (1 - b) g- - lambda_l (E[max(0, L - W_k)] + pi)
      V_wait(k) = E[ max(S(k+1), C(k+1)) ]          (dispatch later, or never)

               It speculates iff S(k) > 0 and S(k) >= V_wait(k). With the
               passages in flight it waits; at the end of turn it USES them
               iff U_use(b_end) > 0 with the realised hiding window, else
               DISCARDS them (compute already spent). Without a dispatch by
               the end it decides like the commit controller there.
               b_end | b_k is the model's end-of-turn belief distribution,
               obtained by composing next_dist down to the last partial and
               final_dist from there (exact under the model, at any depth).
               THEORY.md Prop. 4': with rho = 0 and lambda_c = 0 this equals
               SPEC (b10_speculative_v2); in general it is >= SPEC and >=
               Ours-v2[COMMIT] under the model, and beats SPEC by at most
               lambda_c (1 + w_1) per turn.

The legacy flag controller.e3v2.ours.speculative is superseded by
`architecture` and raises if set.
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
    priced_latency_ms,
    quality_model,
    read_prices_v2,
    rho_at,
    speculative_use_utility,
    step_terms,
    utility_retrieve,
    waste_at,
)
from controller.sequential_v2 import (  # noqa: E402
    DISCARD,
    RETRIEVE,
    SKIP,
    SPECULATE,
    USE,
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
            raise ValueError("controller.e3v2.ours.speculative is superseded: set "
                             "controller.e3v2.architecture: speculative instead")
        self.speculative = self.cfg.get("architecture") == "speculative"
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
        cost = priced_latency_ms(self.lat, expected_exposed_ms(self.lat, 0.0, True, self.cfg))
        return utility_retrieve(b, 0.0, cost, 0.0, self.prices, self.quality)

    def _u_partial(self, b: float, remaining: float, elapsed: float) -> float:
        # a hypothetical step after the end of speech (remaining < 0, still
        # inside the end gap) hears the whole question: f = 1
        frac = estimated_frac(elapsed, max(0.0, remaining), False)
        return utility_retrieve(
            b, rho_at(frac, True, self.cfg),
            priced_latency_ms(self.lat, expected_exposed_ms(self.lat, remaining, False, self.cfg)),
            waste_at(frac, False, self.cfg), self.prices, self.quality)

    def _branches(self, remaining: Optional[float], interval: float, rq=None, sched=None):
        """
        THE LOOKAHEAD FIX (2026-10-16). From a step, the next decision point
        is an information-carrying partial with probability p and the end-of-
        turn decision otherwise; p = P(R > spacing) from the remaining-speech
        predictor's unclipped aged quantiles `rq`, or the deterministic
        schedule `sched` (prefixes still to come); see
        _objective_v2.next_partial_branch. Returns
        ([(prob, remaining at the next partial | None for end of turn)],
        spacing).
        """
        from controller.baselines._objective_v2 import RemainingDist, next_partial_branch, partial_spacing_ms
        if remaining is None:
            return [(1.0, None)], partial_spacing_ms(interval, self.cfg)
        dist = RemainingDist(*rq) if rq else RemainingDist.point(remaining)
        spacing = partial_spacing_ms(interval, self.cfg)
        p_next, r_next = next_partial_branch(dist, spacing, self.cfg, sched)
        out = []
        if p_next > 0:
            out.append((p_next, r_next))
        if p_next < 1:
            out.append((1.0 - p_next, None))
        return out, spacing

    def _v_end(self, b: float, frac: Optional[float]) -> float:
        """Value of waiting until the end of turn and deciding then."""
        return sum(p * max(self._u_final(b2), 0.0) for p, b2 in self.transitions.final_dist(b, frac))

    def v_wait(self, b: float, remaining: Optional[float], elapsed: float, interval: float, depth: int,
               rq=None, sched=None) -> float:
        """n-step lookahead value of waiting from a step with belief b: a
        mixture of the next-partial value (probability p) and the end-of-turn
        decision value (1 - p); see _branches."""
        key = (round(b, 12), None if remaining is None else round(remaining, 6),
               round(elapsed, 6), round(interval, 6), depth, rq, sched)
        if key in self._memo:
            return self._memo[key]
        frac = estimated_frac(elapsed, remaining, False)
        branches, spacing = self._branches(remaining, interval, rq, sched)
        nxt = None if sched is None else sched - 1
        value = 0.0
        for prob, r1 in branches:
            if r1 is None:
                value += prob * self._v_end(b, frac)
                continue
            e1 = elapsed + spacing
            f1 = estimated_frac(e1, max(0.0, r1), False)
            sub = 0.0
            for p, b1 in self.transitions.next_dist(b, frac):
                cont = (self.v_wait(b1, r1, e1, interval, depth - 1, None, nxt) if depth > 1
                        else self._v_end(b1, f1))
                sub += p * max(self._u_partial(b1, r1, e1), cont)
            value += prob * sub
        self._memo[key] = value
        return value

    # -- speculative architecture: option value of a dispatch ----------------
    def _use_value(self, b: float, window: float, rho: float) -> float:
        """U_use at the end of turn: quality term minus exposed latency + pi,
        compute excluded (it is sunk once dispatched)."""
        return speculative_use_utility(b, rho, self.lat, window, self.prices, self.quality)

    def _option_value(self, b: float, remaining: Optional[float], elapsed: float, interval: float,
                      query_ready: bool = True, rq=None, sched=None) -> Tuple[float, float, float]:
        """S(k) for a speculative dispatch now: (S, rho_k, w_k)."""
        frac = estimated_frac(elapsed, None if remaining is None else max(0.0, remaining), False)
        rho = rho_at(frac, query_ready, self.cfg)
        waste = waste_at(frac, False, self.cfg)
        from controller.baselines._objective_v2 import hiding_window_ms
        window = hiding_window_ms(remaining, False, self.cfg)
        use = lambda b_end, w=window, r=rho: max(self._use_value(b_end, w, r), 0.0)  # noqa: E731
        # memo key must identify the function: use its (window, rho) parameters
        use.__qualname__ = f"use[{round(window, 6)},{round(rho, 12)}]"
        option = self._end_expect_named(use, b, remaining, elapsed, interval, rq, sched)
        return option - self.prices.lambda_c * (1.0 + waste), rho, waste

    def _end_expect_named(self, fn, b, remaining, elapsed, interval, rq=None, sched=None) -> float:
        """E[fn(b_end) | b at a partial with (remaining, elapsed)] under the
        model: next_dist down to the last partial, then final_dist. fn is
        re-created per call, so the memo is keyed on its __qualname__, which
        callers set to encode its parameters."""
        key_fn = fn.__qualname__
        key = ("endn", key_fn, round(b, 12), None if remaining is None else round(remaining, 6),
               round(elapsed, 6), round(interval, 6), rq, sched)
        if key in self._memo:
            return self._memo[key]
        frac = estimated_frac(elapsed, None if remaining is None else max(0.0, remaining), False)
        branches, spacing = self._branches(remaining, interval, rq, sched)
        nxt = None if sched is None else sched - 1
        value = 0.0
        for prob, r1 in branches:
            if r1 is None:
                value += prob * sum(p * fn(b2) for p, b2 in self.transitions.final_dist(b, frac))
            else:
                value += prob * sum(p * self._end_expect_named(fn, b1, r1, elapsed + spacing, interval, None, nxt)
                                    for p, b1 in self.transitions.next_dist(b, frac))
        self._memo[key] = value
        return value

    def _no_dispatch_value(self, b, remaining, elapsed, interval, rq=None, sched=None) -> float:
        """Wait to the end without dispatching, then commit-decide there."""
        fn = lambda b_end: max(self._u_final(b_end), 0.0)  # noqa: E731
        fn.__qualname__ = "final_commit"
        return self._end_expect_named(fn, b, remaining, elapsed, interval, rq, sched)

    def v_wait_spec(self, b: float, remaining: Optional[float], elapsed: float, interval: float,
                    depth: int, rq=None, sched=None) -> float:
        key = ("vws", round(b, 12), None if remaining is None else round(remaining, 6),
               round(elapsed, 6), round(interval, 6), depth, rq, sched)
        if key in self._memo:
            return self._memo[key]
        frac = estimated_frac(elapsed, None if remaining is None else max(0.0, remaining), False)
        branches, spacing = self._branches(remaining, interval, rq, sched)
        nxt = None if sched is None else sched - 1
        value = 0.0
        for prob, r1 in branches:
            if r1 is None:
                value += prob * sum(p * max(self._u_final(b2), 0.0)
                                    for p, b2 in self.transitions.final_dist(b, frac))
                continue
            e1 = elapsed + spacing
            sub = 0.0
            for p, b1 in self.transitions.next_dist(b, frac):
                s1 = self._option_value(b1, r1, e1, interval, True, None, nxt)[0]
                cont = (self.v_wait_spec(b1, r1, e1, interval, depth - 1, None, nxt) if depth > 1
                        else self._no_dispatch_value(b1, r1, e1, interval, None, nxt))
                sub += p * max(s1, cont)
            value += prob * sub
        self._memo[key] = value
        return value

    def _step_speculative(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT,
                           remaining_ms=obs.remaining_ms)
        spec = memory.get("spec")
        if spec is not None:
            if not obs.is_final:
                record.note = "speculative retrieval in flight"
                return WAIT, record
            b_end, source = self._belief(obs)
            window = obs.available_ms - spec["t_ms"]
            a = self._use_value(b_end, window, spec["rho"])
            record.belief, record.rho, record.u_retrieve = b_end, spec["rho"], a
            record.exposed_ms = self.lat.expected_exposed_ms(window)
            action = USE if a > 0 else DISCARD
            record.note = (f"Ours[spec] use-decision: b_K={b_end:.3f}({source}) window={window}ms "
                           f"U_use={a:+.3f} -> {action}")
            return action, record
        if obs.is_final:
            # never dispatched: the ordinary end-of-turn commit decision
            return self._step_commit(obs, memory)
        terms = step_terms(obs, self.cfg, self.calibration, self.lat)
        s_now, rho, waste = self._option_value(terms.belief, obs.remaining_ms, obs.speech_elapsed_ms,
                                               obs.partial_interval_ms, obs.query_ready, obs.remaining_quantiles,
                                               obs.scheduled_partials_left)
        v = self.v_wait_spec(terms.belief, obs.remaining_ms, obs.speech_elapsed_ms,
                             obs.partial_interval_ms, self.lookahead, obs.remaining_quantiles,
                             obs.scheduled_partials_left)
        record.belief, record.rho, record.waste = terms.belief, rho, waste
        record.u_retrieve, record.v_wait, record.exposed_ms = s_now, v, terms.exposed_ms
        if s_now > 0 and s_now >= v:
            memory["spec"] = {"t_ms": obs.available_ms, "rho": rho, "waste": waste}
            action = SPECULATE
        else:
            action = WAIT
        record.note = (f"Ours[spec] b={terms.belief:.3f} S(option-compute)={s_now:+.3f} "
                       f"V_wait={v:+.3f} n={self.lookahead} -> {action}")
        return action, record

    def _belief(self, obs: StepObs):
        from controller.baselines._objective_v2 import belief_from_obs
        return belief_from_obs(obs, self.cfg, self.calibration)

    # -- the step --------------------------------------------------------------
    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        if self.speculative:
            return self._step_speculative(obs, memory)
        return self._step_commit(obs, memory)

    def _step_commit(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        terms = step_terms(obs, self.cfg, self.calibration, self.lat)
        u = utility_retrieve(terms.belief, terms.rho, terms.cost_ms, terms.waste,
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
                        obs.partial_interval_ms, self.lookahead, obs.remaining_quantiles,
                        obs.scheduled_partials_left)
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
        if obs.is_final:
            return "sentence_end"
        return "joint_speculative" if self.speculative else "joint_partial"

    def _note(self, obs, terms, u, v, action) -> str:
        note = (f"b={terms.belief:.3f}({terms.belief_source}) rho={terms.rho:.2f} "
                f"exp={terms.exposed_ms:.0f}ms+pi={self.lat.nonhideable_ms:.0f}ms w={terms.waste:.3f} U={u:+.3f} V_wait={v:+.3f} "
                f"n={self.lookahead} -> {action}")
        if action == RETRIEVE:
            # Diagnostic only: would a full-latency pricer (B9) take this?
            band = divergence_band_v2(self.prices, self.quality, terms.cost_ms,
                                      priced_latency_ms(self.lat, self.lat.mean_ms),
                                      terms.rho, terms.waste, terms.waste)
            if band.contains(terms.belief) and band.width > 0:
                note += f" | DIVERGENCE_V2[ours-takes] {band.describe()}"
        return note
