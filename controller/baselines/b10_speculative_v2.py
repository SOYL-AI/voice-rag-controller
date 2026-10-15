"""
b10_speculative_v2.py — SPEC: the speculative-prefetch decomposed rival
(THEORY.md §6.2, Proposition 4'). Requires
controller.e3v2.architecture = speculative.

Two decisions, neither of which sees the other:

  1. DISPATCH (timing only). At the trigger of an independently chosen
     timing policy (controller.e3v2.b10.timing, default "immediate" = the
     first query-ready partial; "forecast" is the other sensible choice) it
     ALWAYS dispatches a retrieval, whatever the belief. That dispatch costs
     lambda_c (1 + w_dispatch) and the cost is sunk.

  2. USE (belief only, at the end of turn). With the FINAL belief b_K it
     uses the passages iff

         b_K g+ (1 - rho_d) - (1 - b_K) g- - lambda_l (E[max(0, L - W)] + pi) > 0,

     W = end_of_turn - dispatch (realised, known by then). Using pays the
     exposed retrieval latency and the non-hideable increment pi and gets
     the asymmetric quality term; discarding changes nothing but the compute
     already spent.

If the trigger only fires at the end of turn (e.g. no partial is ever
query-ready), there is nothing to speculate on and it decides like any
end-of-turn controller: retrieve iff the full commit utility (with
compute) is positive.

Nothing in it is joint: the dispatch time is a function of the timing
stream only, the use decision of b_K and fixed quantities. Prop. 4': with
rho = 0 and lambda_c = 0 it is optimal over ALL policies; with lambda_c > 0
the joint optimum beats it by at most lambda_c (1 + w_1) per turn. It is
therefore the strongest rival whenever the pipeline can discard results,
and ours_v2 (joint-commit) can lose to it.
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
    belief_from_obs,
    estimated_frac,
    latency_dist_for,
    load_belief_calibration,
    policy_fires,
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


class B10SpeculativeV2(SequentialController):
    priced = True

    def __init__(self, cfg: Optional[dict] = None, timing: Optional[str] = None, **params):
        super().__init__(cfg=cfg, **params)
        self.policy = timing or self.cfg["b10"]["timing"]
        if self.policy not in POLICIES_V2:
            raise ValueError(f"b10 timing must be one of {POLICIES_V2}, got {self.policy!r}")
        self.calibration = load_belief_calibration(self.cfg)
        self.quality = quality_model(self.cfg)

    def begin_turn(self, static: TurnStatic) -> None:
        self.prices = read_prices_v2(static.prices, self.cfg)
        self.lat = latency_dist_for(static.retrieval, self.cfg)

    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT,
                           remaining_ms=obs.remaining_ms)
        spec = memory.get("spec")
        if spec is None:
            if not policy_fires(self.policy, obs, self.cfg, self.lat):
                record.note = f"waiting for {self.policy} trigger"
                return WAIT, record
            if obs.is_final:
                # Nothing was speculated: an ordinary end-of-turn decision.
                terms = step_terms(obs, self.cfg, self.calibration, self.lat)
                u = utility_retrieve(terms.belief, terms.rho, terms.cost_ms, terms.waste,
                                     self.prices, self.quality)
                record.belief, record.u_retrieve, record.exposed_ms = terms.belief, u, terms.exposed_ms
                record.note = f"no speculative dispatch before end; U={u:+.3f} @{self.policy}"
                return (RETRIEVE if u > 0 else SKIP), record
            # Dispatch decision: timing only. The belief is NOT read here.
            frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
            memory["spec"] = {"t_ms": obs.available_ms,
                              "rho": rho_at(frac, obs.query_ready, self.cfg),
                              "waste": waste_at(frac, obs.is_final, self.cfg)}
            record.rho, record.waste = memory["spec"]["rho"], memory["spec"]["waste"]
            record.note = (f"speculative dispatch @{self.policy} "
                           f"(compute {self.prices.lambda_c * (1 + record.waste):.4f} sunk)")
            return SPECULATE, record
        if not obs.is_final:
            record.note = "speculative retrieval in flight"
            return WAIT, record
        # Use decision: final belief only.
        b_end, source = belief_from_obs(obs, self.cfg, self.calibration)
        window = obs.available_ms - spec["t_ms"]
        a = speculative_use_utility(b_end, spec["rho"], self.lat, window, self.prices, self.quality)
        record.belief, record.rho, record.u_retrieve = b_end, spec["rho"], a
        record.exposed_ms = self.lat.expected_exposed_ms(window)
        action = USE if a > 0 else DISCARD
        record.note = (f"use-decision: b_K={b_end:.3f}({source}) window={window}ms "
                       f"exp={record.exposed_ms:.0f}ms+pi={self.lat.nonhideable_ms:.0f}ms "
                       f"A-lambda_l*X={a:+.3f} -> {action}")
        return action, record

    def timing_label(self, obs: StepObs, memory: dict) -> str:
        return self.policy
