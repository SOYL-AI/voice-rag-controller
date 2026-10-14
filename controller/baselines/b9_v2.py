"""
b9_v2.py — the DECOMPOSED v2 controller (B9-v2): two sub-optimizers that
share prices and the v2 cost model (_objective_v2) but no state.

  1. Timing sub-optimizer (_optimize_timing), run once before the turn from
     dev-estimated policy constants and the prices only. It picks one policy
     from {sentence_end, immediate, fixed_early_proxy, forecast} by the v1
     B9 timing objective restricted to the terms that vary with timing:

         T(p) = lambda_l * Hbar_p  -  bbar * g+ * rhobar_p  -  lambda_c * wbar_p

     (Hbar_p = dev-mean realised hidden latency of p on this latency
     profile, rhobar_p / wbar_p = dev-mean rho / waste at p's trigger, bbar
     = dev-mean end-of-turn belief). With controller.e3v2.b9.timing_objective
     = "hidden_only" the risk and waste terms are dropped. It never sees a
     belief from this turn and never sees the retrieval decision.

  2. Retrieval sub-optimizer (_optimize_retrieval), run once at the chosen
     policy's trigger with the belief available THEN, priced at the FULL
     expected retrieval latency E[L] — it has no timing to condition on:

         U = b g+ (1 - rho) - (1 - b) g- - lambda_l (E[L] + pi) - lambda_c (1 + waste)

     retrieve iff U > 0, else skip (final). It never sees the policy name.

Same g+/g-/rho/waste/calibration as ours_v2 and b9plus_v2. b9plus_v2 differs
in exactly one line: the latency it prices (see that module).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective import Prices  # noqa: E402
from controller.baselines._objective_v2 import (  # noqa: E402
    POLICIES_V2,
    QualityModel,
    belief_from_obs,
    divergence_band_v2,
    estimated_frac,
    expected_exposed_ms,
    latency_dist_for,
    load_belief_calibration,
    policy_fires,
    priced_latency_ms,
    quality_model,
    read_prices_v2,
    resolve_policy_constants,
    rho_at,
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

# Used only when no dev estimates exist: v1's measured overlaps, labelled
# UNFITTED in every reason string. A real run always passes dev constants.
_V1_HIDDEN_FALLBACK_MS = {"sentence_end": 0.0, "immediate": 50.0,
                          "fixed_early_proxy": 188.0, "forecast": 188.0}


def profile_constants(constants: dict, profile: str, mean_latency_ms: float) -> dict:
    prof = (constants.get("profiles") or {}).get(profile)
    if prof:
        return prof
    return {p: {"hidden_ms": h, "exposed_ms": max(0.0, mean_latency_ms - h), "rho": 0.0, "waste": 0.0}
            for p, h in _V1_HIDDEN_FALLBACK_MS.items()}


# ---------------------------------------------------------------------------
# Sub-optimizer 1: timing (prices + dev constants only)
# ---------------------------------------------------------------------------
def _optimize_timing(prices: Prices, prof_constants: dict, prior_belief: float,
                     quality: QualityModel, objective: str) -> Tuple[str, str]:
    def score(policy: str) -> float:
        row = prof_constants.get(policy) or {}
        value = prices.lambda_l * float(row.get("hidden_ms", 0.0)) / 1000.0
        if objective == "v1_full":
            value -= prior_belief * quality.g_plus * float(row.get("rho", 0.0))
            value -= prices.lambda_c * float(row.get("waste", 0.0))
        elif objective != "hidden_only":
            raise ValueError(f"unknown b9 timing_objective {objective!r}")
        return value

    candidates = [p for p in POLICIES_V2 if p in prof_constants]
    best = max(candidates, key=score)  # first maximal = most conservative on ties
    row = prof_constants[best]
    return best, (f"timing-opt[{objective}]: {best} T={score(best):+.4f} "
                  f"(Hbar={float(row.get('hidden_ms', 0)):.0f}ms rhobar={float(row.get('rho', 0)):.2f} "
                  f"wbar={float(row.get('waste', 0)):.3f})")


# ---------------------------------------------------------------------------
# Sub-optimizer 2: retrieve/skip at the trigger (belief + a FIXED latency price)
# ---------------------------------------------------------------------------
def _optimize_retrieval(belief: float, rho: float, waste: float, priced_latency_ms: float,
                        prices: Prices, quality: QualityModel) -> Tuple[bool, float, str]:
    u = utility_retrieve(belief, rho, priced_latency_ms, waste, prices, quality)
    return u > 0, u, (f"retrieval-opt: b={belief:.3f} rho={rho:.2f} w={waste:.3f} "
                      f"priced_lat={priced_latency_ms:.0f}ms U={u:+.3f}")


class B9V2(SequentialController):
    priced = True
    #: "full" = E[L] (B9); "policy_expected_exposed" = dev E[exposed | policy] (B9+)
    latency_pricing = "full"

    def __init__(self, cfg: Optional[dict] = None, policy_constants: Optional[dict] = None, **params):
        super().__init__(cfg=cfg, **params)
        self.calibration = load_belief_calibration(self.cfg)
        self.quality = quality_model(self.cfg)
        self.constants, self.constants_source = resolve_policy_constants(self.cfg, policy_constants)
        self.objective = self.cfg["b9"]["timing_objective"]

    def begin_turn(self, static: TurnStatic) -> None:
        self.prices = read_prices_v2(static.prices, self.cfg)
        self.lat = latency_dist_for(static.retrieval, self.cfg)
        self.prof = profile_constants(self.constants, self.lat.profile, self.lat.mean_ms)
        prior = float(self.constants.get("prior_belief", self.cfg["belief"]["prior"]))
        self.policy, self.timing_why = _optimize_timing(self.prices, self.prof, prior,
                                                        self.quality, self.objective)
        self.priced_latency_ms = self._priced_latency_ms()

    def _priced_latency_ms(self) -> float:
        # B9: the full expected retrieval latency, whatever the policy (+ pi,
        # added for every priced controller by priced_latency_ms).
        return priced_latency_ms(self.lat, self.lat.mean_ms)

    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT,
                           remaining_ms=obs.remaining_ms)
        if not policy_fires(self.policy, obs, self.cfg, self.lat):
            record.note = f"waiting for {self.policy} trigger"
            return WAIT, record
        belief, source = belief_from_obs(obs, self.cfg, self.calibration)
        frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
        rho = rho_at(frac, obs.query_ready, self.cfg)
        waste = waste_at(frac, obs.is_final, self.cfg)
        retrieve, u, why = _optimize_retrieval(belief, rho, waste, self.priced_latency_ms,
                                               self.prices, self.quality)
        record.belief, record.rho, record.waste, record.u_retrieve = belief, rho, waste, u
        record.exposed_ms = self.priced_latency_ms
        # ---- diagnostic only: both sub-optimizers have returned ----
        ours_exp = priced_latency_ms(
            self.lat, expected_exposed_ms(self.lat, obs.remaining_ms, obs.is_final, self.cfg))
        band = divergence_band_v2(self.prices, self.quality, ours_exp, self.priced_latency_ms,
                                  rho, waste, waste)
        tag = ""
        if not retrieve and band.contains(belief) and band.width > 0:
            tag = f" | DIVERGENCE_V2[{self.name or 'b9'}-declines] {band.describe()}"
        record.note = (f"{self.timing_why} | {why} [{source}] constants={self.constants_source}"
                       f"{tag}")
        return (RETRIEVE if retrieve else SKIP), record

    def timing_label(self, obs: StepObs, memory: dict) -> str:
        return self.policy
