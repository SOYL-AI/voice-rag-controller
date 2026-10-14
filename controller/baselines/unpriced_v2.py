"""
unpriced_v2.py — the v2 unpriced baselines. None of them reads prices; each
is one operating point per (condition, latency profile), as in v1.

  B0  never retrieve.
  B1  always retrieve, at one timing policy's trigger (one baseline per
      policy: b1_v2_sentence_end / _immediate / _fixed_early_proxy /
      _forecast).
  B4  FLARE-style: retrieve iff ANY generated token's probability < theta.
      v1's B4 read a key that never existed ('one_minus_mean_confidence'),
      defaulted it to 0.0 and therefore retrieved on 0% of turns; this reads
      the token-probability signals E1-v2 records (min_token_prob,
      min_token_logprob, token_probs, token_logprobs), and only if none is
      present the documented lower-bound proxy exp(-max_entropy_nats)
      (the greedy token's probability is >= exp(-H) at every position, so
      min_t p_t >= exp(-max_t H_t)). Timing: sentence_end (FLARE acts on a
      generation of the full question).
  B5  Adaptive-RAG: a query-complexity classifier on the transcript text
      (multinomial naive Bayes over word tokens, trained on dev with the
      E1-v2 empirical needs-retrieval labels by controller/tune_v2.py),
      retrieve iff P(needed | text) >= threshold. Without a trained model it
      falls back to v1's keyword rule. Timing: sentence_end.
  B6  DRAGIN-style: retrieve iff max_entropy_nats >= theta. sentence_end.
  B8  independent-best: the best of B4/B5/B6 on dev (tuned by tune_v2.py)
      bolted to the single timing policy with the most dev hidden latency,
      chosen in isolation; the trigger is evaluated on the signal available
      at that policy's trigger.

Thresholds come from (in order): constructor params, the estimates artifact
written by tune_v2.py (controller.e3v2.estimates_artifact), the config's
controller.e3v2.unpriced block.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    POLICIES_V2,
    latency_dist_for,
    load_estimates,
    policy_fires,
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

# v1 B5's keyword list, kept verbatim as the untrained fallback.
_V1_KEYWORDS = {"hotel", "room", "check-in", "checkin", "checkout", "check-out", "pool", "wifi",
                "parking", "pet", "pets", "breakfast", "restaurant", "spa", "reservation",
                "booking", "gym", "airport", "shuttle"}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


# ---------------------------------------------------------------------------
# Signal readers
# ---------------------------------------------------------------------------
def min_token_prob(entry: Optional[dict]) -> Tuple[Optional[float], str]:
    if not entry:
        return None, "missing"
    if entry.get("min_token_prob") is not None:
        return float(entry["min_token_prob"]), "min_token_prob"
    if entry.get("min_token_logprob") is not None:
        return math.exp(float(entry["min_token_logprob"])), "min_token_logprob"
    if entry.get("token_probs"):
        return min(float(p) for p in entry["token_probs"]), "token_probs"
    if entry.get("token_logprobs"):
        return math.exp(min(float(p) for p in entry["token_logprobs"])), "token_logprobs"
    if entry.get("max_entropy_nats") is not None:
        return math.exp(-float(entry["max_entropy_nats"])), "proxy:exp(-max_entropy)"
    return None, "missing"


def max_entropy(entry: Optional[dict]) -> Optional[float]:
    if not entry or entry.get("max_entropy_nats") is None:
        return None
    return float(entry["max_entropy_nats"])


# ---------------------------------------------------------------------------
# Adaptive-RAG classifier
# ---------------------------------------------------------------------------
def train_adaptive_rag_classifier(texts: Sequence[str], labels: Sequence[bool], alpha: float = 1.0) -> dict:
    """Multinomial naive Bayes, Laplace-smoothed; JSON-serialisable."""
    counts = {"1": {}, "0": {}}
    totals = {"1": 0, "0": 0}
    docs = {"1": 0, "0": 0}
    vocab = set()
    for text, label in zip(texts, labels):
        c = "1" if label else "0"
        docs[c] += 1
        for tok in tokenize(text):
            counts[c][tok] = counts[c].get(tok, 0) + 1
            totals[c] += 1
            vocab.add(tok)
    v = max(1, len(vocab))
    n = max(1, docs["1"] + docs["0"])
    model = {"kind": "nb", "alpha": alpha, "log_prior": {}, "log_lik": {}, "log_unk": {}}
    for c in ("1", "0"):
        model["log_prior"][c] = math.log((docs[c] + alpha) / (n + 2 * alpha))
        denom = totals[c] + alpha * (v + 1)
        model["log_lik"][c] = {t: math.log((k + alpha) / denom) for t, k in counts[c].items()}
        model["log_unk"][c] = math.log(alpha / denom)
    return model


def classifier_score(model: dict, text: str) -> float:
    scores = {}
    for c in ("1", "0"):
        s = model["log_prior"][c]
        lik, unk = model["log_lik"][c], model["log_unk"][c]
        for tok in tokenize(text):
            s += lik.get(tok, unk)
        scores[c] = s
    m = max(scores.values())
    e1, e0 = math.exp(scores["1"] - m), math.exp(scores["0"] - m)
    return e1 / (e1 + e0)


# ---------------------------------------------------------------------------
# Triggers: (obs, params) -> (retrieve, why)
# ---------------------------------------------------------------------------
def trigger_b4(obs: StepObs, params: dict) -> Tuple[bool, str]:
    p, src = min_token_prob(obs.belief_entry)
    theta = float(params["theta"])
    if p is None:
        return False, "FLARE: no token-probability signal -> skip"
    return p < theta, f"FLARE: min_token_prob={p:.3f}[{src}] {'<' if p < theta else '>='} {theta:g}"


def trigger_b5(obs: StepObs, params: dict) -> Tuple[bool, str]:
    model = params.get("model")
    if not model:
        hit = any(k in obs.text.lower() for k in _V1_KEYWORDS)
        return hit, f"Adaptive-RAG: untrained -> v1 keyword fallback ({'hit' if hit else 'miss'})"
    score = classifier_score(model, obs.text)
    thr = float(params["threshold"])
    return score >= thr, f"Adaptive-RAG: P(needed|text)={score:.3f} {'>=' if score >= thr else '<'} {thr:g}"


def trigger_b6(obs: StepObs, params: dict) -> Tuple[bool, str]:
    h = max_entropy(obs.belief_entry)
    theta = float(params["threshold"])
    if h is None:
        return False, "DRAGIN: no max_entropy_nats -> skip"
    return h >= theta, f"DRAGIN: max_entropy={h:.3f} {'>=' if h >= theta else '<'} {theta:g}"


TRIGGERS: Dict[str, Callable[[StepObs, dict], Tuple[bool, str]]] = {
    "b4": trigger_b4, "b5": trigger_b5, "b6": trigger_b6,
}


def _resolved_params(cfg: dict, key: str, explicit: dict) -> dict:
    base = dict(cfg["unpriced"].get(key) or {})
    tuned = ((load_estimates(cfg).get("unpriced") or {}).get(key) or {}).get("chosen") or {}
    base.update(tuned)
    base.update({k: v for k, v in explicit.items() if v is not None})
    return base


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class B0V2(SequentialController):
    def step(self, obs, memory):
        return SKIP, StepTrace(k=obs.k, available_ms=obs.available_ms, action=SKIP, note="B0: never retrieve")


class _PolicyTriggered(SequentialController):
    """Waits for a timing policy's trigger, then applies a trigger rule once."""

    policy = "sentence_end"

    def begin_turn(self, static: TurnStatic) -> None:
        self.lat = latency_dist_for(static.retrieval, self.cfg)

    def decide_at_trigger(self, obs: StepObs) -> Tuple[bool, str]:
        raise NotImplementedError

    def step(self, obs, memory):
        record = StepTrace(k=obs.k, available_ms=obs.available_ms, action=WAIT)
        if not policy_fires(self.policy, obs, self.cfg, self.lat):
            record.note = f"waiting for {self.policy}"
            return WAIT, record
        retrieve, why = self.decide_at_trigger(obs)
        record.note = f"{why} @{self.policy}"
        return (RETRIEVE if retrieve else SKIP), record

    def timing_label(self, obs, memory):
        return self.policy


class B1V2(_PolicyTriggered):
    def __init__(self, cfg=None, policy: str = "sentence_end", **params):
        super().__init__(cfg=cfg, **params)
        if policy not in POLICIES_V2:
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy

    def decide_at_trigger(self, obs):
        return True, "B1: always retrieve"


class _ThresholdTrigger(_PolicyTriggered):
    key = ""

    def __init__(self, cfg=None, **params):
        super().__init__(cfg=cfg, **params)
        self.trigger_params = _resolved_params(self.cfg, self.key, params)

    def decide_at_trigger(self, obs):
        return TRIGGERS[self.key](obs, self.trigger_params)


class B4V2(_ThresholdTrigger):
    key = "b4"


class B5V2(_ThresholdTrigger):
    key = "b5"


class B6V2(_ThresholdTrigger):
    key = "b6"


class B8V2(_PolicyTriggered):
    def __init__(self, cfg=None, trigger: Optional[str] = None, timing: Optional[str] = None,
                 trigger_params: Optional[dict] = None, **params):
        super().__init__(cfg=cfg, **params)
        chosen = _resolved_params(self.cfg, "b8", {"trigger": trigger, "timing": timing})
        self.trigger_key = chosen["trigger"]
        self.policy = chosen["timing"]
        if self.trigger_key not in TRIGGERS:
            raise ValueError(f"b8 trigger must be one of {sorted(TRIGGERS)}")
        if self.policy not in POLICIES_V2:
            raise ValueError(f"b8 timing must be one of {POLICIES_V2}")
        self.trigger_params = _resolved_params(self.cfg, self.trigger_key, trigger_params or {})

    def decide_at_trigger(self, obs):
        retrieve, why = TRIGGERS[self.trigger_key](obs, self.trigger_params)
        return retrieve, f"B8[{self.trigger_key}] {why}"
