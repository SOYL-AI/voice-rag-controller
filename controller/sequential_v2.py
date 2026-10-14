"""
sequential_v2.py — the v2 SEQUENTIAL decision contract.

v1 controllers answered one question per turn, decide(ctx) -> Decision, with
the timing as a label. v2 controllers are evaluated CAUSALLY over a turn's
partial-transcript stream: at each step k (one per ASR partial, plus one
end-of-turn step) a controller sees only what is available at that step's
available_ms and chooses

    retrieve now  |  skip (final)  |  wait

The turn's outcome is a TurnDecision (dispatch time or None, timing label,
the query text dispatched on, a reason, and the per-step trace).

CAUSALITY IS STRUCTURAL, not a convention:

  * build_steps() constructs StepObs k from data with timestamps <= the
    step's available_ms only (partials, prefix beliefs, remaining-time
    predictions). The final transcript appears only in the end-of-turn
    step.
  * decide_turn() hands the controller ONE StepObs at a time plus a memory
    dict the controller itself wrote on earlier steps, and stops at the
    first retrieve/skip. A controller never receives the TurnCtx, so it
    cannot look ahead even by accident.
  * tests/test_v2_theory.py perturbs every post-tau partial/belief/
    prediction and asserts the pre-tau trace and any pre-tau dispatch are
    unchanged.

The only per-turn information available before the first step is static:
the price point and the latency profile (TurnStatic).
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import load_e3v2_config  # noqa: E402
from controller.timing import is_query_ready  # noqa: E402

WAIT = "wait"
RETRIEVE = "retrieve"
SKIP = "skip"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TurnCtx:
    """
    Everything the evaluator knows about one turn. Controllers never see
    this object — see decide_turn(). `signals` follows CONTRACTS.md §6;
    `oracle` is populated only for oracle_v2 and is never read by a causal
    controller.

    signals["timing"] may additionally carry (optional, additive to §6):
      user_end_ms / vad_user_end_ms   VAD end of the user's speech
      asr_final {t_ms, text}          final ASR transcript and its time
      partial_interval_ms             the ASR's partial cadence
    """

    item_id: str
    signals: dict
    prices: dict = field(default_factory=dict)
    condition: str = "default"
    transcript_final: str = ""
    turn_id: int = 1
    conversation_id: str = ""
    oracle: Optional[dict] = None


@dataclass(frozen=True)
class TurnStatic:
    """What a controller may know before the first step: prices and the
    latency profile block. Both are fixed for the whole turn."""

    prices: dict
    retrieval: dict


@dataclass(frozen=True)
class StepObs:
    """One causal observation. Every field is computed from data stamped at
    or before available_ms."""

    k: int
    partial_index: Optional[int]  # index into the turn's partials; None at end of turn
    available_ms: int
    is_final: bool
    text: str
    query_ready: bool
    speech_elapsed_ms: float
    remaining_ms: Optional[float]  # predicted remaining speech; 0 at end of turn
    belief_entry: Optional[dict]
    belief_source: str  # "prefix" | "full" | "none"
    partial_interval_ms: float
    retrieval: dict
    prices: dict
    history_texts: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
@dataclass
class StepTrace:
    k: int
    available_ms: int
    action: str
    belief: Optional[float] = None
    remaining_ms: Optional[float] = None
    exposed_ms: Optional[float] = None
    rho: Optional[float] = None
    waste: Optional[float] = None
    u_retrieve: Optional[float] = None
    v_wait: Optional[float] = None
    note: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TurnDecision:
    dispatch_t_ms: Optional[int]
    timing_label: str
    query_text: Optional[str]
    reason: str
    trace: List[StepTrace]
    dispatch_step: Optional[int] = None
    dispatch_partial_index: Optional[int] = None  # None with a dispatch => final transcript
    cancelled_dispatches: int = 0  # speculative-dispatch hook (unused unless enabled)

    @property
    def retrieve(self) -> bool:
        return self.dispatch_t_ms is not None


# ---------------------------------------------------------------------------
# Step construction
# ---------------------------------------------------------------------------
def _timing_block(ctx: TurnCtx) -> dict:
    return (ctx.signals or {}).get("timing") or {}


def _partials(ctx: TurnCtx) -> List[dict]:
    parts = [p for p in (_timing_block(ctx).get("partials") or []) if p.get("t_ms") is not None]
    return sorted(parts, key=lambda p: float(p["t_ms"]))


def user_end_ms(ctx: TurnCtx) -> float:
    timing = _timing_block(ctx)
    ue = timing.get("user_end_ms", timing.get("vad_user_end_ms"))
    if ue is not None:
        return float(ue)
    times = [float(p["t_ms"]) for p in _partials(ctx)]
    fin = (timing.get("asr_final") or {}).get("t_ms")
    if fin is not None:
        times.append(float(fin))
    return max(times) if times else 0.0


def end_of_turn_ms(ctx: TurnCtx, cfg: dict) -> float:
    """When the end-of-turn step happens: VAD end + end-of-turn detection
    delay, or the final ASR transcript's arrival if that is later."""
    end = user_end_ms(ctx) + float(cfg["timing"]["eot_detection_ms"])
    fin = (_timing_block(ctx).get("asr_final") or {}).get("t_ms")
    if fin is not None:
        end = max(end, float(fin))
    return end


def _belief_entries(ctx: TurnCtx, cfg: dict) -> List[Tuple[float, dict]]:
    model = (ctx.signals or {}).get("model") or {}
    lag = float(cfg["belief"]["availability_lag_ms"])
    out = []
    for entry in model.get("uncertainty_by_prefix") or []:
        if entry.get("t_ms") is None:
            continue
        avail = entry.get("available_ms")
        avail = float(entry["t_ms"]) + lag if avail is None else float(avail)
        out.append((avail, entry))
    out.sort(key=lambda pair: pair[0])
    return out


def _is_full_entry(entry: dict) -> bool:
    if entry.get("kind") == "full":
        return True
    wf = entry.get("word_frac")
    return wf is not None and float(wf) >= 1.0


def build_steps(ctx: TurnCtx, cfg: Optional[dict] = None) -> List[StepObs]:
    cfg = cfg if cfg is not None else load_e3v2_config()
    timing = _timing_block(ctx)
    onset = float(timing.get("speech_onset_ms", 0) or 0)
    end_ms = end_of_turn_ms(ctx, cfg)
    interval = float(timing.get("partial_interval_ms") or cfg["timing"]["default_partial_interval_ms"])
    qr_cfg = cfg["timing"]["query_ready"]
    retrieval = dict((ctx.signals or {}).get("retrieval") or {})
    prices = dict(ctx.prices or {})
    beliefs = _belief_entries(ctx, cfg)
    preds = sorted(
        [p for p in (timing.get("predicted_remaining_speech_ms") or []) if p.get("t_ms") is not None],
        key=lambda p: float(p["t_ms"]),
    )
    partials = [p for p in _partials(ctx) if float(p["t_ms"]) < end_ms]

    def latest_belief(at: float, final: bool) -> Tuple[Optional[dict], str]:
        avail = [e for a, e in beliefs if a <= at]
        if final:
            full = [e for e in avail if _is_full_entry(e)]
            if full:
                return full[-1], "full"
            v1 = ((ctx.signals or {}).get("model") or {}).get("uncertainty")
            if v1:
                return dict(v1), "full"
        if avail:
            return avail[-1], ("full" if _is_full_entry(avail[-1]) else "prefix")
        return None, "none"

    def predicted_remaining(at: float) -> Optional[float]:
        avail = [p for p in preds if float(p["t_ms"]) <= at]
        if not avail:
            return None
        last = avail[-1]
        return max(0.0, float(last["remaining_ms"]) - (at - float(last["t_ms"])))

    steps: List[StepObs] = []
    history: List[str] = []
    for idx, part in enumerate(partials):
        at = float(part["t_ms"])
        text = str(part.get("text") or "")
        history.append(text)
        entry, source = latest_belief(at, final=False)
        steps.append(StepObs(
            k=len(steps), partial_index=idx, available_ms=int(round(at)), is_final=False,
            text=text, query_ready=bool(text.strip()) and is_query_ready(text, qr_cfg),
            speech_elapsed_ms=max(0.0, at - onset), remaining_ms=predicted_remaining(at),
            belief_entry=entry, belief_source=source, partial_interval_ms=interval,
            retrieval=retrieval, prices=prices, history_texts=tuple(history),
        ))

    final_text = (timing.get("asr_final") or {}).get("text") or ctx.transcript_final or (
        history[-1] if history else "")
    entry, source = latest_belief(end_ms, final=True)
    steps.append(StepObs(
        k=len(steps), partial_index=None, available_ms=int(math.ceil(end_ms)), is_final=True,
        text=str(final_text), query_ready=True,
        speech_elapsed_ms=max(0.0, user_end_ms(ctx) - onset), remaining_ms=0.0,
        belief_entry=entry, belief_source=source, partial_interval_ms=interval,
        retrieval=retrieval, prices=prices, history_texts=tuple(history + [str(final_text)]),
    ))
    return steps


# ---------------------------------------------------------------------------
# Controller base
# ---------------------------------------------------------------------------
class SequentialController(ABC):
    """Base for every v2 controller. Implement step(); decide_turn() is the
    causal driver and should not be overridden (oracle_v2 is the one,
    clearly-labelled, exception)."""

    name: str = ""
    priced: bool = False

    def __init__(self, cfg: Optional[dict] = None, **params):
        self.cfg = cfg if cfg is not None else load_e3v2_config()
        self.params = params

    def begin_turn(self, static: TurnStatic) -> None:
        """Reset per-turn state; may read only prices and the latency profile."""

    @abstractmethod
    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:
        raise NotImplementedError

    def timing_label(self, obs: StepObs, memory: dict) -> str:
        return "sentence_end" if obs.is_final else "partial"

    def decide_turn(self, ctx: TurnCtx) -> TurnDecision:
        steps = build_steps(ctx, self.cfg)
        self.begin_turn(TurnStatic(prices=dict(ctx.prices or {}),
                                   retrieval=dict((ctx.signals or {}).get("retrieval") or {})))
        memory: dict = {}
        trace: List[StepTrace] = []
        for obs in steps:
            action, record = self.step(obs, memory)
            record.action = action
            trace.append(record)
            if action == RETRIEVE:
                return TurnDecision(
                    dispatch_t_ms=obs.available_ms,
                    timing_label=self.timing_label(obs, memory),
                    query_text=obs.text,
                    reason=f"{self.name}: retrieve@k={obs.k} t={obs.available_ms}ms | {record.note}",
                    trace=trace,
                    dispatch_step=obs.k,
                    dispatch_partial_index=obs.partial_index,
                )
            if action == SKIP:
                return TurnDecision(
                    dispatch_t_ms=None, timing_label="skip", query_text=None,
                    reason=f"{self.name}: skip@k={obs.k} | {record.note}", trace=trace,
                )
            if action != WAIT:
                raise ValueError(f"{self.name}: unknown action {action!r}")
            if obs.is_final:
                raise RuntimeError(f"{self.name} returned WAIT at the end-of-turn step")
        raise RuntimeError("build_steps produced no end-of-turn step")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_FACTORIES: Dict[str, Tuple[str, str, dict]] = {
    "ours_v2": ("controller.baselines.ours_v2", "OursV2", {}),
    "b9_v2": ("controller.baselines.b9_v2", "B9V2", {}),
    "b9plus_v2": ("controller.baselines.b9plus_v2", "B9PlusV2", {}),
    "oracle_v2": ("controller.baselines.oracle_v2", "OracleV2", {}),
    "b0_v2": ("controller.baselines.unpriced_v2", "B0V2", {}),
    "b1_v2_sentence_end": ("controller.baselines.unpriced_v2", "B1V2", {"policy": "sentence_end"}),
    "b1_v2_immediate": ("controller.baselines.unpriced_v2", "B1V2", {"policy": "immediate"}),
    "b1_v2_fixed_early_proxy": ("controller.baselines.unpriced_v2", "B1V2", {"policy": "fixed_early_proxy"}),
    "b1_v2_forecast": ("controller.baselines.unpriced_v2", "B1V2", {"policy": "forecast"}),
    "b4_v2": ("controller.baselines.unpriced_v2", "B4V2", {}),
    "b5_v2": ("controller.baselines.unpriced_v2", "B5V2", {}),
    "b6_v2": ("controller.baselines.unpriced_v2", "B6V2", {}),
    "b8_v2": ("controller.baselines.unpriced_v2", "B8V2", {}),
}

PRICED_V2 = ("ours_v2", "b9_v2", "b9plus_v2")


def list_controllers_v2() -> List[str]:
    return sorted(_FACTORIES)


def get_controller_v2(name: str, cfg: Optional[dict] = None, **params) -> SequentialController:
    import importlib

    if name not in _FACTORIES:
        raise KeyError(f"no v2 controller {name!r}; known: {list_controllers_v2()}")
    module, cls_name, defaults = _FACTORIES[name]
    cls = getattr(importlib.import_module(module), cls_name)
    ctrl = cls(cfg=cfg, **{**defaults, **params})
    ctrl.name = name
    return ctrl


# ---------------------------------------------------------------------------
# Model expected utility of a decision (theory checks, diagnostics)
# ---------------------------------------------------------------------------
def model_utility(decision: TurnDecision, ctx: TurnCtx, cfg: Optional[dict] = None) -> float:
    """
    The shared v2 objective evaluated at the decision's dispatch step with
    that step's OWN terms (belief, rho, waste, exposure from the turn's
    remaining-speech prediction): what the decision is worth under the
    model, whichever controller made it. 0 for a skip. On synthetic data
    whose predictions equal the truth this is the true expected utility,
    which is what tests/test_v2_theory.py compares controllers on.
    """
    if not decision.retrieve:
        return 0.0
    from controller.baselines._objective_v2 import (
        latency_dist_for,
        load_belief_calibration,
        quality_model,
        read_prices_v2,
        step_terms,
        utility_retrieve,
    )

    cfg = cfg if cfg is not None else load_e3v2_config()
    obs = build_steps(ctx, cfg)[decision.dispatch_step]
    lat = latency_dist_for((ctx.signals or {}).get("retrieval"), cfg)
    terms = step_terms(obs, cfg, load_belief_calibration(cfg), lat)
    return utility_retrieve(terms.belief, terms.rho, terms.exposed_ms, terms.waste,
                            read_prices_v2(ctx.prices, cfg), quality_model(cfg))
