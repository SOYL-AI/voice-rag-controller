"""
baseline.py — Experiment 3's shared baseline contract.

E3 compares our unified controller against rival baselines (B0-B9) that
independent teammates build in parallel — the pre-registration's rival A
("three independent best controllers sharing nothing") and rival B ("the
same three given a shared latency/compute price but no shared state"), plus
whatever simpler heuristics B0-B8 turn out to be. For that comparison to be
apples-to-apples, every baseline has to plug into the exact same sweep the
exact same way: same input snapshot, same output shape, same logged event.
This module is that shared contract — the E3 equivalent of eventlog/
schema.md being the frozen event-log contract for Phase 1.

Three pieces:

  Ctx       — read-only snapshot of "everything a baseline is allowed to
              look at" for one turn's decision.
  Decision  — the answer shape every baseline must return, regardless of
              how it decides.
  BaselineController — the base class every baseline subclasses, plus a
              name -> class registry so pipeline/spine.py and simulator/
              e3_sweep.py can select a baseline by string name.

controller/decide.py's decide() stays exactly as-is and untouched — it's
what "Ours" wraps (see controller/baselines/ for that when it lands). This
module does not change it; it only gives every OTHER baseline a way to
plug into the same sweep decide() already plugs into.

See controller/baselines/b0_never_retrieve.py for a fully worked reference
implementation — copy its structure for a new baseline.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Type

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.timing import VALID_POLICIES  # noqa: E402 -- see _REPO_ROOT sys.path fixup above

if TYPE_CHECKING:
    # Same pattern as controller/decide.py: only needed for type checkers,
    # so this module stays importable even where eventlog/ isn't on the
    # path yet (e.g. a standalone baseline unit test).
    from eventlog.log_writer import LogWriter


# ---------------------------------------------------------------------------
# Ctx — what a baseline is allowed to see
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ctx:
    """
    Read-only snapshot of the current turn, handed to BaselineController.
    decide(). Every baseline gets exactly this and nothing else — no
    reaching into the pipeline, no shared globals, so the E3 comparison
    actually isolates decision logic rather than incidental access
    differences between implementations.

    transcript: str
        The transcript this decision is being made over. In the text
        (question=) path this is the scenario question (or, for an
        early-timing decision, whatever prefix of it has been "spoken" so
        far); on the live/synthetic-audio path it would be the current
        partial or final transcript. Never mutated, never re-fetched — a
        baseline sees the same string every time it's called for a given
        decision.

    signals: dict
        Belief signals a baseline may use to decide, keyed by name.
        Currently populated by pipeline/spine.py:
          - "tracker_slots": the dict llm.tracker.track() returns (the
            same state-tracker slot-fill state controller/decide.py's
            live rule reads) — a real, already-available uncertainty
            signal.
        Expected to grow as Experiment 1's other belief signals get wired
        in ahead of the full B1-B9 sweep, e.g.:
          - "max_entropy_nats" / "mean_entropy_nats": per-token output
            entropy from the last model call (see llm/client.py's
            GenerationResult).
          - retrieval scores from the previous turn's retrieval_result
            (retrieval/search.py's FAISS distances — lower is a better
            match).
        A baseline must treat a missing key as "signal not available" (use
        .get(), never []) — not every signal is wired into every call
        site yet, and a baseline that assumes otherwise will break the
        moment it's tested against a call site that hasn't wired that
        signal in.

    elapsed_ms: int
        Milliseconds since this turn started (turn_start_ms in
        pipeline/spine.py's run_conversation), at the moment this decision
        is being requested. 0 if the caller has no turn-start reference.

    user_end_known: bool
        Whether the user has (or, on the text path's modeled speech
        timeline, is known to have) finished speaking as of this decision.
        Mirrors controller/timing.py's user_end_known_at_launch field —
        see mechanism_launch_fields().

    turn_id: int
        Which turn this decision belongs to — forwarded straight to
        log_writer.log_event()'s turn_id, same as decide()/
        controller_decide_forced().

    conversation_id: str
        Which conversation this turn belongs to (LogWriter.conversation_id)
        — informational; most baselines won't need it, but it's here for
        any baseline that wants to key per-conversation state (e.g. a
        memory-budget baseline tracking spend across turns).

    prices: dict
        {"lambda_l": ..., "lambda_c": ...} — the shared latency/compute
        price Experiment 3's rival B and "Ours" (B9) are compared under
        (see paper/preregistration.md's Experiment 3 section). Every other
        baseline (B0-B8, rival A) is expected to ignore this field
        entirely — it's here so B9/Ours don't need a different Ctx shape.
        Defaults to {} (no price given) rather than None, so
        `ctx.prices.get("lambda_l")` is always safe to call without a
        None-check.
    """

    transcript: str
    signals: dict
    elapsed_ms: int
    user_end_known: bool
    turn_id: int
    conversation_id: str
    prices: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decision — what every baseline must return
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    """
    The one return shape every baseline's decide() must produce, no matter
    how it decides internally.

    retrieve: bool
        Whether to retrieve this turn. Maps to controller_decide_forced()'s
        "retrieve" / "skip_retrieval" action.

    timing: str
        WHEN retrieval should launch, if retrieve is True — one of
        controller/timing.py's VALID_POLICIES ("sentence_end", "immediate",
        "fixed_early_proxy"). Validated in __post_init__ below so a typo'd
        policy name fails loudly at decision time, not silently downstream.
        A baseline that never varies timing (e.g. B0) can just always
        return "sentence_end". NOTE (current wiring): pipeline/spine.py's
        baseline= wiring logs this field but does not yet use it to choose
        WHEN retrieval launches — WHEN is still governed by
        controller.timing.policy for the whole conversation, same as
        today. Per-decision timing control is part of the full E3 sweep,
        not this interface stub.

    reason: str
        Human/log-readable justification, same role as decide()'s
        {"reason": ...} — required (not optional) because "why did this
        baseline decide this" is exactly what someone debugging a sweep
        result needs, for every baseline, every turn.

    memory_op: Optional[str] = None
        What memory operation (if any) to perform this turn — e.g.
        "write", "summarize", "evict". None means "no memory operation",
        the correct default until a baseline actually implements memory
        scheduling (Experiment 4 is cut for this deadline per
        paper/preregistration.md, but the field exists now so a baseline
        that does touch memory doesn't need a different Decision shape
        later).
    """

    retrieve: bool
    timing: str
    reason: str
    memory_op: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timing not in VALID_POLICIES:
            raise ValueError(
                f"Decision.timing={self.timing!r} is not one of "
                f"controller.timing.VALID_POLICIES ({sorted(VALID_POLICIES)})."
            )


# ---------------------------------------------------------------------------
# BaselineController — the base class every baseline subclasses
# ---------------------------------------------------------------------------
class BaselineController(ABC):
    """
    Base class for every E3 baseline (B0-B9, Ours). Subclass this, set
    `name`, implement decide(), and register it — see
    controller/baselines/b0_never_retrieve.py for the full template.

    Subclasses must NOT reimplement event logging: call self.log_decision()
    (or the decide_and_log() convenience wrapper) instead, so every
    baseline logs controller_action in exactly the same shape
    controller/decide.py's decide() and pipeline/spine.py's
    controller_decide_forced() already do. That shape is additive here —
    the "baseline"/"timing"/"memory_op"/"lambda_l"/"lambda_c" keys are new
    payload fields, not replacements for "action"/"reason", so a log reader
    written against the Phase 2 schema keeps working unmodified.
    """

    #: Registry key this baseline was registered under — set automatically
    #: by @register(name), or set it yourself if constructing a
    #: BaselineController subclass without the decorator. Never "" in a
    #: real, registered baseline; left as "" here only so an unregistered
    #: subclass fails obviously (a blank "baseline" field in the log)
    #: rather than raising at import time.
    name: str = ""

    @abstractmethod
    def decide(self, ctx: Ctx) -> Decision:
        """
        The one method every baseline implements: given a read-only Ctx
        snapshot, return a Decision. Must be a pure function of `ctx` —
        no reaching into the pipeline, no hidden global state, so every
        baseline is comparable under the same sweep conditions. (Some
        baselines may reasonably keep instance state across calls within
        one baseline instance's lifetime — e.g. a running memory-budget
        counter — but never mutate `ctx` itself, which is frozen.)
        """
        raise NotImplementedError

    def log_decision(
        self, decision: Decision, ctx: Ctx, log_writer: Optional["LogWriter"]
    ) -> dict:
        """
        Log `decision` as a controller_action event, in the same shape
        decide()/controller_decide_forced() use ({"action": ...,
        "reason": ...} as the payload's core), additively carrying which
        baseline made the call, its timing choice, its memory_op (if any),
        and the shared price fields (if ctx.prices has them). Returns the
        payload dict — reused as-is as the "action" dict callers like
        pipeline/spine.py's run_retrieval_pipeline() already expect from
        controller_decide()/controller_decide_forced().

        log_writer=None is allowed (mirrors decide()'s log_writer=None
        support) so a baseline stays unit-testable without a real
        LogWriter — no event is logged in that case, only the dict is
        returned.
        """
        payload = {
            "action": "retrieve" if decision.retrieve else "skip_retrieval",
            "reason": decision.reason,
            "baseline": self.name,
            "timing": decision.timing,
        }
        if decision.memory_op is not None:
            payload["memory_op"] = decision.memory_op
        # Additive, price-bearing baselines only (B9/Ours) — every other
        # baseline's ctx.prices is {} per Ctx's docstring, so these two
        # keys simply don't appear in their logged payload.
        lambda_l = ctx.prices.get("lambda_l")
        lambda_c = ctx.prices.get("lambda_c")
        if lambda_l is not None:
            payload["lambda_l"] = lambda_l
        if lambda_c is not None:
            payload["lambda_c"] = lambda_c

        if log_writer is not None:
            log_writer.log_event(
                actor="system",
                event_type="controller_action",
                payload=payload,
                turn_id=ctx.turn_id,
            )
        return payload

    def decide_and_log(
        self, ctx: Ctx, log_writer: Optional["LogWriter"] = None
    ) -> dict:
        """
        Convenience wrapper: decide(ctx) then log_decision(...) in one
        call — what pipeline/spine.py and simulator/e3_sweep.py actually
        call. Returns the same payload dict log_decision() returns.
        """
        decision = self.decide(ctx)
        return self.log_decision(decision, ctx, log_writer)


# ---------------------------------------------------------------------------
# Registry — select a baseline by name
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type[BaselineController]] = {}
_INSTANCES: Dict[str, BaselineController] = {}


def register(name: str):
    """
    Class decorator: register a BaselineController subclass under `name`.

        @register("b0")
        class NeverRetrieve(BaselineController):
            ...

    Also stamps cls.name = name, so subclasses don't have to set `name`
    themselves and it can never drift from the registry key it's looked up
    under. Raises if `name` is already taken — two baselines silently
    sharing a registry key would make the sweep run whichever one imported
    last, which is exactly the kind of quiet bug this shared contract
    exists to prevent.
    """

    def _decorator(cls: Type[BaselineController]) -> Type[BaselineController]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            # One benign case: a baseline module's own `if __name__ ==
            # "__main__":` self-test being run via `python -m
            # controller.baselines.<mod>`. That import path loads the
            # module twice under two different identities — once as
            # controller.baselines.<mod> (via this package's __init__.py,
            # which runs first) and once as __main__ — so the *same*
            # source line runs @register(name) twice with two distinct
            # class objects that happen to share a __qualname__. That's
            # not a real collision between two different baselines, so
            # don't raise for it; just let the second registration win.
            # A genuine collision (two different real modules both
            # claiming `name`) always has __qualname__ differ or neither
            # side be "__main__", so it still raises below.
            self_test_rerun = (
                "__main__" in (existing.__module__, cls.__module__)
                and existing.__qualname__ == cls.__qualname__
            )
            if not self_test_rerun:
                raise ValueError(
                    f"baseline name {name!r} is already registered to "
                    f"{existing.__module__}.{existing.__name__} — pick a "
                    f"different name for {cls.__module__}.{cls.__name__}."
                )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_baseline(name: str) -> BaselineController:
    """
    Look up a registered baseline by name and return a (lazily
    instantiated, cached) instance of it.

    Raises KeyError with the list of currently-registered names if `name`
    isn't registered — the most common cause is forgetting to `import
    controller.baselines` first (that package's __init__.py is what
    actually runs every baseline module's @register(...) side effect;
    importing controller.baseline alone does not).
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No baseline registered as {name!r}. Registered baselines: "
            f"{sorted(_REGISTRY) or '(none)'}. Did you `import "
            f"controller.baselines` first, so its __init__.py could run "
            f"each baseline module's @register(...) side effect?"
        )
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


def list_baselines() -> list:
    """Names of every currently-registered baseline, sorted. Handy for a
    sweep runner's --help text or input validation."""
    return sorted(_REGISTRY)
