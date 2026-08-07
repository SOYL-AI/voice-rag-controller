"""
decide.py — the controller's decision function.

This is "the decision brain" the top-level README promises: the single place
the pipeline asks, every turn, "what should we do right now?" Phase 1 always
answered "nothing". Phase 2 replaces that with the first real rule: decide
whether to retrieve, based on how uncertain the state tracker still is about
what the user wants.

See controller/README.md's Phase 2 section for the rule this implements.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Imported only for type checkers/IDEs, not at runtime. decide() only calls
# log_writer.log_event(...), so it doesn't actually need the LogWriter class
# to exist at import time — this keeps the controller usable even in a
# context where eventlog/ isn't on the path yet (e.g. a standalone test).
if TYPE_CHECKING:
    from eventlog.log_writer import LogWriter

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"

# Used only if configs/config.yaml can't be read at all (e.g. a standalone
# unit test that doesn't ship the file). Mirrors the values committed in
# configs/config.yaml under controller.decision — keep the two in sync.
_DEFAULT_DECISION_CFG = {
    "key_slots": ["intent", "room", "time", "people"],
    "unknown_fraction_threshold": 0.5,
}


def _load_decision_config(path: Optional[str] = None) -> dict:
    """
    Read the `controller.decision` section of configs/config.yaml.

    Deliberately not llm.config.load_config() — that function returns only
    the `llm:` section (Ryan's). Per configs/README.md, each component owns
    its own top-level section, so controller/ reads its own here.

    Falls back to _DEFAULT_DECISION_CFG (merged under whatever the file did
    have) if the file is missing, so decide() stays usable without a config
    file present, same as it stays usable without a log_writer.
    """
    import yaml  # imported here so `import controller.decide` doesn't require it

    cfg_file = Path(path) if path else Path(
        os.environ.get("VOICE_RAG_CONFIG") or _DEFAULT_CONFIG_PATH
    )
    try:
        with open(cfg_file, "r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return dict(_DEFAULT_DECISION_CFG)

    decision_cfg = (full.get("controller") or {}).get("decision") or {}
    return {**_DEFAULT_DECISION_CFG, **decision_cfg}


def decide(
    belief: Any,
    log_writer: Optional["LogWriter"] = None,
    turn_id: Optional[int] = None,
) -> dict:
    """
    Decide what the system should do right now.

    This is the stable interface every other component wires against:
    decide(belief) -> action. The signature is deliberately final — only
    the body changes as later phases add more signals.

    The Phase 2 rule reads one signal: the state tracker's slot-fill state,
    expected at belief["tracker_slots"] (the dict llm.tracker.track()
    returns — see llm/tracker.py). The tracker doesn't expose a numeric
    confidence; its uncertainty signal is which slots are still "unknown",
    which is a documented, correct value for "not stated yet", not an
    error. From that:

    - If "intent" is unknown, we don't yet know what the user wants at
      all, so we always retrieve — there's no safe way to skip.
    - Otherwise, if at least `unknown_fraction_threshold` of the
      configured `key_slots` are unknown, the tracker is judged too
      uncertain to trust and we retrieve.
    - Otherwise we skip retrieval.

    `key_slots` and `unknown_fraction_threshold` come from
    configs/config.yaml under controller.decision, never hardcoded here,
    so the rule can be tuned for experiments without touching this file.
    A missing/empty tracker_slots dict makes every key slot read as
    "unknown" (see _load_decision_config's default and the lookup below),
    which is the safe default: retrieve rather than guess.

    Args:
        belief: the current situation snapshot. Phase 2 reads
            belief["tracker_slots"]; other keys (transcript, timing, model
            confidence) are accepted but not yet used by this rule.
        log_writer: if given (together with turn_id), the decision is
            logged as a controller_action event via log_writer.log_event(),
            per eventlog/schema.md.
        turn_id: which turn this decision belongs to. Required alongside
            log_writer for logging to happen.

    Returns:
        {"action": "retrieve", "reason": "..."} or
        {"action": "skip_retrieval"}. A dict rather than a bare string
        because actions carry parameters too — keeping the shape
        consistent means callers never have to special-case it.
    """
    belief = belief or {}
    slots = belief.get("tracker_slots") or {}
    decision_cfg = _load_decision_config()

    key_slots = decision_cfg.get("key_slots") or _DEFAULT_DECISION_CFG["key_slots"]
    threshold = decision_cfg.get(
        "unknown_fraction_threshold",
        _DEFAULT_DECISION_CFG["unknown_fraction_threshold"],
    )

    unknown_count = sum(1 for s in key_slots if slots.get(s, "unknown") == "unknown")
    unknown_fraction = (unknown_count / len(key_slots)) if key_slots else 0.0
    intent_unknown = slots.get("intent", "unknown") == "unknown"
    threshold_hit = unknown_fraction >= threshold

    if intent_unknown or threshold_hit:
        reasons = []
        if intent_unknown:
            reasons.append("intent unknown")
        if threshold_hit:
            reasons.append(
                f"{unknown_count}/{len(key_slots)} key slots unknown "
                f"(>= {threshold:.0%} threshold)"
            )
        action = {"action": "retrieve", "reason": "; ".join(reasons)}
    else:
        action = {"action": "skip_retrieval"}

    # Only log if the caller gave us somewhere to write and a turn to
    # attribute it to — decide() must work standalone (e.g. in tests)
    # without a log_writer.
    if log_writer is not None and turn_id is not None:
        log_writer.log_event(
            actor="system",
            event_type="controller_action",
            payload=action,
            turn_id=turn_id,
        )

    return action


if __name__ == "__main__":
    # Roughly the shape the pipeline passes in: transcript plus the
    # tracker's slot dict. Two examples: one where the tracker knows
    # nothing yet (should retrieve), one where it's confident (should skip).
    uncertain_belief = {
        "transcript_so_far": "what time does the pool close",
        "tracker_slots": {
            "intent": "unknown",
            "room": "unknown",
            "time": "unknown",
            "people": "unknown",
            "confirmed": False,
        },
    }
    confident_belief = {
        "transcript_so_far": "book the spa for two people at 4pm",
        "tracker_slots": {
            "intent": "book_spa",
            "room": "unknown",
            "time": "4pm",
            "people": 2,
            "confirmed": False,
        },
    }

    for label, belief in [("uncertain", uncertain_belief), ("confident", confident_belief)]:
        action = decide(belief)
        print(f"decide({label} belief)\n  -> {action}")
