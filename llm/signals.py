"""Phase 2 belief-signal assembly owned by ``llm/``.

The shared event log already stores retrieval scores in ``retrieval_result``
and model data in ``llm_final``. This module gives the integration layer one
stable way to combine those values with the tracker's uncertainty without
changing another owner's pipeline or event schema on Ryan's branch.
"""

from __future__ import annotations

from typing import Iterable, Optional

from llm.tracker import uncertainty as tracker_uncertainty


SIGNAL_SCHEMA_VERSION = 1


def retrieval_summary(scores: Optional[Iterable[float]], score_kind="distance") -> dict:
    """Summarize raw retrieval scores while preserving their direction.

    FAISS currently emits L2 distances (lower is better). Other retrievers
    commonly emit similarity scores (higher is better), so ``score_kind`` is
    explicit and recorded with the values instead of silently assuming one.
    """
    values = [float(value) for value in (scores or [])]
    if score_kind not in {"distance", "similarity"}:
        raise ValueError("score_kind must be 'distance' or 'similarity'")
    if not values:
        return {
            "available": False,
            "score_kind": score_kind,
            "scores": [],
            "best_score": None,
            "top_two_margin": None,
        }

    ordered = sorted(values, reverse=(score_kind == "similarity"))
    margin = None
    if len(ordered) >= 2:
        margin = (
            ordered[0] - ordered[1]
            if score_kind == "similarity"
            else ordered[1] - ordered[0]
        )
    return {
        "available": True,
        "score_kind": score_kind,
        "scores": values,
        "best_score": ordered[0],
        "top_two_margin": margin,
    }


def build_signal_snapshot(
    generation=None,
    retrieval_scores=None,
    retrieval_score_kind="distance",
    tracker_slots=None,
    tracker_parsed_ok=True,
    tracker_cfg=None,
    low_confidence_threshold=0.5,
) -> dict:
    """Build the full per-turn Phase 2 signal payload.

    This is deliberately side-effect free. Renya can attach the returned
    object to the approved shared event/belief location without Ryan's branch
    editing ``pipeline/``, ``controller/``, or the frozen event schema.
    """
    model_signals = (
        generation.uncertainty_signals(low_confidence_threshold)
        if generation is not None
        else {}
    )
    return {
        "signal_schema_version": SIGNAL_SCHEMA_VERSION,
        "model": {
            "available": generation is not None,
            "model": getattr(generation, "model", None),
            "token_count": len(getattr(generation, "tokens", []) or []),
            "uncertainty": model_signals,
        },
        "retrieval": retrieval_summary(retrieval_scores, retrieval_score_kind),
        "tracker": tracker_uncertainty(
            tracker_slots or {}, cfg=tracker_cfg, parsed_ok=tracker_parsed_ok
        ),
    }
