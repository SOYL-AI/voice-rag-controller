"""
e3v2_synthetic.py — synthetic turns, signals and Phase A cell stores for the
v2 controllers' unit/theory tests and for a wiring smoke run of
simulator/e3v2_decide.py. No model, ASR or judge is involved; nothing here
is a result.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.sequential_v2 import TurnCtx  # noqa: E402

DEFAULT_WORDS = ("what is the late payment fee on the northwind postpaid mobile plan "
                 "this month").split()


def synthetic_turn(
    item_id: str,
    beliefs: Sequence[float],
    final_belief: float,
    interval_ms: float = 300.0,
    user_end_ms: Optional[float] = None,
    words: Sequence[str] = DEFAULT_WORDS,
    retrieval: Optional[dict] = None,
    prices: Optional[dict] = None,
    remaining_pred: Optional[Sequence[Optional[float]]] = None,
    entry_extra: Optional[Sequence[dict]] = None,
    final_extra: Optional[dict] = None,
    condition: str = "synthetic",
    oracle: Optional[dict] = None,
    first_words: int = 4,
) -> TurnCtx:
    """
    One turn with len(beliefs) partials at t = interval*(k+1) and the end of
    turn at user_end_ms (default interval*(n+1)). beliefs are pre-calibrated
    P(needed) values on each prefix ('p_needed'); the true remaining speech is
    the predictor's output unless remaining_pred overrides it (None entries
    = no prediction at that partial). The k-th partial carries
    first_words + k words, so every partial is query-ready by default.
    """
    n = len(beliefs)
    end = float(user_end_ms if user_end_ms is not None else interval_ms * (n + 1))
    partials, by_prefix, preds = [], [], []
    for k, b in enumerate(beliefs):
        t = interval_ms * (k + 1)
        m = min(len(words), first_words + k)
        partials.append({"t_ms": t, "text": " ".join(words[:m]), "stable": False})
        entry = {"t_ms": t, "word_frac": m / len(words), "p_needed": float(b)}
        if entry_extra:
            entry.update(entry_extra[k])
        by_prefix.append(entry)
        r = (end - t) if remaining_pred is None else remaining_pred[k]
        if r is not None:
            preds.append({"t_ms": t, "remaining_ms": float(r)})
    full = {"t_ms": end, "word_frac": 1.0, "kind": "full", "p_needed": float(final_belief)}
    if final_extra:
        full.update(final_extra)
    by_prefix.append(full)
    signals = {
        "model": {"available": True, "uncertainty_by_prefix": by_prefix},
        "timing": {
            "speech_onset_ms": 0, "partials": partials, "predicted_remaining_speech_ms": preds,
            "user_end_ms": end, "partial_interval_ms": interval_ms,
            "asr_final": {"t_ms": end, "text": " ".join(words)},
        },
        "retrieval": dict(retrieval or {"latency_profile": "synthetic",
                                        "latency_quantiles_ms": {"p50": 600}}),
    }
    return TurnCtx(item_id=item_id, signals=signals, prices=dict(prices or {}), condition=condition,
                   transcript_final=" ".join(words), oracle=oracle)


def synthetic_cfg_overrides(**extra) -> dict:
    """Config overrides that make the objective use the synthetic p_needed
    beliefs directly and keep everything else explicit."""
    base = {
        "calibration": {"inline": {"signal": "p_needed", "kind": "identity", "calibrated": False,
                                   "source": "synthetic"}, "signal": "p_needed"},
        "timing": {"eot_detection_ms": 0},
        "ours": {"belief_transitions_artifact": None},
        "estimates_artifact": None,
    }
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def synthetic_dataset(n_items: int = 40, seed: int = 0, n_partials: int = 6,
                      interval_ms: float = 300.0, profiles: Optional[Dict[str, dict]] = None):
    """
    Items, turns and a Phase A cell store for a smoke run of the evaluator.
    Returns (turns_by_condition_item, cell_store_records, labels, items).
    """
    rng = random.Random(seed)
    items, turns, cells, labels = [], {}, [], {}
    for i in range(n_items):
        item_id = f"syn-{i:04d}"
        needed = rng.random() < 0.5
        labels[item_id] = needed
        base = 0.75 if needed else 0.25
        beliefs = [min(1, max(0, base + rng.gauss(0, 0.25) * (1 - k / n_partials)))
                   for k in range(n_partials)]
        final_b = min(1, max(0, base + rng.gauss(0, 0.1)))
        end = interval_ms * (n_partials + 1) + rng.uniform(-150, 150)
        h = 0.2 + (1.2 if needed else 0.2) + rng.gauss(0, 0.2)
        extra = [{"max_entropy_nats": h + rng.gauss(0, 0.3), "min_token_prob": max(0.01, 0.9 - 0.4 * h)}
                 for _ in range(n_partials)]
        turn = synthetic_turn(item_id, beliefs, final_b, interval_ms, user_end_ms=end,
                              entry_extra=extra,
                              final_extra={"max_entropy_nats": h, "min_token_prob": max(0.01, 0.9 - 0.4 * h)})
        turns[("synthetic", item_id)] = turn
        gold = f"doc-{i}-gold"
        items.append({"id": item_id, "question": turn.transcript_final, "gold_doc_ids": [gold],
                      "split": "dev"})
        variants = []
        for k in range(n_partials):
            hit = k >= n_partials // 2 or rng.random() < 0.3
            docs = [gold, f"doc-{i}-x"] if hit else [f"doc-{i}-y{k}", f"doc-{i}-z"]
            variants.append(docs)
        final_docs = [gold, f"doc-{i}-x"]
        for k, docs in enumerate(variants + [final_docs]):
            cells.append({"kind": "retrieval", "item_id": item_id, "condition": "synthetic",
                          "partial_index": None if k == n_partials else k, "retrieved_doc_ids": docs})
        for seed_ in (11, 23):
            cb_correct = (not needed) and rng.random() < 0.9 or (needed and rng.random() < 0.15)
            cells.append({"kind": "cell", "item_id": item_id, "seed": seed_, "condition": "synthetic",
                          "path": "skip", "query_variant": "none", "answer_correct": bool(cb_correct),
                          "retrieved_doc_ids": [], "n_prompt_tokens": 120})
            seen = set()
            for docs in variants + [final_docs]:
                from simulator.e3v2_decide import query_variant_for  # lazy: avoid import cycle
                qv = query_variant_for(docs)
                if qv in seen:
                    continue
                seen.add(qv)
                has_gold = gold in docs
                p = (0.9 if has_gold else 0.3) if needed else (0.35 if has_gold else 0.3)
                cells.append({"kind": "cell", "item_id": item_id, "seed": seed_, "condition": "synthetic",
                              "path": "retrieve", "query_variant": qv,
                              "answer_correct": rng.random() < p, "retrieved_doc_ids": docs,
                              "n_prompt_tokens": 600})
    return turns, cells, labels, items
