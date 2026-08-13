"""
e3_signals.py — Experiment 3's frozen per-scenario belief-signal precompute.

Baselines like B4 (FLARE confidence) and B6 (DRAGIN entropy) need the
model's belief signals — token entropy, logprob-derived confidence, tracker
uncertainty — at decision time, the same signals Experiment 1 validated
(llm/experiment1/run.py). Computing them live, once per baseline per turn,
would mean every baseline in the sweep pays for its own extra model call
before it even gets to compare against another baseline's decision logic —
an unfair, costly coupling this module exists to remove, and the
pre-registration's "differ only in decision logic" requirement forbids it
outright.

Instead, per scenario, we run ONE closed-book generation (the question
alone, no retrieved passages — exactly Experiment 1's setup) and freeze its
belief-signal snapshot (llm/signals.py's build_signal_snapshot()) before any
baseline runs. Every baseline in the sweep is then handed the identical
frozen snapshot via pipeline/spine.py's run_conversation(..., signals=...)
— B0-B9 differ only in decision logic, never in what they were shown. Same
pattern as Experiment 2's frozen retrieve/skip mask and frozen query
rewrite (simulator/batch_runner.py): precompute once, outside any
timed/compared path, freeze it, replay it identically everywhere.

Why closed-book, and why reuse Experiment 1's exact prompt: Experiment 1
validated that the model's OWN uncertainty on a question — answered without
retrieval, before any passage biases its confidence — predicts whether
retrieval was actually needed. A baseline gating on entropy/confidence
should trigger on that same signal, not a differently-prompted
approximation of it. So this reuses llm/experiment1/run.py's
EXPERIMENT_SYSTEM_PROMPT verbatim rather than defining a second, possibly
drifting, closed-book prompt here.

retrieval stays unavailable ({"available": False, ...}) in this snapshot on
purpose: retrieval hasn't run yet at the point a baseline is deciding
whether to retrieve at all, so there is no honest score to freeze. A
baseline that wants last-turn retrieval scores is a multi-turn concern
outside this stub's single-turn scenarios.

On the mock backend (configs/config.yaml's llm.backend, the default) this
returns mock signals — synthetic but shaped exactly like the real ones, per
llm/client.py's MockClient. Fine for plumbing; real signal values come once
this runs against the real model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.client import make_client
from llm.config import load_config as load_llm_config
from llm.experiment1.run import EXPERIMENT_SYSTEM_PROMPT
from llm.signals import build_signal_snapshot
from llm.tracker import track_detailed


def _closed_book_messages(question: str) -> list:
    """Experiment 1's exact prompt shape (llm/experiment1/run.py's
    _messages()) — question only, no passages, no retrieval."""
    return [
        {"role": "system", "content": EXPERIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def precompute_signals(
    question: str, client=None, cfg: Optional[dict] = None
) -> dict:
    """
    Run one closed-book generation and one tracker extraction for
    `question`, and return the frozen llm/signals.py snapshot:

        {
          "signal_schema_version": ...,
          "model": {"available": True, ..., "uncertainty": {
              "negative_mean_logprob": ..., "perplexity": ...,
              "mean_entropy_nats": ..., "max_entropy_nats": ..., ...
          }},
          "retrieval": {"available": False, ...},  # see module docstring
          "tracker": {"unknown_fraction": ..., "parsed_ok": ..., ...},
        }

    Call this once per scenario, before any baseline runs — see
    simulator/e3_sweep.py's run_one_scenario() — and pass the result into
    pipeline/spine.py's run_conversation(signals=...) so every baseline
    compared against this scenario sees the identical, frozen signals.

    client/cfg: pass a shared llm.client instance / llm.config.load_config()
    result to avoid re-opening a client per scenario in a sweep. Built
    fresh here if omitted, same defaulting convention as
    llm/client.py's generate_answer().
    """
    cfg = cfg or load_llm_config()
    client = client or make_client(cfg)

    generation = client.generate(_closed_book_messages(question))
    tracker_slots, _raw_text, tracker_parsed_ok = track_detailed(
        [question], client=client, cfg=cfg
    )

    return build_signal_snapshot(
        generation=generation,
        tracker_slots=tracker_slots,
        tracker_parsed_ok=tracker_parsed_ok,
        tracker_cfg=cfg,
    )
