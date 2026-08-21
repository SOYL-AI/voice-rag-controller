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

Instead, per scenario, we run Experiment 1's closed-book protocol (the
question alone, no retrieved passages) and freeze its belief-signal snapshot
(llm/signals.py's build_signal_snapshot()) before any baseline runs. Every
baseline in the sweep is then handed the identical frozen snapshot via
pipeline/spine.py's run_conversation(..., signals=...) — B0-B9 differ only
in decision logic, never in what they were shown. Same pattern as Experiment
2's frozen retrieve/skip mask and frozen query rewrite (simulator/
batch_runner.py): precompute once, outside any timed/compared path, freeze
it, replay it identically everywhere.

---------------------------------------------------------------------------
WHY N DRAWS AND NOT ONE — REPRODUCIBILITY, AND CONSISTENCY WITH E1
---------------------------------------------------------------------------
This module used to take ONE draw, with no seed, at whatever configs/
config.yaml's llm.generation said. Two problems, both of which reach the
committed price grid:

1. NOT REPRODUCIBLE. Temperature 0.2 is stochastic, and with no seed the
   real backend returns a different sample every run. The derived grid
   (simulator/e3_grid.py) moved by roughly 10% between two runs that
   differed in nothing else — a "result" that is a resample, not a change.

2. NOT THE QUANTITY THE CALIBRATION WAS FITTED ON. The VoI estimator is a
   sigmoid in max_entropy_nats fitted on E1's 180 real samples
   (controller/baselines/e1_calibration.json), and E1 produced those samples
   by generating each of its 36 questions under FIVE seeds at temperature
   0.2 (llm/experiment1/config.json's "seeds" and "generation", consumed in
   run.py's per-seed loop). A single unseeded draw is one arbitrary member
   of that per-question sample, not a stable estimate of it.

So precompute_signals() now runs E1's protocol: one generation per E1 seed,
at E1's temperature/top_p/max_tokens, averaged per signal across the draws.
That gives an EXPECTED entropy under the same distribution E1 sampled, and
it is exactly reproducible because the seeds are fixed.

Deliberately NOT temperature 0. Greedy decoding is trivially reproducible,
but it samples a different distribution than the one the calibration was fit
on — it would shift the entropy distribution away from E1's temp-0.2
samples, and the sigmoid's slope/intercept would no longer be about the
numbers being fed to it. Reproducibility is bought with fixed seeds instead,
which costs nothing in consistency.

*** THE PROTOCOL IS READ FROM E1'S OWN CONFIG, NOT RESTATED HERE. *** Seeds,
temperature, top_p, max_tokens and low_confidence_threshold all come from
llm/experiment1/config.json (the same file llm/experiment1/run.py reads), so
E3's precompute cannot drift from the run its calibration was fitted on. In
particular max_tokens is E1's 80, not config.yaml's 256: max_entropy_nats is
a MAX over emitted tokens, so a longer generation budget raises it
mechanically, and a 256-token E3 draw compared against an 80-token E1 fit
would be a silent scale mismatch on the calibrated signal.

One caveat, checked rather than assumed: the calibration was fitted on E1's
180 INDIVIDUAL samples, not on its 36 per-question means, so averaging could
in principle feed the sigmoid a narrower distribution than it was fitted
across. On E1's own numbers it does not, because the seed-to-seed variation
is small next to the question-to-question variation the fit is actually
about — within-question sd 0.0325 nats (median seed spread 0.018) against a
between-question sd of 0.4527. Collapsing the 180 samples to 36 per-question
means moves the distribution's sd from 0.4565 to 0.4527, under 1%, and
leaves the mean unchanged at 1.2363. The averaged signal is therefore the
same quantity at the same scale, with the sampling noise taken out.

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
this runs against the real model. (MockClient seeds its RNG on the prompt
and the seed, so it was already reproducible; the fix above is what makes
the REAL backend reproducible.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.client import make_client
from llm.config import load_config as load_llm_config
from llm.experiment1.run import (
    DEFAULT_EXPERIMENT_CONFIG as E1_CONFIG_PATH,
    EXPERIMENT_SYSTEM_PROMPT,
)
from llm.signals import build_signal_snapshot
from llm.tracker import track_detailed

# Bumped when the precompute protocol changes in a way that makes previously
# cached signals incomparable to newly computed ones. simulator/e3_grid.py
# compares this against its committed cache and rebuilds on a mismatch, so a
# cache holding one-draw signals cannot be silently reused under the
# multi-draw protocol. Version 1 was the single unseeded draw.
PRECOMPUTE_PROTOCOL_VERSION = 2


def load_e1_protocol(config_path: Path = E1_CONFIG_PATH) -> dict:
    """
    Read the generation protocol out of Experiment 1's own config.

    Returns {"seeds", "temperature", "top_p", "max_tokens",
    "low_confidence_threshold", "source"}.

    Read rather than restated so E3's precompute cannot drift from the run
    its VoI calibration was fitted on — see this module's docstring. Every
    key is required: a default silently substituted here would be a protocol
    difference from E1 that nothing downstream could detect.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        experiment_cfg = json.load(handle)

    try:
        generation = experiment_cfg["generation"]
        protocol = {
            "seeds": list(experiment_cfg["seeds"]),
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "max_tokens": generation["max_tokens"],
            "low_confidence_threshold": experiment_cfg["low_confidence_threshold"],
        }
    except KeyError as exc:
        raise ValueError(
            f"{config_path} is missing {exc} — E3's precompute reads its "
            f"generation protocol from Experiment 1's config so the two "
            f"cannot drift apart; it will not guess a default."
        ) from exc

    if not protocol["seeds"]:
        raise ValueError(f"{config_path} lists no seeds; cannot average draws.")

    try:
        source = str(config_path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        source = str(config_path)
    protocol["source"] = source
    return protocol


def _closed_book_messages(question: str) -> list:
    """Experiment 1's exact prompt shape (llm/experiment1/run.py's
    _messages()) — question only, no passages, no retrieval."""
    return [
        {"role": "system", "content": EXPERIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _mean_across_draws(per_draw: list) -> tuple:
    """
    Average each uncertainty signal across the per-draw dicts.

    Returns (means, n_by_signal). A signal is averaged over the draws where
    it is actually present and not None — llm/client.py's
    uncertainty_signals() omits the entropy keys entirely when the server
    returned no top-k alternatives, and dropping a draw is more honest than
    imputing a zero into a MAX statistic. n_by_signal records how many draws
    each mean was taken over, so a signal that was only intermittently
    available is visible rather than looking like a clean 5-draw mean.
    """
    keys = []
    for draw in per_draw:
        for key in draw:
            if key not in keys:
                keys.append(key)

    means, n_by_signal = {}, {}
    for key in keys:
        values = [
            float(draw[key])
            for draw in per_draw
            if draw.get(key) is not None
        ]
        n_by_signal[key] = len(values)
        means[key] = (sum(values) / len(values)) if values else None
    return means, n_by_signal


def precompute_signals(
    question: str, client=None, cfg: Optional[dict] = None
) -> dict:
    """
    Run Experiment 1's closed-book protocol for `question` — one generation
    per E1 seed at E1's temperature/top_p/max_tokens — plus one tracker
    extraction, and return the frozen llm/signals.py snapshot with every
    uncertainty signal averaged across the draws:

        {
          "signal_schema_version": ...,
          "model": {"available": True, ..., "uncertainty": {
              "negative_mean_logprob": ..., "perplexity": ...,
              "mean_entropy_nats": ..., "max_entropy_nats": ..., ...
          }},                                  # each one a mean over N draws
          "retrieval": {"available": False, ...},  # see module docstring
          "tracker": {"unknown_fraction": ..., "parsed_ok": ..., ...},
          "precompute": {...},                 # provenance, see below
        }

    The "precompute" block records the protocol this snapshot was produced
    under — protocol version, N, the seeds, temperature/top_p/max_tokens,
    the E1 config they were read from, the averaging rule, and the per-draw
    signal values. Provenance, not decoration: the averaged numbers are only
    interpretable against E1's calibration if they were produced under E1's
    sampling distribution, and a consumer that cannot see the protocol
    cannot check that. The per-draw values are also what makes the spread
    behind each mean auditable after the fact.

    ONE tracker call, not N. E1 does the same — llm/experiment1/run.py
    builds its tracker cache once per question, outside the per-seed loop —
    because the tracker runs at temperature 0.0 (configs/config.yaml's
    llm.tracker) and is extraction, not sampling. Averaging its output
    across draws would average a constant.

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
    protocol = load_e1_protocol()

    messages = _closed_book_messages(question)
    per_draw_uncertainty, token_counts, model_name = [], [], None
    for seed in protocol["seeds"]:
        generation = client.generate(
            messages,
            temperature=protocol["temperature"],
            top_p=protocol["top_p"],
            max_tokens=protocol["max_tokens"],
            seed=seed,
        )
        per_draw_uncertainty.append(
            generation.uncertainty_signals(protocol["low_confidence_threshold"])
        )
        token_counts.append(len(generation.tokens))
        model_name = generation.model

    mean_uncertainty, n_by_signal = _mean_across_draws(per_draw_uncertainty)

    tracker_slots, _raw_text, tracker_parsed_ok = track_detailed(
        [question], client=client, cfg=cfg
    )

    # build_signal_snapshot() summarizes ONE generation, so it is called with
    # no generation and the model block is filled in from the draws instead —
    # passing one draw and then overwriting its uncertainty would leave that
    # draw's token_count behind, describing a snapshot that is not the one
    # being returned.
    snapshot = build_signal_snapshot(
        tracker_slots=tracker_slots,
        tracker_parsed_ok=tracker_parsed_ok,
        tracker_cfg=cfg,
        low_confidence_threshold=protocol["low_confidence_threshold"],
    )
    snapshot["model"] = {
        "available": bool(mean_uncertainty),
        "model": model_name,
        # The mean over draws, like every other number in this block. Kept a
        # float on purpose: rounding it to a whole token would imply a single
        # generation that does not exist.
        "token_count": sum(token_counts) / len(token_counts),
        "uncertainty": mean_uncertainty,
    }
    snapshot["precompute"] = {
        "protocol_version": PRECOMPUTE_PROTOCOL_VERSION,
        "protocol": "e1_matched_multi_seed_mean",
        "n_draws": len(protocol["seeds"]),
        "seeds": list(protocol["seeds"]),
        "temperature": protocol["temperature"],
        "top_p": protocol["top_p"],
        "max_tokens": protocol["max_tokens"],
        "low_confidence_threshold": protocol["low_confidence_threshold"],
        "protocol_source": protocol["source"],
        "aggregation": (
            "arithmetic mean over draws, per signal; draws where a signal is "
            "absent or None are skipped (see n_draws_by_signal)"
        ),
        "n_draws_by_signal": n_by_signal,
        "per_draw_uncertainty": per_draw_uncertainty,
        "token_counts_per_draw": token_counts,
        "tracker_draws": 1,
        "tracker_note": (
            "one temperature-0.0 extraction, matching llm/experiment1/run.py's "
            "per-question tracker cache; not averaged"
        ),
    }
    return snapshot
