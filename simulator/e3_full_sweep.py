"""
e3_full_sweep.py — Experiment 3's full frontier sweep runner.

This is the real one. simulator/e3_sweep.py is the stub it grew from —
that file proves ONE baseline plugs into the real pipeline at ONE price
point and stops there, deliberately (see its docstring). This file
produces the headline data analysis/e3_frontier.py consumes: every
baseline, crossed with the derived price grid, crossed with E1's five
seeds, over all 50 scenarios, scored.

    python -m simulator.e3_full_sweep --phase-a-only --limit-scenarios 2
    python -m simulator.e3_full_sweep            # both phases, full sweep
    python -m analysis.e3_frontier simulator/e3_sweep_results

---------------------------------------------------------------------------
THE OUTPUT SCHEMA (analysis/e3_frontier.py's validate_records)
---------------------------------------------------------------------------
    simulator/e3_sweep_results/e3_<baseline>_seed<seed>.json

each a JSON LIST of records shaped:

    {
      "baseline": "ours",
      "seed": 11,
      "scenario_id": 1,
      "prices": {"lambda_l": 0.533, "lambda_c": 0.05},   # or null
      "outcome": {"answer_correct": true,
                  "ttfa_ms": 3395.3,
                  "retrievals": 1}
    }

plus a "decision" block (what the baseline actually chose, and why) that
the analysis ignores and a human auditing a surprising point needs. The
validator requires a fixed set of keys and tolerates extras, so carrying
provenance costs nothing downstream.

Priced baselines (ours, b9) emit one record per grid price; unpriced ones
(b0, b4, b6, b8) emit "prices": null and appear as single operating points,
which is E3_BASELINE_INTERFACE.md's intended contrast, not an omission.

---------------------------------------------------------------------------
WHY TWO PHASES
---------------------------------------------------------------------------
The naive sweep is baseline x price x seed x scenario runs of the real
pipeline: 6 baselines, 26 prices for two of them, 5 seeds, 50 scenarios —
14,000 conversations at ~10s each, which is 39 hours of GPU time to
measure a decision function that is deterministic and free.

It is also 14,000 generations of text that is mostly identical, because of
what actually varies:

  * RETRIEVAL IS CONTENT, NOT TIMING. retrieval/search.py's FAISS lookup
    over a frozen query returns the same k passages regardless of when it
    was launched or what price motivated the launch. So the retrieved
    answer for a (scenario, seed) is one string, not 26 x 6 strings.
  * THE CLOSED-BOOK ANSWER LIKEWISE depends only on (scenario, seed).
  * SO answer_correct is a function of (scenario, seed, retrieve-or-not)
    ONLY. Three inputs, not five.
  * TTFA IS the timing-dependent one: retrieval launched during speech
    (immediate / fixed_early_proxy) finishes before it is needed, while
    sentence_end pays for it after user_end. That is E2's whole mechanism,
    and it is why ttfa_ms has to be measured per timing policy rather than
    modelled.
  * RETRIEVALS is just the count, 0 or 1 on a single-turn scenario.

So the measurable space is (scenario, seed, skip|retrieve, timing) — 1500
cells — and everything else is a lookup. Phase A measures the 1500 cells
against the real backend and caches them. Phase B runs every baseline at
every price over the frozen belief signals, reads each decision's outcome
out of that cache, and writes the schema. Phase B does not open a client.

The saving is not the point by itself. The point is that two baselines
that make the SAME decision on a scenario are then scored on the SAME
measured outcome, instead of on two independent samples of a stochastic
decoder that differ by noise. Under the naive sweep, a frontier gap
between Ours and B9 at a price where they agreed would be pure resampling.
Here it is exactly zero, by construction.

---------------------------------------------------------------------------
PHASE A — WHAT IS MEASURED, AND HOW IT RESUMES
---------------------------------------------------------------------------
One cell = one real run_conversation() through pipeline/spine.py:

    force_retrieve = (path == "retrieve")   # E2's mask fixes WHETHER
    VOICE_RAG_TIMING_POLICY = timing        # controller/timing.py, WHEN
    frozen_query = precomputed rewrite      # E2's frozen-rewrite mechanism
    seed = one of E1's five                 # generation, reproducible

Note the sweep does NOT run Phase A under a baseline. A baseline's job is
to pick (retrieve, timing); Phase A's job is to measure what each of those
six (path, timing) combinations costs. Driving it with the E2 mask instead
means the measurement is baseline-independent — which is what makes it
legitimate to share one cell between two baselines that chose it.

Cached per cell at simulator/e3_sweep_cache/cells/. A cell file existing
means that cell is done; re-running skips it. Nothing is written until the
run succeeds, so a crash mid-cell leaves it to be retried, never
half-recorded.

STALE-LOG HAZARD. eventlog/log_writer.py opens in append mode, so a
re-run under a conversation_id that already has a log leaves the previous
run's events in the file ahead of this one's — the hazard
simulator/e3_sweep.py's _read_controller_action() documents and works
around by taking the LAST matching event. This runner does not need that
workaround and does not use it: conversation ids are unique per cell, and
the log file for a cell is DELETED before the cell runs. Every log this
runner reads therefore contains exactly one run, so "first" and "last"
event coincide and there is nothing to get wrong.

---------------------------------------------------------------------------
SCORING — answer_correct
---------------------------------------------------------------------------
An LLM judge at temperature 0 compares the generated answer to the
scenario's expected_answer and returns a strict yes/no on FACTUAL
CORRECTNESS. The judge is Azure OpenAI and is a SEPARATE client from the
system's self-hosted Mistral: Mistral generates the answers, Azure grades
them, and the two are never the same object (see make_judge()). Mistral-7B
graded itself for one run and scored "the spa opens at 9AM" correct
against a 10AM reference, and a wrong wifi network correct — under a judge
that cannot see an error, skipping retrieval costs no accuracy and B0 wins
the frontier by construction. Judge configuration is read from the
environment only (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION); a missing variable is
a hard error, never a downgrade to a weaker judge. `--judge-selftest`
grades three fixed pairs to prove the wiring and the judge's strength
before a sweep.

Swapping the judge does not require regenerating anything. `--rejudge`
re-scores the answers already in the cache with the current judge and
rewrites answers/*.json in place; cells/ (ttfa_ms, retrievals, logs) is
measurement of the SYSTEM, not of the judge, and is not touched, so no
model client is opened and no latency is re-measured. The verdict the
previous judge gave is kept on the record as `previous_verdict`. It is not asked whether retrieval fired, and must not be: a
closed-book answer that happens to state the fact correctly scores true.
Retrieval's cost shows up in the compute axis (the retrievals count) and
in the latency axis, which is exactly the trade-off the frontier is about.
Scoring "did it retrieve" as quality would build the answer into the
question.

Verdicts are cached by sha256 over (judge identity, judge cache version,
scenario_id, expected_answer, answer text) and stored under judge/<judge
slug>/, so an answer generated under five seeds that lands identically is
judged once. The key covers the reference, not just the answer: identical
answer STRINGS occur across scenarios (a refusal, a generic greeting) and
they are correct against different references, so keying on the answer
alone would let one scenario's verdict answer another scenario's question.
It covers the JUDGE for the same class of reason: this cache has already
held a mock run's steered YESes and Mistral's, and if the key ignored who
graded, swapping in a stronger judge would silently re-serve the weak
verdicts and look like it had worked.

---------------------------------------------------------------------------
PHASE B — WHERE THE SIGNALS COME FROM
---------------------------------------------------------------------------
The frozen belief signals are read from the COMMITTED cache at
simulator/e3_scenario_signals.json — the same artifact configs/
e3_price_grid.json was derived from (simulator/e3_grid.py's
load_or_build_signals). Not recomputed here, on purpose: the grid and the
sweep have to be about the same beliefs, and a fresh precompute would
resample them. Ctx is assembled exactly as simulator/e3_divergence_check.py
assembles it, which is exactly what pipeline/spine.py's
controller_decide_baseline() hands a baseline.

elapsed_ms / user_end_known are set to the same fixed values that module
uses. Checked, not assumed: no registered baseline's decide() reads
either field — they read ctx.signals, ctx.prices, and (b5) ctx.transcript.

Seeds do not enter the decision. The signals are frozen per scenario, so
every baseline makes the same decision on a scenario under all five seeds;
the seed varies the measured OUTCOME of that decision, which is where the
frontier's per-seed spread legitimately comes from.

---------------------------------------------------------------------------
BACKEND
---------------------------------------------------------------------------
Phase A must run against the real backend:

    VOICE_RAG_LLM_BACKEND=vllm VOICE_RAG_MODEL=mistral7b_v03 \
    VOICE_RAG_LLM_BASE_URL=<endpoint>/v1 \
    python -m simulator.e3_full_sweep --phase-a-only

Under the mock backend this refuses to write the real cache, in the same
spirit as e3_grid.py's guard on the signals cache: mock draws mixed into a
real cache are undetectable downstream. --allow-mock redirects BOTH the
cache and the results to *_mock siblings, so a wiring smoke test can never
contaminate the real artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# import-for-side-effect: registers every baseline module (see
# controller/baselines/__init__.py) so get_baseline() can find them by name.
import controller.baselines  # noqa: F401,E402
from controller.baseline import Ctx, get_baseline, list_baselines  # noqa: E402
from llm.client import make_client  # noqa: E402
from llm.config import load_config as load_llm_config  # noqa: E402
from pipeline.spine import run_conversation  # noqa: E402
from simulator.batch_runner import precompute_frozen_query  # noqa: E402

SCENARIOS_FILE = Path(__file__).parent / "scenarios.json"
SIGNALS_CACHE = Path(__file__).parent / "e3_scenario_signals.json"
GRID_FILE = _REPO_ROOT / "configs" / "e3_price_grid.json"
LOG_DIR = _REPO_ROOT / "logs"

CACHE_DIR = Path(__file__).parent / "e3_sweep_cache"
RESULTS_DIR = Path(__file__).parent / "e3_sweep_results"
MOCK_CACHE_DIR = Path(__file__).parent / "e3_sweep_cache_mock"
MOCK_RESULTS_DIR = Path(__file__).parent / "e3_sweep_results_mock"

# E1's five fixed seeds (llm/experiment1/config.json), the same set
# simulator/e3_signals.py draws its signals under and the same set
# analysis/e3_frontier.py's EXPECTED_SEEDS requires. Fixed, not sampled:
# generation is temperature 0.2, so an unseeded sweep would not reproduce.
SEEDS = (11, 23, 37, 53, 71)

# controller/timing.py's VALID_POLICIES, in the order Phase A measures them.
TIMINGS = ("sentence_end", "immediate", "fixed_early_proxy")

# The two retrieve-or-not paths a baseline can choose between.
PATHS = ("skip", "retrieve")

# Sweep the full price grid for these (E3_BASELINE_INTERFACE.md: they read
# ctx.prices and optimize the scalarized objective against it) ...
PRICED_BASELINES = ("ours", "b9")
# ... and emit a single unpriced operating point for these. B5 is
# registered too but is not in E3's headline set; --baselines adds it.
UNPRICED_BASELINES = ("b0", "b4", "b6", "b8")

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for a hotel-concierge question-answering "
    "system. You compare a candidate answer against a reference answer "
    "and judge FACTUAL CORRECTNESS only."
)

# The judge REASONS before it answers, and the omission rule binds the
# conflict LIST rather than only the verdict.
#
# Both of those are measured choices, not style. Scored against the 19
# hand labels in simulator/judge_validation/labels.json (run
# `python -m simulator.judge_validation.score_judges`):
#
#     terse, answer unstripped   14/19   74%   <- what the first sweep used
#     terse, answer stripped     15/19   79%
#     reasoned, stripped         18/19   95%   <- this
#
# The terse form got 8 tokens and no room to think, which left it deciding
# borderline cases on surface form: an answer identical to the reference
# except for the words "The Grand Horizon Hotel & Suites offers" at the
# front scored NO five times out of five, and YES five out of five with
# that phrase removed. Reasoning first removes that sensitivity.
#
# The omission rule has to constrain the list. Told only that omission is
# correct, the model still WROTE "CONFLICT: the reference says menu items
# may contain allergens, but the candidate does not mention this" and then
# voted NO on its own list. So the instruction is placed on what may be
# listed as a conflict at all.
#
# Every error made by every variant was in one direction — marking a
# correct answer incorrect, never the reverse. That direction matters: it
# under-credits retrieval, and so penalises whichever baseline retrieves
# most. The remaining miss at 18/19 is s018/retrieve, which is labelled
# BORDERLINE in labels.json by the person who labelled it.
JUDGE_INSTRUCTIONS = (
    "Decide whether the candidate answer is factually correct with respect "
    "to the reference answer.\n"
    "Only FACTUAL CONFLICT counts. Specifically:\n"
    "- Wording, length, style, politeness, and naming the hotel do NOT "
    "matter.\n"
    "- Restating the reference in different words is CORRECT.\n"
    "- Omitting some of the reference's details is CORRECT.\n"
    "- Adding detail the reference does not mention is CORRECT, unless it "
    "states a specific figure, time, name, price or location that the "
    "reference gives differently.\n"
    "- It is INCORRECT only if it contradicts a specific fact in the "
    "reference, declines to answer, or answers a different question.\n\n"
    "List every DIRECT CONFLICT, one per line, as "
    "'CONFLICT: reference says X, candidate says Y'.\n"
    "A missing detail is NOT a conflict — do not list it. A conflict "
    "requires the candidate to STATE something incompatible with the "
    "reference. If there are none, write 'CONFLICT: none'.\n"
    "Then on the final line write exactly 'VERDICT: YES' or 'VERDICT: NO'."
)

# Enough room to list conflicts and still reach the verdict line. The terse
# judge ran at 8, which is what made reasoning impossible.
JUDGE_MAX_TOKENS = 300


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload) -> None:
    """
    Write JSON via a temp file + rename, so a cache entry is either fully
    there or not there at all. Resumability depends on this: a half-written
    cell file that a later run happily read back would silently poison the
    results with a truncated record, and the run that produced it would be
    long gone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        # Length-prefixed separator: without it, ("ab", "c") and ("a",
        # "bc") hash identically, and this key is what decides whether two
        # answers get the same verdict.
        digest.update(b"\x00")
    return digest.hexdigest()


def load_scenarios(limit: int = None) -> list:
    """The same scenarios.json every other runner reads — one shared
    scenario set across E1/E2/E3."""
    scenarios = _read_json(SCENARIOS_FILE)
    return scenarios[:limit] if limit is not None else scenarios


def load_grid() -> tuple:
    """
    The DERIVED price grid (configs/e3_price_grid.json, written by
    `python -m simulator.e3_grid`). Returns (lambda_l values, lambda_c).

    Read, never hand-written: E3_BASELINE_INTERFACE.md's "the lambda_l
    grid is derived, not chosen" — a hand-picked grid can place every
    divergence band off the observed VoI distribution, where Ours and B9
    cannot disagree, and return a null that is an artifact of the grid.
    """
    grid = _read_json(GRID_FILE)["grid"]
    return list(grid["lambda_l_grid"]), float(grid["lambda_c"])


# ---------------------------------------------------------------------------
# Backend guard
# ---------------------------------------------------------------------------
def resolve_backend(allow_mock: bool) -> dict:
    """
    Resolve which backend/model this run will use, decide where its
    artifacts belong, and refuse the one combination that silently
    corrupts data: mock draws written into the real cache.

    Same guard e3_grid.py's load_or_build_signals() applies to the signals
    cache, and for the same reason — a cache half real and half mock is
    undetectable by every consumer downstream. Here it is enforced by
    SEPARATION rather than refusal: --allow-mock is legitimate (it is how
    you smoke-test the wiring without a GPU), it just sends everything to
    *_mock paths that no real analysis reads.

    Reads llm/config.py's RESOLVED model_key, not config.yaml's
    active_model literal: the real runs are driven with
    VOICE_RAG_MODEL=mistral7b_v03 against a config whose active_model is
    still llama31_8b, so the literal would stamp these artifacts with a
    model that never answered.
    """
    cfg = load_llm_config()
    backend = cfg.get("backend")
    is_mock = backend == "mock"

    if is_mock and not allow_mock:
        raise SystemExit(
            "E3's sweep measures real model behaviour; the configured "
            "backend is 'mock'.\n"
            "Run it against the real backend:\n"
            "  VOICE_RAG_LLM_BACKEND=vllm VOICE_RAG_MODEL=mistral7b_v03 \\\n"
            "  VOICE_RAG_LLM_BASE_URL=<endpoint>/v1 \\\n"
            "  python -m simulator.e3_full_sweep\n"
            "or pass --allow-mock to smoke-test the wiring into "
            f"{MOCK_CACHE_DIR.name}/ and {MOCK_RESULTS_DIR.name}/ instead."
        )

    return {
        "backend": backend,
        "model_key": cfg.get("model_key") or cfg.get("active_model"),
        "model_hf_id": cfg.get("model"),
        "is_mock": is_mock,
        "cache_dir": MOCK_CACHE_DIR if is_mock else CACHE_DIR,
        "results_dir": MOCK_RESULTS_DIR if is_mock else RESULTS_DIR,
        "llm_cfg": cfg,
    }


def check_cache_provenance(env: dict) -> None:
    """
    A cache directory belongs to one (backend, model). Refuse to top up
    one built under a different model — the cells would disagree about
    which checkpoint answered, and nothing downstream could tell.
    """
    meta_path = env["cache_dir"] / "meta.json"
    meta = {
        "backend": env["backend"],
        "model_key": env["model_key"],
        "model_hf_id": env["model_hf_id"],
        "seeds": list(SEEDS),
        "timings": list(TIMINGS),
        "scenarios_file": str(SCENARIOS_FILE.relative_to(_REPO_ROOT)),
    }
    if meta_path.exists():
        existing = _read_json(meta_path)
        drifted = [
            key
            for key in ("backend", "model_key", "model_hf_id")
            if existing.get(key) != meta[key]
        ]
        if drifted:
            raise SystemExit(
                f"Cache at {env['cache_dir']} was built under "
                f"{ {k: existing.get(k) for k in drifted} } but this run is "
                f"{ {k: meta[k] for k in drifted} }. Measured cells from two "
                "models must not be mixed — point at a different cache dir, "
                "or delete this one to rebuild it."
            )
        return
    _atomic_write_json(meta_path, meta)


# ---------------------------------------------------------------------------
# Phase A — cache paths
# ---------------------------------------------------------------------------
def cell_path(cache_dir: Path, scenario_id, seed: int, path: str, timing: str) -> Path:
    return cache_dir / "cells" / f"s{scenario_id:03d}_seed{seed}_{path}_{timing}.json"


def answer_path(cache_dir: Path, scenario_id, seed: int, path: str) -> Path:
    """The canonical answer + verdict for a (scenario, seed, path).

    Separate from the cell files on purpose. The answer is a function of
    three inputs; the cell is a function of four. Storing the verdict per
    cell would mean three copies of one judgement per pair, which is three
    chances for them to disagree."""
    return cache_dir / "answers" / f"s{scenario_id:03d}_seed{seed}_{path}.json"


def judge_path(cache_dir: Path, judge_slug: str, key: str) -> Path:
    """Verdicts live under the judge that produced them.

    The directory is belt to the cache key's braces (judge_cache_key()
    already mixes the judge identity into the hash). Two independent
    guards, because the failure this closes is silent: a verdict cached by
    one judge and served to another looks exactly like a cache hit. It is
    also the version you can see — `ls judge/` names every judge that has
    ever written into this cache."""
    return cache_dir / "judge" / judge_slug / f"{key}.json"


def judge_cache_key(judge, scenario: dict, answer: str) -> str:
    """
    The verdict cache key: (judge identity, cache version, scenario id,
    reference answer, candidate answer).

    The judge belongs in the key. Verdicts are a function of WHO graded as
    much as of what was graded, and this cache has already held verdicts
    from a weaker grader — a mock run's steered YES, and Mistral-7B's,
    which scored "the spa opens at 9AM" correct against a 10AM reference.
    Keying on the text alone would let those stale YESes shadow the Azure
    judge's fresh verdict on the identical answer string, so swapping in a
    stronger judge would change nothing at all and the swap would look
    like it had worked. JUDGE_CACHE_VERSION invalidates on top of that,
    for changes (prompt, parsing, request shape) that a deployment name
    does not capture.

    Everything in the key is public: a deployment name and an API version.
    No endpoint, no credential.
    """
    return _sha256(
        JUDGE_CACHE_VERSION,
        judge.identity,
        scenario["id"],
        scenario["expected_answer"],
        answer,
    )


def scenario_path(cache_dir: Path, scenario_id) -> Path:
    return cache_dir / "scenarios" / f"s{scenario_id:03d}.json"


# ---------------------------------------------------------------------------
# Phase A — the frozen query rewrite, once per scenario
# ---------------------------------------------------------------------------
def frozen_query_for(scenario: dict, cache_dir: Path) -> str:
    """
    E2's frozen-rewrite mechanism (simulator/batch_runner.py's
    precompute_frozen_query): run rewrite_query() ONCE per scenario,
    outside any timed window, and hand the result to every run.

    Cached because it is a model call whose ~1.5s would otherwise be paid
    1500 times, and because a rewrite that varied between two cells of the
    same scenario would mean they retrieved different passages — which
    would break the one invariant this whole design rests on (retrieved
    content is a function of the scenario, not of timing or price).
    """
    path = scenario_path(cache_dir, scenario["id"])
    if path.exists():
        return _read_json(path)["frozen_query"]
    query = precompute_frozen_query(scenario)
    _atomic_write_json(
        path,
        {
            "scenario_id": scenario["id"],
            "question": scenario["question"],
            "frozen_query": query,
        },
    )
    return query


# ---------------------------------------------------------------------------
# Phase A — reading one run's outcome back out of its log
# ---------------------------------------------------------------------------
def read_outcome_from_log(log_path: Path) -> dict:
    """
    Pull this run's measured outcome out of the event log it just wrote:

      ttfa_ms    tts_first_audio.t_ms - user_end.t_ms, the same definition
                 harness/metrics.py uses.
      retrievals count of retrieval_launched events.
      answer     llm_final.payload["text"].

    Safe to read positionally (first/only match) because run_cell()
    deleted this log before the run — see this module's stale-log
    paragraph. Anything missing raises: a cell with no tts_first_audio has
    no TTFA, and recording a guess for it would put a fabricated number on
    the latency axis.
    """
    user_end_ms = None
    first_audio_ms = None
    retrievals = 0
    answer = None
    n_turns_seen = set()

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            kind = event.get("event_type")
            n_turns_seen.add(event.get("turn_id"))
            if kind == "user_end" and user_end_ms is None:
                user_end_ms = event["t_ms"]
            elif kind == "tts_first_audio" and first_audio_ms is None:
                first_audio_ms = event["t_ms"]
            elif kind == "retrieval_launched":
                retrievals += 1
            elif kind == "llm_final":
                answer = event["payload"].get("text")

    missing = [
        name
        for name, value in (
            ("user_end", user_end_ms),
            ("tts_first_audio", first_audio_ms),
            ("llm_final", answer),
        )
        if value is None
    ]
    if missing:
        raise RuntimeError(
            f"{log_path} is missing {', '.join(missing)} — the run did not "
            "reach the end of the pipeline, so this cell has no measurable "
            "outcome."
        )

    return {
        "ttfa_ms": float(first_audio_ms - user_end_ms),
        "retrievals": retrievals,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Phase A — the JUDGE CLIENT (Azure OpenAI), separate from the system client
# ---------------------------------------------------------------------------
# Two clients, two roles, deliberately not shared:
#
#   system client  — llm/client.py's make_client(env["llm_cfg"]), the
#                    self-hosted Mistral that GENERATES the answers. It is
#                    the system under measurement. It must not change.
#   judge  client  — AzureOpenaiJudge below, which GRADES those answers. It
#                    is the measuring instrument and is deliberately a
#                    stronger, different model.
#
# They are constructed separately, configured separately (llm/config.json
# vs. the AZURE_* environment), and never substituted for one another.
# Mistral-7B graded "the spa opens at 9AM" as correct against a 10AM
# reference, and a wrong wifi network as correct: a judge that cannot see
# an error cannot see the accuracy that skipping retrieval costs, and the
# frontier then reports "skip is free" and B0 wins by construction. The
# judge being weak is not a small measurement error — it is the one that
# manufactures the headline result.


class JudgeConfigError(RuntimeError):
    """Judge configuration is missing or unusable. Never fall back to a
    weaker judge on this — a silently downgraded instrument produces
    numbers that look fine and are not."""


class JudgeCallError(RuntimeError):
    """The judge was configured but the call did not produce a verdict."""


# Read from the environment, never from a config file and never a literal
# in this repo. The key in particular must not reach disk, logs, cached
# verdicts or git; see _judge_record() for what a verdict is allowed to
# carry.
AZURE_ENV_VARS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)

# Bumped whenever the judge changes in a way that invalidates verdicts.
# Part of the cache key, so old verdicts cannot be read back. v1 was the
# Mistral/mock era.
JUDGE_CACHE_VERSION = "v4-azure-reasoned"


def _slugify(text: str) -> str:
    """Filesystem-safe identity fragment. Purely cosmetic — the cache key
    is the hash; this only decides which directory a verdict lands in."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in text)[:64]


class AzureOpenaiJudge:
    """
    The Azure OpenAI judge, at temperature 0.

    Configuration is read from the environment ONLY (AZURE_ENV_VARS). A
    missing variable is a hard error naming the variable; there is no
    fallback to the system's Mistral client and no fallback to a mock,
    because both of those failure modes are invisible downstream — the
    sweep would complete, write a full results file, and be wrong.

    Transport is the openai SDK's AzureOpenAI client when it is installed,
    and a direct httpx POST to the deployment's chat/completions endpoint
    when it is not. Same request either way.
    """

    def __init__(self, api_key, endpoint, deployment, api_version):
        # Held in memory for the process lifetime and never written
        # anywhere. Not stored on any record, not printed, not logged.
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        # Sticky request-shape flags, resolved on first use: newer Azure
        # deployments (o-series / reasoning models) reject `max_tokens` in
        # favour of `max_completion_tokens`, and some reject a temperature
        # override entirely. Discovering that once per process beats
        # failing the whole sweep on the first call.
        self._token_param = "max_tokens"
        self.temperature_supported = True

        try:
            from openai import AzureOpenAI  # noqa: PLC0415 -- optional dep, probed at construction
        except ImportError:
            self._sdk = None
            self.transport = "httpx"
        else:
            self._sdk = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=self._endpoint,
                api_version=api_version,
            )
            self.transport = "openai-sdk"

    @classmethod
    def from_env(cls) -> "AzureOpenaiJudge":
        missing = [name for name in AZURE_ENV_VARS if not (os.environ.get(name) or "").strip()]
        if missing:
            raise JudgeConfigError(
                "The E3 judge is Azure OpenAI and is not configured: "
                f"{', '.join(missing)} " + ("is" if len(missing) == 1 else "are")
                + " not set in the environment.\n"
                "Set every one of "
                f"{', '.join(AZURE_ENV_VARS)} and re-run, e.g.\n\n"
                "  export AZURE_OPENAI_API_KEY=...        # never commit this\n"
                "  export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com\n"
                "  export AZURE_OPENAI_DEPLOYMENT=<deployment-name>\n"
                "  export AZURE_OPENAI_API_VERSION=<api-version>\n\n"
                "There is deliberately no fallback judge: grading these "
                "answers with a weak model (or a mock) scores wrong answers "
                "correct, which makes skipping retrieval look free and hands "
                "the frontier to B0 by construction."
            )
        return cls(
            api_key=os.environ["AZURE_OPENAI_API_KEY"].strip(),
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"].strip(),
            api_version=os.environ["AZURE_OPENAI_API_VERSION"].strip(),
        )

    @property
    def identity(self) -> str:
        """What this judge IS, for the cache key and the verdict record.

        Deployment name and API version only. The endpoint is left out
        (it names the Azure resource) and the key obviously so. Two
        different deployments therefore get two different cache
        namespaces, which is the whole point."""
        return f"azure-openai/{self.deployment}@{self.api_version}"

    @property
    def slug(self) -> str:
        return _slugify(f"{JUDGE_CACHE_VERSION}-{self.deployment}-{self.api_version}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """One graded call at temperature 0. Returns the raw text.

        Bounded retries on transient failures (rate limit, 5xx, dropped
        connection) because the sweep makes hundreds of judge calls over
        hours and a single 429 must not end it. Anything else is re-raised:
        a judge that is failing for a real reason should stop the run, not
        quietly leave answers ungraded."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        transient = 0
        for _ in range(6):  # >= 3 transient retries, plus room for shape adaptation
            try:
                return self._call(messages)
            except Exception as error:  # noqa: BLE001 -- classified below, then retried or re-raised
                if self._adapt_request_shape(error):
                    continue
                transient += 1
                if not _is_retryable(error) or transient > 3:
                    raise
                time.sleep(2 ** (transient - 1))
        raise JudgeCallError("The Azure judge call did not settle after retries.")

    def _call(self, messages) -> str:
        kwargs = {self._token_param: JUDGE_MAX_TOKENS}
        if self.temperature_supported:
            kwargs["temperature"] = 0
        if self._sdk is not None:
            response = self._sdk.chat.completions.create(
                model=self.deployment, messages=messages, **kwargs
            )
            return (response.choices[0].message.content or "").strip()
        return self._call_httpx(messages, kwargs)

    def _call_httpx(self, messages, kwargs) -> str:
        import httpx  # noqa: PLC0415 -- only needed when the openai SDK is absent

        url = (
            f"{self._endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        response = httpx.post(
            url,
            # The key travels in the header and nowhere else. Do not add it
            # to the URL: URLs end up in exception strings and proxy logs.
            headers={"api-key": self._api_key, "Content-Type": "application/json"},
            json={"messages": messages, **kwargs},
            timeout=60.0,
        )
        if response.status_code >= 400:
            raise _HttpJudgeError(response.status_code, response.text)
        payload = response.json()
        return (payload["choices"][0]["message"].get("content") or "").strip()

    def _adapt_request_shape(self, error) -> bool:
        """Some deployments reject `max_tokens` (want `max_completion_tokens`)
        or reject any temperature override. Adapt once and retry, rather
        than failing a multi-hour sweep on the request shape. Returns True
        if something was changed and the call is worth retrying."""
        text = str(error).lower()
        if "max_tokens" in text and self._token_param == "max_tokens":
            self._token_param = "max_completion_tokens"
            print("[judge] deployment rejects max_tokens; retrying with max_completion_tokens")
            return True
        if "temperature" in text and self.temperature_supported:
            self.temperature_supported = False
            print(
                "[judge] deployment rejects a temperature override; retrying at its "
                "default. NOTE: verdicts from this deployment are not "
                "temperature-0 and are recorded as such."
            )
            return True
        return False


class _HttpJudgeError(RuntimeError):
    def __init__(self, status, body):
        self.status = status
        # Azure error bodies echo the request, never the api-key header.
        super().__init__(f"HTTP {status}: {body[:500]}")


def _is_retryable(error) -> bool:
    status = getattr(error, "status", None) or getattr(error, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    name = type(error).__name__
    return name in (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
    )


class MockJudge:
    """
    Steered verdicts for --allow-mock wiring smoke tests ONLY.

    This is not a fallback: it is reachable only when the operator passes
    --allow-mock, which already redirects the entire cache and every
    result file into *_mock siblings that no analysis reads. Its verdicts
    additionally land under their own cache namespace (see slug), so a
    mock YES can never be served to a real Azure-keyed lookup.
    """

    identity = "mock-steered/not-a-judge"
    slug = "mock-steered-not-data"
    transport = "none"
    temperature_supported = True

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "YES"


def make_judge(is_mock: bool):
    """Build the judge client. The ONLY place a mock judge can appear is
    an explicit --allow-mock run; every real run gets Azure or a hard
    error."""
    if is_mock:
        print(
            "[judge] MOCK BACKEND — using steered verdicts, quarantined in "
            f"judge/{MockJudge.slug}/. Not data."
        )
        return MockJudge()
    judge = AzureOpenaiJudge.from_env()
    # Deployment + api-version only. Never the endpoint, never the key.
    print(
        f"[judge] Azure OpenAI judge: deployment={judge.deployment} "
        f"api-version={judge.api_version} transport={judge.transport} "
        "(system client stays Mistral)"
    )
    return judge


# ---------------------------------------------------------------------------
# Phase A — the judge
# ---------------------------------------------------------------------------
def judge_answer(
    judge,
    scenario: dict,
    answer: str,
    cache_dir: Path,
) -> dict:
    """
    Score one answer against the scenario's expected_answer with the Azure
    judge at temperature 0, and cache the verdict.

    Temperature 0, not 0.2: the judge is a measuring instrument, not a
    sample from the model's answer distribution. The ANSWERS are sampled
    at 0.2 with fixed seeds (E1's protocol); grading them is extraction,
    and re-grading identical text should never return two answers.

    The judge is `judge`, NOT the system's client: the Mistral that
    produced `answer` never grades it. See make_judge().

    A verdict that cannot be parsed as YES/NO is recorded as
    parsed_ok=False and scored incorrect, and Phase A's summary counts
    them. Scored-incorrect is the conservative reading, but it is a real
    bias if it ever happens at scale, so it is surfaced rather than
    absorbed: a run with a nonzero parse-failure count needs looking at
    before its accuracy numbers are believed.
    """
    # Strip before BOTH the key and the prompt, never one or the other:
    # the cache key has to describe the text the judge was actually shown.
    #
    # Not cosmetic. The generator's answers begin with a leading space, and
    # on a borderline answer that single character decides the verdict. On
    # s003 ("Tell me about deluxe room.") the retrieved answer reproduces
    # the reference and omits one list item, "air conditioning" — which
    # JUDGE_INSTRUCTIONS explicitly calls correct ("omits detail but
    # contradicts nothing"). Graded with the leading space it returned NO
    # five times out of five; graded stripped, YES five out of five, same
    # deployment at temperature 0. That flipped all 15 of s003's retrieve
    # cells to incorrect, and 5 of the 12 retrieve-path failures in the
    # first 360 cells of the first full sweep were this artifact.
    #
    # The judge is deterministic (10/10 identical on repeat calls); it is
    # the whitespace that moved it, so the fix is to stop varying it.
    # Leading/trailing whitespace is not content and must not reach a
    # grader. NOTE this does not make borderline grading robust in
    # general — it removes one known perturbation. See the omission rule
    # in JUDGE_INSTRUCTIONS.
    answer = (answer or "").strip()

    key = judge_cache_key(judge, scenario, answer)
    path = judge_path(cache_dir, judge.slug, key)
    if path.exists():
        return _read_json(path)

    user_prompt = (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"Question:\n{scenario['question']}\n\n"
        f"Reference answer:\n{scenario['expected_answer']}\n\n"
        f"Candidate answer:\n{answer}\n"
    )
    raw = judge.complete(JUDGE_SYSTEM_PROMPT, user_prompt)
    verdict = _judge_record(judge, raw, key, scenario["id"])
    _atomic_write_json(path, verdict)
    return verdict


def parse_verdict(raw: str) -> tuple:
    """(answer_correct, parsed_ok) from the judge's raw reply.

    The judge now reasons before it answers, so the verdict is the LAST
    line containing "VERDICT:" rather than the first token. Last, not
    first: the conflict list above it quotes the reference and the
    candidate, and either can contain the word "verdict" or a stray YES.

    The bare-token form is still accepted as a fallback so a verdict cached
    under the pre-reasoning prompt stays readable.
    """
    text = (raw or "").strip()
    lines = [line for line in text.splitlines() if "VERDICT" in line.upper()]
    if lines:
        tail = lines[-1].upper().split("VERDICT:")[-1].strip().lstrip("*\"' \n\t.:-")
        if tail.startswith("YES"):
            return True, True
        if tail.startswith("NO"):
            return False, True
        return False, False

    token = text.upper().lstrip("*\"' \n\t.:-")[:3]
    if token.startswith("YES"):
        return True, True
    if token.startswith("NO"):
        return False, True
    return False, False


def _judge_record(judge, raw: str, key: str, scenario_id) -> dict:
    """The cached verdict.

    Carries the judge's IDENTITY (deployment + api version) and nothing
    else about its configuration: no endpoint, and above all no API key.
    These files are written to the cache directory and are read back by
    later runs, so anything recorded here is on disk permanently.
    """
    raw = (raw or "").strip()
    correct, parsed_ok = parse_verdict(raw)
    record = {
        "answer_correct": correct,
        "parsed_ok": parsed_ok,
        "judge_raw": raw,
        "judge_key": key,
        "judge_identity": judge.identity,
        "judge_cache_version": JUDGE_CACHE_VERSION,
        "scenario_id": scenario_id,
        "temperature": 0.0 if judge.temperature_supported else None,
    }
    if isinstance(judge, MockJudge):
        record["judge_note"] = (
            "MOCK JUDGE — this verdict was steered, not judged. Not data."
        )
    elif not judge.temperature_supported:
        record["judge_note"] = (
            "This deployment rejected a temperature override; the verdict "
            "was produced at the deployment default, not at 0."
        )
    return record


# ---------------------------------------------------------------------------
# Phase A — judge self-test
# ---------------------------------------------------------------------------
# Three pairs with unarguable answers, two of them the exact failures that
# retired the Mistral judge. Run it the moment credentials are set: it
# proves the Azure judge is wired up AND that it is strong enough to see a
# wrong fact, in one call each, before committing to a multi-hour sweep.
JUDGE_SELFTEST_CASES = (
    {
        "name": "correct-answer",
        "question": "What time does the hotel gym open?",
        "expected_answer": "The gym opens at 6AM and closes at 10PM.",
        "answer": "The gym is open from 6AM until 10PM.",
        "want_correct": True,
    },
    {
        "name": "spa-9AM-vs-reference-10AM",
        "question": "What time does the spa open?",
        "expected_answer": "The spa opens at 10AM.",
        "answer": "The spa opens at 9AM.",
        "want_correct": False,
    },
    {
        "name": "contradiction",
        "question": "Is breakfast included with the room rate?",
        "expected_answer": "Breakfast is included with every room rate.",
        "answer": "No, breakfast is not included; it costs 25 dollars extra.",
        "want_correct": False,
    },
)


def run_judge_selftest() -> int:
    """
    Judge three fixed pairs and report. Bypasses the cache entirely (both
    read and write) — a self-test that could be answered from cache would
    verify nothing about the live credentials.

    Returns a process exit code: 0 if every case matched.
    """
    judge = make_judge(is_mock=False)
    print(f"\n  judge identity: {judge.identity}")
    print(f"  cache namespace: judge/{judge.slug}/\n")

    failures = []
    for case in JUDGE_SELFTEST_CASES:
        user_prompt = (
            f"{JUDGE_INSTRUCTIONS}\n\n"
            f"Question:\n{case['question']}\n\n"
            f"Reference answer:\n{case['expected_answer']}\n\n"
            f"Candidate answer:\n{case['answer']}\n"
        )
        raw = judge.complete(JUDGE_SYSTEM_PROMPT, user_prompt)
        correct, parsed_ok = parse_verdict(raw)
        ok = parsed_ok and correct == case["want_correct"]
        want = "YES" if case["want_correct"] else "NO"
        got = raw if parsed_ok else f"{raw!r} (UNPARSEABLE)"
        print(
            f"  {'PASS' if ok else 'FAIL'}  {case['name']:34s} "
            f"want {want:3s}  got {got}"
        )
        if not ok:
            failures.append(case["name"])

    if failures:
        print(
            f"\n{len(failures)} of {len(JUDGE_SELFTEST_CASES)} cases failed: "
            f"{', '.join(failures)}.\n"
            "This judge is not fit to score E3. A judge that scores a wrong "
            "fact as correct makes skipping retrieval look free, which hands "
            "the frontier to B0 by construction — the exact failure that "
            "retired Mistral-7B as the judge. Do not run the sweep until "
            "this passes."
        )
        return 1
    print(f"\nAll {len(JUDGE_SELFTEST_CASES)} cases passed. Judge is fit to score E3.")
    return 0


# ---------------------------------------------------------------------------
# Phase A — one measured cell
# ---------------------------------------------------------------------------
def run_cell(
    scenario: dict,
    seed: int,
    path_kind: str,
    timing: str,
    frozen_query: str,
    verbose: bool,
) -> dict:
    """
    Run ONE (scenario, seed, skip|retrieve, timing) cell through the real
    pipeline and return its measured outcome.

    Driven by E2's mask (force_retrieve) and E2's timing policy, NOT by a
    baseline: this measures what a (path, timing) choice costs, so that
    any baseline choosing it gets the same measured cost. See the module
    docstring's Phase A section.

    The cell's log file is deleted first so the log this run reads back
    contains this run only.
    """
    conversation_id = (
        f"e3full_s{scenario['id']:03d}_seed{seed}_{path_kind}_{timing}"
    )
    log_path = LOG_DIR / f"{conversation_id}.jsonl"
    log_path.unlink(missing_ok=True)

    previous_policy = os.environ.get("VOICE_RAG_TIMING_POLICY")
    os.environ["VOICE_RAG_TIMING_POLICY"] = timing
    captured = io.StringIO()
    try:
        # The spine narrates every stage to stdout (and its TTS worker
        # thread does too). Useful for one run, unreadable across 1500 —
        # captured by default and replayed only if the cell fails, so a
        # failure is still fully diagnosable.
        with contextlib.nullcontext() if verbose else contextlib.redirect_stdout(
            captured
        ):
            written_log = asyncio.run(
                run_conversation(
                    conversation_id,
                    question=scenario["question"],
                    force_retrieve=(path_kind == "retrieve"),
                    frozen_query=frozen_query,
                    seed=seed,
                )
            )
    except Exception:
        if not verbose:
            print(captured.getvalue())
        raise
    finally:
        if previous_policy is None:
            os.environ.pop("VOICE_RAG_TIMING_POLICY", None)
        else:
            os.environ["VOICE_RAG_TIMING_POLICY"] = previous_policy

    outcome = read_outcome_from_log(Path(written_log))

    if path_kind == "retrieve" and outcome["retrievals"] != 1:
        raise RuntimeError(
            f"{conversation_id}: forced a retrieve but the log records "
            f"{outcome['retrievals']} retrieval_launched events."
        )
    if path_kind == "skip" and outcome["retrievals"] != 0:
        raise RuntimeError(
            f"{conversation_id}: forced a skip but the log records "
            f"{outcome['retrievals']} retrieval_launched events."
        )

    outcome["conversation_id"] = conversation_id
    outcome["log_path"] = str(Path(written_log))
    return outcome


# ---------------------------------------------------------------------------
# Phase A — warm-up
# ---------------------------------------------------------------------------
def run_warmup(scenarios: list, timings: tuple, cache_dir: Path, verbose: bool) -> None:
    """
    A few throwaway runs before measurement, so first-call cost (client
    connection, piper/TTS model load, the backend's cold start) lands here
    instead of inside the first measured cell's TTFA. Same reasoning and
    same shape as simulator/batch_runner.py's run_warmup(); results are
    discarded and nothing is cached.
    """
    warmup = scenarios[: min(2, len(scenarios))]
    if not warmup:
        return
    print(f"WARM-UP ({len(warmup)} scenario(s) x {len(timings)} timing(s), discarded)")
    for scenario in warmup:
        query = frozen_query_for(scenario, cache_dir)
        for timing in timings:
            previous = os.environ.get("VOICE_RAG_TIMING_POLICY")
            os.environ["VOICE_RAG_TIMING_POLICY"] = timing
            captured = io.StringIO()
            try:
                with contextlib.nullcontext() if verbose else (
                    contextlib.redirect_stdout(captured)
                ):
                    asyncio.run(
                        run_conversation(
                            f"e3warm_s{scenario['id']:03d}_{timing}",
                            question=scenario["question"],
                            force_retrieve=True,
                            frozen_query=query,
                            seed=0,
                        )
                    )
            finally:
                if previous is None:
                    os.environ.pop("VOICE_RAG_TIMING_POLICY", None)
                else:
                    os.environ["VOICE_RAG_TIMING_POLICY"] = previous
    print("Warm-up complete (not recorded).\n")


# ---------------------------------------------------------------------------
# Phase A — the loop
# ---------------------------------------------------------------------------
def phase_a(
    scenarios: list,
    seeds: tuple,
    timings: tuple,
    env: dict,
    judge,
    verbose: bool,
    skip_warmup: bool,
) -> dict:
    """
    Measure every (scenario, seed, path, timing) cell not already cached,
    and judge each distinct answer once.

    Ordering is scenario -> seed -> path -> timing, so all six cells of a
    (scenario, seed) pair run back to back. Same reasoning as
    batch_runner.py's interleaving: any gradual drift over a multi-hour run
    (a warming cache, a throttling endpoint) then hits all three timing
    policies alike instead of landing entirely on whichever one was swept
    last, where it would masquerade as a timing effect.
    """
    cache_dir = env["cache_dir"]
    # The system client (generation) and the judge client (grading) are
    # two separate objects from two separate providers, and stay that way:
    # `client` answers the questions, `judge` grades the answers.
    client = make_client(env["llm_cfg"])

    if env["backend"] == "vllm" and not client.health_check():
        raise SystemExit(
            "The vLLM backend is not reachable. Start the server (see "
            "llm/SETUP.md) or export VOICE_RAG_LLM_BASE_URL=<endpoint>/v1, "
            "then re-run. Nothing has been written."
        )

    pending = sum(
        1
        for scenario in scenarios
        for seed in seeds
        for path_kind in PATHS
        for timing in timings
        if not cell_path(cache_dir, scenario["id"], seed, path_kind, timing).exists()
    )
    # Skipping warm-up when nothing is pending is not just a saving: a
    # resumed run that warmed up would spend six real conversations
    # warming a client it is about to close without measuring anything.
    if not skip_warmup and pending:
        run_warmup(scenarios, timings, cache_dir, verbose)

    stats = {
        "cells_total": len(scenarios) * len(seeds) * len(PATHS) * len(timings),
        "cells_measured": 0,
        "cells_cached": 0,
        "cells_failed": 0,
        "answers_generated": 0,
        "judge_calls": 0,
        "judge_parse_failures": 0,
        "answer_text_divergences": 0,
        "failures": [],
    }
    started = time.time()

    for scenario in scenarios:
        query = frozen_query_for(scenario, cache_dir)
        for seed in seeds:
            for path_kind in PATHS:
                canonical_file = answer_path(cache_dir, scenario["id"], seed, path_kind)
                canonical = _read_json(canonical_file) if canonical_file.exists() else None

                for timing in timings:
                    target = cell_path(cache_dir, scenario["id"], seed, path_kind, timing)
                    label = (
                        f"s{scenario['id']:03d} seed{seed} {path_kind:8s} {timing:18s}"
                    )
                    if target.exists():
                        stats["cells_cached"] += 1
                        print(f"  {label}  cached")
                        continue

                    try:
                        outcome = run_cell(
                            scenario, seed, path_kind, timing, query, verbose
                        )
                    except Exception as error:  # noqa: BLE001 -- one bad cell must not end a multi-hour resumable sweep; it is recorded and retried on the next run
                        stats["cells_failed"] += 1
                        stats["failures"].append({"cell": label, "error": str(error)})
                        print(f"  {label}  FAILED: {error}")
                        continue

                    answer = outcome["answer"]
                    answer_hash = _sha256(scenario["id"], answer)

                    if canonical is None:
                        # First cell for this (scenario, seed, path) — this
                        # is the answer the pair is scored on, for every
                        # timing.
                        verdict = judge_answer(judge, scenario, answer, cache_dir)
                        stats["judge_calls"] += 1
                        if not verdict["parsed_ok"]:
                            stats["judge_parse_failures"] += 1
                        canonical = {
                            "scenario_id": scenario["id"],
                            "seed": seed,
                            "path": path_kind,
                            "answer_text": answer,
                            "answer_sha256": answer_hash,
                            "answer_correct": verdict["answer_correct"],
                            "judge": verdict,
                            "measured_under_timing": timing,
                        }
                        _atomic_write_json(canonical_file, canonical)
                        stats["answers_generated"] += 1

                    matches_canonical = answer_hash == canonical["answer_sha256"]
                    if not matches_canonical:
                        # Seeded decoding on a batching server is
                        # best-effort, not a guarantee, so this is recorded
                        # rather than asserted. answer_correct still comes
                        # from the canonical answer, which keeps quality a
                        # function of (scenario, seed, path) exactly as the
                        # design requires — but a run with many of these has
                        # a reproducibility problem worth knowing about.
                        stats["answer_text_divergences"] += 1
                        print(
                            f"  {label}  NOTE: answer text differs from the "
                            "canonical answer for this (scenario, seed, path); "
                            "scoring the canonical one."
                        )

                    _atomic_write_json(
                        target,
                        {
                            "scenario_id": scenario["id"],
                            "seed": seed,
                            "path": path_kind,
                            "timing": timing,
                            "ttfa_ms": outcome["ttfa_ms"],
                            "retrievals": outcome["retrievals"],
                            "answer_sha256": answer_hash,
                            "matches_canonical_answer": matches_canonical,
                            "conversation_id": outcome["conversation_id"],
                            "log_path": outcome["log_path"],
                            "frozen_query": query,
                        },
                    )
                    stats["cells_measured"] += 1
                    done = stats["cells_measured"] + stats["cells_cached"]
                    print(
                        f"  {label}  ttfa={outcome['ttfa_ms']:.0f}ms "
                        f"retrievals={outcome['retrievals']} "
                        f"correct={canonical['answer_correct']}  "
                        f"[{done}/{stats['cells_total']}]"
                    )

    stats["elapsed_s"] = time.time() - started
    return stats


# ---------------------------------------------------------------------------
# Re-scoring an existing cache with a new judge
# ---------------------------------------------------------------------------
def rejudge(
    scenarios: list,
    seeds: tuple,
    env: dict,
    judge,
    dry_run: bool,
) -> dict:
    """
    Re-score already-cached answers with the CURRENT judge, generating
    nothing.

    What this touches and what it does not:

      answers/*.json   REWRITTEN — answer_correct and the judge record are
                       replaced with the current judge's verdict. The
                       answer TEXT is read, never regenerated.
      cells/*.json     UNTOUCHED — ttfa_ms, retrievals, conversation_id
                       and the log pointers are measurements of the
                       system, not of the judge, and re-scoring must not
                       disturb them. No cell is read or written here.
      judge/<slug>/    Written under the current judge's namespace only.

    No model client is opened: the Mistral that produced these answers is
    never contacted, so a re-score cannot perturb the measured latencies
    or spend hours regenerating text that is already on disk. This is the
    supported alternative to deleting answers/ by hand — that also works,
    but it silently discards what the previous judge said, and it is one
    slipped keystroke away from taking cells/ with it.

    The previous verdict is kept on the record as `previous_verdict`, so
    "the old judge said YES here and the new one says NO" stays
    answerable without re-running anything. It is single-level: a second
    re-score overwrites it with the first re-score's result.

    Re-scoring cannot be done with a mock judge against real data. It is
    structurally impossible rather than checked: a mock judge only exists
    under --allow-mock, which points env["cache_dir"] at the *_mock
    sibling, so a steered verdict has no path into the real cache.
    """
    cache_dir = env["cache_dir"]
    by_id = {scenario["id"]: scenario for scenario in scenarios}
    stats = {
        "answers_found": 0,
        "rejudged": 0,
        "unchanged": 0,
        "flipped_to_incorrect": 0,
        "flipped_to_correct": 0,
        "parse_failures": 0,
        "skipped": [],
        "flips": [],
    }

    targets = [
        (scenario_id, seed, path_kind, answer_path(cache_dir, scenario_id, seed, path_kind))
        for scenario_id in by_id
        for seed in seeds
        for path_kind in PATHS
    ]
    present = [item for item in targets if item[3].exists()]
    stats["answers_found"] = len(present)

    if not present:
        print(
            f"  no cached answers under {cache_dir / 'answers'} for the "
            "requested scenarios and seeds — nothing to re-score."
        )
        return stats

    if dry_run:
        print(f"  {len(present)} cached answer(s) would be re-scored by {judge.identity}:")
        for scenario_id, seed, path_kind, file_path in present:
            record = _read_json(file_path)
            previous = record.get("judge", {}).get("judge_identity", "unknown judge")
            print(
                f"    s{scenario_id:03d} seed{seed} {path_kind:8s} "
                f"currently answer_correct={record.get('answer_correct')} "
                f"(by {previous})"
            )
        print("  --dry-run: no judge calls made, nothing written.")
        return stats

    for scenario_id, seed, path_kind, file_path in present:
        label = f"s{scenario_id:03d} seed{seed} {path_kind:8s}"
        record = _read_json(file_path)
        answer = record.get("answer_text")
        if not answer:
            # An answer file with no text cannot be re-scored, and
            # inventing a verdict for it is exactly the failure this whole
            # change is about. Left alone and reported.
            stats["skipped"].append({"cell": label, "reason": "no answer_text on record"})
            print(f"  {label}  SKIPPED: no answer_text on record")
            continue

        was = record.get("answer_correct")
        verdict = judge_answer(judge, by_id[scenario_id], answer, cache_dir)
        now = verdict["answer_correct"]

        record["answer_correct"] = now
        record["previous_verdict"] = {
            "answer_correct": was,
            "judge_identity": record.get("judge", {}).get("judge_identity", "unknown"),
            "judge_raw": record.get("judge", {}).get("judge_raw"),
        }
        record["judge"] = verdict
        _atomic_write_json(file_path, record)

        stats["rejudged"] += 1
        if not verdict["parsed_ok"]:
            stats["parse_failures"] += 1
        if was == now:
            stats["unchanged"] += 1
            print(f"  {label}  {was} -> {now}")
        else:
            key = "flipped_to_correct" if now else "flipped_to_incorrect"
            stats[key] += 1
            stats["flips"].append({"cell": label, "was": was, "now": now})
            print(f"  {label}  {was} -> {now}   *** FLIPPED ***")

    return stats


# ---------------------------------------------------------------------------
# Phase B — the frozen signals, and the Ctx built from them
# ---------------------------------------------------------------------------
def load_signal_entries(scenario_ids: set) -> dict:
    """
    The committed per-scenario belief signals (simulator/
    e3_scenario_signals.json), keyed by scenario id.

    Read, never recomputed — see the module docstring's Phase B section.
    The backend it was built under is checked against 'mock' here for the
    same reason e3_grid.py checks it: a grid and a sweep derived from mock
    beliefs are not a result.
    """
    if not SIGNALS_CACHE.exists():
        raise SystemExit(
            f"No frozen signals at {SIGNALS_CACHE}. Build them with "
            "`python -m simulator.e3_grid --refresh-signals` under the real "
            "backend — Phase B reads them rather than recomputing, so the "
            "sweep and the price grid are about the same beliefs."
        )
    cache = _read_json(SIGNALS_CACHE)
    if cache.get("backend") == "mock":
        print(
            "WARNING: the frozen signals cache was built under the mock "
            "backend. Every baseline decision below is about mock beliefs."
        )
    entries = {entry["scenario_id"]: entry for entry in cache["signals"]}
    missing = sorted(scenario_ids - entries.keys())
    if missing:
        raise SystemExit(
            f"Frozen signals are missing scenarios {missing}. Rebuild them "
            "with `python -m simulator.e3_grid --refresh-signals`."
        )
    return entries


def ctx_for(entry: dict, prices: dict) -> Ctx:
    """
    The Ctx pipeline/spine.py's controller_decide_baseline() would hand a
    baseline for this scenario, assembled from the frozen signals — the
    same construction simulator/e3_divergence_check.py's
    scenario_contexts() uses, so a decision made here is the decision the
    pipeline would have logged.

    elapsed_ms=0 / user_end_known=True are fixed rather than measured
    because no registered baseline's decide() reads either field (checked
    across controller/baselines/); they exist in Ctx for baselines that
    might. If one ever does, this is the line that has to grow a real
    per-timing value, and this comment is the reason it would not be
    correct to leave as is.
    """
    return Ctx(
        transcript=entry["question"],
        signals={
            "model": {"available": True, "uncertainty": entry["model_uncertainty"]},
            "tracker": entry["tracker"],
        },
        elapsed_ms=0,
        user_end_known=True,
        turn_id=1,
        conversation_id=f"e3_full_sweep_{entry['scenario_id']:03d}",
        prices=prices,
    )


def phase_b(scenarios: list, seeds: tuple, baselines: list, env: dict) -> dict:
    """
    Cross every baseline with every price point it reads, every seed and
    every scenario; ask the baseline what it would do; look the outcome up
    in Phase A's cache; write the schema.

    No model call happens here. If a decision lands on a cell Phase A never
    measured, this raises rather than guessing — a missing cell means the
    frontier would be plotted partly on invented data.
    """
    cache_dir = env["cache_dir"]
    results_dir = env["results_dir"]
    scenario_ids = {scenario["id"] for scenario in scenarios}
    entries = load_signal_entries(scenario_ids)
    lambda_l_grid, lambda_c = load_grid()

    # Cache reads are repeated across 26 prices x 5 seeds; hold them in
    # memory so 14,000 records are not 14,000 file opens.
    canonical_cache = {}
    cell_cache = {}

    def canonical(scenario_id, seed, path_kind):
        key = (scenario_id, seed, path_kind)
        if key not in canonical_cache:
            file = answer_path(cache_dir, scenario_id, seed, path_kind)
            if not file.exists():
                raise SystemExit(
                    f"Phase A never measured an answer for scenario "
                    f"{scenario_id}, seed {seed}, path '{path_kind}' "
                    f"({file}). Run Phase A to completion first."
                )
            canonical_cache[key] = _read_json(file)
        return canonical_cache[key]

    def cell(scenario_id, seed, path_kind, timing):
        key = (scenario_id, seed, path_kind, timing)
        if key not in cell_cache:
            file = cell_path(cache_dir, scenario_id, seed, path_kind, timing)
            if not file.exists():
                raise SystemExit(
                    f"Phase A never measured scenario {scenario_id}, seed "
                    f"{seed}, path '{path_kind}', timing '{timing}' ({file}). "
                    f"Run Phase A to completion first."
                )
            cell_cache[key] = _read_json(file)
        return cell_cache[key]

    stats = {"records": 0, "files": 0, "decisions": 0, "by_baseline": {}}

    for name in baselines:
        controller = get_baseline(name)
        price_points = (
            [{"lambda_l": value, "lambda_c": lambda_c} for value in lambda_l_grid]
            if name in PRICED_BASELINES
            else [None]
        )

        # decide() is a pure function of (signals, prices) — the signals are
        # frozen per scenario and the seed never reaches Ctx — so the
        # decision is computed once per (scenario, price) and reused across
        # the five seeds rather than recomputed 5x. The seed varies the
        # measured outcome of the decision, not the decision.
        decisions = {}
        for prices in price_points:
            price_key = None if prices is None else (prices["lambda_l"], prices["lambda_c"])
            for scenario in scenarios:
                decision = controller.decide(ctx_for(entries[scenario["id"]], prices or {}))
                decisions[(price_key, scenario["id"])] = decision
                stats["decisions"] += 1

        retrieve_count = sum(1 for d in decisions.values() if d.retrieve)
        stats["by_baseline"][name] = {
            "price_points": len(price_points),
            "decisions": len(decisions),
            "retrieve_fraction": retrieve_count / max(1, len(decisions)),
            "timings_chosen": sorted({d.timing for d in decisions.values()}),
        }

        for seed in seeds:
            records = []
            for prices in price_points:
                price_key = (
                    None if prices is None else (prices["lambda_l"], prices["lambda_c"])
                )
                for scenario in scenarios:
                    decision = decisions[(price_key, scenario["id"])]
                    path_kind = "retrieve" if decision.retrieve else "skip"
                    measured = cell(scenario["id"], seed, path_kind, decision.timing)
                    scored = canonical(scenario["id"], seed, path_kind)
                    records.append(
                        {
                            "baseline": name,
                            "seed": seed,
                            "scenario_id": scenario["id"],
                            "prices": prices,
                            "outcome": {
                                "answer_correct": bool(scored["answer_correct"]),
                                "ttfa_ms": float(measured["ttfa_ms"]),
                                "retrievals": int(measured["retrievals"]),
                            },
                            # Ignored by analysis/e3_frontier.py (its
                            # validator requires a fixed set of keys and
                            # tolerates extras). Kept because "why is this
                            # point here" is otherwise unanswerable without
                            # re-running the baseline.
                            "decision": {
                                "retrieve": decision.retrieve,
                                "timing": decision.timing,
                                "memory_op": decision.memory_op,
                                "reason": decision.reason,
                            },
                            "source_cell": measured["conversation_id"],
                        }
                    )
            out = results_dir / f"e3_{name}_seed{seed}.json"
            _atomic_write_json(out, records)
            stats["records"] += len(records)
            stats["files"] += 1
            print(f"  wrote {out}  ({len(records)} records)")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3's full frontier sweep: Phase A measures "
        "every (scenario, seed, retrieve/skip, timing) cell against the "
        "real backend and caches it; Phase B crosses every baseline with "
        "the derived price grid and writes the result schema "
        "analysis/e3_frontier.py reads."
    )
    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Only use the first N scenarios (default: all 50). Applies to "
        "both phases — a Phase B run with a limit produces a partial "
        "frontier, so use it for testing, not for the headline.",
    )
    parser.add_argument(
        "--phase-a-only",
        action="store_true",
        help="Measure and cache only; write no result files.",
    )
    parser.add_argument(
        "--phase-b-only",
        action="store_true",
        help="Write result files from an already-complete Phase A cache. "
        "Opens no model client.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help=f"Seeds to run (default: E1's {list(SEEDS)}).",
    )
    parser.add_argument(
        "--timings",
        nargs="+",
        choices=list(TIMINGS),
        default=list(TIMINGS),
        help="Timing policies Phase A measures (default: all three). "
        "Restricting this leaves cells unmeasured, and Phase B will refuse "
        "to invent them if a baseline chooses one.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(UNPRICED_BASELINES) + list(PRICED_BASELINES),
        help="Baselines Phase B runs (default: "
        f"{list(UNPRICED_BASELINES) + list(PRICED_BASELINES)}).",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Permit the mock backend, writing to *_mock cache/results "
        "directories. For wiring smoke tests only — mock runs are not data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Let the pipeline's own per-stage output through (default: "
        "captured, and replayed only when a cell fails).",
    )
    parser.add_argument(
        "--judge-selftest",
        action="store_true",
        help="Judge three fixed pairs with the Azure judge and exit "
        "non-zero if any is graded wrong. Run this the moment the AZURE_* "
        "environment is set: it verifies the credentials and that the "
        "judge is strong enough to fail a wrong fact. Bypasses the cache; "
        "measures nothing.",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="Re-score every cached answer with the CURRENT judge and "
        "exit. Rewrites answers/*.json only — no generation, no model "
        "client, and cells/ (timing, retrievals, logs) is not touched. "
        "Scoped by --limit-scenarios and --seeds. Use after swapping the "
        "judge; combine with --dry-run to list what would be re-scored "
        "without spending a judge call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the work plan and what is already cached, then exit.",
    )
    args = parser.parse_args()

    if args.judge_selftest:
        # Before any other validation: this is the one mode that needs
        # nothing but credentials.
        try:
            raise SystemExit(run_judge_selftest())
        except JudgeConfigError as error:
            # A misconfigured environment is an operator error, not a bug;
            # print the instructions, not a traceback.
            raise SystemExit(str(error)) from None

    if args.phase_a_only and args.phase_b_only:
        parser.error("--phase-a-only and --phase-b-only are mutually exclusive.")
    if args.rejudge and (args.phase_a_only or args.phase_b_only):
        parser.error(
            "--rejudge is a mode of its own: it re-scores the cache and "
            "exits, measuring nothing and writing no result files. Run it "
            "alone, then re-run with --phase-b-only to rewrite the results "
            "from the new verdicts."
        )

    registered = list_baselines()
    unknown = [name for name in args.baselines if name not in registered]
    if unknown:
        parser.error(
            f"unknown baseline(s) {unknown}. Registered: {registered}"
        )

    run_a = not args.phase_b_only
    run_b = not args.phase_a_only

    env = resolve_backend(args.allow_mock)
    # Built up front, not lazily inside Phase A: a missing AZURE_* variable
    # must fail in the first second, not after warm-up has already spent
    # six conversations against the real backend.
    try:
        judge = make_judge(env["is_mock"]) if (run_a or args.rejudge) else None
    except JudgeConfigError as error:
        raise SystemExit(str(error)) from None
    scenarios = load_scenarios(args.limit_scenarios)
    seeds = tuple(args.seeds)
    timings = tuple(args.timings)
    lambda_l_grid, lambda_c = load_grid()

    if args.rejudge:
        check_cache_provenance(env)
        print("=" * 70)
        print("E3 — RE-SCORING CACHED ANSWERS WITH THE CURRENT JUDGE")
        print("=" * 70)
        print(f"judge      : {judge.identity}")
        print(f"cache      : {env['cache_dir']}")
        print(f"scope      : {len(scenarios)} scenario(s), seeds {list(seeds)}")
        print("generation : none — cells/ (timing, retrievals) is not touched")
        print()
        stats = rejudge(scenarios, seeds, env, judge, args.dry_run)
        if args.dry_run:
            return
        print()
        print("-" * 70)
        print(f"cached answers found  : {stats['answers_found']}")
        print(f"re-scored             : {stats['rejudged']}")
        print(f"verdict unchanged     : {stats['unchanged']}")
        print(f"flipped to INCORRECT  : {stats['flipped_to_incorrect']}")
        print(f"flipped to correct    : {stats['flipped_to_correct']}")
        print(f"judge parse failures  : {stats['parse_failures']}")
        if stats["skipped"]:
            print(f"skipped               : {len(stats['skipped'])}")
            for item in stats["skipped"]:
                print(f"  {item['cell']}: {item['reason']}")
        if stats["parse_failures"]:
            print(
                "\nSome verdicts did not parse as YES/NO and were scored "
                "incorrect. That is the conservative reading, not a safe one "
                "— check them before believing the accuracy numbers."
            )
        print(
            "\nanswers/*.json rewritten. Re-run with --phase-b-only to "
            "rebuild the result files from these verdicts."
        )
        return

    priced = [name for name in args.baselines if name in PRICED_BASELINES]
    unpriced = [name for name in args.baselines if name not in PRICED_BASELINES]
    expected_records = len(seeds) * len(scenarios) * (
        len(priced) * len(lambda_l_grid) + len(unpriced)
    )

    print("=" * 70)
    print("EXPERIMENT 3 — FULL FRONTIER SWEEP")
    print("=" * 70)
    print(f"backend    : {env['backend']} / {env['model_key']} ({env['model_hf_id']})")
    if env["is_mock"]:
        print("             *** MOCK — smoke test only, not data ***")
    print(f"scenarios  : {len(scenarios)}")
    print(f"seeds      : {list(seeds)}")
    print(f"timings    : {list(timings)}")
    print(f"grid       : {len(lambda_l_grid)} lambda_l points, lambda_c={lambda_c:g}")
    print(f"baselines  : priced={priced}  unpriced={unpriced}")
    if judge is not None:
        print(f"judge      : {judge.identity}  (separate client — the system "
              "model does not grade itself)")
    print(f"cache      : {env['cache_dir']}")
    print(f"results    : {env['results_dir']}")
    print()
    cells_total = len(scenarios) * len(seeds) * len(PATHS) * len(timings)
    cached_now = sum(
        1
        for scenario in scenarios
        for seed in seeds
        for path_kind in PATHS
        for timing in timings
        if cell_path(env["cache_dir"], scenario["id"], seed, path_kind, timing).exists()
    )
    print(f"Phase A    : {cells_total} cells, {cached_now} already cached, "
          f"{cells_total - cached_now} to measure  ({'ON' if run_a else 'skipped'})")
    print(f"Phase B    : {expected_records} records over "
          f"{len(args.baselines) * len(seeds)} files  "
          f"({'ON' if run_b else 'skipped'})")
    print()

    if args.dry_run:
        print("--dry-run: nothing measured, nothing written.")
        return

    if run_a:
        check_cache_provenance(env)
        print("-" * 70)
        print("PHASE A — measuring cells against the real pipeline")
        print("-" * 70)
        stats = phase_a(
            scenarios, seeds, timings, env, judge, args.verbose, skip_warmup=False
        )
        print()
        print("-" * 70)
        print("PHASE A COMPLETE")
        print("-" * 70)
        print(f"cells measured        : {stats['cells_measured']}")
        print(f"cells already cached  : {stats['cells_cached']}")
        print(f"cells failed          : {stats['cells_failed']}")
        print(f"answers generated     : {stats['answers_generated']}")
        print(f"judge calls           : {stats['judge_calls']}")
        print(f"judge parse failures  : {stats['judge_parse_failures']}")
        print(f"answer-text drift     : {stats['answer_text_divergences']}")
        print(f"elapsed               : {stats['elapsed_s'] / 60:.1f} min")
        if stats["cells_measured"]:
            per_cell = stats["elapsed_s"] / stats["cells_measured"]
            remaining = cells_total - stats["cells_measured"] - stats["cells_cached"]
            print(f"per measured cell     : {per_cell:.1f}s")
            if remaining > 0:
                print(
                    f"projected remaining   : {remaining} cells "
                    f"~ {remaining * per_cell / 3600:.1f}h"
                )
        if stats["cells_failed"]:
            print()
            print("FAILED CELLS (re-run to retry — nothing was cached for them):")
            for failure in stats["failures"][:20]:
                print(f"  {failure['cell']}: {failure['error']}")
            print()
            print("Phase B not run: it would be scored on an incomplete cache.")
            sys.exit(1)
        print()

    if run_b:
        print("-" * 70)
        print("PHASE B — baselines x prices x seeds x scenarios (no generation)")
        print("-" * 70)
        stats = phase_b(scenarios, seeds, args.baselines, env)
        print()
        print("-" * 70)
        print("PHASE B COMPLETE")
        print("-" * 70)
        print(f"records written : {stats['records']} (expected {expected_records})")
        print(f"files written   : {stats['files']}")
        print(f"decisions made  : {stats['decisions']}")
        for name, summary in stats["by_baseline"].items():
            print(
                f"  {name:5s} {summary['price_points']:3d} price point(s)  "
                f"retrieve={summary['retrieve_fraction']:.2%}  "
                f"timings={summary['timings_chosen']}"
            )
        print()
        print(f"Next: python -m analysis.e3_frontier {env['results_dir']}")


if __name__ == "__main__":
    main()
