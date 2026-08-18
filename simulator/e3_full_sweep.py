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
CORRECTNESS. It is not asked whether retrieval fired, and must not be: a
closed-book answer that happens to state the fact correctly scores true.
Retrieval's cost shows up in the compute axis (the retrievals count) and
in the latency axis, which is exactly the trade-off the frontier is about.
Scoring "did it retrieve" as quality would build the answer into the
question.

Verdicts are cached by sha256 over (scenario_id, expected_answer, answer
text), so an answer generated under five seeds that lands identically is
judged once. The key covers the reference, not just the answer: identical
answer STRINGS occur across scenarios (a refusal, a generic greeting) and
they are correct against different references, so keying on the answer
alone would let one scenario's verdict answer another scenario's question.

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

JUDGE_INSTRUCTIONS = (
    "Judge whether the candidate answer is factually correct with respect "
    "to the reference answer.\n"
    "- Correct means: it states the facts the reference states, or a "
    "subset of them, without contradicting the reference.\n"
    "- Wording, length, style and politeness do not matter.\n"
    "- A candidate that omits detail but contradicts nothing is correct.\n"
    "- A candidate that invents a fact the reference contradicts, or that "
    "declines to answer, or that answers a different question, is "
    "incorrect.\n"
    "Reply with exactly one word: YES if correct, NO if incorrect."
)


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


def judge_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "judge" / f"{key}.json"


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
# Phase A — the judge
# ---------------------------------------------------------------------------
def judge_answer(
    client,
    is_mock: bool,
    scenario: dict,
    answer: str,
    cache_dir: Path,
) -> dict:
    """
    Score one answer against the scenario's expected_answer with an LLM
    judge at temperature 0, and cache the verdict.

    Temperature 0, not 0.2: the judge is a measuring instrument, not a
    sample from the model's answer distribution. The ANSWERS are sampled
    at 0.2 with fixed seeds (E1's protocol); grading them is extraction,
    and re-grading identical text should never return two answers.

    A verdict that cannot be parsed as YES/NO is recorded as
    parsed_ok=False and scored incorrect, and Phase A's summary counts
    them. Scored-incorrect is the conservative reading, but it is a real
    bias if it ever happens at scale, so it is surfaced rather than
    absorbed: a run with a nonzero parse-failure count needs looking at
    before its accuracy numbers are believed.
    """
    key = _sha256(scenario["id"], scenario["expected_answer"], answer)
    path = judge_path(cache_dir, key)
    if path.exists():
        return _read_json(path)

    user_prompt = (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"Question:\n{scenario['question']}\n\n"
        f"Reference answer:\n{scenario['expected_answer']}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        "Verdict (YES or NO):"
    )
    overrides = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4,
        "seed": 0,
    }
    if is_mock:
        # MockClient returns a canned hotel answer, which would fail to
        # parse for every single cell and make a mock smoke test prove
        # nothing about the happy path. Steering it to a parseable verdict
        # exercises the real plumbing; the verdict itself is meaningless,
        # which is why it is labelled as such in the cached record and why
        # mock runs are quarantined in *_mock directories.
        overrides["mock_answer"] = "YES"

    generation = client.generate(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        **overrides,
    )
    raw = (generation.text or "").strip()
    token = raw.upper().lstrip("*\"' \n\t.:-")[:3]
    if token.startswith("YES"):
        correct, parsed_ok = True, True
    elif token.startswith("NO"):
        correct, parsed_ok = False, True
    else:
        correct, parsed_ok = False, False

    verdict = {
        "answer_correct": correct,
        "parsed_ok": parsed_ok,
        "judge_raw": raw,
        "judge_key": key,
        "scenario_id": scenario["id"],
        "temperature": 0.0,
    }
    if is_mock:
        verdict["judge_note"] = (
            "MOCK BACKEND — this verdict was steered, not judged. Not data."
        )
    _atomic_write_json(path, verdict)
    return verdict


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
                        verdict = judge_answer(
                            client, env["is_mock"], scenario, answer, cache_dir
                        )
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
        "--dry-run",
        action="store_true",
        help="Print the work plan and what is already cached, then exit.",
    )
    args = parser.parse_args()

    if args.phase_a_only and args.phase_b_only:
        parser.error("--phase-a-only and --phase-b-only are mutually exclusive.")

    registered = list_baselines()
    unknown = [name for name in args.baselines if name not in registered]
    if unknown:
        parser.error(
            f"unknown baseline(s) {unknown}. Registered: {registered}"
        )

    run_a = not args.phase_b_only
    run_b = not args.phase_a_only

    env = resolve_backend(args.allow_mock)
    scenarios = load_scenarios(args.limit_scenarios)
    seeds = tuple(args.seeds)
    timings = tuple(args.timings)
    lambda_l_grid, lambda_c = load_grid()

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
        stats = phase_a(scenarios, seeds, timings, env, args.verbose, skip_warmup=False)
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
