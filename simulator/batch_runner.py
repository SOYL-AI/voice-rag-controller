import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os

from pipeline.spine import run_conversation
from retrieval.rewrite import rewrite_query


SCENARIOS_FILE = Path(__file__).parent / "scenarios.json"
RESULTS_FILE = Path(__file__).parent / "batch_results.json"  # --single mode's output

# Experiment 2: sweep the three timing policies (controller/timing.py's
# VALID_POLICIES) across a handful of seeds, with retrieval decisions
# pinned to each scenario's gold retrieval_needed label (force_retrieve)
# so the only thing that varies between policies is WHEN retrieval fires,
# never WHETHER it fires.
TIMING_POLICIES = ["sentence_end", "immediate", "fixed_early_proxy"]
SEEDS = [0, 1, 2, 3, 4]
E2_RESULTS_DIR = Path(__file__).parent / "e2_results"


def load_scenarios():
    """Load simulator scenarios from scenarios.json."""
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def precompute_frozen_query(scenario):
    """
    Experiment 2's frozen-rewrite precompute step (this file is
    Shubhankit's — flagging the change here). Runs rewrite_query() once for
    a scenario, up front, outside the timed sweep below, and returns the
    resulting string so run_scenario() can hand it to run_conversation() as
    frozen_query.

    Why this needs to happen here and not just inline in
    run_retrieval_pipeline(): rewrite_query() takes ~1.5s on the real Modal
    model. Left in the timed path, that latency gets paid by every timing
    policy — including immediate/fixed_early_proxy, which exist specifically
    to launch retrieval early, during speech — eating into (or entirely
    consuming) the retrieval-overlap-with-speech window Experiment 2 is
    trying to measure. That's a rewrite-latency confound, not a placement
    effect: E2 measures WHEN retrieval launches relative to speech, not how
    long the rewrite itself takes. Precomputing here, once per scenario,
    isolates the two.

    Uses the scenario's own conversation history (scenarios.json's
    "history" field, [] for standalone questions) — the same history a
    live rewrite_query() call would have used — so the frozen query matches
    what the live path would have produced, not the queried question alone.
    """
    return rewrite_query(scenario.get("history", []), scenario["question"])


def run_scenario(scenario, conversation_id, force_retrieve=None, seed=None, frozen_query=None):
    """
    Run one simulator scenario through the pipeline and return its result
    record. force_retrieve/seed/frozen_query are forwarded to
    run_conversation() unchanged (all default to None there, i.e. "use the
    live decision" / "use the config's generation seed" / "rewrite live,
    in the timed path") — see pipeline/spine.py's run_conversation
    docstring. frozen_query is normally produced by
    precompute_frozen_query() above, ahead of the timed sweep.
    """
    scenario_id = scenario["id"]

    print(f"  Running scenario {scenario_id} ({conversation_id})")
    print(f"    Question: {scenario['question']}")

    try:
        log_path = asyncio.run(
            run_conversation(
                conversation_id,
                question=scenario["question"],
                force_retrieve=force_retrieve,
                seed=seed,
                frozen_query=frozen_query,
            )
        )

        return {
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "question": scenario["question"],
            "expected_answer": scenario["expected_answer"],
            "retrieval_needed": scenario["retrieval_needed"],
            "status": "success",
            "log_path": str(log_path),
        }

    except Exception as error:
        print(f"    ERROR: Scenario {scenario_id} failed: {error}")

        return {
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "question": scenario["question"],
            "expected_answer": scenario["expected_answer"],
            "retrieval_needed": scenario["retrieval_needed"],
            "status": "failed",
            "error": str(error),
        }


# ---------------------------------------------------------------------------
# --single: the batch runner's original behavior — one pass over every
# scenario, live retrieval decision, no timing/seed sweep. Kept as an
# opt-in flag for quick sanity checks against the pipeline.
# ---------------------------------------------------------------------------
def run_single_pass(scenarios):
    print("=" * 60)
    print("VOICE RAG BATCH RUNNER (single pass, no sweep)")
    print("=" * 60)
    print(f"Loaded {len(scenarios)} scenarios.")
    print()

    results = []
    for scenario in scenarios:
        conversation_id = f"conv_{scenario['id']:03d}"
        results.append(run_scenario(scenario, conversation_id))

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print()
    print("=" * 60)
    print("BATCH RUN COMPLETE")
    print("=" * 60)
    print(f"Total scenarios : {len(results)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")
    print(f"Results saved   : {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Warm-up: a few throwaway runs of every policy before the measured sweep
# starts, so model loading / connection setup / first-call cold-start cost
# lands here instead of contaminating the timing comparison below. Results
# are discarded on purpose -- nothing here is written to disk or added to
# any summary.
# ---------------------------------------------------------------------------
def run_warmup(scenarios, policies):
    warmup_scenarios = scenarios[: min(2, len(scenarios))]
    if not warmup_scenarios:
        return

    print("-" * 60)
    print(
        f"WARM-UP ({len(warmup_scenarios)} scenario(s) x "
        f"{len(policies)} polic{'y' if len(policies) == 1 else 'ies'}, "
        "results discarded, not recorded)"
    )
    print("-" * 60)

    # Precompute each warm-up scenario's rewrite once, up front (Experiment
    # 2's frozen-query mechanism — see precompute_frozen_query()), so
    # warm-up exercises the exact same frozen-rewrite code path the
    # measured sweep below does, instead of warming up a live-rewrite path
    # that never actually runs during measurement.
    frozen_queries = {
        scenario["id"]: precompute_frozen_query(scenario) for scenario in warmup_scenarios
    }

    for policy in policies:
        os.environ["VOICE_RAG_TIMING_POLICY"] = policy
        for scenario in warmup_scenarios:
            conversation_id = f"warmup_{policy}_{scenario['id']:03d}"
            run_scenario(
                scenario,
                conversation_id,
                force_retrieve=scenario["retrieval_needed"],
                seed=0,
                frozen_query=frozen_queries[scenario["id"]],
            )
            # Result intentionally dropped: warm-up runs are not part of
            # Experiment 2's measured data.

    print("Warm-up complete (not recorded).")
    print()


# ---------------------------------------------------------------------------
# Default mode: Experiment 2's full sweep.
#
# Run order is interleaved by (scenario, seed): for every scenario/seed
# pair we run all policies back-to-back before moving to the next scenario,
# rather than running policy A across every scenario, then all of B, then
# all of C. Any gradual drift over the course of the sweep (caching warming
# further, connection pooling settling in, etc.) then lands on all three
# policies roughly equally instead of systematically favoring whichever
# policy's block happened to run last. Results are still bucketed and
# written out per (policy, seed), identical to the old block-order output.
# ---------------------------------------------------------------------------
def run_sweep(scenarios, policies, seeds):
    print("=" * 60)
    print("VOICE RAG BATCH RUNNER — EXPERIMENT 2 SWEEP")
    print("=" * 60)
    print(f"Loaded {len(scenarios)} scenarios.")
    print(f"Policies: {policies}")
    print(f"Seeds   : {seeds}")
    print()

    E2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_warmup(scenarios, policies)

    # Experiment 2's frozen-query precompute: run each scenario's rewrite
    # exactly once, here, outside the timed sweep below, and reuse the
    # result across every (policy, seed) run of that scenario. See
    # precompute_frozen_query()'s docstring for why this needs to happen
    # up front rather than inline per run — in short, isolating rewrite
    # latency out of the retrieval-placement measurement.
    print("-" * 60)
    print(f"PRECOMPUTING REWRITES ({len(scenarios)} scenario(s), outside the timed sweep)")
    print("-" * 60)
    frozen_queries = {}
    for scenario in scenarios:
        frozen_queries[scenario["id"]] = precompute_frozen_query(scenario)
        print(f"  scenario {scenario['id']}: {frozen_queries[scenario['id']]}")
    print()

    print("-" * 60)
    print("MEASURED SWEEP (policy order interleaved per scenario/seed)")
    print("-" * 60)

    runs_by_key = {(policy, seed): [] for policy in policies for seed in seeds}

    for seed in seeds:
        for scenario in scenarios:
            print(f"  Scenario={scenario['id']}  Seed={seed}  "
                  f"policies={policies}")

            for policy in policies:
                # Read by controller/timing.py's resolve_policy() at the
                # top of every run_conversation() call. Set immediately
                # before each individual run now (rather than once per
                # policy block) since policies interleave; asyncio.run()
                # below is called sequentially (never concurrently), so
                # there's no race on this env var.
                os.environ["VOICE_RAG_TIMING_POLICY"] = policy

                conversation_id = (
                    f"conv_{policy}_s{seed}_{scenario['id']:03d}"
                )
                result = run_scenario(
                    scenario,
                    conversation_id,
                    force_retrieve=scenario["retrieval_needed"],
                    seed=seed,
                    frozen_query=frozen_queries[scenario["id"]],
                )
                result["policy"] = policy
                result["seed"] = seed
                runs_by_key[(policy, seed)].append(result)

    print()

    # Write out and summarize in the original policy-major, seed-minor
    # order so file names/contents and the summary table are unchanged
    # from the old block-order run -- only the execution order above
    # changed, not this output structure.
    summary = []

    for policy in policies:
        for seed in seeds:
            results = runs_by_key[(policy, seed)]

            out_path = E2_RESULTS_DIR / f"e2_{policy}_seed{seed}.json"
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump(
                    {"policy": policy, "seed": seed, "results": results},
                    file,
                    indent=2,
                )

            successful = sum(1 for r in results if r["status"] == "success")
            failed = len(results) - successful
            print(f"  -> {out_path}  ({successful} ok, {failed} failed)")

            summary.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "total": len(results),
                    "successful": successful,
                    "failed": failed,
                    "results_file": str(out_path),
                }
            )

    print()
    print("=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    total_runs = sum(row["total"] for row in summary)
    total_ok = sum(row["successful"] for row in summary)
    print(f"Combos (policy x seed) : {len(summary)}")
    print(f"Total scenario runs    : {total_runs}")
    print(f"Successful             : {total_ok}")
    print(f"Failed                 : {total_runs - total_ok}")
    print(f"Results dir            : {E2_RESULTS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Voice RAG batch runner")
    parser.add_argument(
        "--single",
        "--no-sweep",
        dest="single",
        action="store_true",
        help=(
            "Run the original single-pass mode (one run per scenario, "
            "live retrieval decision, no timing/seed sweep) instead of "
            "Experiment 2's full sweep. Writes batch_results.json."
        ),
    )
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help=(
            "Comma-separated timing policies to sweep "
            f"(default: all of {TIMING_POLICIES}). Ignored with --single."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            f"Comma-separated seeds to sweep (default: {SEEDS}). "
            "Ignored with --single."
        ),
    )
    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Only run the first N scenarios (for quick test runs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = load_scenarios()

    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    if args.single:
        run_single_pass(scenarios)
        return

    policies = (
        [p.strip() for p in args.policies.split(",")]
        if args.policies
        else TIMING_POLICIES
    )
    seeds = (
        [int(s.strip()) for s in args.seeds.split(",")]
        if args.seeds
        else SEEDS
    )

    run_sweep(scenarios, policies, seeds)


if __name__ == "__main__":
    main()
