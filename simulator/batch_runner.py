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


def run_scenario(scenario, conversation_id, force_retrieve=None, seed=None):
    """
    Run one simulator scenario through the pipeline and return its result
    record. force_retrieve/seed are forwarded to run_conversation()
    unchanged (both default to None there, i.e. "use the live decision" /
    "use the config's generation seed") — see pipeline/spine.py's
    run_conversation docstring.
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
# Default mode: Experiment 2's full sweep.
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

    summary = []

    for policy in policies:
        # Read by controller/timing.py's resolve_policy() at the top of
        # every run_conversation() call. Set once per policy, before that
        # policy's runs start; asyncio.run() below is called sequentially
        # (never concurrently), so there's no race on this env var.
        os.environ["VOICE_RAG_TIMING_POLICY"] = policy

        for seed in seeds:
            print("-" * 60)
            print(f"Policy={policy}  Seed={seed}")
            print("-" * 60)

            results = []
            for scenario in scenarios:
                conversation_id = (
                    f"conv_{policy}_s{seed}_{scenario['id']:03d}"
                )
                result = run_scenario(
                    scenario,
                    conversation_id,
                    force_retrieve=scenario["retrieval_needed"],
                    seed=seed,
                )
                result["policy"] = policy
                result["seed"] = seed
                results.append(result)

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
