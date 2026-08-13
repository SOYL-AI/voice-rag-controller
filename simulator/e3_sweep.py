"""
e3_sweep.py — Experiment 3's stub sweep runner.

Proves one thing, end to end: a baseline built against
controller/baseline.py's contract (Ctx / Decision / BaselineController)
actually plugs into the real pipeline and produces a real, readable log.
It is deliberately NOT the full E3 sweep — no price grid, no seeds, no
accuracy-vs-latency curves, no comparison across baselines. That's built
once there's more than one baseline (B0) to compare. Teammates use this
file the same way simulator/batch_runner.py's --single mode is used today:
a quick sanity check that a new component (here, a baseline) actually
works against the real pipeline before it's wired into anything bigger.

Usage:
    python -m simulator.e3_sweep --baseline b0 --limit-scenarios 3
    python -m simulator.e3_sweep --list-baselines

See controller/baseline.py for the contract every baseline implements, and
controller/baselines/b0_never_retrieve.py for the reference baseline this
file is tested against.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# import-for-side-effect: registers every baseline module (see
# controller/baselines/__init__.py) so list_baselines()/get_baseline()
# below can find them by name.
import controller.baselines  # noqa: F401
from controller.baseline import list_baselines
from pipeline.spine import run_conversation

SCENARIOS_FILE = Path(__file__).parent / "scenarios.json"
RESULTS_FILE = Path(__file__).parent / "e3_sweep_results.json"


def load_scenarios():
    """Same scenarios.json simulator/batch_runner.py reads — one shared
    scenario set across every experiment's runner, so E3 baselines are
    tested against the same questions E2's sweep already exercised."""
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def run_one_scenario(scenario, baseline_name: str, conversation_id: str) -> dict:
    """
    Run one scenario's question through the real pipeline with
    baseline=baseline_name, and return a small result record — enough to
    eyeball that the baseline actually ran (action/reason logged, log file
    produced) without pulling in harness/metrics.py's full scoring, which
    is out of scope for this stub (see this module's docstring).
    """
    scenario_id = scenario["id"]
    print(f"  Scenario {scenario_id}: {scenario['question']!r}")

    try:
        log_path = asyncio.run(
            run_conversation(
                conversation_id,
                question=scenario["question"],
                baseline=baseline_name,
            )
        )
        action = _read_controller_action(log_path)
        print(f"    -> controller_action: {action}")
        return {
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "question": scenario["question"],
            "status": "success",
            "log_path": str(log_path),
            "controller_action": action,
        }
    except Exception as error:  # noqa: BLE001 -- a sweep-stub sanity check should report *any* failure, not just the ones we anticipated
        print(f"    ERROR: scenario {scenario_id} failed: {error}")
        return {
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "question": scenario["question"],
            "status": "failed",
            "error": str(error),
        }


def _read_controller_action(log_path: Path) -> dict:
    """
    Pull the controller_action event's payload back out of the log this
    run just wrote — proof the baseline's decision actually reached the
    log, in the shape controller/baseline.py's BaselineController.
    log_decision() produces, not just that decide() returned something in
    memory.
    """
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            event = json.loads(line)
            if event.get("event_type") == "controller_action":
                return event["payload"]
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 stub sweep: run one baseline over a few "
        "scenarios end to end, to prove it plugs into the real pipeline. "
        "Not the full price-grid sweep — see this file's module docstring."
    )
    parser.add_argument(
        "--baseline",
        help="Registered baseline name to run (e.g. b0). See --list-baselines.",
    )
    parser.add_argument(
        "--limit-scenarios",
        type=int,
        default=None,
        help="Only run the first N scenarios from scenarios.json (default: all).",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="Print every currently-registered baseline name and exit.",
    )
    args = parser.parse_args()

    if args.list_baselines:
        names = list_baselines()
        print("Registered baselines:", ", ".join(names) if names else "(none)")
        return

    if not args.baseline:
        parser.error("--baseline is required (or pass --list-baselines).")

    registered = list_baselines()
    if args.baseline not in registered:
        parser.error(
            f"unknown baseline {args.baseline!r}. Registered: "
            f"{', '.join(registered) or '(none)'}"
        )

    scenarios = load_scenarios()
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    print("=" * 60)
    print("EXPERIMENT 3 — STUB SWEEP (proves a baseline plugs in)")
    print("=" * 60)
    print(f"Baseline  : {args.baseline}")
    print(f"Scenarios : {len(scenarios)}")
    print()

    results = []
    for scenario in scenarios:
        conversation_id = f"e3_{args.baseline}_{scenario['id']:03d}"
        results.append(run_one_scenario(scenario, args.baseline, conversation_id))

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print()
    print("=" * 60)
    print("STUB SWEEP COMPLETE")
    print("=" * 60)
    print(f"Total scenarios : {len(results)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")
    print(f"Results saved   : {RESULTS_FILE}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
