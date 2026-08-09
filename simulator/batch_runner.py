import json
import sys
from pathlib import Path

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.spine import run_conversation


SCENARIOS_FILE = Path(__file__).parent / "scenarios.json"
RESULTS_FILE = Path(__file__).parent / "batch_results.json"


def load_scenarios():
    """Load simulator scenarios from scenarios.json."""
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def run_scenario(scenario):
    """
    Run one simulator scenario through the pipeline.

    The current spine.py interface does not yet accept the scenario
    question, so for now we use its existing run_conversation()
    interface. The integration point can be updated later without
    changing the batch runner structure.
    """
    scenario_id = scenario["id"]
    conversation_id = f"conv_{scenario_id:03d}"

    print("=" * 60)
    print(f"Running scenario {scenario_id}")
    print(f"Question: {scenario['question']}")
    print(f"Conversation ID: {conversation_id}")
    print("=" * 60)

    try:
        log_path = run_conversation(conversation_id)

        return {
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "question": scenario["question"],
            "expected_answer": scenario["expected_answer"],
            "status": "success",
            "log_path": str(log_path),
        }

    except Exception as error:
        print(f"ERROR: Scenario {scenario_id} failed: {error}")

        return {
            "scenario_id": scenario_id,
            "conversation_id": conversation_id,
            "question": scenario["question"],
            "expected_answer": scenario["expected_answer"],
            "status": "failed",
            "error": str(error),
        }


def main():
    scenarios = load_scenarios()

    print("=" * 60)
    print("VOICE RAG BATCH RUNNER")
    print("=" * 60)
    print(f"Loaded {len(scenarios)} scenarios.")
    print()

    results = []

    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    successful = sum(
        1 for result in results if result["status"] == "success"
    )
    failed = len(results) - successful

    print()
    print("=" * 60)
    print("BATCH RUN COMPLETE")
    print("=" * 60)
    print(f"Total scenarios : {len(results)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")
    print(f"Results saved   : {RESULTS_FILE}")


if __name__ == "__main__":
    main()