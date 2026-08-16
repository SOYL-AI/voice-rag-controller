import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_simulator_pipeline(tmp_path):
    """The scenario generator runs and emits usable scenarios.

    Writes to tmp_path, NOT to simulator/scenarios.json. This test used to
    delete the committed file and regenerate it in place, which was harmless
    while that file was untracked and generator-derived. It is neither now:
    simulator/scenarios.json is the curated 25-positive / 25-negative E3
    dataset, and the generator hardcodes retrieval_needed=True on every
    scenario it emits (see simulator/generate_scenarios.py). Running the test
    suite therefore silently replaced the balanced dataset with the legacy
    all-positive one, collapsing E3's VoI distribution and its price grid.
    """

    generated = tmp_path / "scenarios.json"

    result = subprocess.run(
        [sys.executable, "simulator/generate_scenarios.py", "--out", str(generated)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    assert generated.exists()

    data = json.loads(generated.read_text(encoding="utf-8"))

    assert len(data) > 0

    for scenario in data:
        assert scenario["expected_answer"].strip() != ""


def test_committed_scenarios_are_the_balanced_e3_set():
    """simulator/scenarios.json is the balanced set E3's grid is derived from.

    A regression guard for the failure above: if anything regenerates or
    overwrites this file with the all-positive generator output, the E3 price
    grid silently loses its negative class and every scenario's VoI saturates.
    Cheap to assert, and it fails loudly at the point of damage rather than
    surfacing later as an unexplained grid.
    """

    scenarios = json.loads(
        (ROOT / "simulator" / "scenarios.json").read_text(encoding="utf-8")
    )

    needed = [s for s in scenarios if s.get("retrieval_needed")]
    not_needed = [s for s in scenarios if not s.get("retrieval_needed")]

    assert len(scenarios) == 50, f"expected 50 scenarios, found {len(scenarios)}"
    assert len(needed) == 25, (
        f"expected 25 retrieval_needed scenarios, found {len(needed)} — has "
        f"simulator/generate_scenarios.py overwritten the curated dataset?"
    )
    assert len(not_needed) == 25, (
        f"expected 25 not-needed scenarios, found {len(not_needed)} — has "
        f"simulator/generate_scenarios.py overwritten the curated dataset?"
    )
