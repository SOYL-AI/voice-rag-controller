import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_simulator_pipeline():

    scenarios = ROOT / "simulator" / "scenarios.json"

    # Remove old file if it exists
    if scenarios.exists():
        scenarios.unlink()

    # Run the real generator
    result = subprocess.run(
        [sys.executable, "simulator/generate_scenarios.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    assert scenarios.exists()

    data = json.loads(scenarios.read_text(encoding="utf-8"))

    assert len(data) > 0

    for scenario in data:
        assert scenario["expected_answer"].strip() != ""