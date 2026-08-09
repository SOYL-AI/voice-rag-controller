import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_rewrite_pipeline():

    result = subprocess.run(
        [sys.executable, "retrieval/rewrite.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    output = result.stdout.strip()

    # The script should print a rewritten query
    assert output != ""

    # In mock mode it should return the deterministic mock response
    assert "What about kids?" in output