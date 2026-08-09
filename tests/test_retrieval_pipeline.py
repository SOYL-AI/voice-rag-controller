import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_retrieval_pipeline():

    # Ensure retrieval artifacts exist
    subprocess.run(
        [sys.executable, "retrieval/build_index.py"],
        cwd=ROOT,
        check=True,
    )

    # Run the real search script
    result = subprocess.run(
        [sys.executable, "retrieval/search.py"],
        cwd=ROOT,
        input="pool\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    output = result.stdout

    assert "Top Results" in output
    assert "Result 1" in output