import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_build_index_pipeline():

    index_file = ROOT / "retrieval" / "index.faiss"
    chunks_file = ROOT / "retrieval" / "chunks.json"

    # Remove old artifacts if they exist
    if index_file.exists():
        index_file.unlink()

    if chunks_file.exists():
        chunks_file.unlink()

    # Run the actual build script
    result = subprocess.run(
        [sys.executable, "retrieval/build_index.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    assert index_file.exists()

    assert chunks_file.exists()