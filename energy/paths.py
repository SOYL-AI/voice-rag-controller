"""Where measurement-time model weights live (outside the repo and OneDrive).

Weights are never committed. Override with VOICE_RAG_ENERGY_CACHE.
"""

import os
from pathlib import Path

CACHE = Path(os.environ.get("VOICE_RAG_ENERGY_CACHE",
                            Path.home() / ".cache" / "voice-rag-energy"))


def model_dir(repo_id: str) -> Path:
    """Local snapshot of a Hugging Face repo, e.g. 'BAAI/bge-small-en-v1.5'."""
    p = CACHE / "models" / repo_id.replace("/", "__")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Download with:\n  python -c \"from huggingface_hub import "
            f"snapshot_download; snapshot_download('{repo_id}', local_dir=r'{p}')\"")
    return p


def piper_voice() -> Path:
    return CACHE / "piper" / "en" / "en_US" / "lessac" / "medium" / "en_US-lessac-medium.onnx"
