"""
asr.py - load the project's streaming STT stack (audio/stt.py) for offline replay.

* Silero VAD and faster-whisper are obtained through audio/stt.py's own loaders
  (get_vad_model, get_whisper_model) and StreamingSTT, so the replay uses the
  same models, VAD frame logic and config values as the live pipeline.
* The only override is where Whisper runs (device / compute_type, and
  optionally model size). The v2 default is the configured model size on the
  GPU (float16); everything else comes from configs/config.yaml `audio:`.

CUDA on Windows: CTranslate2 needs cuBLAS 12 and cuDNN 9 DLLs. If they are not
on PATH, we add the nvidia/*/bin folders of the pip wheels installed in
.venv-speech (or VOICE_RAG_CUDA_DLL_DIRS, os.pathsep-separated).
"""

from __future__ import annotations

import copy
import glob
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DLL_DONE = False


def add_cuda_dll_dirs():
    global _DLL_DONE
    if _DLL_DONE or os.name != "nt":
        return
    dirs = [d for d in os.environ.get("VOICE_RAG_CUDA_DLL_DIRS", "").split(os.pathsep) if d]
    if not dirs:
        dirs = glob.glob(str(_REPO_ROOT / ".venv-speech" / "Lib" / "site-packages" / "nvidia" / "*" / "bin"))
    for d in dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    _DLL_DONE = True


def stt_config(model_size: str | None = None, device: str = "cuda",
               compute_type: str | None = None) -> dict:
    """configs/config.yaml `audio:` section with the Whisper placement overridden."""
    from audio.config import load_config

    cfg = copy.deepcopy(load_config())
    if model_size:
        cfg["stt"]["model_size"] = model_size
    cfg["stt"]["device"] = device
    cfg["stt"]["compute_type"] = compute_type or ("float16" if device == "cuda" else "int8")
    return cfg


def load_streaming_stt(cfg: dict):
    """An audio.stt.StreamingSTT built from `cfg` (loads Whisper + Silero once per process)."""
    if cfg["stt"]["device"] == "cuda":
        add_cuda_dll_dirs()
    from audio.stt import StreamingSTT

    return StreamingSTT(cfg=cfg, sample_rate=16000)


def whisper_for_alignment_check(device: str = "cuda"):
    """The configured Whisper model, for the word-timestamp cross-check in render.py."""
    cfg = stt_config(device=device)
    if device == "cuda":
        add_cuda_dll_dirs()
    from audio.stt import get_whisper_model

    return get_whisper_model(cfg)
