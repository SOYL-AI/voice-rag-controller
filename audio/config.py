"""
config.py — loads the `audio:` section of configs/config.yaml.

No hardcoded model sizes, voice paths, device settings, or intervals in audio/.
Everything is loaded from configs/config.yaml per project rules.

Usage:
    from audio.config import load_config
    cfg = load_config()
    stt_cfg = cfg["stt"]
    tts_cfg = cfg["tts"]
    vad_cfg = cfg["vad"]
"""

import copy
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"

REQUIRED_STT_KEYS = {
    "model_size",
    "device",
    "compute_type",
    "partial_interval_ms",
    "silence_hang_ms",
}

REQUIRED_TTS_KEYS = {
    "engine",
    "voice_model_path",
    "playback_block_ms",
    "sample_rate",
}

REQUIRED_VAD_KEYS = {
    "frame_ms",
}


def config_path() -> Path:
    """Where we read settings from. VOICE_RAG_CONFIG overrides it."""
    override = os.environ.get("VOICE_RAG_CONFIG")
    return Path(override) if override else _DEFAULT_CONFIG_PATH


def load_config(path=None) -> dict:
    """
    Read configs/config.yaml and return the validated `audio:` section.

    Raises FileNotFoundError, KeyError, or ValueError if any required section
    or setting is missing.
    """
    import yaml

    cfg_file = Path(path) if path else config_path()
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"No config at {cfg_file}. It should be committed at "
            f"configs/config.yaml - see configs/README.md."
        )

    with open(cfg_file, "r", encoding="utf-8") as fh:
        full = yaml.safe_load(fh) or {}

    if "audio" not in full:
        raise KeyError(f"{cfg_file} has no `audio:` section.")

    audio_cfg = copy.deepcopy(full["audio"])

    # Validate sub-sections
    for section_name in ("stt", "tts", "vad"):
        if section_name not in audio_cfg:
            raise KeyError(
                f"Missing required sub-section `audio.{section_name}` in {cfg_file}"
            )

    stt = audio_cfg["stt"]
    missing_stt = REQUIRED_STT_KEYS - set(stt.keys())
    if missing_stt:
        raise KeyError(
            f"Missing required keys in `audio.stt`: {sorted(missing_stt)} in {cfg_file}"
        )

    tts = audio_cfg["tts"]
    missing_tts = REQUIRED_TTS_KEYS - set(tts.keys())
    if missing_tts:
        raise KeyError(
            f"Missing required keys in `audio.tts`: {sorted(missing_tts)} in {cfg_file}"
        )

    vad = audio_cfg["vad"]
    missing_vad = REQUIRED_VAD_KEYS - set(vad.keys())
    if missing_vad:
        raise KeyError(
            f"Missing required keys in `audio.vad`: {sorted(missing_vad)} in {cfg_file}"
        )

    return audio_cfg


if __name__ == "__main__":
    c = load_config()
    print(f"config file : {config_path()}")
    print(f"STT config  : {c['stt']}")
    print(f"TTS config  : {c['tts']}")
    print(f"VAD config  : {c['vad']}")
