"""
voices.py - registry of the TTS voices used for v2 speech stimuli.

voice_id naming
---------------
    <engine>-<model>[-spk_<speaker>]

    piper-en_US-lessac-medium               single-speaker piper voice
    piper-en_US-l2arctic-medium-spk_RRBI    speaker RRBI of a multi-speaker piper model
    parler-indic-spk_Rohit                  a named speaker of Indic Parler-TTS
    azure-en-IN-NeerjaNeural                cloud voice (adapter only, never called here)

    human-<code>                            a team member's recordings (record_prompts.py)

A rendered *condition* adds the speaking rate and the noise:
    <voice_id>__<wpm>wpm__<noise>           e.g. piper-en_US-lessac-medium__150wpm__babble_10dB
    <voice_id>__natural__<noise>            uncontrolled rate (human recordings)
which is the trace file stem (datasets/v2/<name>/speech/traces/<condition>.jsonl).

`status` says whether a voice can be rendered on this machine:
    active        downloaded on demand from a public, ungated source; used by default
    screen        local and renderable, but only used in the accent screen
    needs_auth    the model is gated (licence click-through + HF token); adapter ready
    adapter_only  paid / cloud engine; interface only, never invoked by the tooling
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = _REPO_ROOT / "audio" / "v2" / "models"
PIPER_DIR = MODELS_DIR / "piper"
PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Licences are those of the *training data* named in each piper MODEL_CARD
# (piper code and the voice weights repo are MIT). All are fine for research
# stimuli; several are non-commercial, which matters only if the rendered audio
# itself were redistributed commercially.
_LIC = {
    "lessac": "Blizzard 2013 Lessac licence (research use)",
    "hfc": "CC BY-NC-SA 4.0 (Hi-Fi-CAPTAIN)",
    "cori": "public domain (LibriVox)",
    "northern": "CC BY-SA 4.0 (OpenSLR 83)",
    "l2arctic": "CC BY-NC 4.0 (L2-ARCTIC)",
    "arctic": "CMU ARCTIC licence (permissive, attribution)",
    "vctk": "CC BY 4.0 (VCTK)",
    "indic_parler": "Apache-2.0 (model); gated on Hugging Face",
}


def _piper(model, accent, gender, lic, status="active", speaker=None, note=""):
    vid = f"piper-{model}" + (f"-spk_{speaker}" if speaker else "")
    return vid, {
        "voice_id": vid, "engine": "piper", "model": model, "speaker": speaker,
        "accent": accent, "gender": gender, "licence": _LIC[lic], "status": status,
        "note": note,
    }


VOICES: dict[str, dict] = dict([
    # ---- default rendering set --------------------------------------------------
    _piper("en_US-lessac-medium", "en-US", "F", "lessac",
           note="project default voice (configs/config.yaml audio.tts)"),
    _piper("en_US-hfc_male-medium", "en-US", "M", "hfc"),
    _piper("en_GB-cori-medium", "en-GB", "F", "cori"),
    _piper("en_GB-northern_english_male-medium", "en-GB", "M", "northern"),
    _piper("en_US-l2arctic-medium", "en-IN", "F", "l2arctic", speaker="SVBI",
           note="L2-ARCTIC speaker SVBI, Hindi L1; accent screen top-1 indian 14/20"),
    _piper("en_US-arctic-medium", "en-IN", "M", "arctic", speaker="ksp",
           note="CMU ARCTIC speaker ksp, Indian English; accent screen top-1 indian 15/20"),
    # ---- accent-screen candidates (en-IN) and accent controls ----------------------
    _piper("en_US-l2arctic-medium", "en-IN", "M", "l2arctic", "screen", "ASI", "Hindi L1"),
    _piper("en_US-l2arctic-medium", "en-IN", "M", "l2arctic", "screen", "RRBI", "Hindi L1"),
    _piper("en_US-l2arctic-medium", "en-IN", "F", "l2arctic", "screen", "TNI", "Hindi L1"),
    _piper("en_US-arctic-medium", "en-IN", "M", "arctic", "screen", "aup", "Indian English"),
    _piper("en_US-arctic-medium", "en-IN", "M", "arctic", "screen", "gka", "Indian English"),
    _piper("en_US-arctic-medium", "en-IN", "F", "arctic", "screen", "axb", "Indian English"),
    _piper("en_US-arctic-medium", "en-IN", "F", "arctic", "screen", "slp", "Indian English"),
    _piper("en_GB-vctk-medium", "en-IN", "F", "vctk", "screen", "p248", "VCTK, Indian accent"),
    _piper("en_GB-vctk-medium", "en-IN", "M", "vctk", "screen", "p251", "VCTK, Indian accent"),
    _piper("en_GB-vctk-medium", "en-IN", "M", "vctk", "screen", "p376", "VCTK, Indian accent"),
    _piper("en_US-arctic-medium", "en-US", "M", "arctic", "screen", "bdl", "US control"),
    _piper("en_GB-vctk-medium", "en-GB", "F", "vctk", "screen", "p225", "southern English control"),
])

# Indic Parler-TTS (AI4Bharat). Apache-2.0, but the HF repo is gated: an HF
# account must accept the terms and HF_TOKEN must be set. Speaker names are the
# English-capable speakers listed on the model card.
for _spk, _g in (("Thoma", "M"), ("Mary", "F")):
    _vid = f"parler-indic-spk_{_spk}"
    VOICES[_vid] = {
        "voice_id": _vid, "engine": "parler", "model": "ai4bharat/indic-parler-tts",
        "speaker": _spk, "accent": "en-IN", "gender": _g, "licence": _LIC["indic_parler"],
        "status": "needs_auth", "note": "gated repo; see docs/research_v2/SPEECH.md",
    }

# Cloud adapter placeholder (see tts_engines.AzureSpeechEngine). Never called.
VOICES["azure-en-IN-NeerjaNeural"] = {
    "voice_id": "azure-en-IN-NeerjaNeural", "engine": "azure", "model": "en-IN-NeerjaNeural",
    "speaker": None, "accent": "en-IN", "gender": "F", "licence": "paid service",
    "status": "adapter_only", "note": "interface only",
}

DEFAULT_VOICES = [v for v, e in VOICES.items() if e["status"] == "active"]

# Fixed, dataset-independent sentences for per-voice rate calibration and the
# accent screen: spoken-question style, 5-16 words, mixed domains.
CALIBRATION_SENTENCES = [
    "What time does the swimming pool close on weekends?",
    "Can I get a late checkout if my flight leaves in the evening?",
    "How much does it cost to add an international roaming pack to my plan?",
    "Is breakfast included with the deluxe room or do I pay separately?",
    "Who was the first person to walk on the moon?",
    "What is the boiling point of water in Fahrenheit?",
    "My card was charged twice for the same order, what should I do?",
    "Which river flows through the city of Cairo?",
    "Do you have any vegetarian options on the room service menu?",
    "How long does it take for a refund to show up in my bank account?",
    "What documents do I need to open a savings account?",
    "Can you tell me how to reset the password on my router?",
    "Where is the nearest pharmacy to the hotel?",
    "What year did the Second World War end?",
    "Why is my internet so slow in the evenings?",
    "How many calories are there in a banana?",
    "Is there a penalty if I cancel my booking two days before arrival?",
    "What is the difference between a debit card and a credit card?",
    "Could you book a taxi to the airport for six in the morning?",
    "Which planet has the most moons?",
]
SCREEN_VOICES = [v for v, e in VOICES.items() if e["engine"] == "piper"]
DEFAULT_RATES = [130, 150, 180]
DEFAULT_NOISES = ["clean", "babble_10dB", "babble_5dB"]


def get_voice(voice_id: str) -> dict:
    if voice_id not in VOICES:
        raise KeyError(f"unknown voice_id {voice_id!r}; known: {sorted(VOICES)}")
    return VOICES[voice_id]


def rate_token(wpm) -> str:
    """130 -> "130wpm"; "natural" (uncontrolled rate, e.g. human recordings) -> "natural"."""
    return "natural" if wpm in (None, "natural") else f"{int(wpm)}wpm"


def condition_id(voice_id: str, wpm, noise: str) -> str:
    return f"{voice_id}__{rate_token(wpm)}__{noise}"


def parse_condition(cond: str):
    voice_id, rate, noise = cond.split("__")
    return voice_id, ("natural" if rate == "natural" else int(rate.replace("wpm", ""))), noise


def piper_model_path(model: str) -> Path:
    return PIPER_DIR / f"{model}.onnx"


def ensure_piper_model(model: str) -> Path:
    """Download <model>.onnx, .onnx.json and MODEL_CARD from rhasspy/piper-voices if missing."""
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    lang, name, quality = model.split("-")
    family = lang.split("_")[0]
    rel = f"{family}/{lang}/{name}/{quality}"
    out = piper_model_path(model)
    for fname, dest in ((f"{model}.onnx.json", out.with_suffix(".onnx.json")),
                        ("MODEL_CARD", PIPER_DIR / f"{model}.MODEL_CARD"),
                        (f"{model}.onnx", out)):
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{PIPER_BASE_URL}/{rel}/{fname}"
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"  downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    return out


def piper_speaker_id(model: str, speaker: str | None) -> int | None:
    if speaker is None:
        return None
    cfg = json.loads(piper_model_path(model).with_suffix(".onnx.json").read_text(encoding="utf-8"))
    return int(cfg["speaker_id_map"][speaker])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="List voices or download piper models.")
    ap.add_argument("--download", nargs="*", help="voice_ids to fetch (default: all piper voices)")
    a = ap.parse_args()
    if a.download is not None:
        ids = a.download or SCREEN_VOICES
        for m in sorted({get_voice(v)["model"] for v in ids if get_voice(v)["engine"] == "piper"}):
            print(ensure_piper_model(m))
    else:
        for v, e in VOICES.items():
            print(f"{v:48s} {e['accent']:6s} {e['gender']} {e['status']:12s} {e['licence']}")
