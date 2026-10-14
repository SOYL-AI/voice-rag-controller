"""
tts_engines.py - TTS adapter interface used by render.py.

Every engine turns (text, voice, rate control) into a float32 mono waveform at
the engine's native sample rate. render.py does everything else (resampling,
padding, level, alignment, noise), so adding an engine means implementing
`synthesize` and nothing more.

Engines
-------
PiperEngine   local, CPU; runs piper-tts in its own venv through tts_worker.py.
              Rate control: VITS length_scale (supports_length_scale = True).
ParlerEngine  local, GPU; Parler-TTS family (ai4bharat/indic-parler-tts for
              en-IN) in its own venv through tts_worker.py. Rate control: a pace
              word in the description, then render.py time-stretches to target.
AzureSpeechEngine
              cloud adapter (en-IN neural voices). Implemented to the SDK's
              documented interface but refuses to run unless
              VOICE_RAG_ALLOW_PAID_TTS=1 *and* credentials are set. The v2 tooling
              never sets that flag.

Interpreters for worker engines (override with env vars):
    VOICE_RAG_PIPER_PYTHON   default <repo>/.venv-speech/Scripts/python.exe
    VOICE_RAG_PARLER_PYTHON  default <repo>/.venv-parler/Scripts/python.exe
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from audio.v2 import voices as V

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WORKER = Path(__file__).resolve().parent / "tts_worker.py"


def _venv_python(env_var: str, venv: str) -> str:
    if os.environ.get(env_var):
        return os.environ[env_var]
    win = _REPO_ROOT / venv / "Scripts" / "python.exe"
    return str(win if win.exists() else _REPO_ROOT / venv / "bin" / "python")


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


class TTSEngine(ABC):
    engine: str = "?"
    supports_length_scale: bool = False
    version: str | None = None

    @abstractmethod
    def synthesize(self, text: str, voice: dict, *, length_scale: float | None = None,
                   pace: str = "moderate", seed: int = 0) -> tuple[np.ndarray, int]:
        """Return (float32 mono waveform, sample_rate)."""

    def close(self):
        pass


class _WorkerEngine(TTSEngine):
    """Drives tts_worker.py in another interpreter over JSON lines."""

    def __init__(self, python: str, engine: str, model: str, device: str = "cpu"):
        if not Path(python).exists():
            raise FileNotFoundError(
                f"{engine} interpreter not found at {python}; see docs/research_v2/SPEECH.md (setup)")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            [python, str(_WORKER), "--engine", engine, "--model", model, "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", env=env, cwd=str(_REPO_ROOT))
        hello = json.loads(self.proc.stdout.readline() or '{"ready": false, "error": "worker died"}')
        if not hello.get("ready"):
            self.close()
            raise RuntimeError(f"{engine} worker failed to load {model}: {hello.get('error')}")
        self.version = hello.get("engine_version")
        self._tmpdir = tempfile.mkdtemp(prefix=f"tts_{engine}_")

    def _call(self, req: dict) -> tuple[np.ndarray, int]:
        out = os.path.join(self._tmpdir, "out.wav")
        req = dict(req, out=out)
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline() or '{"ok": false, "error": "worker died"}')
        if not resp.get("ok"):
            raise RuntimeError(f"{self.engine} synthesis failed: {resp.get('error')}")
        return read_wav(out)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class PiperEngine(_WorkerEngine):
    engine = "piper"
    supports_length_scale = True

    def __init__(self, model: str):
        self.model = model
        path = V.ensure_piper_model(model)
        super().__init__(_venv_python("VOICE_RAG_PIPER_PYTHON", ".venv-speech"), "piper", str(path))
        self.version = f"piper-tts {self.version}"

    def synthesize(self, text, voice, *, length_scale=None, pace="moderate", seed=0):
        sid = V.piper_speaker_id(voice["model"], voice.get("speaker"))
        return self._call({"text": text, "speaker_id": sid, "length_scale": length_scale, "seed": seed})


class ParlerEngine(_WorkerEngine):
    engine = "parler"
    supports_length_scale = False

    def __init__(self, model: str, device: str = "cuda"):
        self.model = model
        super().__init__(_venv_python("VOICE_RAG_PARLER_PYTHON", ".venv-parler"), "parler", model, device)

    @staticmethod
    def description(voice: dict, pace: str) -> str:
        spk = voice.get("speaker")
        accent = {"en-IN": "an Indian English accent", "en-GB": "a British accent",
                  "en-US": "an American accent"}.get(voice.get("accent"), "a neutral accent")
        who = spk if spk else "A speaker"
        return (f"{who} speaks with {accent} at a {pace} pace in a clear, expressive voice. "
                f"The recording is very high quality, close-sounding, with no background noise.")

    def synthesize(self, text, voice, *, length_scale=None, pace="moderate", seed=0):
        return self._call({"text": text, "description": self.description(voice, pace), "seed": seed})


class AzureSpeechEngine(TTSEngine):
    """Cloud adapter for Azure Speech neural voices (e.g. en-IN-NeerjaNeural).

    Rate control maps to SSML <prosody rate>. Disabled unless the operator opts in,
    because it is a paid service.
    """

    engine = "azure"
    supports_length_scale = True  # via prosody rate = 1/length_scale

    def __init__(self, model: str):
        if os.environ.get("VOICE_RAG_ALLOW_PAID_TTS") != "1":
            raise RuntimeError("AzureSpeechEngine is a paid service and is disabled "
                               "(set VOICE_RAG_ALLOW_PAID_TTS=1 to opt in).")
        key, region = os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION")
        if not (key and region):
            raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set")
        import azure.cognitiveservices.speech as speechsdk  # optional dependency

        self._sdk = speechsdk
        cfg = speechsdk.SpeechConfig(subscription=key, region=region)
        cfg.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
        self._synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
        self.model = model
        self.version = "azure-speech"

    def synthesize(self, text, voice, *, length_scale=None, pace="moderate", seed=0):
        from xml.sax.saxutils import escape

        rate = 1.0 / (length_scale or 1.0)
        lang = "-".join(voice["model"].split("-")[:2])
        ssml = (f'<speak version="1.0" xml:lang="{lang}"><voice name="{voice["model"]}">'
                f'<prosody rate="{rate:.3f}">{escape(text)}</prosody></voice></speak>')
        res = self._synth.speak_ssml_async(ssml).get()
        if res.reason != self._sdk.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError(f"azure synthesis failed: {res.reason}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(res.audio_data)
        try:
            return read_wav(fh.name)
        finally:
            os.unlink(fh.name)


_CACHE: dict[tuple[str, str], TTSEngine] = {}


def get_engine(voice: dict) -> TTSEngine:
    key = (voice["engine"], voice["model"])
    if key not in _CACHE:
        if voice["engine"] == "piper":
            _CACHE[key] = PiperEngine(voice["model"])
        elif voice["engine"] == "parler":
            _CACHE[key] = ParlerEngine(voice["model"])
        elif voice["engine"] == "azure":
            _CACHE[key] = AzureSpeechEngine(voice["model"])
        else:
            raise ValueError(f"no adapter for engine {voice['engine']!r}")
    return _CACHE[key]


def close_all():
    for e in _CACHE.values():
        e.close()
    _CACHE.clear()


if __name__ == "__main__":  # quick manual check: python -m audio.v2.tts_engines <voice_id> "text"
    vid, text = sys.argv[1], sys.argv[2]
    v = V.get_voice(vid)
    x, sr = get_engine(v).synthesize(text, v)
    print(vid, sr, len(x) / sr, "s")
    close_all()
