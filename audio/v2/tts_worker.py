"""
tts_worker.py - a long-lived TTS process that runs inside the engine's own venv.

The rendering driver (render.py, in .venv-e2) talks to it over stdin/stdout,
one JSON object per line, so engines with conflicting dependencies (piper's
onnxruntime build, parler-tts' pinned transformers) never share an interpreter
with faster-whisper.

Request : {"text": "...", "out": "C:/.../x.wav", "speaker_id": 3 | null,
           "speaker": "Thoma" | null, "length_scale": 1.0 | null,
           "description": "..." | null, "seed": 123}
Response: {"ok": true, "sr": 22050, "n_samples": 51712, "synth_ms": 160.2}
          {"ok": false, "error": "..."}

The worker writes a 16-bit mono wav at the engine's native rate; resampling,
padding and level normalisation happen in render.py. This file must only use
the standard library plus the engine's own dependencies.

    python audio/v2/tts_worker.py --engine piper  --model audio/v2/models/piper/en_US-lessac-medium.onnx
    python audio/v2/tts_worker.py --engine parler --model ai4bharat/indic-parler-tts [--device cuda]
"""

import argparse
import json
import sys
import time
import wave


def _write_wav(path, int16_bytes, sr):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_bytes)


class PiperBackend:
    def __init__(self, model, device):
        from piper import PiperVoice

        self.voice = PiperVoice.load(model, use_cuda=(device == "cuda"))
        import piper

        self.version = getattr(piper, "__version__", None)
        try:
            from importlib.metadata import version

            self.version = version("piper-tts")
        except Exception:
            pass

    def synth(self, req):
        from piper import SynthesisConfig

        cfg = SynthesisConfig(speaker_id=req.get("speaker_id"), length_scale=req.get("length_scale"))
        chunks = list(self.voice.synthesize(req["text"], syn_config=cfg))
        sr = chunks[0].sample_rate
        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        _write_wav(req["out"], pcm, sr)
        return sr, len(pcm) // 2


class ParlerBackend:
    """Parler-TTS family (parler-tts/parler-tts-mini-*, ai4bharat/indic-parler-tts)."""

    def __init__(self, model, device):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model, torch_dtype=dtype).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = self.model.config.sampling_rate
        try:
            from importlib.metadata import version

            self.version = "parler-tts " + version("parler-tts")
        except Exception:
            self.version = "parler-tts"

    def synth(self, req):
        import numpy as np

        torch = self.torch
        torch.manual_seed(int(req.get("seed", 0)))
        desc = req.get("description") or "A clear voice speaking at a moderate pace, very clear audio."
        d = self.desc_tok(desc, return_tensors="pt").to(self.device)
        p = self.tok(req["text"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                      prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        audio = gen.float().cpu().numpy().squeeze()
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        _write_wav(req["out"], pcm, self.sr)
        return self.sr, len(pcm) // 2


BACKENDS = {"piper": PiperBackend, "parler": ParlerBackend}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = sys.stdout
    sys.stdout = sys.stderr  # library chatter must not corrupt the protocol
    try:
        backend = BACKENDS[a.engine](a.model, a.device)
        out.write(json.dumps({"ready": True, "engine_version": getattr(backend, "version", None)}) + "\n")
    except Exception as e:  # report load failures (e.g. gated model) to the driver
        out.write(json.dumps({"ready": False, "error": f"{type(e).__name__}: {e}"}) + "\n")
        out.flush()
        return 1
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            t0 = time.perf_counter()
            sr, n = backend.synth(req)
            resp = {"ok": True, "sr": sr, "n_samples": n, "synth_ms": (time.perf_counter() - t0) * 1000}
        except Exception as e:
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(resp) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
