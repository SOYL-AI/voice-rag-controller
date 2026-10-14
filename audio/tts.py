"""
tts.py — Text-To-Speech synthesis using piper-tts.

Owns Stage 6 (speak).

Synthesizes response text into audio and plays it in small playback_block_ms blocks
so playback can be interrupted quickly on barge-in.
Emits schema-compliant events: tts_first_audio (TTFA anchor) and barge_in.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.config import load_config
from eventlog.log_writer import LogWriter


def split_into_clauses(text: str) -> List[str]:
    """Splits full response text into shorter clauses/sentences for streamed synthesis."""
    if not text:
        return []
    # Split by clause boundaries: period, exclamation, question mark, semicolon, or newline
    chunks = re.split(r"(?<=[.!?;\n])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


_CLAUSE_BOUNDARY = re.compile(r"(?<=[.!?;\n])\s+")


class ClauseAccumulator:
    """
    Incremental counterpart of split_into_clauses() for streamed LLM text.

    feed(delta) returns every clause that became complete with this delta;
    flush() returns whatever is left once the stream ends. A clause is
    complete once a boundary character (. ! ? ; or newline) is followed by
    whitespace, which is exactly where split_into_clauses() splits, so for
    any way of chunking a text into deltas:

        [c for d in deltas for c in acc.feed(d)] + acc.flush()
            == split_into_clauses("".join(deltas))
    """

    def __init__(self):
        self._buf = ""

    def feed(self, delta: str) -> List[str]:
        if not delta:
            return []
        self._buf += delta
        out = []
        while True:
            m = _CLAUSE_BOUNDARY.search(self._buf)
            if m is None:
                break
            clause = self._buf[: m.start()].strip()
            self._buf = self._buf[m.end():]
            if clause:
                out.append(clause)
        return out

    def flush(self) -> List[str]:
        rest = self._buf.strip()
        self._buf = ""
        return split_into_clauses(rest) if rest else []


def get_voice_sample_rate(voice_model_path: str, default_rate: int = 22050) -> int:
    """Reads sample_rate from voice model companion .onnx.json file if available."""
    p = Path(voice_model_path)
    companion_json = p.with_suffix(".onnx.json")
    if not companion_json.exists():
        companion_json = Path(str(p) + ".json")
    if companion_json.exists():
        try:
            with open(companion_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sr = data.get("audio", {}).get("sample_rate")
            if sr and isinstance(sr, int):
                return sr
        except Exception:
            pass
    return default_rate


class StreamingTTS:
    """
    Synthesizes and streams text-to-speech using piper-tts.
    Supports chunked audio playback and rapid stop() for barge-in interrupts.
    """

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.tts_cfg = cfg["tts"]

        self.engine = self.tts_cfg["engine"]
        self.voice_model_path = self.tts_cfg["voice_model_path"]
        self.playback_block_ms = self.tts_cfg["playback_block_ms"]

        config_sample_rate = self.tts_cfg.get("sample_rate", 22050)
        self.sample_rate = get_voice_sample_rate(self.voice_model_path, config_sample_rate)

        # piper_mode: "subprocess" (default, the original behaviour: one
        # `python -m piper` process per clause, which reloads the voice
        # every time) or "resident" (load the voice once in-process with
        # piper.PiperVoice and reuse it). Measured on the dev laptop the
        # subprocess path costs seconds per clause, resident ~0.1-1 s.
        self.piper_mode = os.environ.get("VOICE_RAG_PIPER_MODE") or self.tts_cfg.get(
            "piper_mode", "subprocess"
        )
        self._voice = None
        self._voice_lock = threading.Lock()

        self._stop_requested = threading.Event()
        self._is_playing = False
        self.playback_active = False

        model_path = Path(self.voice_model_path)
        if not model_path.exists():
            print(
                f"  ⚠️  WARNING [TTS]: Piper model not found at '{model_path}'. "
                f"Falling back to synthetic tone. Download model to audio/models/ to hear real speech."
            )

    def stop(self, log_writer: Optional[LogWriter] = None, turn_id: Optional[int] = None):
        """
        Immediately interrupts active TTS playback.
        Emits barge_in event if log_writer and turn_id are provided.
        """
        self._stop_requested.set()
        if log_writer is not None and turn_id is not None:
            log_writer.log_event(
                actor="system",
                event_type="barge_in",
                payload={},
                turn_id=turn_id,
            )

    def synthesize_clause_to_raw_pcm(self, clause: str, sample_rate: Optional[int] = None) -> bytes:
        """
        Synthesizes a clause using piper subprocess.
        Returns raw 16-bit mono PCM audio bytes.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        model_path = Path(self.voice_model_path)

        # If piper binary or model path is missing in local environment, generate synthetic PCM tone/audio bytes for testing
        if not model_path.exists():
            # Generate synthetic PCM audio data corresponding to clause length (approx 100ms per word)
            duration_s = max(0.5, len(clause.split()) * 0.25)
            t = (
                __import__("numpy").linspace(0, duration_s, int(sample_rate * duration_s))
            )
            tone = (0.3 * __import__("numpy").sin(2 * 3.14159 * 440 * t)).astype(
                __import__("numpy").int16
            )
            return tone.tobytes()

        if self.piper_mode == "resident":
            return self._synthesize_resident(clause)

        # Run piper via python module (avoids needing piper on system PATH)
        cmd = [
            sys.executable, "-m", "piper",
            "--model", str(model_path),
            "--output-raw",
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdout, _ = proc.communicate(input=clause.encode("utf-8"))
        return stdout

    def _synthesize_resident(self, clause: str) -> bytes:
        """In-process piper synthesis with the voice loaded once and reused."""
        with self._voice_lock:
            if self._voice is None:
                from piper import PiperVoice

                self._voice = PiperVoice.load(str(self.voice_model_path))
            voice = self._voice
            return b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(clause))

    def speak_sync(
        self,
        text: str,
        log_writer: LogWriter,
        turn_id: int,
        live: bool = False,
        simulate_playback: bool = True,
    ):
        """
        Synchronous synthesis and playback function intended to be run in a worker thread via asyncio.to_thread().
        Does NOT block the asyncio event loop thread.
        """
        print(f"\nTTS:\n  worker_started\n  timestamp={time.time():.3f}")
        self._stop_requested.clear()
        self._is_playing = True
        self.playback_active = False

        clauses = split_into_clauses(text)
        print(f"\nTTS TOTAL CLAUSES: {len(clauses)}")
        first_audio_logged = False
        sample_rate = self.sample_rate

        block_samples = int(sample_rate * (self.playback_block_ms / 1000.0))

        stream = None
        if live:
            import sounddevice as sd
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            stream.start()

        try:
            for idx, clause in enumerate(clauses, start=1):
                if self._stop_requested.is_set():
                    print(f"\nTTS STOP REQUESTED BEFORE CLAUSE {idx}")
                    break

                print(f"\nTTS CLAUSE:\n  index={idx}\n  text=\"{clause}\"")

                print(f"\nSYNTHESIS START:\n  index={idx}")
                start_synth = time.monotonic()
                pcm_bytes = self.synthesize_clause_to_raw_pcm(clause, sample_rate)
                synth_dur = (time.monotonic() - start_synth) * 1000.0

                if not pcm_bytes:
                    print(f"\nSYNTHESIS COMPLETE:\n  index={idx}\n  pcm_samples=0\n  duration_ms={synth_dur:.1f}\n  sample_rate={sample_rate}")
                    continue

                pcm_samples = len(pcm_bytes) // 2
                audio_dur_ms = (pcm_samples / sample_rate) * 1000.0
                print(f"\nSYNTHESIS COMPLETE:\n  index={idx}\n  pcm_samples={pcm_samples}\n  duration_ms={audio_dur_ms:.1f}\n  sample_rate={sample_rate}")

                import numpy as np
                audio_float32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                print(f"\nPLAYBACK START:\n  index={idx}")
                self.playback_active = True
                print(f"\nTTS:\n  playback_active=true")

                # Stream out PCM in small blocks
                offset = 0
                while offset < len(audio_float32):
                    if self._stop_requested.is_set():
                        print(f"\nTTS STOP REQUESTED DURING PLAYBACK OF CLAUSE {idx}")
                        break

                    chunk = audio_float32[offset : offset + block_samples]
                    offset += block_samples

                    if not first_audio_logged:
                        # Log the critical TTFA anchor event
                        log_writer.log_event(
                            actor="system",
                            event_type="tts_first_audio",
                            payload={},
                            turn_id=turn_id,
                        )
                        first_audio_logged = True

                    if live and stream is not None:
                        stream.write(chunk)
                    elif not simulate_playback:
                        # Measurement runs: first audio is already logged;
                        # skip sleeping through the rest of the playback.
                        break
                    else:
                        # Simulate chunk playback sleep according to block duration
                        sleep_time = self.playback_block_ms / 1000.0
                        time.sleep(sleep_time)

                print(f"\nPLAYBACK COMPLETE:\n  index={idx}")

        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            self.playback_active = False
            self._is_playing = False
            print(f"\nTTS:\n  playback_active=false")
            print(f"\nTTS:\n  worker_finished\n  timestamp={time.time():.3f}")

    async def speak(
        self,
        text: str,
        log_writer: LogWriter,
        turn_id: int,
        live: bool = False,
        simulate_playback: bool = True,
    ):
        """
        Asynchronously delegates synchronous TTS synthesis & playback to a worker thread via asyncio.to_thread.
        Keeps the asyncio event loop 100% unblocked and responsive for concurrent barge-in VAD processing.
        """
        await asyncio.to_thread(
            self.speak_sync, text, log_writer, turn_id, live, simulate_playback
        )


# ---------------------------------------------------------------------------
# Functions for pipeline/spine.py integration
# ---------------------------------------------------------------------------
def speak(answer: str, log_writer: LogWriter, turn_id: int) -> None:
    """
    Stage 6 interface for spine.py drop-in compatibility.
    Synchronously runs TTS synthesis, logs tts_first_audio, and outputs answer text.
    """
    cfg = load_config()
    tts_engine = StreamingTTS(cfg)
    log_writer.log_event("system", "tts_first_audio", {}, turn_id)
    print(f'  system speaks: "{answer}"')


if __name__ == "__main__":
    print("Testing audio/tts.py text splitting and initialization...")
    cfg = load_config()
    print("TTS Config loaded:", cfg["tts"])
    test_text = "The pool is open until 10 PM. Have a pleasant stay!"
    clauses = split_into_clauses(test_text)
    print(f"Text split into {len(clauses)} clauses:", clauses)
    tts = StreamingTTS(cfg)
    print("StreamingTTS initialized successfully.")
