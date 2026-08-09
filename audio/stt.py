"""
stt.py — Speech-To-Text streaming engine using faster-whisper and Silero VAD.

Owns Stage 1 (audio_in: mic capture + Silero VAD gating) and Stage 2
(transcribe: partial/final transcription).

Reads configuration parameters directly from configs/config.yaml via audio.config.
Emits schema-compliant events: asr_partial, asr_final, user_end.
"""

import asyncio
import sys
import time
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.config import load_config
from eventlog.log_writer import LogWriter

# Lazy imports for heavy libraries to ensure quick import times if required
_WHISPER_MODEL = None
_VAD_MODEL = None


def _save_debug_wav(audio_data: np.ndarray, sample_rate: int, turn_id: int) -> Path:
    """Saves float32 numpy audio array as a 16-bit mono WAV file to debug_audio/turn_XXX.wav."""
    out_dir = _REPO_ROOT / "debug_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"turn_{turn_id:03d}.wav"

    pcm16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    print(f"\n  💾 Audio saved : {out_path.relative_to(_REPO_ROOT)}")
    return out_path


def get_whisper_model(cfg: Optional[dict] = None):
    """Lazy initialization of faster-whisper model."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        if cfg is None:
            cfg = load_config()
        stt_cfg = cfg["stt"]
        from faster_whisper import WhisperModel

        _WHISPER_MODEL = WhisperModel(
            stt_cfg["model_size"],
            device=stt_cfg["device"],
            compute_type=stt_cfg["compute_type"],
        )
    return _WHISPER_MODEL


def get_vad_model():
    """Lazy initialization of Silero VAD model."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        import torch

        # Load silero-vad from torch hub
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        model.eval()
        _VAD_MODEL = model
    return _VAD_MODEL


class StreamingSTT:
    """
    Manages audio frame ingestion, Silero VAD speech detection,
    periodic partial transcription via faster-whisper, and final transcription on user silence.
    """

    def __init__(self, cfg: Optional[dict] = None, sample_rate: int = 16000):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.stt_cfg = cfg["stt"]
        self.vad_cfg = cfg["vad"]
        self.sample_rate = sample_rate

        self.partial_interval_ms = self.stt_cfg["partial_interval_ms"]
        self.silence_hang_ms = self.stt_cfg["silence_hang_ms"]
        self.frame_ms = self.vad_cfg["frame_ms"]
        self.language = self.stt_cfg.get("language", "en")

        # Number of samples per VAD frame (e.g., 32ms at 16kHz = 512 samples)
        self.frame_samples = int(self.sample_rate * (self.frame_ms / 1000.0))

        self.model = get_whisper_model(self.cfg)
        self.vad_model = get_vad_model()

    def process_vad_frame(self, frame_np: np.ndarray) -> float:
        """Runs Silero VAD on a 1D float32 numpy audio frame (16kHz). Returns speech probability."""
        import torch

        if frame_np.dtype != np.float32:
            frame_np = frame_np.astype(np.float32)

        # Silero VAD requires minimum chunk size of 512 samples (32ms @ 16kHz)
        min_samples = 512
        if len(frame_np) < min_samples:
            frame_np = np.pad(frame_np, (0, min_samples - len(frame_np)))
        elif len(frame_np) > min_samples:
            frame_np = frame_np[-min_samples:]

        tensor = torch.from_numpy(frame_np)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            speech_prob = self.vad_model(tensor, self.sample_rate).item()
        return speech_prob


    async def transcribe_stream(
        self,
        audio_chunk_generator: AsyncGenerator[np.ndarray, None],
        log_writer: LogWriter,
        turn_id: int,
        vad_threshold: float = 0.5,
    ) -> str:
        """
        Consumes streaming float32 16kHz audio chunks.
        Performs VAD speech detection, logs asr_partial at partial_interval_ms intervals,
        logs user_end on VAD-confirmed trailing silence, and returns final asr_final transcript.
        """
        # Reset Silero VAD states for clean turn evaluation
        if hasattr(self.vad_model, "reset_states"):
            try:
                self.vad_model.reset_states()
            except Exception:
                pass

        buffer = []          # rolling window for VAD (all audio, for frame evaluation)
        speech_buffer = []   # ONLY audio collected after speech onset — fed to Whisper
        speech_started = False
        last_partial_time = time.monotonic()
        silence_start_time = None
        accumulated_samples = 0
        _frame_count = 0
        # RMS gate: skip VAD on frames below the noise floor.
        # Config-driven (0.003 default for Phase 1A) to prevent suppressing quiet speech.
        RMS_GATE = float(self.stt_cfg.get("rms_gate", 0.003))

        async for chunk in audio_chunk_generator:
            if chunk.ndim > 1:
                chunk = chunk.squeeze()

            buffer.append(chunk)
            accumulated_samples += len(chunk)
            _frame_count += 1

            full_audio = np.concatenate(buffer) if len(buffer) > 1 else buffer[0]

            # Energy gate: skip VAD entirely when signal is below noise floor
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < RMS_GATE:
                speech_prob = 0.0
            elif len(full_audio) >= self.frame_samples:
                vad_frame = full_audio[-self.frame_samples:]
                speech_prob = self.process_vad_frame(vad_frame)
            else:
                speech_prob = 0.0

            # Print VAD level every 5 frames (~160ms) — fast enough to feel live
            if _frame_count % 5 == 0:
                bar = "█" * int(speech_prob * 20)
                if speech_started and silence_start_time:
                    waited_ms = (time.monotonic() - silence_start_time) * 1000
                    status = f"silence {waited_ms:.0f}/{self.silence_hang_ms}ms"
                elif speech_started:
                    status = "SPEAKING"
                else:
                    status = "waiting..."
                print(f"  VAD {speech_prob:.2f} rms={rms:.4f} [{bar:<20}] {status}   ", end="\r")

            now = time.monotonic()

            if speech_prob >= vad_threshold:
                if not speech_started:
                    # First speech frame — start collecting clean speech audio
                    speech_buffer = [chunk]
                else:
                    speech_buffer.append(chunk)
                speech_started = True
                silence_start_time = None
            elif speech_started:
                # Still accumulate trailing audio so Whisper gets word endings
                speech_buffer.append(chunk)
                if silence_start_time is None:
                    silence_start_time = now
                    print(
                        f"\nSPEECH -> POSSIBLE SILENCE\n"
                        f"  rms={rms:.4f}\n"
                        f"  vad_prob={speech_prob:.4f}\n"
                        f"  timestamp={now:.3f}"
                    )

            # If speech is ongoing, periodically re-run transcription for partial guesses
            if speech_started and speech_buffer:
                elapsed_since_partial = (now - last_partial_time) * 1000.0
                if elapsed_since_partial >= self.partial_interval_ms:
                    last_partial_time = now
                    # Cap partial transcription to last 8s of speech for speed (no quadratic slowdown)
                    partial_window = self.sample_rate * 8
                    speech_audio = np.concatenate(speech_buffer)[-partial_window:]
                    segments, _ = await asyncio.to_thread(
                        self.model.transcribe,
                        speech_audio,
                        beam_size=1,
                        word_timestamps=False,
                        language=self.language,
                    )
                    partial_text = " ".join([s.text.strip() for s in segments if s.text]).strip()
                    if partial_text:
                        log_writer.log_event(
                            actor="user",
                            event_type="asr_partial",
                            payload={"text": partial_text},
                            turn_id=turn_id,
                        )
                        # Print partial to terminal so user sees live transcription
                        print(f"\n  ✏️  partial: '{partial_text}'")

            # Check if trailing silence exceeds silence_hang_ms threshold
            if speech_started and silence_start_time is not None:
                silence_duration_ms = (now - silence_start_time) * 1000.0
                if silence_duration_ms >= self.silence_hang_ms:
                    speech_duration_ms = (sum(len(c) for c in speech_buffer) / self.sample_rate) * 1000.0
                    print(
                        f"\nENDPOINT:\n"
                        f"  silence_duration_ms={silence_duration_ms:.0f}\n"
                        f"  silence_hang_ms={self.silence_hang_ms}\n"
                        f"  last_vad_probability={speech_prob:.4f}\n"
                        f"  last_rms={rms:.4f}\n"
                        f"  speech_duration_ms={speech_duration_ms:.0f}"
                    )

                    # User is judged to have stopped speaking
                    log_writer.log_event(
                        actor="user",
                        event_type="user_end",
                        payload={},
                        turn_id=turn_id,
                    )

                    # Final pass with word timestamps — use full clean speech audio, async to thread
                    speech_audio = np.concatenate(speech_buffer)
                    _save_debug_wav(speech_audio, self.sample_rate, turn_id)
                    segments, _ = await asyncio.to_thread(
                        self.model.transcribe,
                        speech_audio,
                        beam_size=5,
                        word_timestamps=True,
                        language=self.language,
                    )
                    final_text = " ".join([s.text.strip() for s in segments if s.text]).strip()

                    log_writer.log_event(
                        actor="user",
                        event_type="asr_final",
                        payload={"text": final_text},
                        turn_id=turn_id,
                    )
                    return final_text

        # If audio stream ended before VAD silence trigger, process final transcript if audio was gathered
        if speech_buffer:
            full_audio = np.concatenate(speech_buffer)
            speech_duration_ms = (len(full_audio) / self.sample_rate) * 1000.0
            print(
                f"\nENDPOINT (stream end):\n"
                f"  silence_duration_ms=0\n"
                f"  silence_hang_ms={self.silence_hang_ms}\n"
                f"  last_vad_probability=0.0000\n"
                f"  last_rms=0.0000\n"
                f"  speech_duration_ms={speech_duration_ms:.0f}"
            )
            _save_debug_wav(full_audio, self.sample_rate, turn_id)
            log_writer.log_event(
                actor="user",
                event_type="user_end",
                payload={},
                turn_id=turn_id,
            )
            segments, _ = await asyncio.to_thread(
                self.model.transcribe,
                full_audio,
                beam_size=5,
                word_timestamps=True,
                language=self.language,
            )
            final_text = " ".join([s.text.strip() for s in segments if s.text]).strip()
            log_writer.log_event(
                actor="user",
                event_type="asr_final",
                payload={"text": final_text},
                turn_id=turn_id,
            )
            return final_text

        return ""


# ---------------------------------------------------------------------------
# Functions for pipeline/spine.py integration
# ---------------------------------------------------------------------------
def audio_in() -> str:
    """
    Stage 1 interface for spine.py stub/drop-in compatibility.
    Returns the initial utterance or raw prompt text.
    """
    return "what time does the pool close"


def transcribe(
    utterance: str,
    log_writer: LogWriter,
    turn_id: int,
    audio_data: Optional[np.ndarray] = None,
) -> str:
    """
    Stage 2 interface for spine.py compatibility.
    If audio_data array is provided, transcribes with faster-whisper and logs events;
    otherwise emits partial + final transcription events for the input utterance.
    """
    if audio_data is not None:
        cfg = load_config()
        model = get_whisper_model(cfg)
        lang = cfg["stt"].get("language", "en")
        segments, _ = model.transcribe(audio_data, word_timestamps=True, language=lang)
        final_text = " ".join([s.text.strip() for s in segments if s.text]).strip()
        log_writer.log_event("user", "asr_partial", {"text": final_text}, turn_id)
        log_writer.log_event("user", "asr_final", {"text": final_text}, turn_id)
        log_writer.log_event("user", "user_end", {}, turn_id)
        return final_text
    else:
        partial_text = utterance.rsplit(" ", 2)[0] if " " in utterance else utterance
        log_writer.log_event("user", "asr_partial", {"text": partial_text}, turn_id)
        log_writer.log_event("user", "asr_final", {"text": utterance}, turn_id)
        log_writer.log_event("user", "user_end", {}, turn_id)
        return utterance


if __name__ == "__main__":
    print("Testing audio/stt.py initialization...")
    cfg = load_config()
    print("STT Config loaded:", cfg["stt"])
    whisper_m = get_whisper_model(cfg)
    vad_m = get_vad_model()
    print("Successfully initialized Whisper and Silero VAD models.")
