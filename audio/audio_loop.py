"""
audio_loop.py — Standalone audio I/O loop & event logger integration.

Ties audio/stt.py and audio/tts.py together with eventlog/log_writer.py.

Two modes:
  Synthetic (default) — headless test with a generated sine-wave "speech"
                        signal. No microphone or speakers needed.
  Live    (--live)    — reads from the real microphone, speaks the reply
                        aloud via sounddevice, and logs every event.

Usage:
    python audio/audio_loop.py           # synthetic / headless test
    python audio/audio_loop.py --live    # real mic + speaker
    python -m audio.audio_loop --live
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.config import load_config
from audio.stt import StreamingSTT
from audio.tts import StreamingTTS
from eventlog.log_writer import LogWriter

SAMPLE_RATE = 16000   # 16 kHz — required by Silero VAD and faster-whisper
FRAME_MS    = 30      # VAD frame duration in milliseconds
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 480 samples


# ---------------------------------------------------------------------------
# Placeholder LLM (Stage 3-5)
# ---------------------------------------------------------------------------
def placeholder_llm_stage(user_transcript: str) -> str:
    """
    Canned response function — stands in for Stages 3-5 of spine.py.

    TODO: Week 2 Pipeline Integration
    Replace this with a call to the real Controller -> Retrieval -> LLM chain.
    """
    if not user_transcript.strip():
        return "I am sorry, I did not catch that. Could you please repeat?"
    return (
        f"You asked: '{user_transcript}'. "
        "The pool is open daily from 6:00 AM to 10:00 PM."
    )


# ---------------------------------------------------------------------------
# Synthetic audio stream (headless / CI testing)
# ---------------------------------------------------------------------------
async def generate_synthetic_speech_stream() -> AsyncGenerator[np.ndarray, None]:
    """
    Yields a synthetic 16 kHz float32 audio stream that looks like:
    short silence -> brief 300 Hz tone (simulated speech) -> trailing silence.
    Used for automated headless testing — no microphone required.
    """
    # 0.5 s initial silence (15 frames × 30 ms)
    for _ in range(15):
        yield np.zeros(FRAME_SAMPLES, dtype=np.float32)
        await asyncio.sleep(FRAME_MS / 1000.0)

    # 1.5 s simulated speech — 300 Hz tone above VAD threshold
    t = np.linspace(0, 1.5, int(SAMPLE_RATE * 1.5))
    speech = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    for i in range(0, len(speech), FRAME_SAMPLES):
        chunk = speech[i : i + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        yield chunk
        await asyncio.sleep(FRAME_MS / 1000.0)

    # 1.0 s trailing silence to trigger VAD user_end (35 frames)
    for _ in range(35):
        yield np.zeros(FRAME_SAMPLES, dtype=np.float32)
        await asyncio.sleep(FRAME_MS / 1000.0)


# ---------------------------------------------------------------------------
# Live microphone stream
# ---------------------------------------------------------------------------
async def generate_mic_stream(mic_gain: float = 2.0) -> AsyncGenerator[np.ndarray, None]:
    """
    Reads audio from the default microphone using sounddevice.
    Yields float32 mono 16 kHz frames of 512 samples (Silero VAD minimum).
    Applies mic_gain boost from config.yaml to compensate for quiet Mac microphone levels.
    """
    import sounddevice as sd

    MIC_BLOCK = 512   # Silero VAD minimum (32ms @ 16kHz)
    MIC_GAIN  = mic_gain

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _callback(indata, frames, time_info, status):
        frame = indata[:, 0].copy() * MIC_GAIN
        loop.call_soon_threadsafe(q.put_nowait, frame)

    print(f"🎙  Listening... (speak now, Ctrl+C to stop)")
    print(f"   Device : {sd.query_devices(kind='input')['name']}")
    print(f"   Gain   : {MIC_GAIN}x (from config) | Block: {MIC_BLOCK} samples (32ms)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=MIC_BLOCK,
        callback=_callback,
    ):
        try:
            while True:
                frame = await q.get()
                yield frame
        except asyncio.CancelledError:
            pass



# ---------------------------------------------------------------------------
# Speaker playback (live mode)
# ---------------------------------------------------------------------------
def play_pcm_audio(pcm_bytes: bytes, sample_rate: int = 22050, block_ms: int = 50):
    """
    Plays raw 16-bit signed mono PCM through the default speaker using sounddevice.
    Plays in block_ms-sized chunks so an external stop() can interrupt quickly.
    """
    import sounddevice as sd

    if not pcm_bytes:
        return
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    block_samples = int(sample_rate * block_ms / 1000)
    for start in range(0, len(audio), block_samples):
        chunk = audio[start : start + block_samples]
        sd.play(chunk, samplerate=sample_rate)
        sd.wait()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def run_audio_loop(live: bool = False, conversation_id: str = None) -> Path:
    """
    One full conversation turn: listen -> transcribe -> placeholder LLM -> speak.
    live=False  uses the synthetic stream (headless / CI).
    live=True   uses the real microphone and plays audio through the speaker.
    """
    if conversation_id is None:
        conversation_id = f"conv_audio_{'live' if live else 'synth'}_{int(time.time())}"

    cfg        = load_config()
    log_writer = LogWriter(conversation_id=conversation_id)
    turn_id    = 1

    stt = StreamingSTT(cfg)
    tts = StreamingTTS(cfg)

    mode = "LIVE mic" if live else "synthetic stream"
    print(f"\n--- Audio Loop ({conversation_id}) | mode: {mode} ---")

    try:
        # ---- Stage 1+2: listen & transcribe --------------------------------
        mic_gain = cfg["stt"].get("mic_gain", 2.0)
        audio_stream = generate_mic_stream(mic_gain=mic_gain) if live else generate_synthetic_speech_stream()
        user_transcript = await stt.transcribe_stream(
            audio_stream, log_writer, turn_id=turn_id, vad_threshold=0.3
        )

        if not user_transcript.strip():
            user_transcript = "what time does the pool close"
            print(f"  [fallback] no speech detected, using: '{user_transcript}'")
        else:
            print(f"  📝 Transcript : '{user_transcript}'")

        # ---- Stage 3-5 placeholder -----------------------------------------
        system_response = placeholder_llm_stage(user_transcript)
        print(f"  💬 Response   : '{system_response}'")

        # ---- Stage 6: speak ------------------------------------------------
        if live:
            # Synthesise each clause and play it immediately
            from audio.tts import split_into_clauses
            clauses = split_into_clauses(system_response)
            first_audio_logged = False
            for clause in clauses:
                pcm = await asyncio.to_thread(
                    tts.synthesize_clause_to_raw_pcm, clause
                )
                if not first_audio_logged:
                    log_writer.log_event("system", "tts_first_audio", {}, turn_id)
                    first_audio_logged = True
                    print("  🔊 Speaking...")
                play_pcm_audio(pcm, sample_rate=22050, block_ms=cfg["tts"]["playback_block_ms"])
        else:
            # Headless: just log the event and simulate the timing
            await tts.speak(system_response, log_writer, turn_id=turn_id)

    finally:
        log_writer.close()

    print(f"\n  ✅ Log written : {log_writer._path}")
    return log_writer._path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone audio loop — speak and get a spoken reply."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real microphone input and speaker output (default: synthetic test mode).",
    )
    args = parser.parse_args()

    try:
        log_path = asyncio.run(run_audio_loop(live=args.live))
        print("\nLog contents:")
        print(log_path.read_text())
    except KeyboardInterrupt:
        print("\nStopped by user.")

