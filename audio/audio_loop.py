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
    return "The pool is open daily from 6:00 AM to 10:00 PM."


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
def play_pcm_audio(pcm_bytes: bytes, sample_rate: int, block_ms: int = 50):
    """
    Plays raw 16-bit signed mono PCM through the default speaker using sounddevice.
    Uses a single persistent OutputStream opened once per utterance to eliminate block clicking.
    """
    import sounddevice as sd

    if not pcm_bytes:
        return
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    block_samples = int(sample_rate * block_ms / 1000)
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        for start in range(0, len(audio), block_samples):
            chunk = audio[start : start + block_samples]
            stream.write(chunk)


# ---------------------------------------------------------------------------
# Concurrent VAD Barge-in Listener
# ---------------------------------------------------------------------------
async def run_barge_in_listener(
    stt: StreamingSTT,
    tts: StreamingTTS,
    log_writer: LogWriter,
    turn_id: int,
    vad_threshold: float = 0.5,
    consecutive_frames: int = 4,
    mic_gain: float = 2.0,
):
    """
    Lightweight VAD-only mic listener running concurrently with TTS playback.
    Opens a short-lived sounddevice InputStream while TTS is active.
    Triggers tts.stop(log_writer, turn_id) upon detecting `consecutive_frames` speech-positive frames.
    """
    import sounddevice as sd

    print(f"\nBARGE-IN:\n  listener_active=true")

    MIC_BLOCK = 512  # Silero VAD minimum (32ms @ 16kHz)
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _callback(indata, frames, time_info, status):
        frame = indata[:, 0].copy() * mic_gain
        loop.call_soon_threadsafe(q.put_nowait, frame)

    playback_started = False
    consecutive_count = 0
    frames_processed = 0
    RMS_GATE = 0.012

    try:
        with sd.InputStream(
            samplerate=stt.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=MIC_BLOCK,
            callback=_callback,
        ):
            while not tts._stop_requested.is_set():
                # Active only while actual TTS PCM audio playback is occurring
                if not tts.playback_active:
                    await asyncio.sleep(0.01)
                    if tts._is_playing and not tts.playback_active:
                        # Piper is synthesizing, wait until PCM playback starts
                        continue
                    if not tts._is_playing and not tts.playback_active:
                        print(f"\nBARGE-IN:\n  listener_stopping=tts_finished")
                        break

                if not playback_started:
                    drained_count = 0
                    while not q.empty():
                        try:
                            q.get_nowait()
                            drained_count += 1
                        except asyncio.QueueEmpty:
                            break
                    print(f"\nBARGE-IN:\n  drained_stale_frames={drained_count}")
                    playback_started = True

                frame = await q.get()
                frames_processed += 1
                rms = float(np.sqrt(np.mean(frame ** 2)))
                if rms < RMS_GATE:
                    speech_prob = 0.0
                else:
                    speech_prob = stt.process_vad_frame(frame)

                if frames_processed % 8 == 0:
                    print(
                        f"\nBARGE-IN HEARTBEAT:\n"
                        f"  timestamp={time.time():.3f}\n"
                        f"  frames_processed={frames_processed}\n"
                        f"  latest_rms={rms:.4f}\n"
                        f"  latest_speech_prob={speech_prob:.4f}"
                    )

                if speech_prob >= vad_threshold:
                    consecutive_count += 1
                    if consecutive_count >= consecutive_frames:
                        print(
                            f"\nBARGE-IN:\n"
                            f"  user_speech_detected=true\n"
                            f"  consecutive={consecutive_count}\n"
                            f"  calling=tts.stop()"
                        )
                        tts.stop(log_writer, turn_id)
                        break
                else:
                    consecutive_count = 0
    except asyncio.CancelledError:
        pass
    except Exception as e:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def run_audio_loop(
    live: bool = False,
    conversation_id: str = None,
    test_barge_in: bool = False,
) -> Path:
    """
    Continuous multi-turn conversation loop:
    while not quit: listen -> transcribe -> LLM -> speak -> listen again.

    live=False  uses the synthetic stream (headless / CI). Runs 2 turns for verification.
    live=True   uses the real microphone and speaker in a continuous loop until Ctrl+C or 'quit'.
    """
    if conversation_id is None:
        conversation_id = f"conv_audio_{'live' if live else 'synth'}_{int(time.time())}"

    cfg        = load_config()
    log_writer = LogWriter(conversation_id=conversation_id)

    # Initialize models ONCE for the entire session
    stt = StreamingSTT(cfg)
    tts = StreamingTTS(cfg)

    mode = "LIVE mic" if live else "synthetic stream"
    print(f"\n--- Audio Loop ({conversation_id}) | mode: {mode} ---")
    print(
        f"  ⚙️  Config loaded: silence_hang_ms={stt.silence_hang_ms}ms, "
        f"TTS sample_rate={tts.sample_rate}Hz, playback_block_ms={tts.playback_block_ms}ms"
    )

    turn_id = 1
    mic_gain = cfg["stt"].get("mic_gain", 2.0)
    barge_in_cfg = cfg.get("tts", {}).get("barge_in", {})
    vad_thresh = barge_in_cfg.get("vad_threshold", 0.5)
    consec_frames = barge_in_cfg.get("consecutive_frames", 4)

    try:
        while True:
            if live:
                print(f"\n🎙  [Turn {turn_id}] Listening... (speak now, Ctrl+C to stop)")
            else:
                print(f"\n--- Turn {turn_id} (synthetic mode) ---")

            # ---- Stage 1+2: listen & transcribe ----------------------------
            audio_stream = (
                generate_mic_stream(mic_gain=mic_gain)
                if live
                else generate_synthetic_speech_stream()
            )
            user_transcript = await stt.transcribe_stream(
                audio_stream, log_writer, turn_id=turn_id, vad_threshold=0.3
            )

            if not user_transcript.strip():
                if not live:
                    user_transcript = "what time does the pool close"
                    print(f"  [fallback] no speech detected, using: '{user_transcript}'")
                else:
                    print("  (no speech detected, listening again...)")
                    turn_id += 1
                    continue

            # Safe exit command check
            if user_transcript.strip().lower() in ("quit", "quit.", "exit", "exit."):
                print("\n  👋 'quit' command detected. Exiting audio loop...")
                break

            print(f"  📝 Transcript : '{user_transcript}'")

            # ---- Stage 3-5 placeholder -------------------------------------
            system_response = placeholder_llm_stage(user_transcript)
            print(f"  💬 Response   : '{system_response}'")

            # ---- Stage 6: speak --------------------------------------------
            barge_in_task = None
            if live:
                barge_in_task = asyncio.create_task(
                    run_barge_in_listener(
                        stt=stt,
                        tts=tts,
                        log_writer=log_writer,
                        turn_id=turn_id,
                        vad_threshold=vad_thresh,
                        consecutive_frames=consec_frames,
                        mic_gain=mic_gain,
                    )
                )

            test_task = None
            if test_barge_in:
                async def _test_trigger():
                    await asyncio.sleep(0.15)
                    print("\n  🛑 [Test Barge-In] Triggering tts.stop() via manual test path...")
                    tts.stop(log_writer, turn_id)

                test_task = asyncio.create_task(_test_trigger())

            try:
                print("  🔊 Speaking...")
                await tts.speak(system_response, log_writer, turn_id=turn_id, live=live)
            finally:
                if barge_in_task is not None:
                    barge_in_task.cancel()
                    try:
                        await barge_in_task
                    except asyncio.CancelledError:
                        pass
                if test_task is not None:
                    test_task.cancel()
                    try:
                        await test_task
                    except asyncio.CancelledError:
                        pass

            turn_id += 1

            # In synthetic mode, run 2 turns to verify multi-turn capability before completing
            if not live and turn_id > 2:
                break

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n  👋 Audio loop interrupted by user.")
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
    parser.add_argument(
        "--test-barge-in",
        action="store_true",
        help="Simulate a manual barge-in trigger after 150ms to verify stop latency and logging.",
    )
    args = parser.parse_args()

    try:
        log_path = asyncio.run(
            run_audio_loop(live=args.live, test_barge_in=args.test_barge_in)
        )
        print("\nLog contents:")
        print(log_path.read_text())
    except KeyboardInterrupt:
        print("\nStopped by user.")


