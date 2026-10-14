"""
streaming.py — streaming LLM -> TTS for the pipeline's model + speak stages.

v1 (batch TTS) waits for the whole answer (llm_final) before synthesizing
anything, so TTFA pays for the full generation. Here the LLM token stream is
consumed as it arrives, cut into clauses with audio/tts.py's
ClauseAccumulator (the same boundaries split_into_clauses() uses), and the
first complete clause goes to TTS while the model is still generating.

Events (eventlog/schema.md):
    llm_first_token / llm_final   logged by llm/client.py, unchanged
    llm_first_clause              first complete clause available
    tts_clause_ready              each clause synthesized (synth_ms, audio_ms)
    tts_first_audio               first clause starts playing (TTFA anchor),
                                  payload {"tts_mode": "streaming"}

Playback is either real (live=True, sounddevice), simulated in wall-clock
time (simulate_playback=True: sleep through each clause's audio duration on
a separate playback task, so synthesis of clause i+1 overlaps playback of
clause i), or skipped (simulate_playback=False, measurement runs: first
audio is logged at the moment playback would start, and later clauses are
only synthesized). TTFA is identical across the three because it is
anchored on the first clause's playback start.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.tts import ClauseAccumulator  # noqa: E402

_END = object()


async def generate_and_speak_streaming(
    generate_fn: Callable[..., str],
    tts,
    log_writer,
    turn_id: int,
    live: bool = False,
    simulate_playback: bool = True,
    playback_block_ms: int = 20,
) -> str:
    """
    Run one answer through streaming LLM -> TTS.

    generate_fn(on_delta) must run the (blocking) LLM call, invoke
    on_delta(text) for every streamed content delta, log llm_first_token /
    llm_final itself (llm/client.py does), and return the full answer text.
    It runs in a worker thread; deltas are handed to the event loop
    thread-safely and in order.

    tts must provide synthesize_clause_to_raw_pcm(clause) -> bytes (16-bit
    mono PCM) and a sample_rate attribute (audio/tts.py's StreamingTTS).

    Returns the full answer text.
    """
    loop = asyncio.get_running_loop()
    deltas: asyncio.Queue = asyncio.Queue()

    def on_delta(text: str) -> None:
        loop.call_soon_threadsafe(deltas.put_nowait, text)

    gen_future = asyncio.ensure_future(asyncio.to_thread(generate_fn, on_delta))
    # The thread's deltas were all scheduled (call_soon_threadsafe) before the
    # to_thread future resolves, so this sentinel always lands after them.
    gen_future.add_done_callback(lambda _f: deltas.put_nowait(_END))

    sample_rate = getattr(tts, "sample_rate", 22050)
    playback_q: asyncio.Queue = asyncio.Queue()
    first_audio_logged = False

    async def playback_worker():
        nonlocal first_audio_logged
        while True:
            item = await playback_q.get()
            if item is _END:
                return
            idx, pcm = item
            if not first_audio_logged:
                log_writer.log_event(
                    "system", "tts_first_audio", {"tts_mode": "streaming"}, turn_id
                )
                first_audio_logged = True
            if live:
                from audio.audio_loop import play_pcm_audio

                await asyncio.to_thread(
                    play_pcm_audio, pcm, sample_rate=sample_rate, block_ms=playback_block_ms
                )
            elif simulate_playback:
                await asyncio.sleep(len(pcm) / 2 / sample_rate)

    player = asyncio.create_task(playback_worker())

    acc = ClauseAccumulator()
    clause_index = 0

    async def handle_clause(clause: str):
        nonlocal clause_index
        idx = clause_index
        clause_index += 1
        if idx == 0:
            log_writer.log_event(
                "system",
                "llm_first_clause",
                {"text": clause, "n_chars": len(clause), "clause_index": 0},
                turn_id,
            )
        t0 = time.perf_counter()
        pcm = await asyncio.to_thread(tts.synthesize_clause_to_raw_pcm, clause)
        synth_ms = (time.perf_counter() - t0) * 1000.0
        audio_ms = len(pcm) / 2 / sample_rate * 1000.0 if pcm else 0.0
        log_writer.log_event(
            "system",
            "tts_clause_ready",
            {
                "clause_index": idx,
                "text": clause,
                "synth_ms": round(synth_ms, 1),
                "audio_ms": round(audio_ms, 1),
            },
            turn_id,
        )
        if pcm:
            await playback_q.put((idx, pcm))

    try:
        while True:
            item = await deltas.get()
            if item is _END:
                break
            for clause in acc.feed(item):
                await handle_clause(clause)
        answer = gen_future.result()  # re-raises an LLM error here
        for clause in acc.flush():
            await handle_clause(clause)
    finally:
        await playback_q.put(_END)
        await player
    return answer
