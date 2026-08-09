"""
spine.py — the end-to-end pipeline skeleton (Week 1 -> Phase 2).

Runs one conversation through every stage in order:

    audio-in -> transcribe -> controller -> retrieve -> model -> speak

As of Phase 2, every stage is real. Audio-in/transcribe use Adithi's
StreamingSTT (audio/stt.py: Silero VAD + faster-whisper) over either the
live microphone or a synthetic speech-shaped stream. The controller calls
the real decide() from controller/decide.py, which reads the state
tracker's uncertainty to make a real retrieve-or-not call and actually
gates whether the retrieve stage runs. Retrieve embeds the query and
searches the FAISS index (retrieval/search.py). Model calls the real LLM
client (llm/client.py). Speak uses Adithi's StreamingTTS (audio/tts.py:
piper), either playing audio through the speaker (--live) or synthesizing
with simulated playback timing so the pipeline still runs with no
microphone or speaker attached (the default).

The point, per pipeline/README.md, isn't intelligence. It's that a
conversation flows all the way through, every step gets logged in the
frozen format, and each stage is a clearly separated function so a
teammate's real component can drop in and replace one stub at a time
without touching this file's overall shape.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path


# controller/ and eventlog/ are sibling top-level folders, not subpackages
# of pipeline/. Running this file directly (`python3 pipeline/spine.py`)
# only puts pipeline/'s own directory on sys.path, so without this, the
# imports below fail with ModuleNotFoundError. Adding the repo root once,
# here, means both `python3 pipeline/spine.py` and `python3 -m
# pipeline.spine` work the same way.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retrieval.rewrite import rewrite_query
from controller.decide import decide
from eventlog.log_writer import LogWriter
from llm.client import generate_answer as llm_generate_answer
from llm.tracker import track as tracker_track
from retrieval.search import search as retrieval_search

from audio.config import load_config
from audio.stt import StreamingSTT
from audio.tts import StreamingTTS, split_into_clauses
from audio.audio_loop import (
    generate_mic_stream,
    generate_synthetic_speech_stream,
    play_pcm_audio,
)


def _sleep_ms(ms: float) -> None:
    """
    Sleep for roughly `ms` milliseconds.

    Only the controller stage still uses this (see controller_decide()) —
    it simulates the moment it takes to assemble a belief and decide,
    since that stage has no real I/O of its own to time it against. Every
    other stage now waits on real work (STT, retrieval, the LLM, TTS)
    instead of a cosmetic sleep.
    """
    time.sleep(ms / 1000)


# ---------------------------------------------------------------------------
# Stage 1 — audio-in
# ---------------------------------------------------------------------------
def audio_in(cfg: dict, live: bool):
    """
    Builds the async audio-frame stream that stage 2 consumes: the live
    microphone (audio/audio_loop.py's generate_mic_stream) when --live is
    passed, otherwise a synthetic speech-shaped stream so a plain
    `python3 pipeline/spine.py` still runs with no microphone attached and
    no person talking — the same synthetic/live split audio/audio_loop.py
    uses for its own headless vs. live modes.

    Unlike the Week 1 stub, this stage hands back a stream, not a finished
    utterance: the real STT component does its own listening *and*
    transcribing in one pass (see transcribe() below), so there's no
    intermediate "the user said this" string to produce here.
    """
    if live:
        mic_gain = cfg["stt"].get("mic_gain", 2.0)
        return generate_mic_stream(mic_gain=mic_gain)
    return generate_synthetic_speech_stream()


# ---------------------------------------------------------------------------
# Stage 2 — transcribe
# ---------------------------------------------------------------------------
async def transcribe(
    audio_stream, stt: StreamingSTT, log_writer: LogWriter, turn_id: int
) -> str:
    """
    Real version: audio/stt.py's StreamingSTT, running Silero VAD frame by
    frame over the stream from audio_in() and faster-whisper on top of it.

    This fuses what used to be two separate stub stages (listen, then
    transcribe) into one call: transcribe_stream() logs asr_partial as
    guesses arrive, user_end once trailing silence crosses the configured
    threshold, and asr_final with the finished transcript itself, then
    returns that transcript. No manual logging needed here, unlike the
    Week 1 stub.
    """
    return await stt.transcribe_stream(
        audio_stream, log_writer, turn_id=turn_id, vad_threshold=0.3
    )


# ---------------------------------------------------------------------------
# Stage 3 — controller
# ---------------------------------------------------------------------------
def controller_decide(transcript: str, log_writer: LogWriter, turn_id: int) -> dict:
    """
    Not a stub. Runs the state tracker to get the current slot-fill state,
    then calls the real decide() from controller/decide.py, which reads
    that state to make a real retrieve-or-not call and logs its own
    controller_action event internally.

    No log_writer is passed to the tracker call — see llm/tracker.py's
    module docstring on why tracker output isn't logged yet (no
    state_update event type in the frozen schema).
    """
    # "belief" is the situation snapshot decide() reasons over. Phase 2's
    # rule reads tracker_slots; other keys are along for the ride for
    # later phases (timing, model confidence, ...).
    tracker_slots = tracker_track([transcript])
    belief = {"transcript": transcript, "tracker_slots": tracker_slots}
    _sleep_ms(15)  # simulate the moment it takes to assemble a belief and decide
    return decide(belief, log_writer=log_writer, turn_id=turn_id)


# ---------------------------------------------------------------------------
# Stage 4 — retrieve
# ---------------------------------------------------------------------------
def retrieve(query: str, log_writer: LogWriter, turn_id: int) -> list:
    """
    Embeds the query and searches the FAISS index built from
    retrieval/corpus/ (retrieval/search.py), returning the top-k passages.

    Note on "scores": retrieval/search.py returns FAISS L2 distances, not a
    similarity score — lower is a better match, the opposite of the Week 1
    stub's hardcoded 0.91/0.78. Logged as-is; anything reading this field
    downstream needs to know it's a distance now.
    """
    log_writer.log_event("system", "retrieval_launched", {"query": query}, turn_id)

    results = retrieval_search(query, k=3)
    passages = [r["text"] for r in results]
    scores = [r["distance"] for r in results]

    log_writer.log_event(
        "system",
        "retrieval_result",
        {"passages": passages, "scores": scores},
        turn_id,
    )
    return passages


# ---------------------------------------------------------------------------
# Stage 5 — model
# ---------------------------------------------------------------------------
def generate_answer(
    passages: list, log_writer: LogWriter, turn_id: int, question: str
) -> str:
    """
    Calls the real llm/client.py generate_answer(): builds the RAG prompt
    from the question + retrieved passages, and runs it through whichever
    backend configs/config.yaml's llm.backend selects (mock or vllm). That
    function logs llm_first_token and llm_final itself, in the exact
    schema shape the Week 1 stub used to log manually, so no logging call
    is needed here.
    """
    return llm_generate_answer(passages, log_writer, turn_id, question=question)


# ---------------------------------------------------------------------------
# Stage 6 — speak
# ---------------------------------------------------------------------------
async def speak(
    answer: str,
    tts: StreamingTTS,
    log_writer: LogWriter,
    turn_id: int,
    live: bool,
    playback_block_ms: int,
) -> None:
    """
    Real version: audio/tts.py's StreamingTTS, synthesizing with piper.

    Two paths, mirroring audio/audio_loop.py's own live/synthetic split:
      live=True  synthesizes each clause and actually plays it through the
                 speaker (sounddevice), logging tts_first_audio — the TTFA
                 anchor the harness uses — the moment the first clause is
                 ready to play.
      live=False awaits StreamingTTS.speak(), which does real synthesis
                 and logs tts_first_audio itself but only simulates
                 playback timing — no speaker needed, so plain `python3
                 pipeline/spine.py` runs without hardware.
    """
    if live:
        clauses = split_into_clauses(answer)
        first_audio_logged = False
        for clause in clauses:
            pcm = await asyncio.to_thread(tts.synthesize_clause_to_raw_pcm, clause)
            if not first_audio_logged:
                log_writer.log_event("system", "tts_first_audio", {}, turn_id)
                first_audio_logged = True
            play_pcm_audio(pcm, sample_rate=22050, block_ms=playback_block_ms)
    else:
        await tts.speak(answer, log_writer, turn_id)

    print(f'  system speaks: "{answer}"')


# ---------------------------------------------------------------------------
# The spine itself
# ---------------------------------------------------------------------------
async def run_conversation(
    conversation_id: str, live: bool = False, question: str = None
) -> Path:
    """
    Runs one full conversation through every stage above, in order,
    logging every event to logs/<conversation_id>.jsonl per
    eventlog/schema.md. Returns the path to that log file.

    live=False (default) listens on a synthetic speech-shaped stream and
    speaks with simulated playback timing — no microphone or speaker
    needed, so this stays the deterministic, hardware-free path `python3
    pipeline/spine.py` has always run. live=True uses the real microphone
    and speaker, per audio/audio_loop.py's --live mode.

    question, when given, bypasses audio-in/transcribe entirely and feeds
    that string straight in as the user's transcript instead — for the
    batch runner and experiments, which need to drive the pipeline with
    real scenario questions rather than always replaying whatever the
    synthetic/live audio stage happens to produce. Since transcribe()
    normally logs asr_partial/user_end/asr_final itself (see audio/stt.py),
    this path logs the user_end/asr_final pair itself so the log stays
    schema-correct — asr_partial is skipped since there's no partial guess
    to report when the text is already final. speak() still runs, but
    always with simulated (non-live) playback, since there's no captured
    audio turn to match hardware timing against.
    """
    cfg = load_config()
    log_writer = LogWriter(conversation_id=conversation_id)
    turn_id = 1  # Week 1's canned conversation is a single turn.

    # Built once, not per turn: StreamingSTT/StreamingTTS lazily load
    # faster-whisper, Silero VAD, and piper on first use, which is too
    # slow to redo for every turn.
    try:
        tts = StreamingTTS(cfg)

        if question is None:
            stt = StreamingSTT(cfg)
            audio_stream = audio_in(cfg, live)

            transcript = await transcribe(audio_stream, stt, log_writer, turn_id)

            if not transcript.strip():
                # Same fallback audio/audio_loop.py uses: the synthetic
                # stream (or a quiet mic) can yield no speech at all.
                transcript = "what time does the pool close"
                print(f'  [fallback] no speech detected, using: "{transcript}"')
            else:
                print(f'  user says:      "{transcript}"')
        else:
            # Text path: no audio, no STT — the given question *is* the
            # transcript. Log the events transcribe_stream() would have
            # logged, in the same order/actor audio/stt.py uses, so a
            # reader of the log can't tell the two paths apart.
            transcript = question
            log_writer.log_event("user", "user_end", {}, turn_id)
            log_writer.log_event("user", "asr_final", {"text": transcript}, turn_id)
            print(f'  user says:      "{transcript}"')

        action = controller_decide(transcript, log_writer, turn_id)
        print(f"  controller:     {action}")

        # The decision now actually gates retrieval, not just logs it: only
        # fetch passages when decide() says "retrieve". Any other action
        # (e.g. skip_retrieval) means the model answers with no passages.
        if action.get("action") == "retrieve":
            history = []

            rewritten_query = rewrite_query(
                history,
                transcript,
            )

            print(f"  rewritten query: {rewritten_query}")

            passages = retrieve(
                rewritten_query,
                log_writer,
                turn_id,
            )

            print(f"  retrieved:      {len(passages)} passage(s)")
        else:
            passages = []
            print("  retrieved:      skipped (controller decided not to retrieve)")

        answer = generate_answer(passages, log_writer, turn_id, transcript)
        # Text mode always speaks non-live (simulated playback, no
        # hardware) regardless of `live`, matching the headless default —
        # there's no captured audio turn here for live playback to pair
        # against.
        speak_live = live if question is None else False
        await speak(
            answer,
            tts,
            log_writer,
            turn_id,
            speak_live,
            cfg["tts"]["playback_block_ms"],
        )
    finally:
        # Always close, even if a stage above raises — otherwise whatever
        # is still sitting in the writer thread's queue never reaches
        # disk, and we'd lose the tail of the conversation.
        log_writer.close()

    return log_writer._path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one conversation through the full pipeline spine."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real microphone and speaker (default: synthetic/headless).",
    )
    args = parser.parse_args()

    print("Running one conversation through the spine...\n")
    log_path = asyncio.run(run_conversation("conv_001", live=args.live))
    print(f"\nWrote log to: {log_path}")
