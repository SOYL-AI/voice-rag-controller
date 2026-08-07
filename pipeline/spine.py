"""
spine.py — the end-to-end pipeline skeleton (Week 1).

Runs one conversation through every stage in order:

    audio-in -> transcribe -> controller -> retrieve -> model -> speak

Every stage except the controller is a stub for Week 1: it logs the
event(s) the frozen schema (eventlog/schema.md) says it should, and hands a
hardcoded value to the next stage. The controller is not a stub — it calls
the real decide() from controller/decide.py, which currently always answers
"noop" (see that module's docstring for why that's the right answer this
week).

The point, per pipeline/README.md, isn't intelligence. It's that a
conversation flows all the way through, every step gets logged in the
frozen format, and each stage is a clearly separated function so a
teammate's real component can drop in and replace one stub at a time
without touching this file's overall shape.
"""

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

from controller.decide import decide
from eventlog.log_writer import LogWriter
from llm.client import generate_answer as llm_generate_answer
from retrieval.search import search as retrieval_search


def _sleep_ms(ms: float) -> None:
    """
    Sleep for roughly `ms` milliseconds.

    Purely cosmetic: with every stage a stub, all six events would
    otherwise land at nearly the same t_ms and the log would read like
    everything happened at once. A small pause between stages spreads the
    timestamps out so a replay through the harness's timeline viewer
    looks like a real conversation's pacing, not an instant burst.
    """
    time.sleep(ms / 1000)


# ---------------------------------------------------------------------------
# Stage 1 — audio-in
# ---------------------------------------------------------------------------
def audio_in() -> str:
    """
    Real version: a live microphone stream. For Week 1, a hardcoded
    utterance stands in for "the user just said this."
    """
    return "what time does the pool close"


# ---------------------------------------------------------------------------
# Stage 2 — transcribe
# ---------------------------------------------------------------------------
def transcribe(utterance: str, log_writer: LogWriter, turn_id: int) -> str:
    """
    Real version: faster-whisper streaming speech-to-text, emitting a
    partial guess every 300-500ms as audio arrives, then a final
    transcription once the user stops (see audio/README.md).

    The stub fakes that shape with a single partial (a leading fragment of
    the sentence, as if the tail end hasn't been heard yet), then the full
    text as the final, then user_end once the user is judged to have
    stopped talking.
    """
    partial_text = utterance.rsplit(" ", 2)[0]  # e.g. "what time does the"
    log_writer.log_event("user", "asr_partial", {"text": partial_text}, turn_id)
    _sleep_ms(280)  # simulate the rest of the utterance streaming in

    log_writer.log_event("user", "asr_final", {"text": utterance}, turn_id)
    _sleep_ms(10)  # simulate the brief silence needed to confirm the user stopped

    log_writer.log_event("user", "user_end", {}, turn_id)
    return utterance


# ---------------------------------------------------------------------------
# Stage 3 — controller
# ---------------------------------------------------------------------------
def controller_decide(transcript: str, log_writer: LogWriter, turn_id: int) -> dict:
    """
    Not a stub. Calls the real decide() from controller/decide.py, which
    logs its own controller_action event internally. The pipeline calls it
    exactly the way it always will, so swapping in the real value-vs-delay
    decision logic later requires no change here — only decide()'s body
    changes.
    """
    # "belief" is whatever situation snapshot the real controller will
    # eventually reason over. decide() ignores it in Week 1; it's passed
    # here so the call shape already matches what it will be later.
    belief = {"transcript": transcript}
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
def speak(answer: str, log_writer: LogWriter, turn_id: int) -> None:
    """
    Real version: local TTS (e.g. piper) synthesizing and streaming audio
    out (see audio/README.md). The stub prints the answer as a stand-in
    for "audio played," and logs tts_first_audio — the TTFA anchor the
    harness uses: Time To First Audio is this event's t_ms minus
    user_end's t_ms.
    """
    _sleep_ms(80)  # simulate synthesis time before the first sample plays
    log_writer.log_event("system", "tts_first_audio", {}, turn_id)
    print(f'  system speaks: "{answer}"')


# ---------------------------------------------------------------------------
# The spine itself
# ---------------------------------------------------------------------------
def run_conversation(conversation_id: str) -> Path:
    """
    Runs one full, hardcoded conversation through every stage above, in
    order, logging every event to logs/<conversation_id>.jsonl per
    eventlog/schema.md. Returns the path to that log file.
    """
    log_writer = LogWriter(conversation_id=conversation_id)
    turn_id = 1  # Week 1's canned conversation is a single turn.

    try:
        utterance = audio_in()
        print(f'  user says:      "{utterance}"')

        transcript = transcribe(utterance, log_writer, turn_id)

        action = controller_decide(transcript, log_writer, turn_id)
        print(f"  controller:     {action}")

        passages = retrieve(transcript, log_writer, turn_id)
        print(f"  retrieved:      {len(passages)} passage(s)")

        answer = generate_answer(passages, log_writer, turn_id, transcript)
        speak(answer, log_writer, turn_id)
    finally:
        # Always close, even if a stage above raises — otherwise whatever
        # is still sitting in the writer thread's queue never reaches
        # disk, and we'd lose the tail of the conversation.
        log_writer.close()

    return log_writer._path


if __name__ == "__main__":
    print("Running one canned conversation through the spine...\n")
    log_path = run_conversation("conv_001")
    print(f"\nWrote log to: {log_path}")
