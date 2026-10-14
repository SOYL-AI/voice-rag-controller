"""
log_writer.py — shared logging helper for the Voice-RAG Controller event log.

Every component (audio, retrieval, llm, controller, ...) writes conversation
events through this module. The exact fields and event types below mirror
eventlog/schema.md (FROZEN v1) — read that file first if you haven't.

Import it as:

    from eventlog.log_writer import LogWriter
"""

import json
import queue
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# The shared clock
# ---------------------------------------------------------------------------
# Schema rule: "Use one monotonic clock for t_ms everywhere." A monotonic
# clock only ever moves forward — unlike the system clock, it can't jump
# backward from an NTP sync or a manual clock change. We capture a single
# reference point (_START_NS) the moment this module is first imported.
# Every event's t_ms afterward is that event's monotonic time minus this
# reference, converted to milliseconds — i.e. "ms since the logger started."
#
# Because _START_NS is a module-level constant, every LogWriter in this
# process (one per conversation, possibly created from different threads —
# audio, retrieval, llm) measures time from the same zero point, so their
# t_ms values land on one shared, comparable timeline. This guarantee only
# holds within a single Python process; if the pipeline ever splits across
# processes, each process would need its own agreed time source.
_START_NS = time.monotonic_ns()


def now_ms() -> int:
    """Milliseconds elapsed since this module was first imported."""
    return (time.monotonic_ns() - _START_NS) // 1_000_000


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
# Mirrors the "Event types (Week 1 set)" table in eventlog/schema.md. This
# set exists purely so log_event() can catch a typo'd event_type early.
# eventlog/schema.md is still the source of truth — add new event types
# there first, then here.
ALLOWED_EVENT_TYPES = {
    "asr_partial",
    "asr_final",
    "user_end",
    "retrieval_launched",
    "retrieval_result",
    "controller_action",
    "llm_first_token",
    "llm_final",
    "tts_first_audio",
    "barge_in",
    # research v2 additions (see eventlog/schema.md)
    "llm_first_clause",
    "tts_clause_ready",
}

ALLOWED_ACTORS = {"user", "system"}

# Sentinel put on the queue to tell the writer thread to drain and stop.
# A dedicated object (not None or a string) so it can never collide with a
# real event.
_SHUTDOWN = object()


class LogWriter:
    """
    Appends one conversation's events to a single .jsonl file.

    Create one LogWriter per conversation_id. Call log_event(...) from any
    thread, as often as you like — it hands the event to a background
    queue and returns immediately, so it never blocks the caller on disk
    I/O. Call close() once the conversation ends so buffered events get
    flushed and the writer thread shuts down cleanly.
    """

    def __init__(self, conversation_id: str, log_dir: str = "logs"):
        self.conversation_id = conversation_id

        # One file per conversation, named after its id:
        # e.g. logs/conv_001.jsonl
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        self._path = log_dir_path / f"{conversation_id}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")

        # Unbounded queue: log_event() just drops an event here and
        # returns. The background thread below is the only thing that
        # ever touches self._file, so writes are naturally serialized —
        # multiple caller threads can log_event() at once without racing
        # on the file.
        self._queue: queue.Queue = queue.Queue()

        # Why a background thread: the schema explicitly forbids writing
        # to the log directly from the audio thread, because even a few
        # milliseconds of disk I/O can show up as an audible stutter.
        # Making it a daemon thread means it won't stop the process from
        # exiting if someone forgets to call close() — but you should
        # still always call close(), or any events still sitting in the
        # queue at that moment are lost.
        self._thread = threading.Thread(
            target=self._run, name=f"LogWriter-{conversation_id}", daemon=True
        )
        self._thread.start()

    # -- public API -----------------------------------------------------

    def log_event(self, actor: str, event_type: str, payload: dict, turn_id):
        """
        Queue one event for writing. Returns immediately.

        Writes exactly the fields eventlog/schema.md defines, in its order:
        t_ms, conversation_id, turn_id, actor, event_type, payload.
        """
        if event_type not in ALLOWED_EVENT_TYPES:
            # We still write the event — dropping data is worse than
            # writing something unrecognized — but we warn loudly so a
            # typo gets caught during development instead of silently
            # producing an event_type nothing downstream understands.
            print(
                f"[log_writer] WARNING: unknown event_type '{event_type}' "
                f"— not in eventlog/schema.md. Writing it anyway."
            )
        if actor not in ALLOWED_ACTORS:
            print(
                f"[log_writer] WARNING: unknown actor '{actor}' "
                f"— schema expects 'user' or 'system'. Writing it anyway."
            )

        event = {
            "t_ms": now_ms(),
            "conversation_id": self.conversation_id,
            "turn_id": turn_id,
            "actor": actor,
            "event_type": event_type,
            "payload": payload if payload is not None else {},
        }
        # queue.Queue.put() is thread-safe and effectively instant — it
        # just appends to an internal deque and wakes the writer thread.
        self._queue.put(event)

    def close(self, timeout: float = 5.0):
        """
        Tell the writer thread to flush whatever's left in the queue and
        stop, then wait for it. Always call this when a conversation ends.
        """
        self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=timeout)
        if not self._file.closed:
            self._file.close()

    # -- background thread ----------------------------------------------

    def _run(self):
        """
        The writer thread's main loop. queue.get() blocks (no busy-waiting,
        no wasted CPU) until an event or the shutdown sentinel arrives.
        """
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._drain_remaining()
                return
            self._write_line(item)

    def _drain_remaining(self):
        """
        On shutdown, write anything still sitting in the queue before the
        thread exits, so close() never silently drops buffered events.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _SHUTDOWN:
                self._write_line(item)

    def _write_line(self, event: dict):
        self._file.write(json.dumps(event) + "\n")
        # Flushed on every line rather than buffered: a conversation
        # produces at most a handful of events per second, so the extra
        # syscall is cheap, and it means a crash mid-conversation loses at
        # most the one event currently mid-flight, not a whole buffer.
        self._file.flush()


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    writer = LogWriter(conversation_id="conv_demo", log_dir="logs")

    writer.log_event("user", "asr_partial", {"text": "what time does the"}, turn_id=1)
    writer.log_event(
        "user", "asr_final", {"text": "what time does the pool close"}, turn_id=1
    )
    writer.log_event("user", "user_end", {}, turn_id=1)
    writer.log_event("system", "controller_action", {"action": "retrieve"}, turn_id=1)
    writer.log_event(
        "system",
        "retrieval_result",
        {"passages": ["Pool hours: 6am-10pm"], "scores": [0.91]},
        turn_id=1,
    )
    writer.log_event(
        "system", "llm_final", {"text": "The pool closes at 10 PM."}, turn_id=1
    )
    writer.log_event("system", "tts_first_audio", {}, turn_id=1)

    # Deliberate typo, to demonstrate the warning path without losing the event.
    writer.log_event("system", "tts_first_audioo", {}, turn_id=1)

    writer.close()

    print(f"\nWrote to {writer._path}\n")
    print(writer._path.read_text())
