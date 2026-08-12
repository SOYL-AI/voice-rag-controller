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
from typing import Optional


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
from controller.timing import (
    MechanismContext,
    PartialSignalWatcher,
    mechanism_launch_fields,
    mechanism_result_fields,
    plan_simulated_speech,
    resolve_policy,
)
from eventlog.log_writer import LogWriter, now_ms
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


def controller_decide_forced(force_retrieve: bool, log_writer: LogWriter, turn_id: int) -> dict:
    """
    Experiment 2's fixed retrieve/skip mask: bypasses controller_decide()
    (and therefore decide()'s tracker-uncertainty rule) entirely and returns
    a decision sourced from the caller instead — the scenario's gold
    retrieval_needed label, per run_conversation()'s force_retrieve
    parameter. The point is to hold the retrieve/skip decision fixed across
    all three timing policies so Experiment 2 isolates WHEN retrieval
    happens from WHETHER it happens; letting decide()'s live rule keep
    picking would let retrieve/skip vary turn to turn for reasons unrelated
    to timing, confounding that comparison.

    Still logs a controller_action event, in the exact shape decide() would
    (same log_writer.log_event() call, same payload keys) — a log reader
    can't tell a forced decision from a live one except by the "e2_mask: "
    prefix on `reason`.
    """
    action = (
        {"action": "retrieve", "reason": "e2_mask: forced retrieve"}
        if force_retrieve
        else {"action": "skip_retrieval", "reason": "e2_mask: forced skip"}
    )
    log_writer.log_event(
        actor="system",
        event_type="controller_action",
        payload=action,
        turn_id=turn_id,
    )
    return action


# ---------------------------------------------------------------------------
# Stage 4 — retrieve
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    log_writer: LogWriter,
    turn_id: int,
    mechanism: MechanismContext = None,
    rewrite_precomputed: bool = False,
) -> list:
    """
    Embeds the query and searches the FAISS index built from
    retrieval/corpus/ (retrieval/search.py), returning the top-k passages.

    Note on "scores": retrieval/search.py returns FAISS L2 distances, not a
    similarity score — lower is a better match, the opposite of the Week 1
    stub's hardcoded 0.91/0.78. Logged as-is; anything reading this field
    downstream needs to know it's a distance now.

    `mechanism` (controller/timing.py's MechanismContext), when given, adds
    Experiment 2's hidden-latency mechanism-check fields to the
    retrieval_launched/retrieval_result payloads — which timing policy and
    trigger were active, and (computed from real captured timestamps, never
    guessed) how this retrieval's real launch/finish times related to the
    real or modeled speech window. See controller/timing.py's docstring for
    why this makes the timing-policy effect measurable rather than asserted.

    `rewrite_precomputed` (Experiment 2's frozen-query mechanism — see
    run_retrieval_pipeline()'s `frozen_query`), when True, just marks the
    retrieval_launched payload so a log reader can tell `query` came from a
    rewrite done ahead of time, outside the timed window, rather than a live
    rewrite_query() call that happened right before this launch.
    """
    launch_ms = now_ms()
    launched_payload = {"query": query, "rewrite_precomputed": rewrite_precomputed}
    if mechanism is not None:
        launched_payload.update(mechanism_launch_fields(mechanism, launch_ms))
    log_writer.log_event("system", "retrieval_launched", launched_payload, turn_id)

    results = retrieval_search(query, k=3)
    finish_ms = now_ms()
    passages = [r["text"] for r in results]
    scores = [r["distance"] for r in results]
    # filenames: the source document each passage came from
    # (retrieval/search.py already returns this per hit — see its
    # "filename" key). Logged alongside passages/scores so hit@k is
    # computable from the log without re-running retrieval; harness/
    # metrics.py's hit_at_k_per_turn() reads it back out.
    filenames = [r["filename"] for r in results]

    result_payload = {"passages": passages, "scores": scores, "filenames": filenames}
    if mechanism is not None:
        result_payload.update(mechanism_result_fields(mechanism, launch_ms, finish_ms))
    log_writer.log_event("system", "retrieval_result", result_payload, turn_id)
    return passages


def run_retrieval_pipeline(
    transcript_snapshot: str,
    log_writer: LogWriter,
    turn_id: int,
    mechanism: MechanismContext,
    force_retrieve: Optional[bool] = None,
    frozen_query: Optional[str] = None,
) -> tuple:
    """
    The controller-decide + query-rewrite + retrieve sequence — unchanged
    from Phase 2's logic (controller/decide.py's retrieve-or-not rule never
    varies with the timing policy). Only *when* run_conversation() calls
    this changes per policy: at user_end for sentence_end, or earlier (on a
    partial transcript, real or modeled) for immediate/fixed_early_proxy. See
    run_conversation()'s timing-policy branches below.

    force_retrieve (Experiment 2's mask, threaded down from
    run_conversation()): None runs the live decide() rule exactly as
    before. True/False bypasses it via controller_decide_forced() instead,
    fixing WHETHER retrieval happens so only WHEN it happens (this
    function's caller, per timing policy) varies.

    frozen_query (Experiment 2's frozen-rewrite mechanism, threaded down
    from run_conversation()): None runs rewrite_query() live, right here,
    inside the timed window — unchanged behavior. On the real Modal model
    that call takes ~1.5s, which E2's early timing policies (immediate/
    fixed_early_proxy) would otherwise pay before retrieve() ever launches,
    eating into (or entirely consuming) the retrieval-overlap-with-speech
    window E2 is trying to measure — a rewrite-latency confound, not a
    placement effect. When frozen_query is a string instead, the live
    rewrite_query() call is skipped entirely and frozen_query is used
    as-is as the retrieval query, so retrieval can launch immediately
    without waiting on the rewrite. The caller (run_conversation(), and
    ultimately simulator/batch_runner.py's precompute helper) is
    responsible for having produced that string ahead of time, outside the
    timed window. This composes with force_retrieve and the timing policy
    as three independent knobs: frozen_query sets WHAT query, force_retrieve
    sets WHETHER retrieval happens, and the policy sets WHEN.
    """
    if force_retrieve is None:
        action = controller_decide(transcript_snapshot, log_writer, turn_id)
    else:
        action = controller_decide_forced(force_retrieve, log_writer, turn_id)
    print(f"  controller:     {action}  (timing_trigger={mechanism.trigger})")
    if action.get("action") != "retrieve":
        print("  retrieved:      skipped (controller decided not to retrieve)")
        return action, []

    if frozen_query is None:
        rewritten_query = rewrite_query([], transcript_snapshot)
        print(f"  rewritten query: {rewritten_query}")
    else:
        rewritten_query = frozen_query
        print(f"  frozen query:   {rewritten_query}  (rewrite precomputed, live rewrite skipped)")

    passages = retrieve(
        rewritten_query,
        log_writer,
        turn_id,
        mechanism=mechanism,
        rewrite_precomputed=frozen_query is not None,
    )
    print(f"  retrieved:      {len(passages)} passage(s)  (timing_trigger={mechanism.trigger})")
    return action, passages


async def _trigger_live_early_retrieval(
    policy: str,
    watcher: PartialSignalWatcher,
    turn_start_ms: int,
    log_writer: LogWriter,
    turn_id: int,
    force_retrieve: Optional[bool] = None,
    frozen_query: Optional[str] = None,
) -> tuple:
    """
    Live/synthetic-audio path only. Waits for the real signal `policy`
    needs from the real STT (first asr_partial for immediate; a stable
    repeated partial as fixed_early_proxy's proxy trigger — see
    controller/timing.py's PartialSignalWatcher), then launches
    run_retrieval_pipeline() as a background task that races the rest of
    transcribe_stream() for real.

    Never returns on its own if the signal it's waiting for doesn't arrive
    before the turn ends (e.g. a very short utterance with no partial
    before user_end) — run_conversation() races this coroutine against the
    transcribe task itself and cancels this one if transcribe wins.

    force_retrieve: forwarded to run_retrieval_pipeline() unchanged — see
    that function's docstring. This function only decides WHEN the pipeline
    launches; it never inspects force_retrieve itself.

    frozen_query: forwarded to run_retrieval_pipeline() unchanged — see that
    function's docstring. Same non-involvement as force_retrieve: this
    function only decides WHEN, never WHAT query gets used.

    Returns (task, mechanism). `task` is None only when fixed_early_proxy's
    fixed-point assumption gets invalidated (the user kept talking past the
    point a pause was assumed) before the early retrieval finished — the
    task is cancelled and run_conversation() falls back to a normal
    post-user_end retrieval on the real final transcript instead. See this
    function's fixed_early_proxy branch for the caveat on what "cancelled"
    actually stops.
    """
    # query_ready gate (Experiment 2 fix): wait for the earliest partial
    # that actually has enough content to retrieve on, not literally the
    # first partial (which on real STT is often one word) — see
    # controller/timing.py's is_query_ready() and PartialSignalWatcher's
    # first_ready_partial / repeated_ready_partial. Unlike the text path
    # below, this partial IS the retrieval query: on real/synthetic audio
    # there's no future knowledge of the final transcript to fall back to.
    if policy == "immediate":
        partial_text = await watcher.first_ready_partial
        trigger = "immediate"
    else:
        partial_text = await watcher.repeated_ready_partial
        trigger = "fixed_early_proxy"

    mechanism = MechanismContext(
        policy=policy,
        trigger=trigger,
        user_end_ms_getter=lambda: watcher.user_end_ms,
        speech_window_ms_getter=lambda: (
            (turn_start_ms, watcher.user_end_ms) if watcher.user_end_ms is not None else None
        ),
    )
    task = asyncio.create_task(
        asyncio.to_thread(
            run_retrieval_pipeline,
            partial_text,
            log_writer,
            turn_id,
            mechanism,
            force_retrieve,
            frozen_query,
        )
    )

    if policy == "fixed_early_proxy":
        # Structural cancel-on-invalidation: if a later partial shows the
        # user kept talking past the fixed point this policy assumed a
        # pause at, the early retrieval was launched on a stale/incomplete
        # query — cancel it rather than use its result. Where a real
        # forecast signal belongs long-term: a genuine mid-utterance
        # silence gap from audio/stt.py's VAD state, distinct from its
        # final user_end endpoint — that needs a small addition to
        # Adithi's file, out of scope here, so wait_for_growth()'s
        # repeated-partial-outgrown proxy stands in for it.
        growth_future = watcher.wait_for_growth(partial_text)
        done, _pending = await asyncio.wait(
            {task, growth_future}, return_when=asyncio.FIRST_COMPLETED
        )
        if growth_future in done and task not in done:
            # Caveat: retrieve() runs inside asyncio.to_thread. If the
            # worker thread has already started, cancelling the wrapping
            # Task stops us from waiting on/using its result — it does not
            # guarantee the underlying thread stops immediately. Either
            # way, retrieval_launched/retrieval_result already logged (or
            # about to be) for this attempt stay in the record with
            # forecast_invalidated=True (field name kept for log-schema
            # stability — see controller/timing.py's
            # mechanism_result_fields()); they just aren't this turn's
            # passages.
            task.cancel()
            mechanism.forecast_invalidated = True
            print(
                "  [fixed_early_proxy] proxy assumption invalidated — speech "
                "continued past the fixed proxy point; discarding the early "
                "retrieval."
            )
            return None, mechanism
        if not growth_future.done():
            growth_future.cancel()

    return task, mechanism


# ---------------------------------------------------------------------------
# Stage 5 — model
# ---------------------------------------------------------------------------
def generate_answer(
    passages: list,
    log_writer: LogWriter,
    turn_id: int,
    question: str,
    seed: Optional[int] = None,
) -> str:
    """
    Calls the real llm/client.py generate_answer(): builds the RAG prompt
    from the question + retrieved passages, and runs it through whichever
    backend configs/config.yaml's llm.backend selects (mock or vllm). That
    function logs llm_first_token and llm_final itself, in the exact
    schema shape the Week 1 stub used to log manually, so no logging call
    is needed here.

    seed, when not None, is forwarded straight through to llm/client.py's
    generate_answer() (Experiment 2's seed sweep — see run_conversation's
    docstring). Default None: unchanged generation behavior.
    """
    return llm_generate_answer(
        passages, log_writer, turn_id, question=question, seed=seed
    )


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
    conversation_id: str,
    live: bool = False,
    question: str = None,
    force_retrieve: Optional[bool] = None,
    seed: Optional[int] = None,
    frozen_query: Optional[str] = None,
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
    this path logs that same trio itself so the log stays schema-correct
    and a reader can't tell the two paths apart.

    Experiment 2 (retrieval timing): WHEN the retrieve-or-not decision and
    the retrieval it can trigger actually run is controlled by
    controller.timing.policy (see controller/timing.py), not hardcoded
    here. sentence_end (baseline) runs it after the transcript is final,
    same as always. immediate/fixed_early_proxy run it earlier — on a real
    partial transcript from the live STT in the live/synthetic-audio path,
    or, in the text path (no real audio to derive one from), on a modeled
    partial produced partway through a real asyncio.sleep() sized to how
    long saying `question` would actually take. Either way the earlier
    launch is a real background task racing real remaining work, not a
    number subtracted from a metric afterward — see controller/timing.py's
    module docstring. Both early paths also apply Experiment 2's
    query_ready gate (controller.timing.query_ready): immediate/
    fixed_early_proxy dispatch at the earliest point (real partial, or
    simulated on the text path) with enough content to retrieve on
    meaningfully, not the first word — see controller/timing.py's
    is_query_ready() / SpeechPlan.dispatch_at_s().

    Experiment 2 (retrieval masking): force_retrieve, when not None,
    overrides WHETHER retrieval happens — True forces a retrieve, False
    forces a skip, bypassing controller/decide.py's live retrieve-or-not
    rule for this turn (see controller_decide_forced() above). It composes
    with every timing policy above unchanged: the mask only ever swaps out
    which decision run_retrieval_pipeline() logs and acts on, never when
    run_retrieval_pipeline() gets called, so a forced-retrieve turn still
    launches early under immediate/fixed_early_proxy and at user_end under
    sentence_end, exactly as an unmasked turn would. Default None leaves
    every other call to run_conversation() (including the batch runner's)
    on the live decision, unchanged.

    Experiment 2 (frozen rewrite): frozen_query, when not None, is
    forwarded unchanged to every run_retrieval_pipeline() call this turn
    (and, in the live/synthetic-audio path, to _trigger_live_early_retrieval()
    too) — see run_retrieval_pipeline()'s docstring for why: rewrite_query()
    takes ~1.5s on the real Modal model, which would otherwise eat into (or
    consume outright) the retrieval-overlap-with-speech window the timing
    policies above are trying to create, confounding rewrite latency with
    retrieval placement. frozen_query lets a caller (simulator/
    batch_runner.py's precompute helper) run the rewrite once, up front,
    outside the timed window, and hand back the resulting string so
    retrieval can launch on it immediately. It composes with force_retrieve
    and the timing policy as independent knobs: frozen_query is WHAT query,
    force_retrieve is WHETHER retrieval happens, policy is WHEN. Default
    None leaves every other call to run_conversation() on the live
    rewrite_query() call, unchanged.

    Experiment 2 (seed sweep): seed, when not None, is forwarded to the
    model stage's generate_answer() -> llm/client.py's generate_answer() ->
    client.generate(seed=seed) — the same seed override
    llm/experiment1/run.py already uses. It only affects model generation
    (retrieval/timing/masking above are all deterministic given the same
    mask and policy); default None leaves generation on whatever
    configs/config.yaml's llm.generation.seed says, unchanged.
    """
    cfg = load_config()
    log_writer = LogWriter(conversation_id=conversation_id)
    turn_id = 1  # Week 1's canned conversation is a single turn.
    policy = resolve_policy()
    turn_start_ms = now_ms()

    # Built once, not per turn: StreamingSTT/StreamingTTS lazily load
    # faster-whisper, Silero VAD, and piper on first use, which is too
    # slow to redo for every turn.
    try:
        tts = StreamingTTS(cfg)

        if question is None:
            stt = StreamingSTT(cfg)
            audio_stream = audio_in(cfg, live)
            # Forwards every log_event() call unchanged to log_writer; only
            # additionally observes asr_partial/user_end as they really
            # happen, so the timing policies below have real signals to
            # trigger on without editing audio/stt.py. See
            # controller/timing.py's PartialSignalWatcher.
            watcher = PartialSignalWatcher(log_writer, turn_id)

            early_task = None
            mechanism = None
            if policy in ("immediate", "fixed_early_proxy"):
                transcribe_task = asyncio.create_task(
                    transcribe(audio_stream, stt, watcher, turn_id)
                )
                trigger_task = asyncio.create_task(
                    _trigger_live_early_retrieval(
                        policy, watcher, turn_start_ms, log_writer, turn_id,
                        force_retrieve, frozen_query,
                    )
                )
                done, _pending = await asyncio.wait(
                    {transcribe_task, trigger_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if trigger_task in done:
                    early_task, mechanism = trigger_task.result()
                else:
                    # transcribe finished (silence hit user_end) before the
                    # signal this policy needs ever arrived -- e.g. a very
                    # short utterance with no partial before user_end. No
                    # early retrieval this turn; fall through to the
                    # post-user_end path below like sentence_end would.
                    trigger_task.cancel()
                transcript = await transcribe_task
            else:
                transcript = await transcribe(audio_stream, stt, watcher, turn_id)

            if not transcript.strip():
                # Same fallback audio/audio_loop.py uses: the synthetic
                # stream (or a quiet mic) can yield no speech at all.
                transcript = "what time does the pool close"
                print(f'  [fallback] no speech detected, using: "{transcript}"')
            else:
                print(f'  user says:      "{transcript}"')

            if early_task is not None:
                action, passages = await early_task
            else:
                if policy == "sentence_end":
                    trigger = "sentence_end"
                elif mechanism is not None and mechanism.forecast_invalidated:
                    trigger = f"{policy}_forecast_invalidated_fallback"
                else:
                    trigger = f"{policy}_no_signal_fallback"

                late_mechanism = MechanismContext(
                    policy=policy,
                    trigger=trigger,
                    user_end_ms_getter=lambda: watcher.user_end_ms,
                    speech_window_ms_getter=lambda: (
                        (turn_start_ms, watcher.user_end_ms)
                        if watcher.user_end_ms is not None
                        else None
                    ),
                )
                action, passages = run_retrieval_pipeline(
                    transcript, log_writer, turn_id, late_mechanism, force_retrieve, frozen_query
                )
        else:
            # Text path: no audio, no STT — the given question *is* the
            # transcript. immediate/fixed_early_proxy need a real
            # elapsed-time window to launch retrieval inside, which the
            # text path doesn't otherwise have — plan_simulated_speech()
            # models how long saying `question` would take from its actual
            # word count, and the sleeps below really happen (real wall
            # clock), so the eventual race against retrieval's real
            # latency is real too. sentence_end sleeps through the exact
            # same modeled duration with no early trigger, so the *only*
            # thing that differs between policies here is when retrieval
            # launches, matching the pre-registration's "vary only WHEN
            # retrieval fires."
            transcript = question
            plan = plan_simulated_speech(transcript)

            def _planned_user_end_ms():
                return turn_start_ms + int(plan.total_speech_s * 1000)

            def _planned_window():
                return (turn_start_ms, _planned_user_end_ms())

            early_task = None
            if policy in ("immediate", "fixed_early_proxy"):
                # query_ready gate (Experiment 2 fix): don't dispatch at the
                # raw trigger point (partial_at_s / pause_at_s — reactive
                # for immediate, a fixed proxy point for fixed_early_proxy)
                # if the transcript revealed by then is still too short / all
                # function words to retrieve on meaningfully — dispatch_at_s
                # also waits for ready_at_s. See controller/timing.py's
                # SpeechPlan.dispatch_at_s() / is_query_ready().
                trigger_s = plan.dispatch_at_s(policy)
                # Simulated partial: the text path has no real STT, so this
                # is the word-prefix a real partial transcript would
                # plausibly contain by trigger_s, per plan_simulated_speech's
                # speech timeline — logged as the honest asr_partial event,
                # but NOT what's retrieved on below (see comment there).
                simulated_partial = plan.text_at(trigger_s)

                await asyncio.sleep(trigger_s)
                log_writer.log_event(
                    "user", "asr_partial", {"text": simulated_partial}, turn_id
                )

                mechanism = MechanismContext(
                    policy=policy,
                    trigger=policy,
                    user_end_ms_getter=_planned_user_end_ms,
                    speech_window_ms_getter=_planned_window,
                )
                early_task = asyncio.create_task(
                    asyncio.to_thread(
                        run_retrieval_pipeline,
                        # Same final query as sentence_end's post-user_end
                        # retrieval — only WHEN this fires differs by
                        # policy, never WHAT it retrieves on (see this
                        # module's Experiment 2 docstring paragraph below).
                        # The text path already knows the full transcript
                        # (it's modeling speech *timing* over a known
                        # scenario string, not transcribing real audio), so
                        # once the query_ready gate says the moment is
                        # meaningful, there's no reason to retrieve on
                        # `simulated_partial` instead of `transcript` —
                        # doing so would just reintroduce the
                        # degraded-query regression this gate exists to
                        # fix. (The live/synthetic-audio path above can't
                        # do this — it has no future knowledge of the
                        # final transcript — so it retrieves on the real
                        # partial it actually has.)
                        transcript,
                        log_writer,
                        turn_id,
                        mechanism,
                        force_retrieve,
                        frozen_query,
                    )
                )
                await asyncio.sleep(plan.total_speech_s - trigger_s)
            else:
                await asyncio.sleep(plan.total_speech_s)

            log_writer.log_event("user", "user_end", {}, turn_id)
            log_writer.log_event("user", "asr_final", {"text": transcript}, turn_id)
            print(f'  user says:      "{transcript}"')

            if early_task is not None:
                action, passages = await early_task
            else:
                mechanism = MechanismContext(
                    policy=policy,
                    trigger="sentence_end",
                    user_end_ms_getter=_planned_user_end_ms,
                    speech_window_ms_getter=_planned_window,
                )
                action, passages = run_retrieval_pipeline(
                    transcript, log_writer, turn_id, mechanism, force_retrieve, frozen_query
                )

        answer = generate_answer(passages, log_writer, turn_id, transcript, seed=seed)
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
