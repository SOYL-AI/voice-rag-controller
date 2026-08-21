# audio/ — Audio In & Out

**Owner: Adithi**  
This README covers Week 1. Refinements (better barge-in, tuning) come later.

## What this folder is

The system's ears and mouth. Two jobs: turn the user's speech into text as they talk 
(speech-to-text), and turn the system's answer back into spoken audio (text-to-speech). 
Plus the ability to stop speaking instantly if the user interrupts.

## Why it matters

This is the front door of the whole system. It is also the highest-risk component in Week 1 
because audio is where the trickiest timing bugs live. Build it carefully and in isolation 
first, and flag early if you get stuck — do not struggle silently.

## Week 1 goal (your deliverable)

1. Real streaming speech-to-text that produces partial guesses every 300–500 ms.
2. Real local text-to-speech that speaks a given sentence.
3. Both working together in a small standalone script (not yet wired into the main 
   pipeline).
4. Every audio event written to the log in the frozen schema format.

**Priority order:** get transcription and speech-out solid first. The interrupt feature 
(barge-in) is important but can slip to Week 2 if time is short — do not let it block the 
core.

## Background: the key ideas

- **Streaming** means producing text *as the person talks*, not after they finish.
- **Partial guesses** are live transcriptions that get revised as more audio arrives (e.g. 
  "I need a rum" becomes "I need a room"). Emitting these every 300–500 ms is the goal.
- **Barge-in** is when the user interrupts while the system is talking. The system must stop 
  speaking within about 100 ms.

## How to start (step by step)

### Step 1 — Speech-to-text in isolation (`stt.py`)

Use `faster-whisper` (a fast, local Whisper implementation). Write a small script that 
takes microphone or audio-file input and prints transcriptions, emitting partial guesses as 
audio streams in. Confirm it revises its guesses as more speech arrives.

### Step 2 — Text-to-speech in isolation (`tts.py`)

Use a local text-to-speech engine such as `piper`. Write a script that takes a sentence and 
speaks it aloud. Confirm it sounds clear.

### Step 3 — Put them together (`audio_loop.py`)

A small standalone loop: you speak, it transcribes (with partials), and it speaks a fixed 
canned reply back. This proves both directions work. Do not connect it to the main pipeline 
yet — that swap happens once the spine is ready.

### Step 4 — Log every audio event

Using the shared log writer (`eventlog/`), record audio events in the frozen schema:
- `asr_partial` — each live guess (with `text`)
- `asr_final` — the finalized transcription (with `text`)
- `user_end` — when the user is judged to have stopped
- `tts_first_audio` — the moment the first sound of the reply goes out (this is the timing 
  anchor the harness needs)
- `barge_in` — if/when the user interrupts (once that feature exists)

### Step 5 (if time allows) — Barge-in

Add the ability to detect the user speaking while the system is talking, and stop the 
speech output within ~100 ms. If Week 1 is tight, note where you stopped and continue in 
Week 2.

## Not in Week 1

- Wiring into the main pipeline (the integration lead does the swap once the spine exists).
- Echo cancellation tuning, prosody tuning — later.

## Definition of done (Week 1)

Real speech in and real speech out, working in a standalone script, with all audio events 
logged in the frozen schema. Ready to replace the pipeline's audio stubs.

## Coordinate with

The integration lead, who will plug this into the spine once your standalone version works.

---

## Phase 2 — Audio Integration and Polish

In Phase 1 the audio worked standalone (a script where you speak and it responds). In Phase 
2 it goes into the real pipeline, and the two known rough edges get fixed.

### Goal

Get the real audio running inside the integrated pipeline (not just standalone), and fix 
the transcription cutoff and the text-to-speech quality, plus finish barge-in.

### How to do it

- Work with Renya to plug stt.py and tts.py into the spine, so audio flows through the full 
  pipeline (speech in, all the way through retrieval and the model, then speech out). The 
  standalone loop stays for testing.
- Fix the transcription cutoff: in Phase 1 it sometimes ended the turn too early (for 
  example "I am in room 427..." became "I am in room"). This is the endpointing deciding the 
  user stopped too soon. Give it a longer silence window before it decides the turn is over, 
  so full sentences get captured in the final transcript.
- Fix the text-to-speech audio quality: the Phase 1 output sounded rough ("bad network"). 
  This is usually a sample-rate mismatch between what Piper outputs and what is played back. 
  Make sure the playback sample rate matches the TTS output rate.
- Finish the barge-in path: when the user speaks while the system is talking, stop the 
  speech output within about 100 ms and start listening. The event is already logged 
  (barge_in), so this is about making the stop reliable and fast.
  *Note:* Requires headphones or AEC hardware; an open mic+speaker setup causes self-triggered barge-in from acoustic echo.

### Definition of done

Real audio runs inside the integrated pipeline, transcription captures full sentences, 
playback sounds clean, and barge-in stops speech quickly when the user interrupts.

### Coordinate with

Renya (plugging audio into the spine), and keep logging every event through the shared 
eventlog log_writer as before.

---

## Experiment 3 (E3) — Adithi's baselines: B3, plus B0/B1

**Owner: Adithi.** The turn-taking/endpointing baseline, plus the two simplest retrieval
baselines. Each implements `controller/baseline.py`'s `BaselineController` interface (copy
`controller/baselines/b0_never_retrieve.py` as the template). Full contract in
[`../E3_BASELINE_INTERFACE.md`](../E3_BASELINE_INTERFACE.md).

- **B3 (tuned silence-timeout endpointing):** your main deliverable — a controller whose
  timing decision is purely a silence-timeout on the endpointing signal, tuned (swept, not
  guessed) on the dev split like every other baseline.
- **B0 (never-retrieve):** the reference baseline — already built at
  `controller/baselines/b0_never_retrieve.py`, nothing to do here except read it as the
  worked example.
- **B1 (always-retrieve-once):** retrieve exactly once per turn, unconditionally.

Same fairness rule as every baseline: same tuning budget, thresholds swept, no strawmen.
