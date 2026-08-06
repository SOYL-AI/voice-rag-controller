# pipeline/ — The Spine (end-to-end integration)

**Owner: integration lead**  
This README covers Week 1.

## What this folder is

The spine that connects every stage into one working loop: audio in → transcribe → retrieve 
→ model → speak → audio out. In Week 1 every stage is a stub that returns a hardcoded value 
and logs that it ran. The point is not intelligence; it is that a conversation flows all the 
way through and every step is logged.

## Why it matters

This is the centre everyone else plugs into. Until the spine exists, nobody can integrate 
their real component. It is also the reference example of the log format that every other 
component copies. The classic project failure is polishing components separately and 
integrating at the end; the spine exists to prevent that by making integration continuous.

## Week 1 goal (your deliverable)

1. An end-to-end pipeline where each stage is a stub, connected in order.
2. A canned conversation runs start to finish: fake audio in, hardcoded answer spoken out.
3. Every stage writes to the log in the frozen schema.
4. As real components become ready, their stubs are swapped out and the log still reads 
   correctly.

## How to start (step by step)

### Step 1 — Freeze the schema first

Before anything, make sure `eventlog/schema.md` is frozen and the shared `log_writer` helper 
exists. Nothing in the pipeline should be built before the schema is set, since every stage 
logs against it.

### Step 2 — Build each stage as a stub

Create a simple stage for each step of the loop. Each stub does three things: log that it 
ran (in the frozen schema), pass a hardcoded value to the next stage, and return. Stages:
- audio-in (start with a canned audio file or a fixed transcript)
- transcribe → logs `asr_final` with fixed text
- controller → logs `controller_action` with `noop`
- retrieve → logs `retrieval_launched` / `retrieval_result` with fixed passages
- model → logs `llm_final` with a hardcoded answer
- speak → logs `tts_first_audio` and "plays" a fixed sentence

### Step 3 — Connect them into one loop

Wire the stages in order so one call runs the whole chain and produces a complete log for 
one conversation. Confirm the log can be read by the harness's timeline viewer.

### Step 4 — Swap stubs for real components

As each teammate's real component becomes ready (audio, retrieval, model), replace the 
matching stub with the real thing and confirm the pipeline still runs and the log still 
parses. This is the integration work that runs through the rest of Week 1 and beyond.

## Definition of done (Week 1)

One full conversation runs end to end from a clean checkout of the repo, every stage logged, 
and the log replays correctly on the harness timeline viewer.

## Coordinate with

Everyone — this is where all components come together. Audio, retrieval, and model each get 
swapped into the spine here as they become ready.

---

## Phase 2 — Integration

Phase 1 built every component in isolation. Phase 2 connects them into one running 
pipeline. This is the critical path: every experiment waits on the pipeline being 
integrated, so this comes first.

### Goal

Swap the spine's stub stages for the real components teammates built, so one real 
conversation runs end to end and produces a correct log. Then begin the controller's first 
real decision logic.

### How to do it

- Replace the retrieval stub with Lakshya's search function (retrieval/search.py), the 
  model stub with Ryan's client (llm/client.py), and the audio stubs with Adithi's stt and 
  tts (audio/stt.py, audio/tts.py). Swap them in one at a time, confirming the log still 
  reads correctly on the harness timeline after each swap.
- Handle the path and import details so the pieces run together. Each was built standalone, 
  so small wiring fixes are expected (for example, making sure retrieval finds its index 
  from wherever the spine runs).
- Once integrated, give the controller its first simple rule (see controller/README.md 
  Phase 2), replacing the no-op stub, keeping the same decide(belief) interface.

### Definition of done

One real conversation runs through the fully integrated pipeline (real audio, real 
retrieval, real model, real speech) and produces a correct log. This unblocks all 
experiments.

### Coordinate with

Everyone. This is where all components come together. Work closely with Adithi on the audio 
swap, Ryan on the model swap, and Lakshya on the retrieval swap.
