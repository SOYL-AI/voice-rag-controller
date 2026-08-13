# harness/ — Metrics & Replay Tools

**Owner: Shubhankit**  
This README covers Week 1. More metrics and analysis tools will be added here as we go.

## What this folder is

The scoreboard and the debugging viewer. Two things: a tool that reads an event log and 
computes metrics (starting with the main timing metric), and a viewer that turns a raw log 
into a readable timeline of what happened and when.

## Why it matters

The replay viewer becomes the whole team's debugging tool for the entire month. Every time 
someone asks "why did this conversation go wrong?", the answer comes from your timeline 
viewer. This folder builds against the frozen log schema (`eventlog/schema.md`), not the 
live pipeline, so you are fully unblockable from Day 1 — you do not have to wait for anyone.

## Week 1 goal (your deliverable)

1. A log-reader that computes TTFA (Time To First Audio) from a log file.
2. A replay-to-timeline viewer that prints a readable sequence of events from a log.
3. Basic statistics helpers (percentiles, a confidence-interval function) for later use.

## Background you need first

Read `eventlog/schema.md`. That is the exact format of the log files you will read. Every 
line is one JSON object with fields `t_ms`, `conversation_id`, `turn_id`, `actor`, 
`event_type`, and `payload`. Your tools parse these lines.

**TTFA (Time To First Audio)** is the key metric: the gap in milliseconds between the user 
finishing speaking and the first sound of the reply going out. In log terms, it is the 
`t_ms` of the `tts_first_audio` event minus the `t_ms` of the `user_end` event, within the 
same turn.

## How to start (step by step)

### Step 1 — Make a fake log to test against

You do not need the real pipeline. Hand-write a small `.jsonl` file with a few events 
(a `user_end`, then a `tts_first_audio` a few hundred ms later, etc.) following the schema. 
This is your test input.

### Step 2 — Log reader + TTFA (`metrics.py`)

Write a script that:
1. Opens a `.jsonl` log file and reads it line by line, parsing each as JSON.
2. Groups events by `turn_id`.
3. For each turn, finds the `user_end` and `tts_first_audio` events and computes TTFA.
4. Prints a summary (per-turn TTFA, and the average / p95 across turns).

Run it on your fake log and confirm the numbers are right.

### Step 3 — Timeline viewer (`replay.py`)

Write a tool that reads a log and prints a clean, ordered, human-readable timeline, e.g.:

[  0 ms] user      asr_partial     "what time does the"
[420 ms] user      asr_final       "what time does the pool close"
[430 ms] user      user_end
[440 ms] system    controller_action  retrieve
[560 ms] system    retrieval_result   (3 passages)
[900 ms] system    tts_first_audio     -> TTFA = 470 ms

This is the tool everyone will use to understand what happened in a conversation. Make it 
clear and pleasant to read.

### Step 4 — Statistics helpers (`stats.py`)

Add simple helpers for later experiments: a percentile function (for p95) and a bootstrap 
confidence-interval function (repeatedly resample a list of numbers to estimate a range). 
You will not use these heavily in Week 1, but having them ready helps later.

## Not in Week 1 (do not start these yet)

- Running actual experiments (that is Weeks 3–4).
- Comparing policies over logs (re-simulation) — later.

## Definition of done (Week 1)

Feed your tools a sample log and they (a) compute TTFA correctly and (b) draw a clean, 
readable timeline. The statistics helpers exist and are tested on toy inputs.

## Coordinate with

Nobody directly for Week 1 — you build against the schema, which is frozen. Just make sure 
you are reading the latest `eventlog/schema.md` if it ever updates.

---

## Phase 2 — Full Metrics, Batch Runner, and Experiment 2

In Phase 1 the harness read a log and computed the key metric (time to first audio). In 
Phase 2 it grows into the full experiment scoreboard, and you run Experiment 2.

### Goal

Extend the harness from a single metric into the full measurement system, build the batch 
runner that runs many conversations unattended, and run Experiment 2 (timing value) with 
Renya.

### How to do it

**1. Full metrics**
- Add the remaining metrics beyond time to first audio: answer accuracy (did it get the 
  right answer, graded against the simulator's answer key), dead-air (silence gaps), and 
  interruptions.
- Add the reporting we need across many conversations: p95 (the slowest 5 percent) and 
  confidence intervals using the bootstrap helper you already built.

**2. Batch runner**
- Build a runner that takes the simulator's conversations (from Lakshya), plays them through 
  the integrated pipeline one after another unattended, and collects all the logs. This is 
  what lets us run 50 or more conversations overnight without sitting there.

**3. Experiment 2 (timing value), with Renya**
- Compare retrieval timing policies: retrieve immediately when a need is detected, versus at 
  sentence boundaries, versus during predicted user pauses (while the user is still talking).
- Keep everything else fixed and change only the timing. Measure which policy gives lower 
  p95 time-to-first-audio at equal answer accuracy.

### Definition of done

The full metrics scoreboard works over real logs, the batch runner can run many 
conversations unattended, and Experiment 2 produces results with confidence intervals.

### Coordinate with

Lakshya (his simulator feeds your batch runner), Renya (co-runs Experiment 2 and owns the 
timing policies in the controller), and Yashmitha (compiles your results into tables).

---

## Experiment 3 (E3) — Shubhankit's frontier analysis, plus B7 if there's room

**Owner: Shubhankit.** The E3 sweep (all baselines × price grid × scenarios × seeds) writes
logs through the shared `controller_action` event, additively carrying `baseline` and, for
B9/Ours, `lambda_l`/`lambda_c` — see [`../E3_BASELINE_INTERFACE.md`](../E3_BASELINE_INTERFACE.md)
for the full contract every baseline (including yours, if you build B7) implements against
`controller/baseline.py`'s `BaselineController`.

- **Frontier analysis (main deliverable):** build the accuracy-vs-p95-TTFA and
  accuracy-vs-compute frontiers, per `(baseline, price point)`, with bootstrap confidence
  bands (reuse `stats.py`'s bootstrap helper). This is what turns per-run metrics into the
  headline comparison — B0-B7 as single points, B9/Ours as swept price-point frontiers. Read
  the dominance verdict off where Ours's frontier sits relative to B9's.
- **B7 (MemGPT-style memory), if you have room:** a baseline whose memory_op decisions follow
  a MemGPT-style write/summarize/evict policy, everything else held simple. Implements the
  same `BaselineController` interface (copy
  `controller/baselines/b0_never_retrieve.py` as the template) — build it only after the
  frontier analysis is solid, since that's the piece every other baseline's result depends on.

Same fairness rule as every baseline: same tuning budget, thresholds swept, no strawmen.
