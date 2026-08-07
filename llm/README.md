# llm/ — Language Model & Signals

**Owner: Ryan**  
This README covers Week 1. Full signal instrumentation and the Experiment 1 label set come 
in Week 2.

## What this folder is

Two things: serving the language model (so the rest of the system can send it a prompt and 
get an answer), and exposing the model's internal confidence numbers so we can read how 
sure it is. Plus the state tracker — a small component that, each turn, writes down what the 
user wants as a structured form.

## Why it matters

A later experiment depends entirely on being able to see the model's per-token confidence numbers (logprobs). If those 
are not exposed and logged, that experiment is impossible. This is exactly why we serve an 
open model ourselves instead of using a closed API — closed APIs do not reveal these 
numbers.

## Week 1 goal (your deliverable)

1. An open ~8B language model served locally via vLLM, answering prompts.
2. The model's confidence numbers (logprobs) visible and logged, not just the answer text.
3. A basic state tracker that outputs the user's goal as structured JSON each turn.

## Background: the key idea

Every time the model produces a word, it also produces a number for how confident it was in 
that word (the logprob). High uncertainty may signal the model has a knowledge gap and 
needs to look something up. We cannot use these signals unless they are exposed and logged, 
so making them visible is the core of this role.

## How to start (step by step)

### Step 1 — Serve the model with vLLM (`serve.py` / setup notes)

Stand up an open ~8B instruct model using vLLM on a cloud GPU (RunPod / Azure). Confirm you 
can send it a prompt and get an answer back.

**Critical:** turn on logprobs in the vLLM request so the response includes the per-token 
confidence numbers. Confirm you can actually see and read them — this is the point of the 
whole role, not just getting answers out.

### Step 2 — Log the confidence numbers

When the model generates, record the relevant events in the frozen log schema 
(`eventlog/schema.md`):
- `llm_first_token` — the first output token
- `llm_final` — the full answer text
- Log the per-token logprobs alongside these so the harness and later experiments can read 
  them.

### Step 3 — State tracker (`tracker.py`)

A small component that, each turn, sends the conversation so far to the model with a strict 
instruction to output ONLY JSON describing what the user wants, as slots. For example:
```json
{"intent": "book_spa", "time": "4pm", "people": 2, "confirmed": false}
```
Use a fixed JSON schema so the output is always parseable. If a slot is unknown, its value 
is "unknown" — that is correct, not an error.

## Important setup note

Capture your setup so it reproduces on a fresh machine. Cloud GPU instances get created and 
destroyed, so the setup must be written down (in a script or notes here), not hand-installed 
once and forgotten. When we move to Docker in Week 2, this setup goes into the shared image. 
Add the Python packages you use to the shared `requirements.txt`.

## Cost note

Use cheap or free compute for development and testing. Reserve the paid GPU only for real 
runs. Spin instances down when not in use.

## Not in Week 1

- The full belief-vector instrumentation (logging every signal every 50 ms) — Week 2.
- The Experiment 1 labeled question set — Week 2.

## Definition of done (Week 1)

The model answers prompts with its confidence numbers visible and logged, and the state 
tracker outputs valid, parseable JSON each turn.

## Coordinate with

The integration lead, who will connect this to the spine once it serves reliably.

---

## Phase 2 — Signal Instrumentation and Experiment 1

In Phase 1 the model was served with its confidence numbers exposed. In Phase 2 we capture 
the full set of signals the controller and experiments need, and run Experiment 1, the 
first and most important experiment.

### Goal

Build the full belief-state signal capture (all the numbers each turn), and own Experiment 
1, the signal validity audit that decides whether the whole trigger approach works.

### How to do it

- Log the full set of signals each turn, not just the first-token confidence: per-token 
  confidence and entropy across the answer, retrieval scores, and the tracker's uncertainty. 
  These are what the controller reads to make decisions and what the experiments analyse.
- Build the Experiment 1 dataset: a set of questions with known correct answers AND a label 
  for whether retrieval was actually needed for each. Coordinate with Lakshya, who is 
  bringing in the public datasets (Spoken-SQuAD, HeySQuAD) that make these results 
  comparable to prior work.
- Run Experiment 1 offline: for each candidate uncertainty signal, measure how well it 
  predicts "retrieval was needed" using AUC (0.5 = useless, 1.0 = perfect). Record whether 
  the best signal reaches 0.7.

### Why it matters (this is the gatekeeper)

The whole project assumes that when the model is uncertain, it needs to look something up. 
Experiment 1 tests exactly that. It runs first and gates everything: if the signals do not 
predict retrieval need (AUC below 0.6), we pivot the approach. So this is the most important 
single experiment.

### Definition of done

Full signals logged every turn, and Experiment 1 executed with the AUC results and the 
pass/fail verdict recorded in the lab log (paper/lab_log.md).

### Ryan branch implementation status

- `client.py` now preserves vLLM's top-k alternatives and logs chosen-token
  confidence plus per-token entropy and summary uncertainty signals.
- `tracker.py::uncertainty()` exposes the fixed-schema unknown-slot fraction.
- `signals.py` combines model, raw retrieval-score, and tracker signals into a
  versioned per-turn snapshot for the integration owner to attach to the
  approved shared event/belief location.
- `experiment1/` contains the balanced answer-bearing label set, frozen
  thresholds, a dependency-light ROC AUC runner, and explicit mock-vs-vLLM
  result validity rules. See `experiment1/README.md` for commands.

Ryan's branch does not edit the frozen event schema or another owner's
pipeline. The existing shared log already records retrieval scores and the
expanded `llm_final` payload records the model signals. Logging the tracker
snapshot into the shared timeline still requires Renya to approve an event
schema home and wire the side-effect-free payload returned by `signals.py`.

### Coordinate with

Lakshya (public datasets for the labelled question set), Renya (the controller reads these 
signals), and Yashmitha (records the Experiment 1 verdict in the lab log and pre-registration 
doc).
