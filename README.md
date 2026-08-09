# Voice-RAG Controller

A research project by the SOYL-AI team, targeting an IEEE conference paper.

## The idea, in plain terms

We are building a voice assistant that answers questions from documents (for example, a
hotel phone concierge answering "is the pool open?" or "what is the cancellation policy?").

Such an assistant has to make three decisions live, during a call:

1. **When to look something up** (retrieval) — searching takes time, and on a phone call
   even one second of silence feels like the line dropped.
2. **What to remember and what to forget** (memory) — a long call cannot all fit in the
   model's working memory, so we compress, without losing the detail that matters later.
3. **When to speak, wait, or stop** (turn-taking) — speak too soon and we interrupt; wait
   too long and we feel slow.

Today, those three decisions are typically handled by three separate, simple mechanisms
that do not talk to each other: a retrieval rule, a memory rule, and a silence timer. Each
is blind to the others, and none of them accounts for how long the user actually waited.

Our claim: these are not three separate problems. They are one. We build a single small
controller that watches everything at once and makes all three decisions together, using
one principle: **take an action only when it is worth the delay it causes.**

We are not building a new speech model or a new search engine. We are assembling standard
open-source parts into a transparent, measurable pipeline and adding a thin decision layer
on top.

---

## Research questions

- **RQ1 — Signal validity (E1, the gatekeeper).** Does the language model's own
  uncertainty (its confidence numbers / logprobs) reliably indicate that it needs to
  retrieve a document? Runs first and gates the rest.
- **RQ2 — Timing value (E2).** Does timing retrieval cleverly — for example, while the
  user is still talking — buy latency without costing accuracy?
- **RQ3 — Coupling value (E3, the headline).** Does one unified controller actually beat
  well-tuned separate controllers for retrieval, memory, and turn-taking?
- **RQ4 — Memory scheduling (E4).** Does deferring memory operations to idle windows in
  the conversation improve latency at equal recall?

Any answer, including "no," is a valid, publishable finding — see
[`paper/preregistration.md`](paper/preregistration.md) for exact success criteria, decided
before results exist.

---

## How the system works

A pipeline chain carries each turn through fixed stages, with the controller sitting above
it and deciding what happens next:

```
audio in -> STT -> [controller decides] -> retrieval -> LLM -> TTS -> audio out
```

- **STT / TTS** turn speech into text and text back into speech.
- **Retrieval** searches a document corpus (embeddings + FAISS) for relevant passages.
- **The LLM** generates the answer and exposes its per-token confidence (logprobs) and a
  structured state tracker of what the user wants.
- **The controller** is the research core: it reads the model's uncertainty, retrieval
  scores, memory state, and timing signals, and decides whether to retrieve, wait, speak,
  or summarize — acting only when the value of an action beats the delay it causes.

Every event (words heard, searches launched, decisions made, audio spoken) is written,
timestamped on a single monotonic clock, to one `.jsonl` log file per conversation, in a
schema that every component shares. That log is the single source of truth for every
number and figure in the paper — see [`eventlog/schema.md`](eventlog/schema.md).

---

## Repository structure

| Folder | What it holds |
|--------|---------------|
| `pipeline/` | The end-to-end spine connecting STT, controller, retrieval, LLM, and TTS into one loop |
| `controller/` | The decision brain: the retrieval-trigger rule and the timing policies (reactive / boundary / forecast-timed) |
| `audio/` | Streaming speech-to-text and text-to-speech |
| `retrieval/` | Embeddings, FAISS index, hotel document corpus, and the search function |
| `llm/` | Model serving (vLLM) with exposed confidence numbers, the state tracker, and signal instrumentation |
| `simulator/` | Fake-user scenario generator, answer keys, and the batch runner for unattended test runs |
| `datasets/` | Public benchmark datasets (SpokenWOZ, Spoken-SQuAD) and loaders for comparability with prior work |
| `harness/` | Metrics (TTFA, accuracy, dead-air, interruptions), the replay-to-timeline viewer, and statistics helpers |
| `eventlog/` | The frozen event-log schema and the shared log writer |
| `paper/` | The paper draft, lab log, and pre-registration document |
| `configs/` | All settings as files — model names, thresholds, paths — never hardcoded |

Each folder has its own README with that component's details and how to work on it.

---

## Experiments

| # | Name | What it compares | Headline metric |
|---|------|-------------------|------------------|
| E1 | Gatekeeper | Candidate uncertainty signals vs. a labeled "was retrieval needed" set | AUC (0.7+ to pass, <0.6 kills the approach) |
| E2 | Timing | Reactive vs. sentence-boundary vs. forecast-timed retrieval, at matched accuracy | p95 time-to-first-audio (TTFA) |
| E3 | Coupling (headline) | Our unified controller vs. tuned separate controllers, with and without a shared latency price | Accuracy-vs-latency curves across a settings grid |
| E4 | Memory scheduling | Deferring memory operations to idle windows vs. immediate | Latency at equal recall |

Full designs, methodology, and pre-committed success/failure criteria live in
[`paper/preregistration.md`](paper/preregistration.md).

---

## Design principles

- **Transparency.** We build our own measurable system from open-source parts rather than
  using a sealed one, because the experiments need to see the model's internal signals.
- **One shared schema, frozen.** The event log format is a contract six people build
  against in parallel; nobody changes it without telling the whole team.
- **Log everything.** Every event, timestamped on one monotonic clock, in one file per
  conversation. Every figure in the paper comes from these logs.
- **Strong baselines.** Rivals get the same tuning effort as our method, including rebuilt
  versions of published approaches.
- **Pre-registration and honesty.** Success criteria are frozen before data exists, and
  failures are reported as honestly as wins.
- **Reproducibility.** A fake-user simulator runs hundreds of test conversations
  unattended, so results don't depend on live humans, and every figure regenerates from
  raw logs by one command.

---

## Working rules for everyone

1. The event log schema (`eventlog/schema.md`) is a frozen, shared contract — do not write
   code that produces or reads logs without going through the shared log writer
   (`eventlog/log_writer.py`).
2. Everything goes into this repository through your own feature branch and a pull
   request, with review. Nobody commits directly to `main`.
3. Everyone installs the same Python packages from the shared `requirements.txt` so
   environments match, plus `requirements-gpu.txt` on GPU/cloud machines that serve the
   language model.
