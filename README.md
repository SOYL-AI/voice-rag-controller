<<<<<<< HEAD
# Voice-RAG Controller

A research project by the SOYL-AI team. Target: IEEE conference paper.

This README is a living document. It currently covers the project overview and Week 1. 
Later phases will be added as we go.

---

## What we are building, in plain terms

We are building a voice assistant that answers questions from documents (for example, a 
hotel phone concierge answering "is the pool open?" or "what is the cancellation policy?").

Such an assistant has to make three decisions live, during a call:

1. **When to look something up** (retrieval) — searching takes time, and on a phone call 
   even one second of silence feels like the line dropped.
2. **What to remember and what to forget** (memory) — a long call cannot all fit in the 
   model's working memory, so we compress, without losing the detail that matters later.
3. **When to speak, wait, or stop** (turn-taking) — speak too soon and we interrupt; wait 
   too long and we feel slow.

We are not building a new speech model or a new search engine. We are assembling standard 
open-source parts and adding a thin decision layer on top.

---

## Problem Statement

Today, those three decisions are handled by three separate, simple mechanisms that do not 
talk to each other: a retrieval rule, a memory rule, and a silence timer. Each is blind to 
the others, and none of them accounts for how long the user actually waited. That 
disconnect is the gap we address.

---

## Objective

Our core idea: these are not three separate problems. They are one. We build a single 
small controller that watches everything at once and makes all three decisions together, 
using one principle: take an action only when it is worth the delay it causes.

Specific objectives:

- Build a working voice-RAG pipeline where all three decisions can be measured.
- Test whether the model's own uncertainty (its confidence numbers) can reliably signal 
  when it needs to look something up.
- Test whether timing retrievals cleverly (for example, while the user is still talking) 
  hides the delay.
- Test the main claim: does one unified controller actually beat three well-tuned separate 
  ones? It might not, and that is a valid, publishable finding either way.

---

## Methodology

- **Build a transparent pipeline** from open-source parts (speech-to-text, search, 
  language model, text-to-speech). We build our own measurable system rather than using a 
  sealed one, because the experiments need to see the model's internal signals.
- **Log everything.** Every event, timestamped, in one file per conversation. Every result 
  in the paper comes from these logs.
- **Use a fake-user simulator** to run hundreds of test conversations overnight, so we do 
  not need live humans for every experiment.
- **Compare against strong baselines**, not weak ones, including rebuilt versions of 
  existing published methods, so our results are credible.
- **Pre-decide what counts as success** for each experiment before running it, and report 
  failures as honestly as wins.

The one-line version: we are building a voice assistant that decides when to look things 
up, what to remember, and when to speak, all with one smart controller instead of three 
separate rules, and rigorously testing whether unifying those decisions actually helps.

---

## Repository structure

| Folder | What it holds |
|--------|---------------|
| `pipeline/` | The end-to-end spine that connects every stage |
| `controller/` | The decision brain (starts as a no-op stub) |
| `audio/` | Streaming speech-to-text and text-to-speech |
| `retrieval/` | Embeddings, FAISS index, hotel corpus, search function |
| `llm/` | Model serving with confidence numbers, and the state tracker |
| `simulator/` | Fake-user scenario generator and answer keys |
| `harness/` | Log reader, metrics, and the replay-to-timeline viewer |
| `eventlog/` | The event-log schema and the log writer |
| `paper/` | The paper draft and supporting docs |
| `configs/` | All settings as files, never hardcoded |

Each folder has its own README with that component's Week 1 goal and how to start.

---

## Working rules for everyone

1. Do not write any code that touches the event log until the log schema 
   (`eventlog/schema.md`) is frozen. It is the shared contract that lets six people work in 
   parallel without their outputs clashing.
2. Everything goes into this repository through your own feature branch and a pull request. 
   Nobody commits directly to `main`.
3. Everyone installs the same Python packages from the shared `requirements.txt` so our 
   environments match. (We will move to Docker in Week 2 once the cloud GPU setup defines 
   the full environment.)
=======
# Testing
>>>>>>> 9ae2c0846a54e9ac48402c86abe8ce43bb83f257
