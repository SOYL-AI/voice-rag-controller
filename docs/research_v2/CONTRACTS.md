# Research v2 — Shared Contracts

Every v2 workstream builds against this file. If you need to change a
contract, change it here in the same commit and say so in your report. Do
not silently fork a schema.

## 0. Why v2 exists (one paragraph)

The v1 E3 null is analytically forced. With (A1) answer quality independent
of retrieval timing, (A2) a constant per-policy overlap, and (A3) a belief
that does not depend on when retrieval is dispatched, the joint controller
at latency price λ makes exactly the decisions the decomposed controller (B9)
makes at λ·(L−o*)/L. The two trace the same frontier. This is observed in the
real E3 data: Ours(λ_i) = B9(λ_{i−1}), and the grid ratio of 1.38 equals
0.672/0.484.

v2 tests the theory's predictions where A1–A3 are broken *by the real
system*:

- **C1, turn-dependent overlap.** Hideable latency depends on how much speech
  remains after dispatch.
- **C2, information-vs-overlap.** The belief computed on a transcript prefix
  sharpens as more speech arrives, so waiting buys information but loses
  overlap.
- **C3, partial-query quality.** Live ASR partials degrade the query, so
  quality depends on timing.

The objective is also corrected: retrieval *hurts* on questions that do not
need it. Measured in v1 E3, retrieval adds +0.72 accuracy on items that need
it and costs −0.59 on items that don't.

## 1. Repository layout (v2)

```
datasets/v2/<dataset_name>/
    items.jsonl            # one item per line, schema §2
    corpus.jsonl           # one document/passage per line, schema §3
    splits.json            # {"dev": [ids], "test": [ids], "seed": ..., "method": ...}
    README.md              # provenance, licence, construction, stats
    speech/traces/<voice_id>.jsonl   # schema §4 (committed)
    speech/audio/...       # .wav files — NOT committed (gitignored)
llm/experiment1_v2/        # E1-v2 code + results (schema §5)
controller/baselines/      # v2 controllers live next to v1 ones (new files)
pipeline/ , harness/       # v2 infra additions
energy/                    # energy instrumentation
docs/research_v2/          # theory, pre-registration addendum, run manifests
```

Dataset names: `d1_open` (public open-domain QA, popularity-stratified) and
`d2_support` (multi-domain customer-support / service-agent benchmark).

## 2. Item schema (`items.jsonl`)

```json
{
  "id": "d2_support-000123",
  "dataset": "d2_support",
  "domain": "telecom",
  "org": "Northwind Mobile",
  "question": "What's the late payment fee on a postpaid plan?",
  "expected_answer": "₹150, charged 7 days after the due date.",
  "answer_aliases": ["150 rupees", "Rs 150"],
  "gold_doc_ids": ["d2-telecom-northwind-billing-03"],
  "item_type": "org_specific | general | counter_prior | ambiguous | popqa",
  "retrieval_needed_authored": true,
  "pair_id": "d2-pair-0061",
  "popularity": null,
  "history": [],
  "split": "dev | test",
  "meta": {}
}
```

- `retrieval_needed_authored` is the author's label. The **empirical** label
  (closed-book wrong across draws) is produced by E1-v2, never by hand, and
  lives in E1-v2 outputs keyed by `id`.
- `pair_id` links matched pairs: same domain, same question template, and
  similar length, where one item needs retrieval and the other does not.
  Required for `d2_support`, optional for `d1_open`. In `d2_support` a
  `pair_id` names a **pair group**:
  - exactly one `general` item
  - one or two retrieval-needed items from the same organisation, each within
    ±3 words of the general item's question

  The whole group always shares a split.
- `history` is a list of prior turns that the question depends on. Each
  turn is `{"role": "user" | "assistant", "content": "..."}`, and the list
  ends with an assistant turn. An empty list means a single-turn item. The
  D2 builder also accepts `{"role": "caller" | "agent", "text": ...}` in raw
  files and normalises it to this shape.

## 3. Corpus schema (`corpus.jsonl`)

```json
{"doc_id": "...", "dataset": "...", "domain": "...", "org": "...|null",
 "title": "...", "text": "...", "source": "authored|wikipedia|...", "licence": "..."}
```

Passages should be ≤ ~350 words so they fit the existing bge-small + FAISS
chunking.

## 4. Speech trace schema (`speech/traces/<voice_id>.jsonl`)

One line per item rendered with that voice:

```json
{
  "item_id": "...", "voice_id": "piper-en_US-lessac-medium", "accent": "en-US|en-IN|en-GB",
  "wpm_target": 150, "noise": "clean|cafe_10dB|street_5dB", "tts_engine": "...",
  "audio_path": "speech/audio/<voice_id>/<item_id>.wav", "duration_ms": 3120,
  "words": [{"w": "What's", "start_ms": 0, "end_ms": 310}],
  "partials": [{"t_ms": 400, "text": "What's the", "stable": false}],
  "asr_final": {"t_ms": 3500, "text": "..."},
  "vad_user_end_ms": 3180,
  "asr_model": "faster-whisper-<size>", "partial_interval_ms": 300
}
```

- `t_ms` is milliseconds from speech onset, on one clock.
- `partials` must be what the real streaming ASR emitted, including errors
  and revisions. Never write the reference transcript into a partial.

## 5. Signal record schema (E1-v2: `llm/experiment1_v2/results/*.jsonl`)

```json
{
  "item_id": "...", "model": "mistralai/Mistral-7B-Instruct-v0.3", "draw_seed": 11,
  "prefix": {"kind": "full|partial", "source": "asr|text", "voice_id": "...|null",
             "t_ms": 1800, "word_frac": 0.6, "text": "..."},
  "signals": {"mean_entropy_nats": 0.0, "max_entropy_nats": 0.0,
              "first_k_mean_entropy_nats": {"4": 0.0, "8": 0.0, "16": 0.0}, "...": 0.0},
  "closed_book_answer": "...", "ttft_ms": 0, "gen_ms": 0, "n_tokens": 0
}
```

## 6. Controller signal interface (additions to `Ctx.signals`)

The v1 keys are unchanged. v2 adds:

```python
signals["model"]["uncertainty_by_prefix"] = [
    {"t_ms": 900,  "word_frac": 0.3, "mean_entropy_nats": ..., "first_k_mean_entropy_nats": {...}},
    ...
]
signals["timing"] = {
    "speech_onset_ms": 0,
    "partials": [{"t_ms": ..., "text": ...}],          # from the speech trace
    "predicted_remaining_speech_ms": [{"t_ms": ..., "remaining_ms": ...}],  # from the end-of-turn/remaining-time predictor
}
signals["retrieval"] = {"latency_profile": "faiss|faiss_rerank|remote_api",
                        "expected_latency_ms": ..., "latency_quantiles_ms": {...}}
```

A controller must treat any missing key as "signal not available" (the v1
`Ctx` contract).

## 7. Splits and sealing (pre-registration hygiene)

- 30% dev / 70% test. Deterministic: sha256(`id`) mod 1000 < 300 → dev.
  Stratify by `item_type` (D2) or popularity bin (D1); `pair_id` partners
  always share a split.
- **The test split is sealed.** No model generation, signal extraction,
  tuning or judging may touch test items until the v2 pre-registration
  addendum is frozen and tagged. Code may *load* the test ids only to exclude
  them. Every runner must take `--split dev|test` and refuse `test` unless
  `docs/research_v2/PREREG_FROZEN` exists.

## 8. Commit rules

- Work on your own branch, commit locally, **do not push**.
- No AI attribution anywhere: no Co-Authored-By lines and no mention of AI
  tools in commit messages, code comments or docs.
- Never commit secrets, endpoint URLs, API keys or `.wav` files.
- The Windows host uses `.venv-e2` (Python 3.10) for anything that imports
  pyarrow, faster-whisper or faiss; the system Python 3.12 crashes in pyarrow.
  Set `PYTHONUTF8=1`.
