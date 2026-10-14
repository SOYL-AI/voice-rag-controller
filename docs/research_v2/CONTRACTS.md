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
    speech/traces/<condition>.jsonl  # schema §4 (committed), one per voice x rate x noise
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
  Required for `d2_support`, optional for `d1_open`.

## 3. Corpus schema (`corpus.jsonl`)

```json
{"doc_id": "...", "dataset": "...", "domain": "...", "org": "...|null",
 "title": "...", "text": "...", "source": "authored|wikipedia|...", "licence": "..."}
```

Passages should be ≤ ~350 words so they fit the existing bge-small + FAISS
chunking.

## 4. Speech trace schema (`speech/traces/<condition>.jsonl`)

One file per **condition** = voice x speaking rate x noise, named
`<voice_id>__<wpm>wpm__<noise>.jsonl` (`<voice_id>__natural__<noise>.jsonl` for
uncontrolled-rate human recordings), one line per item:

```json
{
  "item_id": "...", "voice_id": "piper-en_US-lessac-medium", "accent": "en-US|en-IN|en-GB",
  "wpm_target": 150, "wpm_actual": 148.2, "noise": "clean|babble_10dB|babble_5dB",
  "noise_meta": {"type": "babble", "snr_db": 10, "n_talkers": 6, "seed": 123} | null,
  "tts_engine": "piper-tts 1.x", "length_scale": 1.94,
  "audio_path": "<name>/speech/audio/<voice_id>/<wpm>wpm/<noise>/<item_id>.wav",
  "duration_ms": 3120, "lead_ms": 540, "file_ms": 6200,
  "words": [{"w": "What's", "start_ms": 0, "end_ms": 310}],
  "partials": [{"t_ms": 400, "available_ms": 610, "compute_ms": 205.3, "audio_ms": 368,
                "text": "What's the", "stable": false, "stable_prefix_words": 0}],
  "n_empty_partials": 1,
  "asr_final": {"t_ms": 3980, "available_ms": 4350, "compute_ms": 370.1, "text": "..."},
  "vad_user_end_ms": 3980, "endpoint_reason": "vad|stream_end", "vad_onset_ms": 30,
  "asr_model": "faster-whisper-<size>", "asr_settings": {...}, "partial_interval_ms": 400,
  "word_timing": {"method": "ctc_forced_alignment", "model": "..."},
  "replay_wall_ms": 2500.0, "trace_version": "v2-speech-1"
}
```

- `t_ms` is milliseconds from speech onset (forced-aligned start of the first
  word), on one clock shared by `words`, `partials`, `asr_final` and the VAD
  fields. `duration_ms` is first-word start to last-word end.
- `t_ms` of a partial is the audio time at which the ASR was run on the audio
  received so far; **`available_ms`** is when its text exists for a consumer:
  `max(t_ms, previous available_ms) + compute_ms` (one serial ASR worker,
  compute measured on the host that produced the trace). Controllers and the
  latency simulation must act on a partial no earlier than `available_ms`.
- `vad_user_end_ms` is the Silero-VAD end of turn with the live thresholds;
  `null` (with `endpoint_reason: "stream_end"`) if no endpoint fired before the
  audio ended. The live pipeline ends the turn at the first endpoint, so an
  endpoint inside a pause truncates `asr_final` - that is kept, not repaired.
- `partials` must be what the real streaming ASR emitted, including errors
  and revisions. Never write the reference transcript into a partial.
- Producer: `audio/v2/trace.py`; method: `docs/research_v2/SPEECH.md`.

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
