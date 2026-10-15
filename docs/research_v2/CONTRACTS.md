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
  "item_type": "org_specific | general | counter_prior | ambiguous | popqa | multihop",
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
- `meta.key_facts` (D2) lists the facts a correct answer must contain and
  must not contradict. `meta.core_facts` (D2) is a non-empty, verbatim
  subset of `key_facts`.
  - **Definition.** A key fact is *core* if, without it, the caller's literal
    question is not answered, or is answered wrongly. Everything else is
    supplementary: extra conditions, advice, follow-up steps, contact
    details and rationale.
  - **By type:**
    - `general`: usually one core fact, the direct answer.
    - `org_specific` and `counter_prior`: the org value or values asked
      for, plus any condition that flips the answer when the question
      doesn't settle it.
    - `ambiguous`: the org-specific detail that makes the generic answer
      wrong.
  - **Questions that ask several things** ("how much and how long"): one
    fact per thing asked is core.
  - **Judging.** A judge should require every core fact to be present. A
    supplementary fact counts only if the answer contradicts it.
  - **Provenance.** Core facts were annotated per item from the question,
    history, expected answer and gold documents only, without looking at
    any model output.
- `item_type: "multihop"` marks multi-hop open-domain questions (D1 uses
  HotpotQA bridge and comparison questions).
- **D1 (`d1_open`) has no authored retrieval label.** For every D1 item
  `retrieval_needed_authored` is `null`: whether a public-knowledge question
  needs retrieval depends on what the model memorised, so D1 relies only on
  the empirical label (closed-book correctness computed by E1-v2).
  `popularity` (PopQA `s_pop`, the subject's Wikipedia page views) is the
  covariate; it is `null` for `multihop` items.

## 3. Corpus schema (`corpus.jsonl`)

```json
{"doc_id": "...", "dataset": "...", "domain": "...", "org": "...|null",
 "title": "...", "text": "...", "source": "authored|wikipedia|...", "licence": "...",
 "meta": {}}
```

`meta` is optional and free-form (e.g. Wikipedia `pageid`/`revid`/section);
consumers must not depend on it.

A corpus too large to commit (over ~50 MB) is committed as `corpus.jsonl.gz`
(deterministic gzip of the same JSON lines) and the plain `corpus.jsonl` is
gitignored and regenerated by the dataset build script. Consumers must accept
either file; `retrieval/build_index_v2.py` does. D1 uses this form.

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
  "vad_user_end_ms": 3980, "user_end_ms": 3980,
  "endpoint_reason": "vad|fallback_timeout|stream_end",
  "endpoint_rule": null | "no_text_change", "fallback_rule": "armed-2", "fallback_arm_ms": 410,
  "n_partials_after_end": 0,
  "vad_onset_ms": 30,
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
  `null` if the VAD never confirmed one (typical under babble). The live
  pipeline ends the turn at the first endpoint, so an endpoint inside a pause
  truncates `asr_final` - that is kept, not repaired.
- **`user_end_ms` is the end of turn every consumer uses.** It equals
  `vad_user_end_ms` when the VAD fired (`endpoint_reason: "vad"`). Otherwise the
  fallback applies (`endpoint_reason: "fallback_timeout"`, `fallback_rule:
  "armed-2"`): armed at the first partial with >= 2 words (`fallback_arm_ms`),
  the turn ends 2.0 s after the last change of the normalised partial text
  (`endpoint_rule: "no_text_change"`); every ASR call counts as the user still
  talking. Partials after that moment are dropped (`n_partials_after_end` counts them)
  and `asr_final` covers the audio up to it. `"stream_end"` remains only for
  turns where no partial ever had >= 2 words. Report fallback turns separately.
  Traces from `trace_version` v2-speech-1 lack these fields; `trace.py
  --upgrade-fallback` adds them.
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

Optional, additive keys read by the v2 controllers (`v2/controllers`):

```python
signals["timing"]["user_end_ms"]          # trace user_end_ms (VAD end, or the fallback end of turn)
signals["timing"]["asr_final"]            # {"t_ms": ..., "text": ...}
signals["timing"]["partial_interval_ms"]  # ASR partial cadence
signals["retrieval"]["nonhideable_increment_ms"]  # pi: TTFA increment of USING passages
                                          # (extra prefill/generation) that no timing hides
# per uncertainty_by_prefix entry (all optional):
#   "available_ms"    when the prefix signal is usable (default t_ms + config lag)
#   "kind": "full"    marks the full-transcript entry (or word_frac >= 1)
#   "min_token_prob" | "min_token_logprob" | "token_probs" | "token_logprobs"
#                     token-probability signals for the FLARE baseline (B4)
#   "max_entropy_nats" for the DRAGIN baseline (B6)
```

Causality: controllers see a step's data only if stamped at or before that
step's time; `word_frac` (reference-transcript fraction) is never used at
decision time, only the causal estimate e/(e+R) (see
`controller/sequential_v2.py`).

## 9. Controller-side artifacts (v2/controllers)

- **E1-v2 calibration artifact** (`controller.e3v2.calibration.from_artifact`):
  `{"calibration": {"signal", "kind": "logistic", "intercept", "slope",
  "frac_slope"?, "interaction"?, "calibrated", "source", "auc"?}}` —
  P(needed) = sigmoid(intercept + slope·x + frac_slope·f + interaction·x·f).
- **E1-v2 empirical labels** (for `oracle_v2`, `tune_v2`): JSON
  `{item_id: closed_book_wrong_bool}`.
- **Belief transitions** (`controller/fit_belief_transitions.py`):
  `{"bin_edges", "phase_edges", "next": {phase: {bin: {"probs","values","n"}}},
  "final": {...}, "provenance"}`.
- **Phase A cell store** (`simulator/e3v2_decide.py` docstring): JSONL of
  `{"kind": "retrieval", item_id, condition, partial_index|null,
  retrieved_doc_ids}` and `{"kind": "cell", item_id, seed, condition,
  path, query_variant, answer_correct, n_prompt_tokens?, ...}`;
  `query_variant = "qv-" + sha256(ordered doc ids)[:16]`, `"none"` for skip.
- **Latency model**: `harness/latency_model_v2.LatencyModel.sample_ttfa(path,
  dispatch_ms, user_end_ms, profile, n_prompt_tokens, rng) -> dict` with at
  least `ttfa_ms` and `exposed_retrieval_ms` (stub:
  `harness/latency_model_v2_stub.py`).
- **v2 result records**: v1 frontier schema plus `dispatch_t_ms`,
  `condition`, `latency_profile`, `exposed_latency_ms`, `architecture`
  (`commit | speculative`) and `used` (false = a speculative dispatch was
  discarded: skip-cell answer, compute counted)
  (`analysis/e3_frontier.validate_v2_records`). Files:
  `e3_v2_<architecture>_<controller>_seed<seed>.json`.
- **Pipeline architecture** (`controller.e3v2.architecture`): `commit` = a
  dispatched retrieval is always used; `speculative` = it may be discarded
  at end of turn. The v2 pipeline must state which it implements; under
  `speculative`, `b10_speculative_v2` (SPEC, THEORY.md Prop. 4′) is a
  mandatory rival. Per the PI decision (2026-10-16) both are evaluated:
  `ours_v2` is Ours-v2[COMMIT] or Ours-v2[SPECULATIVE] according to the
  architecture, and `simulator/e3v2_decide.py` writes
  `e3v2_boundary.json` with Δ[COMMIT] (vs B9+), Δ[SPECULATIVE] (vs D*, the
  dev-better of B9+ and SPEC, fixed at freeze for test), their paired
  interaction, q̂, w̄ and m = λ_c(1+w̄)q̂ (PREREG_V2 §3). Records carry
  `realized_utility` = Q − λ_l·(TTFA − skip TTFA) − λ_c·n_retrievals.

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
