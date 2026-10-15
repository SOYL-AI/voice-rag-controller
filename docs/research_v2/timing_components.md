# Research v2: timing components (pipeline branch)

This note covers the latency-side components built on `v2/pipeline`: where
TTFA goes, the live partial-transcript path, retrieval latency profiles,
streaming TTS, the remaining-speech predictor, the `forecast` policy and the
trace-driven latency model. The numbers come from committed artifacts. The
file paths are given so each number can be regenerated.

## 1. TTFA breakdown of the v1 E2 runs

`python -m harness.latency_breakdown --logs "logs/conv_*.jsonl" --label e2 --paper-dir paper`
writes `analysis/latency_breakdown/latency_breakdown_e2.{json,md,png}`,
`paper/figures/latency_breakdown_e2.pdf` and
`paper/tables/latency_breakdown_e2.tex`.

Stages run along the critical path after `user_end` and sum to TTFA exactly.
Values are the mean in ms, with the share of mean TTFA in brackets
(n = 250 per policy):

| policy | TTFA p50 / p95 | exposed retrieval | LLM TTFT | generation until TTS input | TTS | other |
|---|---|---|---|---|---|---|
| sentence_end | 4110 / 7052 | 1311 (29.5%) | 1065 (23.9%) | 1997 (44.9%) | 49 (1.1%) | 26 |
| immediate | 4070 / 6915 | 1219 (28.4%) | 1058 (24.7%) | 1979 (46.2%) | 17 (0.4%) | 12 |
| fixed_early_proxy | 3688 / 7090 | 992 (24.5%) | 1055 (26.0%) | 1979 (48.8%) | 18 (0.5%) | 10 |

Findings:

- **Generation before TTS is the largest stage.** Batch TTS waits for the
  whole answer (about 62 tokens at about 33 ms per token).
- **The retrieval call's in-pipeline time is not its compute cost.** In E2
  it took a median of 657 ms and a p95 of 3.6 s for bge-small over 61
  documents. Isolated microbenchmarks of the same call give 20–60 ms p50
  (section 8). With torch's default 16 CPU threads the time was
  load-dependent: one in-process run on the laptop measured ~1.2 s p50,
  while the artifact run measured 26 ms p50. With the opt-in cap
  `VOICE_RAG_TORCH_THREADS=4`, the v2 real-endpoint pipeline runs measured
  47 ms median (p95 ~250 ms) in-pipeline. The v1 retrieval latency was
  host contention, not intrinsic retrieval cost.
- **The E2 "TTS" stage measured no real synthesis.** Piper was not
  installed and the voice model was missing, so `audio/tts.py` fell back to
  a synthetic tone, and TTFA contains no synthesis cost. With real piper,
  the v1 subprocess path costs 3–6 s per clause, because it starts a new
  process and reloads the model every time. The new resident-voice mode
  (`audio.tts.piper_mode: resident`) takes about 0.35–0.65 s per first
  sentence on CPU.
- The windows timer has about 15.6 ms resolution, so stages are quantised.

## 2. Live partial-transcript path

`pipeline/spine.py` has a trace-driven text path, used when `speech_trace`
is given, when the policy is `forecast`, or when
`query_source="live_partial"`. It replays a CONTRACTS §4 trace in real time:
partials at their `t_ms`, `user_end` at `vad_user_end_ms`. The `user_end`
payload records `speech_end_offset_ms`. When the policy dispatches, the
rewrite and retrieval run on the partial available at that moment.
`frozen_query` is not used. `retrieval_launched` logs `query_source`,
`partial_word_count`, `partial_t_ms`, the retrieval profile and the
forecast inputs.

`retrieval/partial_query_risk.py` estimates rho(t), defined as
P(retrieval on the partial misses gold within k | the full transcript hits).
It fits rho on DEV items only and outputs it isotonically by word fraction
and by remaining speech. Controllers read it with `rho_at()`.

The development run uses the simulated v1 traces with the faiss profile:
`analysis/partial_query_risk/v1_textsim_faiss.json`. It covers the 25
retrieval-needed items × 3 speaking rates, and the rows below are the
query-ready partials only:

| word fraction | n | hit@3 partial | rho@3 |
|---|---|---|---|
| 0.4–0.6 | 15 | 0.40 | 0.60 |
| 0.6–0.8 | 37 | 0.38 | 0.62 |
| 0.8–1.0 | 87 | 0.87 | 0.13 |
| 1.0 | 138 | 1.00 | 0.00 |

So partial-query risk is far from zero once retrieval runs on the live
partial. The E2 value of 0 was structural, because the query was frozen.
These are simulated traces, so they must be refit on the real
`audio/v2/` traces.

## 3. Retrieval latency profiles

See `retrieval/latency_profiles_v2.json`, produced by
`python -m retrieval.measure_latency`, and the table in section 8.
Controllers get `signals["retrieval"]` from
`retrieval.profiles.retrieval_signals(profile)` for the hideable call, or
from `harness/latency_model_v2.py`'s `signals_retrieval()`, which also
returns the non-hideable increment. The `remote_api` profile is measured
against a uvicorn server on 127.0.0.1. That measurement isolates
serialisation and HTTP cost from compute. WAN RTT stays `null` until a
deployment of `retrieval/modal_app.py` is approved.

## 4. Streaming LLM to TTS

`llm/client.py` clients take an `on_delta` callback. `pipeline/streaming.py`
cuts the token stream into clauses with `audio.tts.ClauseAccumulator`, which
splits exactly where `split_into_clauses()` does, and synthesizes clause 0
while the model is still generating. New events are `llm_first_clause` and
`tts_clause_ready`. Select the mode with `tts_mode` in `run_conversation`,
`VOICE_RAG_TTS_MODE`, or `pipeline.tts_mode` in the config. Section 8 has
the measured effect on the real endpoint.

## 5. Remaining-speech predictor

`controller/remaining_speech.py` predicts remaining speech R_t, measured to
the end of the last word, from the partial text and elapsed time. It
returns quantiles q10, q50 and q90. Three models are available:

- `prior`: survival on total duration.
- `rate_words`: survival on total words × the user's measured rate.
- `qr`: linear quantile regression on the rate_words estimate plus
  completeness features. These cover the last word (determiner,
  preposition, auxiliary, wh-word, conjunction or pronoun), the opener
  type, and the content-word count, scaled by the measured per-word time.

Punctuation is ignored, because streaming partials carry unreliable
punctuation.

Grouped 5-fold CV by item on the simulated v1 traces (150 traces), from
`analysis/remaining_speech/remaining_speech_v1_textsim_cv.json`:

| model | MAE q50 | median AE | pinball q10 / q50 / q90 | 80% interval coverage |
|---|---|---|---|---|
| prior | 538 | 430 | 86.5 / 268.8 / 166.8 | 0.79 |
| rate_words | 592 | 512 | 138.6 / 296.0 / 192.7 | 0.69 |
| qr | **471** | **381** | **84.7 / 235.6 / 161.3** | 0.71 |

For the qr model, MAE falls from 1374 ms in the first quarter of the
utterance to 307 ms in the last quarter. The last quarter is the part that
matters for just-in-time dispatch. These utterances are 1.2–4 s,
template-like v1 questions with simulated timing. Refit on the real dev
traces before using any of this.

External options:

- **LiveKit turn detector**: not usable. Its licence (LiveKit Model License)
  allows use only together with the LiveKit Agents framework. It forbids
  standalone use, and it forbids using its outputs "to improve or otherwise
  develop any other models". So we can neither run it in this pipeline nor
  use it as a feature or teacher. It is also a text EOU classifier that
  returns P(end of turn), not a remaining time.
- **VAP (Voice Activity Projection)**: a poor fit. It models dyadic
  dialogue from two-channel audio. It predicts voice activity over a
  roughly 2 s horizon in discrete bins, and it was trained on telephone
  dialogue (Switchboard/Fisher). Our setting is a single user speaking a
  question to a silent system. We need the remaining time of the whole
  utterance, which is often more than 2 s at dispatch, and the development
  data has no audio. At most VAP could give a near-end hazard on the real
  `audio/v2` audio.
- **Pipecat Smart Turn v3** (BSD-2, about 8M parameters, ONNX, CPU):
  usable. It is an audio-native end-of-turn classifier and a good candidate
  for a near-end feature, or for the EOT detector itself, once real audio
  traces exist. It also returns a probability, not a remaining time.

## 6. The `forecast` policy

`controller/timing.py` adds the `forecast` policy. R(t) comes from
`signals["timing"]["predicted_remaining_speech_ms"]` or a configured
predictor. L is the expected latency of the hideable retrieval call for the
active profile, plus `margin_ms` and `extra_latency_ms` (for example a live
rewrite). The default rule, `jit`, dispatches at the first query-ready
partial where R(t) < L + Δt, where Δt is the partial interval. That is the
last partial at which the remaining speech is still predicted to cover the
retrieval, which gives the query as many words as possible without exposing
latency (the C2 trade-off). The alternative rule, `earliest`, is a gated
reactive ablation. If the policy never fires, retrieval falls back to
sentence end. `fixed_early_proxy` is unchanged.

## 7. Trace-driven latency model

`harness/latency_model_v2.py` provides `LatencyModel.sample_ttfa(path,
dispatch_ms, user_end_ms, profile, n_prompt_tokens, rng, tts_mode=...)`.
It draws the components from measured pools:

- end-of-turn detection;
- dispatch and hand-off overheads;
- the retrieval profile;
- TTFT as residual + slope × prompt tokens;
- generation until TTS input, from joint per-run records (n_tokens,
  first-clause token index, per-token time);
- TTS first audio, either as logged in v1 or as resident piper synthesis
  versus clause length.

The model splits retrieval's TTFA cost into the HIDEABLE retrieval call and
the NON-HIDEABLE remainder: prefill of the passages, a longer answer before
TTS input, and the first clause's synthesis. `signals_retrieval()` exports
both. `expected_latency_ms` and `latency_quantiles_ms` describe the call,
and `nonhideable_increment_ms` (mean and quantiles, per TTS mode) describes
the remainder.

Fidelity, from `analysis/latency_model_v2/fidelity_report.md`, using
out-of-fold predictions from 5-fold CV grouped by scenario:

- **E2 TTFA.** p50 is predicted within 1–4%, the mean within 1.5%, and the
  KS distance is 0.04–0.06 per policy. A single 60/40 hold-out is noisier,
  with KS 0.12–0.14.
- **v1 E3 paired retrieve−skip increments.** Predicted vs observed means are
  1560 vs 1566 ms (sentence_end), 1332 vs 1295 ms (immediate) and 1163 vs
  1266 ms (fixed_early_proxy). Medians are 1566 vs 1390, 1371 vs 1196 and
  1146 vs 1125 ms. The model compares the median of per-pair expectations
  with the median of single noisy draws, which explains the median gaps.
  E3 TTFA distributions have KS 0.04–0.10.

## 8. Measurements on the current endpoint and host

Host: RTX 4060 Laptop and a shared CPU. Other jobs were running, and as
little as 0.2–1.4 GB of RAM was free, so tails are inflated.

### Retrieval profiles (`retrieval/latency_profiles_v2.json`)

785 timed queries per profile: the 50 v1 questions plus their ≥3-word
prefixes, in 5 shuffled passes after warm-up. Columns are ms.

| profile | p50 | p90 | p95 | p99 | notes |
|---|---|---|---|---|---|
| **faiss** (CPU embed, 4 threads) | 41.3 | 179.7 | 246.9 | 422.8 | embed is ~all of it; flat search 0.1 ms |
| faiss via asyncio.to_thread | 53.4 | 168.5 | 234.2 | 580.1 | how spine.py calls it |
| faiss, v1 default threads (16) | 26.4 | 39.3 | 46.6 | 76.5 | load-dependent, see section 1 |
| faiss_cuda | 22.3 | 27.2 | 30.2 | 40.9 | |
| **faiss_rerank** = top-50 + ms-marco-MiniLM-L6-v2 fp16 on CUDA | 65.9 | 74.8 | 78.0 | 110.4 | rerank 43 ms |
| faiss_rerank, MiniLM fp32 | 156.7 | 221.4 | 240.3 | 361.3 | |
| faiss_rerank, bge-reranker-base fp16 / fp32 | 165.3 / 536.9 | 293.0 / 564.4 | 306.0 / 575.8 | 322.7 / 604.2 | lower hit@1 (0.87 vs 0.96) on v1 |
| **remote_api** = faiss behind HTTP on loopback | 23.3 | 26.6 | 27.9 | 39.6 | server 20.1, transport 3.1 |
| remote_api with faiss_rerank backend | 66.1 | 72.2 | 74.4 | 90.8 | transport 3.4 |

MiniLM fp16 is the canonical reranker. It ties on quality with fp32
(hit@1 0.96, hit@3 1.00 on the retrieval-needed v1 questions), and the
bge-reranker-base variants are worse on hit@1. The loopback HTTP overhead
is about 3 ms. WAN RTT is not measured, because that needs the Modal
deployment.

FAISS flat-search component, single query, k=3, p50: 0.006 ms at 61
vectors, 0.06 ms at 1k, 1.2 ms at 10k and about 25–41 ms at 100k (noisy
host). 1M vectors was not run, to keep memory modest.

### LLM and TTS on the real endpoint

From `analysis/llm_tts_v2/measure_llm_tts.json` and `logs/v2_llm_tts/`:
Mistral-7B-Instruct-v0.3 on vLLM, sentence_end, frozen E3 queries, resident
piper on CPU. That is 50 scenarios × {skip, retrieve} × {batch, streaming},
plus a 60-request prefill sweep, 261 requests in total.

| path / TTS | TTFA p50 | p95 | mean | TTFT p50 | first token → TTS input p50 | TTS first audio p50 |
|---|---|---|---|---|---|---|
| retrieve / batch | 2664 | 4548 | 2816 | 328 | 1640 (full answer) | 617 |
| retrieve / streaming | 1836 | 3085 | 1946 | 328 | 718 (first clause) | 586 |
| skip / batch | 1696 | 3410 | 1890 | 297 | 813 | 609 |
| skip / streaming | 1430 | 2590 | 1499 | 313 | 547 | 500 |

- **Streaming TTS** cuts the retrieve-path TTFA by 31% at p50 and by 32% at
  p95.
- **Prefill** costs about 0.14–0.15 ms per prompt token. Median TTFT is
  321 / 355 / 462 ms at 79 / 596 / 1068 prompt tokens, which is 0, 5 and 10
  passages. About 0.3 s of TTFT is network round trip. So the passages'
  prefill is a small part of π.
- **Real piper synthesis** is now a large stage, about 0.5–0.6 s for the
  first clause on this loaded CPU.

### Hideable vs non-hideable cost of retrieval (π)

Paired retrieve−skip TTFA for the same scenario. π is that difference minus
the retrieval call exposed after the end of turn. All values in ms.

| source | TTS | total increment, mean / median | π mean | π p10 / p50 / p90 |
|---|---|---|---|---|
| v1 E3, sentence_end | batch (tone) | 1566 / 1390 | 739 | 782 at p50 |
| v1 E3, fixed_early_proxy | batch (tone) | 1266 / 1125 | 809 | 952 at p50 |
| v2 endpoint, sentence_end | batch (piper) | 926 / 874 | 855 | −431 / 812 / 1968 |
| v2 endpoint, sentence_end | streaming (piper) | 447 / 430 | **371** | −133 / 382 / 969 |

The model's π from rank-matched draws is 886 ms (batch) and 364 ms
(streaming), against 855 and 371 measured.

`analysis/latency_model_v2/signals_retrieval_v2.json` carries, per profile
and TTS mode, the hideable call (`expected_latency_ms`,
`latency_quantiles_ms`) and π (`nonhideable_increment_ms` as a number, plus
`nonhideable_increment_quantiles_ms`). This is the shape
`controller/baselines/_objective_v2.py` reads.

The controllers' config, `controller.e3v2` in `configs/config.yaml`, still
carries placeholders: faiss p50 672 / p90 3187, and π = 900. The measured
replacements are:

- faiss: p10/p50/p90 = 23/41/180.
- faiss_rerank: 57/66/75.
- remote_api: 20/23/27 on loopback, plus WAN RTT once deployed.
- π = 371 (streaming) or 855 (batch).

`LatencyModel(cfg=...)` samples the measured profiles regardless of what
that config says.

### Model fidelity (`analysis/latency_model_v2/fidelity_report.md`)

Out-of-fold predictions, 5-fold CV grouped by scenario:

| check | observed | predicted |
|---|---|---|
| E2 TTFA, sentence_end p50 / p95 / KS | 4110 / 7052 | 3998 / 7202, KS 0.062 |
| E2 TTFA, fixed_early_proxy p50 / p95 / KS | 3688 / 7090 | 3718 / 6859, KS 0.038 |
| E3 paired increment, sentence_end mean / median | 1566 / 1390 | 1557 / 1557 |
| E3 paired increment, fixed_early_proxy mean / median | 1266 / 1125 | 1160 / 1163 |
| E3 paired increment, immediate mean / median | 1295 / 1196 | 1340 / 1374 |
| v2 paired increment, batch mean | 926 | 964 |
| v2 paired increment, streaming mean | 447 | 453 |
| v2 TTFA KS (four path × TTS groups) | | 0.07–0.14 |

The model's "median" is the median of per-pair expectations. The observed
median is of single noisy draws, so the medians are not like for like and
should not be read too closely. The time to first clause reconstructed
from token indices matches the logged `llm_first_clause` to within 7.7 ms
MAE (n = 65).
