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
- **The retrieval call is slow because of CPU thread oversubscription.**
  It takes a median of 657 ms and a p95 of 3.6 s for bge-small over 61
  documents. The cause is torch's default 16 intra-op CPU threads on the run
  laptop. With 1–4 threads the query embedding takes about 20 ms. The
  pipeline now has an opt-in cap, `VOICE_RAG_TORCH_THREADS`. Section 3 has
  the measured profiles.
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

Filled from `retrieval/latency_profiles_v2.json`,
`analysis/llm_tts_v2/measure_llm_tts.json` and
`analysis/latency_model_v2/`, see below.
