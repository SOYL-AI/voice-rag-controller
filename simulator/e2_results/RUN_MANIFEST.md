# Experiment 2 Modal Rerun Manifest (frozen rewrite)

This directory contains the result manifests for the full Experiment 2 timing sweep,
rerun after the frozen-rewrite fix. The raw event logs referenced by each result
record are versioned under `logs/` in the same branch.

## Why this rerun exists

The previous sweep (`experiment2/modal-mistral-results`) paid `rewrite_query()`'s
~1.5s latency *inside* the timed window. That latency was charged to every timing
policy — including `immediate` and `fixed_early_proxy`, which exist specifically to
launch retrieval early, during speech — and so ate into or entirely consumed the
retrieval-overlap window Experiment 2 measures. That is a rewrite-latency confound,
not a placement effect.

This run uses the merged fix (`090ec85`): each scenario's rewrite is precomputed
exactly once, up front, outside the timed sweep, and passed into `run_conversation()`
as `frozen_query`. Retrieval therefore launches immediately with the genuine
rewritten query. All 750 runs logged `rewrite_precomputed: true`.

## Run provenance

- Run window: 2026-08-12 15:56 to 21:10 (Asia/Calcutta), 5.2 hours
- Repository commit: `090ec85` (merge of `fix/e2-frozen-rewrite`)
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Serving: Modal-hosted vLLM (`*.modal.run`), OpenAI-compatible endpoint
- Timing policies: `sentence_end`, `immediate`, `fixed_early_proxy`
- Seeds: `0, 1, 2, 3, 4`
- Scenarios per policy/seed: 50
- Total scenario runs: 750
- Successful runs: 750
- Failed runs: 0
- Retrieval decision: frozen from each scenario's `retrieval_needed` label
  (`force_retrieve`), so only *when* retrieval fires varies, never *whether*
- Command: `python -u simulator/batch_runner.py`

Endpoint URLs and credentials are intentionally not committed.

## Environment constraints (required to reproduce)

- Run under `.venv-e2` (Python 3.10), not the system Python 3.12. Python 3.12.10
  on the run host dies with an access violation (`0xc0000005`) in `pyarrow\arrow.dll`
  during import — a native crash with no Python traceback, which also makes
  `pytest` hang silently. The 3.10 environment has no pyarrow.
- Set `VOICE_RAG_MODEL=mistral7b_v03`. `configs/config.yaml` defaults to
  `active_model: llama31_8b`, which this Modal endpoint does not serve.
- Set `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8`. The runner prints Unicode
  diagnostics that abort on the Windows charmap console codec before the pipeline
  executes.

## Methodology changes since the previous run

- **Warm-up**: 2 scenarios x 3 policies run before measurement, results discarded
  (`warmup_*.jsonl`, not committed). Warm-up exercises the frozen-rewrite path, so
  it warms the code path actually measured.
- **Interleaved policy order**: all three policies run back-to-back per
  (scenario, seed) rather than in per-policy blocks. Any gradual drift over the
  sweep lands on all three policies roughly equally instead of systematically
  favoring whichever block ran last. The previous run used block order.
- **Policy rename**: `predicted_pause` is now `fixed_early_proxy` (commit
  `87274e1`), honest framing — it is a fixed proxy point, not a forecaster.
- **Clean log state**: all colliding `conv_*`/`warmup_*` logs were archived before
  launch, so every committed JSONL contains exactly one complete run. The previous
  manifest's append-mode caveat on `conv_sentence_end_s0_001.jsonl` does **not**
  apply here.

## Artifacts

- `simulator/e2_results/e2_<policy>_seed<seed>.json`: one result manifest per
  policy/seed combination (15 files).
- `logs/conv_<policy>_s<seed>_<scenario>.jsonl`: raw event stream for each of the
  750 scenario runs, including timestamps, retrieval mechanism fields, generated
  answers, and token log probabilities.

Each result record's `log_path` points to its corresponding raw JSONL file.

## Measured launch offset vs user end

`retrieval_launched.payload.launch_offset_ms_vs_user_end`, negative = retrieval
fired before the user finished speaking. n = 250 per policy.

| policy | p10 | p25 | median | p75 | p90 | min | max |
|---|---|---|---|---|---|---|---|
| `fixed_early_proxy` | -729 | -359 | **-297** | 57 | 228 | -1575 | 5875 |
| `immediate` | -338 | 41 | **78** | 187 | 312 | -1177 | 6359 |
| `sentence_end` | 31 | 47 | **78** | 156 | 265 | 6 | 1469 |

## Important analysis notes

- The text-input path models user speech at 150 words per minute (actually slept
  through in real wall-clock time); it does not use recorded speech.
- `fixed_early_proxy` used the repository's documented timing proxy, not a trained
  or live pause forecaster.
- Generation used `temperature: 0.0` for the answer path with per-run `seed`.
- `immediate` and `sentence_end` share the same median (78 ms) in this run. Their
  quartiles differ, but the distributions overlap heavily; whether the two are
  meaningfully separable in this setup should not be asserted from the median alone.
- Six runs launched retrieval >1000 ms after user end, two of them >2000 ms under
  early-launch policies (scenarios 38 and 40, one run each). These are outliers
  against the intended behavior of those policies and are not yet explained.
- Both early policies trigger from a short partial transcript, so answer accuracy
  must be evaluated alongside TTFA rather than in isolation.
