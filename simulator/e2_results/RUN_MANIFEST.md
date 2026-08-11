# Experiment 2 Modal Run Manifest

This directory contains the result manifests for the full Experiment 2 timing sweep. The raw event logs referenced by each result record are versioned under `logs/` in the same branch.

## Run provenance

- Run window: 2026-08-11 to 2026-08-12 (Asia/Calcutta)
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Serving: Modal-hosted vLLM on one NVIDIA A10, unquantized `bfloat16`
- Timing policies: `sentence_end`, `immediate`, `predicted_pause`
- Seeds: `0, 1, 2, 3, 4`
- Scenarios per policy/seed: 50
- Total scenario runs: 750
- Successful runs: 750
- Failed runs: 0
- Retrieval decision: frozen from each scenario's `retrieval_needed` label
- Command: `python -m simulator.batch_runner`

The local environment selected the real vLLM backend and set `VOICE_RAG_MODEL=mistral7b_v03`. Endpoint URLs and credentials are intentionally not committed.

## Artifacts

- `simulator/e2_results/e2_<policy>_seed<seed>.json`: one result manifest per policy/seed combination.
- `logs/conv_<policy>_s<seed>_<scenario>.jsonl`: raw event stream for each scenario run, including timestamps, retrieval mechanism fields, generated answers, and token log probabilities.
- `logs/e2_modal_full_20260811.console.log`: complete console output from the sweep.

Each result record's `log_path` points to its corresponding raw JSONL file.

## Important analysis notes

- The text-input path models user speech at 150 words per minute; it does not use recorded speech.
- Piper was unavailable during this run, so TTS used the pipeline's synthetic-tone fallback. `tts_first_audio` still marks the simulated playback boundary.
- `predicted_pause` used the repository's documented timing proxy, not a trained or live pause forecaster.
- Both early policies triggered from a one-word partial transcript in this dataset. Answer accuracy must therefore be evaluated alongside TTFA.
- Policies were executed in blocks in this order: `sentence_end`, `immediate`, `predicted_pause`.
- `logs/conv_sentence_end_s0_001.jsonl` was opened in append mode after pre-run smoke attempts. For the full sweep observation, select the final complete event segment ending at the last `tts_first_audio` event. All other referenced JSONL files contain one completed sweep run.
