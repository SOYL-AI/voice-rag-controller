# Ryan — Phase 2 Execution Plan

Branch: `feature/phase2-signal-experiment-1`

Baseline: `main` at `4618840` after the requested one-time pull

Branch review snapshot: 2026-08-08

## Scope boundary

Ryan owns signal instrumentation and Experiment 1. Changes on this branch
stay under `llm/`. The branch does not edit the shared pipeline, controller,
event schema, retrieval implementation, harness, audio, or paper records.

## Dependency map from the other branches

- Renya's `phase2-integration`, `phase2-controller-rule`, and
  `phase2-readmes` are already merged into `main`. The real retrieval and LLM
  calls are therefore available as the integration baseline.
- Lakshya's `phase2-retrieval` is one commit ahead of `main`. It adds query
  rewriting, public-dataset loaders, and simulator scenarios. Ryan can run the
  local balanced seed audit now; the public benchmark extension waits for
  that branch to be reviewed/merged.
- Shubhankit's `feature/dead-air-metric` is separate Experiment 2 work and is
  based behind current `main`. It is not a dependency for Experiment 1.
- Audio and Phase 1 documentation branches are read-only context for Ryan and
  are not touched here.

## Execution order

1. **Instrument model signals.** Preserve chosen-token logprob, probability,
   top-k alternatives, entropy, and answer-level uncertainty summaries in the
   existing `llm_first_token` / `llm_final` payloads.
2. **Expose belief signals.** Provide tracker unknown-slot uncertainty and a
   versioned snapshot that combines model, retrieval-score, and tracker data.
3. **Freeze the seed audit.** Keep an answer-bearing balanced label set,
   five seeds, candidate signals, and the pre-registered 0.7 pass / 0.6 pivot
   thresholds in `llm/experiment1/`.
4. **Run the local smoke experiment.** Validate collection, deterministic
   mock data, AUC direction/ties, serialization, and invalid-for-gate marking.
5. **Extend with public data.** After `phase2-retrieval` lands, validate its
   SpokenWOZ/Spoken-SQuAD examples and add only examples with usable answers
   and defensible retrieval-needed labels. Do not change thresholds after
   looking at AUC.
6. **Run the real gate.** On the unquantized ~8B vLLM GPU host, run the same
   command/config. Review samples, then hand the signed result JSON to
   Yashmitha for the lab log. PASS is best mean AUC >= 0.7, PIVOT is < 0.6,
   and the middle band is INCONCLUSIVE.
7. **Integration handoff.** Give Renya the side-effect-free signal snapshot
   contract. Renya owns approving an event-schema home and wiring it into the
   shared controller belief; Ryan does not change those files unilaterally.

## Current result

The five-seed, 36-example mock run completed and is stored at
`experiment1/results/smoke_mock.json`. It is correctly marked
`NOT_EVALUATED_SMOKE`; synthetic logprobs are plumbing evidence only. The real
gate remains pending a healthy unquantized vLLM host with sufficient VRAM.
