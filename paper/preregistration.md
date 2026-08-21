# Pre-Registration — Voice-RAG Controller Experiments

This document records what each experiment measures and what counts as success or failure, 
decided BEFORE running them. We do not change success criteria after seeing results. 
Owner: Yashmitha. Sign-off: Ryan.

Status: DRAFT — remaining criteria for later experiments are still to be confirmed with
the team. Experiment 1 is FINAL/FROZEN. Its model and dataset were frozen on
2026-08-10 before any real data collection, and Ryan confirmed on 2026-08-11 that
Experiment 1 should remain final before the first run began.

## Guiding principle
The project claims three live decisions (when to retrieve, what to remember, when to speak) 
are better made by one unified controller than three separate rules. Each experiment below 
tests one assumption behind that claim. All outcomes, including failures, will be reported.

## Experiment 1 — Signal validity (the gatekeeper)
**Status: FINAL / FROZEN**

Confirmed by Ryan on 2026-08-11, before the first experimental run. The success
criterion, kill criterion, model, datasets, and seeds below must not be changed after
results are observed.
- Question: When the language model is uncertain (measured from its confidence numbers / 
  logprobs), does that reliably indicate it needs to retrieve a document?
- How: Offline, over questions with known answers and known "was retrieval needed" labels. 
  Score each candidate signal with AUC (0.5 = useless, 1.0 = perfect).
- Success: best signal AUC >= 0.7.
- Kill criterion: best signal AUC < 0.6 -> pivot away from uncertainty-triggered retrieval 
  to a backup approach (draft an answer, verify it, retrieve only if verification fails).
- Runs first; gates the rest.

### Frozen before collection (2026-08-10)
- Model: ~~`meta-llama/Llama-3.1-8B-Instruct`~~ → **`mistralai/Mistral-7B-Instruct-v0.3`**
  (see the deviation note below), served unquantized at `--dtype bfloat16`. 
  Reported on this one model only — the other entries in configs/config.yaml are 
  fallbacks for a pending gated download, not a comparison set. Quantized serving is 
  excluded because the per-token logprobs are the measurement instrument and 
  compression shifts the distribution being measured.

#### Recorded deviation — model substitution (logged 2026-08-16)
The frozen model above was `meta-llama/Llama-3.1-8B-Instruct`. Every real run
reported by this project — Experiment 1's gate run, Experiment 2's 750-run timing
sweep, and Experiment 3's belief-signal precompute — was served with
`mistralai/Mistral-7B-Instruct-v0.3` instead.

This is the fallback path this section already anticipated ("fallbacks for a pending
gated download"), taken because the Llama-3.1 gated download had not been approved
when collection started. It is logged here rather than left implicit because this
section also requires the active model be named in this document, and until now it
named the wrong one.

- Substitution decided before Experiment 1 collection, not after seeing results; the
  AUC thresholds (success 0.7 / kill 0.6) were not touched.
- All three experiments use the SAME model, so the substitution is not a confound
  across experiments — E1's calibration and E3's signals are on one checkpoint.
- Verifiable in the artifacts, which record the served model:
  `controller/baselines/e1_calibration.json` (`provenance.model`),
  `simulator/e2_results/RUN_MANIFEST.md`, `configs/e3_price_grid.json`
  (`signals.model_hf_id`).
- 7B not 8B, so any parameter-count-sensitive claim must say Mistral-7B, never
  "an 8B model".

Reported results name Mistral-7B. No result on Llama-3.1-8B exists.
- Primary gate dataset: `llm/experiment1/dataset.jsonl`, 36 examples, balanced 18/18. 
  The AUC thresholds above apply to THIS dataset and no other.
- Seeds: the five in `llm/experiment1/config.json` (11, 23, 37, 53, 71).
- Secondary analysis (NOT the gate): `llm/experiment1/dataset_extended.jsonl`, 120 
  examples — 18 hotel positives, 42 Spoken-SQuAD positives (alinet/spoken_squad, 
  default config), 60 authored negatives. Built to address a known weakness in the 
  36-item set, where positives all name a fictional hotel and negatives are 
  primary-school arithmetic, leaving the classes separable by surface form alone. 
  Reported alongside the gate result, never in place of it. Applying the frozen 
  thresholds to this set would require re-freezing here first.
- Both runs record the dataset sha256, so which set produced which number is checkable.
- Known limitation, stated in advance: at 18 vs 18 the standard error on AUC is roughly 
  ±0.09, so a result near 0.7 will not be decisive. Spread across the five seeds measures 
  generation variance on fixed questions, not sampling variance over questions, and will 
  not be reported as a confidence interval on AUC.

## Experiment 2 — Timing value
- Question: Does timing retrievals to happen while the user is still talking hide the delay?
- How: Fix the trigger and retrieval budget; vary only WHEN retrieval fires — reactive vs 
  at sentence boundaries vs during predicted user-still-talking windows. Compare TTFA (time 
  to first audio) at matched answer accuracy. 5 seeds, paired bootstrap for intervals.
- Success: forecast-timed retrieval gives lower p95 TTFA at equal accuracy.
- Failure: no timing benefit -> "timing is decoration"; contribution shrinks to trigger 
  quality.

## Experiment 3 — Coupling value (the headline)
- Question: Does one unified controller beat well-tuned separate controllers?
- How: Compare our unified controller against two rivals we build and tune equally: 
  (A) three independent best controllers sharing nothing; (B) the same three given a shared 
  latency/compute price but no shared state. Compare as accuracy-vs-latency curves across a 
  grid of settings. 5 seeds.
- Success (pre-committed): our curve clearly beats rival B over a meaningful region, with 
  non-overlapping confidence bands, in at least 3 of 5 seeds. Ties are reported as ties.
- Three honest outcomes, all publishable: unified wins (thesis confirmed); rival B ties 
  ("shared prices suffice, full unification unnecessary"); nothing beats simple heuristics 
  ("coupling intuition was wrong") + we publish the benchmark.

## Experiment 4 — Memory scheduling (CUT for this deadline)
- Deferring memory operations to idle windows, tested for latency benefit at equal recall. 
  Cut from scope for the current deadline (per our Risk-4 scope rule). Memory still runs 
  (simple size-triggered) but we make no scheduling claims.

## Quality bars (never cut, apply to all experiments)
- 5 seeds per reported result.
- Every baseline gets the same tuning effort as our method.
- Success criteria frozen before data exists.
- TTFA measured at the audio boundary only.
- Every figure regenerable from raw logs by one command.

### Implementation note

The policy pre-registered as `forecast-timed` was implemented for E2 as
`fixed_early_proxy`.

`fixed_early_proxy` is not a live forecast or pause-prediction policy. It
dispatches retrieval at a fixed early point in the modeled speech timeline and
therefore serves as a proxy for early/overlapped retrieval placement.

True forecast timing using a live VAD/VAP or pause-prediction signal was not
implemented for this experiment and is left as future work.

This note documents the implementation mapping and does not alter the frozen
pre-registered success criterion.
