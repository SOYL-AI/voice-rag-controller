# Pre-Registration — Voice-RAG Controller Experiments

This document records what each experiment measures and what counts as success or failure, 
decided BEFORE running them. We do not change success criteria after seeing results. 
Owner: Yashmita. Sign-off: Ryan.

Status: FROZEN — methods, metrics, decision criteria, and price-grid construction
are finalized. Experiment outcomes are recorded separately in the results compilation
and lab log.

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

### Status

FROZEN — E3 methods, metrics, decision rule, tuning protocol, and price-grid
construction are finalized. Experiment outcomes are recorded separately in
`results.md` and `lab_log.md`.

### Question

Does one unified controller beat well-tuned separate controllers?

### Primary Comparison

The headline comparison is:

- **B9 — decomposed-with-prices:** retrieval trigger, timing, and memory each
  separately optimize against the same latency and compute prices, with no
  shared state between the components.

- **Ours — unified controller:** one controller reasons jointly about retrieval,
  timing, and memory under the same objective.

B9 vs Ours is the primary comparison because both use the same prices and the
same scalarization objective. The difference being tested is decomposed versus
joint decision-making.

B8 (`independent-best`) is also reported as the strongest unpriced rival.

### Scalarization Objective

`quality − lambda_l * latency − lambda_c * compute`

B9 and Ours are swept across the same `(lambda_l, lambda_c)` price grid to
produce their trade-off frontiers.

### Primary Metric

Frontier dominance.

The harness constructs:

- accuracy vs p95 TTFA frontiers
- accuracy vs compute frontiers

for each `(baseline, price point)`.

B0-B7 appear as individual points. B9 and Ours appear as swept price-point
frontiers.

### Pre-Committed Decision Rule

Success requires the unified controller's frontier to clearly beat B9's
frontier over a meaningful region, with non-overlapping bootstrap confidence
bands, in at least 3 of 5 seeds.

Ties are reported as ties.

The three pre-committed outcomes are:

1. **Unified wins:** coupling hypothesis supported.
2. **B9 ties:** shared prices are sufficient and full unification is not
   necessary.
3. **Neither beats simple heuristics:** coupling hypothesis is not supported;
   the benchmark and negative result are still reported.

### Lambda Grid

The E3 sweep uses a fixed price grid:

- `lambda_c = 0.05`
- 26 fixed `lambda_l` points spanning approximately 0.022 to 1.713
- The `lambda_l` points were derived by tiling the empirical
  value-of-information distribution rather than selecting arbitrary values.

This is the same grid used in the completed E3 run.

**Price-grid construction confirmed by:** Renya

### Tuning Protocol

All baselines receive the same tuning budget on the same development split.

Thresholds are swept rather than left at default values. No baseline may be
used as an untuned strawman.

B8, B9, and Ours receive the same tuning treatment as the other baselines.

### Seeds

5 seeds per reported result.

### Metrics Recorded

**Quality**
- Task success
- Answer accuracy against gold
- Hit@3 retrieval accuracy

**Latency**
- p50 TTFA
- p95 TTFA
- Dead-air rate

**Compute**
- Retrievals per dialogue
- Wasted-retrieval rate
- Prompt tokens per turn
- Controller overhead per decision

### Confidence Analysis

Bootstrap confidence bands are used for the frontier comparison.

### Sign-Off

**Overall experiment sign-off:** Ryan  
**Price-grid confirmation:** Renya  
**Status:** FROZEN  
**Freeze date:** 2026-08-21

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

### Operational Uncertainty Signal

The operational uncertainty signal used by the controller is
`mean_entropy_nats`.

Experiment 1's strongest gate result was obtained with `max_entropy_nats`.
Both signals exceeded the pre-registered AUC success threshold of 0.70.

The distinction is retained explicitly: `max_entropy_nats` was the strongest
E1 gate signal, while `mean_entropy_nats` is the signal committed for
downstream controller operation.

### Answer Accuracy Metric

Answer correctness is evaluated using the Azure GPT-4o judge.

- Temperature: 0
- Deterministic evaluation configuration
- Used as the locked `answer_correct` metric for the E3 sweep
- Configuration verified for reproducibility

### Methodological Disclosures

1. **Operational signal vs. E1 best signal**

   Experiment 1's strongest AUC was obtained with `max_entropy_nats`, while
   the downstream controller uses `mean_entropy_nats` as its operational
   uncertainty signal. Both exceeded the frozen E1 success threshold of 0.70.

2. **Waste-rate term**

   `waste_rate` appears in the compute/cost formulation as an unmeasured
   placeholder. It is not treated as an empirically measured constant in the
   current experiments.

3. **E1 dataset limitation**

   The primary E1 gate dataset has a known surface-form separability issue,
   already identified in the preregistration: retrieval-positive examples are
   hotel-specific while many negatives are simple general-knowledge or
   arithmetic questions. The gate result is reported with this limitation.

### Final Sign-Off

- **Document owner:** Yashmitha
- **Overall pre-registration sign-off:** Ryan
- **E2 implementation mapping confirmation:** Renya
- **E3 price-grid construction confirmation:** Renya
- **Status:** FROZEN
- **Freeze date:** 2026-08-21
