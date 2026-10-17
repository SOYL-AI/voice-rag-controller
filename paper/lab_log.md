# Voice-RAG Controller — Lab Log

This file records important project decisions, scope changes, tool choices,
experiment decisions, and blockers.

## Decision Log

### 2026-08-06 — Phase 1 documentation started

**Decision:** Created the project lab log for recording technical and research decisions.

**Reason:** The team needs a dated record explaining why important choices were made.

**Decided by:** Project team

**Recorded by:** Yashmita Sudhir

### 2026-08-11 — Experiment 1 finalized and first run started

**Decision:** Experiment 1 remains final/frozen before results are observed.

**Confirmed by:** Ryan

**Run status:** Ryan began the first Experiment 1 run.

**Blocker:** GPU issues delayed experiment execution.

**Impact:** No changes will be made to the frozen Experiment 1 criteria as a result of
execution issues.

**Recorded by:** Yashmita

### 2026-08-15 — Experiment 1 result recorded

**Experiment:** Signal validity

**Pre-registered success criterion:** Best signal AUC >= 0.70.

**Observed result:** `max_entropy_nats` was the best signal with a mean AUC of
0.9784 across five seeds.

**Verdict:** PASS.

**Impact:** Experiment 1 passed its pre-registered AUC gate, allowing the team
to proceed to the later experiments.

**Documentation issue identified:** The Experiment 1 result artifact reports
`mistralai/Mistral-7B-Instruct-v0.3`, while the current pre-registration lists
`meta-llama/Llama-3.1-8B-Instruct`. This discrepancy was subsequently resolved
and recorded on 2026-08-16.

**Recorded by:** Yashmita

### 2026-08-15 — Experiment 2 timing-value result finalized

**Experiment:** Timing value

**Implementation:** The pre-registered forecast-timed condition was implemented
as `fixed_early_proxy`, a fixed early point in the modeled speech timeline
rather than a live forecasting or pause-prediction policy.

**Accuracy:** Hit@3 = 100% for all three timing policies.

**p95 TTFA:**
- `fixed_early_proxy`: 7089.90 ms
- `immediate`: 6915 ms
- `sentence_end`: 7051.70 ms

**Pre-registered verdict:** NOT MET. `fixed_early_proxy` had a p95 TTFA
38.20 ms higher than `sentence_end`.

**Secondary paired result:** `fixed_early_proxy` was 479.36 ms faster in mean
TTFA than `sentence_end`; the paired-bootstrap 95% confidence interval excluded
zero.

Against `immediate`, the improvement was approximately 180 ms, but the
confidence interval crossed zero and was not statistically significant.

**Mechanism check:** Speech overlap correlated with lower TTFA
(Pearson r = -0.25).

**Interpretation:** The pre-registered p95 criterion was not met. The paired
analysis is retained as a secondary finding showing that earlier retrieval
placement can reduce average latency relative to sentence-end retrieval.

**Limitation:** E2 tested a fixed early proxy rather than true live
forecast-timing. VAD/VAP-based prediction remains future work.

**Recorded by:** Yashmita

### 2026-08-16 — Model substitution logged against the frozen pre-registration

**Decision:** Recorded in `paper/preregistration.md` that every real run
(E1 gate, E2 750-run sweep, E3 signal precompute) was served with
`mistralai/Mistral-7B-Instruct-v0.3`, not the frozen
`meta-llama/Llama-3.1-8B-Instruct`. Reported results name Mistral-7B; no
Llama-3.1-8B result exists.

**Reason:** The prereg anticipated this fallback (the Llama gated download was
still pending at collection time) but also requires the active model be named
in that document, and it still named Llama. The substitution predates data
collection and no threshold was altered, so this is a recording gap, not a
changed criterion — but it has to be visible in the document a reviewer reads,
not only in the run artifacts.

**Decided by:** Project team

**Recorded by:** Ryan Gomez

### 2026-08-16 — E2 reported against its pre-registered p95 criterion

**Decision:** `harness/e2_bootstrap.py` now evaluates E2's pre-registered
criterion explicitly and prints MET / NOT MET. On the 750-run sweep the verdict
is **NOT MET**: `fixed_early_proxy` p95 TTFA is 7089.90 ms against
`sentence_end` at 7051.70 ms (+38.20 ms, i.e. higher). The paired-mean result
(−479.36 ms, CI excluding 0) is retained and reported as a secondary finding.

**Reason:** The script previously described p95 as reported "for
completeness/pre-registration, not as the primary evidence" and directed the
reader to the paired-mean test instead. That is a post-hoc substitution of the
success criterion, which the prereg's guiding principle forbids, and it matters
more than usual here because the two tests disagree in direction rather than
only in significance. The power argument for why p95 is noisy at 50 clusters is
sound and is kept — as an explanation accompanying a failed criterion, not as
grounds for swapping which test counts.

**Decided by:** Project team

**Recorded by:** Ryan Gomez

### 2026-08-19 — E3 full sweep complete; state of all three results

**Decision:** Recording where every experiment stands against its
pre-registered criterion, now that E3's headline run is in. Two of the three
criteria are NOT MET and the third carries a caveat. All are reported as they
fell, per the prereg's guiding principle.

| experiment | pre-registered criterion | result |
| --- | --- | --- |
| E1 | best signal AUC >= 0.7 | **PASS** — 0.9784 (`max_entropy_nats`) |
| E2 | forecast-timed gives lower **p95** TTFA | **NOT MET** — 7089.90 ms vs 7051.70 ms baseline |
| E3 | Ours beats B9 in >= 3 of 5 seeds | **NOT MET** — 0 of 5 seeds, 0 dominated points of 26 |

**E3 — the headline.** 1500 Phase A cells on the Mistral-7B Modal endpoint,
500 answers, 0 failed cells, 14000 Phase B records. Zero dominated operating
points in every seed. This is the prereg's second honest outcome, "rival B
ties: shared prices suffice, full unification unnecessary", and it is reported
as a tie because that document says ties are reported as ties.

Why it ties is the substantive finding, and it is not "no effect":

- At 15 of 26 prices Ours and B9 make IDENTICAL decisions. The divergence band
  is empty there, so the frontier gap is exactly zero by construction rather
  than by measurement.
- Where they differ in the middle of the grid Ours is slightly WORSE (-0.020
  accuracy at five prices). Its lower retrieve threshold buys extra retrievals,
  and those land on negatives the belief signal scores falsely high. Joint
  reasoning inherits the belief's errors — coupling's value scales with belief
  quality, which is a mechanism with a boundary condition, not a null.
- Ours wins only at the top two prices, and at lambda_l=1.7127 B9 retrieves
  0 of 50: its cut has passed the maximum achievable VoI, so it is disabled
  rather than beaten. That point should not be claimed as a win.

Retrieval itself works. B0 (never retrieve) scores 0.40-0.42 accuracy against
0.62-0.71 for the priced controllers. What failed is the coupling hypothesis,
not the pipeline.

**E1 — passed, with two things the paper must state.** The gate ran on
`llm/experiment1/dataset.jsonl`, the 36-item set the prereg itself describes as
"separable by surface form alone". `dataset_extended.jsonl` — the 120-item
harder check with authored negatives — is still not built, so that known
weakness remains unmeasured. Separately, the gate was won by
`max_entropy_nats` (0.9784) while the system now runs on `mean_entropy_nats`
(0.9253 on E1). Both clear the 0.70 bar so the gate holds either way, but the
signal that passed the gate and the signal being reported are not the same one.

**The thread connecting all three.** Every measurement in this project
saturates or collapses, and each time the cause is the evaluation substrate
rather than the system. E1's classes are separable by surface form. E2's hit@3
is at a 100% ceiling, so "at equal accuracy" is satisfied trivially. E3's VoI
distribution is bimodal with an empty middle, and the 20 borderline scenarios
authored to fill it all landed in the high cluster because `max_entropy_nats`
is near-binary by construction. The controllers, harness and statistics are the
strongest part of the work.

**Also needs attention: b4 is degenerate.** It retrieves on 0.00% of turns and
its operating point is identical to b0 on accuracy, p95 TTFA and compute in all
five seeds. It is not currently a distinct baseline and should not be reported
as one until its trigger is examined.

**Judge replaced and validated before grading.** The Azure gpt-4o judge had an
8-token budget and no room to reason, so it decided borderline answers on
surface form: an answer identical to its reference except for a leading phrase
scored NO 5/5 and YES 5/5 with the phrase removed. Scored against 19 hand
labels (`simulator/judge_validation/`): terse 74%, terse+strip 79%,
reasoned+strip 95%. Every error under every variant was marking a CORRECT
answer wrong — the direction that under-credits retrieval and so penalises
whichever baseline retrieves most. All 500 answers were re-scored under the
validated judge before Phase B: 456 unchanged, 34 flipped to correct, 10 to
incorrect.

**Reason for recording all this together:** the three results are usually read
one at a time, and read that way E3 looks like an isolated null. Read together
they point at one cause, which changes what should be done next and what the
paper can honestly claim.

**Cheapest open lever:** the extended E1 run. It is the only unexercised check,
roughly 20 minutes of endpoint time, and it either shores up the gate or
establishes that the signal was surface form — both of which are worth knowing
before results are written.

**Decided by:** Project team

**Recorded by:** Ryan Gomez

### 2026-08-21 — E3 pre-registration finalized

- Experiment 3 methods and decision criteria finalized.
- Primary comparison: B9 (`decomposed-with-prices`) vs Ours (`unified controller`), with B8 (`independent-best`) reported as the strongest unpriced rival.
- Primary metric: frontier dominance across accuracy vs p95 TTFA and accuracy vs compute.
- Pre-committed success rule remains dominance over B9 with non-overlapping bootstrap confidence bands in at least 3 of 5 seeds.
- E3 price grid finalized at `lambda_c = 0.05` with 26 fixed `lambda_l` points spanning approximately 0.022 to 1.713, derived by tiling the empirical value-of-information distribution.
- Operational uncertainty signal confirmed as `mean_entropy_nats`.
- Answer correctness for E3 is evaluated using the Azure GPT-4o judge at temperature 0 through the locked `answer_correct` metric.
- E3 numerical results are not compiled here yet; they will be added once the final result artifacts and exact numbers are available on `main`.

### 2026-10-16 — v2 primary design: the paper tests the commit/speculative boundary

**Decision:** The v2 pre-registration (draft, `docs/research_v2/PREREG_V2.md` §3) tests BOTH pipeline architectures on the same test items, price and latency profile:

- **COMMIT:** a dispatched retrieval is always used.
- **SPECULATIVE:** a dispatched retrieval may be discarded at end of turn.

There are three co-primary hypotheses:

- **P1:** the joint controller beats the strongest decomposed rival under COMMIT.
- **P2:** that advantage is larger under COMMIT than under SPECULATIVE (an interaction contrast).
- **P3:** under SPECULATIVE, the joint advantage over speculative prefetch is below the THEORY Prop. 4′ margin λ_c(1+w̄)q̂.

The boundary claim requires all three (an intersection–union test at one-sided α = 0.025).

**Reason:** THEORY.md shows that:

- joint and decomposed control are equivalent under A1–A3 (Prop. 1);
- turn-dependent overlap is a second-order effect (Prop. 3);
- the information-vs-overlap channel separates the two only when retrieval is committed (Prop. 4 / 4′).

Testing both architectures makes the result informative whichever way it falls, instead of staking the study on one side of the boundary.

**Not yet done before freeze:** extend the power analysis to P2/P3, fix m on dev, and name D\*[arch].

**Decided by:** Ryan (PI)

**Recorded by:** Ryan Gomez

### 2026-10-15 — Boundary-test design corrected before freeze (follow-up to the entry above)

**Proposed change** (draft; PREREG_V2 is not frozen and no test data have been touched):

- **New decomposed rival.** **SPEC-θ** (`b11_speculative_gated_v2`) is added to D\*[SPECULATIVE]. SPEC-θ is belief-gated speculative prefetch: the dispatch time comes from the timing policy, and whether to prefetch is decided from the belief and dev-mean scalars only.
- **P3 replaced.** P3 is now a TOST equivalence test, |Δ[SPECULATIVE]| < m₃ with m₃ = ½·Δ[COMMIT] estimated on dev. It replaces the non-superiority margin λ_c(1+w̄)q̂.
- **Price rule.** The primary price is now chosen to maximize conjunction power.

**Reasons** (THEORY.md §6.2; `analysis/v2_power.py`, block `boundary_test`):

1. **λ_c(1+w̄)q̂ is not a bound** on the joint speculative controller's advantage over SPEC. There is an analytic counterexample, and the bound is exceeded in simulation in 4 of 6 cells at the lowest price. With partial-query risk active, no compute-only bound exists at all.
2. **Without SPEC-θ, the speculative contrast measures a decomposable effect.** The joint speculative controller beats {B9+, SPEC} mainly by not prefetching on confident-skip turns, and a decomposed controller can do that from the belief alone. In simulation this makes Δ[SPECULATIVE] (0.013–0.015) larger than Δ[COMMIT] (0.009–0.011), so P2 is predicted false and the conjunction cannot pass at any N. With SPEC-θ, Δ[SPECULATIVE] ≈ 0 and the interaction ≈ Δ[COMMIT].
3. **The q̂ margin is uninformative.** It is about 3× Δ[COMMIT], so P3 would follow from P2.

**Power** (placeholder assumptions): 80% conjunction power needs about 3,400 (faiss), 4,600 (faiss_rerank) or 6,500 (remote_api) pooled test items. A direct bootstrap check suggests adding about 10%.

**Implementation:**

- `boundary_contrasts` now reports m₃. On test, m₃ must be passed in, as D\* is.
- q̂ and λ_c(1+w̄)q̂ are kept as labelled diagnostics.
- `tests/test_v2_speculative.py` checks SPEC ≤ SPEC-θ ≤ Ours-v2[SPEC] in exact expectation. It also checks that SPEC-θ strictly beats SPEC when λ_c > 0, and that its gate ignores the turn's timing state.

**Needs PI sign-off before freeze.**

**Recorded by:** methods/theory workstream

### 2026-10-16 — E1-v2 dev results; proposed operational belief and judge rubric (pending PI sign-off)

**Evidence (dev only; the test split is sealed):**

- **d2_support** (222 dev items):
  - Closed-book token entropy is at chance as a retrieval-need signal: mean entropy has AUC 0.55 [0.47, 0.65] on the full question against r3 labels. In the support-agent persona, Mistral-7B states fabricated org policies fluently.
  - A self-assessment probe (frozen wording `sa-v1-frozen`: "answerable from general knowledge alone, without {org}'s own policies…? Yes/No", reading P(No)) gives AUC 0.55 / 0.58 / 0.62 / 0.70 at word fraction 0.3 / 0.5 / 0.7 / 1.0, and 0.78 against the authored labels.
- **d1_open** (1,566 dev items; alias-match labels, need 877/1566):
  - mean entropy: 0.44 / 0.43 / 0.57 / 0.70;
  - self-assessed need: 0.51 / 0.56 / 0.60 / 0.69.
- **Combined belief** (logistic on entropy + self-assessment): +0.02 on d1's full question, worse at every prefix, no gain on d2.
- **Latency of each belief:** self-assessment 0.55 s; first-16-token entropy 1.10 s; full-answer entropy 1.9–3.0 s.

**Proposed:** the operational belief AND the E1-v2 gate signal is `self_assessed_need`, calibrated on dev. One signal is used for both, which avoids v1's gate-signal vs operational-signal mismatch. The combined belief is reported as exploratory.

**Rubric:** judge rubric r3 (core facts) is final. It is the second and last dev revision allowed by PREREG §2.4.

- Judge κ: 0.525 under r1, 0.704 under r2, 0.737 under r3.
- Authored-vs-empirical κ under r3: 0.424.
- The core facts were annotated blind to model outputs.

**Status:** pending PI sign-off; to be written into PREREG_V2 before the freeze.

**Recorded by:** Ryan Gomez

### 2026-10-16 — E3-v2 dev: an artifact in the first result, the lookahead flaw, two pre-freeze fixes, and the single rerun

**Status:** dev only; the test split is sealed and no test outcome has been seen. Nothing is frozen. Decisions by the PI; recorded by Ryan Gomez.

**1. The first dev result was an artifact.** The first dev run (`docs/research_v2/v2_predictions_pre_lookahead_fix.json`) showed Ours-v2 beating B9+ under commit on d2_support ASR by +0.10 to +0.14 utility. That "win" came from a gap in B9+'s pre-registered timing family (early / late / R̂-threshold). The family had no rule that dispatches *late in the utterance*, where the end-of-turn gap still hides retrieval. Ours-v2 could do that and B9+ could not. A diagnostic run that added an R̂-below rule to the family (`v2_predictions_b9plus_extended.json`) removed the d2 ASR advantage (≈ 0). The pooled Δ[COMMIT] became −0.034 to −0.055.

**2. The lookahead flaw.** In the diagnostic run, Ours-v2 lost because its lookahead assumed the next partial was certain whenever the point R̂ exceeded the nominal partial interval. So it kept waiting for partials that did not come before the end-of-turn decision. There it priced the full retrieval latency and usually skipped:
- d1 ASR: 550 of 1,566 turns skipped at the end;
- d2 text150: early dispatch on only 11 of 222 turns.

**3. The two fixes (PI decision, before freeze, applied once).**
- **(a) R̂-below enters the pre-registered B9+ family and the shared SPEC / SPEC-θ trigger family.**
  - It uses a 21-point dev quantile grid for each.
  - For SPEC-θ, the gate scalars are the dev means of remaining and elapsed speech at the firing step.
  - The opt-in flag is removed. PREREG_V2 §4.5 is updated.
- **(b) Ours-v2's lookahead is corrected once, in both architectures.**
  - The value of waiting is p·(next-partial value) + (1−p)·(end-of-turn decision value).
  - p comes from the remaining-speech predictor's q10/q50/q90, aged to the step's availability time.
  - The step spacing is the dev-median availability spacing between partials, which includes the ASR lag.
  - No other change was made to Ours-v2, and no parameter was chosen by looking at Δ on dev.
  - Unit tests (`tests/test_v2_lookahead_fix.py`) cover three cases: no further partial certain; further partial certain, which recovers the old recursion; and the mixed case.
- **Also done:** the Phase A cells for the 5 re-traced dev items were topped up (d1_open-popqa-2474729; d2_support-03210, -03215, -06201, -10123). Newly needed d2 cells were lazy-graded: 59 primary-judge calls, about ₹9.

**4. Pre-commitment.** Before the rerun, the PI committed to freezing after this single rerun, whatever it showed.

**5. Result of the single rerun (`docs/research_v2/v2_predictions.json`).** Primary (ASR) condition, belief quantiles 0.5 / 0.65 / 0.8 / 0.9, rewrite_faiss:

| contrast | 0.5 | 0.65 | 0.8 | 0.9 | cluster SE |
|---|---|---|---|---|---|
| d1_open Δ[COMMIT] | −0.040 | −0.049 | −0.051 | −0.058 | ≈ 0.010 |
| d2_support Δ[COMMIT] | −0.021 | −0.017 | −0.013 | −0.009 | ≈ 0.017–0.027 |
| pooled Δ[COMMIT] (all three profiles) | −0.036 to −0.037 | −0.044 to −0.045 | −0.046 | −0.051 to −0.052 | ≈ 0.009–0.010 |

- **Interaction:** within ±0.003 everywhere.
- **text150:** d1 ≈ 0; d2 −0.085 to −0.107.
- **Power** (`docs/research_v2/v2_power_dev.json`, `analysis/v2_power_dev.py`, from the dev per-item paired differences, test N = 3,734 + 498): power of P1 and of the conjunction is 0 in every profile and at every price, because dev Δ[COMMIT] < 0. The minimum detectable Δ[COMMIT] at that N is 0.016–0.018 (P2: 0.004–0.005).

**6. Finding after the rerun: the fix in (3b) was implemented in a way that makes it inert on dev.** It was found while diagnosing the rerun, after the dev results had been seen.
- **How p was implemented.** p was computed as P(R + end gap > spacing), with the aged R̂ quantiles clipped at 0.
- **What that does on dev.** On every dev condition the end gap is at least the spacing:
  - d1 ASR: 1,090 vs 421 ms;
  - d2 ASR: 1,731 vs 433 ms;
  - text150: 800 vs 400 / 1,200 ms.

  So p ≈ 1.00 at every partial step, including the last one (a descriptive check over all dev turns). The mixture never takes its end-of-turn branch, and the rerun mostly reflects fix (3a), not fix (3b).
- **Two structural facts behind it:**
  - In the ASR traces partials keep arriving after speech end (about 2.5 per turn on d1, 3.6 on d2). But once R̂ is clipped at 0, the predicted time to the end of turn stops shrinking.
  - The text150 timeline has exactly three prefixes, so no partial ever follows the 0.7 prefix.
- **Behaviour on d2 ASR.** Ours-v2 now dispatches at a partial on 182 of 222 turns (B9+: 199).
- **Behaviour on d1 ASR** (rewrite_faiss, q = 0.65, seed 11):
  - Ours-v2 retrieves on 764 of 1,566 turns (353 at a partial, 411 at the end) and skips 802.
  - B9+ retrieves on 1,406 turns, 1,309 of them at its R̂-below partial.

**Whether to correct this implementation and rerun is the PI's decision.** It would be a second rerun, against the pre-commitment in (4). Nothing has been changed or rerun since the finding.

### 2026-10-16 — E3-v2 dev: the lookahead correction re-implemented once (code defect), second and final dev rerun

**Status:** dev only; the test split is sealed and no test outcome has been seen. Nothing is frozen and no profile or price has been chosen. Decision by the PI; recorded by Ryan Gomez.

**Why a second correction.** It is justified by a verifiable code-level defect, not by Δ. The entry above (§6) shows that the first implementation of the lookahead correction computed p = P(R + end gap > spacing), with the aged R̂ quantiles clipped at 0. On every dev condition the end gap is at least the spacing, so p ≈ 1.00 at every partial step and the specified mixture never took its end-of-turn branch. The PI therefore had the *implementation* of the already-specified fix corrected, once, with no new tunable parameter:
1. **Information channel.** p is the probability that the next partial carries *new information*: p = P(remaining speech R > spacing).
   - R uses the predictor's q10/q50/q90, aged to the step's availability time and **not clipped**, so p → 0 after the predicted end of speech.
   - Post-speech re-transcription partials carry no information, so their belief transition is the identity. In the transition fit they are merged into the previous step (cut after the trace's speech end).
2. **Latency channel.** The end-of-turn gap enters only the exposure term, never p.
3. **text150.** The prefix schedule is deterministic: p = 1 while a prefix is still to come, and p = 0 after the last prefix.
4. **Scope.** The same correction applies in Ours-v2[SPECULATIVE], and through the shared option machinery in SPEC-θ (whose point scalars now give p = 1[R̄ > spacing]).

Tests (`tests/test_v2_lookahead_fix.py`) check that:
- p < 0.5 at the last pre-end partial of a synthetic ASR-like trace;
- post-speech partials give identity transitions and p = 0;
- p = 0 after text150's last prefix;
- the gap does not enter p.

**Diagnostic, reported and not used for any choice** (`docs/research_v2/v2_lookahead_p_diagnostic.json`). p at each turn's last partial:

| dataset × condition | mean p | share of turns with p < 0.5 | note |
|---|---|---|---|
| d1 ASR | 0.55 | 0.47 | 3,171 post-speech partials, mean p 0.58: the d1 remaining-speech predictor over-predicts remaining speech |
| d2 ASR | 0.23 | 0.86 | |
| d1 text150 | 0 | 1.00 | |
| d2 text150 | 0 | 1.00 | |

**Both dev results** (primary ASR condition; the first column is the rerun with r_below and the inert correction, `v2_predictions_post_rbelow_inert_fix.json`; the second is this rerun, `v2_predictions.json`):

| | rerun 1 | rerun 2 (final) |
|---|---|---|
| d1 ASR Δ[COMMIT], rewrite_faiss, q = 0.5…0.9 | −0.040 … −0.058 | −0.049, −0.045, −0.046, −0.053 (SE ≈ 0.009) |
| d2 ASR Δ[COMMIT], rewrite_faiss | −0.021 … −0.009 | −0.015, −0.009, −0.011, −0.007 (SE ≈ 0.018–0.021) |
| pooled Δ[COMMIT], all profiles | −0.036 … −0.052 | −0.040 … −0.048 (SE ≈ 0.009) |
| pooled interaction | within ±0.003 | +0.002 … +0.005 (SE ≈ 0.003) |
| d2 text150 Δ[COMMIT] | −0.085 … −0.107 | +0.006 … +0.010 (SE ≈ 0.018) |
| d1 text150 Δ[COMMIT] | ≈ 0 | −0.001 … −0.005 |

**Action counts** (rewrite_faiss, belief-quantile 0.65 price, seed 11, commit; Ours-v2 vs B9+), as retrieve-at-partial / retrieve-at-end / skip:

| dataset × condition | Ours-v2 | B9+ |
|---|---|---|
| d1 ASR | 1,541 / 9 / 16 | 1,309 / 97 / 160 |
| d2 ASR | 210 / 2 / 10 | 199 / 4 / 19 |
| d2 text150 | 222 / 0 / 0 | 196 / 7 / 19 |
| d1 text150 | 48 / 530 / 988 | 0 / 463 / 1,103 |

- **The end-of-turn branch now fires.** Ours-v2 no longer waits for the end of turn and then skips.
- **On d1 ASR it now retrieves on 99% of turns,** dispatching earlier than B9+ (median partial index 5 vs 6). Its accuracy is 0.438 against B9+'s 0.500, at a similar p95 TTFA.
- **The d1 ASR loss now comes from dispatching too early,** on partial queries that retrieve worse. It no longer comes from skipping.

**Power** (`docs/research_v2/v2_power_dev.json`, dev per-item paired differences, test N = 3,734 + 498):
- P1 and the conjunction have power 0 in every profile and at every price, because dev Δ[COMMIT] < 0.
- P2 alone has power 0.10–0.65.
- The minimum detectable Δ[COMMIT] at that N is 0.015–0.017 (P2: 0.006–0.007).

**Spend.** No new grounded answers were needed. gpt-6.1-sol: ₹8,338 spent of the ₹20,000 cap. gpt-4o: ₹3,094.

**Hard pre-commitment.** This is the last change to any controller before the freeze. The test split will be run with exactly the code frozen after this rerun, whatever its outcome. No further correction, tuning or rerun of Ours-v2 or any baseline will be made on the basis of these or any later dev numbers.

### 2026-10-16 — PREREG_V2 made freeze-ready (PI decisions, pre-freeze; nothing frozen)

**Status:** the controller code is frozen at `ddde67f`. The test split is sealed, and no test outcome has been seen.

**PI decisions written into `docs/research_v2/PREREG_V2.md`:**

1. **Primary cell: `rewrite_faiss` at the 0.65 dev belief-quantile price.**
   * Values: λ_l = 0.17305735434823866 (d1_open) and 0.2157823714003309 (d2_support); λ_c = 0.05.
   * Why: the §4.3 rule (maximum dev conjunction power) is degenerate, because power is 0 at every cell. Its fallback is written in now, and the choice was made without reference to Δ.
2. **P3 margin m₃ = 0.016**, a pre-specified smallest effect size of interest equal to the dev MDE at test N. It replaces ½·δ̂_C, which is 0 on dev.
3. **Rivals.**
   * D\*[COMMIT] = B9+.
   * D\*[SPECULATIVE] = B9+ for d1_open, and SPEC (`b10_speculative_v2`) for d2_support. For d2 this is the dev tuning's pick: it ties SPEC-θ at 0.35795, and the tie is broken by candidate order.
4. **P1–P3 and the IUT are unchanged.**
5. **H-REV added:** Δ[COMMIT] < 0 at the primary cell, pooled, one-sided α = 0.025. Dev: −0.041 (SE 0.0085). It sits in one Holm family with P1–P3 for individual claims.
6. **Headline:** Δ[COMMIT] with its 95% CI, per dataset and pooled, reported whatever the test outcomes.
7. **Judge.** gpt-6.1-sol 2026-09-29; k = 1; reasoning_effort medium; max_completion_tokens 2000; seed 20261015. Rubric e1v2-r3, sha256 `83fde728…a7ccbc23`. The GPT-4o 30% subsample uses seed 20261016.
8. **Freeze procedure.** Test generation starts right after the freeze. d2_support test grading waits for the human judge-validation gate.

**Removed before freeze** (not implementable in the frozen code): the channel-set ablation cells of the earlier draft.

**Open items:**
* the human judge gate;
* the d2 authored-label review;
* the max_completion_tokens truncation rate;
* a run-time check of the test guards;
* OSF;
* **the d1_open secondary judge.** The draft lists it, but the test budget has no d1 judge calls. The PI must decide before the freeze.

**Recorded by:** methods/theory workstream

### 2026-10-16 — Test run launched; deviation D1 (d2_support Phase A before labels) and note N1

**Status:** v2 pre-registration FROZEN (tag `prereg-v2-frozen`, frozen code `ddde67f`). Test generation started at 14:32 UTC per PREREG_V2 §9 step 7, on the frozen code. No test outcome has been seen, and no test answer has been sent to any judge. Recorded by Ryan Gomez; decisions by the PI.

- **D1 (approved by the PI).** `PhaseA.e1_closed_book` loaded `labels_<split>.json` unconditionally. For d2_support test, that file exists only after judge grading, which waits for the human judge-validation gate (§9 step 8). This blocked the d2_support Phase A store, the Phase B lazy plan and the required-cell generation.
  - **Change:** when the labels file is missing, no verdicts are attached. Reused skip cells stay unresolved until the store is rebuilt after gate-approved grading.
  - **Effect:** none on results. Decisions and the lazy plan never read verdicts; `tests/test_v2_deviation_d1.py` shows they are identical with and without verdicts.
  - Recorded in PREREG_V2 "Deviations".
- **N1 (accepted by the PI; not a deviation in method).** The frozen E1 runner cannot generate the full question alone. It runs its frozen default (`--no-asr`): text prefixes 0.3 / 0.5 / 0.7 plus the full question, at the frozen 80-token protocol cap.
  - The extra prefix records are not used by any pre-registered analysis.
  - Skip answers are E1's answers that ended on "stop", reused, or regenerated at 256 tokens where truncated, as on dev.
- **Grading on hold.** The d1_open 400-answer judge subsample (O6) also waits for the gate.

### 2026-10-17 — Test generation complete (PREREG_V2 §9 step 7); no judge call, no outcome analysis

**Status:** frozen code `ddde67f` plus deviation D1 (`43ca350`). Every stage ran from `simulator/e3v2_phase_a/test_logs/` (gitignored), in two resumable per-dataset pipelines. Recorded by Ryan Gomez.

| | d1_open | d2_support |
|---|---|---|
| E1 self-assessment calls | 45,590 | 8,704 |
| E1 closed-book calls (text prefixes + full, 80-token cap; N1) | 74,515 | 9,960 |
| rewrites | 25,900 | 8,397 |
| lazy plan (written before any grading) | 29,261 cells (21,793 retrieve, 7,468 skip) | 4,897 cells (3,901 retrieve, 996 skip) |
| plan sha256 | c49c44f7781a4ef5210724fbcc6e18ceb9b2f5d83dd8bfe63f8e4e1f1ed33a54 | f59b33bce622fb2a637da0f418d3ca3d05eae27f0bf80313a18a161e3b8d50fc |
| grounded answers | 103,560 | 18,045 |
| skip answers regenerated at 256 | 15,278 | 2,580 |
| grading | alias only (118,838 Phase A answers + E1 labels), done | none; waits for the judge gate |

- **Index path.** d1_open retrieval used this worktree's `retrieval/indexes/d1_open`, the index dev used (same corpus_sha256 724624b9…). d2_support used the main checkout's index, as on dev. Only the path argument changed; no code changed.
- **Duplicate run.** For about 6 seconds two d2 continuation processes ran the store stage together. The resulting `store_test.jsonl` was verified afterwards: 10,707 retrieval rows equal the 10,707 dispatch points, and 996 skip-prompt rows.
- **Judge spend.** The ledger is unchanged throughout: gpt-6.1-sol ₹8,338, gpt-4o ₹3,094.
- **No outcome analysis has been run.** The E1 label step printed d1's empirical-need base rate by design. It has not been used.
- **Next, after the PI's go (gate passed):**
  1. d2 Phase A `grade,store` on the required cells;
  2. d2 E1 labels;
  3. d2 store rebuild;
  4. the d1 400-answer judge subsample;
  5. `simulator.e3v2_lazy evaluate` with D*[SPEC] and m3 = 0.016 fixed (§9 step 9).

### 2026-10-17 — d1_open-only test analysis (PI go-ahead; NOT the confirmatory pooled test)

**Status:** d1_open only. The pooled confirmatory results wait for d2_support grading after the judge gate. Frozen code plus D1, with no judge calls. Sources: `docs/research_v2/v2_test_results_d1.json` and `.md` (`analysis/v2_test_d1.py`), and the frozen `simulator.e3v2_lazy evaluate` (D\*[COMMIT] = D\*[SPEC] = B9+, m₃ = 0.016). Recorded by Ryan Gomez.

- **Primary cell** (rewrite_faiss, q = 0.65, λ_l = 0.17305735434823866, primary ASR, 3,734 items; paired bootstrap B = 10,000):

  | contrast | estimate | 95% CI |
  |---|---|---|
  | Δ[COMMIT] | −0.0367 | [−0.0481, −0.0252] |
  | Δ[SPEC] | −0.0415 | [−0.0528, −0.0302] |
  | interaction | +0.0047 | [+0.0007, +0.0088] |

- **d1-only one-sided tests:**
  - P1: p = 1.0, does not reject.
  - H-REV: p < 1e-4, rejects.
  - P2: p = 0.0097. Rejects at α = 0.025, but not after Holm (threshold 0.0083).
  - P3 TOST: fails on the lower side.
  - Boundary claim: false.
  - These are the outcomes the dev predictions gave (§3.6).
- **E1-v2 on d1 test:**
  - Full-question AUC is 0.665 [0.648, 0.682]: below the 0.70 success line, above 0.60.
  - Text prefix curve: 0.537 / 0.543 / 0.582 / 0.665.
  - C2 is not available.
  - The dev-fitted calibration has ECE 0.077 on the full question.
  - The boundary mass is 0 (b* < 0 with dev ĝ⁻ < 0), as predicted.

### 2026-10-17T17:13Z — Gate supplement (deviation D2) and per-company retrieval finding (note N2)
* **Gate supplement (D2).** The human-labelling kit had closed-book answers only, so the pre-registered retrieve-vs-skip judge criterion could not be evaluated. 100 d2 dev retrieval-path answers were added (`build_supplement.py`, seed 20261017). The same two annotators label them. Recorded before any d2 test judging.
* **Per-company retrieval (N2).** The d2 retriever searches all 24 companies' documents together: on dev, the top-1 doc came from the caller's company 37% of the time, with 64% vs 17% correct on org-specific questions. The confirmatory analysis is unchanged (frozen). It is reported as a limitation, with a clearly labelled exploratory rerun of d2 test using per-company retrieval.
* **Decided by:** Ryan (PI). **Recorded by:** Ryan Gomez.

### 2026-10-17 — Note N3: the d2_support lazy plan is re-run after test labels (PI decision; transparency)

The frozen lazy plan includes `oracle_v2` only once test labels exist, and the d2_support plan was written before the judge gate.
- After `label.py`, the plan is re-run deterministically. It is asserted that the non-oracle cells equal the original plan (f59b33bc…).
- The oracle's extra cells are generated and graded, and the store is rebuilt. They are pre-approved up to ₹1,000.
- The other controllers' cells and decisions are unchanged; this was verified on 747,000 decision records. The oracle enters no confirmatory test. Recorded as PREREG_V2 note N3.
- Recorded by Ryan Gomez.
