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
