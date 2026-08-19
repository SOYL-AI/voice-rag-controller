# Voice-RAG Controller — Lab Log

This file records important project decisions, scope changes, tool choices,
experiment decisions, and blockers.

## Decision Log

### 2026-08-06 — Phase 1 documentation started

**Decision:** Created the project lab log for recording technical and research decisions.

**Reason:** The team needs a dated record explaining why important choices were made.

**Decided by:** Project team

**Recorded by:** Yashmita Sudhir

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
more than usual here because the two tests disagree in DIRECTION rather than
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
