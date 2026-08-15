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
`meta-llama/Llama-3.1-8B-Instruct`. Ryan has been asked to confirm the reason
for this difference before the Experiment 1 documentation is closed.

**Recorded by:** Yashmita
