# Voice-RAG Controller — Experiment Results

This file compiles the final experiment results against the criteria recorded in
`preregistration.md`.

---

## Experiment 1 — Signal Validity

### Status

Completed — PASS

### Pre-Registered Criterion

- Primary metric: AUC
- Success criterion: best signal AUC >= 0.70
- Kill criterion: best signal AUC < 0.60

### Primary Result

| Item | Result |
|---|---|
| Verdict | PASS |
| Best signal | `max_entropy_nats` |
| Mean AUC | 0.9784 |
| Minimum AUC across seeds | 0.9630 |
| Maximum AUC across seeds | 0.9907 |
| Number of seeds | 5 |
| Primary dataset size | 36 |
| Retrieval needed | 18 |
| Retrieval not needed | 18 |

### Seed Results — Best Signal

| Seed | AUC |
|---:|---:|
| 11 | 0.9907 |
| 23 | 0.9784 |
| 37 | 0.9815 |
| 53 | 0.9630 |
| 71 | 0.9784 |

### Result Summary

The best-performing signal was `max_entropy_nats`, with a mean AUC of 0.9784
across the five seeds. This exceeds the pre-registered success threshold of 0.70.

### Documentation Check

The result artifact reports the model as
`mistralai/Mistral-7B-Instruct-v0.3`.

The current pre-registration names
`meta-llama/Llama-3.1-8B-Instruct`.

This discrepancy is awaiting confirmation from Ryan before the Experiment 1
record is considered fully reconciled.

### Supporting Information

- Dataset: `llm/experiment1/dataset.jsonl`
- Examples: 36
- Dataset SHA256:
  `01050b552346c78601ba58d11584a4ebec47d42bc7e347bbb45777faf3a460fc`
- Backend: vLLM
- Raw result artifact: Experiment 1 JSON shared by Ryan
