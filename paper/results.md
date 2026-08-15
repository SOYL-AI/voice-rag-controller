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

  ## Experiment 2 — Timing Value

### Status

Analysis complete — verdict pending policy-name confirmation.

### Pre-Registered Success Criterion

Forecast-timed retrieval gives lower p95 TTFA at equal answer accuracy.

### Results

| Policy | Hit@3 Accuracy | p95 TTFA |
|---|---:|---:|
| `fixed_early_proxy` | 100% | 7090 ms |
| `immediate` | 100% | 6915 ms |
| `sentence_end` | 100% | 7052 ms |

### Median-Latency Observation

`fixed_early_proxy` has the lowest median TTFA at approximately 3.69 seconds,
compared with approximately 4.07 seconds for `immediate` and 4.11 seconds for
`sentence_end`.

This median improvement is useful as a secondary observation, but the
pre-registered success criterion is based on p95 TTFA.

### Primary Verdict

Pending confirmation that `fixed_early_proxy` corresponds to the
forecast-timed policy.

If confirmed, the pre-registered p95 success criterion is NOT MET:
`fixed_early_proxy` p95 TTFA = 7090 ms versus 6915 ms for `immediate`.

### Supporting Figures

- `analysis/e2_accuracy_vs_latency.png`
- `analysis/e2_ttfa_distribution.png`
- Raw sweep: `logs/` — 750-run E2 Modal sweep

### Paired-Scenario Analysis

Accuracy was matched across all three policies at 100% Hit@3.

Compared with `sentence_end`, `fixed_early_proxy` reduced median TTFA by
479 ms. The paired-scenario bootstrap 95% confidence interval excluded zero,
indicating a statistically significant latency reduction.

Compared with `immediate`, `fixed_early_proxy` reduced median TTFA by
approximately 180 ms, but the 95% confidence interval crossed zero. This
difference is therefore directional but not statistically significant.

### Mechanism Check

Speech overlap was negatively correlated with TTFA:

- Pearson r = -0.25

Greater overlap between retrieval work and the user's speech was therefore
associated with lower response latency.

### Interpretation

The p95 TTFA values were relatively similar across policies and were
tail-noisy. The paired-scenario analysis provides the stronger evidence for
the timing effect.

The E2 result supports the claim that earlier retrieval placement can reduce
latency at matched answer accuracy. The strongest evidence is against the
`sentence_end` policy, where `fixed_early_proxy` produced a significant
479 ms median-TTFA reduction.

The effect is modest because overall TTFA remains largely dominated by
language-model latency.

### Implementation Limitation

`fixed_early_proxy` is a fixed early dispatch point rather than a true
forecast-timed policy. Live forecast timing using a VAD/VAP or pause-prediction
signal remains future work.

### Supporting Material

- Raw sweep: `logs/` — clean 750-run E2 sweep
- `analysis/e2_accuracy_vs_latency.png`
- `analysis/e2_ttfa_distribution.png`
- `analysis/e2_overlap_vs_ttfa.png`
- `analysis/e2_launch_offset_distribution.png`
