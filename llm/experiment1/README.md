# Experiment 1 — Signal Validity

Owner: Ryan. This is the Phase 2 gatekeeper audit described in
`paper/preregistration.md`.

## Frozen question and thresholds

Can closed-book model uncertainty distinguish questions that require a
project document from questions answerable with stable general knowledge?
The metric is ROC AUC, oriented so higher signal values mean more likely to
need retrieval. The pre-registered thresholds are read from `config.json`:

- pass: best mean AUC >= 0.7;
- pivot: best mean AUC < 0.6;
- otherwise: inconclusive.

The run uses the five preconfigured seeds. `dataset.jsonl` is balanced and
stores a known answer and provenance for every label. Hotel-specific examples
avoid the five intentionally contradictory corpus pairs.

## Run locally (plumbing only)

```powershell
$env:VOICE_RAG_LLM_BACKEND = "mock"
python -m llm.experiment1.run
```

This writes `results/smoke_mock.json`. Its synthetic logprobs are useful only
for validating collection, entropy calculation, AUC scoring, and result
serialization. The runner marks it `NOT_EVALUATED_SMOKE`; it is not research
evidence and must not be copied to the lab log as the gate verdict.

## Run the real gate on the GPU host

Start the unquantized vLLM setup from `llm/SETUP.md`, then:

```powershell
$env:VOICE_RAG_LLM_BACKEND = "vllm"
$env:VOICE_RAG_LLM_BASE_URL = "http://<host>:8000/v1"
python -m llm.experiment1.run --output llm/experiment1/results/gate_vllm.json
```

The runner refuses a vLLM run if the configured server/model health check
fails. Review the sample answers and labels before Yashmitha copies the real
PASS/PIVOT/INCONCLUSIVE verdict into the shared lab log.

Lakshya's unmerged `phase2-retrieval` branch adds public SpokenWOZ and
Spoken-SQuAD loaders. After that branch is reviewed, extend this seed dataset
with its answer-bearing public examples; do not silently change the frozen
thresholds.
