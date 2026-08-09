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

## The extended dataset (secondary analysis)

`dataset.jsonl` has a known weakness: every positive names a fictional hotel
and every negative is arithmetic or primary-school trivia, so the two classes
are separable by surface form alone. A high AUC on it partly measures "the
model is less sure about invented proper nouns than about 2+2", which is a
weaker claim than the one Experiment 1 makes.

`build_extended_dataset.py` builds a 120-example set that removes that
confound, in the same schema `run.py` validates:

- 18 hotel positives (kept — they are the deployment domain);
- 42 Spoken-SQuAD positives, passage-dependent by construction;
- 60 negatives from `negatives_hard.jsonl`, authored to match Spoken-SQuAD's
  question form and difficulty.

```powershell
pip install pandas pyarrow          # not in requirements.txt
python -m llm.experiment1.build_extended_dataset
python -m llm.experiment1.run --dataset llm/experiment1/dataset_extended.jsonl `
  --output llm/experiment1/results/gate_vllm_extended.json
```

The builder drops any Spoken-SQuAD row whose question field embeds its source
passage — that would hand the model the answer and silently break the
closed-book design. Read the sample questions it prints before spending GPU
time.

**The frozen thresholds belong to `dataset.jsonl`.** The 36-item run is the
pre-registered gate; the extended run is a labelled secondary analysis
reported alongside it. Applying `success_auc` to a different dataset without
re-freezing in `paper/preregistration.md` first is exactly the substitution
pre-registration exists to prevent. Each result records the dataset sha256, so
which set produced which number stays checkable.

SpokenWOZ is deliberately not used here: its rows are dialogue turns with
neither a `retrieval_needed` label nor an answer key, so there is nothing for
an AUC to score. It is the conversation source for E2/E3.
