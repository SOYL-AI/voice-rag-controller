# Experiment 1, v2: the belief at dispatch time

v1 E1 validated an entropy signal on 36 trivially separable items, fitted the
calibration in-sample, reported max-entropy AUC while the controller used mean
entropy, and computed the belief on the full question although early-timing
policies dispatch retrieval before the user finishes. E1-v2 fixes each of these:

- harder data: `d1_open` (public open-domain QA) and `d2_support` (multi-domain
  customer support), CONTRACTS.md sections 2 to 4;
- **empirical** retrieval-need labels (closed-book wrong across draws), next to
  the authored ones;
- signals on **partial transcripts**: text prefixes at word fractions
  0.25/0.5/0.75 and every streaming-ASR partial of the speech traces;
- the measured **latency of obtaining the belief** (TTFT, arrival of token k,
  generation time), from unqueued calls;
- calibration fitted on **dev only**, evaluated by grouped cross-validation;
- one model: `mistralai/Mistral-7B-Instruct-v0.3` (`mistral7b_v03`). The code
  takes `--model` from the `configs/config.yaml` registry, but v2 runs Mistral
  only.

The generation protocol is v1 E1's, read at run time from
`llm/experiment1/config.json`: seeds 11, 23, 37, 53, 71; temperature 0.2;
top_p 0.9; max_tokens 80; top-5 logprobs (`configs/config.yaml`).

## Pipeline

| step | module | output (`results/<dataset>/mistral7b_v03/`) |
|---|---|---|
| signals per prefix condition | `run.py` | `signals_dev.jsonl` (CONTRACTS section 5) |
| grounded answers for the full question | `run.py --with-retrieval` | `retrieval_dev.jsonl` |
| unqueued latency of the belief | `run.py --latency-probe` | `latency_dev.jsonl` |
| empirical labels, g per item, judge agreement | `label.py` | `labels_dev.json` |
| AUCs, dispatch-time curves, first-k vs latency, calibration, g+/g-, figures | `analyze.py` | `../analysis_dev/` |

Every entry point takes `--split dev|test` and refuses `test` until
`docs/research_v2/PREREG_FROZEN` exists.

## Real dev run (Mistral-7B)

Load credentials from the environment (`VOICE_RAG_LLM_BASE_URL`,
`VLLM_API_KEY`, `AZURE_OPENAI_*`, `VOICE_RAG_JUDGE_*`); nothing reads or writes
them elsewhere. Use `.venv-e2` and `PYTHONUTF8=1`.

```powershell
$env:VOICE_RAG_MODEL = "mistral7b_v03"
$env:VOICE_RAG_JUDGE_MODEL = "gpt-6.1-sol-2026-09-29"   # model version, recorded on every verdict
foreach ($ds in "d1_open", "d2_support") {
  python -m llm.experiment1_v2.run --dataset $ds --split dev --backend vllm --dry-run
  python -m llm.experiment1_v2.run --dataset $ds --split dev --backend vllm --latency-probe --limit 50
  python -m llm.experiment1_v2.run --dataset $ds --split dev --backend vllm --with-retrieval
  python -m llm.experiment1_v2.label --dataset $ds --split dev --model mistral7b_v03
  python -m llm.experiment1_v2.analyze --dataset $ds --split dev --controller-artifact
}
```

- `run.py` is resumable: re-run the same command after a crash or failures
  (`errors_dev.jsonl`). Calls with identical prompt text are made once per
  (item, seed) and copied to every condition that shares them.
- `--with-retrieval` needs the v2 index and `retrieval.search.search(query,
  index_dir, k)` (default index `datasets/v2/<dataset>/index`); see
  `retrieval_adapter.py`.
- Run `--latency-probe` on a warm endpoint with nothing else running. Latency
  under load includes server queueing and is not used as belief latency.
- `analyze.py --controller-artifact` writes
  `controller/baselines/e1v2_calibration_<dataset>_mistral7b_v03_mean_entropy_nats.json`
  in the format `_objective.py:load_voi_calibration()` reads. It refuses mock
  results and slopes <= 0.

## Judges (`judges.json`)

| role | deployment | settings |
|---|---|---|
| primary | `VOICE_RAG_JUDGE_DEPLOYMENT` (gpt-6.1-sol), `VOICE_RAG_JUDGE_API_VERSION` (2025-04-01-preview) | no temperature (rejected), `max_completion_tokens` 2000, `reasoning_effort` medium, seed 20261015, k=3 seeded samples with a majority vote |
| secondary | `AZURE_OPENAI_DEPLOYMENT` (gpt-4o), `AZURE_OPENAI_API_VERSION` | v4 settings: temperature 0, `max_tokens` 300, one sample |

You can override any request parameter with `VOICE_RAG_JUDGE[2]_{TEMPERATURE,
TOKEN_PARAM,MAX_TOKENS,REASONING_EFFORT,SEED,SAMPLES}`. Every verdict records:
- the judge identity;
- the exact parameters sent;
- the model string the server returned;
- token usage, including reasoning tokens;
- every sample.

The cache key covers identity, request config and prompt version.
`label.py` reports alias vs judge vs judge agreement (percent and Cohen's
kappa) and the primary judge's self-consistency across its samples.

Correctness of record: `d1_open` uses a normalized alias match; `d2_support`
uses the primary judge with key facts.

To score both judges against the team's double-labelled answers, run:

```powershell
python -m simulator.judge_validation.score_judges --v2-labels human_labels.csv --out judge_report.json
```

## Cost and time

Generation calls = items x seeds x distinct prefix texts. Per (item, seed):

- up to 4 for the text conditions;
- about 1 per distinct ASR partial per trace condition, plus the final;
- 1 for retrieval.

`run.py --dry-run` prints the exact count after dedup and resume.

Measured on the Modal Mistral endpoint (2026-10-15):
- each call takes about 2.9 s (TTFT about 0.84 s, about 31.5 ms per token, median 69 tokens);
- throughput is 0.35 calls/s at client concurrency 1 and 0.38 at 4, so the endpoint serves one request at a time;
- until that changes, wall time is roughly calls x 2.9 s.

Example for 300 dev items and 5 seeds:

| conditions | calls | wall time (1 stream) |
|---|---|---|
| text (4) | 6,000 | 4.8 h |
| + 1 clean ASR voice (about 10 distinct partials + final) | +16,500 | +13.3 h |
| + retrieval | +1,500 | +1.2 h |

Judging covers 3,000 answers (closed-book and grounded). That is 9,000
gpt-6.1-sol samples (k=3) plus 3,000 gpt-4o calls, run in parallel and
cached.

## Mock smoke run

```powershell
python -m llm.experiment1_v2.convert_v1 --out .tmp_e1v2/v1_e1 --all-dev --plumbing-traces
python -m llm.experiment1_v2.run --items .tmp_e1v2/v1_e1/items.jsonl --split dev --backend mock --with-retrieval --retriever lexical
python -m llm.experiment1_v2.label --items .tmp_e1v2/v1_e1/items.jsonl --split dev --model mistral7b_v03 --llm-backend mock --judge-backend mock
python -m llm.experiment1_v2.analyze --items .tmp_e1v2/v1_e1/items.jsonl --split dev --llm-backend mock
python -m pytest llm/tests/test_experiment1_v2.py
```

The mock backend reuses `llm/client.py`'s `MockClient`. Its logprobs and
timings are synthetic and its output goes to the gitignored `results_mock/`.
`--plumbing-traces` partials are reference words, not ASR, so `run.py` refuses
them for the real backend.
