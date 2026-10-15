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
  python -m llm.experiment1_v2.self_assess --dataset $ds --split dev --backend vllm --concurrency 16
  python -m llm.experiment1_v2.label --dataset $ds --split dev --model mistral7b_v03 --workers 4
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

## Self-assessed need (`self_assess.py`)

Entropy of the closed-book answer is near chance on d2_support dev. In the
support-agent persona Mistral fabricates org policies fluently, so its
uncertainty does not track whether it needs the company's documents.

The signal `self_assessed_need` asks the model directly instead. Per distinct
prefix text it makes one deterministic call (temperature 0, max_tokens 1,
top-20 logprobs) with the question:

> Can this be answered correctly from general knowledge alone, without
> {org}'s own policies, prices, plans or records? Answer with one word: Yes or No.

d1_open uses a neutral variant. The signal is
P(No) = mass of tokens starting "no" / (mass "yes" + mass "no"). Each
call's own TTFT is recorded.

**The wording is frozen** (`config.json` `self_assessment`, `sa-v1-frozen`).
It was taken as-is from one exploratory probe on d2_support dev and must not
be tuned on dev: with 222 dev items, any search over wordings would overfit.

```powershell
python -m llm.experiment1_v2.self_assess --dataset d2_support --split dev --concurrency 16
```

This writes `self_assessment_dev.jsonl`. `analyze.py` joins it to every
record by (item, prefix text) and treats it like any other signal: AUC,
prefix and ASR curves, a latency row, and full and prefix-aware calibration
artifacts.

## Judges (`judges.json`)

**Rubric r2** (`rubrics/e1v2-r2/`, default) applies to both judges. A
d2_support answer is correct iff it states **every** `meta.key_facts` item
(paraphrase and unit/format variants allowed) **and** contradicts nothing in
the reference. Deferring, declining and key-fact-free hedges are incorrect.
The judge also sees the dialogue history. Other datasets keep r1's text
(`default.txt`); d1_open correctness stays alias match. r1
(`--rubric e1v2-reasoned-strip-r1`) reproduces the first d2_support grading
byte-for-byte. The rubric's sha256 is in every judge identity and cache key.

The human labelling kit for the §2.4 gate is in
`simulator/judge_validation/v2_labeling/`:
- the guide: `LABELING_GUIDE.md`;
- the sheets: `sheet_annotator_A.csv` and `sheet_annotator_B.csv`;
- `build_kit.py` and `merge_sheets.py`.

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

**Empirical need** (PREREG_V2 §1) means the answer to the full question is
wrong in at least 3 of 5 closed-book draws:
- draws with no valid judge majority are excluded, and the label is
  undetermined if they could decide it;
- a sensitivity variant counts them as wrong.

If fewer than 2 of 3 judge samples parse, or they tie, up to 3 more are
drawn.

Every sample stores the exact request body the SDK sent, captured from the
httpx request (body only, never the URL or headers). It also stores the
response id, finish reason and usage. The judge identity includes the
prompt sha256.

`score_judges.py --v2-labels` ends with the §2.4 gate. The holdout is the
cases with sha256(case_id) mod 3 ≠ 0; the rest are prompt-development. On
the holdout it checks:
- human κ ≥ 0.70, first;
- judge κ ≥ 0.80, with lower 95% bound ≥ 0.70;
- accuracy ≥ 0.90;
- |error(retrieve) − error(skip)| ≤ 0.05, which needs a `path` column;
- unanimity ≥ 0.90 and Fleiss' κ ≥ 0.80.

## Outputs the v2 controllers read

`analyze.py` writes two calibration artifacts per model:
- `calibration_<ds>_mistral7b_v03_<signal>.json`: the full transcript, in v1
  format, loadable by `_objective.py`;
- `calibration_prefix_<ds>_mistral7b_v03_<signal>.json`: the prefix-aware form
  `sigmoid(intercept + slope·x + frac_slope·f + interaction·x·f)`. This is
  the one to point `controller.e3v2.calibration.from_artifact` at. It was
  checked against `_objective_v2.py:load_belief_calibration()` on
  research/v2-base.

It also prints ĝ⁺ and ĝ⁻ for `controller.e3v2.quality`. The analysis JSON
holds the remaining PREREG H-E1v2 quantities:
- the ASR prefix curve at word_frac {0.3, 0.5, 0.7, 1.0};
- the paired bootstrap of AUC(1.0) − AUC(0.3);
- the C2-availability flag, AUC(0.5) ≥ 0.65;
- the boundary-mass check.

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

Planned dev sizes are 30% of `d1_open` ≈ 5,300 items and `d2_support` ≈ 720
items. That gives 1,590 + 216 = 1,806 dev items, or 9,030 (item, seed) pairs
over 5 seeds:

| conditions | d1_open | d2_support | total calls | wall time (1 stream) |
|---|---|---|---|---|
| text: full + 3 prefixes (≤ 4) | 31,800 | 4,320 | 36,120 | 29.1 h |
| ASR, per voice condition (assumes ~10 distinct partials + final) | 87,450 | 11,880 | 99,330 | 80.0 h |
| retrieval (full question) | 7,950 | 1,080 | 9,030 | 7.3 h |
| latency probe (50 items x 4 texts x 1 seed, each) | 200 | 200 | 400 | 0.3 h |
| **total, one ASR voice** | | | **144,880** | **≈ 117 h** |

The ASR row is the uncertain one; `--dry-run` gives the exact deduplicated
count once the traces exist. At one stream the full run is about 5 days.
Moving the endpoint to the concurrent Modal app (`llm/modal_serve.py` on
research/v2-base declares `@modal.concurrent(max_inputs=64)`) or adding
replicas divides the wall time by the effective parallelism, which must be
measured (`--assume-parallel-streams`).

Judging covers 18,060 answers: 9,030 closed-book plus 9,030 grounded.
Identical answers share cache entries; in the smoke run 68 of 80 were
distinct. Expected calls:
- gpt-6.1-sol: 54,180 (k=3), plus redraws on parse failures (none in the smoke run);
- gpt-4o: 18,060.

Measured per call in the smoke run:
- prompt about 323 tokens (expect about 350 with d2 key facts);
- visible completion about 26 tokens;
- 0 reasoning tokens at `medium` on these easy items.

That puts gpt-6.1-sol at about 19 M prompt tokens plus at least 1.4 M
completion tokens, and harder items will add reasoning tokens. The
`max_completion_tokens` 2000 cap bounds the worst case at about 108 M.
gpt-4o comes to about 6.3 M prompt plus 0.4 M completion tokens. Judge wall
time was not measured; `label.py` runs `--workers` parallel calls (default 8).

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
