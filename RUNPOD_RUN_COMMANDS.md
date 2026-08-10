# RunPod Experiment Commands

Run these commands from PowerShell. Never paste the RunPod API key into this
file or commit it to the repository; keep it in the ignored `.env` file.

## 1. Open the repository

```powershell
cd C:\Users\ryang\OneDrive\Desktop\voice-rag\voice-rag-controller
```

## 2. Load `.env` into the current PowerShell session

The Python application does not load `.env` automatically. Repeat this step
whenever you open a new PowerShell window.

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
        [Environment]::SetEnvironmentVariable(
            $matches[1].Trim(),
            $matches[2].Trim(),
            "Process"
        )
    }
}
```

The relevant `.env` entries are:

```dotenv
VOICE_RAG_LLM_BACKEND=vllm
VOICE_RAG_LLM_BASE_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1
VLLM_API_KEY=YOUR_RUNPOD_API_KEY
```

## 3. Verify the resolved configuration

```powershell
python -m llm.config
```

Confirm that the output reports:

- backend `vllm`;
- model `meta-llama/Llama-3.1-8B-Instruct`;
- the expected RunPod `/openai/v1` base URL;
- logprobs enabled with `top_k=5`.

## 4. Test one request

Always do this before paying for the complete experiment:

```powershell
python -m llm.client
```

A successful test prints an answer, TTFT, token count, logprob statistics, and
per-token confidence values. Its event log is appended to:

```text
logs/conv_llm_demo.jsonl
```

The first request may time out while a Serverless worker cold-starts. If the
RunPod worker is healthy and still loading, wait for it to become ready and
retry the test. The configured request timeout is 600 seconds.

## 5. Run Experiment 1 and preserve each result

Do not run multiple copies concurrently. Create a timestamped result name so
later runs do not overwrite earlier ones:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$result = "llm/experiment1/results/gate_vllm_$stamp.json"
$console = "llm/experiment1/results/gate_vllm_$stamp.console.log"

python -m llm.experiment1.run --output $result 2>&1 |
    Tee-Object -FilePath $console
```

The runner makes 216 sequential model calls: 36 tracker calls and 180 answer
calls across five seeds. It prints little while running and writes the final
JSON only after every call completes, so leave the PowerShell window open.

## Result locations

Each timestamped run produces:

```text
llm/experiment1/results/gate_vllm_YYYYMMDD_HHMMSS.json
llm/experiment1/results/gate_vllm_YYYYMMDD_HHMMSS.console.log
```

The JSON is the complete research result. Important fields include:

- `verdict`: `PASS`, `PIVOT`, or `INCONCLUSIVE`;
- `best_signal` and `best_mean_auc`;
- `aggregate_auc` and per-seed AUC values;
- every model answer, expected answer, retrieval label, and uncertainty signal.

The registered verdict thresholds are:

- `PASS`: best mean AUC greater than or equal to `0.70`;
- `PIVOT`: best mean AUC below `0.60`;
- `INCONCLUSIVE`: best mean AUC from `0.60` up to, but not including, `0.70`.

To inspect all saved runs later:

```powershell
Get-ChildItem llm\experiment1\results\gate_vllm_*.json |
    Sort-Object LastWriteTime
```
