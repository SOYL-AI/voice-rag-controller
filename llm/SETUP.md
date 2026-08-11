# llm/ — Setup

How to stand up the model server and verify the confidence numbers are real.

Written provider-agnostic on purpose. The launch command is identical
everywhere; only the environment around it differs. RunPod is our primary,
Azure the fallback. This file is the thing that gets turned into the shared
Dockerfile in Week 2, so keep it accurate — if you run a command that isn't
written here, add it.

---

## What the hardware has to be

An 8B model at bf16 is ~16 GB of weights, plus KV cache and activations.

| VRAM | Verdict |
|------|---------|
| 16 GB | Only fits quantized — **don't**, see below |
| 24 GB (RTX 4090, A10G) | Fits, ~6 GB left for KV cache. Fine for Week 1 |
| 48 GB (A40, L40S, A6000) | Comfortable, and leaves room for the batched simulator runs later |
| 80 GB (A100, H100) | Overkill for 8B — paying for capacity we can't use |

**Do not serve a quantized model for real runs.** The per-token logprobs are
the measurement instrument for Experiment 1. AWQ/GPTQ/FP8 shift the output
distribution, which distorts the exact thing we are measuring, and a reviewer
can fairly ask whether the uncertainty signal survived compression. Serve
`--dtype bfloat16`. That constraint is what sets the 24 GB floor.

---

## Development: no GPU, no cost

`llm.backend: mock` in `configs/config.yaml` is the default, and it's what you
should use for everything except real data collection. The mock returns
synthetic logprobs in the same shape the real server produces, so the whole
logging path, the event payloads, and the tracker's JSON parsing can be
finished and tested on a laptop.

```bash
pip install openai pyyaml         # that's all the mock needs
python llm/client.py              # streams, logs, prints per-token confidence
python llm/tracker.py             # JSON slots for several conversations
```

Only start a paid instance once those both pass.

---

## Real runs: any provider

### 0. Secrets

Never in the repo, never pasted into chat or a ticket. Put them in a `.env`
file at the repo root — `.gitignore` already covers `.env`, and
`.dockerignore` keeps it out of every image layer:

```bash
# voice-rag-controller/.env   (never committed)
HF_TOKEN=hf_...          # required for gated models, see table below
VLLM_API_KEY=...         # optional locally; required if the port is public
```

Then pass it in rather than exporting by hand:

```bash
docker run --env-file .env ...
set -a && . ./.env && set +a          # for running outside Docker
```

If a token is ever pasted somewhere it shouldn't be, revoke it at
huggingface.co/settings/tokens and issue a new one. Rotating is a
30-second job; a leaked token on a public repo is not.

### 1. Install

Use the image. `Dockerfile` at the repo root has two targets:

```bash
docker build --target dev -t voice-rag-llm:dev .    # CPU, mock, ~240 MB
docker build --target gpu -t voice-rag-llm:gpu .    # CUDA + vLLM
```

The `gpu` target already contains vLLM, CUDA, and our launcher, so there is
nothing to install on the instance:

```bash
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -v /workspace/models:/models \
  -v $(pwd)/logs:/app/logs \
  voice-rag-llm:gpu
```

`-v /workspace/models:/models` must point at a persistent volume, or every
new instance re-downloads ~16 GB before it can answer anything.

Falling back to a bare `pip install -r requirements.txt` on the instance also
works, but nothing about it is reproducible, so use it only to debug.

### 2. Start the server

```bash
python llm/serve.py                     # reads every flag from configs/config.yaml
python llm/serve.py --print-only        # show the command without running it
VOICE_RAG_MODEL=qwen3_8b python llm/serve.py
```

First start downloads ~16 GB. Point `llm.serve.download_dir` at a persistent
volume so a destroyed instance doesn't mean downloading it again.

### 3. Verify the logprobs — this is the actual gate

```bash
VOICE_RAG_LLM_BACKEND=vllm \
VOICE_RAG_LLM_BASE_URL=http://localhost:8000/v1 \
python llm/client.py
```

You should see a per-token confidence listing and a written log file. "The
model answered" is not the deliverable; visible, logged confidence numbers
are. If you get a 400 mentioning logprobs, the server was started without
`--max-logprobs` >= `llm.logprobs.top_k`.

### 4. Point the rest of the system at it

```bash
export VOICE_RAG_LLM_BACKEND=vllm
export VOICE_RAG_LLM_BASE_URL=https://<host>:<port>/v1
```

### 5. Shut it down

Billing is per second and nobody remembers at 2am. Stop the instance the
moment a run finishes.

---

## RunPod: serverless vs. pod

These are two different products and they are not interchangeable for our
purposes.

**Serverless does not run our `gpu` image.** A RunPod Serverless endpoint is
job-based: it expects a worker that implements their handler interface and
returns a result, not a long-lived HTTP server. Deploying
`voice-rag-llm:gpu` there won't work, because our container's job is to hold
port 8000 open.

For serverless, use RunPod's **official vLLM worker** instead of a custom
image. Configure the endpoint with
`MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3`; it is public and does not
require `HF_TOKEN`. The worker exposes an OpenAI-compatible route, so our
client needs no API code change:

```bash
VOICE_RAG_LLM_BACKEND=vllm
VOICE_RAG_LLM_BASE_URL=https://api.runpod.ai/v2/<endpoint-id>/openai/v1
VLLM_API_KEY=<your RunPod API key>
```

Two things to verify on the first call, because we don't control that
worker's launch flags the way `serve.py` controls ours:

1. **Logprobs come back.** vLLM's default `--max-logprobs` ceiling is 20 and
   our `top_k` is 5, so it should work — but if you get a 400 mentioning
   logprobs, drop `llm.logprobs.top_k` to 1 and raise it once the worker's
   ceiling is known.
2. **The served model name matches** `cfg["model"]`, or `health_check()`
   reports a mismatch.

### Serverless is fine for verification, wrong for latency data

Serverless scales to zero, so a cold request loads ~16 GB of weights before
it answers — tens of seconds to minutes. Every TTFT and TTFA number measured
across a cold start is meaningless, and TTFA is this project's headline
metric.

So: **use serverless to confirm the logprobs are real and the tracker behaves
against a live model** — that is exactly the cheap functional check this week
needs. For Experiment 1 data collection, either set minimum workers to 1
(always warm, which costs roughly what a pod costs) or use a GPU pod. Don't
report timing numbers collected from a cold-starting endpoint.

## RunPod pod specifics

- Expose HTTP port `8000` in the pod template. RunPod gives you a proxy URL
  shaped like `https://<pod-id>-8000.proxy.runpod.net` — that becomes
  `VOICE_RAG_LLM_BASE_URL` with `/v1` appended.
- Because that URL is public, set `VLLM_API_KEY` before starting. `serve.py`
  picks it up automatically.
- Attach a network volume and set `llm.serve.download_dir` to a path on it,
  or you re-download the weights on every new pod.
- The pod hostname is regenerated each time, which is why the base URL is an
  environment variable and not a committed setting.

## Modal serverless specifics

Modal runs the pinned Mistral model on an A10, caches both Hugging Face weights
and vLLM compilation artifacts in persistent volumes, and scales the GPU to
zero after five idle minutes. From the repository root:

```powershell
python -m pip install modal
python -m modal setup

# Load .env, including a strong, dedicated VLLM_API_KEY. Do not reuse a
# RunPod, Azure, Hugging Face, or other provider credential for this value.
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
        [Environment]::SetEnvironmentVariable(
            $matches[1].Trim(), $matches[2].Trim(), "Process"
        )
    }
}

python -m modal secret create voice-rag-vllm `
    "VLLM_API_KEY=$env:VLLM_API_KEY" --force
$env:PYTHONUTF8 = "1"
python -m modal deploy llm/modal_vllm.py
```

Set `VOICE_RAG_LLM_BASE_URL` to the URL printed by Modal with `/v1`
appended, then run `python -m llm.client`. The bearer-auth middleware keeps
the endpoint key out of vLLM's process arguments and startup logs. The
deployment stays available after its container scales down; a later request
cold-starts it again.

## Azure specifics

- NC-series (A100) or NV-series. Check quota **before** planning around it:
  GPU quota on a new or student subscription is request-gated and approval
  can take days. If we're near a deadline, use RunPod.
- Open port 8000 in the network security group, or tunnel it:
  `ssh -L 8000:localhost:8000 user@host` — which is safer, and then the
  default `http://localhost:8000/v1` just works with no config change.

---

## The three models

All three live in `configs/config.yaml`. Switch with one environment variable:
`VOICE_RAG_MODEL=qwen3_8b`.

| Key | Hugging Face id | Gated | Note |
|-----|-----------------|-------|------|
| `llama31_8b` | `meta-llama/Llama-3.1-8B-Instruct` | yes | Accept the licence early — approval isn't always instant |
| `qwen3_8b` | `Qwen/Qwen3-8B` | no | Downloads immediately. Use if a licence is still pending |
| `mistral7b_v03` | `mistralai/Mistral-7B-Instruct-v0.3` | no | Current default. Public Apache-2.0 model; 7B, not 8B |

**Qwen3 gotcha:** it emits a `<think>` block by default, which breaks strict
JSON from the tracker and fills the logprob sequence with reasoning tokens
that aren't part of the answer. The config already disables it via
`chat_template_kwargs.enable_thinking: false`; don't remove that.

**These are fallbacks, not a comparison set.** Experiment 1 must be reported
on one model or the model becomes a confound. Whichever is active when real
data collection begins has to be named in the pre-registration doc first.

---

## Cost discipline

- Develop on `mock`. Rent nothing.
- Boot the GPU, verify, collect, shut down. Don't leave it idling overnight.
- Prefer on-demand over reserved until we know how many hours the experiments
  actually need.
