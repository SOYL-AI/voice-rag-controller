"""Energy-instrumented variant of the Modal vLLM deploy script. NOT DEPLOYED.

Base: ``llm/modal_vllm.py`` on branch ``experiment1/modal-mistral-results``
(commit 9149692), the script that served Mistral-7B-Instruct-v0.3 for E1-E3.
Everything that determines model behaviour is identical (image, vLLM
version, model + pinned revision, dtype, max-model-len, memory fraction,
max-logprobs, GPU type, auth). The differences, all additive:

1. ``nvidia-ml-py`` is added to the image.
2. The ``energy`` package from this repo is shipped into the image
   (``add_local_python_source``) so ``energy.serving.gpu_energy`` imports.
3. A third Modal Volume, ``voice-rag-energy-logs``, is mounted at
   ``/energy-logs``; the middleware appends ``requests.jsonl`` and
   ``timeline.jsonl`` there.
4. vLLM gets a second ``--middleware`` (EnergyMiddleware). vLLM calls
   ``app.add_middleware`` in argument order and Starlette makes the
   last-added middleware the outermost, so auth is listed last: requests
   are authenticated before they are measured, and the ``/energy/requests``
   log endpoint sits behind the same bearer auth. (Verify the order against
   the pinned vLLM version before deploying.)
5. ``max_containers=1`` so every request in a run lands on one GPU (one
   log, one idle floor). The E1-E3 runs were effectively single-container.

To use it later (requires Modal credentials; costs money; not done here):

    python -m modal deploy energy/serving/modal_vllm_energy.py
    # client side, per request: send stream_options={"include_usage": true}
    # when streaming, keep requests sequential, then pull the log:
    curl -H "Authorization: Bearer $VLLM_API_KEY" "$URL/energy/requests?since=<ts>"
    # or: modal volume get voice-rag-energy-logs requests.jsonl .

Before trusting the numbers, check in the log that ``counter_vs_power`` is
~1.0 (counter sane on this driver) and ``inflight_max`` is 1, and record
the ``timeline.jsonl`` idle floor alongside the run manifest.
"""

import hmac
import os
import subprocess

import modal

APP_NAME = "voice-rag-vllm-energy"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
VLLM_PORT = 8000
MINUTES = 60

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "nvidia-ml-py")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "10",
          "VOICE_RAG_ENERGY_LOG_DIR": "/energy-logs"})
    .add_local_python_source("energy")
)

hf_cache = modal.Volume.from_name("voice-rag-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("voice-rag-vllm-cache", create_if_missing=True)
energy_logs = modal.Volume.from_name("voice-rag-energy-logs", create_if_missing=True)
api_secret = modal.Secret.from_name("voice-rag-vllm")


class BearerAuthMiddleware:
    """Unchanged from llm/modal_vllm.py."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = headers.get(b"authorization", b"").decode("utf-8")
        expected = f"Bearer {os.environ['VLLM_API_KEY']}"
        if not hmac.compare_digest(supplied, expected):
            body = b'{"error":"Unauthorized"}'
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode("ascii"))]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


@app.function(
    image=vllm_image,
    gpu="A10",
    cpu=4,
    memory=16384,
    secrets=[api_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/energy-logs": energy_logs,
    },
    scaledown_window=5 * MINUTES,
    timeout=24 * 60 * MINUTES,
    max_containers=1,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve() -> None:
    command = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--max-logprobs", "20",
        # Starlette runs the LAST-added middleware outermost; vLLM adds them
        # in the order given, so auth (listed second) wraps energy.
        "--middleware", "energy.serving.gpu_energy.EnergyMiddleware",
        "--middleware", "modal_vllm_energy.BearerAuthMiddleware",
    ]
    subprocess.Popen(command)
