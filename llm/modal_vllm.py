"""Deploy the experiment's vLLM server on a serverless Modal A10 GPU.

Deploy from the repository root with::

    python -m modal deploy llm/modal_vllm.py

The resulting URL is OpenAI-compatible. Configure the local client with the
URL plus ``/v1`` and the same ``VLLM_API_KEY`` stored in the Modal secret
named ``voice-rag-vllm``.
"""

import hmac
import os
import subprocess

import modal


APP_NAME = "voice-rag-vllm"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# Pin the exact public Hugging Face revision used by the experiment.
MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
VLLM_PORT = 8000
MINUTES = 60

app = modal.App(APP_NAME)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)

hf_cache = modal.Volume.from_name("voice-rag-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("voice-rag-vllm-cache", create_if_missing=True)
api_secret = modal.Secret.from_name("voice-rag-vllm")


class BearerAuthMiddleware:
    """Authenticate without putting the bearer secret in vLLM's argv/logs."""

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
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
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
    },
    scaledown_window=5 * MINUTES,
    timeout=24 * 60 * MINUTES,
)
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=20 * MINUTES,
)
def serve() -> None:
    """Start an authenticated OpenAI-compatible vLLM HTTP server."""
    command = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--max-logprobs",
        "20",
        "--middleware",
        "modal_vllm.BearerAuthMiddleware",
    ]
    subprocess.Popen(command)
