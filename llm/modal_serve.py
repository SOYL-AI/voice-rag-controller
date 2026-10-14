"""
modal_serve.py — serve a project LLM behind an OpenAI-compatible vLLM
endpoint on Modal.

The serving flags mirror configs/config.yaml's llm.serve block and
llm/serve.py: bfloat16 (never quantized, because the logprobs are the
measurement instrument), max_model_len 8192, max_logprobs 20, and
--api-key read from the VLLM_API_KEY in the `voice-rag-vllm` Modal secret.
That secret must also carry HF_TOKEN, which gated models need.

Each model is its own Modal app, so deploying a second model never touches
an endpoint another run is using:

    # Llama-3.1-8B-Instruct (the originally pre-registered model)
    VR_MODEL_HF_ID=meta-llama/Llama-3.1-8B-Instruct VR_APP_NAME=voice-rag-vllm-llama31 \
        modal deploy llm/modal_serve.py

    # endpoint: https://<workspace>--<VR_APP_NAME>-serve.modal.run/v1

The original Mistral-7B endpoint (app `voice-rag-vllm`) predates this file.
Every run records the exact model id and vLLM version it was served with.
"""

import os
import subprocess

import modal

MODEL_HF_ID = os.environ.get("VR_MODEL_HF_ID", "meta-llama/Llama-3.1-8B-Instruct")
APP_NAME = os.environ.get("VR_APP_NAME", "voice-rag-vllm-llama31")
VLLM_VERSION = "0.10.1.1"
GPU = os.environ.get("VR_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VR_MODEL_HF_ID": MODEL_HF_ID})
)

hf_cache = modal.Volume.from_name("voice-rag-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("voice-rag-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=15 * 60,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("voice-rag-vllm")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", os.environ["VR_MODEL_HF_ID"],
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--max-logprobs", "20",
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    subprocess.Popen(cmd)
