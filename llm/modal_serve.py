"""
modal_serve.py — serve a project LLM behind an OpenAI-compatible vLLM
endpoint on Modal.

The serving flags mirror configs/config.yaml's llm.serve block and
llm/serve.py: bfloat16 (never quantized, because the logprobs are the
measurement instrument), max_model_len 8192, max_logprobs 20, and
--api-key read from the VLLM_API_KEY in the `voice-rag-vllm` Modal secret.
That secret must also carry HF_TOKEN, which gated models need.

The study's model is Mistral-7B-Instruct-v0.3. The live endpoint (Modal app
`voice-rag-vllm`) was deployed before this file existed, and its source
was not kept. This file is the reproducible recipe for serving the same
model under the same flags. It deploys as a SEPARATE app by default, so it
never replaces the endpoint a run is using:

    VR_APP_NAME=voice-rag-vllm-repro modal deploy llm/modal_serve.py

    # endpoint: https://<workspace>--<VR_APP_NAME>-serve.modal.run/v1

Every run records the exact model id and vLLM version it was served with.
"""

import os
import subprocess

import modal

MODEL_HF_ID = os.environ.get("VR_MODEL_HF_ID", "mistralai/Mistral-7B-Instruct-v0.3")
APP_NAME = os.environ.get("VR_APP_NAME", "voice-rag-vllm-repro")
VLLM_VERSION = "0.10.1.1"
GPU = os.environ.get("VR_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers is pinned below 4.57: an unpinned install resolves to a
    # release whose tokenizer API vLLM 0.10.1 cannot load (crash at startup).
    .pip_install(f"vllm=={VLLM_VERSION}", "transformers>=4.55,<4.57", "huggingface_hub[hf_transfer]")
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
