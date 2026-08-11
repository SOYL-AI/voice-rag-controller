"""
config.py — loads the `llm:` section of configs/config.yaml.

Nothing in llm/ hardcodes a model name, a URL, or a threshold. Everything
comes through here, per the rule in configs/README.md.

Three things can be overridden by environment variable, because they are the
three that change per machine rather than per experiment:

    VOICE_RAG_LLM_BACKEND   mock | vllm
    VOICE_RAG_MODEL         a key from the `models:` map (e.g. qwen3_8b)
    VOICE_RAG_LLM_BASE_URL  where the vLLM server is reachable

That last one matters more than it looks: on RunPod the pod's hostname is
generated fresh every time you start one, so it cannot live in a file that
gets committed.

Usage:

    from llm.config import load_config
    cfg = load_config()
    cfg["model"]        # -> "mistralai/Mistral-7B-Instruct-v0.3"
    cfg["backend"]      # -> "mock"
"""

import copy
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"


def config_path() -> Path:
    """Where we read settings from. VOICE_RAG_CONFIG overrides it."""
    override = os.environ.get("VOICE_RAG_CONFIG")
    return Path(override) if override else _DEFAULT_CONFIG_PATH


def load_config(path=None, model_key: str = None) -> dict:
    """
    Read configs/config.yaml and return the `llm:` section, with the active
    model already resolved.

    The returned dict adds three resolved keys on top of what's in the file:

        model        the Hugging Face id of the active model
        model_key    which entry in `models:` that came from
        extra_body   per-model request extras (e.g. Qwen's thinking toggle)

    so callers never have to know which model is selected or how to look it up.
    """
    import yaml  # imported here so `import llm.config` works before pip install

    cfg_file = Path(path) if path else config_path()
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"No config at {cfg_file}. It should be committed at "
            f"configs/config.yaml - see configs/README.md."
        )

    with open(cfg_file, "r", encoding="utf-8") as fh:
        full = yaml.safe_load(fh) or {}

    if "llm" not in full:
        raise KeyError(f"{cfg_file} has no `llm:` section.")

    # Deep copy so a caller mutating the result can't affect a later load.
    cfg = copy.deepcopy(full["llm"])

    # -- resolve which model is active ----------------------------------
    models = cfg.get("models") or {}
    key = model_key or os.environ.get("VOICE_RAG_MODEL") or cfg.get("active_model")
    if key not in models:
        raise KeyError(
            f"Model key '{key}' is not in configs/config.yaml under "
            f"llm.models. Available: {sorted(models)}"
        )
    entry = models[key]

    cfg["model_key"] = key
    cfg["model"] = entry["hf_id"]
    cfg["model_gated"] = bool(entry.get("gated", False))
    cfg["extra_body"] = entry.get("extra_body") or {}

    # -- environment overrides ------------------------------------------
    backend = os.environ.get("VOICE_RAG_LLM_BACKEND")
    if backend:
        cfg["backend"] = backend

    base_url = os.environ.get("VOICE_RAG_LLM_BASE_URL")
    if base_url:
        cfg.setdefault("server", {})["base_url"] = base_url

    return cfg


def api_key(cfg: dict) -> str:
    """
    The key the client sends. vLLM ignores it unless the server was started
    with --api-key, but the OpenAI SDK refuses to construct a client with an
    empty one, so we fall back to a placeholder.
    """
    env_name = (cfg.get("server") or {}).get("api_key_env", "VLLM_API_KEY")
    return os.environ.get(env_name) or "EMPTY"


if __name__ == "__main__":
    c = load_config()
    print(f"config file : {config_path()}")
    print(f"backend     : {c['backend']}")
    print(f"active model: {c['model_key']} -> {c['model']}")
    print(f"gated       : {c['model_gated']}")
    print(f"base_url    : {c['server']['base_url']}")
    print(f"logprobs    : enabled={c['logprobs']['enabled']} top_k={c['logprobs']['top_k']}")
    print(f"tracker slots: {list(c['tracker']['slots'])}")
