"""Per-turn energy model of the voice-RAG pipeline, with gCO2e conversion.

    from energy.model import turn_energy_j, marginal_retrieval

    e = turn_energy_j("retrieve", "faiss", n_prompt_tokens=534, n_gen_tokens=56,
                      asr_seconds=3.9, tts_chars=230)
    e["total_j"], e["components"]["llm_decode"], e["g_co2e"]

Every component is a triple (central, low, high) in joules. Parameters come
from energy/params.json, which energy/fit_params.py derives from the
committed measurements (energy/results/*.json); see energy/REPORT.md for how
each number was obtained and what its band means. A caller can pass its own
`params` dict (same schema) to explore assumptions.

Accounting conventions
----------------------
* All energies are NET (marginal above the idle floor of the device doing
  the work), unless `include_idle=True`, which adds the idle floor of the
  local machine and of the remote GPU over the turn's wall time (gross).
* Local CPU stages use the attributional CPU energy (CPU-seconds x package
  watts per busy logical CPU); local GPU stages use NVML net energy.
* The LLM runs remotely. Its energy is an ESTIMATE: observed A10G timings
  (E2/E3 logs) x an A10G power band derived from local measurements and
  published board power (energy/llm_estimate.py). `pue` multiplies remote
  (datacenter) energy; default 1.0 = GPU board energy only.
* Bands: low/high of a sum are the sums of the component lows/highs, i.e.
  components are treated as perfectly correlated. That is conservative
  (wide) for the total.
* gCO2e = kWh x grid factor (energy/grid_factors.json; default CEA v21.0
  weighted average for India). The remote GPU's real grid is not known;
  by default the same factor is applied (`remote_grid_g_per_kwh` overrides).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PARAMS_PATH = HERE / "params.json"
GRID_PATH = HERE / "grid_factors.json"

PATHS = ("skip", "retrieve")
PROFILES = ("faiss", "faiss_rerank", "remote_api")
J_PER_KWH = 3.6e6

_cache: dict = {}


def load_params(path: Path = PARAMS_PATH) -> dict:
    key = str(path)
    if key not in _cache:
        _cache[key] = json.loads(Path(path).read_text(encoding="utf-8"))
    return _cache[key]


def grid_factor(name: Optional[str] = None, path: Path = GRID_PATH) -> dict:
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    name = name or g["default"]
    f = dict(g["factors"][name])
    f["name"] = name
    return f


def _t(x) -> tuple:
    """Normalise a param to a (central, low, high) triple."""
    if isinstance(x, dict):
        return (float(x["central"]), float(x["low"]), float(x["high"]))
    if isinstance(x, (list, tuple)):
        return tuple(float(v) for v in x)
    return (float(x),) * 3


def _scale(t, s):
    return tuple(v * s for v in t)


def _add(*ts):
    return tuple(sum(v) for v in zip(*ts))


def turn_energy_j(path: str, profile: Optional[str], n_prompt_tokens: float,
                  n_gen_tokens: float, asr_seconds: float, tts_chars: float, *,
                  params: Optional[dict] = None, include_idle: bool = False,
                  turn_wall_s: Optional[float] = None, pue: float = 1.0,
                  grid: Optional[str] = None,
                  remote_grid_g_per_kwh: Optional[float] = None,
                  index_size: Optional[str] = None,
                  gpu_wake: Optional[bool] = None) -> dict:
    """Energy of one turn, by component, in joules (central, low, high).

    path            "skip" or "retrieve"
    profile         retrieval latency profile ("faiss" | "faiss_rerank" |
                    "remote_api"); ignored for skip
    n_prompt_tokens LLM prompt tokens (prefill), incl. passages when retrieving
    n_gen_tokens    generated tokens (decode)
    asr_seconds     seconds of user speech run through streaming ASR + VAD
    tts_chars       characters of answer text synthesised
    """
    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}")
    if path == "retrieve" and profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    for name, v in (("n_prompt_tokens", n_prompt_tokens), ("n_gen_tokens", n_gen_tokens),
                    ("asr_seconds", asr_seconds), ("tts_chars", tts_chars)):
        if v < 0:
            raise ValueError(f"{name} must be >= 0")
    p = params or load_params()
    loc, rem = p["local"], p["remote_llm"]

    comp = {}
    comp["vad"] = _scale(_t(loc["vad_j_per_audio_s"]), asr_seconds)
    comp["asr"] = _scale(_t(loc["asr_j_per_audio_s"]), asr_seconds)
    comp["tts"] = _add(_t(loc["tts_j_fixed"]), _scale(_t(loc["tts_j_per_char"]), tts_chars))

    if path == "retrieve":
        idx = index_size or loc.get("default_index_size", "60")
        embed = _t(loc["embed_query_j"])
        search = _t(loc["faiss_search_j"][idx])
        if profile == "remote_api":
            # identical compute on a retrieval server (datacenter PUE applies)
            # plus the network transfer of query + passages
            srv = _t(rem.get("retrieval_server_scale", 1.0))
            comp["retrieval_embed"] = tuple(a * b * pue for a, b in zip(embed, srv))
            comp["retrieval_search"] = tuple(a * b * pue for a, b in zip(search, srv))
            comp["retrieval_network"] = _t(loc.get("remote_api_network_j", 0.0))
        else:
            comp["retrieval_embed"] = embed
            comp["retrieval_search"] = search
            if profile == "faiss_rerank":
                comp["retrieval_rerank"] = _t(loc["rerank50_j"])

    use_wake = loc.get("gpu_wake_per_turn", False) if gpu_wake is None else gpu_wake
    if use_wake:
        comp["local_gpu_wake_tail"] = _t(loc["gpu_wake_tail_j"])

    comp["llm_prefill"] = _scale(_add(_t(rem["prefill_j_fixed"]),
                                      _scale(_t(rem["prefill_j_per_token"]), n_prompt_tokens)), pue)
    comp["llm_decode"] = _scale(_t(rem["decode_j_per_token"]), n_gen_tokens * pue)

    remote_keys = {"llm_prefill", "llm_decode"}
    if path == "retrieve" and profile == "remote_api":
        remote_keys |= {"retrieval_embed", "retrieval_search"}

    if include_idle:
        if turn_wall_s is None:
            raise ValueError("include_idle=True needs turn_wall_s")
        comp["idle_local"] = _scale(_t(loc["idle_w"]), turn_wall_s)
        comp["idle_remote_gpu"] = _scale(_t(rem["idle_w"]), turn_wall_s * pue)
        remote_keys.add("idle_remote_gpu")

    total = _add(*comp.values())
    remote = _add(*(v for k, v in comp.items() if k in remote_keys)) if remote_keys & comp.keys() else (0.0, 0.0, 0.0)
    local = tuple(t - r for t, r in zip(total, remote))

    gf = grid_factor(grid)
    g_local = gf["g_co2_per_kwh"]
    g_remote = remote_grid_g_per_kwh if remote_grid_g_per_kwh is not None else g_local
    co2 = tuple(l / J_PER_KWH * g_local + r / J_PER_KWH * g_remote for l, r in zip(local, remote))

    def tri(t):
        return {"central": t[0], "low": t[1], "high": t[2]}

    return {
        "path": path, "profile": profile if path == "retrieve" else None,
        "inputs": {"n_prompt_tokens": n_prompt_tokens, "n_gen_tokens": n_gen_tokens,
                   "asr_seconds": asr_seconds, "tts_chars": tts_chars, "pue": pue,
                   "include_idle": include_idle, "turn_wall_s": turn_wall_s},
        "components": {k: tri(v) for k, v in comp.items()},
        "local_j": tri(local), "remote_j": tri(remote), "total_j": tri(total),
        "g_co2e": tri(co2),
        "grid_factor": {"name": gf["name"], "g_co2_per_kwh_local": g_local,
                        "g_co2_per_kwh_remote": g_remote, "source": gf.get("source"),
                        "url": gf.get("url")},
        "params_version": p.get("version"),
    }


def marginal_retrieval(profile: str, *, n_prompt_skip: float, n_prompt_retrieve: float,
                       n_gen_skip: float, n_gen_retrieve: float, asr_seconds: float,
                       tts_chars_skip: float, tts_chars_retrieve: float, **kw) -> dict:
    """turn(retrieve, profile) - turn(skip), component by component."""
    r = turn_energy_j("retrieve", profile, n_prompt_retrieve, n_gen_retrieve, asr_seconds,
                      tts_chars_retrieve, **kw)
    s = turn_energy_j("skip", None, n_prompt_skip, n_gen_skip, asr_seconds, tts_chars_skip, **kw)
    keys = set(r["components"]) | set(s["components"])
    zero = {"central": 0.0, "low": 0.0, "high": 0.0}
    diff = {}
    for k in sorted(keys):
        a, b = r["components"].get(k, zero), s["components"].get(k, zero)
        diff[k] = {q: a[q] - b[q] for q in ("central", "low", "high")}
    return {"profile": profile, "components": diff,
            "total_j": {q: r["total_j"][q] - s["total_j"][q] for q in ("central", "low", "high")},
            "g_co2e": {q: r["g_co2e"][q] - s["g_co2e"][q] for q in ("central", "low", "high")},
            "retrieve": r, "skip": s}


def implied_quality_per_joule(lambda_c_per_retrieval: float, marginal_j: float) -> float:
    """If one retrieval is priced at lambda_c quality units and costs marginal_j
    joules, the implied energy price in quality units per joule."""
    if marginal_j <= 0:
        raise ValueError("marginal energy must be positive")
    return lambda_c_per_retrieval / marginal_j
