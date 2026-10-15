"""Derive energy/params.json (and the GPU-placement variant) from the results.

Inputs (all committed under energy/results/): stages.json, llm_local.json,
a10g_timing.json, token_budget.json. Output schema is what energy/model.py
reads. Every triple is (central, low, high) in joules; the band rules:

Local CPU stages (production placement):
    central = cpu_s_per_op * kappa            (attributional, measured kappa)
    low     = min(cpu_s_per_op * 28 W / 22,   (Core Ultra 7 155H processor
                  RAPL net marginal if > 0)    base power spread over its 22
                                               logical CPUs, i.e. the CodeCarbon-
                                               style TDP x utilisation share; or
                                               the measured package marginal,
                                               which is depressed when the
                                               package sits at its power cap)
    high    = 1.5 * central                   (judgement: a non-contended
                                               machine runs the op at higher
                                               clocks/voltage per CPU-second)
    + any measured GPU net energy (clipped at 0) in all three.
Local GPU stages:
    central = gpu_net mean + CPU central; low/high use the 95% CI ends of the
    GPU net energy (clipped at 0) and the CPU low/high above.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.fit_params
"""

from __future__ import annotations

import json
from pathlib import Path

from energy.llm_estimate import a10g_llm_params

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
BASE_W_PER_CPU = 28.0 / 22.0
HIGH_CPU_FACTOR = 1.5


def _tri(c, lo, hi):
    return {"central": float(c), "low": float(min(lo, c)), "high": float(max(hi, c))}


def is_cpu_only(name: str) -> bool:
    """Stages that never touch the GPU; their GPU 'net' is pure noise from
    other workloads on the shared host and is excluded."""
    return not ("cuda" in name or name == "gpu_wake_tail")


def stage_triple(s: dict, name: str = "") -> dict:
    """(central, low, high) for one measured stage; see module docstring."""
    if name and is_cpu_only(name):
        s = dict(s)
        s["gpu_net_j"] = {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    cpu_s = s["cpu_s_per_op"]["mean"] or 0.0
    attr = s["cpu_attr_j"]["mean"] or 0.0
    rapl = s["cpu_net_j"]["mean"]
    cpu_low = cpu_s * BASE_W_PER_CPU
    if rapl is not None and rapl > 0:
        cpu_low = min(cpu_low, rapl)        # the measured package marginal, if lower
    g = s["gpu_net_j"]
    g_c = max(g["mean"] or 0.0, 0.0)
    g_lo = max(g["lo"] if g["lo"] is not None else g_c, 0.0)
    g_hi = max(g["hi"] if g["hi"] is not None else g_c, 0.0)
    return _tri(attr + g_c, cpu_low + g_lo, HIGH_CPU_FACTOR * attr + g_hi)


def scale(t: dict, k: float) -> dict:
    return {q: v * k for q, v in t.items()}


def add(*ts) -> dict:
    return {q: sum(t[q] for t in ts) for q in ("central", "low", "high")}


def build(stages: dict, util: dict, llm: dict, placement: str) -> dict:
    S = stages["stages"]
    utt = stages["utterance"]
    tri = lambda name: stage_triple(S[name], name)
    gpu = placement == "gpu"
    wake = _tri(0, 0, 0)
    if "gpu_wake_tail" in S:   # wake/tail is GPU-only: use its own CI directly
        w = S["gpu_wake_tail"]["gpu_net_j"]
        wake = _tri(max(w["mean"], 0), max(w["lo"], 0), max(w["hi"], 0))

    asr_name = "asr_stream_cuda_fp16" if gpu and "asr_stream_cuda_fp16" in S else "asr_stream_cpu_int8"
    emb_name = "embed_query_cuda" if gpu else "embed_query_cpu"

    # TTS: linear fit over short/typical/long in-process, + the per-turn
    # overhead of today's subprocess-per-clause implementation
    xs, ys = [], []
    for lab in ("short", "typical", "long"):
        n = f"tts_piper_inproc_{lab}"
        if n in S:
            xs.append(S[n]["chars"]); ys.append(stage_triple(S[n], n))
    import numpy as np
    fit = {}
    for q in ("central", "low", "high"):
        b, a = np.polyfit(xs, [y[q] for y in ys], 1) if len(xs) >= 2 else (ys[0][q] / xs[0], 0.0)
        fit[q] = (max(a, 0.0), max(b, 0.0))
    order = lambda d: _tri(d["central"], d["low"], d["high"])
    tts_fixed = order({q: fit[q][0] for q in fit})
    tts_per_char = order({q: fit[q][1] for q in fit})
    sub_over = None
    if "tts_piper_subprocess_typical" in S and "tts_piper_inproc_typical" in S:
        sub_over = {q: max(stage_triple(S["tts_piper_subprocess_typical"], "tts")[q]
                           - stage_triple(S["tts_piper_inproc_typical"], "tts")[q], 0.0)
                    for q in ("central", "low", "high")}
        if not gpu:   # production = as implemented today
            tts_fixed = add(tts_fixed, sub_over)

    rerank = tri("rerank50_minilm_cuda") if "rerank50_minilm_cuda" in S else _tri(0, 0, 0)
    if not gpu:
        # production keeps ASR/embedding on the CPU, so a CUDA reranker wakes
        # the GPU from rest on every retrieval and pays the clock-down tail
        rerank = add(rerank, wake)

    local = {
        "vad_j_per_audio_s": scale(tri("vad_utterance_cpu"), 1 / utt["total_s"]),
        "asr_j_per_audio_s": scale(tri(asr_name), 1 / utt["speech_s"]),
        "tts_j_fixed": tts_fixed, "tts_j_per_char": tts_per_char,
        "tts_subprocess_overhead_j": sub_over,
        "embed_query_j": tri(emb_name),
        "faiss_search_j": {lab: tri(f"faiss_search_{lab}") for lab in ("60", "10k", "100k")
                           if f"faiss_search_{lab}" in S},
        "default_index_size": "60",
        "rerank50_j": rerank,
        "remote_api_network_j": _tri(0.0, 0.0, 0.0),
        "gpu_wake_tail_j": wake,
        "gpu_wake_per_turn": gpu,
        "idle_w": _tri(util["cpu_idle_w"] + util["gpu_idle_w"],
                       util["cpu_idle_w"] * 0.25 + util["gpu_idle_w"],
                       util["cpu_idle_w"] + util["gpu_idle_w"]),
    }
    remote = {
        "prefill_j_fixed": llm["prefill_j_fixed"],
        "prefill_j_per_token": llm["prefill_j_per_token"],
        "decode_j_per_token": llm["decode_j_per_token"],
        "idle_w": llm["idle_w"],
        "retrieval_server_scale": _tri(1.0, 0.5, 2.0),
    }
    return {"version": f"energy-params-1/{placement}", "placement": placement,
            "local": local, "remote_llm": remote,
            "llm_estimate_detail": llm,
            "sources": ["energy/results/stages.json", "energy/results/llm_local.json",
                        "energy/results/a10g_timing.json", "energy/results/token_budget.json"]}


def main():
    stages = json.loads((RES / "stages.json").read_text())
    timing = json.loads((RES / "a10g_timing.json").read_text())
    tokens = json.loads((RES / "token_budget.json").read_text())
    lp = RES / "llm_local.json"
    pp = RES / "phase_power.json"
    llm = a10g_llm_params(timing, tokens, json.loads(lp.read_text()) if lp.exists() else None,
                          json.loads(pp.read_text()) if pp.exists() else None)
    S = stages["stages"]
    cpu_idle = [s["cpu_idle_w"]["mean"] for s in S.values() if "cpu_idle_w" in s and s["cpu_idle_w"]["mean"]]
    gpu_idle = [s["gpu_idle_w"]["mean"] for s in S.values() if "gpu_idle_w" in s and s.get("gpu_idle_w", {}).get("mean")]
    util = {"cpu_idle_w": float(sum(cpu_idle) / len(cpu_idle)),
            "gpu_idle_w": float(min(gpu_idle))}
    for placement, name in (("production", "params.json"), ("gpu", "params_gpu_placement.json")):
        p = build(stages, util, llm, placement)
        (HERE / name).write_text(json.dumps(p, indent=2))
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
