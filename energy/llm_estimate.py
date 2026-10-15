"""Model-based energy estimate for the production LLM on the Modal A10G.

ESTIMATE, not a measurement. energy(request) ~ prefill_tokens * e_prefill
+ gen_tokens * e_decode, all NET of the A10G's idle floor (which is reported
separately so gross numbers can be formed).

Each coefficient is   e = (P_phase - P_idle) * t_phase_per_token   where

* t_phase_per_token is OBSERVED on the production endpoint itself
  (energy/results/a10g_timing.json, 2,250 E2/E3 turns of
  Mistral-7B-Instruct-v0.3 bf16 under vLLM on the Modal A10G):
    - decode: median inter-token time over each streamed answer (33 ms);
    - prefill: (median TTFT of retrieve turns - median TTFT of skip turns)
      / mean extra prompt tokens of retrieve (455, energy/results/
      token_budget.json). WAN RTT and client overhead cancel in the
      difference.
  Roofline sanity checks (printed): 14.5 GB of bf16 weights per decode step
  over 33 ms = ~440 GB/s, 73% of the A10G's 600 GB/s; prefill at 2*N FLOP
  per token over the observed time ~= 50% of the A10G's dense BF16 peak.

* P_phase is the A10G board power in that phase. It cannot be measured from
  here, so three scenarios are carried as (low, central, high):
    high    = 300 W: the A10G board power limit (cannot be exceeded).
    central = f_phase * 300 W, where f_phase = (mean GPU power) / (enforced
              power limit) measured LOCALLY on the RTX 4060 Laptop running the
              same phase (memory-bound batch-1 decode; compute-bound prefill)
              of a Qwen2.5 bf16 model (energy/results/llm_local.json, largest
              model measured). Assumes the phase loads both GPUs to the same
              fraction of their power limit.
    low     = f_phase * 150 W: as if the board drew like the NVIDIA A10, the
              150 W card built on the same GA102 die with the same 24 GB /
              600 GB/s GDDR6 (Modal describes the A10G as the AWS variant of
              the A10 with the same VRAM and bandwidth).
  P_idle (the serving container's GPU between requests, model resident) is
  an ASSUMPTION: 60 W (30-90 W). It only enters through the net subtraction
  and the gross idle term.

Sources for the A10G/A10 figures: NVIDIA A10 datasheet (150 W max TDP,
600 GB/s, 125 TF dense BF16; e.g. Lenovo Press LP1816); A10G 300 W board
power, 600 GB/s, ~70 TF dense FP16 (NVIDIA A10G datasheet as summarised at
https://getdeploying.com/gpus/nvidia-a10g); Modal on A10 vs A10G:
https://modal.com/blog/nvidia-a10g-price-article.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.llm_estimate
"""

from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"

A10G = {"board_power_w": 300.0, "a10_sibling_tdp_w": 150.0, "mem_bw_gbs": 600.0,
        "dense_bf16_tflops": 70.0}
MISTRAL_7B = {"n_params": 7.248e9, "bytes_per_param": 2}
IDLE_W = {"central": 60.0, "low": 30.0, "high": 90.0}

# Used only if energy/results/llm_local.json is missing (documented in REPORT).
FALLBACK_FRACTIONS = {"decode": 0.8, "prefill": 0.95}


def local_fractions(local: dict | None) -> dict:
    """Phase power as a fraction of the enforced limit, from the largest local model."""
    if not local or not local.get("models"):
        return {**FALLBACK_FRACTIONS, "source": "fallback (no local measurement)"}
    size, m = max(local["models"].items(), key=lambda kv: kv[1]["n_params"])
    c = m["coefficients"]
    lim = c.get("enforced_power_limit_w_end")
    ops = m["ops"]
    # limits move on a laptop (Dynamic Boost); use the limit recorded at the end of each op
    def frac(op):
        o = ops[op]
        w = o["gpu_mean_active_w"]["mean"]
        l = (o.get("device_end", {}).get("gpu") or {}).get("enforced_power_limit_w") or lim
        return w / l, w, l
    fd, wd, ld = frac(f"decode_N64_ctx530")
    fp, wp, lp = frac("prefill_L512")
    fp2, wp2, lp2 = frac("prefill_L2048")
    return {"decode": min(fd, 1.0), "prefill": min(max(fp, fp2), 1.0),
            "decode_w": wd, "decode_limit_w": ld, "prefill512_w": wp, "prefill2048_w": wp2,
            "prefill_limit_w": lp, "source": f"Qwen2.5-{size} bf16 on RTX 4060 Laptop"}


def a10g_llm_params(timing: dict, tokens: dict, local: dict | None) -> dict:
    t_dec = timing["decode_ms_per_token_median"] / 1e3
    d_ttft = timing["ttft_median_diff_retrieve_minus_skip_ms"] / 1e3
    d_tok = tokens["extra_prefill_tokens"]["mean"]
    t_pf = d_ttft / d_tok
    fr = local_fractions(local)

    def phase(f, t):
        P = {"low": f * A10G["a10_sibling_tdp_w"], "central": f * A10G["board_power_w"],
             "high": A10G["board_power_w"]}
        # net of idle: pair the low power with the high idle and vice versa so
        # the band is the widest consistent one
        net = {"low": max(P["low"] - IDLE_W["high"], 0.0) * t,
               "central": max(P["central"] - IDLE_W["central"], 0.0) * t,
               "high": max(P["high"] - IDLE_W["low"], 0.0) * t}
        return P, net

    Pd, ed = phase(fr["decode"], t_dec)
    Pp, ep = phase(fr["prefill"], t_pf)
    bw = MISTRAL_7B["n_params"] * MISTRAL_7B["bytes_per_param"] / t_dec / 1e9
    mfu = 2 * MISTRAL_7B["n_params"] / t_pf / (A10G["dense_bf16_tflops"] * 1e12)
    return {
        "decode_s_per_token": t_dec, "prefill_s_per_token": t_pf,
        "decode_power_w": Pd, "prefill_power_w": Pp, "idle_w": IDLE_W,
        "decode_j_per_token": ed, "prefill_j_per_token": ep,
        "prefill_j_fixed": 0.0,
        "power_fractions": fr,
        "sanity": {"decode_effective_bw_gbs": bw, "decode_bw_utilisation": bw / A10G["mem_bw_gbs"],
                   "prefill_mfu_vs_dense_bf16": mfu},
        "assumptions": {"a10g": A10G, "model": MISTRAL_7B,
                        "timing_source": timing.get("source"), "n_turns": timing.get("n_turns")},
    }


def main():
    timing = json.loads((RES / "a10g_timing.json").read_text())
    tokens = json.loads((RES / "token_budget.json").read_text())
    lp = RES / "llm_local.json"
    local = json.loads(lp.read_text()) if lp.exists() else None
    p = a10g_llm_params(timing, tokens, local)
    print(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
