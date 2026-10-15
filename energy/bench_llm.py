"""Local LLM energy coefficients: prefill J/token and decode J/token.

The production LLM (Mistral-7B-Instruct-v0.3, bf16, vLLM, Modal A10G) cannot
be sampled from here, and a 7B bf16 model does not fit this 8 GB GPU (the
gated Mistral weights are also not downloadable without a token, which rules
out a 4-bit Mistral). We therefore measure the same transformer workload on
Qwen2.5-{0.5B,1.5B,3B}-Instruct in bf16 on the RTX 4060 Laptop and use the
results for three things (energy/llm_estimate.py):

1. the *shape* of the cost: E_prefill(L) = a + b*L is linear in prompt
   length; decode costs a roughly constant J/token that is far larger than
   the prefill J/token;
2. how J/token scales with parameter count (decode ~ bytes moved, prefill ~
   FLOPs), checked across the three sizes;
3. the GPU's power during each phase as a fraction of its enforced power
   limit (memory-bound decode vs compute-bound prefill), which is the one
   number transferred to the A10G.

Ops (each a paired-protocol op, energy/protocol.py):
  prefill_L{L}   one forward pass over L prompt tokens (batch 1, fresh cache)
  decode_N{N}_ctx{C}  N single-token decode steps on top of a C-token cache
                 (the C-token prefill happens outside the measured window)

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.bench_llm [--models 0.5B,1.5B,3B]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from energy.paths import model_dir  # noqa: E402
from energy.protocol import mean_ci, run_paired  # noqa: E402
from energy.sampler import EnergyMeter  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "llm_local.json"
PREFILL_LENS = (128, 512, 2048)
DECODE_CTX = (80, 530)     # ~ skip / retrieve prompt lengths (energy/results/token_budget.json)
DECODE_N = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.5B,1.5B,3B")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--target-s", type=float, default=5.0)
    ap.add_argument("--idle-s", type=float, default=3.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    meter = EnergyMeter(rate_hz=50).start()
    time.sleep(2)
    res = {"started": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
           "device_start": meter.device_info(), "models": {}}
    if Path(args.out).exists():
        res["models"].update(json.loads(Path(args.out).read_text()).get("models", {}))

    corpus = "\n".join(p.read_text(encoding="utf-8")
                       for p in sorted((REPO / "retrieval" / "corpus").glob("*.txt")))
    sync = torch.cuda.synchronize

    for size in args.models.split(","):
        rid = f"Qwen/Qwen2.5-{size}-Instruct"
        tok = AutoTokenizer.from_pretrained(str(model_dir(rid)))
        # load straight onto the GPU: host RAM was nearly exhausted by other workloads
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir(rid)), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True, device_map="cuda").eval()
        n_params = sum(p.numel() for p in model.parameters())
        ids_all = tok(corpus, return_tensors="pt").input_ids[0]
        entry = {"repo": rid, "n_params": n_params,
                 "weight_bytes": n_params * 2, "ops": {}}
        print(f"=== {rid}: {n_params / 1e9:.3f} B params", flush=True)

        for L in PREFILL_LENS:
            ids = ids_all[:L].unsqueeze(0).cuda()

            @torch.no_grad()
            def prefill(ids=ids):
                model(input_ids=ids, use_cache=True)
            ts = run_paired(meter, f"prefill_L{L}", prefill, n_trials=args.trials,
                            target_s=args.target_s, idle_s=args.idle_s, sync=sync)
            s = ts.summary(); s["tokens"] = L
            s["device_end"] = meter.device_info()
            entry["ops"][f"prefill_L{L}"] = s

        for C in DECODE_CTX:
            ids = ids_all[:C].unsqueeze(0).cuda()
            state = {}

            @torch.no_grad()
            def setup(ids=ids):
                out = model(input_ids=ids, use_cache=True)
                state["pkv"] = out.past_key_values
                state["next"] = out.logits[:, -1:].argmax(-1)

            @torch.no_grad()
            def decode():
                # the cache is rebuilt between ops by `setup`, outside the
                # timed loop: run_paired calls `fn`, so wrap setup+decode and
                # subtract the measured prefill of C tokens afterwards.
                setup()
                pkv, nxt = state["pkv"], state["next"]
                for _ in range(DECODE_N):
                    out = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
                    pkv = out.past_key_values
                    nxt = out.logits[:, -1:].argmax(-1)

            name = f"decode_N{DECODE_N}_ctx{C}"
            ts = run_paired(meter, name, decode, n_trials=args.trials,
                            target_s=args.target_s, idle_s=args.idle_s, sync=sync)
            s = ts.summary(); s["new_tokens"] = DECODE_N; s["ctx"] = C
            # prefill of C tokens, to subtract
            pf = run_paired(meter, f"prefill_L{C}_for_decode", setup, n_trials=args.trials,
                            target_s=args.target_s, idle_s=args.idle_s, sync=sync)
            s["prefill_ctx"] = pf.summary()
            s["device_end"] = meter.device_info()
            entry["ops"][name] = s

        # Derived coefficients
        Ls = np.array(PREFILL_LENS, dtype=float)
        E = np.array([entry["ops"][f"prefill_L{L}"]["gpu_net_j"]["mean"] for L in PREFILL_LENS])
        T = np.array([entry["ops"][f"prefill_L{L}"]["latency_s"]["mean"] for L in PREFILL_LENS])
        b, a = np.polyfit(Ls, E, 1)
        bt, at = np.polyfit(Ls, T, 1)
        dec = {}
        for C in DECODE_CTX:
            o = entry["ops"][f"decode_N{DECODE_N}_ctx{C}"]
            e_tok = (o["gpu_net_j"]["mean"] - o["prefill_ctx"]["gpu_net_j"]["mean"]) / DECODE_N
            t_tok = (o["latency_s"]["mean"] - o["prefill_ctx"]["latency_s"]["mean"]) / DECODE_N
            dec[str(C)] = {"j_per_token": e_tok, "s_per_token": t_tok,
                           "mean_power_w": o["gpu_mean_active_w"]["mean"],
                           "achieved_bw_gbs": entry["weight_bytes"] / t_tok / 1e9}
        lim = (meter.device_info()["gpu"] or {}).get("enforced_power_limit_w")
        entry["coefficients"] = {
            "prefill_j_per_token": float(b), "prefill_j_fixed": float(a),
            "prefill_s_per_token": float(bt), "prefill_s_fixed": float(at),
            "prefill_mean_power_w_L2048": entry["ops"]["prefill_L2048"]["gpu_mean_active_w"]["mean"],
            "decode": dec,
            "enforced_power_limit_w_end": lim,
        }
        print(json.dumps(entry["coefficients"], indent=1), flush=True)
        res["models"][size] = entry
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        del model
        torch.cuda.empty_cache()

    meter.stop()
    res["finished"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
