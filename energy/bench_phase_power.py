"""GPU power in the two LLM phases, measured with dispatch-free proxy kernels.

Why: on this shared, CPU-contended laptop, batch-1 Hugging Face generation
is launch-bound (each Qwen2.5-0.5B forward costs ~0.1 CPU-second of Python
dispatch and the GPU idles between kernels), so its power is not what a
CUDA-graph server such as vLLM draws. The A10G transfer in
energy/llm_estimate.py needs the power the GPU draws when the phase's
bottleneck is saturated, so we measure that directly:

  decode proxy  : bf16 GEMV y = W x with W = 2 GiB (cannot sit in L2), the
                  arithmetic of batch-1 decode: every weight read once per
                  token. Memory-bound.
  prefill proxy : bf16 GEMM (L x 4096) @ (4096 x 4096), L = 512 — the
                  arithmetic of prefilling a 512-token prompt. Compute-bound.

Each is run back-to-back for ~5 s windows under the paired protocol; we
report mean GPU power, the enforced power limit at the time, the power
fraction, and the achieved bandwidth / TFLOPS.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.bench_phase_power
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from energy.protocol import mean_ci, run_paired
from energy.sampler import EnergyMeter

OUT = Path(__file__).resolve().parent / "results" / "phase_power.json"


def main():
    import torch

    dev = "cuda"
    meter = EnergyMeter(rate_hz=50).start()
    time.sleep(2)
    n = 32768
    W = torch.randn(n, n, device=dev, dtype=torch.bfloat16)          # 2 GiB
    x = torch.randn(n, 1, device=dev, dtype=torch.bfloat16)
    A = torch.randn(512, 4096, device=dev, dtype=torch.bfloat16)
    B = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    sync = torch.cuda.synchronize

    res = {"started": datetime.now(timezone.utc).isoformat(), "device_start": meter.device_info(),
           "phases": {}}
    for name, fn, work in (
        ("decode_gemv_2GiB", lambda: W @ x, ("bytes", W.numel() * 2)),
        ("prefill_gemm_512x4096x4096", lambda: A @ B, ("flops", 2 * 512 * 4096 * 4096)),
    ):
        reps = 20
        def op(fn=fn):
            for _ in range(reps):
                fn()
        ts = run_paired(meter, name, op, n_trials=8, target_s=5.0, idle_s=3.0, sync=sync,
                        tail_s=0.5)
        s = ts.summary()
        limit = (meter.device_info()["gpu"] or {}).get("enforced_power_limit_w")
        t_kernel = s["latency_s"]["mean"] / reps
        rate = work[1] / t_kernel
        w = s["gpu_mean_active_w"]["mean"]
        s.update({"enforced_power_limit_w": limit, "power_fraction_of_limit": w / limit if limit else None,
                  "kernel_s": t_kernel,
                  ("achieved_gbs" if work[0] == "bytes" else "achieved_tflops"):
                      rate / 1e9 if work[0] == "bytes" else rate / 1e12,
                  "device_end": meter.device_info()})
        res["phases"][name] = s
        print(name, "power", w, "W of limit", limit, "->", s["power_fraction_of_limit"], flush=True)
    meter.stop()
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
