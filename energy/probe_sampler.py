"""Characterise the sampler on this machine.

Reports, for each requested rate: the achieved sampling rate idle and under a
GPU load, how often NVML's power reading and energy counter actually change
(their true update interval), the RAPL counter update interval, and the
agreement between the two GPU integration methods (power trapezoid vs the
cumulative energy counter) on a known load.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.probe_sampler
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from energy.sampler import EnergyMeter

OUT = Path(__file__).resolve().parent / "results" / "sampler_probe.json"


def _update_interval_ms(ts, ys):
    ts = np.asarray(ts); ys = np.asarray(ys)
    ok = np.isfinite(ys)
    ts, ys = ts[ok], ys[ok]
    ch = np.nonzero(np.diff(ys))[0]
    if len(ch) < 2:
        return None
    return float(np.median(np.diff(ts[ch + 1])) * 1e3)


def gpu_load(seconds: float):
    import torch

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t_end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < t_end:
        for _ in range(8):
            a @ b
        torch.cuda.synchronize()
        n += 8
    return n


def main():
    import torch  # noqa: F401  (warm CUDA context before timing)

    gpu_load(2.0)
    out = {"rates": {}}
    for rate in (20, 50, 100, 200):
        m = EnergyMeter(rate_hz=rate).start()
        time.sleep(1.0)
        with m.measure("idle") as idle:
            time.sleep(4.0)
        with m.measure("load") as load:
            gpu_load(6.0)
        time.sleep(1.5)  # let the power trace settle back
        ts, ps, es, cs = m.snapshot()
        m.stop()
        out["device"] = m.device_info()
        out["rates"][str(rate)] = {
            "achieved_hz_idle": idle.result.achieved_hz,
            "achieved_hz_load": load.result.achieved_hz,
            "nvml_power_update_ms": _update_interval_ms(ts, ps),
            "nvml_energy_update_ms": _update_interval_ms(ts, es),
            "rapl_update_ms": _update_interval_ms(ts, cs),
            "load_gpu_j_power": load.result.gpu_j_power,
            "load_gpu_j_counter": load.result.gpu_j_counter,
            "load_rel_diff_power_vs_counter": (
                (load.result.gpu_j_power - load.result.gpu_j_counter) / load.result.gpu_j_counter
                if load.result.gpu_j_counter else None),
            "load_gpu_mean_w": load.result.gpu_mean_w,
            "idle_gpu_w": idle.result.gpu_j / idle.result.duration_s,
            "idle_cpu_w": idle.result.cpu_pkg_j / idle.result.duration_s if idle.result.cpu_pkg_j else None,
        }
        print(rate, json.dumps(out["rates"][str(rate)], indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
