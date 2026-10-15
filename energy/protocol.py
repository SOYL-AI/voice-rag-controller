"""Paired idle/active measurement protocol and the statistics used everywhere.

Why paired windows
------------------
The host is a personal laptop with background load (sync clients, other
experiments; CPU package draw at "idle" was 34-44 W during this work). A
single idle calibration at the start would drift. Instead every active
window is flanked by idle windows of fixed length:

    idle_0, active_1, idle_1, active_2, idle_2, ..., active_n, idle_n

and the idle power subtracted from active_i is the mean of idle_{i-1} and
idle_i. Slow drift (thermal, background) cancels to first order.

Why repetition inside a window
------------------------------
NVML power on this laptop GPU refreshes every ~500 ms, and the RAPL package
counter every ~8-16 ms. An operation shorter than that cannot be resolved
alone. Each active window therefore runs the operation K times back-to-back,
with K chosen from a warm-up timing so the window lasts ~`target_s`, plus a
`tail_s` settle period (counted in the window, so the idle subtraction covers
it) that captures NVML reporting lag and part of the GPU clock-down tail.
Per-operation energy = (E_active - P_idle * D_active) / K.

CPU attribution (contended host)
--------------------------------
When other workloads hold the CPU package at its power limit, adding work
does not raise package power: it displaces other processes' cycles, and the
RAPL *marginal* reads ~0 (observed here, see REPORT.md). Each trial
therefore also records this process's CPU time per op (psutil, net of the
sampler thread's own rate measured in the idle windows) and the system-wide
busy logical-CPU count, giving an *attributional* CPU energy:

    cpu_attr_j = cpu_s_per_op * kappa,   kappa = P_pkg / busy_logical_cpus

(package watts per busy logical CPU in the same window). kappa includes the
uncore/static share, so it is an upper-side attribution; the report also
gives a TDP-based (CodeCarbon-style) lower figure.

GPU contamination
-----------------
The flanking idle GPU power is recorded per trial; a trial is flagged when
either flank exceeds `gpu_idle_flag_w` (another process was using the GPU).
Summaries are computed over unflagged trials and the flag count is reported.

Statistics: per-trial values are the unit; the reported interval is a
two-sided 95% Student-t CI on the mean across trials.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import psutil
from scipy import stats


def mean_ci(xs, conf: float = 0.95) -> dict:
    """Mean with a Student-t confidence interval (n-1 dof)."""
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None, "half": None}
    m = float(xs.mean())
    if n == 1:
        return {"n": 1, "mean": m, "sd": None, "lo": None, "hi": None, "half": None}
    sd = float(xs.std(ddof=1))
    half = float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / math.sqrt(n))
    return {"n": n, "mean": m, "sd": sd, "lo": m - half, "hi": m + half, "half": half}


def paired_net(active_j: float, active_s: float, idle_w_before: float,
               idle_w_after: Optional[float], k: int) -> float:
    """Per-op net energy from one active window and its flanking idle powers."""
    idle_w = idle_w_before if idle_w_after is None else 0.5 * (idle_w_before + idle_w_after)
    return (active_j - idle_w * active_s) / k


@dataclass
class TrialSet:
    name: str
    k: int
    op_s: list = field(default_factory=list)          # seconds per op
    gpu_net_j: list = field(default_factory=list)     # per op
    cpu_net_j: list = field(default_factory=list)     # per op
    gpu_gross_j: list = field(default_factory=list)   # per op
    cpu_gross_j: list = field(default_factory=list)   # per op
    gpu_idle_w: list = field(default_factory=list)
    cpu_idle_w: list = field(default_factory=list)
    gpu_counter_j: list = field(default_factory=list)  # per op, cross-check
    gpu_mean_w: list = field(default_factory=list)
    achieved_hz: list = field(default_factory=list)
    cpu_s_per_op: list = field(default_factory=list)
    kappa_w_per_cpu: list = field(default_factory=list)
    sys_busy_cpus: list = field(default_factory=list)
    flagged: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        keep = [not f for f in self.flagged] if self.flagged else [True] * len(self.op_s)
        all_flagged = bool(self.flagged) and not any(keep)
        if all_flagged:   # nothing clean: report all trials and say so
            keep = [True] * len(self.op_s)
        sel = lambda xs: [x for x, k in zip(xs, keep) if k]
        tot_net = [g + c for g, c in zip(self.gpu_net_j, self.cpu_net_j)]
        tot_gross = [g + c for g, c in zip(self.gpu_gross_j, self.cpu_gross_j)]
        attr = [c * k for c, k in zip(self.cpu_s_per_op, self.kappa_w_per_cpu)]
        return {
            "name": self.name, "k_per_window": self.k, "n_trials": len(self.op_s),
            "n_flagged_gpu_contended": int(sum(self.flagged)),
            "all_trials_flagged_used_anyway": all_flagged,
            "latency_s": mean_ci(sel(self.op_s)),
            "gpu_net_j": mean_ci(sel(self.gpu_net_j)),
            "cpu_net_j": mean_ci(sel(self.cpu_net_j)),
            "total_net_j": mean_ci(sel(tot_net)),
            "gpu_gross_j": mean_ci(sel(self.gpu_gross_j)),
            "cpu_gross_j": mean_ci(sel(self.cpu_gross_j)),
            "total_gross_j": mean_ci(sel(tot_gross)),
            "gpu_idle_w": mean_ci(self.gpu_idle_w),
            "cpu_idle_w": mean_ci(self.cpu_idle_w),
            "gpu_counter_j_crosscheck": mean_ci(sel(self.gpu_counter_j)),
            "gpu_mean_active_w": mean_ci(sel(self.gpu_mean_w)),
            "achieved_hz": mean_ci(self.achieved_hz),
            "cpu_s_per_op": mean_ci(sel(self.cpu_s_per_op)),
            "kappa_w_per_busy_cpu": mean_ci(self.kappa_w_per_cpu),
            "sys_busy_cpus": mean_ci(self.sys_busy_cpus),
            "cpu_attr_j": mean_ci(sel(attr)),
            "gpu_net_plus_cpu_attr_j": mean_ci(sel([g + a for g, a in zip(self.gpu_net_j, attr)])),
            "meta": self.meta,
        }


def _sys_busy_s() -> float:
    t = psutil.cpu_times()
    return sum(t) - t.idle


def _proc_cpu_s(proc) -> float:
    t = proc.cpu_times()
    return t.user + t.system


def run_paired(meter, name: str, fn: Callable[[], None], *, n_trials: int = 10,
               target_s: float = 5.0, idle_s: float = 3.0, tail_s: float = 0.75,
               k: Optional[int] = None, sync: Optional[Callable[[], None]] = None,
               warmup: int = 2, max_k: int = 100000, log: Callable = print,
               gpu_idle_flag_w: Optional[float] = None, **meta) -> TrialSet:
    """Measure per-operation energy of `fn` with the paired protocol."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    if k is None:
        t = time.perf_counter()
        reps = 0
        while time.perf_counter() - t < 0.5 or reps < 1:
            fn()
            reps += 1
        sync()
        per = (time.perf_counter() - t) / reps
        k = int(min(max_k, max(1, math.ceil(target_s / max(per, 1e-7)))))
    ts = TrialSet(name=name, k=k, meta=dict(meta))
    proc = psutil.Process()
    settle_s = []

    def idle():
        # Let the GPU clock back down to rest before measuring the idle floor.
        # Without this the flanking "idle" is the op's own clock-down tail
        # (~15 W for seconds after GPU work, vs ~3-4 W at rest) and the
        # subtraction would remove real energy. The tail beyond `tail_s` is
        # accounted once per turn as the GPU wake/tail overhead
        # (bench_stages: gpu_wake_tail).
        settle_s.append(settle_gpu(meter, gpu_rest_w(meter)))
        c0 = _proc_cpu_s(proc)
        r = meter.idle_window(idle_s)
        overhead = (_proc_cpu_s(proc) - c0) / r.duration_s   # sampler thread etc.
        return (r.gpu_j / r.duration_s if r.gpu_j is not None else 0.0,
                r.cpu_pkg_j / r.duration_s if r.cpu_pkg_j is not None else 0.0,
                overhead)

    prev = idle()
    for i in range(n_trials):
        c0, b0 = _proc_cpu_s(proc), _sys_busy_s()
        with meter.measure(name, tail_s=tail_s) as m:
            for _ in range(k):
                fn()
            sync()
        c1, b1 = _proc_cpu_s(proc), _sys_busy_s()
        r = m.result
        nxt = idle()
        work_s = r.meta["work_s"]
        ts.op_s.append(work_s / k)
        ts.gpu_idle_w.append(0.5 * (prev[0] + nxt[0]))
        ts.cpu_idle_w.append(0.5 * (prev[1] + nxt[1]))
        flag_w = gpu_idle_flag_w if gpu_idle_flag_w is not None else gpu_rest_w(meter) + 2.0
        ts.flagged.append(max(prev[0], nxt[0]) > flag_w)
        g = r.gpu_j or 0.0
        c = r.cpu_pkg_j or 0.0
        ts.gpu_net_j.append(paired_net(g, r.duration_s, prev[0], nxt[0], k))
        ts.cpu_net_j.append(paired_net(c, r.duration_s, prev[1], nxt[1], k))
        # gross per op = energy over the work part only, pro rata (tail excluded)
        ts.gpu_gross_j.append(g * (work_s / r.duration_s) / k)
        ts.cpu_gross_j.append(c * (work_s / r.duration_s) / k)
        if r.gpu_j_counter is not None:
            ts.gpu_counter_j.append((r.gpu_j_counter - ts.gpu_idle_w[-1] * r.duration_s) / k)
        ts.gpu_mean_w.append(r.gpu_mean_w)
        ts.achieved_hz.append(r.achieved_hz)
        overhead = 0.5 * (prev[2] + nxt[2])
        ts.cpu_s_per_op.append(max(0.0, (c1 - c0) - overhead * r.duration_s) / k)
        busy = (b1 - b0) / r.duration_s
        ts.sys_busy_cpus.append(busy)
        ts.kappa_w_per_cpu.append((c / r.duration_s) / busy if busy > 0 else float("nan"))
        prev = nxt
        log(f"  [{name}] trial {i + 1}/{n_trials}: {work_s / k * 1e3:9.3f} ms/op  "
            f"gpu_net {ts.gpu_net_j[-1]:9.4f} J  cpu_net {ts.cpu_net_j[-1]:8.4f} J  "
            f"cpu_s {ts.cpu_s_per_op[-1]:8.4f}  kappa {ts.kappa_w_per_cpu[-1]:5.2f} W  "
            f"gpu_idle {ts.gpu_idle_w[-1]:5.1f} W{'  FLAG' if ts.flagged[-1] else ''}  (k={k})",
            flush=True)
    ts.meta["gpu_settle_s"] = settle_s
    ts.meta["gpu_rest_w"] = gpu_rest_w(meter)
    return ts


_REST_W: dict = {}


def gpu_rest_w(meter, margin_w: float = 1.5) -> float:
    """Session GPU rest power (+margin), measured once after a long settle."""
    if id(meter) not in _REST_W:
        if not meter.has_gpu:
            _REST_W[id(meter)] = float("inf")
        else:
            time.sleep(8.0)
            r = meter.idle_window(3.0)
            _REST_W[id(meter)] = r.gpu_j / r.duration_s + margin_w
    return _REST_W[id(meter)]


def settle_gpu(meter, rest_w: float, hold_s: float = 1.0, timeout_s: float = 20.0) -> float:
    """Wait until reported GPU power has stayed <= rest_w for hold_s. Returns wait time."""
    if not meter.has_gpu or not np.isfinite(rest_w):
        return 0.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        ts, ps, _, _ = meter.snapshot()
        now = time.perf_counter()
        recent = ps[ts > now - hold_s]
        if len(recent) and np.nanmax(recent) <= rest_w and now - t0 >= hold_s:
            break
        time.sleep(0.1)
    return time.perf_counter() - t0
