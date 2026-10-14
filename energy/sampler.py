"""Energy sampling: NVML GPU power in a background thread + RAPL CPU package.

Usage
-----
    from energy.sampler import EnergyMeter

    meter = EnergyMeter(rate_hz=50)
    meter.start()
    meter.calibrate_idle(seconds=10)          # measured idle baseline
    with meter.measure("asr") as m:
        run_asr()
    print(m.result.as_dict())                 # gross and net J, per device
    meter.stop()

`measure()` is also exposed as a module-level context manager backed by a
lazily created default meter, so instrumented code can just write
``with measure("stage_name"): ...``.

Integration
-----------
GPU energy is computed two independent ways and both are reported:

* ``gpu_j_power``   — trapezoidal integral of the sampled power trace over
  the window [t0, t1], with linear interpolation at the window edges
  (`integrate_trapezoid`). This is the method the sampling rate matters for.
* ``gpu_j_counter`` — difference of NVML's cumulative energy counter
  (nvmlDeviceGetTotalEnergyConsumption, mJ) read ONCE at each window
  boundary. It is deliberately not polled by the sampling thread: on this
  Windows/WDDM laptop driver (592.82) the counter's apparent slope grows
  with the polling frequency (idle: ~4 W at 1 Hz, ~150 W at 10 Hz, ~290 W
  at 100 Hz, against a reported draw of ~3 W), so a polled counter is
  meaningless. Read only at the boundaries it agrees with the power
  integral under sustained load (28.7 W vs ~29 W) but is noisy at idle.
  It is therefore a cross-check, not the headline.

``gpu_j`` (the headline) is the power integral.

NVML power on this GPU refreshes every ~500 ms whatever the polling rate
(measured: energy/results/sampler_probe.json), with <=0.25 s reporting lag,
so the information rate is ~2 Hz even when the thread polls at 50 Hz.
Short operations must therefore be measured by repeating them inside a
multi-second window and dividing (see energy/bench_stages.py), and windows
can be extended by ``tail_s`` to capture the reporting lag.

CPU energy is the RAPL package counter read through the Windows Energy
Meter Interface (energy/winpdh.py), sampled on the same thread. It is a
cumulative counter with ~8 ms update granularity, so it is interpolated to
the window edges exactly like the GPU counter.

Net vs gross
------------
``net = gross - P_idle * duration`` per device, where P_idle comes from
`calibrate_idle` (or from a paired idle window supplied by the caller —
the benchmark scripts use paired idle windows to cancel slow drift).
Net energy is the *marginal* energy of the work; gross includes the share
of the machine's idle floor that elapsed during the work.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Pure integration math (unit-tested, no hardware needed)
# --------------------------------------------------------------------------


def interp_at(ts: np.ndarray, ys: np.ndarray, t: float) -> float:
    """Linear interpolation of a sampled series at time t (clamped at the ends)."""
    if len(ts) == 0:
        raise ValueError("empty series")
    return float(np.interp(t, ts, ys))


def integrate_trapezoid(ts, ps, t0: float, t1: float) -> float:
    """Integral of a sampled power trace P(t) over [t0, t1] (W*s = J).

    Samples are treated as point values joined linearly. The trace is clipped
    to the window, and the edge values at t0 and t1 are linearly interpolated
    from the neighbouring samples, so partial sample intervals at the edges
    are counted exactly once. Outside the sampled range the nearest sample is
    held constant (zero-order extrapolation).
    """
    ts = np.asarray(ts, dtype=float)
    ps = np.asarray(ps, dtype=float)
    if t1 < t0:
        raise ValueError("t1 < t0")
    if t1 == t0:
        return 0.0
    if len(ts) == 0:
        raise ValueError("no samples")
    if len(ts) == 1:
        return float(ps[0]) * (t1 - t0)
    inside = (ts > t0) & (ts < t1)
    grid_t = np.concatenate(([t0], ts[inside], [t1]))
    grid_p = np.concatenate(([interp_at(ts, ps, t0)], ps[inside], [interp_at(ts, ps, t1)]))
    # np.trapz was renamed np.trapezoid in NumPy 2.0; keep both working.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(grid_p, grid_t))


def counter_delta(ts, cum, t0: float, t1: float) -> float:
    """Increase of a cumulative counter between t0 and t1 (interpolated)."""
    ts = np.asarray(ts, dtype=float)
    cum = np.asarray(cum, dtype=float)
    return interp_at(ts, cum, t1) - interp_at(ts, cum, t0)


def net_energy(gross_j: float, idle_w: float, duration_s: float) -> float:
    """Marginal energy above an idle floor: gross - P_idle * duration."""
    return gross_j - idle_w * duration_s


def achieved_rate_hz(ts) -> float:
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 2:
        return 0.0
    return (len(ts) - 1) / (ts[-1] - ts[0])


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class StageEnergy:
    name: str
    t0: float
    t1: float
    duration_s: float
    gpu_j_power: Optional[float] = None
    gpu_j_counter: Optional[float] = None
    gpu_j: Optional[float] = None
    gpu_idle_w: Optional[float] = None
    gpu_net_j: Optional[float] = None
    gpu_mean_w: Optional[float] = None
    gpu_peak_w: Optional[float] = None
    cpu_pkg_j: Optional[float] = None
    cpu_idle_w: Optional[float] = None
    cpu_net_j: Optional[float] = None
    n_gpu_samples: int = 0
    achieved_hz: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def total_gross_j(self) -> float:
        return (self.gpu_j or 0.0) + (self.cpu_pkg_j or 0.0)

    @property
    def total_net_j(self) -> float:
        return (self.gpu_net_j or 0.0) + (self.cpu_net_j or 0.0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_gross_j"] = self.total_gross_j
        d["total_net_j"] = self.total_net_j
        return d


class _Handle:
    """What `with meter.measure(...) as m` yields; `.result` is set on exit."""

    def __init__(self, name):
        self.name = name
        self.result: Optional[StageEnergy] = None


# --------------------------------------------------------------------------
# Hardware sampler
# --------------------------------------------------------------------------


class EnergyMeter:
    """Background sampler for NVML GPU power/energy and RAPL CPU energy."""

    def __init__(self, rate_hz: float = 50.0, gpu_index: int = 0,
                 use_gpu: bool = True, use_cpu: bool = True,
                 max_samples: int = 2_000_000):
        self.rate_hz = float(rate_hz)
        self.gpu_index = gpu_index
        self.max_samples = max_samples
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._t: list = []
        self._p: list = []       # GPU power, W
        self._e: list = []       # GPU cumulative energy, J
        self._c: list = []       # CPU package cumulative energy, J
        self.gpu_idle_w: Optional[float] = None
        self.cpu_idle_w: Optional[float] = None
        self._nvml = None
        self._h = None
        self._rapl = None
        self._gpu_counter_ok = False
        if use_gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                try:
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)
                    self._gpu_counter_ok = True
                except pynvml.NVMLError:
                    self._gpu_counter_ok = False
            except Exception:
                self._nvml = None
        if use_cpu:
            try:
                from energy.winpdh import RaplPdhReader

                self._rapl = RaplPdhReader()
                if "RAPL_Package0_PKG" not in self._rapl.read_j():
                    self._rapl = None
            except Exception:
                self._rapl = None

    # ---- device metadata -------------------------------------------------
    @property
    def has_gpu(self) -> bool:
        return self._nvml is not None

    @property
    def has_cpu(self) -> bool:
        return self._rapl is not None

    def device_info(self) -> dict:
        info = {"gpu": None, "cpu_rapl": self.has_cpu}
        if not self.has_gpu:
            return info
        N, h = self._nvml, self._h

        def safe(f, *a):
            try:
                v = f(*a)
                return v.decode() if isinstance(v, bytes) else v
            except Exception:
                return None

        info["gpu"] = {
            "name": safe(N.nvmlDeviceGetName, h),
            "driver": safe(N.nvmlSystemGetDriverVersion),
            "nvml": safe(N.nvmlSystemGetNVMLVersion),
            "temperature_c": safe(N.nvmlDeviceGetTemperature, h, N.NVML_TEMPERATURE_GPU),
            "pstate": safe(N.nvmlDeviceGetPerformanceState, h),
            "sm_clock_mhz": safe(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_SM),
            "mem_clock_mhz": safe(N.nvmlDeviceGetClockInfo, h, N.NVML_CLOCK_MEM),
            "sm_clock_max_mhz": safe(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM),
            "mem_clock_max_mhz": safe(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_MEM),
            "enforced_power_limit_w": (safe(N.nvmlDeviceGetEnforcedPowerLimit, h) or 0) / 1000 or None,
            "default_power_limit_w": (safe(N.nvmlDeviceGetPowerManagementDefaultLimit, h) or 0) / 1000 or None,
            "energy_counter_supported": self._gpu_counter_ok,
        }
        return info

    # ---- sampling thread -------------------------------------------------
    def _read_once(self):
        t = time.perf_counter()
        p = e = c = np.nan
        if self._nvml is not None:
            try:
                p = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            except Exception:
                pass
        if self._rapl is not None:
            try:
                c = self._rapl.read_j().get("RAPL_Package0_PKG", np.nan)
            except Exception:
                pass
        return t, p, e, c

    def _loop(self):
        period = 1.0 / self.rate_hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            t, p, e, c = self._read_once()
            with self._lock:
                self._t.append(t)
                self._p.append(p)
                self._e.append(e)
                self._c.append(c)
                if len(self._t) > self.max_samples:
                    drop = len(self._t) - self.max_samples
                    del self._t[:drop], self._p[:drop], self._e[:drop], self._c[:drop]
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                nxt = time.perf_counter()

    def start(self):
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="energy-sampler", daemon=True)
        self._thread.start()
        time.sleep(3.0 / self.rate_hz)
        return self

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def clear(self):
        with self._lock:
            self._t.clear(); self._p.clear(); self._e.clear(); self._c.clear()

    def snapshot(self):
        with self._lock:
            return (np.array(self._t), np.array(self._p),
                    np.array(self._e), np.array(self._c))

    # ---- windows ---------------------------------------------------------
    def _wait_past(self, t1: float):
        """Block until at least one sample later than t1 exists."""
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._t and self._t[-1] > t1:
                    return
            time.sleep(0.5 / self.rate_hz)

    def window(self, name: str, t0: float, t1: float,
               gpu_idle_w: Optional[float] = None,
               cpu_idle_w: Optional[float] = None) -> StageEnergy:
        """Energy over an arbitrary past window [t0, t1] of the sample stream."""
        ts, ps, es, cs = self.snapshot()
        r = StageEnergy(name=name, t0=t0, t1=t1, duration_s=t1 - t0)
        sel = (ts >= t0) & (ts <= t1)
        r.n_gpu_samples = int(sel.sum())
        r.achieved_hz = achieved_rate_hz(ts[sel]) if sel.sum() > 1 else 0.0
        gi = self.gpu_idle_w if gpu_idle_w is None else gpu_idle_w
        ci = self.cpu_idle_w if cpu_idle_w is None else cpu_idle_w
        if self.has_gpu and np.isfinite(ps).any():
            ok = np.isfinite(ps)
            r.gpu_j_power = integrate_trapezoid(ts[ok], ps[ok], t0, t1)
            if sel.any():
                r.gpu_mean_w = r.gpu_j_power / r.duration_s if r.duration_s > 0 else None
                r.gpu_peak_w = float(np.nanmax(ps[sel]))
            r.gpu_j = r.gpu_j_power
            if gi is not None:
                r.gpu_idle_w = gi
                r.gpu_net_j = net_energy(r.gpu_j, gi, r.duration_s)
        if self.has_cpu and np.isfinite(cs).any():
            ok = np.isfinite(cs)
            r.cpu_pkg_j = counter_delta(ts[ok], cs[ok], t0, t1)
            if ci is not None:
                r.cpu_idle_w = ci
                r.cpu_net_j = net_energy(r.cpu_pkg_j, ci, r.duration_s)
        return r

    def read_gpu_counter_j(self) -> Optional[float]:
        if not (self.has_gpu and self._gpu_counter_ok):
            return None
        try:
            return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._h) / 1000.0
        except Exception:
            return None

    @contextlib.contextmanager
    def measure(self, name: str, tail_s: float = 0.0, **meta):
        """Measure the enclosed block. ``tail_s`` extends the window after the
        block ends (sleeping) so reporting lag / clock-down tail is included;
        the extension is counted in ``duration_s`` and so in the idle
        subtraction."""
        if self._thread is None:
            self.start()
        h = _Handle(name)
        c0 = self.read_gpu_counter_j()
        t0 = time.perf_counter()
        try:
            yield h
        finally:
            t_work = time.perf_counter()
            if tail_s > 0:
                time.sleep(tail_s)
            t1 = time.perf_counter()
            c1 = self.read_gpu_counter_j()
            self._wait_past(t1)
            h.result = self.window(name, t0, t1)
            if c0 is not None and c1 is not None:
                h.result.gpu_j_counter = c1 - c0
            h.result.meta.update(meta)
            h.result.meta["work_s"] = t_work - t0
            h.result.meta["tail_s"] = t1 - t_work

    def idle_window(self, seconds: float) -> StageEnergy:
        """Measure an idle window (caller must keep the machine quiet)."""
        with self.measure("idle") as m:
            time.sleep(seconds)
        return m.result

    def calibrate_idle(self, seconds: float = 10.0) -> dict:
        r = self.idle_window(seconds)
        if r.gpu_j is not None:
            self.gpu_idle_w = r.gpu_j / r.duration_s
        if r.cpu_pkg_j is not None:
            self.cpu_idle_w = r.cpu_pkg_j / r.duration_s
        return {"gpu_idle_w": self.gpu_idle_w, "cpu_idle_w": self.cpu_idle_w,
                "seconds": seconds, "achieved_hz": r.achieved_hz}


# --------------------------------------------------------------------------
# Module-level convenience
# --------------------------------------------------------------------------

_DEFAULT: Optional[EnergyMeter] = None


def default_meter(rate_hz: float = 50.0) -> EnergyMeter:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = EnergyMeter(rate_hz=rate_hz).start()
    return _DEFAULT


@contextlib.contextmanager
def measure(stage_name: str, **meta):
    """``with measure("asr") as m: ...`` then ``m.result``."""
    with default_meter().measure(stage_name, **meta) as h:
        yield h
