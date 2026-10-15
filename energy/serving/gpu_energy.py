"""Per-request GPU energy logging inside the vLLM serving container.

NOT DEPLOYED. This is the in-container half of the energy instrumentation
for the remote LLM (Mistral-7B-Instruct-v0.3 under vLLM on a Modal A10G).
The laptop cannot sample that GPU, so the GPU has to sample itself.

What it does
------------
* `GpuEnergyMonitor` — a daemon thread that polls NVML on the serving GPU at
  `rate_hz` (default 20 Hz) and keeps a ring buffer of
  (t, power_W, cumulative_energy_J). On datacenter GPUs under Linux the
  cumulative counter (nvmlDeviceGetTotalEnergyConsumption) is the primary
  measurement and the power integral is the cross-check. (On the Windows
  laptop the counter is distorted by polling, see energy/sampler.py, which
  is why the laptop sampler does the opposite. Check `counter_vs_power` in
  the log to confirm the counter is sane on the A10G before trusting it.)
  It also appends a 1 Hz timeline to `timeline.jsonl` so the idle floor and
  the between-request idle share can be computed offline.

* `EnergyMiddleware` — a pure-ASGI middleware (same shape as the existing
  BearerAuthMiddleware in llm/modal_vllm.py) that brackets every
  /v1/completions and /v1/chat/completions request: it records the wall
  window, GPU energy over the window (counter and integral), the number of
  requests in flight (energy is only exclusive when that is 1), and the
  token usage parsed from the response body (`usage.prompt_tokens`,
  `usage.completion_tokens`; for streaming requests the client must send
  `stream_options: {"include_usage": true}` for the usage chunk to exist).
  One JSON line per request goes to `requests.jsonl`. A response header
  `x-energy-request-id` lets the client join its own log to the server log,
  and `GET /energy/requests?since=<unix_ts>` returns the server log lines so
  a run can be pulled without a separate Modal call.

Attribution rules (written down so the analysis cannot drift)
-------------------------------------------------------------
* gross_j       = GPU energy over [t_start, t_end] of the request.
* idle_w        = median power over the most recent >= 30 s with zero
                  requests in flight (from the timeline); recorded per request.
* net_j         = gross_j - idle_w * duration.
* If `inflight_max > 1` the window energy is shared; the analysis must
  apportion it (e.g. by tokens) or drop those rows. The voice-rag runs are
  sequential, so this should not occur.

Only CPU-light work happens on the request path: two NVML reads and a
dict update. Parsing the body is a bounded scan of the last chunk(s).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

LOG_DIR = Path(os.environ.get("VOICE_RAG_ENERGY_LOG_DIR", "/energy-logs"))
TRACKED_PATHS = ("/v1/completions", "/v1/chat/completions")


def _trapz(ts, ps, t0, t1):
    """Trapezoid integral of (ts, ps) over [t0, t1] with edge interpolation."""
    if not ts:
        return None
    pts = []

    def interp(t):
        if t <= ts[0]:
            return ps[0]
        if t >= ts[-1]:
            return ps[-1]
        lo, hi = 0, len(ts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        f = (t - ts[lo]) / (ts[hi] - ts[lo]) if ts[hi] > ts[lo] else 0.0
        return ps[lo] + f * (ps[hi] - ps[lo])

    pts.append((t0, interp(t0)))
    pts.extend((t, p) for t, p in zip(ts, ps) if t0 < t < t1)
    pts.append((t1, interp(t1)))
    return sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))


class GpuEnergyMonitor:
    def __init__(self, rate_hz: float = 20.0, gpu_index: int = 0,
                 window_s: float = 900.0, nvml=None, log_dir: Path = LOG_DIR):
        if nvml is None:
            import pynvml as nvml  # nvidia-ml-py
        self.nvml = nvml
        nvml.nvmlInit()
        self.h = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.rate_hz = rate_hz
        self.buf = deque(maxlen=int(window_s * rate_hz))
        self.inflight = 0
        self.last_busy_t = 0.0
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        try:
            nvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.counter_ok = True
        except Exception:
            self.counter_ok = False
        self.device = self._device_info()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-energy")

    def _device_info(self):
        N, h = self.nvml, self.h

        def safe(f, *a):
            try:
                v = f(*a)
                return v.decode() if isinstance(v, bytes) else v
            except Exception:
                return None

        return {
            "name": safe(N.nvmlDeviceGetName, h),
            "driver": safe(N.nvmlSystemGetDriverVersion),
            "power_limit_w": (safe(N.nvmlDeviceGetEnforcedPowerLimit, h) or 0) / 1000,
            "sm_clock_max_mhz": safe(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM),
            "mem_clock_max_mhz": safe(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_MEM),
            "energy_counter_supported": self.counter_ok,
        }

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def read(self):
        t = time.time()
        p = self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
        e = (self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0
             if self.counter_ok else None)
        return t, p, e

    def _loop(self):
        period = 1.0 / self.rate_hz
        next_tl = 0.0
        tl = open(self.log_dir / "timeline.jsonl", "a", encoding="utf-8")
        while not self._stop.is_set():
            try:
                t, p, e = self.read()
                with self._lock:
                    self.buf.append((t, p, e, self.inflight))
                if t >= next_tl:
                    N, h = self.nvml, self.h
                    try:
                        sm = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)
                        temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
                    except Exception:
                        sm = temp = None
                    tl.write(json.dumps({"t": t, "power_w": p, "energy_j": e,
                                         "inflight": self.inflight, "sm_mhz": sm,
                                         "temp_c": temp}) + "\n")
                    tl.flush()
                    next_tl = t + 1.0
            except Exception:
                pass
            self._stop.wait(period)

    def idle_w(self, min_s: float = 30.0) -> Optional[float]:
        """Median power over recent samples with nothing in flight."""
        with self._lock:
            rows = [r for r in self.buf if r[3] == 0]
        if len(rows) < 2 or rows[-1][0] - rows[0][0] < min_s:
            return None
        ps = sorted(r[1] for r in rows[-int(min_s * self.rate_hz):])
        return ps[len(ps) // 2]

    def integral(self, t0, t1):
        with self._lock:
            rows = list(self.buf)
        ts = [r[0] for r in rows]
        ps = [r[1] for r in rows]
        return _trapz(ts, ps, t0, t1)


def _parse_usage(body_tail: bytes) -> dict:
    """Pull `usage` out of a JSON body or the last SSE chunks that carry it."""
    text = body_tail.decode("utf-8", errors="ignore")
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if '"usage"' in line:
            candidates.append(line)
    if not candidates and '"usage"' in text:
        candidates.append(text)
    for c in reversed(candidates):
        try:
            u = json.loads(c).get("usage") or {}
        except Exception:
            continue
        if u:
            return {"prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens")}
    return {"prompt_tokens": None, "completion_tokens": None}


class EnergyMiddleware:
    """ASGI middleware: one energy record per completion request."""

    monitor: Optional[GpuEnergyMonitor] = None   # set once per process
    TAIL_BYTES = 64 * 1024

    def __init__(self, app):
        self.app = app
        if EnergyMiddleware.monitor is None:
            EnergyMiddleware.monitor = GpuEnergyMonitor().start()
        self.mon = EnergyMiddleware.monitor
        self.log_path = self.mon.log_dir / "requests.jsonl"

    async def _serve_log(self, scope, send):
        since = 0.0
        for kv in scope.get("query_string", b"").decode().split("&"):
            if kv.startswith("since="):
                try:
                    since = float(kv[6:])
                except ValueError:
                    pass
        rows = []
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("t_start", 0) >= since:
                    rows.append(r)
        body = json.dumps({"device": self.mon.device, "requests": rows}).encode()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path == "/energy/requests" and scope.get("method") == "GET":
            await self._serve_log(scope, send)
            return
        if path not in TRACKED_PATHS:
            await self.app(scope, receive, send)
            return

        rid = uuid.uuid4().hex
        mon = self.mon
        tail = bytearray()
        state = {"inflight_max": 0, "status": None}

        async def send_wrapper(msg):
            if msg["type"] == "http.response.start":
                state["status"] = msg.get("status")
                msg = dict(msg)
                msg["headers"] = list(msg.get("headers", [])) + [
                    (b"x-energy-request-id", rid.encode())]
            elif msg["type"] == "http.response.body":
                tail.extend(msg.get("body", b""))
                if len(tail) > self.TAIL_BYTES:
                    del tail[: len(tail) - self.TAIL_BYTES]
                state["inflight_max"] = max(state["inflight_max"], mon.inflight)
            await send(msg)

        mon.inflight += 1
        state["inflight_max"] = mon.inflight
        t0, _, e0 = mon.read()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            t1, _, e1 = mon.read()
            mon.inflight -= 1
            dur = t1 - t0
            gross_counter = (e1 - e0) if (e0 is not None and e1 is not None) else None
            gross_power = mon.integral(t0, t1)
            idle_w = mon.idle_w()
            gross = gross_counter if gross_counter is not None else gross_power
            rec = {
                "request_id": rid, "path": path, "status": state["status"],
                "t_start": t0, "t_end": t1, "duration_s": dur,
                "gpu_j_counter": gross_counter, "gpu_j_power": gross_power,
                "counter_vs_power": (gross_counter / gross_power
                                     if gross_counter and gross_power else None),
                "idle_w": idle_w,
                "net_j": (gross - idle_w * dur) if (gross is not None and idle_w is not None) else None,
                "inflight_max": state["inflight_max"],
                **_parse_usage(bytes(tail)),
                "gpu": mon.device.get("name"),
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
