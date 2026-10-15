"""Tests for energy/: integration math, the paired protocol, the serving
middleware, and the per-turn energy model API. No GPU or NVML needed."""

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest

from energy import model as em
from energy.protocol import mean_ci, paired_net
from energy.sampler import (achieved_rate_hz, counter_delta, integrate_trapezoid,
                            interp_at, net_energy)
from energy.serving.gpu_energy import EnergyMiddleware, GpuEnergyMonitor, _parse_usage, _trapz


# ---------------------------------------------------------------- integration
def test_constant_power_integrates_exactly():
    ts = np.linspace(0, 10, 501)
    ps = np.full_like(ts, 20.0)
    assert integrate_trapezoid(ts, ps, 2.0, 7.0) == pytest.approx(100.0)


def test_linear_ramp_is_exact_including_fractional_edges():
    ts = np.arange(0, 10.01, 0.02)          # 50 Hz
    ps = 3.0 * ts                            # P = 3t
    t0, t1 = 1.013, 8.377                    # not on sample points
    exact = 1.5 * (t1 ** 2 - t0 ** 2)
    assert integrate_trapezoid(ts, ps, t0, t1) == pytest.approx(exact, rel=1e-12)


def test_window_edges_are_not_double_counted():
    ts = np.arange(0, 10.01, 0.1)
    ps = np.sin(ts) + 2
    whole = integrate_trapezoid(ts, ps, 0.37, 9.11)
    parts = integrate_trapezoid(ts, ps, 0.37, 4.444) + integrate_trapezoid(ts, ps, 4.444, 9.11)
    assert whole == pytest.approx(parts, rel=1e-12)


def test_square_pulse_energy_converges_with_rate():
    # a 1 s, 30 W pulse on a 5 W floor; integral over [0, 3] = 3*5 + 25
    for hz in (20, 50, 200):
        ts = np.arange(0, 3.0001, 1 / hz)
        ps = np.where((ts >= 1) & (ts < 2), 30.0, 5.0)
        assert integrate_trapezoid(ts, ps, 0, 3) == pytest.approx(40.0, abs=25.0 / hz + 1e-9)


def test_zero_and_degenerate_windows():
    ts, ps = np.array([0.0, 1.0]), np.array([5.0, 5.0])
    assert integrate_trapezoid(ts, ps, 0.5, 0.5) == 0.0
    assert integrate_trapezoid(np.array([0.0]), np.array([7.0]), 0, 2) == 14.0
    with pytest.raises(ValueError):
        integrate_trapezoid(ts, ps, 1.0, 0.5)


def test_extrapolation_holds_edge_value():
    ts, ps = np.array([1.0, 2.0]), np.array([4.0, 4.0])
    assert integrate_trapezoid(ts, ps, 0.0, 3.0) == pytest.approx(12.0)


def test_counter_delta_interpolates():
    ts = np.array([0.0, 1.0, 2.0])
    cum = np.array([100.0, 110.0, 130.0])
    assert counter_delta(ts, cum, 0.5, 1.5) == pytest.approx(120.0 - 105.0)
    assert interp_at(ts, cum, 1.5) == 120.0


def test_net_energy_and_rate():
    assert net_energy(50.0, 4.0, 10.0) == 10.0
    assert achieved_rate_hz(np.arange(0, 1.0001, 0.02)) == pytest.approx(50.0)


def test_rapl_unit_conversion():
    from energy.winpdh import PWH_TO_J
    # 1 Wh = 1e12 pWh = 3600 J
    assert 1e12 * PWH_TO_J == pytest.approx(3600.0)


# ---------------------------------------------------------------- protocol
def test_paired_net_uses_flanking_mean():
    # active 10 s at 15 W; idle 4 W before, 6 W after -> net = 150 - 50 = 100, k=4
    assert paired_net(150.0, 10.0, 4.0, 6.0, 4) == pytest.approx(25.0)
    assert paired_net(150.0, 10.0, 5.0, None, 1) == pytest.approx(100.0)


def test_mean_ci_matches_t_interval():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = mean_ci(xs)
    assert r["mean"] == 3.0
    assert r["half"] == pytest.approx(2.776445 * math.sqrt(2.5) / math.sqrt(5), rel=1e-5)
    assert mean_ci([])["n"] == 0 and mean_ci([2.0])["half"] is None


class _FakeMeter:
    """Deterministic meter: GPU power 3 W idle, +10 W while 'working'."""

    has_gpu = True

    def __init__(self):
        self.t = 0.0
        self.busy = False

    def snapshot(self):
        ts = np.array([self.t - 1.0, self.t])
        ps = np.array([3.0, 3.0])
        return ts, ps, ps * 0, ps * 0

    def idle_window(self, seconds):
        from energy.sampler import StageEnergy
        self.t += seconds
        return StageEnergy("idle", 0, seconds, seconds, gpu_j=3.0 * seconds, cpu_pkg_j=30.0 * seconds)

    def measure(self, name, tail_s=0.0, **kw):
        import contextlib
        from energy.sampler import StageEnergy

        meter = self

        @contextlib.contextmanager
        def cm():
            class H: ...
            h = H()
            start = meter.t
            meter.work = 0.0
            yield h
            dur = meter.work + tail_s
            meter.t = start + dur
            h.result = StageEnergy(name, start, meter.t, dur, gpu_j=3.0 * dur + 10.0 * meter.work,
                                   cpu_pkg_j=30.0 * dur, gpu_mean_w=None)
            h.result.meta.update({"work_s": meter.work, "tail_s": tail_s})
        return cm()


def test_run_paired_recovers_per_op_energy(monkeypatch):
    import energy.protocol as pr

    meter = _FakeMeter()
    monkeypatch.setattr(pr, "gpu_rest_w", lambda m, margin_w=1.5: 4.5)
    monkeypatch.setattr(pr, "settle_gpu", lambda *a, **k: 0.0)

    def op():
        meter.work += 0.25   # each op = 0.25 s of +10 W -> 2.5 J net

    ts = pr.run_paired(meter, "fake", op, n_trials=4, k=8, warmup=0, idle_s=1.0,
                       tail_s=0.5, log=lambda *a, **k: None)
    s = ts.summary()
    assert s["gpu_net_j"]["mean"] == pytest.approx(2.5)
    assert s["cpu_net_j"]["mean"] == pytest.approx(0.0)
    assert s["latency_s"]["mean"] == pytest.approx(0.25)
    assert s["n_flagged_gpu_contended"] == 0


# ---------------------------------------------------------------- serving patch
class _FakeNvml:
    NVML_CLOCK_SM = NVML_CLOCK_MEM = NVML_TEMPERATURE_GPU = 0

    def __init__(self):
        self.e_mj = 0

    def nvmlInit(self): ...
    def nvmlDeviceGetHandleByIndex(self, i): return "h"
    def nvmlDeviceGetName(self, h): return "FakeA10G"
    def nvmlSystemGetDriverVersion(self): return "0"
    def nvmlDeviceGetEnforcedPowerLimit(self, h): return 300000
    def nvmlDeviceGetMaxClockInfo(self, h, c): return 1
    def nvmlDeviceGetClockInfo(self, h, c): return 1
    def nvmlDeviceGetTemperature(self, h, c): return 40
    def nvmlDeviceGetPowerUsage(self, h): return 150000

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self.e_mj += 1500           # +1.5 J per read
        return self.e_mj


def test_parse_usage_streaming_and_plain():
    sse = (b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":534,"completion_tokens":56}}\n\n'
           b'data: [DONE]\n\n')
    assert _parse_usage(sse) == {"prompt_tokens": 534, "completion_tokens": 56}
    plain = json.dumps({"usage": {"prompt_tokens": 79, "completion_tokens": 28}}).encode()
    assert _parse_usage(plain) == {"prompt_tokens": 79, "completion_tokens": 28}
    assert _parse_usage(b"no usage here")["prompt_tokens"] is None


def test_trapz_helper_matches_numpy_version():
    ts = list(np.arange(0, 5.01, 0.05))
    ps = [2.0 + t for t in ts]
    assert _trapz(ts, ps, 0.33, 4.21) == pytest.approx(
        integrate_trapezoid(ts, ps, 0.33, 4.21), rel=1e-12)


def test_middleware_logs_one_record_per_completion(tmp_path):
    mon = GpuEnergyMonitor(nvml=_FakeNvml(), log_dir=tmp_path)   # thread not started
    EnergyMiddleware.monitor = mon

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body",
                    "body": b'{"usage":{"prompt_tokens":10,"completion_tokens":3}}'})

    mw = EnergyMiddleware(app)
    sent = []

    async def send(m):
        sent.append(m)

    async def receive():
        return {"type": "http.request"}

    asyncio.run(mw({"type": "http", "path": "/v1/chat/completions", "method": "POST"}, receive, send))
    asyncio.run(mw({"type": "http", "path": "/health", "method": "GET"}, receive, send))
    rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["gpu_j_counter"] == pytest.approx(1.5)
    assert r["prompt_tokens"] == 10 and r["completion_tokens"] == 3
    assert r["inflight_max"] == 1 and mon.inflight == 0
    assert any(h[0] == b"x-energy-request-id" for h in sent[0]["headers"])
    EnergyMiddleware.monitor = None


# ---------------------------------------------------------------- model API
P = {
    "version": "test",
    "local": {
        "vad_j_per_audio_s": 0.05, "asr_j_per_audio_s": {"central": 10.0, "low": 5.0, "high": 20.0},
        "tts_j_fixed": 1.0, "tts_j_per_char": 0.1,
        "embed_query_j": 0.2, "faiss_search_j": {"60": 0.001, "100k": 0.05},
        "rerank50_j": 3.0, "remote_api_network_j": {"central": 0.0, "low": 0.0, "high": 1.0},
        "gpu_wake_tail_j": 2.0, "gpu_wake_per_turn": False, "idle_w": 40.0,
        "default_index_size": "60",
    },
    "remote_llm": {
        "prefill_j_fixed": 0.0, "prefill_j_per_token": {"central": 0.1, "low": 0.05, "high": 0.2},
        "decode_j_per_token": {"central": 8.0, "low": 4.0, "high": 10.0},
        "idle_w": 60.0, "retrieval_server_scale": 1.0,
    },
}


def test_turn_energy_components_add_up():
    e = em.turn_energy_j("retrieve", "faiss", 500, 50, 4.0, 200, params=P)
    c = e["components"]
    assert c["llm_prefill"]["central"] == pytest.approx(50.0)
    assert c["llm_decode"]["central"] == pytest.approx(400.0)
    assert c["asr"]["central"] == pytest.approx(40.0)
    assert c["tts"]["central"] == pytest.approx(21.0)
    total = sum(v["central"] for v in c.values())
    assert e["total_j"]["central"] == pytest.approx(total)
    assert e["remote_j"]["central"] == pytest.approx(450.0)
    assert e["local_j"]["central"] == pytest.approx(total - 450.0)
    assert e["total_j"]["low"] <= e["total_j"]["central"] <= e["total_j"]["high"]


def test_skip_has_no_retrieval_components_and_profile_matters():
    s = em.turn_energy_j("skip", None, 80, 28, 4.0, 120, params=P)
    assert not any(k.startswith("retrieval") for k in s["components"])
    f = em.turn_energy_j("retrieve", "faiss", 500, 50, 4.0, 200, params=P)
    r = em.turn_energy_j("retrieve", "faiss_rerank", 500, 50, 4.0, 200, params=P)
    assert r["total_j"]["central"] - f["total_j"]["central"] == pytest.approx(3.0)
    ra = em.turn_energy_j("retrieve", "remote_api", 500, 50, 4.0, 200, params=P, pue=1.5)
    assert ra["components"]["retrieval_embed"]["central"] == pytest.approx(0.3)
    assert ra["remote_j"]["central"] == pytest.approx(1.5 * (50 + 400) + 0.3 + 0.0015)


def test_marginal_retrieval_is_difference():
    m = em.marginal_retrieval("faiss", n_prompt_skip=80, n_prompt_retrieve=535, n_gen_skip=28,
                              n_gen_retrieve=56, asr_seconds=4.0, tts_chars_skip=120,
                              tts_chars_retrieve=230, params=P)
    assert m["components"]["llm_prefill"]["central"] == pytest.approx(0.1 * 455)
    assert m["components"]["llm_decode"]["central"] == pytest.approx(8.0 * 28)
    assert m["components"]["asr"]["central"] == pytest.approx(0.0)
    assert m["total_j"]["central"] == pytest.approx(45.5 + 224 + 11 + 0.201)


def test_co2_conversion_and_override():
    e = em.turn_energy_j("skip", None, 0, 0, 0.0, 0, params=P, include_idle=True, turn_wall_s=3.6)
    # idle: 40 W local + 60 W remote for 3.6 s = 360 J = 1e-4 kWh, plus tts fixed 1 J
    j = e["total_j"]["central"]
    assert j == pytest.approx(361.0)
    g = em.grid_factor()["g_co2_per_kwh"]
    assert e["g_co2e"]["central"] == pytest.approx(j / 3.6e6 * g)
    e2 = em.turn_energy_j("skip", None, 0, 0, 0.0, 0, params=P, include_idle=True,
                          turn_wall_s=3.6, remote_grid_g_per_kwh=0.0)
    assert e2["g_co2e"]["central"] == pytest.approx((1.0 + 144.0) / 3.6e6 * g)


def test_input_validation():
    with pytest.raises(ValueError):
        em.turn_energy_j("maybe", None, 1, 1, 1, 1, params=P)
    with pytest.raises(ValueError):
        em.turn_energy_j("retrieve", "bm25", 1, 1, 1, 1, params=P)
    with pytest.raises(ValueError):
        em.turn_energy_j("skip", None, -1, 1, 1, 1, params=P)
    with pytest.raises(ValueError):
        em.turn_energy_j("skip", None, 1, 1, 1, 1, params=P, include_idle=True)


def test_grid_factors_file_is_cited():
    g = json.loads((Path(em.__file__).parent / "grid_factors.json").read_text())
    for name, f in g["factors"].items():
        assert f["g_co2_per_kwh"] > 0 and f["url"].startswith("https://") and f["source"]
    assert g["default"] in g["factors"]


def test_implied_price():
    assert em.implied_quality_per_joule(0.05, 250.0) == pytest.approx(2e-4)
    with pytest.raises(ValueError):
        em.implied_quality_per_joule(0.05, 0.0)


def test_shipped_params_load_if_present():
    if not em.PARAMS_PATH.exists():
        pytest.skip("params.json not generated yet")
    e = em.turn_energy_j("retrieve", "faiss", 534, 56, 3.9, 230)
    assert e["total_j"]["central"] > 0
