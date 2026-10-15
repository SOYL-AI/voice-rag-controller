"""Tests for the research-v2 pipeline additions (no endpoint, no GPU)."""

import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class RecordingWriter:
    """Minimal LogWriter stand-in that keeps events in memory."""

    def __init__(self, conversation_id="test"):
        from eventlog.log_writer import now_ms

        self._now = now_ms
        self.conversation_id = conversation_id
        self.events = []

    def log_event(self, actor, event_type, payload, turn_id):
        self.events.append({"t_ms": self._now(), "actor": actor, "event_type": event_type,
                            "payload": payload or {}, "turn_id": turn_id})

    def first(self, et):
        return next(e for e in self.events if e["event_type"] == et)


class FakeTTS:
    sample_rate = 16000

    def __init__(self, *a, **k):
        self.synthesized = []

    def synthesize_clause_to_raw_pcm(self, clause, sample_rate=None):
        self.synthesized.append(clause)
        return b"\x00\x00" * 1600  # 100 ms

    async def speak(self, text, log_writer, turn_id, live=False, simulate_playback=True):
        log_writer.log_event("system", "tts_first_audio", {}, turn_id)


def _write_log(path, events, cid="conv_sentence_end_s0_001"):
    with open(path, "w", encoding="utf-8") as fh:
        for t, et, payload in events:
            fh.write(json.dumps({"t_ms": t, "conversation_id": cid, "turn_id": 1,
                                 "actor": "system", "event_type": et, "payload": payload}) + "\n")


# ---------------------------------------------------------------------------
# streaming LLM -> TTS
# ---------------------------------------------------------------------------
def test_clause_accumulator_matches_split_into_clauses():
    from audio.tts import ClauseAccumulator, split_into_clauses

    rng = random.Random(0)
    texts = [
        " The pool opens at 6 a.m. daily; it closes at 10 PM! Towels? Yes.\nBye",
        "Single clause without a boundary",
        "Check-in is 2:00 PM. Early check-in costs ₹1000.",
        "",
        "Trailing space.  ",
    ]
    for text in texts:
        for _ in range(20):
            cuts = sorted(rng.sample(range(len(text) + 1), min(len(text), rng.randint(0, 8))))
            chunks = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
            acc = ClauseAccumulator()
            out = [c for ch in chunks for c in acc.feed(ch)] + acc.flush()
            assert out == split_into_clauses(text)


def test_mock_client_on_delta_reassembles_text(monkeypatch):
    from llm.client import MockClient

    client = MockClient({"model": "m", "generation": {}, "logprobs": {"top_k": 1}})
    deltas = []
    gen = client.generate([{"role": "user", "content": "x"}], on_delta=deltas.append,
                          mock_answer="One two. Three four.", mock_ttft_s=0.0, mock_token_s=0.0)
    assert "".join(deltas) == gen.text == "One two. Three four."


def test_streaming_tts_starts_before_llm_final():
    from llm.client import MockClient, generate_answer
    from pipeline.streaming import generate_and_speak_streaming

    client = MockClient({"model": "m", "generation": {}, "logprobs": {"top_k": 1}})
    writer = RecordingWriter()
    tts = FakeTTS()
    answer_text = "The pool opens at six. It closes at ten tonight and towels are provided at the desk."

    def gen(on_delta):
        return generate_answer([], writer, 1, question="q", client=client, on_delta=on_delta,
                               mock_answer=answer_text, mock_ttft_s=0.05, mock_token_s=0.03)

    answer = asyncio.run(generate_and_speak_streaming(gen, tts, writer, 1, simulate_playback=False))
    assert answer == answer_text
    types = [e["event_type"] for e in writer.events]
    for et in ("llm_first_token", "llm_first_clause", "tts_clause_ready", "tts_first_audio", "llm_final"):
        assert et in types, et
    assert writer.first("llm_first_clause")["payload"]["text"] == "The pool opens at six."
    assert writer.first("tts_first_audio")["t_ms"] < writer.first("llm_final")["t_ms"]
    assert writer.first("tts_first_audio")["payload"] == {"tts_mode": "streaming"}
    assert tts.synthesized == ["The pool opens at six.", "It closes at ten tonight and towels are provided at the desk."]


def test_streaming_propagates_llm_errors():
    from pipeline.streaming import generate_and_speak_streaming

    def boom(on_delta):
        on_delta("Partial text")
        raise RuntimeError("endpoint down")

    with pytest.raises(RuntimeError, match="endpoint down"):
        asyncio.run(generate_and_speak_streaming(boom, FakeTTS(), RecordingWriter(), 1, simulate_playback=False))


def test_new_event_types_are_registered():
    from eventlog.log_writer import ALLOWED_EVENT_TYPES

    assert {"llm_first_clause", "tts_clause_ready"} <= ALLOWED_EVENT_TYPES
    schema = (ROOT / "eventlog" / "schema.md").read_text(encoding="utf-8")
    assert "llm_first_clause" in schema and "tts_clause_ready" in schema


# ---------------------------------------------------------------------------
# forecast policy
# ---------------------------------------------------------------------------
def test_forecast_jit_waits_then_dispatches():
    from controller.timing import forecast_dispatch, forecast_should_dispatch

    qr = {"min_words": 4, "require_content_word": True}
    # not ready: too short
    assert forecast_should_dispatch("what is", 5000, 300, 300, query_ready_cfg=qr) == (False, "query_not_ready")
    # ready, lots of speech left -> wait for more query
    ok, reason = forecast_should_dispatch("what time does the pool", 2000, 300, 300, query_ready_cfg=qr)
    assert not ok and reason.startswith("jit:waiting")
    # remaining speech about to drop below the latency -> dispatch now
    ok, _ = forecast_should_dispatch("what time does the pool close", 500, 300, 300, query_ready_cfg=qr)
    assert ok
    # no forecast signal -> gated reactive
    assert forecast_should_dispatch("what time does the pool", None, 300, 300, query_ready_cfg=qr)[0]

    partials = [{"t_ms": 300 * i, "text": t} for i, t in enumerate(
        ["what", "what time", "what time does", "what time does the pool", "what time does the pool close", "what time does the pool close tonight"], 1)]
    preds = [{"t_ms": p["t_ms"], "remaining_ms": 2100 - p["t_ms"]} for p in partials]
    hit = forecast_dispatch(partials, preds, expected_latency_ms=600, partial_interval_ms=300, query_ready_cfg=qr)
    # remaining < 600 + 300 first at t=1500 (remaining 600)
    assert hit["t_ms"] == 1500 and hit["text"] == "what time does the pool close"
    # earliest rule dispatches at the first ready partial
    hit_e = forecast_dispatch(partials, preds, 600, 300, forecast_cfg={"rule": "earliest"}, query_ready_cfg=qr)
    assert hit_e["t_ms"] == 1200
    # never triggers if never ready -> None (sentence_end fallback)
    assert forecast_dispatch(partials[:2], preds, 600, 300, query_ready_cfg=qr) is None


def test_forecast_is_a_valid_policy_and_proxy_untouched(monkeypatch):
    from controller.baseline import Decision
    from controller import timing

    assert {"sentence_end", "immediate", "fixed_early_proxy", "forecast"} == timing.VALID_POLICIES
    Decision(retrieve=True, timing="forecast", reason="t")
    monkeypatch.setenv("VOICE_RAG_TIMING_POLICY", "forecast")
    assert timing.resolve_policy() == "forecast"
    plan = timing.plan_simulated_speech("What is the hotel's check-in policy?")
    assert plan.dispatch_at_s("fixed_early_proxy") == max(plan.pause_at_s, plan.ready_at_s)


def test_resolve_expected_latency_prefers_signals():
    from controller.timing import resolve_expected_latency_ms

    ms, src = resolve_expected_latency_ms({"retrieval": {"latency_quantiles_ms": {"p50": 21.0}}})
    assert ms == 21.0 and src.startswith("signals")
    ms, src = resolve_expected_latency_ms({"retrieval": {"expected_latency_ms": 33.0}})
    assert ms == 33.0


# ---------------------------------------------------------------------------
# remaining-speech predictor
# ---------------------------------------------------------------------------
def _v1_traces():
    from controller.remaining_speech import load_traces

    return load_traces([str(ROOT / "simulator" / "v1_text_traces" / "traces" / "*.jsonl")])


def test_v1_text_traces_are_labelled_simulated_and_causal():
    traces = _v1_traces()
    assert len(traces) == 150
    for t in traces:
        assert t["simulated"] is True and t["asr_model"].startswith("none:")
        assert t["vad_user_end_ms"] >= t["duration_ms"]
        for p in t["partials"]:
            heard = [w for w in t["words"] if w["end_ms"] <= p["t_ms"]]
            assert len(p["text"].split()) <= len(heard)


@pytest.mark.parametrize("model", ["prior", "rate_words", "qr"])
def test_remaining_speech_fit_predict_roundtrip(tmp_path, model):
    from controller.remaining_speech import RemainingSpeechPredictor, evaluate

    traces = _v1_traces()
    train = [t for t in traces if t["item_id"] < "v1-040"]
    test = [t for t in traces if t["item_id"] >= "v1-040"]
    p = RemainingSpeechPredictor.fit(train, model=model)
    pred = p.predict("what time does the", 900)
    assert set(pred) == {"q10", "q50", "q90"}
    assert 0 <= pred["q10"] <= pred["q50"] <= pred["q90"]
    entries = p.signal_entries(test[0]["partials"])
    assert entries and {"t_ms", "remaining_ms", "remaining_q10_ms", "remaining_q90_ms"} <= set(entries[0])
    p.save(tmp_path / "m.json")
    q = RemainingSpeechPredictor.load(tmp_path / "m.json")
    assert q.predict("what time does the", 900) == pytest.approx(pred)
    res = evaluate(p, test)
    assert res["n_samples"] > 0 and res["mae_q50_ms"] >= 0 and 0 <= res["coverage_q10_q90"] <= 1


def test_remaining_speech_cli_refuses_sealed_test(tmp_path, monkeypatch):
    from controller.remaining_speech import main
    from harness.v2_split import SealedSplitError, check_split
    import harness.v2_split as vs

    monkeypatch.setattr(vs, "PREREG_FROZEN", tmp_path / "PREREG_FROZEN")
    with pytest.raises(SealedSplitError):
        check_split("test")
    check_split("dev")
    with pytest.raises(SealedSplitError):
        main(["fit", "--traces", str(ROOT / "simulator/v1_text_traces/traces/*.jsonl"),
              "--split", "test", "--out", str(tmp_path / "x.json")])


def test_item_split_rule():
    from harness.v2_split import filter_split, item_split

    ids = [f"d2_support-{i:06d}" for i in range(2000)]
    dev = sum(item_split(i) == "dev" for i in ids)
    assert 0.25 < dev / len(ids) < 0.35
    assert item_split("abc") == item_split("abc")
    items = [{"id": "x", "split": "test"}, {"id": "y", "split": "dev"}]
    assert [i["id"] for i in filter_split(items, "dev")] == ["y"]


# ---------------------------------------------------------------------------
# latency breakdown
# ---------------------------------------------------------------------------
def test_latency_breakdown_stages_sum_to_ttfa(tmp_path):
    from harness.latency_breakdown import STAGES, parse_log, summarize

    log = tmp_path / "conv_fixed_early_proxy_s0_001.jsonl"
    _write_log(log, [
        (1000, "controller_action", {"action": "retrieve"}),
        (1000, "retrieval_launched", {"query": "q"}),
        (1700, "retrieval_result", {"passages": ["p"]}),
        (1500, "user_end", {}),
        (1500, "asr_final", {"text": "q"}),
        (2750, "llm_first_token", {"token": "x"}),
        (4000, "llm_final", {"text": "a", "ttft_ms": 1000, "n_tokens": 40}),
        (4020, "tts_first_audio", {}),
    ])
    r = parse_log(log)
    assert r["policy"] == "fixed_early_proxy" and r["ttfa_ms"] == 2520
    assert sum(r["stages_ms"].values()) == r["ttfa_ms"]
    assert r["stages_ms"]["retrieval_exposed"] == 200 and r["retrieval_hidden_ms"] == 500
    assert r["stages_ms"]["pre_llm_overhead"] == 50 and r["stages_ms"]["llm_prefill"] == 1000
    s = summarize([r])
    assert abs(sum(s["fixed_early_proxy"]["stages_ms"][k]["share_of_mean_ttfa"] for k in STAGES) - 1) < 1e-9


def test_latency_breakdown_on_committed_e2_logs():
    from harness.latency_breakdown import load_runs

    runs = load_runs([str(ROOT / "logs" / "conv_sentence_end_s0_00*.jsonl")])
    assert runs
    for r in runs:
        assert abs(sum(r["stages_ms"].values()) - r["ttfa_ms"]) < 1e-6


# ---------------------------------------------------------------------------
# latency model
# ---------------------------------------------------------------------------
def test_first_clause_token_index():
    from harness.latency_model_v2 import first_clause_token_index

    assert first_clause_token_index(["▁The", "▁pool", ".", "▁It", "▁closes", "."]) == 3
    assert first_clause_token_index(["▁Yes", "▁indeed", "."]) is None
    assert first_clause_token_index(["▁Hi", "!", "<0x0A>", "Bye"]) == 2


def _small_model():
    from harness.latency_model_v2 import LatencyModel, load_records

    recs = load_records([str(ROOT / "logs" / "conv_*_s0_00*.jsonl")], "e2")
    # synthesize a skip pool from the same runs (no retrieval) for the test
    skip = []
    for r in recs[:10]:
        s = dict(r, path="skip", retrieval_ms=None, skip_overhead_ms=5.0, n_tokens=max(5, r["n_tokens"] // 2))
        skip.append(s)
    return LatencyModel.fit(recs + skip)


def test_latency_model_sample_ttfa_and_modes():
    m = _small_model()
    rng = np.random.default_rng(0)
    d = m.sample_ttfa("retrieve", -500.0, 0.0, "pipeline_logged", 600, rng)
    for k in ("ttfa_ms", "eot_detect_ms", "retrieval_ms", "retrieval_exposed_ms", "ttft_ms",
              "gen_to_tts_input_ms", "tts_first_audio_ms"):
        assert k in d
    assert d["ttfa_ms"] > 0
    # identical random streams: streaming never slower than batch
    for seed in range(30):
        b = m.sample_ttfa("retrieve", None, 0.0, "pipeline_logged", 600, np.random.default_rng(seed), tts_mode="batch")
        s = m.sample_ttfa("retrieve", None, 0.0, "pipeline_logged", 600, np.random.default_rng(seed), tts_mode="streaming")
        assert s["ttfa_ms"] <= b["ttfa_ms"] + 1e-9
    # earlier dispatch never exposes more retrieval
    for seed in range(30):
        late = m.sample_ttfa("retrieve", 0.0, 0.0, "pipeline_logged", 600, np.random.default_rng(seed))
        early = m.sample_ttfa("retrieve", -5000.0, 0.0, "pipeline_logged", 600, np.random.default_rng(seed))
        assert early["retrieval_exposed_ms"] == 0.0
        assert early["ttfa_ms"] <= late["ttfa_ms"] + 1e-9
    with pytest.raises(KeyError):
        m.sample_ttfa("retrieve", None, 0.0, "nope", 600, rng)


def test_latency_model_signals_and_roundtrip(tmp_path):
    from harness.latency_model_v2 import LatencyModel

    m = _small_model()
    sig = m.signals_retrieval("pipeline_logged", "batch")
    assert sig["latency_profile"] == "pipeline_logged"
    assert sig["expected_latency_ms"] == sig["latency_quantiles_ms"]["p50"]
    # _objective_v2.nonhideable_ms() does float(...) on this key
    assert isinstance(sig["nonhideable_increment_ms"], float)
    assert set(sig["nonhideable_increment_quantiles_ms"]) >= {"p10", "p50", "p90"}
    assert sig["nonhideable_increment_tts_mode"] == "batch"
    m.save(tmp_path / "lm.json")
    m2 = LatencyModel.load(tmp_path / "lm.json")
    a = m.sample_ttfa("skip", None, 0.0, "pipeline_logged", 100, np.random.default_rng(1))
    b = m2.sample_ttfa("skip", None, 0.0, "pipeline_logged", 100, np.random.default_rng(1))
    assert a == b


def test_latency_model_controller_interface():
    """The interface simulator/e3v2_decide.py and _objective_v2.py rely on."""
    import random as _random

    from harness.latency_model_v2 import DEFAULT_FIT_PATH, LatencyModel

    m = _small_model()
    cfg = {"timing": {"eot_detection_ms": 300}}
    m.cfg = cfg
    a = m.sample_ttfa("retrieve", 500.0, 1000.0, "pipeline_logged", 700, _random.Random(7))
    b = m.sample_ttfa("retrieve", 500.0, 1000.0, "pipeline_logged", 700, _random.Random(7))
    assert a == b and a["eot_ms"] == 300.0 and a["stub"] is False
    for k in ("ttfa_ms", "exposed_retrieval_ms", "retrieval_ms", "nonhideable_ms", "skip_ttfa_ms"):
        assert k in a
    # common random numbers: the skip draw with the same seed is the matched counterfactual
    s = m.sample_ttfa("skip", None, 1000.0, "pipeline_logged", None, _random.Random(7))
    assert s["ttfa_ms"] == pytest.approx(a["skip_ttfa_ms"])
    assert s["nonhideable_ms"] == 0.0 and s["exposed_retrieval_ms"] == 0.0
    # decomposition: ttfa = skip + exposed + pi (+ pre-LLM overhead difference, which is part of pi)
    assert a["ttfa_ms"] == pytest.approx(a["skip_ttfa_ms"] + a["exposed_retrieval_ms"] + a["nonhideable_ms"])
    if DEFAULT_FIT_PATH.exists():
        from controller.baselines._objective_v2 import load_e3v2_config
        from simulator.e3v2_decide import load_latency_model

        e3cfg = load_e3v2_config()
        real = load_latency_model(e3cfg)  # what the controllers' simulator does
        assert isinstance(real, LatencyModel)  # the real module, not the stub
        for profile in e3cfg["latency_profiles"]:
            r = real.sample_ttfa("retrieve", 800.0, 1000.0, profile, 0, _random.Random(3))
            s = real.sample_ttfa("skip", None, 1000.0, profile, 0, _random.Random(3))
            assert r["stub"] is False and r["ttfa_ms"] > 0 and r["nonhideable_ms"] is not None
            assert s["ttfa_ms"] == pytest.approx(r["skip_ttfa_ms"])
        sig = real.signals_retrieval("faiss", "streaming")
        from controller.baselines._objective_v2 import nonhideable_ms

        assert nonhideable_ms(sig, e3cfg) == pytest.approx(sig["nonhideable_increment_ms"])


def test_paired_increments():
    from harness.latency_model_v2 import paired_increments

    base = dict(scenario="001", seed="11", policy="sentence_end", tts_mode="batch",
                retrieval_exposed_ms=0.0, dispatch_wait_ms=0.0)
    recs = [dict(base, path="retrieve", ttfa_ms=3000.0, retrieval_exposed_ms=500.0),
            dict(base, path="skip", ttfa_ms=1500.0)]
    out = paired_increments(recs)["sentence_end/batch"]
    assert out["total_mean_ms"] == 1500.0 and out["nonhideable_mean_ms"] == 1000.0


# ---------------------------------------------------------------------------
# measurement harness safety
# ---------------------------------------------------------------------------
def test_measure_env_loader_and_budget(tmp_path, monkeypatch, capsys):
    from harness.measure_llm_tts import Budget, load_env_file

    env = tmp_path / ".env"
    env.write_text("SECRET_TEST_KEY=supersecretvalue\n# comment\nOTHER='x'\n", encoding="utf-8")
    keys = load_env_file(str(env))
    assert keys == ["SECRET_TEST_KEY", "OTHER"]
    assert "supersecretvalue" not in capsys.readouterr().out
    monkeypatch.delenv("SECRET_TEST_KEY", raising=False)
    b = Budget(2)
    b.take(); b.take()
    with pytest.raises(RuntimeError):
        b.take()


# ---------------------------------------------------------------------------
# spine: trace-driven live-partial path (mock LLM, fake TTS, fake retriever)
# ---------------------------------------------------------------------------
class FakeRetriever:
    profile = "fake"

    def __init__(self):
        self.queries = []
        self.last_timing = {"total_ms": 1.0}

    def search(self, query, k=3):
        self.queries.append(query)
        return [{"doc_id": f"d{i}", "filename": f"d{i}.txt", "text": f"passage {i}", "score": 1.0 - i, "rank": i}
                for i in range(k)]


def _trace():
    words = "what time does the pool close tonight".split()
    wt = [{"w": w, "start_ms": 100 * i, "end_ms": 100 * i + 80} for i, w in enumerate(words)]
    partials = [{"t_ms": t, "text": " ".join(words[: max(1, t // 100)])} for t in range(100, 800, 100)]
    return {"item_id": "x", "voice_id": "v", "words": wt, "partials": partials,
            "asr_final": {"t_ms": 900, "text": "What time does the pool close tonight?"},
            "vad_user_end_ms": 900, "duration_ms": 680, "partial_interval_ms": 100}


@pytest.mark.parametrize("tts_mode", ["batch", "streaming"])
def test_spine_trace_path_retrieves_on_live_partial(tmp_path, monkeypatch, tts_mode):
    monkeypatch.setenv("VOICE_RAG_LLM_BACKEND", "mock")
    monkeypatch.setenv("VOICE_RAG_TIMING_POLICY", "forecast")
    import pipeline.spine as spine

    monkeypatch.setattr(spine, "StreamingTTS", FakeTTS)
    retr = FakeRetriever()
    # predictions: 600 ms left at t=100 ... -> jit fires once remaining < 50 + 100
    signals = {"timing": {"predicted_remaining_speech_ms": [{"t_ms": t, "remaining_ms": 700 - t} for t in range(100, 800, 100)]},
               "retrieval": {"expected_latency_ms": 50}}
    path = asyncio.run(spine.run_conversation(
        "trace_test", question="What time does the pool close tonight?", force_retrieve=True,
        speech_trace=_trace(), query_source="live_partial", live_rewrite=False, retriever=retr,
        tts_mode=tts_mode, simulate_playback=False, signals=signals, log_dir=str(tmp_path)))
    ev = [json.loads(l) for l in open(path, encoding="utf-8")]
    launched = next(e for e in ev if e["event_type"] == "retrieval_launched")["payload"]
    assert launched["query_source"] == "live_partial"
    assert launched["timing_policy"] == "forecast"
    assert launched["retrieval_profile"] == "fake"
    assert launched["query"] == "what time does the pool close"  # the partial at t=600, not the final text
    assert retr.queries == [launched["query"]]
    assert launched["predicted_remaining_ms"] == 100.0
    ue = next(e for e in ev if e["event_type"] == "user_end")
    assert ue["payload"]["speech_end_offset_ms"] == -220.0
    assert any(e["event_type"] == "tts_first_audio" for e in ev)
    if tts_mode == "streaming":
        assert any(e["event_type"] == "llm_first_clause" for e in ev)
