"""Model-free tests for the v2 speech tooling (audio/v2)."""

import json

import numpy as np

from audio.v2 import voices as V
from audio.v2.align import refine_with_energy
from audio.v2.textnorm import align_chars, edit_ops, int_to_words, wer
from audio.v2.trace import append_line, done_ids, finalize
from audio.v2.trace_report import condition_metrics


def test_condition_ids_roundtrip():
    c = V.condition_id("piper-en_US-lessac-medium", 150, "babble_5dB")
    assert c == "piper-en_US-lessac-medium__150wpm__babble_5dB"
    assert V.parse_condition(c) == ("piper-en_US-lessac-medium", 150, "babble_5dB")
    assert V.parse_condition(V.condition_id("human-s03", "natural", "clean"))[1] == "natural"


def test_default_voices_cover_accents():
    acc = {V.VOICES[v]["accent"] for v in V.DEFAULT_VOICES}
    assert {"en-US", "en-GB", "en-IN"} <= acc and len(V.DEFAULT_VOICES) >= 4


def test_numbers_and_align_chars():
    assert int_to_words(64) == "sixty four"
    assert int_to_words(1947) == "nineteen forty seven"
    assert align_chars("18") == "eighteen"
    assert align_chars("check-out?") == "check out"


def test_edit_ops_and_wer():
    assert edit_ops(["a", "b", "c"], ["a", "x", "c", "d"]) == (1, 0, 1)
    assert wer("Is 17 a prime number?", "is seventeen a prime number") == 0.0
    assert abs(wer("tell me about spa", "tell me about spaw") - 0.25) < 1e-9


def test_refine_extends_peaky_word_end():
    sr = 16000
    x = np.zeros(sr, np.float32)
    x[1600:8000] = 0.3 * np.sin(np.arange(6400) / 5.0)       # 100-500 ms voiced
    words = [{"w": "hi", "start_ms": 150, "end_ms": 400}]
    out = refine_with_energy(x, words)
    assert out[0]["start_ms"] == 100 and out[0]["end_ms"] == 500


def _rec(item, partials, final, end):
    words = [{"w": "tell", "start_ms": 0, "end_ms": 200}, {"w": "me", "start_ms": 300, "end_ms": 400},
             {"w": "about", "start_ms": 500, "end_ms": 800}, {"w": "spa", "start_ms": 900, "end_ms": 1300}]
    return {"item_id": item, "words": words, "wpm_actual": 150.0, "partials": partials,
            "asr_final": {"t_ms": end, "available_ms": end + 300, "text": final},
            "vad_user_end_ms": end, "asr_settings": {"silence_hang_ms": 800}}


def test_condition_metrics():
    p = [{"t_ms": 400, "available_ms": 600, "compute_ms": 200, "text": "Tell me."},
         {"t_ms": 800, "available_ms": 950, "compute_ms": 150, "text": "Tell me a bit"},
         {"t_ms": 1200, "available_ms": 1400, "compute_ms": 200, "text": "Tell me about spa"}]
    m = condition_metrics([_rec("a", p, "Tell me about spa.", 2150)])
    assert m["final_wer"] == 0.0
    assert m["eot_median_ms"] == 850 and m["eot_premature"] == 0.0
    assert m["rev_strict"] == 0.5          # "a bit" -> "about spa" rewrites the earlier tail
    assert m["rev_lenient"] == 0.5         # ...and more than the last word changed
    assert abs(m["partial_wer"] - 2 / 9) < 1e-9  # 0 + 2 + 0 errors over prefixes of 2, 3, 4 words


def test_trace_file_resume_and_finalize(tmp_path):
    p = tmp_path / "c.jsonl"
    append_line(p, {"item_id": "b", "x": 1})
    append_line(p, {"item_id": "a", "x": 2})
    append_line(p, {"item_id": "b", "x": 3})
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"item_id": "torn"')
    assert done_ids(p) == {"a", "b"}
    finalize(p)
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert [r["item_id"] for r in rows] == ["a", "b"] and rows[1]["x"] == 1
