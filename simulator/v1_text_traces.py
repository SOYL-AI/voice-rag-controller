"""
v1_text_traces.py — SIMULATED speech traces for the 50 v1 scenarios, in the
CONTRACTS §4 trace shape, for developing the v2 timing components
(remaining-speech predictor, forecast policy, partial-query risk) before
real ASR traces exist.

These are NOT ASR output. No audio is rendered and no recogniser runs:
word timings come from a speaking-rate model and every "partial" is a
prefix of the reference text. That violates §4's rule for real traces
("never write the reference transcript into a partial"), which is why they
live here (simulator/v1_text_traces/) and not under datasets/v2/, and why
every record carries "simulated": true and asr_model
"none:text-timing-simulation". Use them for development and unit tests
only; anything reported in the paper must be refit on real traces.

Timing model (per voice):
    - nominal rate wpm (130 / 150 / 175 by default);
    - per-word duration proportional to 0.45 + 0.55 * len(word)/mean_len,
      with +-15 % seeded jitter, rescaled so the utterance runs at the
      nominal rate, plus a 40 ms inter-word gap;
    - a partial every partial_interval_ms containing the words whose end
      is at least asr_lag_ms in the past (streaming ASR lag), lower-cased
      and with punctuation stripped (real streaming partials carry
      unreliable punctuation, so the predictor must not learn from it);
    - vad_user_end_ms = speech end + silence_hang_ms (audio.stt default);
    - asr_final at vad_user_end_ms with the reference text.

    python -m simulator.v1_text_traces --out simulator/v1_text_traces
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_VOICES = {
    "text-timing-wpm130": 130,
    "text-timing-wpm150": 150,
    "text-timing-wpm175": 175,
}

_PUNCT = re.compile(r"[^\w\s'-]")


def normalize_partial(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def word_timings(words: List[str], wpm: float, rng: random.Random, gap_ms: float = 40.0) -> List[dict]:
    if not words:
        return []
    lens = [max(1, len(w.strip(".,!?;:\"'"))) for w in words]
    mean_len = sum(lens) / len(lens)
    raw = [(0.45 + 0.55 * l / mean_len) * rng.uniform(0.85, 1.15) for l in lens]
    total_ms = len(words) * 60000.0 / wpm
    speech_ms = max(total_ms - gap_ms * (len(words) - 1), 100.0 * len(words))
    scale = speech_ms / sum(raw)
    out, t = [], 0.0
    for w, r in zip(words, raw):
        d = r * scale
        out.append({"w": w, "start_ms": int(round(t)), "end_ms": int(round(t + d))})
        t += d + gap_ms
    return out


def make_trace(
    item_id: str,
    text: str,
    voice_id: str,
    wpm: float,
    seed: int = 0,
    partial_interval_ms: int = 300,
    asr_lag_ms: int = 250,
    silence_hang_ms: int = 800,
) -> dict:
    rng = random.Random(f"{item_id}|{voice_id}|{seed}")
    words = text.split()
    wt = word_timings(words, wpm, rng)
    duration = wt[-1]["end_ms"] if wt else 0
    vad_end = duration + silence_hang_ms
    partials = []
    t = partial_interval_ms
    last_text = None
    while t < vad_end:
        n = sum(1 for w in wt if w["end_ms"] <= t - asr_lag_ms)
        if n > 0:
            ptxt = normalize_partial(" ".join(words[:n]))
            partials.append({"t_ms": t, "text": ptxt, "stable": ptxt == last_text})
            last_text = ptxt
        t += partial_interval_ms
    return {
        "item_id": item_id,
        "voice_id": voice_id,
        "accent": None,
        "wpm_target": wpm,
        "noise": "clean",
        "tts_engine": "none:text-timing-simulation",
        "audio_path": None,
        "duration_ms": duration,
        "words": wt,
        "partials": partials,
        "asr_final": {"t_ms": vad_end, "text": text},
        "vad_user_end_ms": vad_end,
        "asr_model": "none:text-timing-simulation",
        "partial_interval_ms": partial_interval_ms,
        "simulated": True,
    }


def v1_items(scenarios_path: Optional[Path] = None) -> List[dict]:
    path = scenarios_path or (_REPO_ROOT / "simulator" / "scenarios.json")
    scenarios = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        {
            "id": f"v1-{s['id']:03d}",
            "question": s["question"],
            "gold_doc_ids": [s["document"]],
            "retrieval_needed_authored": bool(s.get("retrieval_needed")),
            "dataset": "v1_hotel",
        }
        for s in scenarios
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Simulated v1 speech traces (development only)")
    ap.add_argument("--out", default="simulator/v1_text_traces")
    ap.add_argument("--partial-interval-ms", type=int, default=300)
    ap.add_argument("--asr-lag-ms", type=int, default=250)
    ap.add_argument("--silence-hang-ms", type=int, default=800)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    items = v1_items()
    with open(out / "items.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    for voice_id, wpm in DEFAULT_VOICES.items():
        with open(out / "traces" / f"{voice_id}.jsonl", "w", encoding="utf-8") as fh:
            for it in items:
                tr = make_trace(
                    it["id"], it["question"], voice_id, wpm,
                    partial_interval_ms=args.partial_interval_ms,
                    asr_lag_ms=args.asr_lag_ms,
                    silence_hang_ms=args.silence_hang_ms,
                )
                fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items x {len(DEFAULT_VOICES)} voices to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
