"""
render.py - text -> 16 kHz wav stimuli for v2 speech traces.

For every (voice, target rate, item) it renders a clean utterance, and for
every noise condition a babble-noise mix of it.

    python -m audio.v2.render --items simulator/scenarios.json
    python -m audio.v2.render --items datasets/v2/d1_open/items.jsonl \
        --voices piper-en_US-lessac-medium --rates 150 --noises clean babble_5dB --shard 0/4

Layout (under datasets/v2/<name>/speech/audio/, gitignored):
    <voice_id>/<wpm>wpm/<noise>/<item_id>.wav     16 kHz mono int16
    <voice_id>/<wpm>wpm/<noise>/<item_id>.json    sidecar: words, onset, render metadata

Clean render
------------
1. Engine synthesis (tts_engines). Rate control:
   piper  -> length_scale from the per-voice calibration (voice_calibration.json);
   others -> natural pace, then time-stretch (librosa phase vocoder) to target.
2. Resample to 16 kHz; pad LEAD_MS of silence before and TAIL_MS after (enough
   for the VAD end-of-turn hang); scale so speech-active RMS = TARGET_DBFS.
3. Per-word timestamps by CTC forced alignment (align.py); optional Whisper
   cross-check (--align-check-frac).

Babble noise
------------
N_TALKERS streams, each a chain of *other* items' clean renders at the same
rate by voices drawn from the active babble pool (voices.DEFAULT_VOICES),
started at random offsets; summed and scaled so that
    10*log10(P_speech / P_babble) = SNR,
with P_speech the target's mean power between its first-word start and
last-word end and P_babble the babble's mean power over the whole file. The
babble covers the lead and tail too, as real background talk would. Talker
choice is seeded from (item_id, voice_id, wpm, noise) and listed in the sidecar.

Resumable (a sidecar marks a finished file; written last, atomically) and
parallel-safe (atomic replace; use --shard k/n to split work across processes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.v2 import voices as V  # noqa: E402
from audio.v2.align import ALIGN_MODEL, cross_check, energy_bounds, force_align  # noqa: E402
from audio.v2.items import dataset_name_for, load_items, shard, speech_root  # noqa: E402
from audio.v2.textnorm import reference_words  # noqa: E402

SR = 16000
LEAD_MS = 500
TAIL_MS = 2500
TARGET_DBFS = -26.0
N_TALKERS = 6
CALIBRATION_PATH = Path(__file__).resolve().parent / "voice_calibration.json"


# ---------------------------------------------------------------------------- audio io
def write_wav(path: Path, x: np.ndarray, sr: int = SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def read_wav16(path: Path) -> np.ndarray:
    from audio.v2.tts_engines import read_wav

    x, sr = read_wav(path)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return x


def write_json(path: Path, obj):
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def active_rms(x: np.ndarray) -> float:
    a, b = energy_bounds(x)
    seg = x[int(a * SR / 1000): int(b * SR / 1000)]
    seg = seg if len(seg) else x
    n = int(0.01 * SR)
    fr = seg[: len(seg) // n * n].reshape(-1, n)
    e = np.mean(fr ** 2, axis=1)
    keep = e > e.max() * 10 ** (-35 / 10)  # frames within 35 dB of the loudest
    return float(np.sqrt(np.mean(e[keep]))) if keep.any() else float(np.sqrt(np.mean(seg ** 2)))


def trim_silence(x: np.ndarray, pad_ms: int = 30) -> np.ndarray:
    a, b = energy_bounds(x, rel_db=-45)
    a = max(0, a - pad_ms)
    b = b + pad_ms
    return x[int(a * SR / 1000): int(b * SR / 1000)]


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def audio_dir(name: str, voice_id: str, wpm: int, noise: str) -> Path:
    return speech_root(name) / "audio" / voice_id / f"{wpm}wpm" / noise


# ---------------------------------------------------------------------------- calibration
def _load_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return {}


def _speech_wpm(x16: np.ndarray, text: str) -> tuple[float, float]:
    words = force_align(x16, text)
    dur = (words[-1]["end_ms"] - words[0]["start_ms"]) / 1000.0
    return len(reference_words(text)), dur


def calibrate_voice(voice_id: str, rates: list[int]) -> dict:
    """Per-voice rate control: natural wpm over CALIBRATION_SENTENCES, then length_scale
    per target (one refinement pass, since duration is close to linear in length_scale)."""
    from audio.v2.tts_engines import get_engine

    cal = _load_calibration()
    entry = cal.get(voice_id, {})
    if entry and all(str(r) in entry.get("rates", {}) for r in rates):
        return entry
    v = V.get_voice(voice_id)
    eng = get_engine(v)

    def measure(ls):
        nw = du = 0.0
        for s in V.CALIBRATION_SENTENCES:
            x, sr = eng.synthesize(s, v, length_scale=ls)
            a, b = _speech_wpm(resample(x, sr), s)
            nw += a
            du += b
        return 60.0 * nw / du

    natural = entry.get("natural_wpm") or measure(None)
    rates_out = dict(entry.get("rates", {}))
    for r in rates:
        if str(r) in rates_out:
            continue
        if eng.supports_length_scale:
            ls = natural / r
            got = measure(ls)
            ls = ls * got / r
            got = measure(ls)
            rates_out[str(r)] = {"length_scale": round(ls, 4), "measured_wpm": round(got, 1)}
        else:
            rates_out[str(r)] = {"time_stretch": round(natural / r, 4), "measured_wpm": None}
    entry = {"natural_wpm": round(natural, 1), "engine_version": eng.version,
             "n_sentences": len(V.CALIBRATION_SENTENCES), "rates": rates_out}
    cal = _load_calibration()  # re-read: another process may have added voices
    cal[voice_id] = entry
    tmp = CALIBRATION_PATH.with_name(CALIBRATION_PATH.name + f".{os.getpid()}.part")
    tmp.write_text(json.dumps(cal, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, CALIBRATION_PATH)
    print(f"  calibrated {voice_id}: natural {natural:.0f} wpm -> {rates_out}", flush=True)
    return entry


# ---------------------------------------------------------------------------- clean render
_WHISPER = None


def render_clean(name: str, item: dict, voice_id: str, wpm: int, align_check: bool) -> dict:
    from audio.v2.tts_engines import get_engine

    d = audio_dir(name, voice_id, wpm, "clean")
    wav_p, meta_p = d / f"{item['item_id']}.wav", d / f"{item['item_id']}.json"
    if meta_p.exists() and wav_p.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))
    v = V.get_voice(voice_id)
    eng = get_engine(v)
    rc = calibrate_voice(voice_id, [wpm])["rates"][str(wpm)]
    t0 = time.perf_counter()
    x, sr = eng.synthesize(item["text"], v, length_scale=rc.get("length_scale"),
                           seed=seed_of(item["item_id"], voice_id, wpm))
    synth_ms = (time.perf_counter() - t0) * 1000
    x = trim_silence(resample(x, sr))
    if rc.get("time_stretch"):
        import librosa

        x = librosa.effects.time_stretch(x, rate=float(rc["time_stretch"])).astype(np.float32)
    x = x * (10 ** (TARGET_DBFS / 20) / max(active_rms(x), 1e-6))
    x = np.concatenate([np.zeros(int(SR * LEAD_MS / 1000), np.float32), x,
                        np.zeros(int(SR * TAIL_MS / 1000), np.float32)])
    peak = float(np.max(np.abs(x)))
    if peak > 0.98:
        x *= 0.98 / peak
    words = force_align(x, item["text"])
    onset, offset = words[0]["start_ms"], words[-1]["end_ms"]
    meta = {
        "item_id": item["item_id"], "text": item["text"], "voice_id": voice_id,
        "accent": v["accent"], "wpm_target": wpm, "noise": "clean",
        "tts_engine": eng.version, "length_scale": rc.get("length_scale"),
        "time_stretch": rc.get("time_stretch"), "synth_ms": round(synth_ms, 1),
        "sample_rate": SR, "file_ms": int(len(x) * 1000 / SR),
        "speech_onset_ms_in_file": onset, "speech_end_ms_in_file": offset,
        "words_in_file": words, "align_model": ALIGN_MODEL,
        "wpm_actual": round(60000.0 * len(words) / max(offset - onset, 1), 1),
        "level_dbfs": TARGET_DBFS,
    }
    if align_check:
        global _WHISPER
        if _WHISPER is None:
            from audio.v2.asr import whisper_for_alignment_check

            _WHISPER = whisper_for_alignment_check()
        meta["align_check"] = cross_check(x, words, _WHISPER)
    write_wav(wav_p, x)
    write_json(meta_p, meta)
    return meta


# ---------------------------------------------------------------------------- babble
def _talker_stream(rng, pool, name, wpm, length, align_check, cache):
    out = np.zeros(length, np.float32)
    pos = int(rng.integers(-SR * 2, SR // 2))  # random phase: stream may already be mid-utterance
    used = []
    while pos < length:
        it, vid = pool[int(rng.integers(len(pool)))]
        key = (it["item_id"], vid)
        if key not in cache:
            m = render_clean(name, it, vid, wpm, align_check=False)
            y = read_wav16(audio_dir(name, vid, wpm, "clean") / f"{it['item_id']}.wav")
            a = int(m["speech_onset_ms_in_file"] * SR / 1000)
            b = int(m["speech_end_ms_in_file"] * SR / 1000)
            cache[key] = y[max(0, a - 800): b + 800]
        seg = cache[key]
        lo, hi = max(pos, 0), min(pos + len(seg), length)
        if hi > lo:
            out[lo:hi] += seg[lo - pos: hi - pos]
        used.append(list(key))
        pos += len(seg) + int(rng.integers(int(0.05 * SR), int(0.4 * SR)))
    return out, used


def render_noisy(name, item, voice_id, wpm, noise, items_all, babble_voices, cache) -> dict:
    d = audio_dir(name, voice_id, wpm, noise)
    wav_p, meta_p = d / f"{item['item_id']}.wav", d / f"{item['item_id']}.json"
    if meta_p.exists() and wav_p.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))
    if not noise.startswith("babble_") or not noise.endswith("dB"):
        raise ValueError(f"unsupported noise {noise!r} (use clean or babble_<snr>dB)")
    snr = float(noise[len("babble_"):-2])
    clean_meta = render_clean(name, item, voice_id, wpm, align_check=False)
    x = read_wav16(audio_dir(name, voice_id, wpm, "clean") / f"{item['item_id']}.wav")
    seed = seed_of(item["item_id"], voice_id, wpm, noise)
    rng = np.random.default_rng(seed)
    pool = [(it, vid) for it in items_all if it["item_id"] != item["item_id"] for vid in babble_voices]
    if not pool:
        raise ValueError("babble needs at least one other item")
    babble = np.zeros(len(x), np.float32)
    talkers = []
    for _ in range(N_TALKERS):
        s, used = _talker_stream(rng, pool, name, wpm, len(x), False, cache)
        babble += s
        talkers.append(used)
    a = int(clean_meta["speech_onset_ms_in_file"] * SR / 1000)
    b = int(clean_meta["speech_end_ms_in_file"] * SR / 1000)
    p_speech = float(np.mean(x[a:b] ** 2))
    p_noise = float(np.mean(babble ** 2)) + 1e-12
    gain = np.sqrt(p_speech / (p_noise * 10 ** (snr / 10)))
    y = x + gain * babble
    peak = float(np.max(np.abs(y)))
    scale = 0.98 / peak if peak > 0.98 else 1.0
    y = y * scale
    meta = dict(clean_meta)
    meta.pop("align_check", None)
    meta.update({
        "noise": noise,
        "noise_meta": {"type": "babble", "snr_db": snr, "n_talkers": N_TALKERS, "seed": seed,
                       "talker_voices_pool": babble_voices, "talkers": talkers,
                       "post_mix_scale": round(scale, 4), "covers": "whole file incl. lead/tail"},
        "clean_audio": str((audio_dir(name, voice_id, wpm, "clean") / f"{item['item_id']}.wav")
                           .relative_to(speech_root(name)).as_posix()),
    })
    write_wav(wav_p, y)
    write_json(meta_p, meta)
    return meta


# ---------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", required=True, help="items.jsonl (CONTRACTS §2) or v1 scenarios.json")
    ap.add_argument("--name", help="dataset name (default: inferred from --items)")
    ap.add_argument("--split", default="all", help="dev|test|all (speech prep may touch all splits)")
    ap.add_argument("--voices", nargs="+", default=V.DEFAULT_VOICES)
    ap.add_argument("--rates", nargs="+", type=int, default=V.DEFAULT_RATES)
    ap.add_argument("--noises", nargs="+", default=V.DEFAULT_NOISES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="k/n: render items k, k+n, ... (parallel processes)")
    ap.add_argument("--align-check-frac", type=float, default=0.1,
                    help="fraction of clean renders that get the Whisper timestamp cross-check")
    ap.add_argument("--calibrate-only", action="store_true")
    a = ap.parse_args(argv)

    from audio.v2.tts_engines import close_all

    name = a.name or dataset_name_for(Path(a.items))
    items_all = load_items(a.items, a.split)
    if a.limit:
        items_all = items_all[: a.limit]
    todo = shard(items_all, a.shard)
    print(f"{name}: {len(todo)}/{len(items_all)} items, voices={a.voices}, rates={a.rates}, "
          f"noises={a.noises}", flush=True)
    try:
        for vid in a.voices:
            calibrate_voice(vid, a.rates)
        if a.calibrate_only:
            return
        cache: dict = {}
        for wpm in a.rates:
            # clean first (all voices), so babble sources exist and get the alignment check
            for vid in a.voices:
                t0 = time.perf_counter()
                for it in todo:
                    chk = (seed_of("check", it["item_id"], vid, wpm) % 10000) < a.align_check_frac * 10000
                    render_clean(name, it, vid, wpm, align_check=chk)
                dt = time.perf_counter() - t0
                print(f"  {vid} {wpm}wpm clean: {len(todo)} items in {dt:.1f}s "
                      f"({dt / max(len(todo), 1):.2f} s/item)", flush=True)
            for vid in a.voices:
                for nz in a.noises:
                    if nz == "clean":
                        continue
                    t0 = time.perf_counter()
                    for it in todo:
                        render_noisy(name, it, vid, wpm, nz, items_all, V.DEFAULT_VOICES, cache)
                    dt = time.perf_counter() - t0
                    print(f"  {vid} {wpm}wpm {nz}: {len(todo)} items in {dt:.1f}s "
                          f"({dt / max(len(todo), 1):.2f} s/item)", flush=True)
    finally:
        close_all()


if __name__ == "__main__":
    main()
