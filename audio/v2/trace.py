"""
trace.py - replay rendered speech through the project's streaming STT on a
simulated clock and write speech traces (CONTRACTS.md §4).

    python -m audio.v2.trace --items simulator/scenarios.json
    python -m audio.v2.trace --items datasets/v2/d2_support/items.jsonl \
        --voices piper-en_US-lessac-medium --rates 150 --noises clean babble_5dB --shard 1/3

What is replayed
----------------
The loop below is audio/stt.py StreamingSTT.transcribe_stream, statement for
statement, with wall-clock time replaced by audio time:

* the wav is fed in 512-sample (32 ms) blocks, as generate_mic_stream does;
* RMS gate (stt.rms_gate) -> Silero VAD on the last 512 samples via
  StreamingSTT.process_vad_frame, threshold 0.3 (the value audio_loop.py
  passes for live turns; --vad-threshold to change);
* speech audio is collected from the first speech frame; every
  partial_interval_ms of *audio* after that (config: stt.partial_interval_ms)
  the last 8 s of speech audio is transcribed with beam_size=1, no word
  timestamps; empty results are not emitted (as live);
* end of turn = trailing non-speech for stt.silence_hang_ms; the final
  transcript is beam_size=5 with word timestamps over all speech audio;
  the turn ends at the first endpoint, exactly as live.
* mic_gain is NOT applied (it compensates a quiet laptop mic; renders are
  already levelled to -26 dBFS speech-active RMS).

Clocks and the trace fields added to §4
---------------------------------------
All times are ms from true speech onset (forced-aligned start of the first
word), so word times, partial times and the VAD endpoint share one clock.
  partials[].t_ms         audio time at which the partial is requested
  partials[].compute_ms   measured ASR wall time for that partial (GPU)
  partials[].available_ms when the text exists for a consumer, modelling one
                          serial ASR worker: max(t_ms, previous available) + compute
  partials[].stable       text identical to the previous partial (normalised)
  partials[].stable_prefix_words  words shared with the previous partial's prefix
  asr_final.{t_ms, compute_ms, available_ms}  same convention; t_ms = VAD endpoint
  vad_user_end_ms         VAD end-of-turn (null when the VAD never confirmed one)
  user_end_ms             the turn end every consumer should use: vad_user_end_ms, or the
                          fallback below when the VAD never fired
  endpoint_reason         "vad" | "fallback_timeout" | "stream_end" (no partial text at all)
  endpoint_rule           for fallback_timeout: "no_new_partial" | "no_text_change"

Fallback end of turn (VAD never fired, e.g. under babble)
---------------------------------------------------------
Evaluated online over the emitted partials: the turn ends at the first of
  * last partial's t_ms + silence_hang_ms (no new non-empty partial for the hang), or
  * last text change + FALLBACK_NO_CHANGE_MS (2.0 s with no change of the
    normalised partial text).
Partials after that moment are dropped (the live turn would be over; their
count is kept in n_partials_after_end), and asr_final is the beam-5 pass over
the speech audio up to that moment.
  vad_onset_ms            first VAD speech frame
Compute times depend on GPU load: run one trace process per GPU when the
timings matter (several shards on one GPU inflate compute_ms).

Resumable: items already in a trace file are skipped. Parallel-safe: every
append and the final sort hold <trace>.lock (exclusive create).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.v2 import voices as V  # noqa: E402
from audio.v2.asr import load_streaming_stt, stt_config  # noqa: E402
from audio.v2.items import dataset_name_for, load_items, shard, speech_root  # noqa: E402
from audio.v2.render import SR, audio_dir, read_wav16  # noqa: E402

TRACE_VERSION = "v2-speech-2"  # 2: fallback end of turn, user_end_ms
FALLBACK_NO_CHANGE_MS = 2000
MIC_BLOCK = 512           # audio_loop.generate_mic_stream block size
PARTIAL_WINDOW_S = 8      # StreamingSTT: partials see the last 8 s of speech
LIVE_VAD_THRESHOLD = 0.3  # audio_loop.py passes vad_threshold=0.3 for live turns


def _words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^\w' ]", " ", text.lower()).split() if w]


def fallback_end(partials: list[dict], hang_ms: float, no_change_ms: float = FALLBACK_NO_CHANGE_MS):
    """Online fallback end of turn over emitted partials (times on the trace clock).
    Returns (t_ms, rule) or (None, None) when there is no partial."""
    if not partials:
        return None, None
    change_t = partials[0]["t_ms"]
    prev = _words(partials[0]["text"])
    for a, b in zip(partials, partials[1:]):
        cands = []
        if b["t_ms"] - a["t_ms"] > hang_ms:
            cands.append((a["t_ms"] + hang_ms, "no_new_partial"))
        if b["t_ms"] > change_t + no_change_ms:
            cands.append((change_t + no_change_ms, "no_text_change"))
        if cands:
            t, rule = min(cands)
            return int(round(t)), rule
        w = _words(b["text"])
        if w != prev:
            change_t, prev = b["t_ms"], w
    t, rule = min((partials[-1]["t_ms"] + hang_ms, "no_new_partial"),
                  (change_t + no_change_ms, "no_text_change"))
    return int(round(t)), rule


def _lcp(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Replayer:
    def __init__(self, cfg: dict, vad_threshold: float, partial_interval_ms: int | None):
        self.cfg = cfg
        self.stt = load_streaming_stt(cfg)
        self.vad_threshold = vad_threshold
        self.interval = int(partial_interval_ms or self.stt.partial_interval_ms)
        self.hang = float(self.stt.silence_hang_ms)
        self.rms_gate = float(cfg["stt"].get("rms_gate", 0.003))
        self.lang = self.stt.language
        # warm-up so CUDA initialisation is not billed to the first partial
        self._transcribe(np.zeros(SR, np.float32), 1, False)

    def settings(self) -> dict:
        s = self.cfg["stt"]
        return {"model_size": s["model_size"], "device": s["device"], "compute_type": s["compute_type"],
                "language": self.lang, "partial_beam_size": 1, "final_beam_size": 5,
                "final_word_timestamps": True, "partial_window_s": PARTIAL_WINDOW_S,
                "vad": "silero-vad (torch.hub snakers4/silero-vad)", "vad_threshold": self.vad_threshold,
                "vad_frame_samples": self.stt.frame_samples, "block_samples": MIC_BLOCK,
                "silence_hang_ms": self.hang, "rms_gate": self.rms_gate, "mic_gain_applied": 1.0,
                "clock": "simulated audio time; compute measured on this host"}

    def _transcribe(self, audio, beam, word_ts):
        t0 = time.perf_counter()
        segs, _ = self.stt.model.transcribe(audio, beam_size=beam, word_timestamps=word_ts,
                                            language=self.lang)
        text = " ".join([s.text.strip() for s in segs if s.text]).strip()  # consumes the generator
        return text, (time.perf_counter() - t0) * 1000.0

    def replay(self, x: np.ndarray, onset_ms: float) -> dict:
        stt = self.stt
        if hasattr(stt.vad_model, "reset_states"):
            try:
                stt.vad_model.reset_states()
            except Exception:
                pass
        n_blocks = int(np.ceil(len(x) / MIC_BLOCK))
        x = np.pad(x, (0, n_blocks * MIC_BLOCK - len(x)))
        blk_ms = MIC_BLOCK * 1000.0 / SR
        buffer, speech_buffer = [], []
        speech_started = False
        last_partial = 0.0
        silence_start = None
        busy_until = -1e9
        partials, n_empty, vad_onset = [], 0, None
        prev_words = None
        prev_norm = None

        def rel(t):
            return int(round(t - onset_ms))

        final = None
        endpoint = None
        for k in range(n_blocks):
            chunk = x[k * MIC_BLOCK:(k + 1) * MIC_BLOCK]
            now = (k + 1) * blk_ms
            buffer.append(chunk)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < self.rms_gate:
                p = 0.0
            elif (k + 1) * MIC_BLOCK >= stt.frame_samples:
                full = np.concatenate(buffer[-2:]) if len(buffer) > 1 else buffer[0]
                p = stt.process_vad_frame(full[-stt.frame_samples:])
            else:
                p = 0.0

            if p >= self.vad_threshold:
                if not speech_started:
                    speech_buffer = [chunk]
                    vad_onset = now - blk_ms
                else:
                    speech_buffer.append(chunk)
                speech_started = True
                silence_start = None
            elif speech_started:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = now

            if speech_started and speech_buffer and now - last_partial >= self.interval:
                last_partial = now
                audio = np.concatenate(speech_buffer)[-SR * PARTIAL_WINDOW_S:]
                text, c = self._transcribe(audio, 1, False)
                start = max(now, busy_until)
                busy_until = start + c
                if text:
                    w = _words(text)
                    partials.append({
                        "t_ms": rel(now), "available_ms": rel(busy_until), "compute_ms": round(c, 1),
                        "audio_ms": int(len(audio) * 1000 / SR), "text": text,
                        "stable": prev_norm is not None and w == prev_norm,
                        "stable_prefix_words": _lcp(prev_words, w) if prev_words is not None else 0,
                    })
                    prev_words, prev_norm = w, w
                else:
                    n_empty += 1

            if speech_started and silence_start is not None and now - silence_start >= self.hang:
                endpoint = ("vad", now)
                break

        rule, n_after = None, 0
        if endpoint is None and speech_buffer:
            fb, rule = fallback_end(partials, self.hang)
            if fb is None:
                endpoint = ("stream_end", n_blocks * blk_ms)
            else:
                end_file = fb + onset_ms
                kept = [p for p in partials if p["t_ms"] <= fb]
                n_after = len(partials) - len(kept)
                partials = kept
                busy_until = max([p["available_ms"] + onset_ms for p in kept] + [-1e9])
                a = int(round(vad_onset * SR / 1000))
                speech_buffer = [x[a: max(a + 1, int(round(end_file * SR / 1000)))]]
                endpoint = ("fallback_timeout", end_file)
        if endpoint is not None:
            text, c = self._transcribe(np.concatenate(speech_buffer), 5, True)
            start = max(endpoint[1], busy_until)
            final = {"t_ms": rel(endpoint[1]), "available_ms": rel(start + c),
                     "compute_ms": round(c, 1), "text": text}
        return {
            "partials": partials, "n_empty_partials": n_empty, "asr_final": final,
            "vad_user_end_ms": rel(endpoint[1]) if endpoint and endpoint[0] == "vad" else None,
            "user_end_ms": rel(endpoint[1]) if endpoint else None,
            "endpoint_reason": endpoint[0] if endpoint else "no_speech",
            "endpoint_rule": rule if endpoint and endpoint[0] == "fallback_timeout" else None,
            "n_partials_after_end": n_after,
            "vad_onset_ms": rel(vad_onset) if vad_onset is not None else None,
        }


# ---------------------------------------------------------------------------- file plumbing
@contextmanager
def file_lock(path: Path, timeout_s: float = 600.0):
    lock = path.with_name(path.name + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"{lock} held for > {timeout_s}s; remove it if its owner died")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass


def max_compute_ms(rec: dict) -> float:
    cs = [p["compute_ms"] for p in rec["partials"]]
    if rec.get("asr_final"):
        cs.append(rec["asr_final"]["compute_ms"])
    return max(cs) if cs else 0.0


def done_ids(path: Path, retrace_over_ms: float | None = None) -> set[str]:
    """Item ids already traced; with retrace_over_ms, items whose slowest ASR call exceeded it
    (host stalls: paging, contention) are treated as not done and get traced again."""
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    r = json.loads(line)
                except Exception:
                    continue  # a torn last line from a killed process; finalize drops it
                if retrace_over_ms is None or max_compute_ms(r) <= retrace_over_ms:
                    out.add(r["item_id"])
    return out


def append_line(path: Path, rec: dict):
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def finalize(path: Path):
    """Drop torn lines and duplicates (the last record of an item wins, so a re-trace
    replaces the earlier one), sort by item_id."""
    if not path.exists():
        return
    with file_lock(path):
        recs = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                recs[r["item_id"]] = r
        tmp = path.with_name(path.name + f".{os.getpid()}.part")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for k in sorted(recs):
                fh.write(json.dumps(recs[k], ensure_ascii=False) + "\n")
        os.replace(tmp, path)


def upgrade_fallback(path: Path, name: str, rep: "Replayer") -> int:
    """Apply the fallback end of turn to v2-speech-1 records with no VAD endpoint:
    truncate partials at the fallback and redo the final pass on the audio up to it."""
    if not path.exists():
        return 0
    n = 0
    with file_lock(path):
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in recs:
            if "user_end_ms" in r:
                continue
            r["user_end_ms"] = r["vad_user_end_ms"]
            r["endpoint_rule"], r["n_partials_after_end"] = None, 0
            if r["endpoint_reason"] == "stream_end":
                fb, rule = fallback_end(r["partials"], rep.hang)
                if fb is not None:
                    vid, wpm, noise = r["voice_id"], r["wpm_target"], r["noise"]
                    x = read_wav16(audio_dir(name, vid, wpm, noise) / f"{r['item_id']}.wav")
                    onset = r["lead_ms"]
                    kept = [p for p in r["partials"] if p["t_ms"] <= fb]
                    r["n_partials_after_end"] = len(r["partials"]) - len(kept)
                    r["partials"] = kept
                    a = int(round((r["vad_onset_ms"] + onset) * SR / 1000))
                    b = int(round((fb + onset) * SR / 1000))
                    text, c = rep._transcribe(x[a:max(a + 1, b)], 5, True)
                    busy = max([p["available_ms"] for p in kept] + [fb])
                    r["asr_final"] = {"t_ms": fb, "available_ms": int(round(busy + c)),
                                      "compute_ms": round(c, 1), "text": text}
                    r["user_end_ms"], r["endpoint_reason"], r["endpoint_rule"] = fb, "fallback_timeout", rule
            r["trace_version"] = TRACE_VERSION
            n += 1
        tmp = path.with_name(path.name + f".{os.getpid()}.part")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    return n


def build_record(name, item, vid, wpm, noise, side, res, rep, wall_ms):
    onset = side["speech_onset_ms_in_file"]
    words = [{"w": w["w"], "start_ms": w["start_ms"] - onset, "end_ms": w["end_ms"] - onset}
             for w in side["words_in_file"]]
    wav = audio_dir(name, vid, wpm, noise) / f"{item['item_id']}.wav"
    nm = side.get("noise_meta")
    return {
        "item_id": item["item_id"], "voice_id": vid, "accent": side["accent"],
        "wpm_target": wpm, "wpm_actual": side["wpm_actual"], "noise": noise,
        "noise_meta": ({k: nm[k] for k in ("type", "snr_db", "n_talkers", "seed")} if nm else None),
        "tts_engine": side["tts_engine"], "length_scale": side.get("length_scale"),
        "audio_path": wav.relative_to(speech_root(name).parent).as_posix(),
        "duration_ms": words[-1]["end_ms"] - words[0]["start_ms"],
        "lead_ms": onset, "file_ms": side["file_ms"],
        "words": words,
        "partials": res["partials"], "n_empty_partials": res["n_empty_partials"],
        "asr_final": res["asr_final"],
        "vad_user_end_ms": res["vad_user_end_ms"], "user_end_ms": res["user_end_ms"],
        "endpoint_reason": res["endpoint_reason"], "endpoint_rule": res["endpoint_rule"],
        "n_partials_after_end": res["n_partials_after_end"],
        "vad_onset_ms": res["vad_onset_ms"],
        "asr_model": f"faster-whisper-{rep.cfg['stt']['model_size']}",
        "asr_settings": rep.settings(),
        "partial_interval_ms": rep.interval,
        "word_timing": {"method": "ctc_forced_alignment", "model": side["align_model"]},
        "replay_wall_ms": round(wall_ms, 1), "trace_version": TRACE_VERSION,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay rendered speech through streaming STT.")
    ap.add_argument("--items", required=True)
    ap.add_argument("--name")
    ap.add_argument("--split", default="all")
    ap.add_argument("--voices", nargs="+", default=V.DEFAULT_VOICES)
    ap.add_argument("--rates", nargs="+", type=int, default=V.DEFAULT_RATES)
    ap.add_argument("--noises", nargs="+", default=V.DEFAULT_NOISES)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="explicit condition ids <voice>__<wpm>wpm__<noise> (overrides the three lists)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard")
    ap.add_argument("--render-missing", action="store_true",
                    help="render audio that is not there yet (otherwise such items are skipped)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default=None)
    ap.add_argument("--model-size", default=None, help="default: configs/config.yaml audio.stt.model_size")
    ap.add_argument("--vad-threshold", type=float, default=LIVE_VAD_THRESHOLD)
    ap.add_argument("--partial-interval-ms", type=int, default=None)
    ap.add_argument("--traces-dir", default=None, help="default: datasets/v2/<name>/speech/traces")
    ap.add_argument("--upgrade-fallback", action="store_true",
                    help="only apply the fallback end-of-turn rule to existing v2-speech-1 traces")
    ap.add_argument("--retrace-over-ms", type=float, default=None,
                    help="re-trace items whose slowest ASR call took longer than this (host stalls)")
    a = ap.parse_args(argv)

    name = a.name or dataset_name_for(Path(a.items))
    items_all = load_items(a.items, a.split)
    if a.limit:
        items_all = items_all[: a.limit]
    todo = shard(items_all, a.shard)
    conds = a.conditions or [V.condition_id(v, r, n) for v in a.voices for r in a.rates for n in a.noises]
    out_dir = Path(a.traces_dir) if a.traces_dir else speech_root(name) / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = Replayer(stt_config(a.model_size, a.device, a.compute_type), a.vad_threshold,
                   a.partial_interval_ms)
    print(f"{name}: {len(todo)} items x {len(conds)} conditions; ASR {rep.settings()['model_size']} "
          f"on {a.device}, partial every {rep.interval} ms", flush=True)
    if a.upgrade_fallback:
        for cond in conds:
            n = upgrade_fallback(out_dir / f"{cond}.jsonl", name, rep)
            print(f"  {cond}: {n} records upgraded", flush=True)
        return
    for cond in conds:
        vid, wpm, noise = V.parse_condition(cond)
        path = out_dir / f"{cond}.jsonl"
        have = done_ids(path, a.retrace_over_ms)
        t0, n, skipped = time.perf_counter(), 0, 0
        for it in todo:
            if it["item_id"] in have:
                continue
            d = audio_dir(name, vid, wpm, noise)
            side_p, wav_p = d / f"{it['item_id']}.json", d / f"{it['item_id']}.wav"
            if not (side_p.exists() and wav_p.exists()):
                if not a.render_missing:
                    skipped += 1
                    continue
                from audio.v2 import render as R

                if noise == "clean":
                    R.render_clean(name, it, vid, wpm, align_check=False)
                else:
                    R.render_noisy(name, it, vid, wpm, noise, items_all, V.DEFAULT_VOICES, {})
            side = json.loads(side_p.read_text(encoding="utf-8"))
            x = read_wav16(wav_p)
            w0 = time.perf_counter()
            res = rep.replay(x, side["speech_onset_ms_in_file"])
            rec = build_record(name, it, vid, wpm, noise, side, res, rep, (time.perf_counter() - w0) * 1000)
            append_line(path, rec)
            n += 1
        finalize(path)
        dt = time.perf_counter() - t0
        print(f"  {cond}: +{n} traced ({dt / max(n, 1):.2f} s/item), {len(have)} already done, "
              f"{skipped} skipped (no audio)", flush=True)


if __name__ == "__main__":
    main()
    # CTranslate2's CUDA teardown at interpreter exit can return a spurious non-zero
    # code on Windows after all work is written; exit explicitly instead.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
