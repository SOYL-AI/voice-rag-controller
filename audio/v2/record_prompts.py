"""
record_prompts.py - record team members reading the v2 questions aloud.

Protocol (also in docs/research_v2/SPEECH.md, "Human speech subset"):
  * Consent first. The tool prints what is recorded and how it is used and
    needs an explicit "yes". Speakers are identified only by a short code
    (e.g. "s03"); no names are stored. Recordings stay out of git.
  * Quiet room, door closed, fans/AC off if possible; laptop or headset mic
    ~20-30 cm away; same setup for a whole session.
  * Read each question once, naturally, as if asking a support agent or a
    voice assistant. Do not over-articulate. Small hesitations are fine and
    realistic; restart with "r" only if a word is wrong or skipped.
  * Press Enter, wait about half a second, speak, wait about one second, press
    Enter. The trailing second matters for end-of-turn detection.
  * ~200 questions per speaker in sessions of <= 30 minutes; the tool resumes
    where it stopped.

    # record (16 kHz mono; one wav per question)
    python -m audio.v2.record_prompts record --items datasets/v2/d2_support/items.jsonl \
        --speaker s03 --accent en-IN --l1 Hindi --n 200 --seed 3
    # after a session: align + level + pad into the render layout, so trace.py can replay it
    python -m audio.v2.record_prompts prepare --items ... --speaker s03
    python -m audio.v2.trace --items ... --conditions human-s03__natural__clean

Layout (gitignored, under datasets/v2/<name>/speech/audio/):
    human-<code>/raw/<item_id>.wav + .json        what the microphone captured
    human-<code>/natural/clean/<item_id>.wav+.json  levelled, padded, aligned (render format)
    human-<code>/session.json                     consent + speaker metadata
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.v2.items import dataset_name_for, load_items, speech_root  # noqa: E402

SR = 16000
CONSENT = """
You are about to record yourself reading questions aloud for the voice-rag
research project (SOYL). Your recordings are used only to measure how speech
recognition and end-of-turn detection behave on real speech. They are stored on
project machines, never committed to the public code repository, and are
identified only by your speaker code. You may stop at any time and ask for your
recordings to be deleted. Do not say anything personal while recording.
"""


def _dirs(name, code):
    base = speech_root(name) / "audio" / f"human-{code}"
    return base, base / "raw", base / "natural" / "clean"


def record(a):
    import sounddevice as sd

    from audio.v2.render import write_json, write_wav

    name = a.name or dataset_name_for(Path(a.items))
    items = load_items(a.items, a.split)
    random.Random(a.seed).shuffle(items)
    items = items[: a.n] if a.n else items
    base, raw, _ = _dirs(name, a.speaker)
    raw.mkdir(parents=True, exist_ok=True)
    sess_p = base / "session.json"
    if not sess_p.exists():
        print(CONSENT)
        if input('Type "yes" to consent and continue: ').strip().lower() != "yes":
            print("No consent; nothing recorded.")
            return
        write_json(sess_p, {
            "speaker_code": a.speaker, "accent": a.accent, "l1": a.l1, "gender": a.gender,
            "consent": True, "consent_at": dt.datetime.now().isoformat(timespec="seconds"),
            "input_device": sd.query_devices(kind="input")["name"], "sample_rate": SR,
            "items_file": str(Path(a.items).as_posix()), "seed": a.seed})
    todo = [it for it in items if not (raw / f"{it['item_id']}.wav").exists()]
    print(f"{len(items) - len(todo)} done, {len(todo)} to go. Keys: Enter=start/stop, "
          f"r=redo last, p=play last, s=skip, q=quit")
    i = 0
    while i < len(todo):
        it = todo[i]
        print(f"\n[{len(items) - len(todo) + i + 1}/{len(items)}]  {it['text']}")
        cmd = input("  Enter to start (s=skip, q=quit): ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        chunks = []
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            callback=lambda d, f, t, s: chunks.append(d[:, 0].copy())):
            input("  recording... Enter to stop ")
        x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        print(f"  {len(x) / SR:.1f} s, peak {peak:.2f}" + ("  (CLIPPED - move back)" if peak > 0.99 else "")
              + ("  (very quiet - move closer)" if peak < 0.05 else ""))
        nxt = input("  Enter=keep, r=redo, p=play: ").strip().lower()
        while nxt == "p":
            sd.play(x, SR)
            sd.wait()
            nxt = input("  Enter=keep, r=redo, p=play: ").strip().lower()
        if nxt == "r":
            continue
        write_wav(raw / f"{it['item_id']}.wav", x)
        write_json(raw / f"{it['item_id']}.json", {
            "item_id": it["item_id"], "text": it["text"], "speaker_code": a.speaker,
            "recorded_at": dt.datetime.now().isoformat(timespec="seconds"), "sample_rate": SR,
            "peak": round(peak, 3)})
        i += 1


def prepare(a):
    """Raw take -> render-format clean file: level to -26 dBFS, pad, force-align, sanity-check."""
    from audio.v2.align import ALIGN_MODEL, force_align
    from audio.v2.asr import whisper_for_alignment_check
    from audio.v2.render import LEAD_MS, TAIL_MS, TARGET_DBFS, active_rms, read_wav16, trim_silence, \
        write_json, write_wav
    from audio.v2.textnorm import wer

    name = a.name or dataset_name_for(Path(a.items))
    base, raw, out = _dirs(name, a.speaker)
    sess = json.loads((base / "session.json").read_text(encoding="utf-8"))
    whisper = whisper_for_alignment_check(a.device)
    n = flagged = 0
    for meta_p in sorted(raw.glob("*.json")):
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        dst = out / f"{m['item_id']}.json"
        if dst.exists():
            continue
        x = trim_silence(read_wav16(raw / f"{m['item_id']}.wav"), pad_ms=150)
        x = x * (10 ** (TARGET_DBFS / 20) / max(active_rms(x), 1e-6))
        x = np.concatenate([np.zeros(int(SR * LEAD_MS / 1000), np.float32), x,
                            np.zeros(int(SR * TAIL_MS / 1000), np.float32)])
        x *= min(1.0, 0.98 / max(float(np.max(np.abs(x))), 1e-6))
        words = force_align(x, m["text"])
        segs, _ = whisper.transcribe(x, beam_size=5, language="en")
        hyp = " ".join(s.text.strip() for s in segs)
        w = wer(m["text"], hyp)
        flagged += w > 0.3
        on, off = words[0]["start_ms"], words[-1]["end_ms"]
        write_wav(out / f"{m['item_id']}.wav", x)
        write_json(dst, {
            "item_id": m["item_id"], "text": m["text"], "voice_id": f"human-{a.speaker}",
            "accent": sess.get("accent"), "wpm_target": None, "noise": "clean",
            "tts_engine": "human", "length_scale": None, "sample_rate": SR,
            "file_ms": int(len(x) * 1000 / SR), "speech_onset_ms_in_file": on,
            "speech_end_ms_in_file": off, "words_in_file": words, "align_model": ALIGN_MODEL,
            "wpm_actual": round(60000.0 * len(words) / max(off - on, 1), 1), "level_dbfs": TARGET_DBFS,
            "reading_check": {"whisper_final": hyp, "wer": round(w, 3), "suspect_misread": w > 0.3}})
        n += 1
    print(f"prepared {n} recordings for human-{a.speaker}; {flagged} flagged as possible misreads "
          f"(WER > 0.3 against the prompt) - listen to those before use")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("record", "prepare"):
        p = sub.add_parser(c)
        p.add_argument("--items", required=True)
        p.add_argument("--name")
        p.add_argument("--speaker", required=True, help="anonymous speaker code, e.g. s03")
        p.add_argument("--split", default="all")
    r = sub.choices["record"]
    r.add_argument("--accent", required=True, help="self-reported, e.g. en-IN")
    r.add_argument("--l1", default=None)
    r.add_argument("--gender", default=None)
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--seed", type=int, default=0)
    sub.choices["prepare"].add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    record(a) if a.cmd == "record" else prepare(a)


if __name__ == "__main__":
    main()
