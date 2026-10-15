"""
align_report.py - summarise the word-timestamp accuracy checks stored in clean-render sidecars.

    python -m audio.v2.align_report --name v1_scenarios

Reports, over all clean renders that carry `align_check` (render.py --check-existing):
signed Whisper-minus-forced start/end differences for words both agree on
(median and median absolute), the share within 50/100 ms, and the first-word
onset / last-word offset relative to an energy-based speech-activity bound. The
energy comparison is not independent of the refinement step (refine_with_energy
uses the same energy criterion); the Whisper comparison is independent but
Whisper's DTW word times carry their own bias, so its spread (IQR, share within
50 ms after removing the median shift) is the more informative number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.v2.items import speech_root  # noqa: E402


def summarise(paths) -> dict:
    ds, de, on, off, nm, nw, n = [], [], [], [], 0, 0, 0
    for p in paths:
        m = json.loads(Path(p).read_text(encoding="utf-8"))
        c = m.get("align_check")
        if not c:
            continue
        n += 1
        ds += c["start_diff_ms"]
        de += c["end_diff_ms"]
        on.append(c["onset_vs_energy_ms"])
        off.append(c["offset_vs_energy_ms"])
        nm += c["n_matched"]
        nw += c["n_words"]
    ds, de, on, off = map(np.array, (ds, de, on, off))

    def st(x):
        if not len(x):
            return None
        return {"median": float(np.median(x)), "median_abs": float(np.median(np.abs(x))),
                "within_50ms": float(np.mean(np.abs(x) <= 50)), "within_100ms": float(np.mean(np.abs(x) <= 100)),
                "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
                "after_median_shift_within_50ms": float(np.mean(np.abs(x - np.median(x)) <= 50))}

    return {"n_renders": n, "words_matched_share": nm / nw if nw else None,
            "start_whisper_minus_forced": st(ds), "end_whisper_minus_forced": st(de),
            "first_word_onset_minus_energy": st(on), "last_word_end_minus_energy": st(off)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    paths = sorted((speech_root(a.name) / "audio").glob("*/*/clean/*.json"))
    out = summarise(paths)
    dst = speech_root(a.name) / "align_check.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
