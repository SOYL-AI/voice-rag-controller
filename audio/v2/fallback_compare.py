"""
fallback_compare.py - premature fallback ends of turn: unarmed rule vs the armed rule.

    python -m audio.v2.fallback_compare --name v1_scenarios [--noise babble_10dB babble_5dB]

For every trace whose VAD never fired, the end of turn is recomputed from the
stored partials with the original rule (armed=False) and with the armed rule
"armed-1" (trace.fallback_end). Premature = end of turn before the last
reference word ends. Records whose partials were already truncated at an
earlier fallback end only allow the rule that produced them to be recomputed
exactly; for those the other rule's value is reported as "unknown" when it
would fall after the truncation point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio.v2.items import speech_root  # noqa: E402
from audio.v2.trace import fallback_arm_ms, fallback_end  # noqa: E402
from audio.v2.voices import parse_condition  # noqa: E402

HANG_MS = 800


def compare(recs):
    out = {"n_no_vad_endpoint": 0, "unarmed": {"premature": 0, "unknown": 0, "no_end": 0},
           "armed": {"premature": 0, "unknown": 0, "no_end": 0}}
    for r in recs:
        if r.get("vad_user_end_ms") is not None:
            continue
        out["n_no_vad_endpoint"] += 1
        end = r["words"][-1]["end_ms"]
        truncated_at = r["user_end_ms"] if r.get("n_partials_after_end") else None
        arm = fallback_arm_ms(r["partials"], r.get("vad_onset_ms"))
        for key, (t, _) in (("unarmed", fallback_end(r["partials"], HANG_MS, armed=False)),
                            ("armed", fallback_end(r["partials"], HANG_MS, arm_ms=arm))):
            if t is None:
                out[key]["no_end"] += 1
            elif truncated_at is not None and t > truncated_at and truncated_at < end:
                out[key]["unknown"] += 1
            elif t < end:
                out[key]["premature"] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--noise", nargs="*", default=None)
    a = ap.parse_args(argv)
    tot, per = None, {}
    for p in sorted((speech_root(a.name) / "traces").glob("*.jsonl")):
        if a.noise and parse_condition(p.stem)[2] not in a.noise:
            continue
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        c = compare(recs)
        if c["n_no_vad_endpoint"]:
            per[p.stem] = c
    agg = {"n_no_vad_endpoint": sum(c["n_no_vad_endpoint"] for c in per.values())}
    for k in ("unarmed", "armed"):
        agg[k] = {m: sum(c[k][m] for c in per.values()) for m in ("premature", "unknown", "no_end")}
    print(json.dumps({"total": agg}, indent=1))
    return agg, per


if __name__ == "__main__":
    main()
