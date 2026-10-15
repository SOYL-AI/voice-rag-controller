"""
trace_report.py - per-condition quality of speech traces.

    python -m audio.v2.trace_report --name v1_scenarios
    python -m audio.v2.trace_report --traces datasets/v2/d1_open/speech/traces

Metrics (all text scored after textnorm.norm_words, i.e. Whisper's English
normaliser applied to reference and hypothesis alike):

partial_wer      micro WER of every emitted partial against the reference
                 prefix spoken by then: reference words whose midpoint
                 (start+end)/2 <= t_ms. Partials before any reference word is
                 half-spoken are excluded (counted in n_partials_early).
final_wer        micro WER of asr_final against the full reference.
final_ser        share of items whose final is not an exact (normalised) match.
rev_strict       share of consecutive-partial transitions where any word of the
                 earlier partial is changed or removed in the later one.
rev_lenient      same, ignoring the earlier partial's last word (it may have been
                 cut mid-word).
items_revised    share of items with at least one lenient revision.
eot_*            end-of-turn: vad_user_end_ms - true speech end (last word end).
                 Includes the configured silence hang (800 ms); eot_lag_beyond_hang
                 subtracts it. eot_premature = endpoint fired before the speech ended
                 (turn cut mid-utterance); user_end_* is the same for user_end_ms
                 (VAD or fallback end of turn), fallback_share = share of turns ended
                 by the fallback rule; eot_missed = no VAD endpoint before the
                 audio ran out.
partial latency  compute_ms (ASR time per partial), avail_lag = available_ms - t_ms
                 (compute + queueing), first_partial_avail = available_ms of the
                 first partial (from speech onset), final_avail_after_end =
                 asr_final.available_ms - true speech end.
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
from audio.v2.textnorm import NORMALISER, edit_ops, norm_words  # noqa: E402
from audio.v2.voices import parse_condition, rate_token  # noqa: E402


def _q(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def _is_prefix(a, b):
    return len(a) <= len(b) and b[: len(a)] == a


def condition_metrics(recs: list[dict]) -> dict:
    p_err = p_ref = early = n_part = 0
    f_err = f_ref = f_wrong = 0
    trans = rev_s = rev_l = items_rev = 0
    eot, lag_beyond, premature, missed = [], [], 0, 0
    uend, n_fallback, fb_lag = [], 0, []
    comp, avail_lag, first_av, final_after, wpm, n_parts_item = [], [], [], [], [], []
    hang = None
    for r in recs:
        ref_words = r["words"]
        ref_text = " ".join(w["w"] for w in ref_words)
        speech_end = ref_words[-1]["end_ms"]
        wpm.append(r.get("wpm_actual") or np.nan)
        hang = r.get("asr_settings", {}).get("silence_hang_ms", hang)
        prev = None
        item_rev = False
        n_parts_item.append(len(r["partials"]))
        for p in r["partials"]:
            n_part += 1
            hyp = norm_words(p["text"])
            pref = [w["w"] for w in ref_words if (w["start_ms"] + w["end_ms"]) / 2 <= p["t_ms"]]
            if not pref:
                early += 1
            else:
                rw = norm_words(" ".join(pref))
                s, d, i = edit_ops(rw, hyp)
                p_err += s + d + i
                p_ref += len(rw)
            if prev is not None:
                trans += 1
                if not _is_prefix(prev, hyp):
                    rev_s += 1
                if not _is_prefix(prev[:-1], hyp):
                    rev_l += 1
                    item_rev = True
            prev = hyp
            comp.append(p["compute_ms"])
            avail_lag.append(p["available_ms"] - p["t_ms"])
        if r["partials"]:
            first_av.append(r["partials"][0]["available_ms"])
        items_rev += item_rev
        fin = (r.get("asr_final") or {}).get("text", "")
        rw, hw = norm_words(ref_text), norm_words(fin)
        s, d, i = edit_ops(rw, hw)
        f_err += s + d + i
        f_ref += len(rw)
        f_wrong += (rw != hw)
        if r.get("asr_final"):
            final_after.append(r["asr_final"]["available_ms"] - speech_end)
        ue = r.get("user_end_ms", r["vad_user_end_ms"])
        if ue is not None:
            uend.append(ue - speech_end)
        if r.get("endpoint_reason") == "fallback_timeout":
            n_fallback += 1
            fb_lag.append(r["user_end_ms"] - speech_end)
        if r["vad_user_end_ms"] is None:
            missed += 1
        else:
            e = r["vad_user_end_ms"] - speech_end
            eot.append(e)
            lag_beyond.append(e - (hang or 0))
            premature += e < 0
    n = len(recs)
    return {
        "n_items": n, "wpm_actual_mean": float(np.nanmean(wpm)) if n else None,
        "partials_per_item": float(np.mean(n_parts_item)) if n else None,
        "partial_wer": p_err / p_ref if p_ref else None, "n_partials": n_part,
        "n_partials_early": early,
        "final_wer": f_err / f_ref if f_ref else None, "final_ser": f_wrong / n if n else None,
        "rev_strict": rev_s / trans if trans else None, "rev_lenient": rev_l / trans if trans else None,
        "items_revised": items_rev / n if n else None,
        "eot_median_ms": _q(eot, 50), "eot_mean_ms": float(np.mean(eot)) if eot else None,
        "eot_p90_ms": _q(eot, 90), "eot_lag_beyond_hang_median_ms": _q(lag_beyond, 50),
        "eot_premature": premature / n if n else None, "eot_missed": missed / n if n else None,
        "user_end_median_ms": _q(uend, 50), "user_end_p90_ms": _q(uend, 90),
        "user_end_premature": float(np.mean(np.array(uend) < 0)) if uend else None,
        "fallback_share": n_fallback / n if n else None,
        "fallback_n": n_fallback,
        "fallback_premature_n": int(sum(x < 0 for x in fb_lag)),
        "fallback_lag_median_ms": _q(fb_lag, 50), "fallback_lag_p90_ms": _q(fb_lag, 90),
        "compute_median_ms": _q(comp, 50), "compute_p90_ms": _q(comp, 90),
        "avail_lag_median_ms": _q(avail_lag, 50), "avail_lag_p90_ms": _q(avail_lag, 90),
        "first_partial_avail_median_ms": _q(first_av, 50),
        "final_avail_after_end_median_ms": _q(final_after, 50),
        "replay_wall_s_per_item": float(np.mean([r.get("replay_wall_ms", np.nan) for r in recs])) / 1000
        if n else None,
    }


def _fmt(v, pct=False, nd=0):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def fallback_markdown(rows: dict) -> str:
    """Fallback-ended turns listed separately: their end-of-turn latency is structurally higher."""
    out = ["| condition | n | fallback turns | premature | end-of-turn lag median ms | p90 ms |",
           "|---|---|---|---|---|---|"]
    for cond, m in rows.items():
        if not m.get("fallback_n"):
            continue
        out.append(f"| {cond} | {m['n_items']} | {m['fallback_n']} | {m['fallback_premature_n']} | "
                   f"{_fmt(m['fallback_lag_median_ms'])} | {_fmt(m['fallback_lag_p90_ms'])} |")
    return "\n".join(out) if len(out) > 2 else "(no fallback turns)"


def to_markdown(rows: dict) -> str:
    hdr = ("| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | "
           "EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | "
           "final avail after end |")
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for cond, m in rows.items():
        out.append(
            f"| {cond} | {m['n_items']} | {_fmt(m['wpm_actual_mean'])} | {_fmt(m['partial_wer'], True)} | "
            f"{_fmt(m['final_wer'], True)} | {_fmt(m['rev_lenient'], True)} | {_fmt(m['items_revised'], True)} | "
            f"{_fmt(m['eot_median_ms'])} | {_fmt(m['eot_p90_ms'])} | {_fmt(m['eot_premature'], True)} | "
            f"{_fmt(m['eot_missed'], True)} | {_fmt(m.get('fallback_share'), True)} | "
            f"{_fmt(m.get('user_end_median_ms'))} | {_fmt(m['compute_median_ms'])} | {_fmt(m['avail_lag_p90_ms'])} | "
            f"{_fmt(m['final_avail_after_end_median_ms'])} |")
    return "\n".join(out)


def group_rows(rows: dict, key_fn) -> dict:
    """Pool raw per-condition metrics by a grouping key (item-weighted means)."""
    groups: dict = {}
    for cond, m in rows.items():
        groups.setdefault(key_fn(cond), []).append(m)
    out = {}
    for g, ms in sorted(groups.items()):
        w = np.array([m["n_items"] for m in ms], float)
        agg = {"n_items": int(w.sum())}
        for k in ms[0]:
            if k == "n_items":
                continue
            vals = [(m[k], wi) for m, wi in zip(ms, w) if m[k] is not None]
            agg[k] = (sum(v * wi for v, wi in vals) / sum(wi for _, wi in vals)) if vals else None
        out[g] = agg
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="dataset name under datasets/v2/")
    ap.add_argument("--traces", help="explicit traces dir")
    ap.add_argument("--out", help="output stem (default: <speech>/trace_report)")
    a = ap.parse_args(argv)
    tdir = Path(a.traces) if a.traces else speech_root(a.name) / "traces"
    rows = {}
    for p in sorted(tdir.glob("*.jsonl")):
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if recs:
            rows[p.stem] = condition_metrics(recs)
    by_noise = group_rows(rows, lambda c: parse_condition(c)[2])
    by_rate = group_rows(rows, lambda c: rate_token(parse_condition(c)[1]))
    by_voice = group_rows(rows, lambda c: parse_condition(c)[0])
    md = ["# Speech trace quality", "",
          f"Traces: `{tdir.relative_to(_REPO_ROOT).as_posix() if tdir.is_absolute() else tdir}`; "
          f"text normaliser: {NORMALISER}. Metric definitions: audio/v2/trace_report.py.", "",
          "## By noise", "", to_markdown(by_noise), "", "## By rate", "", to_markdown(by_rate), "",
          "## By voice", "", to_markdown(by_voice), "", "## All conditions", "", to_markdown(rows), "",
          "## Fallback-ended turns (VAD never fired; listed separately, structurally later)", "",
          fallback_markdown(rows), ""]
    stem = Path(a.out) if a.out else tdir.parent / "trace_report"
    stem.with_suffix(".json").write_text(json.dumps(
        {"normaliser": NORMALISER, "conditions": rows, "by_noise": by_noise, "by_rate": by_rate,
         "by_voice": by_voice}, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
