"""
latency_breakdown.py — where does time-to-first-audio (TTFA) actually go?

Reads conversation event logs (eventlog/schema.md) and decomposes each
turn's TTFA into consecutive, non-overlapping stages along the critical
path from ``user_end`` to ``tts_first_audio``. The stages always sum to
the turn's TTFA exactly, so the shares are additive and a stacked bar of
stage means equals mean TTFA.

Critical-path stages (all clamped so the path is monotone in time):

    dispatch_wait      user_end -> retrieval_launched, only the part after
                       user_end (ASR final + controller decision + any
                       dispatch delay). 0 when retrieval launched during
                       speech, or when no retrieval happened.
    retrieval_exposed  the part of retrieval (launched -> result) that runs
                       after user_end, i.e. the retrieval latency the user
                       actually waits through. Retrieval that finished
                       during speech is "hidden" and reported separately.
    pre_llm_overhead   max(user_end, retrieval_result) -> LLM request sent
                       (prompt assembly, thread hop). The LLM request time
                       is reconstructed as llm_first_token - ttft_ms, where
                       ttft_ms is measured by llm/client.py on the same clock.
    llm_prefill        LLM request -> first token (TTFT: queueing, network,
                       prefill).
    llm_to_tts_input   first token -> the text TTS starts on. In batch TTS
                       (v1) that is llm_final (the whole answer); in
                       streaming TTS it is llm_first_clause.
    tts_first_audio    TTS input ready -> tts_first_audio.

Also reported, as offsets from user_end (negative = during speech):
asr_final, controller_action, retrieval_launched, retrieval_result,
llm_request, llm_first_token, llm_first_clause (if logged), llm_final,
tts_first_audio.

Clock note: t_ms comes from time.monotonic_ns() on the run host. On the
Windows host used for E2/E3 the effective resolution is ~15.6 ms, so stage
values are quantised to multiples of ~16 ms and small stages read as 0.

Usage:
    python -m harness.latency_breakdown --logs "logs/conv_*.jsonl" \\
        --label e2 --out-dir analysis/latency_breakdown \\
        --paper-dir paper
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

POLICIES = ("sentence_end", "immediate", "fixed_early_proxy", "forecast")

STAGES = (
    "dispatch_wait",
    "retrieval_exposed",
    "pre_llm_overhead",
    "llm_prefill",
    "llm_to_tts_input",
    "tts_first_audio",
)

STAGE_LABELS = {
    "dispatch_wait": "ASR final + controller + dispatch wait",
    "retrieval_exposed": "Retrieval (exposed after user end)",
    "pre_llm_overhead": "Prompt assembly / hand-off",
    "llm_prefill": "LLM time to first token",
    "llm_to_tts_input": "LLM generation until TTS input",
    "tts_first_audio": "TTS to first audio",
}

OFFSET_EVENTS = (
    "asr_final",
    "controller_action",
    "retrieval_launched",
    "retrieval_result",
    "llm_request",
    "llm_first_token",
    "llm_first_clause",
    "llm_final",
    "tts_first_audio",
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _read_events(path: Path) -> List[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def policy_from_name(name: str) -> Optional[str]:
    """Timing policy encoded in a conversation id / file stem, if any.

    Checks the longest names first so ``fixed_early_proxy`` is never
    mistaken for a substring match of something shorter.
    """
    for policy in sorted(POLICIES, key=len, reverse=True):
        if policy in name:
            return policy
    return None


def parse_turn(events: List[dict], turn_id=None) -> Optional[dict]:
    """
    Decompose one turn. ``events`` may contain several turns; ``turn_id``
    selects one (default: the first turn that has a user_end).

    Returns None when the turn never reached tts_first_audio (no TTFA).
    """
    if turn_id is None:
        for e in events:
            if e.get("event_type") == "user_end":
                turn_id = e.get("turn_id")
                break
    turn = sorted(
        (e for e in events if e.get("turn_id") == turn_id), key=lambda e: e["t_ms"]
    )
    first: Dict[str, dict] = {}
    for e in turn:
        first.setdefault(e["event_type"], e)
    if "user_end" not in first or "tts_first_audio" not in first:
        return None

    ue = first["user_end"]["t_ms"]

    def rel(name):
        e = first.get(name)
        return None if e is None else e["t_ms"] - ue

    offsets = {name: rel(name) for name in OFFSET_EVENTS if name != "llm_request"}

    # Reconstruct the LLM request time from the client's own ttft_ms.
    llm_final_payload = (first.get("llm_final") or {}).get("payload") or {}
    ttft = llm_final_payload.get("ttft_ms")
    if offsets.get("llm_first_token") is not None and ttft is not None:
        offsets["llm_request"] = offsets["llm_first_token"] - ttft
    else:
        offsets["llm_request"] = None

    retrieved = offsets.get("retrieval_launched") is not None and offsets.get(
        "retrieval_result"
    ) is not None
    streaming = offsets.get("llm_first_clause") is not None

    ttfa = offsets["tts_first_audio"]

    # Critical path, clamped monotone.
    a = 0
    if retrieved:
        b = max(a, offsets["retrieval_launched"])
        c = max(b, offsets["retrieval_result"])
    else:
        b = a
        c = a
    d = offsets["llm_request"]
    if d is None:
        d = offsets.get("llm_first_token", c)
    d = max(c, d)
    e_ = max(d, offsets.get("llm_first_token") if offsets.get("llm_first_token") is not None else d)
    tts_in = offsets.get("llm_first_clause") if streaming else offsets.get("llm_final")
    f = max(e_, tts_in if tts_in is not None else e_)
    g = max(f, ttfa)
    # If the clamping pushed the path past the real first-audio time (only
    # possible with out-of-order events), shrink the last stage instead so
    # the stages still sum to the observed TTFA.
    stages = {
        "dispatch_wait": b - a,
        "retrieval_exposed": c - b,
        "pre_llm_overhead": d - c,
        "llm_prefill": e_ - d,
        "llm_to_tts_input": f - e_,
        "tts_first_audio": g - f,
    }
    overshoot = g - ttfa
    if overshoot > 0:
        for name in reversed(STAGES):
            take = min(stages[name], overshoot)
            stages[name] -= take
            overshoot -= take
            if overshoot <= 0:
                break

    retrieval_ms = (
        offsets["retrieval_result"] - offsets["retrieval_launched"] if retrieved else None
    )
    hidden = None
    if retrieved:
        hidden = max(0, min(offsets["retrieval_result"], 0) - offsets["retrieval_launched"])

    gen_ms = None
    if offsets.get("llm_final") is not None and offsets.get("llm_first_token") is not None:
        gen_ms = offsets["llm_final"] - offsets["llm_first_token"]

    return {
        "turn_id": turn_id,
        "path": "retrieve" if retrieved else "skip",
        "tts_mode": "streaming" if streaming else "batch",
        "ttfa_ms": ttfa,
        "stages_ms": stages,
        "offsets_ms": offsets,
        "retrieval_ms": retrieval_ms,
        "retrieval_hidden_ms": hidden,
        "ttft_ms": ttft,
        "generation_ms": gen_ms,
        "n_tokens": llm_final_payload.get("n_tokens"),
    }


def parse_log(path) -> Optional[dict]:
    path = Path(path)
    rec = parse_turn(_read_events(path))
    if rec is None:
        return None
    rec["log"] = path.name
    rec["policy"] = policy_from_name(path.stem)
    return rec


def load_runs(patterns: Iterable[str]) -> List[dict]:
    runs = []
    for pattern in patterns:
        for p in sorted(glob.glob(pattern)):
            rec = parse_log(p)
            if rec is not None:
                runs.append(rec)
    return runs


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _quantile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolation quantile (numpy's default), no numpy needed."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def _stats(values: List[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "p50": _quantile(vals, 0.5),
        "p95": _quantile(vals, 0.95),
    }


def group_key(run: dict, group_by: str) -> str:
    policy = run.get("policy") or "unknown"
    if group_by == "policy":
        return policy
    if group_by == "policy_path":
        return f"{policy}/{run['path']}"
    if group_by == "policy_path_tts":
        return f"{policy}/{run['path']}/{run['tts_mode']}"
    raise ValueError(f"unknown group_by {group_by!r}")


def summarize(runs: List[dict], group_by: str = "policy") -> dict:
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r, group_by)].append(r)

    out = {}
    for key in sorted(groups, key=_group_sort_key):
        rs = groups[key]
        ttfa = [r["ttfa_ms"] for r in rs]
        ttfa_stats = _stats(ttfa)
        stage_stats = {}
        for s in STAGES:
            vals = [r["stages_ms"][s] for r in rs]
            st = _stats(vals)
            st["share_of_mean_ttfa"] = (
                st["mean"] / ttfa_stats["mean"] if ttfa_stats["mean"] else None
            )
            stage_stats[s] = st
        offset_stats = {
            name: _stats([r["offsets_ms"].get(name) for r in rs]) for name in OFFSET_EVENTS
        }
        out[key] = {
            "n_runs": len(rs),
            "ttfa_ms": ttfa_stats,
            "stages_ms": stage_stats,
            "offsets_vs_user_end_ms": offset_stats,
            "retrieval_duration_ms": _stats([r["retrieval_ms"] for r in rs]),
            "retrieval_hidden_ms": _stats([r["retrieval_hidden_ms"] for r in rs]),
            "ttft_ms": _stats([r["ttft_ms"] for r in rs]),
            "generation_ms": _stats([r["generation_ms"] for r in rs]),
            "n_tokens": _stats([r["n_tokens"] for r in rs]),
        }
    return out


def _group_sort_key(key: str):
    policy = key.split("/")[0]
    rank = POLICIES.index(policy) if policy in POLICIES else len(POLICIES)
    return (rank, key)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt(v, nd=0):
    if v is None:
        return "–"
    return f"{v:.{nd}f}"


def markdown_table(summary: dict) -> str:
    lines = []
    lines.append(
        "| group | n | TTFA p50 | TTFA p95 | TTFA mean | "
        + " | ".join(f"{s} mean (share)" for s in STAGES)
        + " |"
    )
    lines.append("|" + "---|" * (5 + len(STAGES)))
    for key, g in summary.items():
        t = g["ttfa_ms"]
        cells = [
            key,
            str(g["n_runs"]),
            _fmt(t["p50"]),
            _fmt(t["p95"]),
            _fmt(t["mean"]),
        ]
        for s in STAGES:
            st = g["stages_ms"][s]
            share = st["share_of_mean_ttfa"]
            cells.append(f"{_fmt(st['mean'])} ({_fmt(100 * share if share is not None else None, 1)}%)")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Stage medians / p95 (ms):")
    lines.append("")
    lines.append("| group | " + " | ".join(STAGES) + " |")
    lines.append("|" + "---|" * (1 + len(STAGES)))
    for key, g in summary.items():
        cells = [key] + [
            f"{_fmt(g['stages_ms'][s]['p50'])} / {_fmt(g['stages_ms'][s]['p95'])}" for s in STAGES
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Event offsets vs user_end, median / p95 (ms; negative = during speech):")
    lines.append("")
    lines.append("| group | " + " | ".join(OFFSET_EVENTS) + " |")
    lines.append("|" + "---|" * (1 + len(OFFSET_EVENTS)))
    for key, g in summary.items():
        cells = [key] + [
            f"{_fmt(g['offsets_vs_user_end_ms'][e]['p50'])} / {_fmt(g['offsets_vs_user_end_ms'][e]['p95'])}"
            for e in OFFSET_EVENTS
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Retrieval duration (launch -> result) and hidden part, median / p95 (ms):")
    lines.append("")
    lines.append("| group | retrieval duration | hidden in speech | TTFT | generation | n_tokens |")
    lines.append("|---|---|---|---|---|---|")
    for key, g in summary.items():
        r = g["retrieval_duration_ms"]
        h = g["retrieval_hidden_ms"]
        lines.append(
            f"| {key} | {_fmt(r['p50'])} / {_fmt(r['p95'])} | {_fmt(h['p50'])} / {_fmt(h['p95'])} | "
            f"{_fmt(g['ttft_ms']['p50'])} / {_fmt(g['ttft_ms']['p95'])} | "
            f"{_fmt(g['generation_ms']['p50'])} / {_fmt(g['generation_ms']['p95'])} | "
            f"{_fmt(g['n_tokens']['p50'])} / {_fmt(g['n_tokens']['p95'])} |"
        )
    return "\n".join(lines) + "\n"


_POLICY_DISPLAY = {
    "sentence_end": "sentence-end",
    "immediate": "immediate",
    "fixed_early_proxy": "fixed early proxy",
    "forecast": "forecast",
}


def latex_table(summary: dict, caption: str, label: str) -> str:
    short = {
        "dispatch_wait": "Disp.",
        "retrieval_exposed": "Retr.",
        "pre_llm_overhead": "Hand-off",
        "llm_prefill": "TTFT",
        "llm_to_tts_input": "Gen.",
        "tts_first_audio": "TTS",
    }
    rows = []
    for key, g in summary.items():
        name = "/".join(_POLICY_DISPLAY.get(p, p) for p in key.split("/"))
        t = g["ttfa_ms"]
        cells = [name, _fmt(t["p50"]), _fmt(t["p95"])]
        for s in STAGES:
            st = g["stages_ms"][s]
            cells.append(
                f"{_fmt(st['mean'])} ({_fmt(100 * st['share_of_mean_ttfa'], 0)}\\%)"
            )
        rows.append(" & ".join(cells) + r" \\")
    header = " & ".join(["Policy", "TTFA p50", "TTFA p95"] + [short[s] for s in STAGES]) + r" \\"
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{l" + "r" * (2 + len(STAGES)) + "}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


# Categorical slots 1-6 of the reference palette, fixed order.
_STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot_stacked(summary: dict, out_paths: List[Path], xlabel: str = "Mean time after user end (ms)") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(summary.keys())
    names = ["/".join(_POLICY_DISPLAY.get(p, p) for p in k.split("/")) for k in keys]
    n = len(keys)

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(9.0, 1.0 + 0.75 * n + 1.2))
    ys = list(range(n))[::-1]
    lefts = [0.0] * n
    for si, s in enumerate(STAGES):
        widths = [summary[k]["stages_ms"][s]["mean"] or 0.0 for k in keys]
        ax.barh(
            ys,
            widths,
            left=lefts,
            height=0.6,
            color=_STAGE_COLORS[si],
            edgecolor="white",
            linewidth=1.5,
            label=STAGE_LABELS[s],
        )
        lefts = [l + w for l, w in zip(lefts, widths)]
    for y, k in zip(ys, keys):
        t = summary[k]["ttfa_ms"]
        ax.plot([t["p50"]], [y], marker="|", markersize=22, color="#0b0b0b", mew=2)
        ax.text(
            max(lefts) * 1.01 if False else (summary[k]["ttfa_ms"]["mean"] + 60),
            y,
            f"mean {t['mean']:.0f}  ·  p50 {t['p50']:.0f}  ·  p95 {t['p95']:.0f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#52514e",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_xlabel(xlabel)
    xmax = max(summary[k]["ttfa_ms"]["mean"] for k in keys)
    ax.set_xlim(0, xmax * 1.55)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker="|", color="#0b0b0b", linestyle="", markersize=14, mew=2))
    labels.append("median TTFA")
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32 if n <= 3 else -0.18),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", nargs="+", default=["logs/conv_*.jsonl"], help="glob(s) of event logs")
    ap.add_argument("--label", default="e2", help="name used in output file names")
    ap.add_argument("--group-by", default="policy", choices=["policy", "policy_path", "policy_path_tts"])
    ap.add_argument("--out-dir", default="analysis/latency_breakdown")
    ap.add_argument("--paper-dir", default=None, help="also write figures/ and tables/ under this dir")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)

    runs = load_runs(args.logs)
    if not runs:
        print("no runs with a TTFA found for", args.logs)
        return 1
    summary = summarize(runs, args.group_by)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": args.label,
        "log_globs": args.logs,
        "group_by": args.group_by,
        "n_runs": len(runs),
        "stages": list(STAGES),
        "stage_labels": STAGE_LABELS,
        "clock_note": "t_ms from time.monotonic_ns(); ~15.6 ms effective resolution on the Windows run host",
        "summary": summary,
    }
    (out_dir / f"latency_breakdown_{args.label}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    md = markdown_table(summary)
    (out_dir / f"latency_breakdown_{args.label}.md").write_text(md, encoding="utf-8")
    print(md)

    fig_paths = [out_dir / f"latency_breakdown_{args.label}.png"]
    if args.paper_dir:
        paper = Path(args.paper_dir)
        fig_paths.append(paper / "figures" / f"latency_breakdown_{args.label}.pdf")
        tex = latex_table(
            summary,
            caption=(
                "TTFA decomposition along the critical path after the user stops speaking "
                f"({args.label.upper()} logs, n={len(runs)}). Stage columns are mean ms and share of mean TTFA; "
                "stages sum to TTFA. Disp.: ASR final, controller and dispatch wait; Retr.: retrieval time "
                "exposed after user end; TTFT: LLM time to first token; Gen.: generation until TTS input "
                "(the full answer in batch TTS); TTS: synthesis to first audio."
            ),
            label=f"tab:latency_breakdown_{args.label}",
        )
        (paper / "tables").mkdir(parents=True, exist_ok=True)
        (paper / "tables" / f"latency_breakdown_{args.label}.tex").write_text(tex, encoding="utf-8")
    if not args.no_fig:
        plot_stacked(summary, fig_paths)
        print("figures:", ", ".join(str(p) for p in fig_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
