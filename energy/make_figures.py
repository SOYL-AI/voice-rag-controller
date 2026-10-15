"""Figures and the numeric tables quoted in energy/REPORT.md.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.make_figures

Writes energy/figures/*.png and energy/results/report_tables.json. Figures
carry no titles (captions live in the report), use >= 11 pt text, and use
the first slots of a CVD-validated categorical palette in fixed order.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from energy import model as em  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

STAGE_LABELS = [
    ("vad_utterance_cpu", "VAD, 4.7 s audio (CPU)"),
    ("asr_stream_cpu_int8", "ASR streaming, 3.9 s speech (CPU int8)"),
    ("asr_final_cpu_int8", "ASR final pass only (CPU int8)"),
    ("asr_stream_cuda_fp16", "ASR streaming (GPU fp16)"),
    ("embed_query_cpu", "Query embedding (CPU)"),
    ("embed_query_cuda", "Query embedding (GPU)"),
    ("faiss_search_60", "FAISS top-50, 60 vectors"),
    ("faiss_search_10k", "FAISS top-50, 10k vectors"),
    ("faiss_search_100k", "FAISS top-50, 100k vectors"),
    ("rerank50_minilm_cuda", "Rerank 50, MiniLM-L6 (GPU)"),
    ("rerank50_minilm_cpu", "Rerank 50, MiniLM-L6 (CPU)"),
    ("rerank50_bgebase_cuda", "Rerank 50, bge-reranker-base (GPU)"),
    ("gpu_wake_tail", "GPU wake + clock-down tail"),
    ("tts_piper_inproc_typical", "TTS typical answer (in-process)"),
    ("tts_piper_subprocess_typical", "TTS typical answer (as implemented)"),
]


def stage_central(s):
    if "cpu_attr_j" not in s:  # gpu_wake_tail
        g = s["gpu_net_j"]
        return g["mean"], g["lo"], g["hi"]
    g = s["gpu_net_j"]
    a = s["cpu_attr_j"]
    gm = max(g["mean"] or 0, 0)
    c = (a["mean"] or 0) + gm
    lo = (a["lo"] or 0) + max(g["lo"] or 0, 0)
    hi = (a["hi"] or 0) + max(g["hi"] or 0, 0)
    return c, lo, hi


def fig_stages(stages):
    S = stages["stages"]
    rows = [(lab, *stage_central(S[k])) for k, lab in STAGE_LABELS if k in S]
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))
    c = np.array([max(r[1], 1e-6) for r in rows])
    lo = np.array([max(r[2], 1e-6) for r in rows])
    hi = np.array([max(r[3], 1e-6) for r in rows])
    ax.barh(y, c, color=C[0], height=0.6)
    ax.errorbar(c, y, xerr=[c - np.minimum(lo, c), np.maximum(hi, c) - c], fmt="none",
                ecolor=INK2, elinewidth=1.2, capsize=3)
    ax.set_xscale("log")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Energy per operation, J (net; log scale; 95% CI)")
    for yi, v, h in zip(y, c, hi):
        ax.text(max(v, h) * 1.4, yi, f"{v:.3g} J", va="center", fontsize=10, color=INK2)
    ax.set_xlim(right=c.max() * 30)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "stage_energy.png")
    plt.close(fig)


GROUPS = [
    ("ASR + VAD", ("asr", "vad")),
    ("Retrieval compute", ("retrieval_embed", "retrieval_search", "retrieval_rerank",
                           "retrieval_network")),
    ("LLM prefill", ("llm_prefill",)),
    ("LLM decode", ("llm_decode",)),
    ("TTS", ("tts",)),
]


def grouped(e):
    comp = e["components"]
    return [sum(comp[k]["central"] for k in keys if k in comp) for _, keys in GROUPS]


def fig_turn(tb):
    asr = 3.94
    mean = lambda k: tb[k]["mean"]
    cpt = tb.get("chars_per_gen_token", 4.6)
    chars_s, chars_r = cpt * mean("gen_skip"), cpt * mean("gen_retrieve")
    cases = [("Skip", em.turn_energy_j("skip", None, mean("prompt_skip"), mean("gen_skip"), asr, chars_s))]
    for prof, lab in (("faiss", "Retrieve: faiss"), ("faiss_rerank", "Retrieve: faiss_rerank"),
                      ("remote_api", "Retrieve: remote_api")):
        cases.append((lab, em.turn_energy_j("retrieve", prof, mean("prompt_retrieve"),
                                            mean("gen_retrieve"), asr, chars_r)))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(cases))[::-1]
    left = np.zeros(len(cases))
    for gi, (gname, _) in enumerate(GROUPS):
        vals = np.array([grouped(e)[gi] for _, e in cases])
        ax.barh(y, vals, left=left, color=C[gi], height=0.6, label=gname,
                edgecolor="white", linewidth=1.5)
        left += vals
    for yi, (_, e) in zip(y, cases):
        t = e["total_j"]
        ax.text(t["central"] + 8, yi, f"{t['central']:.0f} J  [{t['low']:.0f}-{t['high']:.0f}]",
                va="center", fontsize=10, color=INK2)
    ax.set_yticks(y, [c[0] for c in cases])
    ax.set_xlabel("Energy per turn, J (central estimate; bracket = low-high band)")
    ax.set_xlim(0, left.max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.28), frameon=False)
    fig.savefig(FIG / "turn_breakdown.png")
    plt.close(fig)
    return {lab: e for lab, e in cases}


def fig_frontier(fr):
    pts = fr["points"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    order = [("ours", "Ours (joint)"), ("b9", "B9 decomposed"), ("b8", "B8 independent"),
             ("b4", "B4 FLARE-style"), ("b6", "B6 DRAGIN-style"), ("b0", "B0 never retrieve (= B4 here)")]
    for i, (b, lab) in enumerate(order):
        p = sorted([q for q in pts if q["baseline"] == b], key=lambda q: q["energy_j"]["central"])
        if not p:
            continue
        x = [q["energy_j"]["central"] for q in p]
        yv = [q["accuracy"] for q in p]
        if b == "ours":   # drawn wide underneath: Ours and B9 coincide point for point
            ax.plot(x, yv, "-", color=C[i], lw=6, alpha=0.45, solid_capstyle="round", label=lab)
        elif len(p) > 1:
            ax.plot(x, yv, "--", color=C[i], lw=1.8, marker="o", ms=4, label=lab)
        else:
            ax.plot(x, yv, "o", color=C[i], ms=9, label=lab,
                    markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(fr["mean_turn_j_always_retrieve"], color=INK2, lw=1, ls="--")
    ax.text(fr["mean_turn_j_always_retrieve"], ax.get_ylim()[1] - 0.005, "always retrieve ",
            color=INK2, fontsize=10, va="top", ha="right")
    ax.set_xlabel("Mean energy per turn, J (central; whole turn incl. local stages)")
    ax.set_ylabel("Accuracy (judged)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "accuracy_vs_energy.png")
    plt.close(fig)


def fig_llm_local(local):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (size, m) in enumerate(sorted(local["models"].items(), key=lambda kv: kv[1]["n_params"])):
        ops = m["ops"]
        Ls = sorted(int(k.split("_L")[1]) for k in ops if k.startswith("prefill_L") and "for_decode" not in k)
        E = [ops[f"prefill_L{L}"]["gpu_net_j"]["mean"] for L in Ls]
        err = [ops[f"prefill_L{L}"]["gpu_net_j"]["half"] or 0 for L in Ls]
        ax[0].errorbar(Ls, E, yerr=err, color=C[i], marker="o", ms=5, lw=2, capsize=3,
                       label=f"Qwen2.5-{size}")
        d = m["coefficients"]["decode"]
        ax[1].bar(np.arange(len(d)) + i * 0.27, [v["j_per_token"] for v in d.values()], width=0.25,
                  color=C[i], label=f"Qwen2.5-{size}")
    ax[0].set_xlabel("Prompt tokens (one prefill pass)")
    ax[0].set_ylabel("GPU net energy, J")
    ax[0].legend(frameon=False)
    ctxs = list(next(iter(local["models"].values()))["coefficients"]["decode"].keys())
    ax[1].set_xticks(np.arange(len(ctxs)) + 0.27, [f"context {c}" for c in ctxs])
    ax[1].set_ylabel("GPU net energy per decoded token, J")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "llm_local.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    stages = json.loads((RES / "stages.json").read_text())
    tb = json.loads((RES / "token_budget.json").read_text())
    fig_stages(stages)
    cases = fig_turn(tb)
    tables = {"turn_cases": {k: {"total_j": v["total_j"], "g_co2e": v["g_co2e"],
                                 "components": v["components"]} for k, v in cases.items()}}
    skip = cases["Skip"]
    tables["marginal_vs_skip"] = {}
    for k, v in cases.items():
        if k == "Skip":
            continue
        comp = {c: v["components"][c]["central"] - skip["components"].get(c, {"central": 0})["central"]
                for c in v["components"]}
        tables["marginal_vs_skip"][k] = {
            "total_j": {q: v["total_j"][q] - skip["total_j"][q] for q in ("central", "low", "high")},
            "g_co2e": {q: v["g_co2e"][q] - skip["g_co2e"][q] for q in ("central", "low", "high")},
            "components_central": comp,
        }
    if (RES / "frontier.json").exists():
        fig_frontier(json.loads((RES / "frontier.json").read_text()))
    if (RES / "llm_local.json").exists():
        fig_llm_local(json.loads((RES / "llm_local.json").read_text()))
    (RES / "report_tables.json").write_text(json.dumps(tables, indent=2))
    print(json.dumps(tables["marginal_vs_skip"], indent=1))


if __name__ == "__main__":
    main()
