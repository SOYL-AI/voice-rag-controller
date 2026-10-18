"""
paper_figures_v2.py -- the two v2 figures of paper_v2/sections/07_results_v2.tex.
Offline: reads committed result files only (no model, judge or cloud call).

    PYTHONUTF8=1 VOICE_RAG_MODEL=mistral7b_v03 .venv-e2/Scripts/python.exe -m analysis.paper_figures_v2

Inputs
  * docs/research_v2/v2_test_results_final.json   (boundary contrasts, all cells)
  * docs/research_v2/paper_numbers_v2.json         (ASR prefix curve + CIs;
                                                     python -m analysis.paper_numbers_v2)
  * llm/experiment1_v2/results/<ds>/analysis_test/analysis_test.json (text prefix curve + CIs)
Outputs (vector PDF, IEEE column width 3.5 in, serif, >= 7 pt, no titles,
greyscale-safe: d1_open black filled markers / solid; d2_support grey open
markers / dotted, labelled exploratory):
  * paper_v2/figures/fig_v2_prefix_auc.pdf
  * paper_v2/figures/fig_v2_boundary.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_final.json"
NUMS = _REPO_ROOT / "docs" / "research_v2" / "paper_numbers_v2.json"
E1 = _REPO_ROOT / "llm" / "experiment1_v2" / "results"
FIG = _REPO_ROOT / "paper_v2" / "figures"
ASR = "piper-en_US-hfc_male-medium__150wpm__clean"
COL_W = 3.5
M3 = 0.016
K1, K2 = "#000000", "#7f7f7f"   # d1_open, d2_support (exploratory)
PREVIEW = None  # optional directory for PNG previews (set via argv[1]); never written into the repo


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load(p: Path):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fig_prefix_auc():
    nums = load(NUMS)
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    fr = [0.3, 0.5, 0.7, 1.0]
    series = []
    for ds, col, mk, mfc, lab in (("d1_open", K1, "o", K1, "d1_open"),
                                  ("d2_support", K2, "s", "white", "d2_support (exploratory)")):
        asr = nums["e1_test_extra"][ds]["asr_prefix_curve_self_assessed_need"]
        a = load(E1 / ds / "analysis_test" / "analysis_test.json")["models"]["mistral7b_v03"]
        txt = {r["word_frac"]: r["empirical"] for r in a["prefix_curve_text"]["self_assessed_need"]}
        ya = [asr[str(f)]["auc"] for f in fr]
        lo = [asr[str(f)]["ci95"][0] for f in fr]
        hi = [asr[str(f)]["ci95"][1] for f in fr]
        yt = [txt[f]["auc"] for f in fr]
        dx = -0.008 if ds == "d1_open" else 0.008
        ls_asr = "-" if ds == "d1_open" else ":"
        ax.fill_between(fr, lo, hi, color=col, alpha=0.13 if ds == "d1_open" else 0.10, lw=0,
                        hatch=None if ds == "d1_open" else "////", edgecolor=col)
        ax.plot([f + dx for f in fr], ya, ls=ls_asr, color=col, marker=mk, ms=4, mfc=mfc, mec=col,
                label=f"{lab}, ASR partials")
        ax.plot([f + dx for f in fr], yt, ls="--", color=col, marker=mk, ms=3, mfc="white", mec=col, lw=0.8,
                dashes=(4, 2) if ds == "d1_open" else (1.5, 1.5), label=f"{lab}, text prefix")
        series.append((ds, ya, yt))
    ax.axhline(0.65, color="0.35", lw=0.6, ls=(0, (6, 3)))
    ax.text(0.305, 0.655, "C2 threshold 0.65", fontsize=7, color="0.25", va="bottom")
    ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
    ax.set_xlim(0.27, 1.03)
    ax.set_ylim(0.40, 0.80)
    ax.set_xticks(fr)
    ax.set_xticklabels(["0.3", "0.5", "0.7", "1.0"])
    ax.set_xlabel("Fraction of the question heard")
    ax.set_ylabel("AUC of belief vs. need label")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=2.6, borderaxespad=0.1)
    out = FIG / "fig_v2_prefix_auc.pdf"
    fig.savefig(out)
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / "fig_v2_prefix_auc.png", dpi=220)
    plt.close(fig)
    return out, series


def fig_boundary():
    fin = load(FINAL)
    cells = [c for c in fin["cells"] if c["condition"] == ASR]
    profiles = ["rewrite_faiss", "rewrite_faiss_rerank", "rewrite_remote"]
    plab = {"rewrite_faiss": "FAISS", "rewrite_faiss_rerank": "FAISS + rerank", "rewrite_remote": "remote API"}
    qs = [0.5, 0.65, 0.8, 0.9]
    xpos = {}
    x = 0.0
    for p in profiles:
        for q in qs:
            xpos[(p, q)] = x
            x += 1.0
        x += 0.8
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax, key, ylab in ((axes[0], "delta_commit", r"$\Delta_{\mathrm{C}}$ (commit)"),
                          (axes[1], "delta_speculative", r"$\Delta_{\mathrm{S}}$ (speculative)")):
        if key == "delta_speculative":
            ax.axhspan(-M3, M3, color="0.88", lw=0)
            gap = (xpos[("rewrite_faiss", 0.9)] + xpos[("rewrite_faiss_rerank", 0.5)]) / 2
            ax.text(gap, 0.0035, r"$\pm m_3$", fontsize=7, va="bottom", ha="center", color="0.2")
        ax.axhline(0, color="0.3", lw=0.6)
        for ds, col, mk, mfc, dx in (("d1_open", K1, "o", K1, -0.17), ("d2_support", K2, "s", "white", 0.17)):
            for c in cells:
                if c["dataset"] != ds:
                    continue
                xx = xpos[(c["profile"], c["belief_quantile"])] + dx
                e = c[key]
                prim = c["profile"] == "rewrite_faiss" and abs(c["belief_quantile"] - 0.65) < 1e-9
                eb = ax.errorbar(xx, e["estimate"],
                                 yerr=[[e["estimate"] - e["ci95"][0]], [e["ci95"][1] - e["estimate"]]],
                                 fmt=mk, ms=4.2 if prim else 3.2, mfc=mfc, mec=col, ecolor=col, elinewidth=0.8,
                                 capsize=1.5, capthick=0.6, mew=1.1 if prim else 0.8)
                if ds == "d2_support":
                    # dotted whisker distinguishes the exploratory dataset in greyscale
                    for bl in eb[2]:
                        bl.set_linestyle((0, (1.2, 1.2)))
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.068, 0.048)
        ax.set_yticks([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04])
    ax = axes[1]
    ticks = [xpos[k] for k in xpos]
    ax.set_xticks(ticks)
    ax.set_xticklabels([("%g" % k[1]) for k in xpos], fontsize=7)
    for p in profiles:
        mid = (xpos[(p, qs[0])] + xpos[(p, qs[-1])]) / 2
        ax.text(mid, -0.22, plab[p], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5)
    ax.text(-0.9, -0.085, "q:", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=7)
    # primary-cell marker
    axes[0].annotate("primary", xy=(xpos[("rewrite_faiss", 0.65)], 0.004), xytext=(xpos[("rewrite_faiss", 0.65)] + 0.2, 0.03),
                     fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=K1, mfc=K1, ls="-", ms=4, lw=0.8, label="d1_open (pre-registered)"),
         Line2D([], [], marker="s", color=K2, mfc="white", ls=(0, (1.2, 1.2)), ms=4, lw=0.8,
                label="d2_support (exploratory)")]
    axes[0].legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                   borderaxespad=0.1, handlelength=2.0, columnspacing=1.2)
    out = FIG / "fig_v2_boundary.pdf"
    fig.savefig(out)
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / "fig_v2_boundary.png", dpi=220)
    plt.close(fig)
    return out


def main() -> int:
    global PREVIEW
    PREVIEW = sys.argv[1] if len(sys.argv) > 1 else None
    style()
    p1, series = fig_prefix_auc()
    p2 = fig_boundary()
    print("wrote", p1)
    print("wrote", p2)
    for ds, ya, yt in series:
        print(ds, "ASR", [round(v, 3) for v in ya], "text", [round(v, 3) for v in yt])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
