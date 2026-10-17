"""
make_fig_framework.py: Fig. "framework" (replaces system_flow.jpeg).

Draws the streaming pipeline and the three decision structures compared in the
paper (joint, decomposed, speculative decomposed) as a vector PDF for a
full-width IEEE figure (figure*, 7.16 in). Text is real text, min 7 pt.

    python paper_v2/tools/make_fig_framework.py
writes paper_v2/figures/fig_framework.pdf and fig_framework.png (preview).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "pdf.fonttype": 42,
})

OUT = Path(__file__).resolve().parents[1] / "figures"
W, H = 7.16, 3.35            # inches; drawing units are 1/10 inch
INK = "#1a1a1a"
GREY = "#f2f2f2"
JOINT = "#dbe8f6"            # light blue: the method
JOINT_EDGE = "#1f4e8c"
DECOMP = "#fbe9d7"           # light orange: decomposed rivals
SPEC = "#e3f1e0"             # light green: speculative rivals

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W * 10)
ax.set_ylim(0, H * 10)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=INK, lw=0.8, ls="-", size=7.5,
        weight="normal", align="center", pad=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    tx = x + w / 2 if align == "center" else x + 0.8
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size,
            weight=weight, linespacing=1.25)


def arrow(x0, y0, x1, y1, lw=0.9, color=INK, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=7, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ row 1 --
Y1, H1 = 23.6, 6.4           # pipeline row
# user speech strip with ASR partial ticks
ax.add_patch(Rectangle((0.6, Y1 + 2.6), 14.2, 2.4, fc=GREY, ec="none"))
ax.text(7.7, Y1 + 3.8, "user speech (streaming)", ha="center", va="center", fontsize=7)
for i, x in enumerate([2.4, 5.2, 8.0, 10.8]):
    ax.plot([x, x], [Y1 + 1.9, Y1 + 2.6], color=INK, lw=0.7)
    ax.text(x, Y1 + 1.2, f"$x_{{{i + 1}}}$", ha="center", va="center", fontsize=7.5)
ax.plot([14.8, 14.8], [Y1 + 1.6, Y1 + 5.6], color=INK, lw=1.4)
ax.text(15.7, Y1 + 1.2, r"$x_K$", ha="center", va="center", fontsize=7.5)
ax.text(14.8, Y1 + 6.35, "VAD end of turn", ha="center", va="center", fontsize=6.8)
ax.text(1.0, Y1 + 0.2, "ASR partial transcripts", ha="left", va="center", fontsize=6.8,
        style="italic", color="#444444")

box(17.0, Y1 + 0.6, 8.8, 5.2,
    "Streaming ASR\n+ VAD\n(faster-whisper,\nSilero)", fc=GREY)
box(28.0, Y1 + 3.3, 11.6, 2.9,
    r"belief  $b_k=\hat P(\mathrm{need}\mid x_k)$", fc=GREY)
box(28.0, Y1 - 0.1, 11.6, 2.9,
    r"remaining speech  $\hat R_k$", fc=GREY)
box(42.2, Y1 + 0.2, 8.6, 5.8, "Controller\n(per partial $k$)\nexpanded below",
    fc=JOINT, ec=JOINT_EDGE, lw=1.3, weight="bold")
box(53.0, Y1 + 0.6, 6.6, 5.2, "Retriever\nlatency $L$", fc=GREY)
box(61.5, Y1 + 0.6, 4.8, 5.2, "LLM\n(Mistral\n7B)", fc=GREY)
box(68.0, Y1 + 0.6, 3.2, 5.2, "TTS", fc=GREY)

arrow(14.8, Y1 + 3.2, 17.0, Y1 + 3.2)
arrow(25.8, Y1 + 4.7, 28.0, Y1 + 4.7)
arrow(25.8, Y1 + 1.4, 28.0, Y1 + 1.4)
arrow(39.6, Y1 + 4.7, 42.2, Y1 + 4.0)
arrow(39.6, Y1 + 1.4, 42.2, Y1 + 2.2)
arrow(50.8, Y1 + 3.2, 53.0, Y1 + 3.2)
arrow(59.6, Y1 + 3.2, 61.5, Y1 + 3.2)
arrow(66.3, Y1 + 3.2, 68.0, Y1 + 3.2)
# skip path straight to the LLM
ax.plot([46.5, 46.5, 63.9], [Y1 + 6.0, Y1 + 7.6, Y1 + 7.6], color=INK, lw=0.8, ls=(0, (3, 2)))
arrow(63.9, Y1 + 7.6, 63.9, Y1 + 5.8, ls=(0, (3, 2)))
ax.text(55.0, Y1 + 8.15, "skip: answer closed-book (no retrieval)", ha="center", va="center", fontsize=6.5,
        style="italic")
# TTFA bracket
yb = Y1 - 1.6
ax.annotate("", xy=(14.8, yb), xytext=(71.2, yb),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK, shrinkA=0, shrinkB=0))
ax.plot([71.2, 71.2], [yb, Y1 + 0.6], color=INK, lw=0.6, ls=":")
ax.plot([14.8, 14.8], [yb, Y1 + 1.6], color=INK, lw=0.6, ls=":")
ax.text(43.0, yb - 0.85, r"TTFA = $t_{\mathrm{first\ audio}}-t_{\mathrm{user\ end}}$"
        "  (retrieval that starts while the user speaks is partly hidden)",
        ha="center", va="center", fontsize=6.8)

# zoom lines from controller to the panel

# ------------------------------------------------------------------ row 2 --
# shared objective bar
ax.add_patch(FancyBboxPatch((1.0, 16.4), 69.6, 3.2, boxstyle="round,pad=0,rounding_size=0.6",
                            fc="white", ec=INK, lw=0.8))
ax.text(35.8, 18.0,
        r"Shared objective:  $U_t(b)=b\,g^{+}(1-\rho_t)\;-\;(1-b)\,g^{-}\;-\;\lambda_l\,e_t\;-\;\lambda_c\,(1+w_t)$,"
        r"   $U(\mathrm{skip})=0$",
        ha="center", va="center", fontsize=7.8)

PY, PH = 1.0, 14.4          # panel row
cols = [(1.0, 22.6), (24.5, 22.6), (48.0, 22.6)]

# (a) joint
x, w = cols[0]
ax.add_patch(FancyBboxPatch((x, PY), w, PH, boxstyle="round,pad=0,rounding_size=0.8",
                            fc=JOINT, ec=JOINT_EDGE, lw=1.3))
ax.text(x + w / 2, PY + PH - 1.5, "(a) Joint (Ours)", ha="center", va="center",
        fontsize=8, weight="bold")
box(x + 1.2, PY + 4.6, w - 2.4, 6.4,
    "at each partial $k$, with $b_k$ and $\\hat R_k$:\n"
    "dispatch now if $U_k>0$ and $U_k\\geq\\mathrm{Wait}_k$,\n"
    "else wait for the next partial;\n"
    "at end of turn: retrieve iff $U_K>0$, else skip", size=7)
ax.text(x + w / 2, PY + 2.4, "whether and when are chosen together:\n"
        "the dispatch time depends on the belief",
        ha="center", va="center", fontsize=6.6, style="italic")

# (b) decomposed
x, w = cols[1]
ax.add_patch(FancyBboxPatch((x, PY), w, PH, boxstyle="round,pad=0,rounding_size=0.8",
                            fc=DECOMP, ec=INK, lw=0.8, ls=(0, (4, 2))))
ax.text(x + w / 2, PY + PH - 1.5, "(b) Decomposed (B9, B9+)", ha="center", va="center",
        fontsize=8, weight="bold")
box(x + 1.0, PY + 4.6, 9.2, 6.4, "timing rule\nfixed trigger $\\tau$,\nset before the turn\n(no belief)", size=7)
box(x + 12.4, PY + 4.6, 9.2, 6.4, "retrieval rule\nat $\\tau$: retrieve\niff $U(b)>0$", size=7)
ax.plot([x + 11.3, x + 11.3], [PY + 4.0, PY + 11.6], color=INK, lw=1.4, ls=(0, (2, 1.5)))
ax.text(x + w / 2, PY + 2.4, "no shared state; B9 prices the full latency $L$,\n"
        "B9+ the expected exposure of $\\tau$",
        ha="center", va="center", fontsize=6.6, style="italic")

# (c) speculative decomposed
x, w = cols[2]
ax.add_patch(FancyBboxPatch((x, PY), w, PH, boxstyle="round,pad=0,rounding_size=0.8",
                            fc=SPEC, ec=INK, lw=0.8, ls=(0, (1, 1.2))))
ax.text(x + w / 2, PY + PH - 1.5, "(c) Speculative (B10, B11)", ha="center", va="center",
        fontsize=8, weight="bold")
box(x + 1.0, PY + 4.6, 9.2, 6.4, "prefetch at first\nquery-ready partial\n(B10 always;\nB11 if belief gate)", size=7)
box(x + 12.4, PY + 4.6, 9.2, 6.4, "end of turn:\nuse result\niff $U(b_K)>0$", size=7)
arrow(x + 10.2, PY + 7.8, x + 12.4, PY + 7.8)
ax.text(x + w / 2, PY + 2.4, "an unused prefetch costs compute only;\n"
        "(a) also runs under this architecture",
        ha="center", va="center", fontsize=6.6, style="italic")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_framework.pdf")
fig.savefig(OUT / "fig_framework.png", dpi=300)
print("wrote", OUT / "fig_framework.pdf")
