"""
make_fig_framework.py: Fig. 1 (framework).

Streaming pipeline plus the three decision structures compared (joint,
decomposed, speculative decomposed). Vector PDF for a full-width IEEE figure
(figure*, 7.16 in). The utility equation and the exact decision rules live in
the text (Eq. 2, Algorithm 1), so the figure only shows structure.

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
    "font.size": 8,
    "pdf.fonttype": 42,
})

OUT = Path(__file__).resolve().parents[1] / "figures"
W, H = 7.16, 2.33            # inches; drawing units are 1/10 inch
INK = "#1a1a1a"
GREY = "#f2f2f2"
JOINT = "#dbe8f6"
JOINT_EDGE = "#1f4e8c"
DECOMP = "#fbe9d7"
SPEC = "#e3f1e0"

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W * 10)
ax.set_ylim(0, H * 10)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=INK, lw=0.8, ls="-", size=8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            weight=weight, linespacing=1.2)


def arrow(x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=INK, ls=ls, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ pipeline row
Y1 = 14.6
ax.add_patch(Rectangle((0.6, Y1 + 2.4), 13.6, 2.3, fc=GREY, ec="none"))
ax.text(7.4, Y1 + 3.55, "user speech", ha="center", va="center", fontsize=8)
for i, x in enumerate([2.4, 5.2, 8.0, 10.8]):
    ax.plot([x, x], [Y1 + 1.8, Y1 + 2.4], color=INK, lw=0.7)
    ax.text(x, Y1 + 1.1, f"$x_{{{i + 1}}}$", ha="center", va="center", fontsize=8)
ax.plot([14.2, 14.2], [Y1 + 1.4, Y1 + 5.3], color=INK, lw=1.4)
ax.text(15.1, Y1 + 1.1, r"$x_K$", ha="center", va="center", fontsize=8)
ax.text(14.2, Y1 + 6.0, "VAD end of turn", ha="center", va="center", fontsize=7.5)

box(16.6, Y1 + 0.7, 8.4, 4.6, "Streaming ASR\n+ VAD", fc=GREY)
box(27.2, Y1 + 3.1, 11.4, 2.6, r"belief $b_k$", fc=GREY)
box(27.2, Y1 + 0.1, 11.4, 2.6, r"remaining speech $\hat R_k$", fc=GREY)
box(41.0, Y1 + 0.4, 8.6, 5.0, "Controller\n(see below)", fc=JOINT, ec=JOINT_EDGE, lw=1.3,
    weight="bold")
box(52.0, Y1 + 0.7, 6.4, 4.6, "Retriever\n$L$", fc=GREY)
box(60.4, Y1 + 0.7, 5.6, 4.6, "LLM", fc=GREY)
box(67.8, Y1 + 0.7, 3.6, 4.6, "TTS", fc=GREY)

arrow(14.2, Y1 + 3.0, 16.6, Y1 + 3.0)
arrow(25.0, Y1 + 4.4, 27.2, Y1 + 4.4)
arrow(25.0, Y1 + 1.4, 27.2, Y1 + 1.4)
arrow(38.6, Y1 + 4.4, 41.0, Y1 + 3.7)
arrow(38.6, Y1 + 1.4, 41.0, Y1 + 2.1)
arrow(49.6, Y1 + 3.0, 52.0, Y1 + 3.0)
arrow(58.4, Y1 + 3.0, 60.4, Y1 + 3.0)
arrow(66.0, Y1 + 3.0, 67.8, Y1 + 3.0)
ax.plot([45.3, 45.3, 63.2], [Y1 + 5.4, Y1 + 6.9, Y1 + 6.9], color=INK, lw=0.8, ls=(0, (3, 2)))
arrow(63.2, Y1 + 6.9, 63.2, Y1 + 5.3, ls=(0, (3, 2)))
ax.text(54.3, Y1 + 7.6, "skip: closed-book answer", ha="center", va="center", fontsize=7.5,
        style="italic")

yb = Y1 - 1.3
ax.annotate("", xy=(14.2, yb), xytext=(71.4, yb),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK, shrinkA=0, shrinkB=0))
ax.plot([71.4, 71.4], [yb, Y1 + 0.7], color=INK, lw=0.6, ls=":")
ax.plot([14.2, 14.2], [yb, Y1 + 1.4], color=INK, lw=0.6, ls=":")
ax.text(42.8, yb - 0.9, "time to first audio (TTFA)", ha="center", va="center", fontsize=7.5)

# ------------------------------------------------------------------ decision structures
PY, PH = 0.6, 10.2
cols = [(1.0, 22.6), (24.5, 22.6), (48.0, 22.6)]


def panel(x, w, title, fc, ec, lw, ls):
    ax.add_patch(FancyBboxPatch((x, PY), w, PH, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, PY + PH - 1.3, title, ha="center", va="center", fontsize=8.5,
            weight="bold")


x, w = cols[0]
panel(x, w, "(a) Joint (Ours, Ours-S)", JOINT, JOINT_EDGE, 1.3, "-")
box(x + 1.2, PY + 3.0, w - 2.4, 4.4, "at each partial: dispatch now or wait,\n"
    r"using $b_k$ and $\hat R_k$ together")
ax.text(x + w / 2, PY + 1.6, "dispatch time depends on the belief", ha="center", va="center",
        fontsize=7.5, style="italic")

x, w = cols[1]
panel(x, w, "(b) Decomposed (B9, B9+)", DECOMP, INK, 0.8, (0, (4, 2)))
box(x + 1.0, PY + 3.0, 9.2, 4.4, "timing rule\n(no belief)")
box(x + 12.4, PY + 3.0, 9.2, 4.4, "retrieve iff\n" r"$U(b)>0$ at trigger")
ax.plot([x + 11.3, x + 11.3], [PY + 2.6, PY + 7.8], color=INK, lw=1.4, ls=(0, (2, 1.5)))
ax.text(x + w / 2, PY + 1.6, "no shared state", ha="center", va="center", fontsize=7.5,
        style="italic")

x, w = cols[2]
panel(x, w, "(c) Speculative (B10, B11)", SPEC, INK, 0.8, (0, (1, 1.2)))
box(x + 1.0, PY + 3.0, 9.2, 4.4, "prefetch early\n(B11: belief gate)")
box(x + 12.4, PY + 3.0, 9.2, 4.4, "use at end of turn\n" r"iff $U(b_K)>0$")
arrow(x + 10.2, PY + 5.2, x + 12.4, PY + 5.2)
ax.text(x + w / 2, PY + 1.6, "an unused prefetch costs compute only", ha="center",
        va="center", fontsize=7.5, style="italic")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_framework.pdf")
fig.savefig(OUT / "fig_framework.png", dpi=300)
print("wrote", OUT / "fig_framework.pdf")
