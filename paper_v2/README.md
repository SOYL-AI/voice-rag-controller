# paper_v2: INDICON 2026 revision (Paper ID 4422)

This folder is the skeleton of the revised paper. Each section contains a
paragraph-level outline: its purpose, the claims it must make with their
evidence, and the only numbers it may quote, each with its source path. The
authors write the prose. That is required by the conference's AI-generated
content rule (score below 10%), so the outline must not be pasted in as text.
Equations, algorithms, tables, captions and definitions are complete.

## Port into the official template (required)

The submission must use the official INDICON 2026 IEEE template. We only have
its Overleaf share link:

<https://www.overleaf.com/project/6accd6b3312f94bd4f391706>

`main.tex` uses the stock `IEEEtran` conference class so that the structure
can be checked here. To port it:

1. Open the template project. Copy `macros.tex`, `references.bib`,
   `sections/`, `tables/` and `figures/` into it.
2. Move this `main.tex`'s preamble additions into the template's preamble:
   `amsthm`, `algorithm` + `algpseudocode` (do not also load `algorithmic`),
   `booktabs`, `multirow`, `tikz` with its libraries, and `xcolor` with the
   `table` option. Then copy the `\input` lines and the title/author block.
3. Keep the template's own page size, fonts, margins and copyright notice.
   Change nothing in the class options it ships with.
4. Compile. Fix anything `tools/check_tex.py` cannot see: overfull boxes, the
   Fig. 1 layout and table widths.
5. Set `\finaltrue` in `macros.tex`. The build then fails while any `\todo`,
   `\TBD` or `\PEND` is left.

There is no TeX installation on the machine these files were written on. The
LaTeX was written conservatively and checked with
`PYTHONUTF8=1 python paper_v2/tools/check_tex.py`, which covers braces,
environments, labels and references, citation keys, figure files and the
15-entry limit. **It has never been compiled.** Compile it in Overleaf before
anything else.

## Layout

| path | content |
|---|---|
| `main.tex` | class, packages, title options, author block, section order |
| `macros.tex` | `\todo{}` (author prose), `\TBD{artifact}` (v2 number not yet produced), `\PEND{}` (open author decision), and notation macros |
| `sections/00_abstract.tex` to `09_conclusion.tex` | paragraph outlines (`P-<section>.<n>` blocks) |
| `sections/theory_formal.tex` | the theory slot, copied from `docs/research_v2/theory_latex.tex` at `8bbe257` |
| `tables/` | controllers, predictions, v1 E3, v1 E2, v2 primary |
| `figures/fig1_architecture.tex` | Fig. 1 in TikZ, with a full drawing spec in its header |
| `figures/*.pdf` | v1 figures, regenerated from the real data by `analysis/paper_figures_v1.py` |
| `RESPONSE_TO_REVIEWERS.md` | point-by-point response (goes to the reviewers, not into the paper) |
| `COMPLIANCE_CHECKLIST.md` | conference rules with their status |
| `tools/check_tex.py` | static checker for use without a TeX installation |

## Regenerating every v1 number and figure

```
PYTHONUTF8=1 python -m analysis.paper_figures_v1     # figures + analysis/paper_v1_numbers.json
PYTHONUTF8=1 python -m analysis.v1_theory_check      # docs/research_v2/v1_theory_check.json
PYTHONUTF8=1 python -m harness.e2_bootstrap logs/    # E2 criterion and paired test
```

These run on the system Python 3.12 (matplotlib, numpy and yaml; no pyarrow).
`paper_figures_v1.py` refuses to read `e3_sample_result/`. That folder holds
the synthetic sample that the submitted Sec. V-C and Figs. 2–3 used by
mistake. Quote v1 numbers in the text from `v1_theory_check.json` where it has
them. The figures use their own bootstrap random stream, so interval end
points can differ from that file in the third decimal place.

## Two corrections to the planning notes, both applied in the outline

1. "Ours(λ_i) = B9(λ_{i−1})" holds **by construction**. `simulator/e3_grid.py`
   tiles divergence bands edge to edge, so the grid is geometric about
   λ_deg = 0.021277 with ratio L/(L−o*) = 1.3884. Raw consecutive ratios run
   from 1.008 to 1.382. Fig. `fig_e3_price_shift.pdf` is therefore presented
   as an implementation check. The evidence for Prop. 1 is:
   - 10,000/10,000 off-grid decisions agree under the exact price map;
   - B9+ agrees with Ours on 11,300/11,300 decisions;
   - the band is non-empty at 11 of 26 prices and covers 41 of 1,300
     decisions (3.15%);
   - every observed disagreement lies in the band.
2. The measured TTFA cost of retrieval (paired retrieve − skip) is 1.57 s at
   sentence_end and 1.20–1.30 s at the early timings. The v1 model used
   0.672/0.484 s.

Two smaller corrections:
- "Accuracy 0.58, compute 0.82, p95 ≈ 5.5 s at λ = 0.022" describes
  **seed 11 only**. Pooled over seeds it is 0.616 [0.488, 0.744] with p95
  5.15 s, and the per-seed range is 0.58–0.66.
- The final d2_support leak-test AUC is **0.576 [0.541, 0.617]** (branch
  `v2/d2-support-a`, commit `0e99cf0`), not ≈ 0.63; 0.625 was an earlier
  part-A round.

## Reference list (15 entries) and what was dropped

Kept from the submitted list (9 of 12):
- Lewis (RAG)
- FLARE
- Self-RAG
- Adaptive-RAG
- DRAGIN
- TurnGPT
- VAP
- Skantze's review. Its venue was wrong in the old bib: it is Computer
  Speech & Language 67, 101178, not AI Magazine.
- Bohus & Horvitz (ICMI 2011)

Dropped:
- **Karpukhin et al. 2020 (DPR).** The system does not use DPR, and
  `lewis2020rag` covers dense retrieval as a component.
- **Cooper 1973.** It is about choosing an evaluation measure. The
  decision-theoretic foundation the paper actually uses is value of
  information.
- **Rao & Daumé III 2018.** It applies EVPI to clarification questions.
  Under the cap, the primary VoI source (Howard 1966) is preferred.

Added:
- Howard 1966 (R1 minor)
- Mallen et al. 2023, PopQA. It is a d1_open source and shows that retrieval
  can hurt (g⁻ > 0).
- Yang et al. 2018 (HotpotQA)
- Radford et al. 2023, Whisper. v2 uses faster-whisper for the partials.
- Guo et al. 2017 (ECE, R1.5)
- Jiang et al. 2023 (Mistral 7B, the only generator)

Verified alternate: Zheng et al. 2023, LLM-as-a-judge. To include it, drop
the Mistral entry and cite the model card in a footnote. Every entry's
verification URL is in a comment in `references.bib`.

## Open items for the authors (also flagged in the files)

- Write every `\todo` paragraph (40 markers). Do not paste the outline
  comments in as prose.
- Choose the title. A title change may need the TPC chair's consent.
- Complete the author list, order and e-mails. The corresponding-author
  marker is a placeholder on Ryan Gomez; confirm it, and keep exactly one.
- Fill the pre-registration link (public repo commit URLs or OSF) for
  `b9ba0c9`, `cd5d145` and `6addd89`.
- Update `docs/research_v2/PREREG_V2.md` to the two-architecture design
  before freezing:
  - two primaries and the alpha allocation between them;
  - the speculative non-superiority margin λ_c(1+w);
  - B10 named and implemented. There is no B10 controller on
    `research/v2-base` yet.
- Decide where the proofs go (6-page budget). The full proofs in
  `theory_formal.tex` will not fit.
- Choose a dead-air threshold for the v2 dead-air rate.
- Waste-rate sensitivity for v1 needs an extension of
  `analysis/v1_theory_check.py`, which has not been written yet.
- Write `analysis/paper_figures_v2.py` for the two v2 figure placeholders
  (prefix AUC, two-sided test).
- Fill every `\TBD` from the named artifact after the frozen test-split
  runs. Never estimate one.
