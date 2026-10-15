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
| `sections/theory_formal.tex` | the theory slot, copied from `docs/research_v2/theory_latex.tex` at `1d8900c` (corrected boundary test, Props. 4′ and 4″) |
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

## Reference list (15 entries), revised after the novelty search (2026-10-16)

A verified literature search changed the positioning:
- **Background, not a contribution:** early and speculative retrieval on
  partial transcripts. Stream RAG (ICML 2026) already compares fixed-interval
  speculative calls with model-triggered calls, which correspond to our
  speculative and commit architectures. Speculative execution for voice
  agents is also published (Jung et al. 2026).
- **Claimed as new:** only the formal joint-vs-decomposed analysis with the
  boundary test, the persona finding on partial transcripts, and the
  matched-pair spoken benchmark. Each is hedged "to our knowledge".
- **Terminology:** never write "Speculative RAG", which names an unrelated
  drafter/verifier method. Write "speculative prefetch".

**Required (10):** Stream RAG (Arora et al., PMLR v306), Galbraith (arXiv
2606.20113), Jung et al. (arXiv 2610.07641), FLARE, DRAGIN, Kadavath et al.
(P(IK)), Moskvoretskii et al. (ACL 2025), Howard 1966, Mallen et al. (PopQA),
Yang et al. (HotpotQA).

**Chosen for the remaining 5 slots:**
- **tau-Voice** (Ray et al. 2026): the closest voice-agent benchmark; the
  benchmark novelty claim needs it as a contrast.
- **MoshiRAG** (Chien et al., ICML 2026): learns whether to retrieve in a
  full-duplex speech model, with retrieval at response onset. It is the
  closest speech system that makes the whether-decision.
- **Adaptive-RAG**: source of the v2 B5 baseline.
- **VAP**: audio-based turn-shift prediction, i.e. the remaining-speech
  input R_k.
- **Bohus & Horvitz 2011**: decision-theoretic timing in dialogue (R2's
  originality comment).

**Dropped this round** (verified records kept as comments at the end of
`references.bib`):
- **Lewis et al. 2020 (RAG).** Canonical, but no claim in the paper depends on
  it, and the cap is binding. It is the first candidate to swap back in,
  e.g. for Bohus & Horvitz.
- **Self-RAG.** FLARE, DRAGIN, Adaptive-RAG and Moskvoretskii already cover
  adaptive retrieval.
- **TurnGPT and the Skantze review.** VAP is kept as the audio-based
  turn-taking reference. R1 asked for TurnGPT/VAP to be cited directly
  instead of the review; the response explains why only VAP remains.
- **Whisper and Mistral 7B.** Identify the tools by footnote URLs (model card;
  faster-whisper project) instead of references.
- **Guo et al. 2017.** ECE is now defined by an equation (Eq. ece) in
  Sec. VI-A, so no citation is needed.
- **Earlier round:** Karpukhin (DPR), Cooper 1973 and Rao & Daumé III, for
  the reasons given in the first revision.

**Not included, verified by the coordinator:** SHANKS (ACL 2026), SKR (Wang
et al., Findings of EMNLP 2023), Ni et al. (Findings of ACL 2024),
RetrievalQA (Zhang et al., Findings of ACL 2024).

**Verification:**
- I fetched every entry added this round myself, and its source URL is in a
  comment in `references.bib`: PMLR pages for Stream RAG and MoshiRAG
  (published venues used), the ACL Anthology for Moskvoretskii, and arXiv for
  Galbraith, Jung, Kadavath and tau-Voice. The last four have no venue
  listed.
- Retained entries keep their verification URLs from the first round.

## Open items for the authors (also flagged in the files)

- Write every `\todo` paragraph (42 markers). Do not paste the outline
  comments in as prose.
- Choose the title. A title change may need the TPC chair's consent.
- Complete the author list, order and e-mails. The corresponding-author
  marker is a placeholder on Ryan Gomez; confirm it, and keep exactly one.
- Fill the pre-registration link (public repo commit URLs or OSF) for
  `b9ba0c9`, `cd5d145` and `6addd89`.
- At the `PREREG_V2.md` freeze, carry the frozen values into the paper:
  - the name of D*[SPECULATIVE] (expected B11, SPEC-θ);
  - the P3 margin m₃ = ½·δ̂_C,dev;
  - the primary (profile, price);
  - the dev estimates of multi-epoch prefetch delay and C3 query
    improvement, which the two-epoch power simulation omits.
  The design itself, the boundary test P1/P2/P3 (intersection–union at
  one-sided α = 0.025), is merged at `1d8900c`. B10 and B11 are implemented
  (`controller/baselines/b10_speculative_v2.py`,
  `b11_speculative_gated_v2.py`).
- Decide where the proofs go (6-page budget). The full proofs in
  `theory_formal.tex` will not fit.
- Choose a dead-air threshold for the v2 dead-air rate.
- Waste-rate sensitivity for v1 needs an extension of
  `analysis/v1_theory_check.py`, which has not been written yet.
- Write `analysis/paper_figures_v2.py` for the two v2 figure placeholders
  (prefix AUC; boundary test with Δ_C, Δ_S, ±m₃ and the interaction).
- Fill every `\TBD` from the named artifact after the frozen test-split
  runs. Never estimate one.
