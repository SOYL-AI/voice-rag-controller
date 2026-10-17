# INDICON 2026 Compliance Checklist (Paper ID 4422, revision)

Status values:
- **pass**: met in the current files;
- **fail**: not met;
- **pending**: can only be checked after compiling, after authors' prose, or
  after v2 results.

| # | Rule | Status | Evidence / what remains |
|---|---|---|---|
| 1 | Official INDICON 2026 IEEE LaTeX template (IEEEtran conference) | **pending** | `main.tex` uses stock `IEEEtran` `[conference]`. It must be ported into the official template (Overleaf share link in `README.md`), keeping that template's class options and copyright block. |
| 2 | The paper compiles | **pending** | No TeX installation on the authoring machine. `tools/check_tex.py` passes: braces, environments, labels and references, citation keys, figure files. The paper has never been compiled; compile in Overleaf. |
| 3 | Title in Title Case | **pass** | The submitted title is kept unchanged and is in Title Case: "Joint vs. Decomposed Control of Retrieval and Timing in Streaming Voice RAG". |
| 4 | No honorifics or designations in author names | **pass** | The author block lists names only (no "Dr.", "Prof." or job titles). Re-check when the remaining authors are added. |
| 5 | Numeric affiliation superscripts | **pass** | `\textsuperscript{1}` is used for SOYL AI Private Limited, Bengaluru. |
| 6 | Exactly one corresponding-author marker | **pending** | One `*` is placed on Ryan Gomez as a placeholder. Authors must confirm who it is. |
| 7 | Author information complete | **fail** | Authors 3–4 and Renya Peter's e-mail are placeholders (`\PEND`). |
| 8 | Every equation numbered and referenced in text | **pending** | All display equations use `equation` (numbered) and carry labels: belief, obj, exposure, ttfa, band, plus those in `theory_formal.tex`. Text references to each must be written in the authors' prose. Unnumbered `\[ \]` displays remain inside `theory_formal.tex` (Prop. 1, Prop. 2) and must become `equation` if the template rule covers every display. |
| 9 | Every table numbered and referenced in text | **pass** (outline) | Five tables (controllers, predictions, v1e3, e2, v2primary) are each referenced (`check_tex.py`: no unreferenced `tab:` label). Re-check once the prose is written. |
| 10 | Every figure numbered and referenced in text ("Fig. 2 shows ...") | **pass** (outline) | Seven figure environments (Fig. 1, frontiers, latency, band, shift, and two v2 placeholders) are each referenced. Author prose must use the "Fig. N shows" form. |
| 11 | Figures high-resolution and legible | **pass** (v1) / **pending** (Fig. 1, v2) | The v1 figures are vector PDFs with embedded TrueType fonts at 8–9 pt at print size and no plot titles. Fig. 1 is TikZ (vector) at ≥ 8 pt but uncompiled. The two v2 figures do not exist yet. |
| 12 | BibTeX `.bib` with `\cite{}` | **pass** | `references.bib`, `\bibliographystyle{IEEEtran}`, all citations through `\cite{}`. |
| 13 | At most 15 references | **pass** | Exactly 15 entries, all cited (`check_tex.py`). |
| 14 | No self-citations | **pass** | No entry has an author from this paper or SOYL AI. |
| 15 | References real and accurate | **pass** | Each entry was checked against the publisher, Anthology, PMLR, ISCA, Crossref or arXiv record, with the URL in a comment. The published venue is used where one exists (Stream RAG and MoshiRAG: PMLR v306). Four entries are arXiv-only (Galbraith, Jung, Kadavath, tau-Voice). |
| 16 | Formal English | **pending** | Depends on author prose. The outline flags informal or overstated phrases to avoid ("dramatically", "architectural strength", "multi-timescale"). |
| 17 | Plagiarism score < 10% | **pending** | Check the final PDF with the conference's checker. Risk: the copied v1 text (paper/, main) and `theory_formal.tex` connecting prose must be re-worded. |
| 18 | AI-generated-content score < 10% | **pending** | All body prose is to be written by the authors (`\todo` markers; 42 remain). `theory_formal.tex` contains connecting sentences and proof wording drafted outside the author team; re-word them before submission. Run the conference's detector on the final PDF. |
| 19 | No AI attribution in the manuscript, commits or files | **pass** | None is present in `paper_v2/` or in the commit messages on `v2/paper`. |
| 20 | No draft markers in the submitted PDF | **pending** | Set `\finaltrue` in `macros.tex`; the build then errors on any remaining `\todo`, `\TBD` or `\PEND`. |
| 21 | Page limit | **pending** | Check the INDICON author kit (usually 6 pages plus paid extras). The full proofs will not fit; decide on a supplementary file or an extended version. |
| 22 | Numbers trace to real data | **pass** (v1) / **pending** (v2) | Every v1 number has a source path in a comment and regenerates from committed data. No number from `e3_sample_result/` (synthetic) is used, and the figure script refuses that folder. All v2 numbers are `\TBD{artifact}`. |
