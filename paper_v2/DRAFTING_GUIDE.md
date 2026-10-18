# Drafting guide: old paper → revised paper (Paper 4422, INDICON 2026)

This maps the **submitted paper** (old) to the **revised outline** (new, `paper_v2/sections/*.tex`).

**How to use it.**
1. Open the old PDF next to the matching `.tex` file.
2. For each paragraph ID (P-x.y), the table below says where the old text is, what to do with it, and the reviewer comment it answers.
3. Each `\todo{P-x.y}` block in the `.tex` file lists the claims and allowed numbers in full.

**Rules.**
- **Prose:** write every paragraph in your own words. The conference requires an AI-generated-text score under 10%. Your own old text may be reused.
- **Numbers:** use only the numbers listed here or in the `\todo` blocks, each with its source file. Never round or change a number without checking that file.
- **Placeholders:** every `\TBD{...}` in `sections/` and `tables/` is now filled in the comment blocks, each number with its source file (final results: `docs/research_v2/v2_test_results_final.md`, commit 341d306; extra offline numbers: `docs/research_v2/paper_numbers_v2.json`). Where a number could not be computed the comment says `NOT AVAILABLE` with the reason: drop that claim. One number is still running: `[waste v2 d1: pending]`.
- **Status of v2 numbers:** d1_open = pre-registered per-dataset analysis (unaffected by D3). d2_support and **every pooled result** = exploratory, because the judge agreed only moderately with the human labels. Describe the human check only as: one human annotator labelled 300 d2 dev answers; judge gpt-6.1-sol 240/300 (80.0%, κ 0.55), GPT-4o 244/300 (81.3%, κ 0.58).
- **Message:** joint control of retrieval and timing rarely pays off with today's belief signals, and a decomposed "retrieve late in the utterance" rule wins, because early partial questions retrieve worse documents. Do **not** frame it as "our controller wins".

**Actions.**
- **REUSE:** keep the old text, light edits.
- **EDIT:** keep the structure, change the content as described.
- **REWRITE:** same topic, new text.
- **NEW:** no old counterpart.

Reviewer tags: R1-n is Reviewer 1 major comment n; R1-m is a Reviewer 1 minor comment; R2-x is a Reviewer 2 comment.

---

## Abstract (`00_abstract.tex`)

**Old:** Abstract. **Action:** REWRITE.

**Reviewer:** R1-m (the old abstract is informal: "pits two approaches", "Building a voice model…"). Use a formal register and give key numbers.

**What changes:**
- **State:** the question; the theory (equivalence and its boundary); two new benchmarks with real streaming ASR; a pre-registered test.
- **Headline:** decomposed beats joint. On D1 test, Δ = −0.037, 95% CI [−0.048, −0.025]; pooled (exploratory) Δ = −0.035 [−0.046, −0.025].
- **Persona finding:** entropy triggers fail on support questions, and self-assessment works better.

---

## I. Introduction (`01_introduction.tex`)

| ID | Old source | Action | What changes | Reviewer |
|---|---|---|---|---|
| P-I.1 | I, paras 1–3 (RAG reliability; not every query needs retrieval; timing during speech) | EDIT | Keep the motivation. Cite Stream RAG [arora2026streamrag] as prior work on retrieving during speech | R2 originality |
| P-I.2 | I, para 4 + abstract's B9 description | EDIT | The joint-vs-decomposed question. The v1 result was a tie (0/5 seeds), which we now **explain** rather than call "underpowered" | R1-1 |
| P-I.3 | I, contribution bullets | REWRITE | The uncertainty trigger is **adopted** (FLARE/DRAGIN), not novel. Early retrieval during speech is background. Novelty (hedge "to our knowledge"): (1) the formal joint-vs-decomposed analysis; (2) a pre-registered boundary test; (3) the persona/entropy finding; (4) a spoken support benchmark | R1-m (contribution 1), R2 originality |
| P-I.4 | none | NEW | Roadmap sentence | none |

---

## II. Related work (`02_related_work.tex`)

| ID | Old source | Action | What changes | Reviewer |
|---|---|---|---|---|
| P-II.1 | none | NEW | Streaming and speculative retrieval in speech: Stream RAG (ICML 2026), Galbraith 2026, Jung et al. 2026, MoshiRAG. They assume every query needs retrieval and have no decision model | R2 originality |
| P-II.2 | II-A (RAG & adaptive retrieval) | EDIT | Keep FLARE, DRAGIN and Adaptive-RAG. Add Moskvoretskii et al. (uncertainty methods often fail) and Kadavath (P(IK) self-assessment). **Drop DPR** (15-reference cap) | R2 originality |
| P-II.3 | none | NEW | Benchmarks: τ-Voice (task success, not when to retrieve); PopQA and HotpotQA (our D1 sources) | R1-2, R2 scale |
| P-II.4 | II-B (turn-taking) + II-C (decision theory) | EDIT, shorten | Keep VAP and Bohus & Horvitz. Cite **Howard 1966** for value of information. **Drop TurnGPT, the Skantze review, Cooper, and Rao & Daumé.** Fix the old wrong citation [9] for TurnGPT/VAP | R1-m (citations) |

---

## III. Problem & controllers (`03_problem.tex`)

| ID | Old source | Action | What changes | Reviewer |
|---|---|---|---|---|
| P-III.1 | III-A (problem formulation, Eqs. 1–3) | EDIT | Keep the action space and utility. Add the per-turn decision made at each partial transcript | none |
| P-III.2 | III-B (belief state) | REWRITE | v1: the belief is entropy on the **full** question, calibrated in-sample. v2: a **self-assessment** probe on each partial transcript, calibrated on dev only, with its latency measured (0.55 s) | R1-4, R1-5, R2 calibration |
| P-III.3 | III-D (quality term) | REWRITE | Quality is no longer equal to the belief. It is b·g⁺ − (1−b)·g⁻: retrieval helps when needed and hurts when not. v1: g⁺ = 0.720 [0.504, 0.888], g⁻ = 0.592 [0.408, 0.768]. **Delete "architectural strength"** | R1-6, R1-m |
| P-III.4 | III-D, "Methodological note on cost parameters" | EDIT | waste_rate is **still a placeholder in v2** (w(f) = 0.1(1 − f), decision only; not measured); v1 sensitivity over w ∈ {0, 0.04, 0.08, 0.16} (B9+ = Ours on 1,300/1,300 at every w); v2 d1 sensitivity `[waste v2 d1: pending]`. Retrieval's real TTFA cost in v1 was 1.57 s (sentence end) / 1.20–1.30 s (early), against the assumed 0.67 / 0.48 s | R1-10, R2 cost model |
| P-III.5 | III-E (Ours, B9, "Mechanism of divergence") | REUSE + extend | Keep the old text; it is good. Add B9+ (the strong decomposed rival), B10/B11 (speculative prefetch), Oracle and the fixed B4. Point to the pseudocode | R1-1, R2 baseline definitions |
| P-III.6 | III-C ("multi-timescale" action space) | REWRITE, shorter | Drop "multi-timescale". Define each dispatch rule precisely, including the "fixed early proxy" | R1-m, R1-7 |

---

## IV. Theory (`04_theory.tex`; the equations are already in `theory_formal.tex`)

| ID | Old source | Action | What changes | Reviewer |
|---|---|---|---|---|
| P-IV.1 to P-IV.4 | none | NEW (prose only) | Write short prose around the propositions. Proofs go in an appendix or the repo link if space is short | R1-1 |

The propositions:
- **Prop. 1:** joint and decomposed are identical under three conditions.
- **Prop. 2:** the gap is bounded by the divergence band.
- **Prop. 3:** turn-varying latency.
- **Prop. 4:** information vs latency.

---

## V. Setup (`05_setup.tex`)

| ID | Old source | Action | What changes | Reviewer |
|---|---|---|---|---|
| P-V.1 | IV-B (datasets, v1) + IV-C (Table I) | EDIT | Disclose the v1 weakness: hotel vs trivia questions are separable by wording, with semantic-detector AUC 1.00 (E1) and 0.94 (E3). Model: Mistral-7B-Instruct-v0.3 only | R1-2 |
| P-V.2 | none | NEW | What the seeds vary (generation and timing, not items). The item is the unit of analysis, with a paired bootstrap over items | R1-3, R2 power |
| P-V.3 | IV-E (metrics) + IV-F (statistics) | EDIT | "Retrievals per **turn**", not per dialogue. Fix the run-on "Latency/Compute" sub-headings. Judge: GPT-6.1-Sol with rubric r3, checked against one human labelling of 300 answers (80.0%, κ 0.55; GPT-4o 81.3%, κ 0.58), one sentence only, no gate details; GPT-4o as secondary. Dead Air: report it or state that it equals TTFA for single-turn runs | R1-9, R1-10, R1-m |
| P-V.4 | none | NEW | **D1:** 5,300 open-domain items (PopQA 5,000, popularity-stratified, + HotpotQA 300); dev 1,566 / test 3,734. **D2:** 720 support items, 12 industries, 24 fictional companies, matched pairs; dev 222 / test 498. **Leak test:** wording AUC 0.576, semantic 0.74 (v1 was 1.00 / 0.94). The empirical "needs retrieval" label means the model is wrong closed-book | R1-2, R2 scale |
| P-V.5 | none | NEW | **Real streaming ASR** on synthetic speech: 6 voices incl. 2 Indian-English passed the accent screen, but the pre-registered **test runs used one voice** (Piper en-US hfc_male, 150 wpm, clean) plus a text-prefix condition; partial WER 23.7% (D1) / 10.5% (D2); final WER 14.2% / 5.5%; VAD end of turn about 830 ms after the last word. Retrieval runs on the live partial transcript, not a frozen query | R1-8, R2 realistic speech |
| P-V.6 | none | NEW | Commit vs speculative architectures. Latency profiles: rewrite + search p50 ≈ 1.04 s; internet retrieval 268 ms; streaming TTS. Early dispatch can hide ≈ 1.07 s | R1-7 |

---

## VI. v1 results, corrected (`06_results_v1.tex`)

| ID | Old source | Action | What changes | Reviewer |
|---|---|---|---|---|
| P-VI.1 | V-A (E1) | EDIT | Keep max-entropy AUC 0.978. **Add** mean-entropy AUC 0.925 (the signal actually used). On held-out E3 items: mean 0.922 vs max 0.882; ECE 0.083. Say the calibration was in-sample | R1-5, R2 calibration |
| P-VI.2 | V-B (E2) | EDIT with **corrections** | See the bullets below | R1-7, R2 E2 tail |
| P-VI.3 | V-C, first sentence | REUSE the verdict | 0/5 seeds of dominance, a tie | none |
| P-VI.4 | V-C (rest) + Figs. 2–3 | REWRITE | The submitted numbers and figures came from **sample data**; disclose this plainly. Real run, with Ours = B9 at low prices: see the bullets below. Use the new Figs. 2–3 | R1-9, figures |
| P-VI.5 | none | NEW | Why it tied (the equivalence): the exact price map reproduces 10,000/10,000 decisions; B9+ = Ours on 11,300/11,300; the divergence band holds 41/1,300 decisions (3.2%) | R1-1 |
| P-VI.6 | none | NEW | The corrected objective predicts the v1 outcome. Use the numbers in the `\todo` (source `v1_theory_check.json`) | R1-6 |

**P-VI.2 corrections (E2).** p95 TTFA was 7,090 / 7,052 / 6,915 ms (early / sentence end / immediate), and medians 3.69 / 4.11 / 4.07 s. Then:
- **479 ms is a MEAN of paired differences**, CI [−690, −275], not a median.
- **Seeds** were 0–4.
- **The query was frozen**, so partial-query risk was zero by design.
- **TTFA ended at a synthetic tone**: no real speech synthesis ran.
- **Retrieval time:** 0.66 s was CPU contention; the isolated call takes 20–60 ms.
- **Add the latency breakdown table.**

**P-VI.4 real-run numbers (E3).**
- Ours = B9: accuracy 0.616, 0.82 retrievals/turn, p95 5.15 s.
- B0 (never retrieve): 0.408.
- B6: 0.636 at 0.76 retrievals/turn.
- B8: 0.708 at 0.68 retrievals/turn.
- B4 never fired, so it equals B0.

---

## VII. v2 results (`07_results_v2.tex`)

| ID | Old source | Action | What changes |
|---|---|---|---|
| P-VII.1 | none | NEW | The pre-registered tests: P1, P2, P3 (boundary test; P3 margin m₃ = 0.016, the smallest effect size of interest), **plus H-REV** (decomposed beats joint) and the headline estimate. Frozen before test data was touched (tag `prereg-v2-frozen`, 2026-10-16). Three logged deviations (D1, D2, D3) and three notes (N1–N3); D3 makes every d2_support and pooled result exploratory; d1_open is unaffected |
| P-VII.2 | none | NEW | **Judge vs human:** 80.0%, κ 0.55, so d2/pooled accuracy results are exploratory. **E1 test:** D1 AUC 0.665 [0.648, 0.682] (below 0.70, above the 0.60 kill line); rises with more of the question heard (0.537 → 0.665); ECE 0.077. D2 (exploratory): AUC 0.701 [0.638, 0.761]; rises 0.579 → 0.701 (text) / 0.558 → 0.708 (ASR); ECE 0.047. C2 not available on either dataset. Persona finding: on D2 mean-entropy AUC is 0.508 (chance), on D1 it is 0.707 (better than the probe's 0.665) |
| P-VII.3 | none | **DELETE** | "Channels-off replication" was removed from the pre-registration (not implemented). Drop this paragraph |
| **P-VII.H (add)** | none | NEW | **Headline + H-REV.** D1 test: joint − decomposed Δ[COMMIT] = **−0.0367 [−0.0481, −0.0252]**. H-REV holds (p < 1e-4). The direction holds at all 12 profile/price settings (−0.035 to −0.046). Pooled (exploratory): −0.0351 [−0.0458, −0.0245], H-REV p < 1e-4. D2 alone (exploratory): −0.0227 [−0.0502, +0.0047] |
| P-VII.4 | none | NEW | **P1** (joint > decomposed, commit): fails (p = 1.0) |
| P-VII.5 | none | NEW | **P2** (interaction): +0.0047 [+0.0007, +0.0088]; p = 0.0097; fails after the Holm correction. Pooled (exploratory): +0.0022 [−0.0020, +0.0065], p = 0.160 |
| P-VII.6 | none | NEW | **P3** (equivalence under speculative, margin 0.016): fails on the lower side. Δ[SPEC] = −0.0415 [−0.0528, −0.0302]. Pooled (exploratory): −0.0373 [−0.0480, −0.0267], fails. Boundary claim not supported (d1, d2, pooled) |
| P-VII.7 | none | NEW | Secondary condition (typed text, no ASR): Δ = −0.0024 [−0.0072, +0.0022], a tie. Secondary family (6 two-sided tests, Holm 0.05; exploratory as a family): D1 Δ[COMMIT] and Δ[SPEC] p < 1e-4 (reject), D1 interaction p = 0.019 (not rejected), D2 all n.s. Predicted vs observed: D1 Spearman 0.870, slope 0.827 [0.582, 1.077], "agree". NOT AVAILABLE (drop): secondary-judge re-report of P1–P3, dead-air rate, frontier dominance, v2 E2 timing on test, max-entropy controller run |

**Source** for all D1 numbers: `docs/research_v2/v2_test_results_d1.md`.

---

## VIII–IX. Discussion & conclusion (`08_discussion.tex`, `09_conclusion.tex`)

| ID | Old source | Action | What changes |
|---|---|---|---|
| P-VIII.1 | VI (Discussion) | REWRITE | The v1 tie was the equivalence regime, not "underpowered" |
| P-VIII.2 | none | NEW | Practical guidance: build a joint controller only when retrieval must be committed **and** early beliefs are informative. Otherwise use decomposed late dispatch. Mechanism: early partial queries retrieve worse documents (dev: Ours dispatched earlier and lost accuracy, 0.44 vs 0.50) |
| P-VIII.3 | VII (Limitations paragraph) | REWRITE as a list | (1) one model (Mistral-7B); (2) synthetic speech and fictional companies; (3) weak beliefs (AUC ≈ 0.67): the result holds *for today's signals*; (4) small effects (≈ 4 points); (5) judge non-determinism (temperature fixed at 1); (6) energy not modelled in the main analysis |
| P-IX.1 | VII (Conclusion paragraph) | REWRITE | Restate the answer and the guidance in 4–5 sentences |

---

## Must-include disclosures (anywhere appropriate; also in RESPONSE_TO_REVIEWERS.md)
1. The submitted E3 numbers and Figs. 2–3 came from sample data. All are now from the real run, and the 0/5 verdict is unchanged.
2. v1 TTFA ended at a synthetic test tone, and v1 retrieval time was inflated by CPU load.
3. The 479 ms figure is a mean, not a median.
4. The consolidated v1 pre-registration file was marked frozen after the run; the criterion itself was committed earlier (commits `b9ba0c9` 2026-08-06, `cd5d145` 2026-08-17).
5. The v2 pre-registration is frozen (tag `prereg-v2-frozen`, 2026-10-16; OSF registration still pending, open item O5), with three logged deviations (D1–D3) and three notes (N1–N3). D3 (the judge agreed only moderately with the human labels) makes every d2_support and pooled result exploratory; d1_open is unaffected.

## Final checks before submission (Oct 18)
1. **Title and authors:** the title is FIXED as submitted ("Joint vs. Decomposed Control of Retrieval and Timing in Streaming Voice RAG", already Title Case; do not change it); authors with numeric affiliations and no honorifics; one corresponding-author mark.
2. **References:** 15 maximum (the new `references.bib` has exactly 15), all cited with `\cite{}`.
3. **Figures:** refer to each in the text as "Fig. N shows …".
4. **Template:** port to the official INDICON Overleaf template, compile, check the page limit.
5. **Checks:** plagiarism and AI-content scores below 10%.
