# Contribution statement for the revised paper

## What is new

Adaptive-retrieval work decides **whether** or **how much** to retrieve:

* FLARE (Jiang et al., 2023) triggers retrieval on low-confidence tokens of an upcoming sentence.
* DRAGIN (Su et al., 2024) triggers on real-time information need during generation.
* Self-RAG (Asai et al., 2024) uses learned reflection tokens.
* Adaptive-RAG (Jeong et al., 2024) routes on predicted query complexity.
* Moskvoretskii et al. (2025) compare uncertainty-based triggers.

Turn-taking work predicts **when** a user will stop speaking: TurnGPT (Ekstedt & Skantze, 2020), VAP (Ekstedt & Skantze, 2022), and the review by Skantze (2021).

None of these asks when deciding the two **jointly** is worth anything over deciding them separately with shared prices. The revised paper answers that question exactly. Its contributions:

1. **An equivalence theorem that explains a null result.** If answer quality does not depend on timing, overlap is a constant per policy, and the belief does not depend on dispatch time, then the joint controller at latency price λ is the decomposed controller at an explicitly mapped price λ′. This holds exactly, including the waste and compute terms (Prop. 1). The two price sweeps therefore trace the same frontier. Our v1 null is a corollary of this theorem, not a failed experiment. It is verified on the real sweep:
   * 1,250/1,250 decisions agree on the grid, and 10,000/10,000 off it;
   * the decomposed controller the reviewer proposed, which prices expected exposure, reproduces the joint controller on all 11,300 decisions checked.
2. **Exact gaps and their sizes.** We give:
   * a closed form for the fixed-price gap (Prop. 2);
   * the joint-vs-decomposed gain under turn-dependent overlap. It is a conditional Jensen gap, $\mathbb E_b[\min(\mathbb E[\Delta^+\mid b],\mathbb E[\Delta^-\mid b])]$, and its second-order form $f(\theta^*)\lambda_l^2\,\mathrm{Var}(e^*)/(2(g^++g^-))$ shows the C1 gain is small by an envelope argument (Prop. 3).
3. **The real coupling is an optimal-stopping problem.** Waiting for a longer transcript buys information (Howard's value of information) and loses overlap. The resulting stopping region is not a product of a belief rule and a timing rule, so no decomposition can represent it (Prop. 4).
   * **Boundary conditions.** Speculative dispatch removes the advantage when compute is cheap (Prop. 4′). Partial-query risk alone does not separate the policy classes (Prop. 5).
4. **A corrected objective grounded in measurement.** Retrieval harms questions that do not need it ($g^-=0.59$ vs $g^+=0.72$ on v1). This moves the zero-price retrieve boundary from 0.05 to 0.45. It also explains the sign of v1's mid-grid gap, which the original objective cannot produce.
5. **A pre-registered confirmatory test of the theory's predictions.** The predictions are signed and sized: exactly 0 where the channels are off, and positive where they are on. The test runs on new open-domain and customer-support data, with a power analysis built from v1 per-item outcomes and a validated judge.

## Title options

1. *When Does Joint Control Pay? Retrieval and Timing in Streaming Voice RAG*
2. *Joint vs. Decomposed Control of Retrieval and Timing in Streaming Voice RAG: An Equivalence Result and Its Boundaries*
3. *Information versus Overlap: Optimal Stopping for Retrieval Dispatch in Streaming Voice Assistants*

## Reviewer comments → changes → evidence

Rows 1–2 restate Reviewer 1's comments as relayed to us. The remaining rows are v1 limitations a reviewer can be expected to raise, inferred from the v1 record. **Replace the left column with the verbatim reviewer text before sending.**

| comment | change in v2 | evidence |
|---|---|---|
| R1: the E3 null is analytically predictable; derive the divergence band, report the fraction of turns in it, show it matches | Props. 1–2 with full proofs; band computed with the controllers' exact cuts at all 26 prices | THEORY §8.3: band non-empty at 11/26 prices, 41/1,300 decisions (3.15%); observed disagreements equal the band at every price. §8.4: predicted vs observed gap, Pearson 0.952 (corrected objective, measured cost) |
| R1: a decomposed controller that prices expected early-dispatch latency would equal Ours, so B9 is near-strawman | B9+ defined, implemented and proved equal under A1–A3; v2's primary rival is the strongest decomposed controller (B9+ family, plus SPEC under speculation) | THEORY §8.5: B9+ = Ours on 11,300/11,300 decisions; Prop. 3(2): B9+ is the best decomposed rule when b ⊥ Z |
| (inferred) the result rests on 50 saturated hotel scenarios | Two new datasets (d1_open, d2_support), a sealed 70% test split, and a power analysis | PREREG_V2 §1, §6: about 4,000 pooled test items needed for 80% power |
| (inferred) the timing policy was a fixed proxy and the query frozen, so the partial-query risk is 0 by construction | Live ASR partials, measured ρ(t), latency profiles; H-E2v2 keeps the v1 p95 criterion | PREREG_V2 H-E2v2 |
| (inferred) the cost model is not grounded | Latency calibrated as the end-to-end TTFA increment; the v1 constants understated it by more than 2× (1.20 s vs 0.484 s) | THEORY §8.4 |
| (inferred) judge validity | A gate on about 300 double-labelled answers, majority-of-3 verdicts from the GPT-6 judge, and every primary result re-reported under the deterministic GPT-4o v4 judge | PREREG_V2 §2 |
| (inferred) generality | Stated: one model (Mistral-7B-Instruct-v0.3), synthetic speech | PREREG_V2 §10 |
