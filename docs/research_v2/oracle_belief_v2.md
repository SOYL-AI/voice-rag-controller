# Oracle-belief ablation, d1_open test (EXPLORATORY)

**Status: EXPLORATORY post hoc ablation (not pre-registered; not part of the pre-registered test).** Post hoc: designed and run after the pre-registered d1_open test was analysed; PREREG_V2 does not mention it and none of its tests or verdicts change. Alias-graded outcomes; no LLM judge. Source: `docs/research_v2/oracle_belief_v2.json` (`analysis/v2_oracle_belief_d1.py`).

## Question and design

Does the joint controller (Ours-v2, "Ours-S" in the paper) beat the decomposed rival B9+ when the belief it plans with is good? Ours-v2 and B9+ (frozen dev-tuned timing rule) are re-run on the d1_open test items under both architectures, with D*[SPEC] = B9+ as frozen for d1, changing ONLY the belief b_k (at every partial and at the end of turn) and what is derived from it:

* **oracle**: b_k = y, the E1-v2 empirical need label (closed-book wrong in ≥ 3 of 5 draws, alias), from the first partial on. The belief never moves, so waiting buys no information (no C2).
* **sharpening oracle**: b_k = p0 + f_k (y − p0), p0 = 0.5600 (dev base rate of y), f_k = the causal fraction-heard estimate e/(e + R̂) the controllers already use (1 at the end of turn; never the reference word fraction). The belief moves from the prior towards the label as speech is heard. It is NOT calibrated at intermediate steps: b_k is a deterministic function of y, so P(y = 1 | b_k) is 0 or 1 while b_k is in between; a controller that reads b_k as a probability misjudges it (a not-needed item at b_k = 0.15 is treated as a 15% chance of need).
* **calibrated reveal** (added by the analyst after the sharpening variant proved miscalibrated; not in the original request): b_k = p0 until f_k ≥ u_i, then y, with u_i ~ U(0, 1) a fixed hash of the item id (independent of y). P(y = 1 | b_k = p0) = p0, so the belief is calibrated at every step and sharpens (reveals the label) as speech is heard; b = y at the end of turn.
* Refitted (dev split only): Ours-v2's belief-transition (lookahead) model, per variant, with the frozen fitter on dev sequences of that variant's belief as the controller computes it (3,132 dev turns over both conditions; phase = causal f; the oracle fit is the identity on {0, 1}, the sharpening fit encodes the drift towards the label, the reveal fit the per-phase reveal probability). Not refitted: B9+ timing rule (frozen dev-tuned, belief-independent r_below / r_threshold rule), B9+ policy constants, prices, g+/g-, rho table, remaining-speech predictor, latency model.
* Gate: with the frozen belief this re-implementation reproduces the per-item Δ[COMMIT] and Δ[SPEC] of the frozen test contrasts to 1e-12 at both cells (first row of each table).
* Statistics: paired item bootstrap, B = 10,000, seed 20261017, the resamples of `analysis/v2_test_d1.py`; percentile 95% CIs. U = accuracy − λ_l (TTFA − skip TTFA) − λ_c · retrievals per item (mean of 5 seeds); Δ = U[Ours-v2] − U[B9+].

## Primary cell (primary ASR, rewrite_faiss, belief quantile 0.65, λ_l = 0.17305735434823866, λ_c = 0.05)

| belief | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction Δ[C] − Δ[S] [95% CI] | n items |
|---|---|---|---|---|
| frozen belief (gate; = pre-registered test) | -0.0367 [-0.0481, -0.0252] | -0.0415 [-0.0528, -0.0302] | +0.0047 [+0.0007, +0.0088] | 3,734 |
| oracle: b = y at every step | +0.0132 [+0.0081, +0.0184] | +0.0132 [+0.0081, +0.0184] | +0.0000 [+0.0000, +0.0000] | 3,734 |
| sharpening oracle: b = p0 + f (y − p0) | -0.0555 [-0.0668, -0.0444] | +0.0132 [+0.0081, +0.0184] | -0.0686 [-0.0791, -0.0584] | 3,734 |
| calibrated reveal: b = p0, then y once f ≥ u | +0.0125 [+0.0070, +0.0180] | +0.0166 [+0.0108, +0.0224] | -0.0041 [-0.0064, -0.0019] | 3,734 |

### Utility decomposition and retrieval behaviour (primary cell)

Means over items. Latency and compute terms are the amounts SUBTRACTED from accuracy (U = accuracy − latency term − compute term). 'uses' = answer generated with passages; 'early' = share of dispatches made on a partial rather than the final transcript; 'uses | need' and 'uses | no need' split by the label y.

| belief | arch | controller | U | accuracy | latency term | compute term | dispatch | uses | early | uses \| need | uses \| no need | TTFA − skip (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | commit | ours_v2 | 0.2945 | 0.4377 | 0.0937 | 0.0495 | 99.1% | 99.1% | 99.4% | 99.2% | 99.0% | 541 |
| frozen | commit | b9plus_v2 | 0.3312 | 0.4879 | 0.1119 | 0.0448 | 89.6% | 89.6% | 94.8% | 90.2% | 88.9% | 647 |
| frozen | speculative | ours_v2 | 0.2898 | 0.4257 | 0.0864 | 0.0496 | 99.1% | 95.5% | 99.4% | 94.9% | 96.3% | 499 |
| frozen | speculative | b9plus_v2 | 0.3312 | 0.4879 | 0.1119 | 0.0448 | 89.6% | 89.6% | 94.8% | 90.2% | 88.9% | 647 |
| oracle | commit | ours_v2 | 0.4058 | 0.5073 | 0.0734 | 0.0281 | 56.2% | 56.2% | 91.1% | 100.0% | 0.0% | 424 |
| oracle | commit | b9plus_v2 | 0.3927 | 0.4971 | 0.0763 | 0.0281 | 56.2% | 56.2% | 87.6% | 100.0% | 0.0% | 441 |
| oracle | speculative | ours_v2 | 0.4058 | 0.5073 | 0.0734 | 0.0281 | 56.2% | 56.2% | 91.1% | 100.0% | 0.0% | 424 |
| oracle | speculative | b9plus_v2 | 0.3927 | 0.4971 | 0.0763 | 0.0281 | 56.2% | 56.2% | 87.6% | 100.0% | 0.0% | 441 |
| sharpen | commit | ours_v2 | 0.3372 | 0.5054 | 0.1185 | 0.0497 | 99.3% | 99.3% | 93.5% | 100.0% | 98.5% | 685 |
| sharpen | commit | b9plus_v2 | 0.3927 | 0.4971 | 0.0763 | 0.0281 | 56.2% | 56.2% | 87.6% | 100.0% | 0.0% | 441 |
| sharpen | speculative | ours_v2 | 0.4058 | 0.5073 | 0.0734 | 0.0281 | 56.2% | 56.2% | 91.1% | 100.0% | 0.0% | 424 |
| sharpen | speculative | b9plus_v2 | 0.3927 | 0.4971 | 0.0763 | 0.0281 | 56.2% | 56.2% | 87.6% | 100.0% | 0.0% | 441 |
| reveal | commit | ours_v2 | 0.3943 | 0.5072 | 0.0815 | 0.0314 | 62.8% | 62.8% | 92.0% | 100.0% | 15.1% | 471 |
| reveal | commit | b9plus_v2 | 0.3818 | 0.4964 | 0.0834 | 0.0312 | 62.5% | 62.5% | 88.9% | 100.0% | 14.4% | 482 |
| reveal | speculative | ours_v2 | 0.3984 | 0.5074 | 0.0775 | 0.0316 | 63.1% | 60.4% | 92.0% | 100.0% | 9.7% | 448 |
| reveal | speculative | b9plus_v2 | 0.3818 | 0.4964 | 0.0834 | 0.0312 | 62.5% | 62.5% | 88.9% | 100.0% | 14.4% | 482 |

Δ decomposed (Ours-v2 − B9+; Δ = Δaccuracy − Δlatency term − Δcompute term):

| belief | arch | Δ accuracy [95% CI] | Δ latency term [95% CI] | Δ compute term [95% CI] |
|---|---|---|---|---|
| frozen | commit | -0.0502 [-0.0617, -0.0385] | -0.0182 [-0.0203, -0.0161] | +0.0047 [+0.0043, +0.0052] |
| frozen | speculative | -0.0622 [-0.0735, -0.0510] | -0.0255 [-0.0275, -0.0234] | +0.0047 [+0.0043, +0.0052] |
| oracle | commit | +0.0103 [+0.0052, +0.0155] | -0.0029 [-0.0038, -0.0020] | +0.0000 [+0.0000, +0.0000] |
| oracle | speculative | +0.0103 [+0.0052, +0.0155] | -0.0029 [-0.0038, -0.0020] | +0.0000 [+0.0000, +0.0000] |
| sharpen | commit | +0.0084 [-0.0025, +0.0190] | +0.0422 [+0.0404, +0.0440] | +0.0216 [+0.0208, +0.0224] |
| sharpen | speculative | +0.0103 [+0.0052, +0.0155] | -0.0029 [-0.0038, -0.0020] | +0.0000 [+0.0000, +0.0000] |
| reveal | commit | +0.0108 [+0.0054, +0.0163] | -0.0019 [-0.0029, -0.0008] | +0.0002 [+0.0000, +0.0003] |
| reveal | speculative | +0.0110 [+0.0053, +0.0168] | -0.0059 [-0.0072, -0.0047] | +0.0003 [+0.0001, +0.0005] |

Effect of the belief on each controller (U[variant] − U[frozen belief], same items, paired):

| belief | arch | controller | ΔU vs frozen belief [95% CI] | decisions ≠ frozen belief | decisions ≠ oracle |
|---|---|---|---|---|---|
| oracle | commit | ours_v2 | +0.1114 [+0.0984, +0.1244] | 3,456 | — |
| oracle | commit | b9plus_v2 | +0.0614 [+0.0522, +0.0709] | 1,662 | — |
| oracle | speculative | ours_v2 | +0.1161 [+0.1033, +0.1291] | 3,451 | — |
| oracle | speculative | b9plus_v2 | +0.0614 [+0.0522, +0.0709] | 1,662 | — |
| sharpen | commit | ours_v2 | +0.0427 [+0.0311, +0.0542] | 2,760 | 2,266 |
| sharpen | commit | b9plus_v2 | +0.0614 [+0.0522, +0.0709] | 1,662 | 0 |
| sharpen | speculative | ours_v2 | +0.1161 [+0.1033, +0.1291] | 3,451 | 0 |
| sharpen | speculative | b9plus_v2 | +0.0614 [+0.0522, +0.0709] | 1,662 | 0 |
| reveal | commit | ours_v2 | +0.0998 [+0.0870, +0.1127] | 3,418 | 296 |
| reveal | commit | b9plus_v2 | +0.0505 [+0.0419, +0.0595] | 1,455 | 235 |
| reveal | speculative | ours_v2 | +0.1086 [+0.0959, +0.1218] | 3,435 | 293 |
| reveal | speculative | b9plus_v2 | +0.0505 [+0.0419, +0.0595] | 1,455 | 235 |

## text150 (contrast; rewrite_faiss, its own 0.65-quantile price 0.161524|0.05, B9+ rule r_threshold)

| belief | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] | n items |
|---|---|---|---|---|
| frozen belief (gate; = pre-registered test) | -0.0024 [-0.0072, +0.0022] | -0.0024 [-0.0072, +0.0022] | +0.0000 [+0.0000, +0.0000] | 3,734 |
| oracle: b = y at every step | -0.0927 [-0.1038, -0.0817] | -0.0927 [-0.1038, -0.0817] | +0.0000 [+0.0000, +0.0000] | 3,734 |
| sharpening oracle: b = p0 + f (y − p0) | +0.0282 [+0.0228, +0.0337] | -0.0924 [-0.1034, -0.0816] | +0.1206 [+0.1116, +0.1299] | 3,734 |
| calibrated reveal: b = p0, then y once f ≥ u | -0.0428 [-0.0523, -0.0334] | -0.0427 [-0.0523, -0.0334] | -0.0000 [-0.0001, +0.0000] | 3,734 |

| belief | arch | controller | U | accuracy | latency term | compute term | uses | early | uses \| need | uses \| no need |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen | commit | ours_v2 | 0.5042 | 0.6099 | 0.0868 | 0.0189 | 37.8% | 9.1% | 46.3% | 27.0% |
| frozen | commit | b9plus_v2 | 0.5066 | 0.5938 | 0.0725 | 0.0147 | 29.4% | 0.0% | 37.8% | 18.6% |
| frozen | speculative | ours_v2 | 0.5042 | 0.6099 | 0.0868 | 0.0189 | 37.8% | 9.1% | 46.3% | 27.0% |
| frozen | speculative | b9plus_v2 | 0.5066 | 0.5938 | 0.0725 | 0.0147 | 29.4% | 0.0% | 37.8% | 18.6% |
| oracle | commit | ours_v2 | 0.5735 | 0.7830 | 0.1637 | 0.0458 | 91.6% | 38.7% | 100.0% | 80.8% |
| oracle | commit | b9plus_v2 | 0.6661 | 0.8100 | 0.1195 | 0.0243 | 48.7% | 1.1% | 85.7% | 1.2% |
| oracle | speculative | ours_v2 | 0.5735 | 0.7830 | 0.1637 | 0.0458 | 91.6% | 38.7% | 100.0% | 80.8% |
| oracle | speculative | b9plus_v2 | 0.6661 | 0.8100 | 0.1195 | 0.0243 | 48.7% | 1.1% | 85.7% | 1.2% |
| sharpen | commit | ours_v2 | 0.6961 | 0.8642 | 0.1394 | 0.0287 | 57.4% | 2.2% | 100.0% | 2.9% |
| sharpen | commit | b9plus_v2 | 0.6680 | 0.8111 | 0.1191 | 0.0241 | 48.2% | 0.0% | 85.7% | 0.0% |
| sharpen | speculative | ours_v2 | 0.5756 | 0.7852 | 0.1638 | 0.0458 | 91.6% | 38.7% | 100.0% | 80.8% |
| sharpen | speculative | b9plus_v2 | 0.6680 | 0.8111 | 0.1191 | 0.0241 | 48.2% | 0.0% | 85.7% | 0.0% |
| reveal | commit | ours_v2 | 0.6235 | 0.8153 | 0.1533 | 0.0385 | 77.0% | 27.0% | 100.0% | 47.5% |
| reveal | commit | b9plus_v2 | 0.6663 | 0.8100 | 0.1194 | 0.0243 | 48.6% | 0.9% | 85.7% | 1.0% |
| reveal | speculative | ours_v2 | 0.6235 | 0.8153 | 0.1533 | 0.0385 | 77.0% | 27.0% | 100.0% | 47.5% |
| reveal | speculative | b9plus_v2 | 0.6663 | 0.8100 | 0.1194 | 0.0243 | 48.6% | 0.9% | 85.7% | 1.0% |

## New generation

* The oracle variants land on 1,125 (item, condition, retrieve, query variant) cells the frozen lazy plan had not generated (5,625 answers over the 5 seeds). They were generated with the frozen Phase A code path (`simulator/e3v2_phase_a.py --required-cells`, same prompts, seeds, retrieval index and passages, temperature/top-p/max-tokens) into a copy under `simulator/e3v2_oracle/d1_open/` (the frozen store is untouched) and graded by alias match only: 5,540 new Mistral-7B-Instruct-v0.3 answers (85 more had an identical answer key in the frozen run and were reused), 263,637 output tokens, ~4,749,047 prompt tokens, answer-stage wall time 7.3 min on the existing Modal vLLM endpoint; no LLM judge or Azure call. GPU cost was not metered: GPU time = this wall time x the number of serving containers, which Phase A does not record (64 requests in flight).
* 5,625 new (item, seed) cells were loaded from the new store; latency draws use the frozen common random numbers (seeded per item, seed, condition, profile).

## Frozen-objective terms behind the decisions

Model utility of a retrieval whose call is fully hidden (exposed = 0, so the priced latency is π alone) for an item with belief b = 0 and b = 1, at estimated word fraction f (frozen g+, g−, λ, π, the condition's dev ρ table, waste), and U(b = 0) at the end of turn (exposed = E[L]). With the dev-fitted g− < 0 (retrieval modelled as helping items that do not need it), U(b = 0) for a hidden dispatch is within ±0.005 of zero, so a confident 'not needed' belief does not by itself rule out a well-hidden retrieval, and on text150 it is slightly positive.

| condition | g+ | g− | λ_l | π (ms) | ρ at f = 0.5 / 0.7 / 0.9 | U(b=0) hidden, f = 0.5 / 0.7 / 0.9 | U(b=1) hidden, f = 0.5 / 0.7 / 0.9 | U(b=0) at end of turn |
|---|---|---|---|---|---|---|---|---|
| primary ASR | 0.4408 | -0.1123 | 0.1731 | 371 | 0.866 / 0.698 / 0.243 | -0.0044 / -0.0034 / -0.0024 | -0.0577 / +0.0174 / +0.2191 | -0.1914 |
| text150 | 0.4408 | -0.1123 | 0.1615 | 371 | 0.953 / 0.766 / 0.255 | -0.0001 / +0.0009 / +0.0019 | -0.0917 / -0.0085 / +0.2178 | -0.1745 |

## Summary

This ablation is exploratory and post hoc, not part of the pre-registered test: Ours-v2 and B9+ are re-run on the d1_open test items with only the belief replaced. With the empirical need label as the belief (oracle), the primary-cell contrast changes sign: Δ[COMMIT] = Δ[SPEC] = +0.0132 [+0.0081, +0.0184], against -0.0367 [-0.0481, -0.0252] with the real belief; both controllers then retrieve exactly the needed items and the gain comes from dispatch timing (Δ accuracy +0.0103, Δ latency term -0.0029), with zero interaction because no information arrives during the turn. With a calibrated belief that reveals the label at a random point of the utterance (a variant added by the analyst), joint control still wins (Δ[COMMIT] +0.0125 [+0.0070, +0.0180], Δ[SPEC] +0.0166 [+0.0108, +0.0224]), and the interaction is -0.0041 [-0.0064, -0.0019], the opposite sign to the P2 prediction. With the requested sharpening belief p0 + f (y − p0), which is informative but not calibrated mid-utterance, Ours-v2[COMMIT] commits early on intermediate beliefs and retrieves for 98.5% of the not-needed items (Δ[COMMIT] -0.0555 [-0.0668, -0.0444]), while Ours-v2[SPEC] decides exactly as under the oracle (Δ[SPEC] +0.0132 [+0.0081, +0.0184]). On text150 the oracle and reveal beliefs make Ours-v2 worse (Δ[COMMIT] -0.0927 [-0.1038, -0.0817] and -0.0428 [-0.0523, -0.0334]): with the dev-fitted g− = -0.112, the frozen objective values a fully hidden retrieval at f = 0.7 for an item known not to need it at +0.0009, so Ours-v2 takes it (80.8% of such items under the oracle) and pays in accuracy and latency. Taken together, the ablation suggests that the pre-registered negative result reflects the belief (and objective terms such as g−) more than the joint decision structure, but it does not show that a realistic belief reaches the regime where joint control pays: it is post hoc, on one dataset and price cell, alias-graded, with B9+'s timing rule tuned under the real belief, and the gain it finds is small.

## Caveats

* Exploratory and post hoc: designed after the pre-registered d1_open analysis; it is not part of PREREG_V2 and changes none of its tests. The 'calibrated reveal' variant was added by the analyst after seeing that the requested sharpening belief is miscalibrated mid-utterance.
* One dataset (d1_open), one latency profile and price per condition (the frozen primary cell and text150's own 0.65-quantile price), alias grading only (no LLM judge); items are the bootstrap clusters.
* Only the belief and Ours-v2's lookahead transitions change. B9+ keeps the timing rule tuned on dev under the REAL belief (r_below on ASR, r_threshold on text150); a rule re-tuned under the oracle belief could do better, so the oracle contrasts may favour Ours-v2. g+/g−, ρ, prices and the remaining-speech predictor are the frozen dev fits; text150 uses the g+/g− fitted on the primary ASR condition, which is what drives its negative oracle result.
* The oracle is the empirical need label (closed-book wrong on the text question in ≥ 3 of 5 draws); it is hindsight information no deployable controller has, and it is not the per-turn outcome (ASR-condition answers with or without passages), so 'perfect' refers to the label, not to the realised gain of retrieval.
* Newly generated cells (Mistral-7B, same Phase A code path) were measured on 2026-10-18, the frozen ones on 2026-10-16, on the same endpoint and model; vLLM sampling with fixed seeds is reproducible per request but not guaranteed bit-identical across server batches, so these cells are comparable to, not identical with, cells the frozen run would have produced.

