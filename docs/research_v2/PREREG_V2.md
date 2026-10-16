# Pre-Registration Addendum v2 — FREEZE-READY DRAFT (NOT FROZEN)

> **Status: DRAFT, ready for freeze.** Every value is filled from dev artifacts except the open items in §11. Nothing in this file binds until the freeze procedure in §9 is complete. Until then the test split stays sealed (`CONTRACTS.md` §7): no generation, signal extraction, tuning or judging may touch test items.
>
> * **Controller and analysis code: frozen at commit `ddde67fdae9b80d80e295f20da45efe9b0dc2540`** (`ddde67f`, "Merge final pre-freeze dev rerun"). The test split is run with exactly this code.
> * **Dev predictions of record:** `docs/research_v2/v2_predictions.json` (generated 2026-10-16T03:28:25Z from commit `d771276`).
> * **Dev power of record:** `docs/research_v2/v2_power_dev.json`.
> * **Earlier prediction files**, kept as history and not used: `v2_predictions_pre_lookahead_fix.json`, `v2_predictions_b9plus_extended.json` and `v2_predictions_post_rbelow_inert_fix.json`.
> * Owner: Yashmita. Sign-off: Ryan (PI). Price-grid confirmation: Renya.

## 0. Relationship to v1

1. **v1 results stand as originally reported.** `paper/preregistration.md` (frozen 2026-08-21) and its criteria and outcomes are not modified by this addendum:
   * E1: PASS (AUC 0.9784);
   * E2: p95 criterion NOT MET;
   * E3: 0/5 seeds of dominance, reported as a tie.

   v1 answers remain graded by the GPT-4o judge (v4 reasoned+strip, temperature 0), exactly as reported.
2. **v2 is a new confirmatory study.** It tests predictions on new datasets and a test split that nothing has touched. No v1 data enter any v2 confirmatory test.
3. **Model.** v2 uses exactly one generator, **`mistralai/Mistral-7B-Instruct-v0.3`**, served unquantized (bf16) with vLLM, as in v1. There is no second model.
   * The v1 note on substituting Llama-3.1-8B with Mistral-7B is a historical fact about v1 only.
   * All v2 claims are about Mistral-7B. Generality across models is untested and is stated as a limitation (§10).
   * Generation and signal-extraction settings are those in `configs/config.yaml` and `llm/experiment1_v2/` at the frozen commit.
   * Seeds: 11, 23, 37, 53, 71.

## 1. Datasets and splits

| dataset | content | items | dev | test | cluster for resampling |
|---|---|---|---|---|---|
| `d1_open` | 5,000 PopQA (5 popularity bins) + 300 HotpotQA (bridge and comparison) | 5,300 | 1,566 | **3,734** | item |
| `d2_support` | customer-support questions across 12 industries and 24 fictional organisations, with matched pairs | 720 | 222 | **498** | `pair_id` |
| **pooled** | | 6,020 | 1,788 | **4,232** | stratified by dataset |

* **Split rule.** Deterministic, 30% dev and 70% test (`CONTRACTS.md` §7; `datasets/v2/*/splits.json`).
  * d1_open: sha256(id) mod 1000 < 300 → dev; no stratum needed rebalancing.
  * d2_support: a pair-preserving split, stratified by organisation and type.
* **Conditions.**
  * **Primary:** the ASR condition `piper-en_US-hfc_male-medium__150wpm__clean`, i.e. the Piper TTS voice at 150 wpm, clean, with live streaming-ASR partials.
  * **Secondary:** `text150`, a deterministic text-prefix timeline at 150 wpm.
* **Need label.** The empirical label from E1-v2: the closed-book answer to the full question is wrong in at least 3 of 5 draws, under the dataset's primary correctness metric (§2). d2_support's authored label is reported alongside it, never instead of it. d1_open has no authored label.

## 2. Correctness measurement (judge)

### 2.1 Primary metric per dataset

| dataset | primary correctness | secondary correctness |
|---|---|---|
| `d1_open` | **normalized alias match**: lower-case and strip punctuation, articles and whitespace; correct iff any normalized alias of `expected_answer` / `answer_aliases` is a substring of the normalized answer | **Secondary (decided by the PI before the freeze, O6):** both LLM judges (gpt-6.1-sol k = 1 and GPT-4o v4) grade a **stratified 400-answer subsample** of d1_open test answers (strata: item type × popularity tercile × alias verdict; seed 20261017). This mirrors the dev agreement check (`label.py --judge-subsample 400`; dev alias vs gpt-6.1-sol κ = 0.78). It is reported as an alias-vs-judge agreement check and as a sensitivity analysis of Δ[COMMIT] on that subsample. Estimated cost: under ₹100 of gpt-6.1-sol, inside the cap. |
| `d2_support` | **LLM judge (2.2)**, conditional on passing the gate (2.4) | GPT-4o v4 judge (2.3), on a 30% subsample |

### 2.2 Primary LLM judge: Azure OpenAI GPT-6.1-sol

| field | value |
|---|---|
| deployment | `gpt-6.1-sol`. A pool of two deployments of the identical model version is allowed for throughput (§4.8). |
| model version | `2026-09-29`, pinned (version upgrade policy: NoAutoUpgrade) |
| api-version | `2025-04-01-preview` |
| temperature | not sent. The service fixes it at 1; a reasoning model rejects any other value. |
| `reasoning_effort` | `medium` |
| `max_completion_tokens` | **2000** (`llm/experiment1_v2/judges.json`). The budget covers reasoning tokens. The < 1% truncation check is open item O3. |
| `seed` | 20261015, sent per call. It is best-effort only: recorded but not relied on. |
| prompt (rubric) | **`e1v2-r3`** (core facts; `llm/experiment1_v2/rubrics/e1v2-r3/`). sha256 `83fde728c0480fd77b146259f6360dd1029befb9904f201b049b4b39a7ccbc23`. This is the second and last dev revision allowed by §2.4. |
| sampling | **k = 1 sample per answer**, decided 2026-10-16 before the freeze to meet the compute budget. If the sample does not parse, up to 3 more are drawn and the first that parses is used. Dev justification: under r3 the 3 samples agreed on 98.8% of answers (Fleiss' κ 0.977). |

**Recording requirement.** For every call, store:
* the exact request body sent, including any parameters the SDK adds;
* the response id and finish reason;
* the model version reported by the service;
* the usage fields, including reasoning tokens;
* the raw output.

A parameter that is rejected or silently dropped is a recorded fact, not an error to work around.

**Non-determinism.** The judge is stochastic, because temperature is fixed at 1 and the seed is best-effort. Re-grading may not reproduce identical verdicts. The stored verdict is the record of what was graded. This is a stated limitation (§10).

### 2.3 Pre-specified secondary judge: GPT-4o v4

* **Model.** `azure-openai/gpt-4o@2025-01-01-preview`, cache version `v4-azure-reasoned` (reasoned+strip prompt), temperature 0 (deterministic). This is the judge validated in v1.
* **Subsample.** It grades a **stratified random 30% subsample** of all judge-graded test answers:
  * strata: dataset × item_type × primary verdict;
  * fixed seed **20261016** (`SECONDARY_SEED` in `simulator/e3v2_phase_a.py`).
* **Use.** On that subsample, all primary v2 results are recomputed under both judges and compared, as a robustness check. A conclusion that holds under one judge and not the other is reported as judge-dependent.

### 2.4 Judge-validation gate (before any TEST GRADING)

**Validation set.** About 300 Mistral-7B answers from **dev items only**:
* stratified by dataset (100 d1 / 200 d2), path (retrieve / skip) and `item_type`;
* each labelled independently by two humans who are blind to path and judge;
* disagreements adjudicated by a third.

The set is split before any judge sees it: cases with sha256(case_id) mod 3 = 0 are **prompt-development** (about 100), and the rest are the **gate holdout** (about 200). This is `simulator/judge_validation/score_judges.py --v2-labels`.

**Gate metrics, in order, all on the holdout:**
1. **Human inter-annotator agreement.** Cohen's κ_human is reported first. If κ_human < 0.70, the labelling guideline is revised and the set re-labelled before the judge is assessed.
2. **Judge (k = 1 verdict of record) vs adjudicated human labels.** Pass requires **all** of:
   * Cohen's κ ≥ 0.80, with lower 95% bootstrap bound ≥ 0.70;
   * accuracy ≥ 0.90;
   * |error rate on retrieve-path answers − error rate on skip-path answers| ≤ 0.05.
3. **Judge self-consistency.** On the validation set only, 3 samples are drawn per answer. The fraction of answers where all 3 agree must be ≥ 0.90, and Fleiss' κ ≥ 0.80. The verdict of record stays sample 0.

*Justification.* 0.80 is the conventional lower edge of "almost perfect" agreement. With n ≈ 200, the standard error of κ is about 0.03–0.04.

**Dev status.** Rubric revisions are exhausted: r3 is final. On dev, the judge's κ against the dev reference labels was 0.525 (r1), 0.704 (r2) and 0.737 (r3). Those reference labels are not the gate's human labels. **The human-labelled gate has not yet been run (open item O1).** Test generation may start at the freeze, but **test grading of d2_support waits until this gate passes** (§9).

**If the GPT-6.1-sol judge fails the gate.**
* The rubric may not be revised again, because r3 was the last allowed revision.
* The **GPT-4o v4 judge becomes primary** for d2_support, provided it passes the same gate. GPT-6.1-sol is then reported as secondary.
* If neither passes, the d2_support hypotheses become exploratory. d1_open is unaffected (alias match).

The GPT-4o v4 judge is put through the same gate in every case, and its gate results are reported.

## 3. Hypotheses

All quantities are computed on **test items** after the freeze. Dev values are predictions (§3.6).

### H-E1v2: signal validity on new data (gatekeeper)

* **Signal (operational belief and gate signal, one and the same).** **`self_assessed_need`** is P(“No”) to the frozen self-assessment prompt **`sa-v1-frozen`**: "answerable from general knowledge alone, without {org}'s own policies…? Yes/No", read from Mistral-7B's token probabilities.
  * **Calibration.** Fitted on dev per dataset: P(need) = sigmoid(a + b·logit(SA) + c·f + d·logit(SA)·f), where f is the spoken word fraction.
  * **Artifacts.** `controller/baselines/e3v2_calibration_{d1_open,d2_support}_mistral7b_v03_self_assessed_need_logit.json`.
  * **Belief transitions.** `controller/baselines/belief_transitions_v2_{d1_open,d2_support}.json`.
* **Primary.** Test AUC of the raw signal on the full question, for the empirical need label, per dataset.
  * **Success:** AUC ≥ 0.70.
  * **Kill:** AUC < 0.60 on both datasets. Then H-E3v2 becomes exploratory.
* **Prefix curve.** AUC at word_frac ∈ {0.3, 0.5, 0.7, 1.0}.
  * Prediction: non-decreasing, tested by the paired bootstrap of AUC(1.0) − AUC(0.3) > 0.
  * **C2 is "available" iff AUC(0.5) ≥ 0.65.** The flag is reported. It does not change any H-E3v2 test, because the frozen code has no channel ablation (§3.4).
* **Calibration (secondary).** Expected calibration error (10 bins) of the dev-fitted calibration, applied to test.
* **Boundary-mass check.** The fraction of test beliefs within ±0.15 of $b^{*}=\hat g^{-}/(\hat g^{+}+\hat g^{-})$ (dev ĝ). The prediction is ≥ 0.15.

### H-E2v2: timing value on live partials, per latency profile

* **Setup.** Primary ASR condition. The comparison is `b1_v2_forecast` (forecast-timed dispatch at a query-ready partial) against `b1_v2_sentence_end`. Both are unpriced, with retrieval fixed. It is run per latency profile: `rewrite_faiss`, `rewrite_faiss_rerank` and `rewrite_remote`. These are the composite profiles, query rewrite + retrieval; their medians are 1,044, 1,043 and 1,241 ms.
* **Measured.** The partial-query risk $\rho(t)=1-\text{hit@3}(t)/\text{hit@3}(\text{end})$, and the paired accuracy difference.
* **Success (v1 form retained).** Forecast-timed dispatch gives **lower p95 TTFA than sentence_end, at equal accuracy**, per profile.
  * "Equal accuracy" means the lower bound of the paired one-sided 97.5% CI for Δaccuracy is > −0.02.
  * The p95 comparison uses the paired cluster bootstrap (§5).

### H-E3v2: joint vs decomposed control

**Controllers** (as registered in `controller/sequential_v2.py` at the frozen commit). All share the dev-fitted belief calibration, the remaining-speech predictor, the latency model (including π), ĝ± and the prices.
* **Ours-v2** (`ours_v2`): the joint optimal-stopping controller, with the lookahead of §4.5.
* **B9+** (`b9plus_v2`): expected-exposure pricing. Its timing rule is tuned on dev over {early, late, R̂-threshold, R̂-below}.
* **B9** (`b9_v2`): the v1 definition.
* **SPEC** (`b10_speculative_v2`): always dispatches at its trigger, and decides at end of turn, from the final belief, whether to use the passages.
* **SPEC-θ** (`b11_speculative_gated_v2`): SPEC with a belief gate that uses dev-scalar exposures (THEORY §6.2).

**Architectures.** Both are tested on the same items, seeds, profile and price:
* **COMMIT:** a dispatched retrieval is always used.
* **SPECULATIVE:** a dispatched retrieval may be discarded at end of turn.

**Rivals (filled from `v2_predictions.json`, primary cell).**
* **D\*[COMMIT] = B9+ (`b9plus_v2`)**, both datasets.
* **D\*[SPECULATIVE]** is the dev-best of {B9+, SPEC, SPEC-θ}, chosen per dataset:
  * **d1_open: `b9plus_v2`.** Dev utility 0.3398, vs SPEC 0.3325 and SPEC-θ 0.3325.
  * **d2_support: `b10_speculative_v2` (SPEC).** Dev utility 0.35795, which ties SPEC-θ exactly (0.35795) and beats B9+ (0.35740). The tie is resolved by the frozen candidate order (B9+, SPEC, SPEC-θ), as `boundary_contrasts` does.

**Outcome per item.** Realized utility, averaged over the 5 seeds:
$$U=Q-\lambda_l\cdot\Delta\text{TTFA}/1000-\lambda_c\cdot n_{\text{retrievals}},$$
where Q is the primary correctness and ΔTTFA is the measured TTFA increase in ms over the same item's skip cell.

* **Δ[arch]** is the mean paired difference $\bar D(\text{Ours-v2[arch]}-D^{*}[\text{arch}])$ over test items.
* **Pooling.** The pooled estimate weights datasets by their test items: 3,734 : 498.

### 3.1 The primary cell

* **Profile: `rewrite_faiss`.** This is the canonical live-partial operation: query rewrite plus FAISS, and the configured default profile.
* **Price: the 0.65 dev end-of-turn belief-quantile point of the pre-registered grid (§4.3).** Its exact values are:

| dataset | λ_l (per s) | λ_c | belief cut b_q |
|---|---|---|---|
| d1_open | **0.17305735434823866** | **0.05** | 0.5828 |
| d2_support | **0.2157823714003309** | **0.05** | 0.8367 |

* **How it was chosen.** It is a PI decision taken before the freeze.
  * The §4.3 rule (maximize dev-predicted conjunction power) is degenerate: the dev power of the conjunction is 0 in every profile and at every price, because dev Δ[COMMIT] < 0 everywhere (`v2_power_dev.json`).
  * The §4.3 rule had no clause for this case, so the PI fixed the fallback now: the canonical profile at the 0.65 quantile, which was the grid point selected in the pre-dev planning.
  * The choice was made **without reference to Δ**. Every one of the 12 pooled cells has dev Δ[COMMIT] between −0.040 and −0.048.

### 3.2 Primary: the boundary test (co-primary P1, P2, P3), unchanged

At the primary cell, pooled, with all channels live (primary ASR condition):
* **P1:** Δ[COMMIT] > 0.
* **P2:** Δ[COMMIT] − Δ[SPECULATIVE] > 0, a paired interaction on the same items.
* **P3 (TOST equivalence):** −m₃ < Δ[SPECULATIVE] < m₃, with **m₃ = 0.016**.

**Why m₃ = 0.016, and why it replaces ½·δ̂_C.**
* **The draft rule is degenerate.** The draft margin m₃ = ½·max(δ̂_C,dev, 0) is 0 on dev because δ̂_C < 0, and an equivalence test with margin 0 is impossible to pass.
* **The new margin.** The PI therefore pre-specifies m₃ as a **smallest effect size of interest**: **0.016**, the dev minimum detectable effect for Δ at the planned test N.
  * At the primary cell, the MDE for P1 is 0.0155, and the TOST half-width needed for 80% power at zero true effect is 0.0156 (`v2_power_dev.json`).
  * Across all cells the MDE ranges from 0.015 to 0.017.
* **What it means.** It is the smallest joint-vs-decomposed difference this test N can resolve. P3 then asserts that, under speculation, the joint and decomposed controllers are equivalent within that resolution.
* **How it was fixed.** It does not depend on any dev estimate of Δ[SPECULATIVE].

**Decision rules.** These are four one-sided tests at α = 0.025, using the paired cluster bootstrap (B = 10,000; the same resamples for every contrast):
* P1;
* P2;
* P3's upper side ($H_0:\delta_S\ge m_3$);
* P3's lower side ($H_0:\delta_S\le-m_3$).

**The boundary claim is supported iff all four reject.** This is an intersection–union test, so the conjunction needs no multiplicity adjustment.

### 3.3 Added confirmatory hypothesis H-REV (pre-freeze, PI decision)

* **H-REV:** **Δ[COMMIT] < 0** at the primary cell, pooled. That is, the strong decomposed controller B9+ beats the joint Ours-v2 under commit.
* **Test.** One-sided α = 0.025, with the same paired cluster bootstrap as §3.2. It rejects iff the 97.5th percentile of $\bar D_C^{*}$ is < 0.
* **Dev prediction (primary cell, pooled over 1,788 dev items):** Δ[COMMIT] = **−0.041, SE 0.0085**.
  * Per dataset: d1_open −0.045 (SE 0.009); d2_support −0.009 (SE 0.019).
  * Across all 12 profile × price cells: −0.040 to −0.048.
* **Why it was added.** It states the dev-predicted direction confirmatorily. It does not replace P1, and both are reported.

**Multiplicity for individual claims.** P1, P2, P3 (at the larger of its two one-sided p-values) and H-REV form one Holm family, at family-wise one-sided α = 0.025. The IUT for the boundary claim is unchanged.

### 3.4 Headline estimate and secondary analyses

* **Headline (reported whatever the test outcomes).** Δ[COMMIT] at the primary cell, with its 95% paired-cluster-bootstrap CI, reported:
  * per dataset (d1_open, d2_support);
  * pooled.

  Δ[SPECULATIVE] and the interaction are reported in the same form.
* **Secondary family (two-sided, Holm-adjusted at family-wise α = 0.05).** Per dataset at the primary cell: Δ[COMMIT], Δ[SPECULATIVE] and the interaction, which makes 6 tests.
  * They are two-sided because the theory's direction (> 0 under commit) and the dev direction (< 0) disagree.
* **Descriptive (no tests).**
  * Every other cell: dataset × condition {ASR, text150} × profile {rewrite_faiss, rewrite_faiss_rerank, rewrite_remote} × price {0.5, 0.65, 0.8, 0.9} × architecture;
  * Ours-v2 vs B9;
  * the two-epoch theory diagnostics $m_{\text{prophet}}$ and $m_{C1}$;
  * q̂ and λ_c(1+w̄)q̂, which are labelled diagnostics and not bounds.
* **Predicted-vs-observed agreement.** This covers all 48 dev cells of `v2_predictions.json` (2 datasets × 2 conditions × 3 profiles × 4 prices), for Δ[COMMIT].
  * (a) The Spearman correlation r_s(δ̂_dev, δ_test) ≥ 0.5, with one-sided 95% bootstrap lower bound > 0. The bootstrap resamples test items and recomputes every cell.
  * (b) The slope of the inverse-variance-weighted regression of δ_test on δ̂_dev has a 95% bootstrap CI that contains 1 and excludes 0.

  Verdict: **agree** if both hold; **directional** if only (a) holds; **disagree** otherwise.
* **Frontier dominance (comparability with v1).** The v1 rule is applied unchanged, per dataset × profile, primary condition: Ours-v2 vs B9, and Ours-v2 vs B9+.
* **Removed before freeze.** The channel-set ablation cells of the earlier draft are not run: the A1–A3-enforced "none" replication, C1-only, and C1+C2.
  * The reason is that the frozen code (`ddde67f`) does not implement them.
  * Proposition 1 is instead verified on the v1 data (THEORY §8: 1,250/1,250 and 10,000/10,000 decisions).

### 3.5 Pre-registered interpretation of outcomes

* **P1 failing** means joint control did not pay under commit at this scale. If H-REV holds, the strong decomposed controller is better. This is reported as a finding against the joint hypothesis, not as a null.
* **P3's lower side failing** means Ours-v2[SPEC] is worse than D\*[SPEC] by more than m₃. It is reported, and the unit tests at the frozen commit are re-run and reported. **The frozen code is not changed.**
* **P3's upper side failing** means the joint controller keeps a speculative advantage greater than m₃ (THEORY §6.2).
* Ties are reported as ties.

### 3.6 Dev-predicted outcomes (stated before the freeze)

Pooled dev at the primary cell (`v2_predictions.json`). The power column is at test N = 4,232 (`v2_power_dev.json`; normal approximation from the dev means and SEs scaled to test N).

| hypothesis | dev estimate (SE) | predicted test SE | predicted outcome |
|---|---|---|---|
| P1: Δ[COMMIT] > 0 | −0.0409 (0.0085) | 0.0055 | **fails** (power 0) |
| P2: interaction > 0 | +0.0036 (0.0037) | 0.0024 | **uncertain** (power 0.31; MDE 0.0068) |
| P3: \|Δ[SPEC]\| < 0.016 | Δ[SPEC] = −0.0445 (0.0085) | 0.0056 | **fails on the lower side**: the predicted mean is 0.029 below −m₃, about 5 test SEs (power ≈ 0). The upper side passes. |
| boundary claim (IUT) | | | **fails** (P1 and P3 fail) |
| H-REV: Δ[COMMIT] < 0 | −0.0409 (0.0085) | 0.0055 | **likely holds** (z ≈ −7.5; power ≈ 1) |
| H-E1v2 success (full-question AUC ≥ 0.70) | d1 0.69, d2 0.70 | — | **borderline** |
| H-E1v2 kill (AUC < 0.60 on both) | | | **not expected** |
| H-E1v2 C2 flag (AUC(0.5) ≥ 0.65) | d1 0.56, d2 0.58 | — | **not available** |
| H-E1v2 boundary mass | d1 ĝ⁻ = −0.11 on ASR, so b* < 0 | — | **fails on d1**: retrieval did not harm non-needed d1 items on dev |

**Why P1 fails on dev.** The lab log of 2026-10-16 gives the action counts. Ours-v2 retrieves on 99% of d1 ASR turns and dispatches earlier than B9+, on partial queries that retrieve worse. Its accuracy is 0.438, against 0.500 for B9+, at a similar p95 TTFA.

**Agreement with THEORY.** THEORY predicted P1 > 0 when C1 and C2 are active. The dev data contradict that prediction for this belief signal, whose out-of-fold calibrated AUC is about 0.57 (d1) and 0.60 (d2). Whichever way the test falls, it is reported.

## 4. Analysis plan

### 4.1 Units, replicates, resampling
* **Unit of analysis:** item. Seeds are within-item replicates; each item's outcome is the mean over its 5 seeds.
* **Paired cluster bootstrap:** B = 10,000. It resamples items (d1) or `pair_id` clusters (d2), with all controllers and both architectures evaluated on the same resample. The pooled analysis resamples within dataset (stratified) and weights datasets by their test items.
* **CIs:** percentile CIs. p95 TTFA is recomputed per resample.

### 4.2 Multiplicity
* **Boundary claim:** one intersection–union test (P1 ∧ P2 ∧ P3: four one-sided tests at α = 0.025).
* **Individual confirmatory claims:** Holm over {P1, P2, P3, H-REV}, at family-wise one-sided α = 0.025.
* **Secondary family:** 6 two-sided tests, Holm at α = 0.05 (§3.4).
* **Separate experiments:** H-E1v2 and H-E2v2, each with its own criteria, as in v1.
* Everything else is descriptive.

### 4.3 Price grid and primary-cell selection (dev only)
* **Price grid.** λ_c = 0.05. λ_l is set at the dev end-of-turn belief quantiles q ∈ {0.5, 0.65, 0.8, 0.9}:
  $\lambda_l=(b_q(\hat g^{+}+\hat g^{-})-\hat g^{-}-\lambda_c)/((E[L]+\pi)/1000)$, per dataset and profile.

  Values, primary condition (`v2_predictions.json`):

| dataset × profile | q = 0.5 | 0.65 | 0.8 | 0.9 |
|---|---|---|---|---|
| d1_open × rewrite_faiss | 0.1657 | **0.1731** | 0.1807 | 0.1889 |
| d1_open × rewrite_faiss_rerank | 0.1676 | 0.1750 | 0.1827 | 0.1910 |
| d1_open × rewrite_remote | 0.1474 | 0.1540 | 0.1607 | 0.1681 |
| d2_support × rewrite_faiss | 0.2053 | **0.2158** | 0.2245 | 0.2337 |
| d2_support × rewrite_faiss_rerank | 0.2076 | 0.2182 | 0.2270 | 0.2363 |
| d2_support × rewrite_remote | 0.1826 | 0.1920 | 0.1997 | 0.2079 |

* **Selection rule.** Choose the (profile, price) that maximizes dev-predicted conjunction power at the planned test N.
* **Pre-freeze fallback (PI, 2026-10-16).** If that power is 0 at every cell, use `rewrite_faiss` at q = 0.65. This applied (§3.1).
* **No test data are used.**

### 4.4 Exclusions and failed runs
* **Failed runs.** A run that crashes or times out is retried up to 3 times with the same seed. If it still fails, the item is **excluded from all controllers and both architectures** (pairwise complete) and counted.
* **Unusable audio.** Items whose ASR final transcript is empty, or whose speech trace violates `CONTRACTS.md` §4, are excluded from all controllers and counted.
* **Judge parse failures.** If the sample does not parse, up to 3 more are drawn. If none parses, the cell is excluded in the primary analysis and counted incorrect in a sensitivity analysis.
* **Flagging.** A cell with exclusions above 5% is flagged and reported with and without the exclusions.
* **No outcome-based exclusions.** No item is excluded because of its outcomes.

### 4.5 Tuning protocol (dev only; complete)
* **Shared estimates.** Fitted once on dev and shared by every controller:
  * the belief calibration and transitions;
  * the remaining-speech predictor;
  * the latency model, including π (`nonhideable_increment_ms` 371, streaming TTS);
  * ĝ± per dataset.
* **Per-controller parameters.** These were tuned on dev with an equal number of evaluations:
  * the B9+ timing rule: early / late / R̂-threshold / R̂-below, each threshold on a 21-point dev quantile grid. R̂-below was added 2026-10-16 before the freeze, because without a late-partial rule the family had a gap that inflated Δ[COMMIT] on dev ASR (`paper/lab_log.md`);
  * the shared SPEC / SPEC-θ trigger: immediate, forecast, or R̂-below on the same grid;
  * the SPEC-θ scalars, which are the dev means at the firing step and are estimates, not tuned.

  Ours-v2 has no free parameter beyond the shared models and its lookahead depth (frozen in `configs/config.yaml` at `ddde67f`).
* **Ours-v2 lookahead** (corrected 2026-10-16 before the freeze; its implementation corrected once more the same day for a code-level defect; PI decisions; `paper/lab_log.md`).
  * **Information channel.** From a step, the next partial carries new information with probability p = P(R > s).
    * R is the remaining speech under the predictor's q10/q50/q90, aged to the step's availability time and not clipped.
    * s is the dev-median spacing between partials, measured by availability time.
  * **Post-speech partials** carry no information: their belief transition is the identity.
  * **End-of-turn gap.** It enters only the exposure term, never p.
  * **text150.** p = 1 while a prefix is still to come, and 0 after the last prefix.
  * **Value of waiting.** It is p·(next-partial value) + (1 − p)·(end-of-turn decision value), in both architectures.
  * **Commitment.** No parameter was selected by looking at Δ on dev, and no further change will be made (hard pre-commitment, lab log 2026-10-16).
* **Cost calibration.** Latency is the **end-to-end TTFA increment** measured on dev (THEORY §8.4).

### 4.6 What may and may not change after freezing
* **May change** (each logged as a deviation): infrastructure; re-running failed items; descriptive analyses labelled as such. A bug fix may change only analysis or infrastructure code; any change that would alter a controller's decision on any item is forbidden.
* **May not change:**
  * the controller code (frozen at `ddde67f`);
  * hypotheses P1–P3 and H-REV, the IUT, the Holm families, α, and m₃ = 0.016;
  * the primary cell (`rewrite_faiss`, q = 0.65; λ_l 0.17305735434823866 for d1, 0.2157823714003309 for d2; λ_c 0.05) and the price grid;
  * D\*[COMMIT] = B9+;
  * D\*[SPECULATIVE] = B9+ for d1_open and SPEC for d2_support;
  * the judge configuration (§2.2, §2.3, including seeds and rubric hash);
  * the exclusion rules, splits, conditions, model and seeds;
  * the dev predictions file (`v2_predictions.json`).

### 4.7 Deviation logging
Every deviation is recorded before the affected analysis is run, in two places:
* `paper/lab_log.md`, with the date, what changed, why, who decided, and whether any test outcome had been seen;
* a "Deviations" section appended to this document. That section is the only part that may be appended after the freeze.

### 4.8 Compute-saving implementation rules (decided 2026-10-16 before the freeze; results-invariant)
* **Lazy outcome evaluation.** Controller decisions depend only on beliefs, the timing state and the latency model, never on answer correctness.
  * So Phase B runs first on test. Only (item, seed, condition, query_variant) cells chosen by at least one pre-registered controller, at at least one pre-registered (profile, price), are then generated and graded.
  * The cell list is written to disk before any grading.
* **E1-v2 on test.** `self_assessed_need` is computed on every text prefix and every ASR partial. Closed-book entropy signals are computed on the full question only.
* **Throughput only.** These changes alter no model, prompt or parameter:
  * more LLM containers;
  * higher client concurrency;
  * a two-deployment pool of the identical judge model version.
* **Budget.** The test-split budget estimate (`docs/research_v2/v2_test_budget.json`):
  * about 240,600 LLM calls;
  * judge cost about ₹3,960 (primary, k = 1) + ₹1,270 (secondary, 30%), d2_support only.

## 5. Statistical details
* **Primary statistic:** $\bar D$, the mean over test items of the seed-averaged paired utility difference.
* **P1:** reject $H_0:\delta_C\le0$ iff the 2.5th percentile of $\bar D_C^{*}$ is > 0.
* **H-REV:** reject $H_0:\delta_C\ge0$ iff the 97.5th percentile of $\bar D_C^{*}$ is < 0.
* **P2:** the P1 rule applied to the per-item difference $D_{C,i}-D_{S,i}$, on items present in both architectures.
* **P3 (TOST):** the 2.5th–97.5th percentile interval of $\bar D_S^{*}$ lies inside (−0.016, 0.016). m₃ is a fixed constant and is not re-estimated.
* **Holm:** one-sided p-values come from the bootstrap. P3's p-value is the larger of its two sides.
* **p95 comparisons:** the paired cluster bootstrap of the difference in p95.

## 6. Power

### 6.1 Dev-based power (of record)
**Source:** `docs/research_v2/v2_power_dev.json` (`analysis/v2_power_dev.py`). It uses the dev Phase B per-item paired differences on the primary condition, scaled to test N = 3,734 + 498 = 4,232.

**At the primary cell** (pooled means weighted by test share):

| contrast | mean | SE at test N | MDE (80%) | power |
|---|---|---|---|---|
| P1: Δ[COMMIT] | −0.0411 | 0.0055 | 0.0155 | 0.00 |
| P2: interaction | +0.0036 | 0.0024 | 0.0068 | 0.31 |
| P3: Δ[SPEC] (TOST half-width needed: 0.0156) | −0.0447 | 0.0056 | — | ≈ 0, lower side |
| H-REV | −0.0411 | 0.0055 | — | ≈ 1.00 |
| conjunction | | | | 0.00 |

**Across all 12 profile × price cells:**
* P1 and the conjunction have power 0;
* P2 has power 0.10–0.65;
* the MDE for Δ[COMMIT] is 0.015–0.017, and for the interaction 0.006–0.007.

The normal approximation was checked against a Wald test on 200 resampled test sets at rewrite_faiss, q = 0.5: both give conjunction power 0.

### 6.2 Pre-dev planning simulation (superseded; kept for the record)
`analysis/v2_power.py` → `docs/research_v2/v2_power.json` holds the simulation used before dev data existed. It used:
* v1 item effects;
* synthetic beliefs and timing;
* placeholder latencies.

It predicted Δ[COMMIT] ≈ +0.009 to +0.011 and the need for SPEC-θ as a rival (THEORY §6.2). Dev reversed the sign of Δ[COMMIT]. The simulation is not used for any decision in this addendum.

## 7. Reporting
* Every hypothesis is reported with its verdict, the point estimate, the CI and the exclusion counts.
* The headline estimate of §3.4 is always reported.
* Ties are reported as ties.
* Every d2_support primary result is also reported under the secondary judge, on its 30% subsample (§2.3).
* v1 and v2 results appear in separate tables.
* The dev-predicted outcomes (§3.6) are shown next to the test outcomes.

## 8. Freeze checklist (sign-offs)

| item | status | owner | sign-off |
|---|---|---|---|
| Dataset sizes and splits (§1) | **done** (`datasets/v2/*/splits.json`) | Yashmita | Ryan |
| E1-v2 dev: signal, calibration, prefix curve, ĝ± (§3) | **done** (lab log 2026-10-16; calibration artifacts) | Yashmita | Ryan |
| Judge configuration filled (§2.2, §2.3) | **done** (truncation check O3 open) | Yashmita | Ryan |
| Human judge-validation gate (§2.4) | **OPEN (O1)**. Not required for the freeze; required before test grading. | Yashmita | Ryan |
| Controller code frozen; commit recorded | **done** (`ddde67f`) | Yashmita | Ryan |
| Primary cell, price grid, D\*, m₃, H-REV written in (§3, §4.3) | **done** (PI decisions 2026-10-16) | Renya (grid) | Ryan |
| Dev predictions and dev power written (§3.6, §6.1) | **done** (`v2_predictions.json`, `v2_power_dev.json`) | Yashmita | Ryan |
| All runners refuse `--split test` without `PREREG_FROZEN` (CONTRACTS §7) | guard code present; run-time check at freeze (O4) | Yashmita | Ryan |
| d1_open secondary judge decided (§2.1) | **DECIDED (O6):** both judges on a stratified 400-answer subsample | Ryan (PI) | Ryan |
| Final read of this document | open | Ryan (PI) | Ryan |

## 9. Freeze procedure (instructions; NOT executed)

Run these only after every item in §8 is signed off, with two exceptions. O1 (human judge gate) does not block the freeze; it blocks test grading. O2 (authored-label review) does not block either. O6 is decided (§2.1).

1. **Check the frozen code.** Confirm that the controller, simulator and analysis code equal commit `ddde67f`:
   `git diff --stat ddde67f HEAD -- controller simulator analysis/e3_frontier.py analysis/v2_power_dev.py configs/config.yaml`
   must print nothing.
2. **Update the header.** Change the status line at the top to `FROZEN`, and add the freeze date and the names of the signers.
3. **Commit the addendum:** `git commit -m "Freeze v2 pre-registration addendum"`. Record the full hash (`git rev-parse HEAD`).
4. **Create the sentinel file** `docs/research_v2/PREREG_FROZEN`, containing:
   * the hash from step 3 and the frozen code commit `ddde67fdae9b80d80e295f20da45efe9b0dc2540`;
   * the UTC timestamp;
   * the sha256 of `PREREG_V2.md`, `v2_predictions.json` and `v2_power_dev.json`;
   * the signers.

   Commit it.
5. **Tag** the sentinel commit: `git tag -a prereg-v2-frozen -m "v2 pre-registration frozen"`. Push the branch and the tag.
6. **Register on OSF** using the "OSF Preregistration" template.
   * Attach `PREREG_V2.md`, `v2_predictions.json`, `v2_power_dev.json` and `THEORY.md`.
   * Give the commit hashes and the tag name.
   * Record the DOI and URL in `PREREG_FROZEN` and in `paper/lab_log.md`, with a commit.
7. **Start test generation immediately after the freeze:**
   * E1-v2 signals;
   * Phase B decisions (lazy cell list written to disk);
   * Phase A generation of the chosen cells.

   d1_open is graded by alias match and can be scored at once.
8. **Test GRADING of d2_support waits until the human judge-validation gate (§2.4) passes** with the GPT-6.1-sol judge, or with its fallback. The gate result is committed before any d2_support test answer is sent to a judge.
9. No analysis of test outcomes is run until step 8 is complete for every dataset it uses.

## 10. Limitations stated in advance
* **Single model.** All v2 results are for Mistral-7B-Instruct-v0.3.
* **Weak operational belief.** `self_assessed_need` has dev full-question AUC of about 0.69–0.70, and out-of-fold calibrated AUC of about 0.57 (d1) and 0.60 (d2). Coupling's value scales with belief quality (THEORY Remark 2.1; v1 lab log). A negative Δ[COMMIT] is a statement about joint control *with this belief*.
* **Judge non-determinism.** The primary judge is a reasoning model run at temperature 1 with a best-effort seed and k = 1. Dev unanimity across 3 samples was 98.8%, so the residual instability is small but not zero.
  * Mitigations: all outputs are stored, self-consistency is a gate metric, and a deterministic GPT-4o v4 judge re-grades a 30% subsample.
* **Judge gate pending at the freeze** (O1). Grading waits for it.
* **Synthetic speech.** The primary condition is a single synthetic voice (Piper `en_US-hfc_male-medium`, 150 wpm, clean) with one streaming-ASR configuration.
* **d2_support labels.** The authored labels are pending human verification (`datasets/v2/d2_support/verification_sample.csv`). This affects only the authored-label analyses; the empirical label is used for confirmatory tests.
* **No channel ablations.** The frozen code cannot isolate C1, C2 or C3 (§3.4).

## 11. Open items (not filled; not invented)
* **O1 — human judge-validation gate (§2.4).** The double-labelled validation set, κ_human, judge κ / accuracy / path-error balance and self-consistency have not been run. This blocks d2_support test grading, not the freeze.
* **O2 — d2_support authored-label verification.** `verification_sample.csv` is unreviewed. It does not block the freeze.
* **O3 — truncation check for `max_completion_tokens` = 2000.** The dev rate of finish_reason = `length` has not been reported in a committed artifact. The value 2000 is fixed regardless. The observed rate is reported, and if it is ≥ 1% the affected verdicts are treated as parse failures (§4.4).
* **O4 — test-split guard.** The guard code is present in `simulator/e3v2_decide.py`, `simulator/e3v2_phase_a.py`, `simulator/e3v2_lazy.py` and `llm/experiment1_v2/common.py` (checked by reading). Before freeze step 7, confirm at run time that each refuses `--split test` without `PREREG_FROZEN`.
* **O5 — OSF registration** (DOI and URL): at freeze step 6.
* **O6 — d1_open secondary judge: DECIDED** (2026-10-16, PI). Both judges grade a stratified 400-answer d1_open test subsample (§2.1).
