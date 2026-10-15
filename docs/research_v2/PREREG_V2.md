# Pre-Registration Addendum v2 — DRAFT (NOT FROZEN)

> **Status: DRAFT.** Nothing in this file binds until the freeze procedure in
> §9 is complete. Until then the test split stays sealed (`CONTRACTS.md` §7):
> no generation, signal extraction, tuning or judging may touch test items.
> Owner: Yashmita. Sign-off: Ryan (PI). Price-grid confirmation: Renya.

## 0. Relationship to v1

1. **v1 results stand as originally reported.** `paper/preregistration.md`
   (frozen 2026-08-21) and its criteria and outcomes are not modified by this
   addendum:
   * E1: PASS (AUC 0.9784).
   * E2: p95 criterion NOT MET.
   * E3: 0/5 seeds of dominance; reported as a tie.

   v1 answers remain graded by the GPT-4o judge (v4 reasoned+strip, temperature 0)
   exactly as reported.
2. **v2 is a new confirmatory study.** It tests predictions derived *after* v1
   (`docs/research_v2/THEORY.md`) on new datasets, new channels and a new test
   split that nothing has touched. No v1 data enter any v2 confirmatory test.
   v1 per-item outcomes are used only for the power analysis (§6).
3. **Model.** v2 uses exactly one generator: **`mistralai/Mistral-7B-Instruct-v0.3`**,
   served unquantized (bf16) with vLLM, as in v1. There is no second model.
   * The v1 note on substituting Llama-3.1-8B with Mistral-7B is a historical
     fact about v1 only.
   * All v2 claims are about Mistral-7B. Generality across models is untested
     and is stated as a limitation (§10).
   * Generation and signal-extraction settings are those in `configs/config.yaml`
     and `llm/experiment1/config.json` at the freeze commit. Seeds: 11, 23, 37, 53, 71.

## 1. Datasets and splits

| dataset | content | unit | cluster for resampling |
|---|---|---|---|
| `d1_open` | PopQA (popularity-stratified) + HotpotQA | item | item |
| `d2_support` | 12-domain customer-support benchmark with matched pairs; fictional organisations | item | `pair_id` |

* **Split rule.** 30% dev / 70% test, deterministic: sha256(id) mod 1000 < 300 → dev. Stratified by popularity bin (d1) or `item_type` (d2), with pair partners in the same split (`CONTRACTS.md` §7).
* **Planned sizes.** [TO FILL AT FREEZE, from the dataset READMEs]. §6 shows the confirmatory test needs about **4,000 pooled test items**. The recommended size is `d1_open` ≥ 5,000 items in total (≥ 3,500 test).
* **Need label.** The empirical label comes from E1-v2, defined as the closed-book answer being incorrect in at least 3 of 5 draws under the dataset's primary correctness metric (§2). The authored label (`retrieval_needed_authored`) is reported alongside it, never instead of it.

## 2. Correctness measurement (judge)

### 2.1 Primary metric per dataset

| dataset | primary correctness | secondary correctness |
|---|---|---|
| `d1_open` | **normalized alias match**: lower-case, strip punctuation, articles and whitespace; correct iff any normalized alias of `expected_answer`/`answer_aliases` is a substring of the normalized answer | LLM judge (2.2), then GPT-4o v4 (2.3) |
| `d2_support` | **LLM judge (2.2)**, conditional on passing the gate (2.4) | GPT-4o v4 judge (2.3) |

### 2.2 Primary LLM judge: Azure OpenAI GPT-6 family

| field | value |
|---|---|
| deployment | `gpt-6.1-sol` |
| model version | `2026-09-29`, pinned (version upgrade policy: NoAutoUpgrade) |
| api-version | `2025-04-01-preview` |
| temperature | 1 (fixed by the service; a reasoning model rejects 0) |
| `reasoning_effort` | `medium` |
| `max_completion_tokens` | [TO FILL AT FREEZE; fixed from the dev pilot so that < 1% of calls truncate] |
| `seed` | sent per call, best-effort only; recorded but not relied on |
| prompt | [PROMPT VERSION ID AND SHA256 TO FILL AT FREEZE]; same rubric as the v4 reasoned+strip judge, adapted to the reasoning model |
| sampling | **k = 3 independent samples per answer; verdict = majority (2 of 3)**; all three raw outputs and parsed verdicts are stored |

**Recording requirement.** For every call, store the exact request body sent (all parameters, including any the SDK adds), the response id, the model version reported by the service, the usage fields and the raw output. A parameter that is rejected, or silently dropped, is a recorded fact, not an error to work around.

**Non-determinism.** Because temperature is fixed at 1, the judge is stochastic. Majority-of-3 reduces this but does not remove it, and the seed is best-effort. Re-grading may not reproduce byte-identical verdicts. This is a stated limitation (§10). The stored per-sample verdicts are the record of what was graded.

### 2.3 Pre-specified secondary judge: GPT-4o v4

`azure-openai/gpt-4o@2025-01-01-preview`, cache version `v4-azure-reasoned` (reasoned+strip prompt), temperature 0 (deterministic). This is the judge validated in v1 (`simulator/judge_validation/`).

* **All primary v2 results are re-reported under it** as a robustness check.
* A conclusion that holds under one judge and not the other is reported as judge-dependent.

### 2.4 Judge-validation gate (before any test grading)

**Validation set.** About 300 Mistral-7B answers from **dev items only**:
* stratified by dataset (100 d1 / 200 d2), path (retrieve / skip) and `item_type`;
* each labelled independently by two humans blind to path and judge;
* disagreements adjudicated by a third.

The set is split, before any judge sees it, into **100 prompt-development** and **200 gate-holdout** answers, by sha256 of the answer id.

**Gate metrics.** In this order:

1. **Human inter-annotator agreement**: Cohen's κ, reported first. If κ_human < 0.70, the labelling guideline is revised and the set re-labelled before the judge is assessed, because a judge cannot be validated against labels less reliable than the bar it must clear.
2. **Judge vs adjudicated human labels** (on the 200 holdout, majority-of-3 verdict). Pass requires **all** of:
   * Cohen's κ ≥ 0.80, with the lower 95% bootstrap bound ≥ 0.70;
   * accuracy ≥ 0.90;
   * |error rate on retrieve-path answers − error rate on skip-path answers| ≤ 0.05.

   The last condition matters because a judge whose errors depend on the path biases every retrieve-vs-skip contrast. The v1 judge's errors all ran in one direction.
3. **Judge self-consistency**: the fraction of answers where all 3 samples agree ≥ 0.90, and Fleiss' κ across the 3 samples ≥ 0.80.

*Justification.* 0.80 is the conventional lower edge of "almost perfect" agreement. With n = 200 the standard error of κ is about 0.03–0.04, so the lower-bound requirement (0.70) is meaningful. The v1 GPT-4o v4 judge scored 95% on 19 hand labels, and the bar should not be lower for its replacement.

**If the GPT-6 judge fails.**
1. Its prompt may be revised **using only the 100 prompt-development answers**, at most twice. Each revision is re-assessed on the holdout, and every attempt is reported.
2. If it still fails, the **GPT-4o v4 judge becomes primary**, provided it passes the same gate. GPT-6 is then reported as secondary.
3. If neither passes, the d2_support accuracy-dependent hypotheses are downgraded to exploratory, and d1_open is unaffected (alias match).

The GPT-4o v4 judge is put through the same gate in every case, and its gate results are reported.

## 3. Hypotheses

All quantities are computed on **test items** after the freeze. Theory references point to `THEORY.md`.

### H-E1v2: signal validity on new data (gatekeeper)

* **Signal.** One signal and one prefix aggregation, chosen on dev and named at freeze: [SIGNAL TO FILL AT FREEZE].
* **Primary.** Test AUC of the chosen signal (full transcript) for the empirical need label, per dataset. **Success: AUC ≥ 0.70.** **Kill: AUC < 0.60 on both datasets** → all belief-driven v2 hypotheses (H-E3v2) become exploratory.
* **Prefix curve.** AUC at word_frac ∈ {0.3, 0.5, 0.7, 1.0} on ASR partials.
  * Prediction: non-decreasing in word_frac, tested by the paired bootstrap of AUC(1.0) − AUC(0.3) > 0.
  * **C2 is "available" iff AUC(0.5) ≥ 0.65.** If not, the C2 cells of H-E3v2 are predicted to be ≈ 0 and are tested as such.
* **Calibration (secondary).** Expected calibration error (10 bins) of the dev-fitted calibration, applied to test.
* **Boundary-mass check.** This tests the corrected-objective premise (THEORY §4). The fraction of test beliefs within ±0.15 of $b^{*}=\hat g^{-}/(\hat g^{+}+\hat g^{-})$, with $\hat g^{\pm}$ estimated on dev, is reported. The prediction is ≥ 0.15. v1 had 0.08.

### H-E2v2: timing value on live partials, per latency profile

* **Setup.** Live ASR partials (no frozen query). Early dispatch at the first meaningful partial, gated by `query_ready` and the remaining-speech predictor, versus `sentence_end`. Run per latency profile: faiss, faiss_rerank, remote_api.
* **Measured.** The partial-query risk $\rho(t)=1-\text{hit@3}(t)/\text{hit@3}(\text{end})$ and the paired accuracy difference.
* **Success (v1 form retained).** Early dispatch gives **lower p95 TTFA than sentence_end, at equal accuracy**, per profile.
  * "Equal accuracy" means non-inferiority: the lower bound of the paired one-sided 97.5% CI for Δaccuracy is > −0.02.
  * The p95 comparison uses the paired cluster bootstrap (§5).
* **Prediction.** The p95 gain increases with profile latency (faiss < faiss_rerank < remote_api). For faiss, most calls are already hidden, so the gain is small.

### H-E3v2: joint vs decomposed under active channels

**Controllers.** All controllers share the dev-fitted belief calibration, the remaining-speech predictor, the latency model, $\hat g^{\pm}$ and the prices.

* **Ours-v2** (joint, THEORY §5–6).
* **B9+**: expected-exposure pricing, with its timing rule tuned on dev over {early, late, R̂-threshold}.
* **B9**: v1 definition.
* **SPEC** (B10): always dispatch at the first meaningful partial, then decide at end of turn, with the final belief, whether to use the passages. Every dispatch pays its retrieval compute. A discarded retrieval adds no quality and no non-hideable TTFA. SPEC is decomposed: the dispatch rule never reads the belief, and the use rule never reads the timing state.

**Architectures (decided by the PI, 2026-10-16): the study tests BOTH, as a boundary test.**
* **COMMIT:** a dispatched retrieval's passages are always used.
* **SPECULATIVE:** a dispatched retrieval may be discarded at end of turn.

Both are run on the same test items, seeds, latency profile and price. The architecture is the only thing that changes.
* **Ours-v2[arch]** is the joint controller for that architecture.
  * Under COMMIT, it is THEORY §5–6.
  * Under SPECULATIVE, it is the same optimal-stopping controller with the dispatch value replaced by the option value of having the passages ready, $E[\max(U_{\text{use}}(b_{\text{end}}),0)]$, less the dispatch compute.
* **D\*[arch]** is the strongest decomposed rival for that architecture:
  * D\*[COMMIT] = B9+;
  * D\*[SPECULATIVE] = whichever of {B9+, SPEC} has the higher dev expected utility at the primary price.
  * Both are named at freeze.
* **Δ[arch]** = $\bar D(\text{Ours-v2[arch]} - D^{*}[\text{arch}])$.

**Outcome per item.** Realized utility under the corrected objective, averaged over the 5 seeds:

$$U=Q-\lambda_l\cdot\Delta\text{TTFA}-\lambda_c\cdot n_{\text{retrievals}},$$

where $Q$ is the primary correctness and ΔTTFA is the measured TTFA increase over the same item's skip cell. The pairwise estimand is the mean paired difference $\bar D$ over test items.

**Primary: the boundary test (three co-primary hypotheses).**

Setup:
* all channels active (C1+C2+C3);
* pooled over d1_open and d2_support test items (item-weighted);
* at **one latency profile and one price**, the same for both architectures, fixed at freeze by the §4.3 rule applied to Δ[COMMIT] on dev: [PROFILE, PRICE TO FILL AT FREEZE].

The three hypotheses follow from THEORY Prop. 4 and Prop. 4′:
* **P1 (joint control pays when retrieval is committed):** Δ[COMMIT] > 0.
* **P2 (the boundary itself):** Δ[COMMIT] − Δ[SPECULATIVE] > 0, a paired interaction contrast on the same items.
* **P3 (speculative prefetch closes the gap):** Δ[SPECULATIVE] < m.
  * m = λ_c·(1 + w̄)·q̂ is the Prop. 4′ bound: w̄ is the dev mean waste, and q̂ is the dev fraction of SPEC dispatches whose passages end up unused.
  * m is computed on dev and fixed at freeze: [m TO FILL AT FREEZE].
  * This is a non-superiority test. It is passed if the 97.5th percentile of the bootstrap distribution of Δ[SPECULATIVE] is < m.

How the tests are decided:
* All three use the paired cluster bootstrap (B = 10,000, the same resamples for every contrast), one-sided.
* **The boundary claim is supported iff P1, P2 and P3 all hold, each at one-sided α = 0.025.** This is an intersection–union test, so the conjunction needs no multiplicity adjustment.
* Individual claims about P1, P2 or P3 alone are reported with Holm adjustment across the three (family-wise one-sided α = 0.025).
* Every possible pattern of outcomes is reported as it falls. In particular:
  * P1 failing means joint control did not pay off even under commit, at this scale.
  * P3 failing would contradict Prop. 4′, so the implementation is audited first.
  * Ties are reported as ties.

**Secondary family (directional, Holm-adjusted at family-wise α = 0.05 one-sided).** Every cell of dataset × profile × channel set ∈ {C1, C1+C2, C1+C2+C3}, at the pre-specified price set, with predictions from THEORY §7:

| channel set | vs B9+ / D\* | vs B9 |
|---|---|---|
| none (A1–A3 enforced: frozen query, per-policy constant overlap, final-transcript belief) | **decisions identical; $\bar D=0$ exactly** | $\bar D\ge0$, within the Prop. 2 bound |
| C1 | > 0 for faiss_rerank and remote_api; ≈ 0 for faiss | > 0 |
| C1+C2 (commit) | > 0 | > 0 |
| C1+C2+C3 | > 0 | > 0 |

* **Replication of Prop. 1.** The "none" cells are tested as **decision identity**: the fraction of identical (retrieve, timing-if-retrieve) decisions between Ours-v2 and B9+, predicted to be 100%. Any non-identity is investigated as an implementation error and reported. The utility difference is tested by TOST with margin ±0.005 (equivalence).
* **Predicted-vs-observed agreement.** For each secondary cell × price, the predicted gap $\hat\delta_{\text{dev}}$ is computed **entirely from dev**: the dev-fitted models are applied to dev items through the THEORY expressions and simulation, and the values are written to `docs/research_v2/v2_predictions.json` before the freeze. The observed gap $\delta_{\text{test}}$ is computed on test. Two pre-specified criteria:
  * (a) Spearman rank correlation $r_s(\hat\delta_{\text{dev}},\delta_{\text{test}})\ge0.5$, with one-sided 95% bootstrap lower bound > 0. The bootstrap resamples test items and recomputes every cell.
  * (b) The slope from the inverse-variance-weighted regression of $\delta_{\text{test}}$ on $\hat\delta_{\text{dev}}$ has a 95% bootstrap CI that includes 1 and excludes 0.

  Verdict: **agree** if (a) and (b) both hold; **directional** if only (a) holds; **disagree** otherwise.
* **Frontier dominance (comparability with v1).** The v1 rule unchanged: strictly higher accuracy with non-overlapping 95% bootstrap bands at no higher p95 TTFA or compute, over a contiguous region of ≥ 2 matched prices, in ≥ 3 of 5 seeds.
  * Ours-v2 vs B9 and vs B9+, per dataset × profile, with all channels active.
  * Predicted: no dominance in the "none" cells (Cor. 1).

## 4. Analysis plan

### 4.1 Units, replicates, resampling
* **Unit of analysis:** item. Seeds are within-item replicates; each item's outcome is the mean over its 5 seeds.
* **Paired cluster bootstrap:** B = 10,000, resampling items (d1) or `pair_id` clusters (d2), with all controllers evaluated on the same resample. The pooled primary resamples within dataset (stratified).
* **CIs:** percentile CIs. p95 TTFA is recomputed per resample.

### 4.2 Multiplicity
* One primary test at α = 0.025 one-sided.
* One secondary family, Holm-adjusted.
* H-E1v2 and H-E2v2 are separate experiments, each with its own criteria, as in v1.
* Everything else is descriptive.

### 4.3 Pre-specified price and profile selection (dev only)
* **Price grid:** $\lambda_l$ at the dev belief quantiles {0.5, 0.65, 0.8, 0.9} of the B9+ cut; $\lambda_c$ = 0.05.
* **Primary (profile, price):** the pair maximizing the dev-predicted standardized effect $\hat\delta/\hat\sigma$ for Ours-v2 vs D\*. It is recorded in §9 before the freeze.
* **No test data are used.** The selection uses only dev, so it involves no winner's-curse leakage into test, although the dev-predicted effect is itself optimistic.

### 4.4 Exclusions and failed runs
* **Failed runs.** A run that crashes or times out is retried up to 3 times with the same seed. If it still fails, the item is **excluded from all controllers** (pairwise complete) and counted.
* **Unusable audio.** Items whose ASR final transcript is empty, or whose speech trace violates `CONTRACTS.md` §4, are excluded from all controllers and counted.
* **Judge parse failures.** If fewer than 2 of 3 samples parse, up to 3 more samples are drawn. If no valid majority emerges, the item is excluded in the primary analysis and counted incorrect in a sensitivity analysis.
* **Flagging.** A cell with exclusions above 5% is flagged, and its result is reported with and without the exclusions.
* **No outcome-based exclusions.** No item is excluded on the basis of its outcomes.

### 4.5 Tuning protocol (dev only, equal budget)
* **Shared estimates.** The belief calibration, the remaining-speech predictor, the latency model (including the non-hideable component π) and $\hat g^{\pm}$ are fitted once on dev and shared by every controller.
* **Per-controller parameters.** These are the B9+ timing rule (early / late / R̂-threshold on a 21-point quantile grid) and nothing else. Every controller gets the same number of dev evaluations; Ours-v2 has no free parameter beyond the shared models.
* **Cost calibration.** The latency model is calibrated as the **end-to-end TTFA increment** of retrieval measured on dev, not the retrieval-call duration. In v1 the call duration understated the TTFA cost by more than 2× (THEORY §8.4).

### 4.6 What may and may not change after freezing
* **May change (each logged as a deviation):** bug fixes that do not alter a controller's decision rule; infrastructure; re-running failed items; adding descriptive analyses that are labelled as such.
* **May not change:** hypotheses, predictions, primary and secondary tests, α, margins, the primary (profile, price), the price grid, controller definitions, D\*, judge configuration (deployment, version, reasoning_effort, k, prompt), exclusion rules, splits, model and seeds.

### 4.7 Deviation logging
Every deviation is recorded before the affected analysis is run in:
* `paper/lab_log.md`, with the date, what changed, why, who decided, and whether any test outcome had been seen;
* a "Deviations" section appended to the frozen addendum. That section is the only part that may be appended after the freeze.

## 5. Statistical details
* Primary statistic: $\bar D$, the mean over test items of the seed-averaged paired utility difference.
* Bootstrap test: reject $H_0:\delta\le0$ if the 2.5th percentile of $\bar D^{*}$ is > 0.
* TOST for the "none" cells: both one-sided 95% bounds lie inside ±0.005.
* p95 comparisons: paired cluster bootstrap of the difference in p95.

## 6. Power analysis

**Source.** `analysis/v2_power.py` → `docs/research_v2/v2_power.json`, reproducible with one command.

**Inputs.**
* **Item-level quality effects** come from the v1 E3 cache: each simulated item borrows the five-seed (skip, retrieve) correctness of a v1 scenario with the same need label. This gives $g^{+}=0.720$, $g^{-}=0.592$, and item-effect SDs of 0.49 and 0.46.
* **Beliefs are synthetic and calibrated:**
  * d1: a smooth three-component Beta mixture;
  * d2: a bimodal Beta mixture.
* **Timing is synthetic:**
  * log-normal utterance durations (median 2.8 s for d1, 4.5 s for d2);
  * remaining speech after the first meaningful partial = 0.6 × duration − 0.3 s;
  * call latency log-normal with median 0.672 s (faiss, the v1 measurement), 1.3 s (faiss_rerank) and 2.2 s (remote_api);
  * non-hideable penalty π = 0.9 s (the v1 TTFA increment of 1.566 s minus the 0.672 s call);
  * remaining-speech predictor log-error SD 0.30.

  **All timing parameters, and the prefix noise, are placeholders to be replaced by dev measurements before the freeze**, after which the script is re-run and this section updated.
* **Test:** one-sided α = 0.025, power 0.80, $N=((z_{.975}+z_{.80})\sigma/\delta)^{2}$, multiplier 2.80.
* **Planned sizes for the table:** 1,200 (d1) and 720 (d2) items, i.e. 840 + 504 = 1,344 test items. These are placeholders.

**C1 alone vs B9+ (second-order effect, THEORY Prop. 3).**
* Best cell: d2_support × remote_api, $\delta=+0.0021$, $\sigma=0.107$ → **N_test = 20,821**.
* Pooled remote_api: $\delta=0.0011$ → N_test = 57,193.
* faiss: δ ≈ 0. The call is hidden on 89% of d1_open turns and 98% of d2_support turns.
* At 1,344 test items, power is about 6% (normal approximation) and 8.5% (direct bootstrap check).

**C1 cannot be the primary.** It is a secondary cell with the minimum detectable effect stated: 0.0070 at 1,344 and 0.0040 at 4,000 pooled test items.

**C1+C2 (commit) vs the best decomposed rival** (tuned with oracle knowledge, which favours the rival), at the dev-selected price:

| | δ | σ | N_test required | N_total (30/70) | MDE at planned N | power at planned N |
|---|---|---|---|---|---|---|
| d1_open × faiss | 0.0077 | 0.170 | 3,851 | 5,502 | 0.0164 (n=840) | 0.26 |
| d1_open × faiss_rerank | 0.0110 | 0.194 | 2,461 | 3,516 | 0.0188 | 0.37 |
| d1_open × remote_api | 0.0066 | 0.164 | 4,784 | 6,835 | 0.0158 | 0.22 |
| d2_support × faiss | 0.0144 | 0.211 | 1,678 | 2,398 | 0.0263 (n=504) | 0.34 |
| d2_support × faiss_rerank | 0.0148 | 0.291 | 3,041 | 4,345 | 0.0363 | 0.21 |
| d2_support × remote_api | 0.0142 | 0.274 | 2,906 | 4,152 | 0.0341 | 0.21 |
| **pooled, faiss** | 0.0102 | 0.187 | **2,618** | 3,741 | 0.0143 (n=1,344) | — |
| **pooled, faiss_rerank** | 0.0124 | 0.235 | **2,824** | 4,035 | 0.0180 | — |
| **pooled, remote_api** | 0.0095 | 0.212 | **3,918** | 5,598 | 0.0162 | — |

**MDE vs pooled test N** (C1+C2, `MDE_table`):

| N_test | 1,344 | 2,000 | 3,000 | 4,000 | 6,000 | 8,000 |
|---|---|---|---|---|---|---|
| faiss | 0.0143 | 0.0117 | 0.0095 | 0.0083 | 0.0067 | 0.0058 |
| faiss_rerank | 0.0180 | 0.0147 | 0.0120 | 0.0104 | 0.0085 | 0.0074 |
| remote_api | 0.0162 | 0.0133 | 0.0108 | 0.0094 | 0.0077 | 0.0066 |

**Conclusions for planning.**
1. At the placeholder sizes (1,344 pooled test items) the primary is underpowered: the MDE of 0.014–0.018 exceeds the predicted 0.009–0.012.
2. **About 4,000 pooled test items** (about 5,700 in total at 30/70) give 80% power for every profile under these assumptions. `d1_open` is where N is cheap (PopQA alone has about 14k questions), so the recommendation is `d1_open` ≥ 5,000 items in total.
3. The effect is sensitive to prefix informativeness. At prefix-noise SD 0.75, 1.5 and 3.0, giving prefix AUC 0.80, 0.76 and 0.61, the d2 × remote_api δ is 0.0034, 0.0138 and 0.0048 at the 0.65 price. **The power analysis must be re-run with the E1-v2 dev prefix curve before the freeze.**
4. Both architectures are tested (§3, P1–P3). Under SPECULATIVE the predicted Δ is at most λ_c(1+w̄)·q̂ and can be negative (THEORY Prop. 4′). **Before the freeze, the power analysis must be extended to P2 (the interaction contrast) and P3 (the non-superiority margin m), and the planned N confirmed for all three.** The boundary claim needs all three, so its power is at most the minimum of the three.

## 7. Reporting
* Every hypothesis is reported with its verdict, the point estimate, the CI and the exclusion counts.
* Ties are reported as ties.
* Every primary result is also reported under the secondary judge.
* v1 and v2 results appear in separate tables.

## 8. Freeze checklist (sign-offs)

| item | owner | sign-off |
|---|---|---|
| Dataset sizes, splits and README stats filled in (§1) | Yashmita | Ryan |
| E1-v2 dev results: signal chosen, prefix curve, $\hat g^{\pm}$, boundary-mass check (§3) | Yashmita | Ryan |
| Judge gate passed and recorded; judge fields filled in (§2) | Yashmita | Ryan |
| Both architectures tested (decided 2026-10-16); D\*[COMMIT] and D\*[SPECULATIVE] named; margin m fixed (§3) | Yashmita | Ryan |
| Latency model calibrated as TTFA increment on dev (§4.5) | Yashmita | Ryan |
| Price grid from dev belief quantiles; primary (profile, price) selected (§4.3) | Renya | Ryan |
| `analysis/v2_power.py` re-run with dev inputs; §6 updated | Yashmita | Ryan |
| `docs/research_v2/v2_predictions.json` written from dev (§3) | Yashmita | Ryan |
| All runners refuse `--split test` without `PREREG_FROZEN` (CONTRACTS §7) | Yashmita | Ryan |
| Final read of this document | Ryan (PI) | Ryan |

## 9. Freeze procedure (instructions; NOT executed)

Run these only after every item in §8 is signed off.

1. Fill every `[TO FILL AT FREEZE]` field. Change the status line at the top to `FROZEN` and add the freeze date and the names of the signers.
2. Commit the addendum together with `v2_predictions.json` and the regenerated `v2_power.json`:
   `git commit -m "Freeze v2 pre-registration addendum"`. Record the full commit hash (`git rev-parse HEAD`).
3. Create the sentinel file `docs/research_v2/PREREG_FROZEN` containing:
   * the commit hash from step 2;
   * the UTC timestamp;
   * the sha256 of `PREREG_V2.md` and of `v2_predictions.json`;
   * the signers.

   Commit it.
4. Tag the sentinel commit with an annotated tag: `git tag -a prereg-v2-frozen -m "v2 pre-registration frozen"`. Push the branch and the tag.
5. **OSF.** Create an OSF registration (template: "OSF Preregistration") under the project's OSF page.
   * Attach `PREREG_V2.md`, `v2_predictions.json`, `v2_power.json` and `THEORY.md`.
   * Paste the commit hash and tag name into the registration.
   * Submit the registration, and record its DOI and URL in `PREREG_FROZEN` and in `paper/lab_log.md` (with a commit).
6. Only after steps 1–5 may any runner be invoked with `--split test`.

## 10. Limitations stated in advance
* **Single model.** All v2 results are for Mistral-7B-Instruct-v0.3. Nothing is claimed about other generators.
* **Judge non-determinism.** The primary LLM judge is a reasoning model that runs only at temperature 1 with a best-effort seed. Verdicts are stochastic, majority-of-3 reduces but does not eliminate this, and re-grading may not reproduce identical verdicts.
  * Mitigations: all individual samples are stored; self-consistency is a gate metric; the deterministic GPT-4o v4 judge re-reports every primary result.
* **Synthetic speech.** Speech is synthetic (TTS), and ASR partials come from one streaming ASR configuration.
* **Placeholders in the power analysis.** The timing parameters in §6 are placeholders until the dev measurements replace them.
* **C1 effect size.** The predicted C1 effect is second order. A null in the C1 cells at the planned N is uninformative beyond the stated MDE.
