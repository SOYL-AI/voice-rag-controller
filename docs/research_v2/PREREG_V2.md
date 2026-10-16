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
* **Planned sizes.** [TO FILL AT FREEZE, from the dataset READMEs]. §6.1 shows the boundary-test primary (P1 ∧ P2 ∧ P3) needs about **3,800–7,100 pooled test items, depending on the primary profile** (including a 10% allowance). The recommended size is therefore `d1_open` ≥ 6,000 items in total (≥ 4,200 test), because d1_open is where N is cheap.
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
| sampling | **k = 1 sample per answer** (decided 2026-10-16, before freeze, to meet the compute budget). Justification from dev: under rubric r3 the 3 samples were unanimous on 98.8% of answers (Fleiss' κ 0.977), so majority-of-3 changes almost no verdicts. The raw output and parsed verdict are stored |

**Recording requirement.** For every call, store the exact request body sent (all parameters, including any the SDK adds), the response id, the model version reported by the service, the usage fields and the raw output. A parameter that is rejected, or silently dropped, is a recorded fact, not an error to work around.

**Non-determinism.** Because temperature is fixed at 1, the judge is stochastic. Majority-of-3 reduces this but does not remove it, and the seed is best-effort. Re-grading may not reproduce byte-identical verdicts. This is a stated limitation (§10). The stored per-sample verdicts are the record of what was graded.

### 2.3 Pre-specified secondary judge: GPT-4o v4

`azure-openai/gpt-4o@2025-01-01-preview`, cache version `v4-azure-reasoned` (reasoned+strip prompt), temperature 0 (deterministic). This is the judge validated in v1 (`simulator/judge_validation/`).

* It grades a **stratified random 30% subsample** of all graded answers (strata: dataset × item_type × primary verdict; fixed seed recorded at freeze). On that subsample, all primary v2 results are recomputed under both judges and compared, as a robustness check. Decided 2026-10-16, before freeze, to meet the compute and cost budget.
* A conclusion that holds under one judge and not the other is reported as judge-dependent.

### 2.4 Judge-validation gate (before any test grading)

**Validation set.** About 300 Mistral-7B answers from **dev items only**:
* stratified by dataset (100 d1 / 200 d2), path (retrieve / skip) and `item_type`;
* each labelled independently by two humans blind to path and judge;
* disagreements adjudicated by a third.

The set is split, before any judge sees it, into **100 prompt-development** and **200 gate-holdout** answers, by sha256 of the answer id.

**Gate metrics.** In this order:

1. **Human inter-annotator agreement**: Cohen's κ, reported first. If κ_human < 0.70, the labelling guideline is revised and the set re-labelled before the judge is assessed, because a judge cannot be validated against labels less reliable than the bar it must clear.
2. **Judge vs adjudicated human labels** (on the 200 holdout, single-sample verdict, k = 1). Pass requires **all** of:
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
* **SPEC-θ** (B11, belief-gated speculation; THEORY §6.2): SPEC with a belief gate. At the same trigger it prefetches only if the belief-only option value of the prefetch exceeds the belief-only value of not prefetching, $E[(A(b_{\text{end}})-\lambda_l\bar X_{\text{trigger}})^{+}\mid b]-\lambda_c(1+w)>E[(A(b_{\text{end}})-\lambda_l\bar X_{\text{end}}-\lambda_c)^{+}\mid b]$, where $\bar X$ are dev-mean exposures (scalars). The use rule is SPEC's. SPEC-θ is decomposed in the same sense as B9+: timing fixes *when*, and the belief with fixed prices decides *whether*.
  * **Why it is needed.** Without SPEC-θ, Ours-v2[SPECULATIVE] beats every listed rival simply by not prefetching on confident-skip turns, which a decomposed controller can do (THEORY Prop. 4″). The speculative contrast would then measure that, not joint control.

**Architectures (decided by the PI, 2026-10-16): the study tests BOTH, as a boundary test.**
* **COMMIT:** a dispatched retrieval's passages are always used.
* **SPECULATIVE:** a dispatched retrieval may be discarded at end of turn.

Both are run on the same test items, seeds, latency profile and price. The architecture is the only thing that changes.
* **Ours-v2[arch]** is the joint controller for that architecture.
  * Under COMMIT, it is THEORY §5–6.
  * Under SPECULATIVE, it is the same optimal-stopping controller with the dispatch value replaced by the option value of having the passages ready, $E[\max(U_{\text{use}}(b_{\text{end}}),0)]$, less the dispatch compute.
* **D\*[arch]** is the strongest decomposed rival for that architecture:
  * D\*[COMMIT] = B9+;
  * D\*[SPECULATIVE] = whichever of {B9+, SPEC, SPEC-θ} has the highest dev expected utility at the primary price. SPEC-θ is expected to win (THEORY §6.2).
  * Both are named at freeze.
* **Δ[arch]** = $\bar D(\text{Ours-v2[arch]} - D^{*}[\text{arch}])$.

**Outcome per item.** Realized utility under the corrected objective, averaged over the 5 seeds:

$$U=Q-\lambda_l\cdot\Delta\text{TTFA}-\lambda_c\cdot n_{\text{retrievals}},$$

where $Q$ is the primary correctness and ΔTTFA is the measured TTFA increase over the same item's skip cell. The pairwise estimand is the mean paired difference $\bar D$ over test items.

**Primary: the boundary test (three co-primary hypotheses).**

Setup:
* all channels active (C1+C2+C3);
* pooled over d1_open and d2_support test items (item-weighted);
* at **one latency profile and one price**, the same for both architectures, fixed at freeze by the §4.3 rule (maximum dev-predicted power of the P1 ∧ P2 ∧ P3 conjunction): [PROFILE, PRICE TO FILL AT FREEZE].

The three hypotheses follow from THEORY Props. 4, 4′ and 4″:
* **P1 (joint control pays when retrieval is committed):** Δ[COMMIT] > 0.
* **P2 (the boundary itself):** Δ[COMMIT] − Δ[SPECULATIVE] > 0, a paired interaction contrast on the same items.
* **P3 (under speculation, joint control adds nothing beyond belief-gated prefetch):** a TOST equivalence test, −m₃ < Δ[SPECULATIVE] < m₃.
  * **Margin.** m₃ = ½·δ̂_C,dev, half of the dev-estimated Δ[COMMIT] at the primary (profile, price). It is fixed at freeze: [m₃ TO FILL AT FREEZE].
  * **Decision.** P3 passes iff the 2.5th and 97.5th percentiles of the bootstrap distribution of Δ[SPECULATIVE] both lie inside (−m₃, m₃).
  * **Why this margin** (THEORY §6.2):
    * The draft margin λ_c(1+w̄)q̂ is not a valid bound on the joint advantage over SPEC: there is an analytic counterexample, and it is exceeded in simulation.
    * It is about 3× the predicted Δ[COMMIT] (0.029 against 0.009–0.011), so P3 would follow from P2 almost for free.
    * With C3 active, no compute-only bound holds.
    * The ½δ̂_C margin makes P3 say something P2 does not: speculation removes at least half the commit advantage in absolute terms.
    * The lower side checks that the joint speculative controller is not broken. Theory predicts Δ[SPECULATIVE] ≥ 0 under correct models.
  * **Diagnostics, not tests.** The two-epoch theory bounds $m_{\text{prophet}}$ (vs SPEC) and $m_{C1}$ (vs SPEC-θ) are computed on dev and reported next to P3.

How the tests are decided:
* All tests use the paired cluster bootstrap (B = 10,000, the same resamples for every contrast). There are four one-sided tests at α = 0.025: P1, P2, and the two sides of P3.
* **The boundary claim is supported iff all four reject.** This is an intersection–union test, so the conjunction needs no multiplicity adjustment.
* Individual claims about P1, P2 or P3 alone are reported with Holm adjustment across the three (family-wise one-sided α = 0.025; P3 counts as one test, at the larger of its two p-values).
* Every possible pattern of outcomes is reported as it falls. In particular:
  * P1 failing means joint control did not pay off even under commit, at this scale.
  * P3's upper side failing means the joint controller keeps a speculative advantage over belief-gated prefetch. THEORY §6.2 names the possible sources: delaying a paid prefetch across several partials, and C3 query improvement. It is reported as such.
  * P3's lower side failing triggers an implementation audit.
  * Ties are reported as ties.

**Secondary family (directional, Holm-adjusted at family-wise α = 0.05 one-sided).** Every cell of architecture × dataset × profile × channel set ∈ {C1, C1+C2, C1+C2+C3}, at the pre-specified price set, with predictions from THEORY §6–7:

| architecture | channel set | vs D\*[arch] | vs B9 | vs SPEC |
|---|---|---|---|---|
| COMMIT | none (A1–A3 enforced: frozen query, per-policy constant overlap, final-transcript belief) | **decisions identical; $\bar D=0$ exactly** | $\bar D\ge0$, within the Prop. 2 bound | — |
| COMMIT | C1 | > 0 for faiss_rerank and remote_api; ≈ 0 for faiss | > 0 | — |
| COMMIT | C1+C2 | > 0 | > 0 | — |
| COMMIT | C1+C2+C3 | > 0 | > 0 | — |
| SPECULATIVE | C1, C1+C2 | ≈ 0 (TOST ±m₃) | > 0 | > 0 (belief-gated compute saving, Prop. 4′) |
| SPECULATIVE | C1+C2+C3 | ≥ 0, possibly > 0 through the C3 query-improvement term (reported, not predicted to exceed m₃) | > 0 | > 0 |

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
* One primary intersection–union test (P1 ∧ P2 ∧ P3, i.e. four one-sided tests, each at α = 0.025; §3).
* One secondary family, Holm-adjusted.
* H-E1v2 and H-E2v2 are separate experiments, each with its own criteria, as in v1.
* Everything else is descriptive.

### 4.3 Pre-specified price and profile selection (dev only)
* **Price grid:** $\lambda_l$ at the dev belief quantiles {0.5, 0.65, 0.8, 0.9} of the B9+ cut; $\lambda_c$ = 0.05.
* **Primary (profile, price):** the pair that maximizes the dev-predicted power of the P1 ∧ P2 ∧ P3 conjunction at the planned test N. Equivalently, it minimizes the N needed for 80% conjunction power, computed as in §6 from the dev per-item paired differences of both architectures.
  * In the placeholder simulation this coincides with the rule "maximize δ̂/σ̂ of Δ[COMMIT]" (the 0.65 belief quantile for every profile).
  * The conjunction rule is used because P2 and P3 also depend on the price.
  * The selected pair and m₃ are recorded in §1 and §3 before the freeze.
* **No test data are used.** The selection uses only dev, so it involves no winner's-curse leakage into test, although the dev-predicted effect is itself optimistic.

### 4.4 Exclusions and failed runs
* **Failed runs.** A run that crashes or times out is retried up to 3 times with the same seed. If it still fails, the item is **excluded from all controllers** (pairwise complete) and counted.
* **Unusable audio.** Items whose ASR final transcript is empty, or whose speech trace violates `CONTRACTS.md` §4, are excluded from all controllers and counted.
* **Judge parse failures.** If the sample does not parse, up to 3 more samples are drawn and the first that parses is used. If none parses, the item is excluded in the primary analysis and counted incorrect in a sensitivity analysis.
* **Flagging.** A cell with exclusions above 5% is flagged, and its result is reported with and without the exclusions.
* **No outcome-based exclusions.** No item is excluded on the basis of its outcomes.

### 4.5 Tuning protocol (dev only, equal budget)
* **Shared estimates.** The belief calibration, the remaining-speech predictor, the latency model (including the non-hideable component π) and $\hat g^{\pm}$ are fitted once on dev and shared by every controller.
* **Per-controller parameters.** The tunable parameters are:
  * the B9+ timing rule (early / late / R̂-threshold / R̂-below, each threshold on a 21-point quantile grid). R̂-below dispatches at the first query-ready partial with R̂ ≤ r₀ (end of turn if none), priced at that rule's dev-mean realised exposure; added 2026-10-16, before freeze (PI decision), because without a late-partial rule the family had a gap that inflated Δ[COMMIT] on dev ASR (see `paper/lab_log.md`);
  * the shared SPEC / SPEC-θ trigger policy (immediate, forecast, or R̂-below on the same 21-point grid; added 2026-10-16, before freeze);
  * the dev-mean scalar exposures used by SPEC-θ, which are estimates and not tuned. For an R̂-below trigger these are the dev means of remaining and elapsed speech at its firing step.

  Nothing else is tuned. Every controller gets the same number of dev evaluations. Ours-v2 (both architectures) has no free parameter beyond the shared models and its lookahead depth, which is fixed at freeze.
* **Ours-v2 lookahead (corrected 2026-10-16, before freeze; PI decision; its implementation corrected once more the same day for a code-level defect).** From a step, the next partial carries new information with probability p = P(R > s), where R is the remaining *speech* under the remaining-speech predictor's q10/q50/q90, aged to the step's availability time and not clipped at 0 (so p → 0 after the predicted end of speech), and s is the dev-median partial-to-partial availability spacing (ASR lag included). Post-speech re-transcription partials carry no new information (identity belief transition; they are merged into the previous step when the transitions are fitted), so the other branch is the end-of-turn decision. The end-of-turn gap enters only the exposure term. On the text150 timeline the prefix schedule is deterministic: p = 1 while a prefix is still to come and 0 after the last. The value of waiting is p · (next-partial value) + (1 − p) · (end-of-turn decision value), in both architectures. The previous version assumed the next partial certain whenever R̂ exceeded the nominal partial interval; the first correction used p = P(R + gap > s) with R clipped at 0, which is ≈ 1 at every dev step, so its end-of-turn branch never fired. No other change was made to Ours-v2 and no parameter was selected by looking at Δ on dev. The partial spacing is a shared estimate, like the end gap. The test split uses exactly the code frozen after this correction, whatever the dev outcome.
* **Cost calibration.** The latency model is calibrated as the **end-to-end TTFA increment** of retrieval measured on dev, not the retrieval-call duration. In v1 the call duration understated the TTFA cost by more than 2× (THEORY §8.4).

### 4.6 What may and may not change after freezing
* **May change (each logged as a deviation):** bug fixes that do not alter a controller's decision rule; infrastructure; re-running failed items; adding descriptive analyses that are labelled as such.
* **May not change:**
  * hypotheses, predictions, the primary and secondary tests, α;
  * margins (m₃ and the ±0.005 replication margin) and δ̂_C,dev;
  * the primary (profile, price) and the price grid;
  * both architectures and their controller definitions, including SPEC-θ, the trigger policy and the lookahead depth;
  * D\*[COMMIT] and D\*[SPECULATIVE];
  * judge configuration (deployment, version, reasoning_effort, k, prompt);
  * exclusion rules, splits, model and seeds.

### 4.7 Deviation logging
Every deviation is recorded before the affected analysis is run in:
* `paper/lab_log.md`, with the date, what changed, why, who decided, and whether any test outcome had been seen;
* a "Deviations" section appended to the frozen addendum. That section is the only part that may be appended after the freeze.


### 4.8 Compute-saving implementation rules (decided 2026-10-16, before freeze; results-invariant)
* **Lazy outcome evaluation.** Controller decisions depend only on beliefs, the timing state and the latency model, never on answer correctness. So Phase B runs first on test, and only (item, seed, condition, query_variant) cells chosen by at least one pre-registered controller at at least one pre-registered (profile, price) are generated and graded. This gives results identical to grading every cell; the cell list is written to disk before any grading.
* **E1-v2 on test.** `self_assessed_need` (the operational and gate signal) is computed on every text prefix and every ASR partial. Closed-book entropy signals are computed on the full question only, for the reported E1 comparison, not on partials.
* **Throughput only.** The LLM is served on more containers and with higher client concurrency, and the primary judge uses a pool of two deployments of the identical model version (2026-09-29). Neither changes any model, prompt or parameter.

## 5. Statistical details
* Primary statistic: $\bar D$, the mean over test items of the seed-averaged paired utility difference.
* Bootstrap test: reject $H_0:\delta\le0$ if the 2.5th percentile of $\bar D^{*}$ is > 0.
* Interaction (P2): the same test applied to the per-item difference $D_{C,i}-D_{S,i}$ on items present in both architectures.
* P3 TOST: reject both $H_0:\delta_S\ge m_3$ and $H_0:\delta_S\le-m_3$, i.e. the 2.5th–97.5th percentile interval of $\bar D_S^{*}$ lies inside $(-m_3,m_3)$. $m_3$ is a fixed constant from dev and is not re-estimated in the bootstrap.
* TOST for the "none" cells: the 2.5th–97.5th percentile interval lies inside ±0.005.
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
2. **About 4,000 pooled test items** (about 5,700 in total at 30/70) give 80% power for the commit contrast alone, for every profile, under these assumptions. The primary conjunction needs more (§6.1).
3. The effect is sensitive to prefix informativeness. At prefix-noise SD 0.75, 1.5 and 3.0, giving prefix AUC 0.80, 0.76 and 0.61, the d2 × remote_api δ is 0.0034, 0.0138 and 0.0048 at the 0.65 price. **The power analysis must be re-run with the E1-v2 dev prefix curve before the freeze.**
4. The table above covers the commit contrast alone. The primary is the P1 ∧ P2 ∧ P3 conjunction, sized below.

### 6.1 Power of the boundary test (P1, P2, P3 and their conjunction)

**Source.** `v2_power.json`, block `boundary_test` (`simulate_boundary`, `boundary_test_power`).

**Setup.**
* The same items under both architectures, two decision epochs (first meaningful partial, end of turn), C1+C2 active.
* Compute is λ_c(1+w) with the v1 waste placeholders (early 0.08, end 0), applied to every controller.
* d1 and d2 are pooled in proportion 840 : 504.
* **Conjunction power.** Each statistic's sample mean is shown > 0 one-sided at α = 0.025; P3 counts as two such statistics. Power is evaluated with the joint normal law of the per-item statistics (their full covariance is estimated from the simulation), using 50,000 draws.
* **Direct check.** At the faiss cell, at the N the normal approximation gives, a full paired percentile bootstrap (200 simulated test sets, B = 1,000) gives conjunction power 0.765. The Wald test on the same test sets gives 0.775 (MC SE 0.028), against 0.80 from the normal approximation. **The N below are therefore lower bounds; plan about 10% above them.**

**Predicted effects at the 0.65 belief-quantile price** (selected by both §4.3 rules in every profile):

| profile | Δ[COMMIT] (σ) | Δ[SPEC] vs {B9+, SPEC, SPEC-θ} (σ) | interaction | m₃ = ½Δ[C] | Δ[SPEC] vs {B9+, SPEC} only | draft m = λ_c(1+w̄)q̂ |
|---|---|---|---|---|---|---|
| faiss | 0.0088 (0.175) | −0.0001 (0.049) | 0.0089 | 0.0044 | 0.0149 | 0.0293 |
| faiss_rerank | 0.0108 (0.229) | −0.0002 (0.095) | 0.0110 | 0.0054 | 0.0133 | 0.0287 |
| remote_api | 0.0097 (0.201) | −0.0003 (0.113) | 0.0100 | 0.0048 | 0.0151 | 0.0291 |

**N_test for 80% conjunction power:**

| design | faiss | faiss_rerank | remote_api |
|---|---|---|---|
| **adopted:** D\*[S] includes SPEC-θ, P3 = TOST at ½δ̂_C | **3,417** (total 4,882) | **4,631** (total 6,616) | **6,492** (total 9,275) |
| variant: P3 one-sided at ½δ̂_C | 3,395 | 4,299 | 5,125 |
| variant: P3 one-sided at the two-epoch theory bound m_C1 | 3,655 | 4,101 | 4,358 |
| draft design (D\*[S] ∈ {B9+, SPEC}, m = λ_c(1+w̄)q̂) | **∞**: P2 predicted false (interaction −0.0062 / −0.0026 / −0.0054) | ∞ | ∞ |

**Conjunction power at fixed pooled N, adopted design:**

| N_test | faiss | faiss_rerank | remote_api |
|---|---|---|---|
| 1,344 (placeholder sizes) | 0.33 | 0.03 | 0.00 |
| 4,000 | 0.86 | 0.71 | 0.43 |

**Conclusions.**
1. With the draft rival set and margin, the boundary claim cannot pass at any N. Ours-v2[SPEC] beats {B9+, SPEC} by 0.013–0.015, which is more than its commit advantage. SPEC-θ must be implemented and added to D\*[SPECULATIVE].
2. With the adopted design, the primary needs about **3,400–6,500 pooled test items**, depending on the profile. Planning for 10% more gives about **3,800 (faiss) to 7,100 (remote_api) test items**, or roughly 5,400–10,200 in total at 30/70.
   * The choice of primary profile therefore drives dataset size. faiss is the cheapest because its Δ[SPEC] has the smallest variance.
3. **What this simulation does not cover.** It has two epochs and Δρ = 0, so it does not quantify two terms that THEORY §6.2 says can keep Δ[SPEC] > 0: multi-epoch prefetch delay, and C3 query improvement.
   * Both must be measured on dev.
   * If they push dev Δ[SPEC] towards m₃, the P3 power above is optimistic.

## 7. Reporting
* Every hypothesis is reported with its verdict, the point estimate, the CI and the exclusion counts.
* Ties are reported as ties.
* Every primary result is also reported under the secondary judge, on its 30% subsample (§2.3).
* v1 and v2 results appear in separate tables.

## 8. Freeze checklist (sign-offs)

| item | owner | sign-off |
|---|---|---|
| Dataset sizes, splits and README stats filled in (§1) | Yashmita | Ryan |
| E1-v2 dev results: signal chosen, prefix curve, $\hat g^{\pm}$, boundary-mass check (§3) | Yashmita | Ryan |
| Judge gate passed and recorded; judge fields filled in (§2) | Yashmita | Ryan |
| Both architectures tested (decided 2026-10-16); SPEC-θ (B11) implemented and unit-tested; `boundary_contrasts` reports Δ against D\* including SPEC-θ and the TOST margin; D\*[COMMIT] and D\*[SPECULATIVE] named; m₃ = ½δ̂_C,dev fixed (§3) | Yashmita | Ryan |
| Multi-epoch prefetch-delay and C3 terms of Δ[SPEC] estimated on dev (THEORY §6.2); §6.1 re-run with dev inputs | Yashmita | Ryan |
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
* **Judge non-determinism.** The primary LLM judge is a reasoning model that runs only at temperature 1 with a best-effort seed. Verdicts are stochastic, k = 1 is used (dev unanimity across 3 samples was 98.8%, so the residual instability is small but non-zero), and re-grading may not reproduce identical verdicts.
  * Mitigations: all individual samples are stored; self-consistency is a gate metric; the deterministic GPT-4o v4 judge re-reports every primary result.
* **Synthetic speech.** Speech is synthetic (TTS), and ASR partials come from one streaming ASR configuration.
* **Placeholders in the power analysis.** The timing parameters in §6 are placeholders until the dev measurements replace them.
* **C1 effect size.** The predicted C1 effect is second order. A null in the C1 cells at the planned N is uninformative beyond the stated MDE.
