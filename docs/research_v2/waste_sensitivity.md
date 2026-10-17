# Waste rate w: what the frozen code uses, what the data say, and sensitivity

Status: **descriptive and exploratory.** Nothing here is pre-registered, and nothing here changes a confirmatory result. Sources:

* `docs/research_v2/waste_measured_v2.json` (`python -m analysis.v2_waste_measured`)
* `docs/research_v2/waste_sensitivity_v1.json` (`python -m analysis.v1_waste_sensitivity`)
* v2 d1_open sensitivity: `analysis/v2_waste_sensitivity_d1.py`, **written but not yet run** (see §4).

## Summary

The v2 waste term is not measured anywhere. The frozen controllers price an early commit at λ_c(1 + w(f)), with the placeholder w(f) = 0.1·(1 − f) and w = 0 at end of turn. w enters only the decisions. The scored utility charges λ_c per dispatched retrieval. Under the COMMIT architecture an early dispatch is never cancelled, by construction, so a "cancellation rate of early dispatches" is identically 0 there and cannot estimate w. The quantities the data do support are SPEC's discard rate under the SPECULATIVE architecture (d1_open test, primary ASR condition: 2.3–10.9 % of SPEC dispatches across profiles and prices) and the ASR revision rate of the partial a commit would be made on (d1_open test: 22.0 % of early query-ready partials, ignoring the possibly cut-off last word). In v1, varying w over {0, 0.04, 0.08, 0.16} moves Ours's retrieve decision on at most 20 of 1,300 scenario-prices. It never separates Ours from B9+ (1,300/1,300 agreement at every w) and never produces dominance (0/5 seeds against B9 or B9+ at every w).

## 1. Definition in the frozen code

| | v1 | v2 (frozen) |
|---|---|---|
| where | `controller/baselines/_objective.py: waste_rate(timing)` | `controller/baselines/_objective_v2.py: waste_at(f, is_final)` |
| value | sentence_end 0, immediate 0.02, fixed_early_proxy 0.08 | w(f) = coef·(1 − f) before end of turn, 0 at end of turn; coef = 0.1 (`configs/config.yaml` `controller.e3v2.waste.coef`); f = e/(e + R̂), the causal word-fraction estimate |
| status in code | "PLACEHOLDER — NOT MEASURED" | "PLACEHOLDER … to be replaced by a measured revision rate from the v2 speech traces" |
| enters | the compute charge λ_c(1 + w) of Ours/B9+ and B9's timing sub-optimiser | the compute charge λ_c(1 + w) at a pre-end-of-turn commit (Ours-v2, B9, B9+, SPEC, SPEC-θ) and B9's dev-mean policy constants w̄_p |
| in the scored utility | — | **no**: U = Q − λ_l·ΔTTFA − λ_c·n_retrievals (`simulator/e3v2_decide.py`) |

The frozen d1_open bundles override `quality`, `calibration`, `ours`, `timing` and `estimates_artifact`. They do not override `waste`, so coef = 0.1 was in force for every v2 dev and test result.

## 2. What the existing v2 data measure

**COMMIT architecture.** The cancellation rate is **0 by construction**. `TurnDecision.cancelled_dispatches` is set only on a SPECULATE → DISCARD path, and the commit architecture forbids that path.

**SPECULATIVE architecture, SPEC (B10) discard rate q̂.** This is the fraction of SPEC dispatches discarded at end of turn, as logged by the frozen evaluator. w̄ is the mean frozen w(f) at those dispatches. Values are given as the range over the 4 frozen prices.

| split | dataset | condition | profile | q̂ (min–max) | w̄ |
|---|---|---|---|---|---|
| dev | d1_open | ASR (primary) | rewrite_faiss | 0.0260–0.0912 | 0.0155 |
| dev | d1_open | ASR (primary) | rewrite_faiss_rerank | 0.0246–0.0897 | 0.0155 |
| dev | d1_open | ASR (primary) | rewrite_remote | 0.0376–0.1103 | 0.0155 |
| dev | d1_open | text150 | rewrite_faiss | 0.0028–0.0188 | 0.0424 |
| dev | d1_open | text150 | rewrite_faiss_rerank | 0.0028–0.0240 | 0.0424 |
| dev | d1_open | text150 | rewrite_remote | 0.0007–0.0038 | 0.0424 |
| dev | d2_support | ASR (primary) | all three | 0.0000 at every price | 0.0029–0.0240 |
| dev | d2_support | text150 | all three | 0.0000 at every price | 0.0369–0.0387 |
| test | d1_open | ASR (primary) | rewrite_faiss | 0.0236–0.0917 | 0.0157 |
| test | d1_open | ASR (primary) | rewrite_faiss_rerank | 0.0227–0.0899 | 0.0157 |
| test | d1_open | ASR (primary) | rewrite_remote | 0.0335–0.1089 | 0.0157 |
| test | d1_open | text150 | rewrite_faiss | 0.0054–0.0169 | 0.0424 |
| test | d1_open | text150 | rewrite_faiss_rerank | 0.0060–0.0179 | 0.0424 |
| test | d1_open | text150 | rewrite_remote | 0.0006–0.0066 | 0.0424 |

The source files store fractions, not counts. Counts are given only where they are exactly and uniquely recoverable (see `waste_measured_v2.json`, field `count`). At the primary cell (d1_open test, ASR, rewrite_faiss, q = 0.65, λ_l = 0.173057), SPEC discarded **149 of 3,441** dispatch decisions (q̂ = 0.0433). Exact counts are not recoverable for any d2_support cell, because a rate of 0 fixes no denominator.

**Revision of the partial a commit is made on.** This is what `waste_at`'s docstring says w should measure. It is computed from the committed speech traces of the primary ASR condition, with the frozen step construction and query-ready gate. A partial counts as *revised* if its words (lower-cased, punctuation stripped) are not a prefix of the final ASR transcript. *Excl. last word* ignores the partial's last, possibly cut-off, word. The text150 condition is revision-free by construction. d2_support test traces were not read, because that run is in progress.

| dataset | split | turns | early query-ready partials | revised | revised, excl. last word | first such partial (SPEC's "immediate") | revised | revised, excl. last word |
|---|---|---|---|---|---|---|---|---|
| d1_open | dev | 1,566 | 12,678 | 6,481 (0.511) | 2,881 (0.227) | 1,566 | 1,237 (0.790) | 347 (0.222) |
| d1_open | test | 3,734 | 30,414 | 15,508 (0.510) | 6,678 (0.220) | 3,733 | 2,914 (0.781) | 826 (0.221) |
| d2_support | dev | 222 | 3,148 | 1,190 (0.378) | 454 (0.144) | 222 | 113 (0.509) | 34 (0.153) |

Neither measure is w. A revision does not force a relaunch under COMMIT, and SPEC's discards are decisions made on the end-of-turn belief, not invalidations. Both are listed so that the paper cites a number with the right name.

**Correct wording for the paper (P-III.4).** Replace "In v2, w is measured as the cancellation rate of early dispatches" with:

> In v2, w is not measured: the frozen controllers use the placeholder w(f) = 0.1(1 − f) for a commit made at estimated word fraction f (w = 0 at end of turn), and w enters only the decision, not the scored utility, which charges λ_c per dispatched retrieval; under the commit architecture an early dispatch is never cancelled.

An optional second sentence: "Under the speculative architecture, SPEC discarded 2.3–10.9% of its dispatches on d1_open test (primary ASR condition), and 22.0% of early query-ready ASR partials were later revised (last word excluded)."

## 3. v1 sensitivity: w ∈ {0, 0.04, 0.08, 0.16}

w is applied to both early timings (immediate and fixed_early_proxy), with w(sentence_end) = 0. The 26-point price grid, the frozen beliefs and the controllers' code are unchanged; only the config dict they read is swapped. Every decision is a lookup in the measured v1 cells, so no generation is needed. Reproduction check: at the logged placeholder, the reconstructed outcome records match the logged sweep on 6,500/6,500 records per controller, with 0 changed decisions.

| scenario | w(imm), w(fep) | Ours = B9 (same price, /1300) | Ours = B9+ (/1300) | Ours(λ_i) = B9(λ_{i−1}) (/1250) | retrievals Ours / B9 / B9+ | Ours retrieve flips vs logged | Ours retrieves at fep / sentence_end | prices with Ours ≠ B9 accuracy (/26) | mean acc. gap Ours − B9 | max acc. gap | dominance seeds vs B9 / vs B9+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logged placeholder | 0.02, 0.08 | 1259 | 1300 | 1250 | 982 / 941 / 982 | 0 | 982 / 0 | 8 | +0.0080 | +0.212 | 0/5 / 0/5 |
| w = 0 | 0, 0 | 1239 | 1300 | 1232 | 1002 / 941 / 1002 | 20 | 1002 / 0 | 17 | +0.0011 | +0.212 | 0/5 / 0/5 |
| w = 0.04 | 0.04, 0.04 | 1246 | 1300 | 1239 | 995 / 941 / 995 | 13 | 995 / 0 | 14 | +0.0034 | +0.212 | 0/5 / 0/5 |
| w = 0.08 | 0.08, 0.08 | 1259 | 1300 | 1250 | 982 / 941 / 982 | 0 | 982 / 0 | 8 | +0.0080 | +0.212 | 0/5 / 0/5 |
| w = 0.16 | 0.16, 0.16 | 1263 | 1300 | 1246 | 978 / 941 / 978 | 4 | 489 / 489 | 8 | +0.0078 | +0.212 | 0/5 / 0/5 |

Notes:

* **B9+ equals Ours at every w** (1,300/1,300 retrieve decisions, and timing on every turn both retrieve).
* **w = 0.08 reproduces the logged run exactly**, because fixed_early_proxy is the chosen timing at every grid price. The immediate waste therefore never binds.
* **w = 0.16.** On 489 of 978 Ours retrievals, sentence_end overtakes fixed_early_proxy. B9's retrieve decisions do not change: it retrieves on 941 at every w, and at w = 0.16 it has 600 timing-label-only changes.
* **The one-step grid shift (Prop. 1)** is exact only at the w the grid was built for. The grid recurrence's fixed point is λ_c(w* − w_se)/o*. Agreement falls to 1,232–1,246 of 1,250 at the other w.
* The dominance test is the unchanged pre-registered `analysis/e3_frontier.py` test, run on the reconstructed records.

## 4. v2 d1_open sensitivity: not run (data access)

`analysis/v2_waste_sensitivity_d1.py` does the following:

* re-runs the frozen evaluator (`simulator/e3v2_lazy.frozen_records`) for Ours-v2 and B9+ with only the waste coefficient changed, coef ∈ {0, 0.04, 0.08, 0.16} plus the frozen 0.1;
* scales the policy-constant w̄ by coef/0.1;
* keeps the tuned B9+ rules frozen;
* looks outcomes up in the existing Phase A test store.

A decision that lands on an ungenerated cell is detected, never scored, and its item is dropped from that cell and counted. The script refuses to write results unless coef = 0.1 reproduces `contrasts_test.json`'s per-item Δ[COMMIT] exactly.

It has **not been run**. The frozen bundles, the d1_open test cell store and `contrasts_test.json` exist only in the worktree of the in-progress confirmatory d2 run, and reading from that worktree was blocked for this task. To run it once access is allowed (read-only; no generation, grading or judge call):

```
PYTHONUTF8=1 VOICE_RAG_MODEL=mistral7b_v03 .venv-e2/Scripts/python.exe -m analysis.v2_waste_sensitivity_d1 \
    --phase-a <worktree>/simulator/e3v2_phase_a
```

It writes `docs/research_v2/waste_sensitivity_v2_d1.json`. The result would be labelled *exploratory sensitivity*. The tuned B9+ timing rules and the price grid were selected on dev at coef = 0.1 and are not re-tuned.

## Caveats

* v1 and v2 sensitivity both change w only in the controllers' decisions. The realised utility and the measured outcomes do not depend on w.
* The v1 grid, and in v2 the frozen tuned parameters, were built at the placeholder w.
* q̂ is SPEC's end-of-turn discard rate: its denominator includes SPEC turns that dispatched only at end of turn. It is a decision rate, not an invalidation rate.
* The revision rates are text-level ASR revisions of the partial. They say nothing about whether retrieval results would have changed; that needs the Phase A retrieval records, which are in the same blocked worktree.
