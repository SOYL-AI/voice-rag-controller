# v2 test results — d1_open only

**Status: d1_open only; pooled confirmatory results pending d2 grading after the judge gate.** Every p-value below is per-dataset and labelled *d1-only, not the confirmatory pooled test*. Frozen code `ddde67f` plus deviation D1; D\*[COMMIT] = D\*[SPECULATIVE] = B9+; m₃ = 0.016. Paired bootstrap over items, B = 10,000, the same resamples for every contrast. Source: `docs/research_v2/v2_test_results_d1.json` (`analysis/v2_test_d1.py`).

## Primary cell (rewrite_faiss, belief quantile 0.65, λ_l = 0.17305735434823866, λ_c = 0.05, primary ASR)

| contrast | estimate | 95% CI | n items |
|---|---|---|---|
| Δ[COMMIT] | -0.0367 | [-0.0481, -0.0252] | 3,734 |
| Δ[SPECULATIVE] | -0.0415 | [-0.0528, -0.0302] | 3,734 |
| interaction Δ[C] − Δ[S] | +0.0047 | [+0.0007, +0.0088] | 3,734 |

**One-sided tests (d1-only, not the confirmatory pooled test):**

| test | H0 | p (bootstrap) | rejects at α = 0.025 | Holm (d1-only) |
|---|---|---|---|---|
| P1 | Δ[C] ≤ 0 | 1.0000 | False | False |
| P2 | Δ[C] − Δ[S] ≤ 0 | 0.0097 | True | False |
| P3 (TOST, ±0.016) | upper Δ[S] ≥ m₃ / lower Δ[S] ≤ −m₃ | < 1e-04 / 1.0000 | False (upper True, lower False) | False |
| H-REV | Δ[C] ≥ 0 | < 1e-04 | True | True |

Boundary claim (P1 ∧ P2 ∧ P3), d1-only: **False**.

**Secondary condition text150** (rewrite_faiss, its own 0.65-quantile price 0.161524|0.05): Δ[COMMIT] -0.0024 [-0.0072, +0.0022], Δ[SPEC] -0.0024 [-0.0072, +0.0022], interaction +0.0000.

## All cells (descriptive)

| condition | profile | q | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |
|---|---|---|---|---|---|
| ASR (primary) | rewrite_faiss | 0.5 | -0.0365 [-0.0477, -0.0252] | -0.0395 [-0.0507, -0.0285] | +0.0030 [-0.0010, +0.0070] |
| ASR (primary) | rewrite_faiss | 0.65 | -0.0367 [-0.0481, -0.0252] | -0.0415 [-0.0528, -0.0302] | +0.0047 [+0.0007, +0.0088] |
| ASR (primary) | rewrite_faiss | 0.8 | -0.0409 [-0.0530, -0.0292] | -0.0453 [-0.0571, -0.0337] | +0.0044 [+0.0003, +0.0084] |
| ASR (primary) | rewrite_faiss | 0.9 | -0.0432 [-0.0559, -0.0309] | -0.0456 [-0.0576, -0.0336] | +0.0023 [-0.0019, +0.0067] |
| ASR (primary) | rewrite_faiss_rerank | 0.5 | -0.0354 [-0.0466, -0.0240] | -0.0385 [-0.0497, -0.0276] | +0.0031 [-0.0008, +0.0071] |
| ASR (primary) | rewrite_faiss_rerank | 0.65 | -0.0363 [-0.0477, -0.0248] | -0.0414 [-0.0527, -0.0302] | +0.0051 [+0.0010, +0.0092] |
| ASR (primary) | rewrite_faiss_rerank | 0.8 | -0.0409 [-0.0529, -0.0293] | -0.0433 [-0.0550, -0.0320] | +0.0024 [-0.0017, +0.0065] |
| ASR (primary) | rewrite_faiss_rerank | 0.9 | -0.0416 [-0.0543, -0.0292] | -0.0442 [-0.0563, -0.0322] | +0.0026 [-0.0015, +0.0068] |
| ASR (primary) | rewrite_remote | 0.5 | -0.0415 [-0.0531, -0.0301] | -0.0417 [-0.0533, -0.0301] | +0.0001 [-0.0040, +0.0043] |
| ASR (primary) | rewrite_remote | 0.65 | -0.0432 [-0.0564, -0.0300] | -0.0435 [-0.0566, -0.0304] | +0.0003 [-0.0040, +0.0046] |
| ASR (primary) | rewrite_remote | 0.8 | -0.0433 [-0.0559, -0.0311] | -0.0443 [-0.0564, -0.0322] | +0.0010 [-0.0033, +0.0054] |
| ASR (primary) | rewrite_remote | 0.9 | -0.0456 [-0.0585, -0.0332] | -0.0444 [-0.0568, -0.0323] | -0.0012 [-0.0056, +0.0031] |
| text150 | rewrite_faiss | 0.5 | -0.0085 [-0.0119, -0.0052] | -0.0085 [-0.0119, -0.0052] | +0.0000 [-0.0001, +0.0001] |
| text150 | rewrite_faiss | 0.65 | -0.0024 [-0.0072, +0.0022] | -0.0024 [-0.0072, +0.0022] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_faiss | 0.8 | -0.0010 [-0.0033, +0.0014] | -0.0010 [-0.0033, +0.0014] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_faiss | 0.9 | -0.0028 [-0.0055, -0.0000] | -0.0028 [-0.0055, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_faiss_rerank | 0.5 | -0.0088 [-0.0122, -0.0055] | -0.0088 [-0.0122, -0.0055] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_faiss_rerank | 0.65 | -0.0028 [-0.0075, +0.0019] | -0.0028 [-0.0075, +0.0019] | -0.0000 [-0.0002, +0.0002] |
| text150 | rewrite_faiss_rerank | 0.8 | -0.0010 [-0.0033, +0.0014] | -0.0010 [-0.0033, +0.0014] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_faiss_rerank | 0.9 | +0.0013 [-0.0070, +0.0101] | +0.0013 [-0.0070, +0.0101] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_remote | 0.5 | -0.0076 [-0.0110, -0.0042] | -0.0076 [-0.0110, -0.0042] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_remote | 0.65 | -0.0023 [-0.0071, +0.0024] | -0.0023 [-0.0071, +0.0024] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_remote | 0.8 | -0.0002 [-0.0027, +0.0023] | -0.0002 [-0.0027, +0.0023] | +0.0000 [+0.0000, +0.0000] |
| text150 | rewrite_remote | 0.9 | -0.0033 [-0.0065, -0.0000] | -0.0033 [-0.0065, -0.0000] | +0.0000 [+0.0000, +0.0000] |

## E1-v2 on test (d1_open only)

* Frozen `analyze.py` output: `C:\Users\ryang\OneDrive\Desktop\voice-rag\voice-rag-controller\llm\experiment1_v2\results\d1_open\analysis_test\analysis_test.json`. The gate criterion (success AUC ≥ 0.70; kill AUC < 0.60 on both datasets) is reported for d1 only.
* Dev-fitted calibration on test, full_text_question: ECE (10 bins) 0.0766; base rate 0.562, mean belief 0.529; boundary mass 0.000 within ±0.15 of b* = -0.342 (prediction ≥ 0.15).
* Dev-fitted calibration on test, primary_asr_final: ECE (10 bins) 0.0664; base rate 0.562, mean belief 0.560; boundary mass 0.000 within ±0.15 of b* = -0.342 (prediction ≥ 0.15).
* **Primary (gate) AUC**, raw self_assessed_need on the full question vs the empirical label: **0.665** [0.648, 0.682] (n = 3,734, 2,097 needed). Success (≥ 0.70): False; below 0.60: False (the kill rule needs both datasets).
* Text prefix curve (AUC at word_frac 0.3 / 0.5 / 0.7 / 1.0): 0.537 / 0.543 / 0.582 / 0.665; AUC(1.0) − AUC(0.3) = +0.128 [+0.100, +0.155] (paired bootstrap, B = 10,000; one-sided p < 1e-04).
* C2 available (AUC(0.5) ≥ 0.65): **False**.
* Primary-ASR curve (latest partial with spoken word fraction ≤ f; f = 1.0 is the ASR final transcript; computed by `analysis/v2_test_d1.py` because the test E1 run has no ASR signal records, note N1): f=0.3: 0.540 (n=3,407) / f=0.5: 0.561 (n=3,733) / f=0.7: 0.583 (n=3,733) / f=1.0: 0.597 (n=3,734).
