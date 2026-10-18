# v2 test results

**Status. The pre-registered PRIMARY test (pooled over d1_open and d2_support) and every result that uses d2_support's judge-graded outcomes — d2_support itself, every pooled result, the 6-test secondary family, the 48-cell predicted-vs-observed check and the two-dataset E1 kill rule — are EXPLORATORY (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)). The d1_open (alias-graded) results below are the pre-registered per-dataset analysis (unaffected by D3); they are not the pooled confirmatory test.**

Frozen code `ddde67f` + D1 (PREREG_V2 Deviations D1-D3; notes N1, N3). Primary cell: rewrite_faiss, 0.65 belief-quantile price, primary ASR condition. D\*[COMMIT] = B9+; D\*[SPEC] = B9+ (d1_open), SPEC (d2_support); m₃ = 0.016. Paired cluster bootstrap, B = 10,000, stratified by dataset, pooled weights 3,734 : 498. Source: `docs/research_v2/v2_test_results_final.json`.

## d1_open, primary cell: pre-registered per-dataset analysis (unaffected by D3)

| | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |
|---|---|---|---|
| d1_open | -0.0367 [-0.0481, -0.0252] | -0.0415 [-0.0528, -0.0302] | +0.0047 [+0.0007, +0.0088] |

| test | p | rejects | Holm (α = 0.025) |
|---|---|---|---|
| P1: Δ[C] > 0 | 1.0000 | False | False |
| P2: Δ[C] − Δ[S] > 0 | 0.0097 | True | False |
| P3: TOST ±0.016 | upper < 1e-04 / lower 1.0000 | False | False |
| H-REV: Δ[C] < 0 | < 1e-04 | True | True |

**Boundary claim (IUT P1 ∧ P2 ∧ P3), d1_open: False.**

Secondary (d1_open; two-sided, Holm α = 0.05 over its 3 tests): delta_commit p < 1e-04 rejects True; delta_speculative p < 1e-04 rejects True; interaction p 0.0194 rejects True

Predicted vs observed (d1_open): 24 cells: Spearman 0.870 (one-sided 95% lower 0.794); IVW slope 0.827 [0.582, 1.077]. Verdict: **agree**.

H-E1v2 (d1_open): AUC 0.665 [0.648, 0.682]; success False; C2 available False.

# EXPLORATORY (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)): d2_support and pooled

## Headline (primary cell) *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*

| | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |
|---|---|---|---|
| d1_open *[pre-registered per-dataset analysis (unaffected by D3)]* | -0.0367 [-0.0481, -0.0252] | -0.0415 [-0.0528, -0.0302] | +0.0047 [+0.0007, +0.0088] |
| d2_support *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]* | -0.0227 [-0.0502, +0.0047] | -0.0061 [-0.0380, +0.0255] | -0.0166 [-0.0381, +0.0044] |
| **pooled** *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]* | -0.0351 [-0.0458, -0.0245] | -0.0373 [-0.0480, -0.0267] | +0.0022 [-0.0020, +0.0065] |

## Pooled tests: the pre-registered PRIMARY (one-sided α = 0.025) *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*

| test | p | rejects | Holm (α = 0.025) |
|---|---|---|---|
| P1: Δ[C] > 0 | 1.0000 | False | False |
| P2: Δ[C] − Δ[S] > 0 | 0.1603 | False | False |
| P3: TOST ±0.016 | upper < 1e-04 / lower 0.9998 | False | False |
| H-REV: Δ[C] < 0 | < 1e-04 | True | True |

**Boundary claim (IUT P1 ∧ P2 ∧ P3), pooled: False** *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*.

## d2_support per-dataset tests *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*

| test | p | rejects | Holm (α = 0.025) |
|---|---|---|---|
| P1: Δ[C] > 0 | 0.9443 | False | False |
| P2: Δ[C] − Δ[S] > 0 | 0.9396 | False | False |
| P3: TOST ±0.016 | upper 0.0860 / lower 0.2594 | False | False |
| H-REV: Δ[C] < 0 | 0.0557 | False | False |

## Secondary family (per dataset, two-sided, Holm α = 0.05 over 6) *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*

| test | p | rejects |
|---|---|---|
| d1_open|delta_commit | < 1e-04 | True |
| d1_open|delta_speculative | < 1e-04 | True |
| d1_open|interaction | 0.0194 | False |
| d2_support|delta_commit | 0.1114 | False |
| d2_support|interaction | 0.1208 | False |
| d2_support|delta_speculative | 0.7166 | False |

## Predicted vs observed (Δ[COMMIT], both datasets) *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*

48 cells: Spearman 0.537 (one-sided 95% lower 0.248); IVW slope 0.762 [0.524, 1.006]. Verdict: **agree**.

## H-E1v2 gate (test)

* d1_open *[pre-registered per-dataset analysis (unaffected by D3)]*: AUC 0.665 [0.648, 0.682]; success False; C2 available False.
* d2_support *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*: AUC 0.701 [0.638, 0.761]; success True; C2 available False.
* Kill (AUC < 0.60 on both) *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*: False.

## Judge checks (d2_support) *[EXPLORATORY (D3: PRE-REGISTERED JUDGE-VALIDATION GATE NOT PASSED (KAPPA_HUMAN 0.52; JUDGE KAPPA VS REFERENCE 0.55 PRIMARY / 0.58 GPT-4O))]*

* Cross-key consistency: 111 (item, answer) texts graded by both label.py (k = 1 key) and Phase A (k = 3 key, sample 0); primary agreement 0.972972972972973.
* E1 label draws, primary vs secondary: pre-registered stratified 30% subsample {'n': 747, 'agreement': 0.9089692101740294}; all draws (superset run) {'n': 2490, 'agreement': 0.9056224899598394}.
* Phase A answers, primary vs secondary (30% subsample): {'n': 10330, 'agreement': 0.886350435624395}.

## All cells (descriptive)

| dataset | condition | profile | q | Δ[COMMIT] | Δ[SPEC] | interaction | status |
|---|---|---|---|---|---|---|---|
| d1_open | ASR | rewrite_faiss | 0.5 | -0.0365 [-0.0477, -0.0252] | -0.0395 [-0.0507, -0.0285] | +0.0030 [-0.0010, +0.0070] | descriptive |
| d1_open | ASR | rewrite_faiss | 0.65 | -0.0367 [-0.0481, -0.0252] | -0.0415 [-0.0528, -0.0302] | +0.0047 [+0.0007, +0.0088] | descriptive |
| d1_open | ASR | rewrite_faiss | 0.8 | -0.0409 [-0.0530, -0.0292] | -0.0453 [-0.0571, -0.0337] | +0.0044 [+0.0003, +0.0084] | descriptive |
| d1_open | ASR | rewrite_faiss | 0.9 | -0.0432 [-0.0559, -0.0309] | -0.0456 [-0.0576, -0.0336] | +0.0023 [-0.0019, +0.0067] | descriptive |
| d1_open | ASR | rewrite_faiss_rerank | 0.5 | -0.0354 [-0.0466, -0.0240] | -0.0385 [-0.0497, -0.0276] | +0.0031 [-0.0008, +0.0071] | descriptive |
| d1_open | ASR | rewrite_faiss_rerank | 0.65 | -0.0363 [-0.0477, -0.0248] | -0.0414 [-0.0527, -0.0302] | +0.0051 [+0.0010, +0.0092] | descriptive |
| d1_open | ASR | rewrite_faiss_rerank | 0.8 | -0.0409 [-0.0529, -0.0293] | -0.0433 [-0.0550, -0.0320] | +0.0024 [-0.0017, +0.0065] | descriptive |
| d1_open | ASR | rewrite_faiss_rerank | 0.9 | -0.0416 [-0.0543, -0.0292] | -0.0442 [-0.0563, -0.0322] | +0.0026 [-0.0015, +0.0068] | descriptive |
| d1_open | ASR | rewrite_remote | 0.5 | -0.0415 [-0.0531, -0.0301] | -0.0417 [-0.0533, -0.0301] | +0.0001 [-0.0040, +0.0043] | descriptive |
| d1_open | ASR | rewrite_remote | 0.65 | -0.0432 [-0.0564, -0.0300] | -0.0435 [-0.0566, -0.0304] | +0.0003 [-0.0040, +0.0046] | descriptive |
| d1_open | ASR | rewrite_remote | 0.8 | -0.0433 [-0.0559, -0.0311] | -0.0443 [-0.0564, -0.0322] | +0.0010 [-0.0033, +0.0054] | descriptive |
| d1_open | ASR | rewrite_remote | 0.9 | -0.0456 [-0.0585, -0.0332] | -0.0444 [-0.0568, -0.0323] | -0.0012 [-0.0056, +0.0031] | descriptive |
| d1_open | text150 | rewrite_faiss | 0.5 | -0.0085 [-0.0119, -0.0052] | -0.0085 [-0.0119, -0.0052] | +0.0000 [-0.0001, +0.0001] | descriptive |
| d1_open | text150 | rewrite_faiss | 0.65 | -0.0024 [-0.0072, +0.0022] | -0.0024 [-0.0072, +0.0022] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_faiss | 0.8 | -0.0010 [-0.0033, +0.0014] | -0.0010 [-0.0033, +0.0014] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_faiss | 0.9 | -0.0028 [-0.0055, -0.0000] | -0.0028 [-0.0055, -0.0000] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_faiss_rerank | 0.5 | -0.0088 [-0.0122, -0.0055] | -0.0088 [-0.0122, -0.0055] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_faiss_rerank | 0.65 | -0.0028 [-0.0075, +0.0019] | -0.0028 [-0.0075, +0.0019] | -0.0000 [-0.0002, +0.0002] | descriptive |
| d1_open | text150 | rewrite_faiss_rerank | 0.8 | -0.0010 [-0.0033, +0.0014] | -0.0010 [-0.0033, +0.0014] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_faiss_rerank | 0.9 | +0.0013 [-0.0070, +0.0101] | +0.0013 [-0.0070, +0.0101] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_remote | 0.5 | -0.0076 [-0.0110, -0.0042] | -0.0076 [-0.0110, -0.0042] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_remote | 0.65 | -0.0023 [-0.0071, +0.0024] | -0.0023 [-0.0071, +0.0024] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_remote | 0.8 | -0.0002 [-0.0027, +0.0023] | -0.0002 [-0.0027, +0.0023] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d1_open | text150 | rewrite_remote | 0.9 | -0.0033 [-0.0065, -0.0000] | -0.0033 [-0.0065, -0.0000] | +0.0000 [+0.0000, +0.0000] | descriptive |
| d2_support | ASR | rewrite_faiss | 0.5 | -0.0095 [-0.0408, +0.0219] | -0.0066 [-0.0381, +0.0241] | -0.0029 [-0.0135, +0.0080] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_faiss | 0.65 | -0.0227 [-0.0502, +0.0047] | -0.0061 [-0.0380, +0.0255] | -0.0166 [-0.0381, +0.0044] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_faiss | 0.8 | -0.0141 [-0.0463, +0.0177] | -0.0059 [-0.0378, +0.0260] | -0.0083 [-0.0201, +0.0032] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_faiss | 0.9 | +0.0136 [-0.0098, +0.0370] | +0.0181 [-0.0051, +0.0418] | -0.0046 [-0.0147, +0.0059] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_faiss_rerank | 0.5 | -0.0089 [-0.0406, +0.0230] | -0.0070 [-0.0387, +0.0241] | -0.0019 [-0.0125, +0.0090] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_faiss_rerank | 0.65 | -0.0237 [-0.0517, +0.0040] | -0.0069 [-0.0390, +0.0251] | -0.0168 [-0.0383, +0.0044] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_faiss_rerank | 0.8 | -0.0145 [-0.0470, +0.0177] | -0.0067 [-0.0391, +0.0254] | -0.0078 [-0.0195, +0.0037] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_faiss_rerank | 0.9 | +0.0136 [-0.0096, +0.0369] | +0.0181 [-0.0051, +0.0417] | -0.0044 [-0.0145, +0.0059] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_remote | 0.5 | -0.0033 [-0.0348, +0.0285] | +0.0142 [-0.0092, +0.0379] | -0.0174 [-0.0491, +0.0145] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_remote | 0.65 | +0.0124 [-0.0119, +0.0368] | +0.0134 [-0.0100, +0.0373] | -0.0011 [-0.0111, +0.0096] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_remote | 0.8 | +0.0132 [-0.0109, +0.0377] | +0.0142 [-0.0092, +0.0379] | -0.0010 [-0.0111, +0.0096] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | ASR | rewrite_remote | 0.9 | +0.0140 [-0.0101, +0.0385] | +0.0139 [-0.0091, +0.0374] | +0.0000 [-0.0103, +0.0108] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss | 0.5 | -0.0129 [-0.0284, +0.0026] | -0.0053 [-0.0206, +0.0103] | -0.0075 [-0.0200, +0.0044] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss | 0.65 | -0.0041 [-0.0197, +0.0119] | -0.0025 [-0.0180, +0.0136] | -0.0017 [-0.0043, -0.0000] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss | 0.8 | -0.0073 [-0.0224, +0.0079] | -0.0099 [-0.0245, +0.0049] | +0.0026 [-0.0084, +0.0142] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss | 0.9 | -0.0067 [-0.0215, +0.0084] | -0.0047 [-0.0194, +0.0103] | -0.0020 [-0.0048, -0.0002] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss_rerank | 0.5 | -0.0138 [-0.0295, +0.0018] | -0.0117 [-0.0272, +0.0035] | -0.0021 [-0.0049, -0.0002] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss_rerank | 0.65 | -0.0103 [-0.0260, +0.0058] | -0.0079 [-0.0233, +0.0080] | -0.0024 [-0.0053, -0.0004] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss_rerank | 0.8 | -0.0123 [-0.0273, +0.0027] | -0.0098 [-0.0246, +0.0050] | -0.0025 [-0.0054, -0.0004] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_faiss_rerank | 0.9 | -0.0080 [-0.0228, +0.0070] | -0.0096 [-0.0242, +0.0051] | +0.0016 [-0.0097, +0.0133] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_remote | 0.5 | -0.0051 [-0.0204, +0.0106] | -0.0051 [-0.0204, +0.0106] | +0.0000 [+0.0000, +0.0000] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_remote | 0.65 | -0.0085 [-0.0242, +0.0075] | -0.0025 [-0.0176, +0.0135] | -0.0060 [-0.0179, +0.0055] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_remote | 0.8 | -0.0108 [-0.0253, +0.0042] | -0.0092 [-0.0236, +0.0057] | -0.0016 [-0.0043, +0.0002] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
| d2_support | text150 | rewrite_remote | 0.9 | -0.0108 [-0.0253, +0.0040] | -0.0052 [-0.0197, +0.0097] | -0.0056 [-0.0172, +0.0056] | descriptive; exploratory (D3: pre-registered judge-validation gate not passed (kappa_human 0.52; judge kappa vs reference 0.55 primary / 0.58 GPT-4o)) |
