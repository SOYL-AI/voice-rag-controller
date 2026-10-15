## E2 hold-out (fit on 60% of scenarios, predict TTFA of the other 40%)

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| fixed_early_proxy/retrieve/batch | 95 | 3969 | 3580 | 7070 | 6889 | 4110 | 4012 | 0.143 |
| immediate/retrieve/batch | 95 | 4187 | 3782 | 6645 | 7173 | 4341 | 4202 | 0.132 |
| sentence_end/retrieve/batch | 95 | 4313 | 3923 | 6751 | 7295 | 4489 | 4338 | 0.137 |

## E2 5-fold grouped CV (every scenario predicted out-of-fold)

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| fixed_early_proxy/retrieve/batch | 250 | 3688 | 3704 | 7090 | 6844 | 4055 | 4074 | 0.038 |
| immediate/retrieve/batch | 250 | 4070 | 3922 | 6915 | 7047 | 4285 | 4285 | 0.054 |
| sentence_end/retrieve/batch | 250 | 4110 | 3999 | 7052 | 7203 | 4449 | 4376 | 0.062 |

## E3 5-fold grouped CV: paired retrieve-minus-skip TTFA increment

| policy/tts | pairs | obs mean | pred mean | obs median | pred median (of expectations) |
|---|---|---|---|---|---|
| fixed_early_proxy/batch | 250 | 1266 | 1160 | 1125 | 1163 |
| immediate/batch | 250 | 1295 | 1341 | 1196 | 1376 |
| sentence_end/batch | 250 | 1566 | 1557 | 1390 | 1558 |

## E3 5-fold grouped CV: TTFA distributions

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| fixed_early_proxy/retrieve/batch | 250 | 2796 | 2782 | 5154 | 4673 | 2933 | 2858 | 0.078 |
| fixed_early_proxy/skip/batch | 250 | 1610 | 1609 | 2619 | 2625 | 1667 | 1702 | 0.082 |
| immediate/retrieve/batch | 250 | 2954 | 2923 | 4828 | 5047 | 2976 | 3033 | 0.099 |
| immediate/skip/batch | 250 | 1640 | 1608 | 2540 | 2578 | 1681 | 1702 | 0.059 |
| sentence_end/retrieve/batch | 250 | 3102 | 3141 | 5781 | 5297 | 3297 | 3260 | 0.090 |
| sentence_end/skip/batch | 250 | 1625 | 1593 | 2668 | 2594 | 1731 | 1695 | 0.039 |

## v2 real-endpoint runs, 5-fold grouped CV (real piper TTS)

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| sentence_end/retrieve/batch | 50 | 2664 | 2707 | 4548 | 4328 | 2816 | 2736 | 0.087 |
| sentence_end/retrieve/streaming | 50 | 1836 | 1811 | 3085 | 3061 | 1946 | 1924 | 0.086 |
| sentence_end/skip/batch | 50 | 1696 | 1739 | 3410 | 3190 | 1890 | 1835 | 0.068 |
| sentence_end/skip/streaming | 50 | 1430 | 1416 | 2590 | 2470 | 1499 | 1511 | 0.140 |

| policy/tts | pairs | obs mean | pred mean | obs median | pred median (of expectations) |
|---|---|---|---|---|---|
| sentence_end/batch | 50 | 926 | 968 | 874 | 958 |
| sentence_end/streaming | 50 | 447 | 450 | 430 | 442 |

