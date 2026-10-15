## E2 hold-out (fit on 60% of scenarios, predict TTFA of the other 40%)

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| fixed_early_proxy/retrieve/batch | 95 | 3969 | 3580 | 7070 | 6889 | 4110 | 4012 | 0.143 |
| immediate/retrieve/batch | 95 | 4187 | 3782 | 6645 | 7173 | 4341 | 4202 | 0.132 |
| sentence_end/retrieve/batch | 95 | 4313 | 3923 | 6751 | 7295 | 4489 | 4338 | 0.137 |

## E2 5-fold grouped CV (every scenario predicted out-of-fold)

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| fixed_early_proxy/retrieve/batch | 250 | 3688 | 3718 | 7090 | 6859 | 4055 | 4074 | 0.038 |
| immediate/retrieve/batch | 250 | 4070 | 3922 | 6915 | 7047 | 4285 | 4286 | 0.054 |
| sentence_end/retrieve/batch | 250 | 4110 | 3998 | 7052 | 7202 | 4449 | 4376 | 0.062 |

## E3 5-fold grouped CV: paired retrieve-minus-skip TTFA increment

| policy/tts | pairs | obs mean | pred mean | obs median | pred median (of expectations) |
|---|---|---|---|---|---|
| fixed_early_proxy/batch | 250 | 1266 | 1160 | 1125 | 1163 |
| immediate/batch | 250 | 1295 | 1340 | 1196 | 1374 |
| sentence_end/batch | 250 | 1566 | 1557 | 1390 | 1557 |

## E3 5-fold grouped CV: TTFA distributions

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| fixed_early_proxy/retrieve/batch | 250 | 2796 | 2782 | 5154 | 4673 | 2933 | 2858 | 0.080 |
| fixed_early_proxy/skip/batch | 250 | 1610 | 1609 | 2619 | 2609 | 1667 | 1702 | 0.082 |
| immediate/retrieve/batch | 250 | 2954 | 2923 | 4828 | 5047 | 2976 | 3032 | 0.099 |
| immediate/skip/batch | 250 | 1640 | 1609 | 2540 | 2578 | 1681 | 1701 | 0.059 |
| sentence_end/retrieve/batch | 250 | 3102 | 3142 | 5781 | 5298 | 3297 | 3261 | 0.090 |
| sentence_end/skip/batch | 250 | 1625 | 1593 | 2668 | 2609 | 1731 | 1695 | 0.039 |

## v2 real-endpoint runs, 5-fold grouped CV (real piper TTS)

| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |
|---|---|---|---|---|---|---|---|---|
| sentence_end/retrieve/batch | 50 | 2664 | 2704 | 4548 | 4349 | 2816 | 2742 | 0.084 |
| sentence_end/retrieve/streaming | 50 | 1836 | 1818 | 3085 | 3103 | 1946 | 1933 | 0.088 |
| sentence_end/skip/batch | 50 | 1696 | 1737 | 3410 | 3177 | 1890 | 1833 | 0.070 |
| sentence_end/skip/streaming | 50 | 1430 | 1416 | 2590 | 2469 | 1499 | 1510 | 0.138 |

| policy/tts | pairs | obs mean | pred mean | obs median | pred median (of expectations) |
|---|---|---|---|---|---|
| sentence_end/batch | 50 | 926 | 964 | 874 | 945 |
| sentence_end/streaming | 50 | 447 | 453 | 430 | 448 |

