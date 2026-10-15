| group | n | TTFA p50 | TTFA p95 | TTFA mean | dispatch_wait mean (share) | retrieval_exposed mean (share) | pre_llm_overhead mean (share) | llm_prefill mean (share) | llm_to_tts_input mean (share) | tts_first_audio mean (share) |
|---|---|---|---|---|---|---|---|---|---|---|
| unknown/retrieve/batch | 50 | 2664 | 4548 | 2816 | 0 (0.0%) | 71 (2.5%) | 3 (0.1%) | 428 (15.2%) | 1632 (57.9%) | 682 (24.2%) |
| unknown/retrieve/streaming | 50 | 1836 | 3085 | 1946 | 0 (0.0%) | 76 (3.9%) | 6 (0.3%) | 404 (20.8%) | 790 (40.6%) | 670 (34.4%) |
| unknown/skip/batch | 50 | 1696 | 3410 | 1890 | 0 (0.0%) | 0 (0.0%) | 3 (0.2%) | 342 (18.1%) | 902 (47.7%) | 643 (34.0%) |
| unknown/skip/streaming | 50 | 1430 | 2590 | 1499 | 0 (0.0%) | 0 (0.0%) | 4 (0.3%) | 345 (23.0%) | 590 (39.4%) | 560 (37.3%) |

Stage medians / p95 (ms):

| group | dispatch_wait | retrieval_exposed | pre_llm_overhead | llm_prefill | llm_to_tts_input | tts_first_audio |
|---|---|---|---|---|---|---|
| unknown/retrieve/batch | 0 / 0 | 47 / 256 | 0 / 0 | 328 / 877 | 1640 / 2995 | 617 / 1285 |
| unknown/retrieve/streaming | 0 / 0 | 47 / 236 | 0 / 16 | 328 / 790 | 718 / 1399 | 586 / 1227 |
| unknown/skip/batch | 0 / 0 | 0 / 0 | 0 / 0 | 297 / 623 | 813 / 1793 | 609 / 1149 |
| unknown/skip/streaming | 0 / 0 | 0 / 0 | 0 / 16 | 313 / 743 | 547 / 1016 | 500 / 1246 |

Event offsets vs user_end, median / p95 (ms; negative = during speech):

| group | asr_final | controller_action | retrieval_launched | retrieval_result | llm_request | llm_first_token | llm_first_clause | llm_final | tts_first_audio |
|---|---|---|---|---|---|---|---|---|---|
| unknown/retrieve/batch | 0 / 0 | 0 / 0 | 0 / 0 | 47 / 256 | 47 / 256 | 390 / 931 | – / – | 2132 / 3543 | 2664 / 4548 |
| unknown/retrieve/streaming | 0 / 0 | 0 / 0 | 0 / 0 | 47 / 236 | 54 / 259 | 414 / 925 | 1172 / 1915 | 2211 / 3669 | 1836 / 3085 |
| unknown/skip/batch | 0 / 0 | 0 / 0 | – / – | – / – | 0 / 0 | 297 / 623 | – / – | 1140 / 2409 | 1696 / 3410 |
| unknown/skip/streaming | 0 / 0 | 0 / 0 | – / – | – / – | 0 / 16 | 313 / 752 | 907 / 1415 | 1188 / 2211 | 1430 / 2590 |

Retrieval duration (launch -> result) and hidden part, median / p95 (ms):

| group | retrieval duration | hidden in speech | TTFT | generation | n_tokens |
|---|---|---|---|---|---|
| unknown/retrieve/batch | 47 / 256 | 0 / 0 | 328 / 877 | 1640 / 2995 | 52 / 95 |
| unknown/retrieve/streaming | 47 / 236 | 0 / 0 | 328 / 790 | 1516 / 2946 | 49 / 93 |
| unknown/skip/batch | – / – | – / – | 297 / 623 | 813 / 1793 | 26 / 57 |
| unknown/skip/streaming | – / – | – / – | 313 / 743 | 804 / 1751 | 26 / 56 |
