| group | n | TTFA p50 | TTFA p95 | TTFA mean | dispatch_wait mean (share) | retrieval_exposed mean (share) | pre_llm_overhead mean (share) | llm_prefill mean (share) | llm_to_tts_input mean (share) | tts_first_audio mean (share) |
|---|---|---|---|---|---|---|---|---|---|---|
| sentence_end | 250 | 4110 | 7052 | 4449 | 19 (0.4%) | 1311 (29.5%) | 7 (0.2%) | 1065 (23.9%) | 1997 (44.9%) | 49 (1.1%) |
| immediate | 250 | 4070 | 6915 | 4285 | 6 (0.1%) | 1219 (28.4%) | 6 (0.1%) | 1058 (24.7%) | 1979 (46.2%) | 17 (0.4%) |
| fixed_early_proxy | 250 | 3688 | 7090 | 4055 | 2 (0.1%) | 992 (24.5%) | 8 (0.2%) | 1055 (26.0%) | 1979 (48.8%) | 18 (0.5%) |

Stage medians / p95 (ms):

| group | dispatch_wait | retrieval_exposed | pre_llm_overhead | llm_prefill | llm_to_tts_input | tts_first_audio |
|---|---|---|---|---|---|---|
| sentence_end | 0 / 110 | 657 / 3646 | 0 / 24 | 1047 / 1219 | 2047 / 2986 | 16 / 212 |
| immediate | 0 / 31 | 718 / 3602 | 0 / 16 | 1032 / 1203 | 2000 / 2986 | 15 / 55 |
| fixed_early_proxy | 0 / 16 | 304 / 3930 | 0 / 32 | 1031 / 1203 | 2031 / 2961 | 16 / 62 |

Event offsets vs user_end, median / p95 (ms; negative = during speech):

| group | asr_final | controller_action | retrieval_launched | retrieval_result | llm_request | llm_first_token | llm_first_clause | llm_final | tts_first_audio |
|---|---|---|---|---|---|---|---|---|---|
| sentence_end | 0 / 63 | 0 / 110 | 0 / 110 | 657 / 3737 | 657 / 3779 | 1664 / 4940 | – / – | 4070 / 6948 | 4110 / 7052 |
| immediate | 0 / 16 | 0 / 0 | 0 / 31 | 726 / 3636 | 726 / 3673 | 1758 / 4705 | – / – | 4016 / 6877 | 4070 / 6915 |
| fixed_early_proxy | 0 / 47 | -406 / 0 | -391 / 16 | 304 / 3930 | 304 / 3945 | 1336 / 4994 | – / – | 3688 / 7074 | 3688 / 7090 |

Retrieval duration (launch -> result) and hidden part, median / p95 (ms):

| group | retrieval duration | hidden in speech | TTFT | generation | n_tokens |
|---|---|---|---|---|---|
| sentence_end | 657 / 3646 | 0 / 0 | 1047 / 1219 | 2047 / 2986 | 63 / 91 |
| immediate | 812 / 3756 | 0 / 406 | 1032 / 1203 | 2000 / 2986 | 62 / 91 |
| fixed_early_proxy | 625 / 4100 | 219 / 743 | 1031 / 1203 | 2031 / 2961 | 62 / 91 |
