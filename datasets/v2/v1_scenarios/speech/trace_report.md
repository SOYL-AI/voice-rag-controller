# Speech trace quality

Traces: `datasets/v2/v1_scenarios/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| babble_10dB | 900 | 139 | 20.6% | 15.2% | 14.2% | 67.3% | 1503 | 1971 | 0.0% | 89.1% | 188 | 1332 | 2993 |
| babble_5dB | 900 | 139 | 33.2% | 33.5% | 22.8% | 79.1% | 1054 | 1054 | 0.0% | 99.9% | 194 | 2461 | 3155 |
| clean | 900 | 139 | 20.7% | 9.7% | 14.9% | 54.0% | 831 | 912 | 0.0% | 0.0% | 177 | 310 | 1163 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 130wpm | 900 | 119 | 27.0% | 21.6% | 19.9% | 75.1% | 1198 | 1513 | 0.0% | 62.6% | 187 | 1459 | 2467 |
| 150wpm | 900 | 135 | 24.8% | 19.2% | 17.0% | 66.6% | 1121 | 1339 | 0.0% | 63.8% | 183 | 1431 | 2412 |
| 180wpm | 900 | 162 | 22.7% | 17.6% | 14.9% | 58.8% | 1177 | 1449 | 0.0% | 62.7% | 189 | 1212 | 2431 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium | 450 | 137 | 24.6% | 16.8% | 18.2% | 66.7% | 1298 | 1461 | 0.0% | 65.1% | 147 | 1124 | 2312 |
| piper-en_GB-northern_english_male-medium | 450 | 144 | 26.2% | 20.3% | 17.7% | 70.0% | 1179 | 1483 | 0.0% | 63.1% | 162 | 1059 | 2318 |
| piper-en_US-arctic-medium-spk_ksp | 450 | 139 | 24.6% | 19.4% | 17.5% | 69.6% | 1171 | 1451 | 0.0% | 61.1% | 176 | 934 | 2417 |
| piper-en_US-hfc_male-medium | 450 | 134 | 13.5% | 6.0% | 9.4% | 53.1% | 1165 | 1517 | 0.0% | 60.0% | 214 | 674 | 2523 |
| piper-en_US-l2arctic-medium-spk_SVBI | 450 | 141 | 44.3% | 45.4% | 29.2% | 80.0% | 1019 | 1290 | 0.0% | 64.2% | 185 | 2853 | 2587 |
| piper-en_US-lessac-medium | 450 | 135 | 15.7% | 8.8% | 11.6% | 61.6% | 1152 | 1382 | 0.0% | 64.4% | 234 | 1561 | 2463 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium__130wpm__babble_10dB | 50 | 117 | 19.8% | 12.6% | 16.5% | 82.0% | 1884 | 2192 | 0.0% | 92.0% | 140 | 1205 | 2916 |
| piper-en_GB-cori-medium__130wpm__babble_5dB | 50 | 117 | 35.7% | 33.8% | 29.4% | 86.0% | - | - | 0.0% | 100.0% | 155 | 3486 | 3018 |
| piper-en_GB-cori-medium__130wpm__clean | 50 | 117 | 21.9% | 8.6% | 17.6% | 60.0% | 828 | 959 | 0.0% | 0.0% | 136 | 181 | 1060 |
| piper-en_GB-cori-medium__150wpm__babble_10dB | 50 | 132 | 18.9% | 12.6% | 13.7% | 66.0% | 1830 | 1830 | 0.0% | 98.0% | 148 | 214 | 2892 |
| piper-en_GB-cori-medium__150wpm__babble_5dB | 50 | 132 | 35.4% | 28.1% | 24.7% | 82.0% | - | - | 0.0% | 100.0% | 156 | 1800 | 2946 |
| piper-en_GB-cori-medium__150wpm__clean | 50 | 132 | 22.1% | 9.3% | 17.7% | 54.0% | 828 | 859 | 0.0% | 0.0% | 146 | 183 | 1056 |
| piper-en_GB-cori-medium__180wpm__babble_10dB | 50 | 163 | 15.8% | 8.6% | 10.7% | 52.0% | 1587 | 2079 | 0.0% | 96.0% | 149 | 369 | 2908 |
| piper-en_GB-cori-medium__180wpm__babble_5dB | 50 | 163 | 33.0% | 30.5% | 19.0% | 70.0% | - | - | 0.0% | 100.0% | 150 | 2476 | 2948 |
| piper-en_GB-cori-medium__180wpm__clean | 50 | 163 | 18.9% | 7.3% | 14.9% | 48.0% | 828 | 850 | 0.0% | 0.0% | 142 | 205 | 1065 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_10dB | 50 | 123 | 29.7% | 26.2% | 20.6% | 88.0% | 1661 | 2043 | 0.0% | 94.0% | 148 | 1179 | 2888 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_5dB | 50 | 123 | 36.2% | 28.5% | 28.5% | 88.0% | - | - | 0.0% | 100.0% | 153 | 1454 | 2918 |
| piper-en_GB-northern_english_male-medium__130wpm__clean | 50 | 123 | 25.7% | 13.2% | 18.3% | 70.0% | 833 | 1030 | 0.0% | 0.0% | 149 | 190 | 1100 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_10dB | 50 | 142 | 21.2% | 16.9% | 13.4% | 70.0% | 1436 | 1751 | 0.0% | 88.0% | 160 | 847 | 2886 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_5dB | 50 | 142 | 34.8% | 40.1% | 20.4% | 78.0% | - | - | 0.0% | 100.0% | 177 | 3318 | 3031 |
| piper-en_GB-northern_english_male-medium__150wpm__clean | 50 | 142 | 20.5% | 12.9% | 16.1% | 58.0% | 834 | 976 | 0.0% | 0.0% | 150 | 192 | 1110 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_10dB | 50 | 168 | 21.0% | 14.6% | 11.6% | 60.0% | 1478 | 2139 | 0.0% | 86.0% | 176 | 263 | 2882 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_5dB | 50 | 168 | 28.1% | 23.5% | 19.0% | 72.0% | - | - | 0.0% | 100.0% | 177 | 1882 | 2934 |
| piper-en_GB-northern_english_male-medium__180wpm__clean | 50 | 168 | 18.2% | 7.3% | 11.4% | 46.0% | 833 | 959 | 0.0% | 0.0% | 165 | 206 | 1114 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_10dB | 50 | 118 | 20.5% | 13.6% | 16.6% | 88.0% | 1620 | 2044 | 0.0% | 82.0% | 177 | 679 | 3002 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_5dB | 50 | 118 | 38.6% | 47.0% | 32.8% | 90.0% | - | - | 0.0% | 100.0% | 188 | 3574 | 3161 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__clean | 50 | 118 | 20.0% | 8.6% | 16.2% | 60.0% | 825 | 928 | 0.0% | 0.0% | 176 | 248 | 1163 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 50 | 136 | 18.4% | 10.6% | 12.7% | 68.0% | 1617 | 2272 | 0.0% | 90.0% | 166 | 298 | 3002 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_5dB | 50 | 136 | 37.2% | 32.5% | 20.4% | 78.0% | 1054 | 1054 | 0.0% | 98.0% | 176 | 1514 | 3048 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__clean | 50 | 136 | 18.3% | 7.3% | 11.2% | 46.0% | 823 | 954 | 0.0% | 0.0% | 165 | 206 | 1143 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_10dB | 50 | 162 | 19.0% | 12.6% | 14.3% | 72.0% | 1440 | 1993 | 0.0% | 80.0% | 182 | 435 | 3011 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_5dB | 50 | 162 | 28.1% | 31.5% | 18.9% | 76.0% | - | - | 0.0% | 100.0% | 182 | 1182 | 3045 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__clean | 50 | 162 | 21.7% | 10.6% | 14.8% | 48.0% | 820 | 911 | 0.0% | 0.0% | 174 | 268 | 1179 |
| piper-en_US-hfc_male-medium__130wpm__babble_10dB | 50 | 114 | 12.4% | 5.0% | 9.3% | 58.0% | 1414 | 1987 | 0.0% | 72.0% | 223 | 882 | 3169 |
| piper-en_US-hfc_male-medium__130wpm__babble_5dB | 50 | 114 | 15.7% | 11.6% | 15.7% | 88.0% | - | - | 0.0% | 100.0% | 227 | 1268 | 3344 |
| piper-en_US-hfc_male-medium__130wpm__clean | 50 | 114 | 16.0% | 4.0% | 9.2% | 48.0% | 831 | 900 | 0.0% | 0.0% | 224 | 308 | 1260 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 50 | 132 | 11.0% | 4.0% | 7.2% | 48.0% | 1406 | 2061 | 0.0% | 90.0% | 225 | 822 | 3204 |
| piper-en_US-hfc_male-medium__150wpm__babble_5dB | 50 | 132 | 13.6% | 8.6% | 10.1% | 66.0% | - | - | 0.0% | 100.0% | 225 | 1312 | 3208 |
| piper-en_US-hfc_male-medium__150wpm__clean | 50 | 132 | 14.3% | 3.6% | 10.8% | 40.0% | 828 | 868 | 0.0% | 0.0% | 174 | 256 | 1138 |
| piper-en_US-hfc_male-medium__180wpm__babble_10dB | 50 | 155 | 10.3% | 5.0% | 5.9% | 44.0% | 1678 | 2442 | 0.0% | 78.0% | 227 | 295 | 3065 |
| piper-en_US-hfc_male-medium__180wpm__babble_5dB | 50 | 155 | 12.6% | 8.6% | 9.1% | 54.0% | - | - | 0.0% | 100.0% | 184 | 608 | 3072 |
| piper-en_US-hfc_male-medium__180wpm__clean | 50 | 155 | 15.2% | 3.6% | 7.8% | 32.0% | 832 | 843 | 0.0% | 0.0% | 221 | 312 | 1246 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_10dB | 50 | 123 | 41.2% | 37.4% | 27.5% | 86.0% | 1354 | 2110 | 0.0% | 92.0% | 168 | 2242 | 2970 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_5dB | 50 | 123 | 66.2% | 86.8% | 42.3% | 94.0% | - | - | 0.0% | 100.0% | 191 | 5925 | 3864 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__clean | 50 | 123 | 34.7% | 26.5% | 22.9% | 74.0% | 851 | 954 | 0.0% | 0.0% | 153 | 403 | 1140 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 50 | 136 | 37.4% | 34.1% | 27.8% | 84.0% | 994 | 1641 | 0.0% | 88.0% | 198 | 2607 | 2977 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_5dB | 50 | 136 | 62.4% | 76.2% | 37.8% | 90.0% | - | - | 0.0% | 100.0% | 207 | 5271 | 3206 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__clean | 50 | 136 | 30.9% | 17.2% | 22.2% | 68.0% | 838 | 900 | 0.0% | 0.0% | 182 | 793 | 1185 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_10dB | 50 | 165 | 36.2% | 40.7% | 21.3% | 70.0% | 1236 | 1236 | 0.0% | 98.0% | 182 | 3043 | 2932 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_5dB | 50 | 165 | 62.4% | 72.2% | 37.8% | 94.0% | - | - | 0.0% | 100.0% | 218 | 5133 | 3852 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__clean | 50 | 165 | 27.3% | 17.5% | 22.8% | 60.0% | 838 | 896 | 0.0% | 0.0% | 169 | 263 | 1161 |
| piper-en_US-lessac-medium__130wpm__babble_10dB | 50 | 117 | 14.0% | 6.0% | 11.6% | 70.0% | 1438 | 2089 | 0.0% | 94.0% | 252 | 798 | 3023 |
| piper-en_US-lessac-medium__130wpm__babble_5dB | 50 | 117 | 20.7% | 12.9% | 14.2% | 68.0% | - | - | 0.0% | 100.0% | 249 | 1582 | 3038 |
| piper-en_US-lessac-medium__130wpm__clean | 50 | 117 | 17.0% | 6.6% | 9.3% | 54.0% | 831 | 916 | 0.0% | 0.0% | 256 | 652 | 1375 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 50 | 133 | 13.0% | 6.6% | 8.6% | 54.0% | 1262 | 1374 | 0.0% | 96.0% | 235 | 4665 | 3113 |
| piper-en_US-lessac-medium__150wpm__babble_5dB | 50 | 133 | 19.3% | 17.9% | 16.7% | 82.0% | - | - | 0.0% | 100.0% | 237 | 1084 | 3109 |
| piper-en_US-lessac-medium__150wpm__clean | 50 | 133 | 16.8% | 6.6% | 14.9% | 66.0% | 828 | 867 | 0.0% | 0.0% | 172 | 384 | 1158 |
| piper-en_US-lessac-medium__180wpm__babble_10dB | 50 | 156 | 10.9% | 6.0% | 6.9% | 52.0% | 1718 | 2195 | 0.0% | 90.0% | 234 | 3128 | 3028 |
| piper-en_US-lessac-medium__180wpm__babble_5dB | 50 | 156 | 16.6% | 12.3% | 12.8% | 68.0% | - | - | 0.0% | 100.0% | 240 | 1426 | 3040 |
| piper-en_US-lessac-medium__180wpm__clean | 50 | 156 | 12.8% | 4.0% | 9.8% | 40.0% | 832 | 848 | 0.0% | 0.0% | 228 | 327 | 1282 |
