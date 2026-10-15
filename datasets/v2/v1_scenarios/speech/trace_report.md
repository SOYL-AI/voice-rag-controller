# Speech trace quality

Traces: `datasets/v2/v1_scenarios/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| babble_10dB | 900 | 139 | 20.7% | 14.8% | 14.2% | 67.4% | 1503 | 1971 | 0.0% | 89.1% | 190 | 1807 | 3004 |
| babble_5dB | 900 | 139 | 33.0% | 33.0% | 22.6% | 78.6% | 1054 | 1054 | 0.0% | 99.9% | 196 | 3229 | 3185 |
| clean | 900 | 139 | 20.7% | 9.7% | 14.9% | 54.0% | 831 | 912 | 0.0% | 0.0% | 177 | 315 | 1164 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 130wpm | 900 | 119 | 26.9% | 20.7% | 19.8% | 74.8% | 1198 | 1513 | 0.0% | 62.6% | 189 | 1840 | 2482 |
| 150wpm | 900 | 135 | 24.8% | 19.2% | 16.9% | 66.6% | 1121 | 1339 | 0.0% | 63.8% | 185 | 1858 | 2417 |
| 180wpm | 900 | 162 | 22.6% | 17.6% | 14.9% | 58.7% | 1177 | 1449 | 0.0% | 62.7% | 190 | 1653 | 2454 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium | 450 | 137 | 24.7% | 16.8% | 18.2% | 66.9% | 1298 | 1461 | 0.0% | 65.1% | 146 | 1136 | 2311 |
| piper-en_GB-northern_english_male-medium | 450 | 144 | 26.0% | 19.5% | 17.7% | 69.6% | 1179 | 1483 | 0.0% | 63.1% | 162 | 1252 | 2321 |
| piper-en_US-arctic-medium-spk_ksp | 450 | 139 | 24.6% | 18.5% | 17.5% | 69.3% | 1171 | 1451 | 0.0% | 61.1% | 177 | 1423 | 2429 |
| piper-en_US-hfc_male-medium | 450 | 134 | 13.5% | 6.0% | 9.5% | 53.3% | 1165 | 1517 | 0.0% | 60.0% | 216 | 1167 | 2543 |
| piper-en_US-l2arctic-medium-spk_SVBI | 450 | 141 | 44.4% | 45.5% | 29.0% | 79.8% | 1019 | 1290 | 0.0% | 64.2% | 187 | 3536 | 2624 |
| piper-en_US-lessac-medium | 450 | 135 | 15.7% | 8.8% | 11.6% | 61.1% | 1152 | 1382 | 0.0% | 64.4% | 238 | 2188 | 2479 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium__130wpm__babble_10dB | 50 | 117 | 19.7% | 12.6% | 16.3% | 82.0% | 1884 | 2192 | 0.0% | 92.0% | 138 | 1636 | 2916 |
| piper-en_GB-cori-medium__130wpm__babble_5dB | 50 | 117 | 35.8% | 33.8% | 29.2% | 86.0% | - | - | 0.0% | 100.0% | 154 | 3289 | 3018 |
| piper-en_GB-cori-medium__130wpm__clean | 50 | 117 | 22.0% | 8.6% | 17.6% | 60.0% | 828 | 959 | 0.0% | 0.0% | 136 | 181 | 1060 |
| piper-en_GB-cori-medium__150wpm__babble_10dB | 50 | 132 | 18.9% | 12.6% | 13.5% | 66.0% | 1830 | 1830 | 0.0% | 98.0% | 148 | 214 | 2892 |
| piper-en_GB-cori-medium__150wpm__babble_5dB | 50 | 132 | 35.5% | 28.1% | 24.6% | 82.0% | - | - | 0.0% | 100.0% | 156 | 1718 | 2930 |
| piper-en_GB-cori-medium__150wpm__clean | 50 | 132 | 22.1% | 9.3% | 17.7% | 54.0% | 828 | 859 | 0.0% | 0.0% | 146 | 183 | 1056 |
| piper-en_GB-cori-medium__180wpm__babble_10dB | 50 | 163 | 15.9% | 8.6% | 10.9% | 54.0% | 1587 | 2079 | 0.0% | 96.0% | 147 | 369 | 2908 |
| piper-en_GB-cori-medium__180wpm__babble_5dB | 50 | 163 | 33.2% | 30.5% | 19.0% | 70.0% | - | - | 0.0% | 100.0% | 150 | 2433 | 2956 |
| piper-en_GB-cori-medium__180wpm__clean | 50 | 163 | 19.0% | 7.3% | 14.9% | 48.0% | 828 | 850 | 0.0% | 0.0% | 142 | 205 | 1065 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_10dB | 50 | 123 | 28.7% | 18.9% | 20.4% | 88.0% | 1661 | 2043 | 0.0% | 94.0% | 148 | 1179 | 2888 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_5dB | 50 | 123 | 36.4% | 28.1% | 28.4% | 88.0% | - | - | 0.0% | 100.0% | 153 | 1435 | 2918 |
| piper-en_GB-northern_english_male-medium__130wpm__clean | 50 | 123 | 25.6% | 13.2% | 18.3% | 70.0% | 833 | 1030 | 0.0% | 0.0% | 149 | 192 | 1102 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_10dB | 50 | 142 | 21.2% | 16.9% | 13.3% | 70.0% | 1436 | 1751 | 0.0% | 88.0% | 160 | 1072 | 2885 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_5dB | 50 | 142 | 33.3% | 40.1% | 20.4% | 76.0% | - | - | 0.0% | 100.0% | 176 | 3208 | 3031 |
| piper-en_GB-northern_english_male-medium__150wpm__clean | 50 | 142 | 20.5% | 12.9% | 16.1% | 58.0% | 834 | 976 | 0.0% | 0.0% | 150 | 192 | 1110 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_10dB | 50 | 168 | 20.9% | 14.6% | 11.8% | 60.0% | 1478 | 2139 | 0.0% | 86.0% | 176 | 307 | 2886 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_5dB | 50 | 168 | 29.1% | 23.5% | 18.8% | 70.0% | - | - | 0.0% | 100.0% | 180 | 3473 | 2956 |
| piper-en_GB-northern_english_male-medium__180wpm__clean | 50 | 168 | 18.2% | 7.3% | 11.4% | 46.0% | 833 | 959 | 0.0% | 0.0% | 165 | 206 | 1114 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_10dB | 50 | 118 | 20.4% | 13.6% | 16.0% | 86.0% | 1620 | 2044 | 0.0% | 82.0% | 177 | 964 | 3006 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_5dB | 50 | 118 | 38.4% | 38.7% | 32.1% | 90.0% | - | - | 0.0% | 100.0% | 191 | 4477 | 3224 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__clean | 50 | 118 | 20.0% | 8.6% | 16.2% | 60.0% | 825 | 928 | 0.0% | 0.0% | 176 | 248 | 1163 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 50 | 136 | 18.4% | 10.6% | 12.7% | 68.0% | 1617 | 2272 | 0.0% | 90.0% | 166 | 298 | 3002 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_5dB | 50 | 136 | 37.2% | 32.5% | 21.0% | 78.0% | 1054 | 1054 | 0.0% | 98.0% | 178 | 1685 | 3048 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__clean | 50 | 136 | 18.3% | 7.3% | 11.2% | 46.0% | 823 | 954 | 0.0% | 0.0% | 165 | 206 | 1143 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_10dB | 50 | 162 | 19.0% | 12.9% | 14.3% | 72.0% | 1440 | 1993 | 0.0% | 80.0% | 183 | 1562 | 3026 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_5dB | 50 | 162 | 28.1% | 31.5% | 18.9% | 76.0% | - | - | 0.0% | 100.0% | 183 | 3102 | 3068 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__clean | 50 | 162 | 21.7% | 10.6% | 14.8% | 48.0% | 820 | 911 | 0.0% | 0.0% | 174 | 268 | 1179 |
| piper-en_US-hfc_male-medium__130wpm__babble_10dB | 50 | 114 | 12.4% | 5.0% | 9.3% | 58.0% | 1414 | 1987 | 0.0% | 72.0% | 229 | 1951 | 3202 |
| piper-en_US-hfc_male-medium__130wpm__babble_5dB | 50 | 114 | 15.7% | 11.6% | 15.7% | 88.0% | - | - | 0.0% | 100.0% | 230 | 1860 | 3412 |
| piper-en_US-hfc_male-medium__130wpm__clean | 50 | 114 | 16.0% | 4.0% | 9.2% | 48.0% | 831 | 900 | 0.0% | 0.0% | 224 | 308 | 1262 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 50 | 132 | 11.0% | 4.0% | 7.3% | 50.0% | 1406 | 2061 | 0.0% | 90.0% | 225 | 880 | 3204 |
| piper-en_US-hfc_male-medium__150wpm__babble_5dB | 50 | 132 | 13.6% | 8.6% | 10.3% | 66.0% | - | - | 0.0% | 100.0% | 229 | 3857 | 3236 |
| piper-en_US-hfc_male-medium__150wpm__clean | 50 | 132 | 14.3% | 3.6% | 10.8% | 40.0% | 828 | 868 | 0.0% | 0.0% | 174 | 256 | 1138 |
| piper-en_US-hfc_male-medium__180wpm__babble_10dB | 50 | 155 | 10.3% | 5.0% | 5.9% | 44.0% | 1678 | 2442 | 0.0% | 78.0% | 230 | 319 | 3106 |
| piper-en_US-hfc_male-medium__180wpm__babble_5dB | 50 | 155 | 12.6% | 8.6% | 9.1% | 54.0% | - | - | 0.0% | 100.0% | 186 | 761 | 3080 |
| piper-en_US-hfc_male-medium__180wpm__clean | 50 | 155 | 15.2% | 3.6% | 7.8% | 32.0% | 832 | 843 | 0.0% | 0.0% | 221 | 314 | 1252 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_10dB | 50 | 123 | 41.3% | 37.4% | 28.1% | 86.0% | 1354 | 2110 | 0.0% | 92.0% | 168 | 2336 | 2970 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_5dB | 50 | 123 | 66.4% | 86.4% | 42.0% | 94.0% | - | - | 0.0% | 100.0% | 192 | 6992 | 3864 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__clean | 50 | 123 | 34.7% | 26.5% | 22.9% | 74.0% | 851 | 954 | 0.0% | 0.0% | 153 | 403 | 1140 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 50 | 136 | 41.4% | 34.1% | 27.4% | 84.0% | 994 | 1641 | 0.0% | 88.0% | 202 | 4046 | 3024 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_5dB | 50 | 136 | 61.4% | 77.5% | 36.3% | 88.0% | - | - | 0.0% | 100.0% | 212 | 6673 | 3219 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__clean | 50 | 136 | 30.8% | 17.2% | 22.2% | 68.0% | 838 | 900 | 0.0% | 0.0% | 183 | 870 | 1185 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_10dB | 50 | 165 | 35.8% | 40.7% | 21.3% | 72.0% | 1236 | 1236 | 0.0% | 98.0% | 187 | 3778 | 2961 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_5dB | 50 | 165 | 60.8% | 72.2% | 38.1% | 92.0% | - | - | 0.0% | 100.0% | 218 | 6463 | 4092 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__clean | 50 | 165 | 27.2% | 17.5% | 22.8% | 60.0% | 838 | 896 | 0.0% | 0.0% | 169 | 263 | 1161 |
| piper-en_US-lessac-medium__130wpm__babble_10dB | 50 | 117 | 14.0% | 6.0% | 11.4% | 68.0% | 1438 | 2089 | 0.0% | 94.0% | 258 | 1200 | 3042 |
| piper-en_US-lessac-medium__130wpm__babble_5dB | 50 | 117 | 20.6% | 12.9% | 13.9% | 66.0% | - | - | 0.0% | 100.0% | 262 | 3816 | 3122 |
| piper-en_US-lessac-medium__130wpm__clean | 50 | 117 | 17.0% | 6.6% | 9.3% | 54.0% | 831 | 916 | 0.0% | 0.0% | 258 | 652 | 1375 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 50 | 133 | 13.0% | 6.6% | 8.8% | 56.0% | 1262 | 1374 | 0.0% | 96.0% | 244 | 6441 | 3123 |
| piper-en_US-lessac-medium__150wpm__babble_5dB | 50 | 133 | 19.2% | 17.9% | 16.5% | 82.0% | - | - | 0.0% | 100.0% | 239 | 1268 | 3120 |
| piper-en_US-lessac-medium__150wpm__clean | 50 | 133 | 16.7% | 6.6% | 14.9% | 66.0% | 828 | 867 | 0.0% | 0.0% | 173 | 384 | 1158 |
| piper-en_US-lessac-medium__180wpm__babble_10dB | 50 | 156 | 10.9% | 6.0% | 6.7% | 50.0% | 1718 | 2195 | 0.0% | 90.0% | 234 | 3977 | 3034 |
| piper-en_US-lessac-medium__180wpm__babble_5dB | 50 | 156 | 16.6% | 12.3% | 12.8% | 68.0% | - | - | 0.0% | 100.0% | 243 | 1617 | 3047 |
| piper-en_US-lessac-medium__180wpm__clean | 50 | 156 | 12.8% | 4.0% | 9.8% | 40.0% | 832 | 848 | 0.0% | 0.0% | 229 | 341 | 1288 |
