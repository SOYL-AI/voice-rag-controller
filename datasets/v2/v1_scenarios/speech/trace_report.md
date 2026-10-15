# Speech trace quality

Traces: `datasets/v2/v1_scenarios/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| babble_10dB | 900 | 139 | 22.5% | 15.1% | 16.0% | 67.1% | 1503 | 1971 | 0.0% | 89.1% | 89.1% | 2024 | 187 | 1449 | 2336 |
| babble_5dB | 900 | 139 | 35.7% | 33.0% | 24.6% | 78.2% | 1054 | 1054 | 0.0% | 99.9% | 99.9% | 2481 | 192 | 2557 | 2946 |
| clean | 900 | 139 | 20.7% | 9.7% | 14.9% | 54.0% | 831 | 912 | 0.0% | 0.0% | 0.0% | 831 | 177 | 310 | 1163 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 130wpm | 900 | 119 | 28.5% | 21.0% | 21.1% | 74.8% | 1198 | 1513 | 0.0% | 62.6% | 62.6% | 1834 | 186 | 1506 | 2264 |
| 150wpm | 900 | 135 | 26.3% | 18.5% | 18.2% | 66.0% | 1121 | 1339 | 0.0% | 63.8% | 63.8% | 1754 | 182 | 1482 | 2094 |
| 180wpm | 900 | 162 | 24.1% | 18.2% | 16.2% | 58.6% | 1177 | 1449 | 0.0% | 62.7% | 62.7% | 1749 | 188 | 1328 | 2087 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium | 450 | 137 | 26.5% | 17.2% | 20.0% | 66.4% | 1298 | 1461 | 0.0% | 65.1% | 65.1% | 1670 | 146 | 1199 | 2013 |
| piper-en_GB-northern_english_male-medium | 450 | 144 | 27.7% | 20.3% | 19.0% | 70.0% | 1179 | 1483 | 0.0% | 63.1% | 63.1% | 1852 | 161 | 1135 | 2168 |
| piper-en_US-arctic-medium-spk_ksp | 450 | 139 | 26.3% | 20.0% | 18.7% | 69.1% | 1171 | 1451 | 0.0% | 61.1% | 61.1% | 1704 | 175 | 1058 | 2113 |
| piper-en_US-hfc_male-medium | 450 | 134 | 14.8% | 5.9% | 10.5% | 52.9% | 1165 | 1517 | 0.0% | 60.0% | 60.0% | 1508 | 212 | 746 | 1807 |
| piper-en_US-l2arctic-medium-spk_SVBI | 450 | 141 | 45.7% | 43.2% | 30.0% | 78.9% | 1019 | 1290 | 0.0% | 64.2% | 64.2% | 2362 | 185 | 2840 | 2860 |
| piper-en_US-lessac-medium | 450 | 135 | 16.8% | 8.9% | 12.7% | 61.3% | 1152 | 1382 | 0.0% | 64.4% | 64.4% | 1578 | 232 | 1653 | 1928 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium__130wpm__babble_10dB | 50 | 117 | 22.4% | 12.6% | 19.6% | 82.0% | 1884 | 2192 | 0.0% | 92.0% | 92.0% | 1881 | 138 | 1380 | 2121 |
| piper-en_GB-cori-medium__130wpm__babble_5dB | 50 | 117 | 38.5% | 33.1% | 31.5% | 86.0% | - | - | 0.0% | 100.0% | 100.0% | 2390 | 155 | 3475 | 3220 |
| piper-en_GB-cori-medium__130wpm__clean | 50 | 117 | 21.9% | 8.6% | 17.6% | 60.0% | 828 | 959 | 0.0% | 0.0% | 0.0% | 828 | 136 | 181 | 1060 |
| piper-en_GB-cori-medium__150wpm__babble_10dB | 50 | 132 | 21.4% | 12.9% | 15.8% | 64.0% | 1830 | 1830 | 0.0% | 98.0% | 98.0% | 1889 | 147 | 267 | 2161 |
| piper-en_GB-cori-medium__150wpm__babble_5dB | 50 | 132 | 38.5% | 29.8% | 27.8% | 82.0% | - | - | 0.0% | 100.0% | 100.0% | 2320 | 156 | 1957 | 2658 |
| piper-en_GB-cori-medium__150wpm__clean | 50 | 132 | 22.1% | 9.3% | 17.7% | 54.0% | 828 | 859 | 0.0% | 0.0% | 0.0% | 828 | 146 | 183 | 1056 |
| piper-en_GB-cori-medium__180wpm__babble_10dB | 50 | 163 | 17.8% | 8.3% | 13.1% | 52.0% | 1587 | 2079 | 0.0% | 96.0% | 96.0% | 1965 | 147 | 496 | 2203 |
| piper-en_GB-cori-medium__180wpm__babble_5dB | 50 | 163 | 36.8% | 33.1% | 22.0% | 70.0% | - | - | 0.0% | 100.0% | 100.0% | 2101 | 150 | 2643 | 2577 |
| piper-en_GB-cori-medium__180wpm__clean | 50 | 163 | 18.9% | 7.3% | 14.9% | 48.0% | 828 | 850 | 0.0% | 0.0% | 0.0% | 828 | 142 | 205 | 1065 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_10dB | 50 | 123 | 32.4% | 25.2% | 22.6% | 88.0% | 1661 | 2043 | 0.0% | 94.0% | 94.0% | 2073 | 148 | 1668 | 2386 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_5dB | 50 | 123 | 38.2% | 32.8% | 30.0% | 88.0% | - | - | 0.0% | 100.0% | 100.0% | 3537 | 153 | 1498 | 3754 |
| piper-en_GB-northern_english_male-medium__130wpm__clean | 50 | 123 | 25.7% | 13.2% | 18.3% | 70.0% | 833 | 1030 | 0.0% | 0.0% | 0.0% | 833 | 149 | 190 | 1100 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_10dB | 50 | 142 | 23.4% | 17.5% | 15.3% | 70.0% | 1436 | 1751 | 0.0% | 88.0% | 88.0% | 2022 | 159 | 919 | 2288 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_5dB | 50 | 142 | 37.8% | 34.4% | 22.6% | 78.0% | - | - | 0.0% | 100.0% | 100.0% | 2237 | 176 | 3128 | 2877 |
| piper-en_GB-northern_english_male-medium__150wpm__clean | 50 | 142 | 20.5% | 12.9% | 16.1% | 58.0% | 834 | 976 | 0.0% | 0.0% | 0.0% | 834 | 150 | 192 | 1110 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_10dB | 50 | 168 | 22.8% | 13.9% | 13.5% | 60.0% | 1478 | 2139 | 0.0% | 86.0% | 86.0% | 1946 | 174 | 329 | 2176 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_5dB | 50 | 168 | 30.2% | 25.8% | 21.1% | 72.0% | - | - | 0.0% | 100.0% | 100.0% | 2352 | 177 | 2086 | 2702 |
| piper-en_GB-northern_english_male-medium__180wpm__clean | 50 | 168 | 18.2% | 7.3% | 11.4% | 46.0% | 833 | 959 | 0.0% | 0.0% | 0.0% | 833 | 165 | 206 | 1114 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_10dB | 50 | 118 | 21.7% | 14.2% | 17.7% | 88.0% | 1620 | 2044 | 0.0% | 82.0% | 82.0% | 2006 | 175 | 717 | 2250 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_5dB | 50 | 118 | 42.7% | 47.4% | 35.4% | 90.0% | - | - | 0.0% | 100.0% | 100.0% | 2409 | 187 | 3547 | 3794 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__clean | 50 | 118 | 20.0% | 8.6% | 16.2% | 60.0% | 825 | 928 | 0.0% | 0.0% | 0.0% | 825 | 176 | 248 | 1163 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 50 | 136 | 20.1% | 9.9% | 14.0% | 66.0% | 1617 | 2272 | 0.0% | 90.0% | 90.0% | 2043 | 165 | 325 | 2272 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_5dB | 50 | 136 | 40.1% | 33.1% | 21.7% | 76.0% | 1054 | 1054 | 0.0% | 98.0% | 98.0% | 2280 | 175 | 1785 | 2527 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__clean | 50 | 136 | 18.3% | 7.3% | 11.2% | 46.0% | 823 | 954 | 0.0% | 0.0% | 0.0% | 823 | 165 | 206 | 1143 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_10dB | 50 | 162 | 21.1% | 13.9% | 16.0% | 72.0% | 1440 | 1993 | 0.0% | 80.0% | 80.0% | 2009 | 179 | 578 | 2266 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_5dB | 50 | 162 | 30.9% | 34.8% | 21.6% | 76.0% | - | - | 0.0% | 100.0% | 100.0% | 2120 | 178 | 1851 | 2421 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__clean | 50 | 162 | 21.7% | 10.6% | 14.8% | 48.0% | 820 | 911 | 0.0% | 0.0% | 0.0% | 820 | 174 | 268 | 1179 |
| piper-en_US-hfc_male-medium__130wpm__babble_10dB | 50 | 114 | 14.0% | 5.0% | 10.4% | 58.0% | 1414 | 1987 | 0.0% | 72.0% | 72.0% | 1747 | 222 | 1136 | 2047 |
| piper-en_US-hfc_male-medium__130wpm__babble_5dB | 50 | 114 | 18.3% | 11.6% | 18.6% | 88.0% | - | - | 0.0% | 100.0% | 100.0% | 1821 | 223 | 1213 | 2150 |
| piper-en_US-hfc_male-medium__130wpm__clean | 50 | 114 | 16.0% | 4.0% | 9.2% | 48.0% | 831 | 900 | 0.0% | 0.0% | 0.0% | 831 | 224 | 308 | 1260 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 50 | 132 | 12.7% | 4.3% | 8.6% | 48.0% | 1406 | 2061 | 0.0% | 90.0% | 90.0% | 1833 | 220 | 929 | 2074 |
| piper-en_US-hfc_male-medium__150wpm__babble_5dB | 50 | 132 | 16.5% | 7.6% | 11.9% | 64.0% | - | - | 0.0% | 100.0% | 100.0% | 1880 | 219 | 1421 | 2114 |
| piper-en_US-hfc_male-medium__150wpm__clean | 50 | 132 | 14.3% | 3.6% | 10.8% | 40.0% | 828 | 868 | 0.0% | 0.0% | 0.0% | 828 | 174 | 256 | 1138 |
| piper-en_US-hfc_male-medium__180wpm__babble_10dB | 50 | 155 | 11.4% | 6.0% | 6.6% | 44.0% | 1678 | 2442 | 0.0% | 78.0% | 78.0% | 1900 | 224 | 294 | 2126 |
| piper-en_US-hfc_male-medium__180wpm__babble_5dB | 50 | 155 | 14.9% | 7.3% | 10.6% | 54.0% | - | - | 0.0% | 100.0% | 100.0% | 1901 | 181 | 844 | 2113 |
| piper-en_US-hfc_male-medium__180wpm__clean | 50 | 155 | 15.2% | 3.6% | 7.8% | 32.0% | 832 | 843 | 0.0% | 0.0% | 0.0% | 832 | 221 | 312 | 1246 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_10dB | 50 | 123 | 43.6% | 36.1% | 29.2% | 86.0% | 1354 | 2110 | 0.0% | 92.0% | 92.0% | 2766 | 168 | 2355 | 3238 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_5dB | 50 | 123 | 68.1% | 75.5% | 42.3% | 90.0% | - | - | 0.0% | 100.0% | 100.0% | 3529 | 187 | 5582 | 4160 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__clean | 50 | 123 | 34.7% | 26.5% | 22.9% | 74.0% | 851 | 954 | 0.0% | 0.0% | 0.0% | 851 | 153 | 403 | 1140 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 50 | 136 | 39.2% | 33.8% | 29.9% | 84.0% | 994 | 1641 | 0.0% | 88.0% | 88.0% | 2368 | 198 | 2493 | 3154 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_5dB | 50 | 136 | 65.1% | 66.9% | 38.3% | 88.0% | - | - | 0.0% | 100.0% | 100.0% | 3744 | 205 | 5339 | 4208 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__clean | 50 | 136 | 30.9% | 17.2% | 22.2% | 68.0% | 838 | 900 | 0.0% | 0.0% | 0.0% | 838 | 182 | 793 | 1185 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_10dB | 50 | 165 | 38.3% | 37.4% | 23.7% | 70.0% | 1236 | 1236 | 0.0% | 98.0% | 98.0% | 2342 | 182 | 3047 | 2825 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_5dB | 50 | 165 | 64.1% | 77.8% | 38.7% | 90.0% | - | - | 0.0% | 100.0% | 100.0% | 3979 | 219 | 5288 | 4672 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__clean | 50 | 165 | 27.3% | 17.5% | 22.8% | 60.0% | 838 | 896 | 0.0% | 0.0% | 0.0% | 838 | 169 | 263 | 1161 |
| piper-en_US-lessac-medium__130wpm__babble_10dB | 50 | 117 | 15.3% | 6.3% | 13.0% | 70.0% | 1438 | 2089 | 0.0% | 94.0% | 94.0% | 1878 | 250 | 827 | 2150 |
| piper-en_US-lessac-medium__130wpm__babble_5dB | 50 | 117 | 22.5% | 11.6% | 15.8% | 66.0% | - | - | 0.0% | 100.0% | 100.0% | 1970 | 248 | 1723 | 2392 |
| piper-en_US-lessac-medium__130wpm__clean | 50 | 117 | 17.0% | 6.6% | 9.3% | 54.0% | 831 | 916 | 0.0% | 0.0% | 0.0% | 831 | 256 | 652 | 1375 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 50 | 133 | 14.8% | 7.3% | 10.1% | 54.0% | 1262 | 1374 | 0.0% | 96.0% | 96.0% | 1915 | 230 | 4983 | 2170 |
| piper-en_US-lessac-medium__150wpm__babble_5dB | 50 | 133 | 21.2% | 19.2% | 18.6% | 82.0% | - | - | 0.0% | 100.0% | 100.0% | 2054 | 233 | 1120 | 2392 |
| piper-en_US-lessac-medium__150wpm__clean | 50 | 133 | 16.8% | 6.6% | 14.9% | 66.0% | 828 | 867 | 0.0% | 0.0% | 0.0% | 828 | 172 | 384 | 1158 |
| piper-en_US-lessac-medium__180wpm__babble_10dB | 50 | 156 | 12.4% | 6.6% | 8.2% | 52.0% | 1718 | 2195 | 0.0% | 90.0% | 90.0% | 1851 | 233 | 3336 | 2142 |
| piper-en_US-lessac-medium__180wpm__babble_5dB | 50 | 156 | 18.5% | 12.3% | 14.8% | 68.0% | - | - | 0.0% | 100.0% | 100.0% | 2041 | 237 | 1523 | 2294 |
| piper-en_US-lessac-medium__180wpm__clean | 50 | 156 | 12.8% | 4.0% | 9.8% | 40.0% | 832 | 848 | 0.0% | 0.0% | 0.0% | 832 | 228 | 327 | 1282 |

## Fallback-ended turns (VAD never fired; listed separately, structurally later)

| condition | n | fallback turns | premature | end-of-turn lag median ms | p90 ms |
|---|---|---|---|---|---|
| piper-en_GB-cori-medium__130wpm__babble_10dB | 50 | 46 | 0 | 1881 | 3091 |
| piper-en_GB-cori-medium__130wpm__babble_5dB | 50 | 50 | 0 | 2390 | 4577 |
| piper-en_GB-cori-medium__150wpm__babble_10dB | 50 | 49 | 0 | 1890 | 3166 |
| piper-en_GB-cori-medium__150wpm__babble_5dB | 50 | 50 | 0 | 2320 | 4489 |
| piper-en_GB-cori-medium__180wpm__babble_10dB | 50 | 48 | 0 | 1965 | 3179 |
| piper-en_GB-cori-medium__180wpm__babble_5dB | 50 | 50 | 0 | 2101 | 4470 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_10dB | 50 | 47 | 0 | 2076 | 4452 |
| piper-en_GB-northern_english_male-medium__130wpm__babble_5dB | 50 | 50 | 0 | 3537 | 4501 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_10dB | 50 | 44 | 0 | 2063 | 3915 |
| piper-en_GB-northern_english_male-medium__150wpm__babble_5dB | 50 | 50 | 0 | 2237 | 4306 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_10dB | 50 | 43 | 0 | 1956 | 3602 |
| piper-en_GB-northern_english_male-medium__180wpm__babble_5dB | 50 | 50 | 0 | 2352 | 4222 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_10dB | 50 | 41 | 0 | 2056 | 3874 |
| piper-en_US-arctic-medium-spk_ksp__130wpm__babble_5dB | 50 | 50 | 1 | 2409 | 4535 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 50 | 45 | 0 | 2054 | 3031 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_5dB | 50 | 49 | 0 | 2358 | 4307 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_10dB | 50 | 40 | 0 | 2052 | 4320 |
| piper-en_US-arctic-medium-spk_ksp__180wpm__babble_5dB | 50 | 50 | 1 | 2120 | 4464 |
| piper-en_US-hfc_male-medium__130wpm__babble_10dB | 50 | 36 | 0 | 1845 | 2201 |
| piper-en_US-hfc_male-medium__130wpm__babble_5dB | 50 | 50 | 0 | 1821 | 3634 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 50 | 45 | 0 | 1850 | 2178 |
| piper-en_US-hfc_male-medium__150wpm__babble_5dB | 50 | 50 | 0 | 1880 | 3115 |
| piper-en_US-hfc_male-medium__180wpm__babble_10dB | 50 | 39 | 0 | 1922 | 2338 |
| piper-en_US-hfc_male-medium__180wpm__babble_5dB | 50 | 50 | 0 | 1901 | 2987 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_10dB | 50 | 46 | 0 | 2873 | 4488 |
| piper-en_US-l2arctic-medium-spk_SVBI__130wpm__babble_5dB | 50 | 50 | 4 | 3529 | 4619 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 50 | 44 | 0 | 2929 | 4327 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_5dB | 50 | 50 | 1 | 3744 | 4476 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_10dB | 50 | 49 | 0 | 2414 | 4335 |
| piper-en_US-l2arctic-medium-spk_SVBI__180wpm__babble_5dB | 50 | 50 | 0 | 3979 | 4488 |
| piper-en_US-lessac-medium__130wpm__babble_10dB | 50 | 47 | 0 | 1882 | 3238 |
| piper-en_US-lessac-medium__130wpm__babble_5dB | 50 | 50 | 0 | 1970 | 3840 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 50 | 48 | 0 | 1922 | 2268 |
| piper-en_US-lessac-medium__150wpm__babble_5dB | 50 | 50 | 0 | 2054 | 3664 |
| piper-en_US-lessac-medium__180wpm__babble_10dB | 50 | 45 | 0 | 1872 | 2565 |
| piper-en_US-lessac-medium__180wpm__babble_5dB | 50 | 50 | 0 | 2041 | 3604 |
