# Speech trace quality

Traces: `datasets/v2/d2_support/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| babble_10dB | 750 | 146 | 18.7% | 10.4% | 28.1% | 92.0% | 1375 | 2202 | 0.1% | 89.6% | 89.6% | 2526 | 220 | 1361 | 3083 |
| clean | 1620 | 144 | 13.1% | 8.2% | 16.4% | 79.6% | 838 | 963 | 6.3% | 0.0% | 0.0% | 838 | 241 | 1121 | 1477 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 130wpm | 150 | 122 | 12.7% | 10.8% | 15.6% | 82.0% | 832 | 984 | 14.0% | 0.0% | 0.0% | 832 | 236 | 354 | 1431 |
| 150wpm | 2070 | 145 | 15.4% | 9.3% | 21.0% | 84.7% | 1034 | 1414 | 4.0% | 32.5% | 32.5% | 1451 | 236 | 1321 | 2075 |
| 180wpm | 150 | 168 | 9.5% | 2.2% | 11.5% | 68.7% | 832 | 916 | 0.0% | 0.0% | 0.0% | 832 | 206 | 322 | 1298 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium | 300 | 148 | 15.6% | 9.0% | 23.9% | 89.7% | 1115 | 1572 | 4.3% | 47.7% | 47.7% | 1516 | 222 | 3722 | 2272 |
| piper-en_US-arctic-medium-spk_ksp | 300 | 149 | 14.9% | 6.6% | 22.9% | 90.0% | 1118 | 1638 | 0.0% | 42.0% | 42.0% | 1486 | 193 | 426 | 1928 |
| piper-en_US-hfc_male-medium | 1170 | 143 | 10.8% | 5.5% | 13.3% | 76.8% | 911 | 1115 | 5.0% | 10.7% | 10.7% | 979 | 254 | 610 | 1647 |
| piper-en_US-l2arctic-medium-spk_SVBI | 300 | 145 | 32.3% | 25.8% | 40.8% | 95.7% | 1082 | 1428 | 5.0% | 46.3% | 46.3% | 2556 | 217 | 1954 | 3062 |
| piper-en_US-lessac-medium | 300 | 144 | 12.4% | 7.5% | 19.4% | 85.3% | 1099 | 1719 | 5.7% | 46.3% | 46.3% | 1465 | 226 | 974 | 2001 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium__150wpm__babble_10dB | 150 | 148 | 17.0% | 8.2% | 27.5% | 92.0% | 1393 | 2161 | 0.0% | 95.3% | 95.3% | 2195 | 219 | 1382 | 3022 |
| piper-en_GB-cori-medium__150wpm__clean | 150 | 148 | 14.2% | 9.9% | 20.3% | 87.3% | 837 | 982 | 8.7% | 0.0% | 0.0% | 837 | 226 | 6062 | 1522 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 150 | 149 | 16.4% | 7.6% | 27.2% | 95.3% | 1391 | 2255 | 0.0% | 84.0% | 84.0% | 2128 | 185 | 534 | 2534 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__clean | 150 | 149 | 13.4% | 5.6% | 18.7% | 84.7% | 844 | 1021 | 0.0% | 0.0% | 0.0% | 844 | 202 | 318 | 1322 |
| piper-en_US-hfc_male-medium__130wpm__clean | 150 | 122 | 12.7% | 10.8% | 15.6% | 82.0% | 832 | 984 | 14.0% | 0.0% | 0.0% | 832 | 236 | 354 | 1431 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 150 | 141 | 11.6% | 3.8% | 18.0% | 83.3% | 1438 | 2249 | 0.7% | 83.3% | 83.3% | 1970 | 222 | 434 | 2432 |
| piper-en_US-hfc_male-medium__150wpm__clean | 720 | 142 | 10.5% | 5.5% | 12.2% | 76.0% | 834 | 948 | 5.0% | 0.0% | 0.0% | 834 | 275 | 760 | 1600 |
| piper-en_US-hfc_male-medium__180wpm__clean | 150 | 168 | 9.5% | 2.2% | 11.5% | 68.7% | 832 | 916 | 0.0% | 0.0% | 0.0% | 832 | 206 | 322 | 1298 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 150 | 145 | 35.4% | 26.9% | 45.7% | 98.7% | 1292 | 1854 | 0.0% | 92.7% | 92.7% | 4241 | 240 | 3529 | 4778 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__clean | 150 | 145 | 29.2% | 24.6% | 35.8% | 92.7% | 871 | 1002 | 10.0% | 0.0% | 0.0% | 871 | 194 | 379 | 1344 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 150 | 144 | 12.9% | 5.4% | 21.9% | 90.7% | 1362 | 2489 | 0.0% | 92.7% | 92.7% | 2094 | 234 | 925 | 2652 |
| piper-en_US-lessac-medium__150wpm__clean | 150 | 144 | 12.0% | 9.7% | 16.9% | 80.0% | 836 | 949 | 11.3% | 0.0% | 0.0% | 836 | 218 | 1023 | 1350 |

## Fallback-ended turns (VAD never fired; listed separately, structurally later)

| condition | n | fallback turns | premature | end-of-turn lag median ms | p90 ms |
|---|---|---|---|---|---|
| piper-en_GB-cori-medium__150wpm__babble_10dB | 150 | 143 | 8 | 2282 | 4655 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 150 | 126 | 0 | 2247 | 4633 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 150 | 125 | 2 | 2064 | 4690 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 150 | 139 | 8 | 4298 | 4576 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 150 | 139 | 5 | 2166 | 4525 |
