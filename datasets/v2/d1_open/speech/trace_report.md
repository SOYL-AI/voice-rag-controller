# Speech trace quality

Traces: `datasets/v2/d1_open/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| babble_10dB | 750 | 129 | 32.6% | 24.5% | 28.2% | 83.1% | 1399 | 2017 | 0.0% | 92.8% | 92.8% | 2419 | 175 | 1897 | 2787 |
| clean | 6200 | 128 | 24.6% | 15.0% | 19.5% | 68.3% | 830 | 893 | 0.7% | 0.0% | 0.0% | 830 | 156 | 233 | 1101 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 130wpm | 150 | 108 | 27.6% | 15.4% | 23.4% | 77.3% | 831 | 942 | 2.7% | 0.0% | 0.0% | 831 | 167 | 255 | 1128 |
| 150wpm | 6650 | 128 | 25.5% | 16.1% | 20.3% | 69.7% | 883 | 996 | 0.6% | 10.5% | 10.5% | 1010 | 157 | 421 | 1290 |
| 180wpm | 150 | 148 | 22.8% | 13.2% | 20.7% | 69.3% | 831 | 893 | 0.0% | 0.0% | 0.0% | 831 | 164 | 231 | 1120 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium | 300 | 125 | 31.1% | 18.9% | 29.0% | 85.3% | 1138 | 1546 | 0.7% | 47.3% | 47.3% | 1473 | 152 | 821 | 1768 |
| piper-en_US-arctic-medium-spk_ksp | 300 | 131 | 32.1% | 23.5% | 27.8% | 76.7% | 1184 | 1474 | 0.0% | 44.7% | 44.7% | 1472 | 158 | 1152 | 1793 |
| piper-en_US-hfc_male-medium | 5750 | 128 | 23.8% | 14.2% | 18.6% | 67.5% | 843 | 914 | 0.6% | 2.2% | 2.2% | 859 | 156 | 230 | 1127 |
| piper-en_US-l2arctic-medium-spk_SVBI | 300 | 135 | 45.1% | 41.8% | 37.0% | 89.3% | 1068 | 1504 | 0.3% | 47.3% | 47.3% | 2360 | 176 | 1763 | 2757 |
| piper-en_US-lessac-medium | 300 | 130 | 25.8% | 15.1% | 22.8% | 72.7% | 830 | 910 | 1.0% | 50.0% | 50.0% | 1433 | 188 | 1424 | 1814 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_GB-cori-medium__150wpm__babble_10dB | 150 | 125 | 32.3% | 20.0% | 29.0% | 86.0% | 1447 | 2156 | 0.0% | 94.7% | 94.7% | 2116 | 150 | 1071 | 2408 |
| piper-en_GB-cori-medium__150wpm__clean | 150 | 125 | 30.0% | 17.7% | 28.9% | 84.7% | 830 | 936 | 1.3% | 0.0% | 0.0% | 830 | 155 | 572 | 1128 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 150 | 131 | 33.4% | 24.7% | 28.9% | 82.0% | 1545 | 2000 | 0.0% | 89.3% | 89.3% | 2122 | 166 | 2074 | 2488 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__clean | 150 | 131 | 30.8% | 22.3% | 26.7% | 71.3% | 823 | 946 | 0.0% | 0.0% | 0.0% | 823 | 149 | 231 | 1098 |
| piper-en_US-hfc_male-medium__130wpm__clean | 150 | 108 | 27.6% | 15.4% | 23.4% | 77.3% | 831 | 942 | 2.7% | 0.0% | 0.0% | 831 | 167 | 255 | 1128 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 150 | 124 | 23.1% | 14.6% | 21.5% | 83.3% | 1320 | 1873 | 0.0% | 85.3% | 85.3% | 1951 | 193 | 871 | 2290 |
| piper-en_US-hfc_male-medium__150wpm__clean | 5300 | 128 | 23.7% | 14.2% | 18.3% | 66.8% | 830 | 886 | 0.6% | 0.0% | 0.0% | 830 | 154 | 211 | 1094 |
| piper-en_US-hfc_male-medium__180wpm__clean | 150 | 148 | 22.8% | 13.2% | 20.7% | 69.3% | 831 | 893 | 0.0% | 0.0% | 0.0% | 831 | 164 | 231 | 1120 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 150 | 135 | 48.1% | 47.8% | 40.0% | 92.7% | 1284 | 2040 | 0.0% | 94.7% | 94.7% | 3869 | 180 | 2927 | 4322 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__clean | 150 | 135 | 42.1% | 35.7% | 34.1% | 86.0% | 852 | 968 | 0.7% | 0.0% | 0.0% | 852 | 171 | 599 | 1191 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 150 | 130 | 26.4% | 15.3% | 21.7% | 71.3% | - | - | 0.0% | 100.0% | 100.0% | 2036 | 184 | 2544 | 2426 |
| piper-en_US-lessac-medium__150wpm__clean | 150 | 130 | 25.2% | 14.9% | 23.9% | 74.0% | 830 | 910 | 2.0% | 0.0% | 0.0% | 830 | 192 | 303 | 1202 |

## Fallback-ended turns (VAD never fired; listed separately, structurally later)

| condition | n | fallback turns | premature | end-of-turn lag median ms | p90 ms |
|---|---|---|---|---|---|
| piper-en_GB-cori-medium__150wpm__babble_10dB | 150 | 142 | 0 | 2162 | 4511 |
| piper-en_US-arctic-medium-spk_ksp__150wpm__babble_10dB | 150 | 134 | 0 | 2243 | 4461 |
| piper-en_US-hfc_male-medium__150wpm__babble_10dB | 150 | 128 | 3 | 1998 | 4396 |
| piper-en_US-l2arctic-medium-spk_SVBI__150wpm__babble_10dB | 150 | 142 | 6 | 4001 | 4511 |
| piper-en_US-lessac-medium__150wpm__babble_10dB | 150 | 150 | 4 | 2036 | 4293 |
