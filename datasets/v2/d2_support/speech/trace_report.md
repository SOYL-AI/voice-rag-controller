# Speech trace quality

Traces: `datasets/v2/d2_support/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 222 | 141 | 10.7% | 5.2% | 11.7% | 73.4% | 834 | 949 | 5.0% | 0.0% | 309 | 1142 | 1730 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150wpm | 222 | 141 | 10.7% | 5.2% | 11.7% | 73.4% | 834 | 949 | 5.0% | 0.0% | 309 | 1142 | 1730 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_US-hfc_male-medium | 222 | 141 | 10.7% | 5.2% | 11.7% | 73.4% | 834 | 949 | 5.0% | 0.0% | 309 | 1142 | 1730 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_US-hfc_male-medium__150wpm__clean | 222 | 141 | 10.7% | 5.2% | 11.7% | 73.4% | 834 | 949 | 5.0% | 0.0% | 309 | 1142 | 1730 |
