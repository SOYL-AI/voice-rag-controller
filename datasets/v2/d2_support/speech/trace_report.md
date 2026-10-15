# Speech trace quality

Traces: `datasets/v2/d2_support/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 720 | 142 | 10.5% | 5.5% | 12.2% | 76.1% | 834 | 948 | 5.0% | 0.0% | 0.0% | 834 | 275 | 760 | 1600 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150wpm | 720 | 142 | 10.5% | 5.5% | 12.2% | 76.1% | 834 | 948 | 5.0% | 0.0% | 0.0% | 834 | 275 | 760 | 1600 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_US-hfc_male-medium | 720 | 142 | 10.5% | 5.5% | 12.2% | 76.1% | 834 | 948 | 5.0% | 0.0% | 0.0% | 834 | 275 | 760 | 1600 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_US-hfc_male-medium__150wpm__clean | 720 | 142 | 10.5% | 5.5% | 12.2% | 76.1% | 834 | 948 | 5.0% | 0.0% | 0.0% | 834 | 275 | 760 | 1600 |
