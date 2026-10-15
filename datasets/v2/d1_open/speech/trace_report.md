# Speech trace quality

Traces: `datasets/v2/d1_open/speech/traces`; text normaliser: whisper.EnglishTextNormalizer. Metric definitions: audio/v2/trace_report.py.

## By noise

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 5300 | 128 | 23.7% | 14.2% | 18.3% | 66.8% | 830 | 886 | 0.6% | 0.0% | 0.0% | 830 | 154 | 211 | 1094 |

## By rate

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 150wpm | 5300 | 128 | 23.7% | 14.2% | 18.3% | 66.8% | 830 | 886 | 0.6% | 0.0% | 0.0% | 830 | 154 | 211 | 1094 |

## By voice

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_US-hfc_male-medium | 5300 | 128 | 23.7% | 14.2% | 18.3% | 66.8% | 830 | 886 | 0.6% | 0.0% | 0.0% | 830 | 154 | 211 | 1094 |

## All conditions

| condition | n | wpm | partial WER | final WER | rev (lenient) | items revised | EOT median ms | EOT p90 | premature | missed | fallback | user_end median | ASR ms med | avail lag p90 | final avail after end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| piper-en_US-hfc_male-medium__150wpm__clean | 5300 | 128 | 23.7% | 14.2% | 18.3% | 66.8% | 830 | 886 | 0.6% | 0.0% | 0.0% | 830 | 154 | 211 | 1094 |

## Fallback-ended turns (VAD never fired; listed separately, structurally later)

(no fallback turns)
