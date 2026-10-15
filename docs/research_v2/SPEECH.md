# v2 speech stimuli: rendered speech, streaming-ASR traces, trace QA

v1 timing experiments "spoke" text at a fixed 150 wpm, so every partial
transcript was a perfect word prefix. v2 replaces that with speech rendered
across voices, accents, rates and noise, replayed through the project's own
streaming STT (faster-whisper + Silero VAD, `audio/stt.py`) on a simulated
clock. The traces keep every recognition error, revision and real end-of-turn
decision, plus the measured ASR compute time, and drive the controllers and the
trace-driven latency simulation (schema: CONTRACTS.md §4).

This is stimulus preparation, not outcome measurement: it may touch all splits,
and it runs no LLM.

Code: `audio/v2/` (`render.py`, `trace.py`, `trace_report.py`, `voices.py`,
`tts_engines.py`, `tts_worker.py`, `align.py`, `asr.py`, `textnorm.py`,
`accent_screen.py`, `record_prompts.py`). Tests: `tests/test_speech_v2.py`.

__RESULTS__

## 1. Setup

Three interpreters, so engines with conflicting pins never share one:

| venv | Python | what | install |
|---|---|---|---|
| `.venv-e2` (existing, repo root) | 3.10 | render driver, alignment, ASR replay, reports | already has faster-whisper, torch (CPU), transformers, librosa, sounddevice |
| `.venv-speech` (worktree, gitignored) | 3.10 | piper TTS worker; CUDA DLLs for faster-whisper; accent screen | `pip install piper-tts nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" requests "huggingface_hub<0.30" soundfile speechbrain==1.0.3 torch==2.5.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu` |
| `.venv-parler` (worktree, gitignored, optional) | 3.10 | Parler-TTS worker (Indic Parler-TTS for en-IN) | `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` then `pip install git+https://github.com/huggingface/parler-tts.git` |

* `.venv-e2`'s torch is CPU-only and CTranslate2 needs cuBLAS 12 + cuDNN 9 on
  Windows. `audio/v2/asr.py` adds the `nvidia/*/bin` folders from
  `.venv-speech` to the DLL search path (override: `VOICE_RAG_CUDA_DLL_DIRS`).
  Nothing is installed into `.venv-e2`.
* Worker interpreters can be overridden with `VOICE_RAG_PIPER_PYTHON` /
  `VOICE_RAG_PARLER_PYTHON`.
* Models download on first use: piper voices to `audio/v2/models/piper/`
  (gitignored, ~60 MB each), `facebook/wav2vec2-base-960h` and
  `Systran/faster-whisper-small` to the HF cache, Silero VAD via torch.hub.
* Always set `PYTHONUTF8=1`.

## 2. Run book

```bash
PY=../../.venv-e2/Scripts/python.exe        # or the absolute path
# 1. render (clean + babble mixes), resumable; split items over processes with --shard k/n
$PY -m audio.v2.render --items datasets/v2/d2_support/items.jsonl [--voices ...] [--rates 130 150 180] \
    [--noises clean babble_10dB babble_5dB] [--shard 0/2] [--align-check-frac 0.1]
# 2. ASR replay -> datasets/v2/<name>/speech/traces/<voice>__<wpm>wpm__<noise>.jsonl
$PY -m audio.v2.trace --items datasets/v2/d2_support/items.jsonl [--conditions <cond> ...] [--shard 0/2]
# 3. quality report -> datasets/v2/<name>/speech/trace_report.{md,json}
$PY -m audio.v2.trace_report --name d2_support
```

* Any items file works: `datasets/v2/<name>/items.jsonl` (CONTRACTS §2) or the
  v1 `simulator/scenarios.json` (item ids `v1_scenarios-%04d`).
* Both stages are resumable (a finished render has its sidecar json; a traced
  item is a line in the trace file) and parallel-safe (atomic file replace;
  trace appends hold `<trace>.lock`). `--shard k/n` splits items across
  processes. For trace timing, run **one trace process per GPU**: several
  processes on one GPU inflate `compute_ms` and hence `available_ms`.
* `--split dev|test|all` filters items (default all: speech prep is allowed on
  the sealed test split because it measures nothing about the models).

## 3. Voices

voice_id = `<engine>-<model>[-spk_<speaker>]`; a condition adds the rate and
noise: `<voice_id>__<wpm>wpm__<noise>` (`__natural__` for human recordings).
Registry and licences: `audio/v2/voices.py`.

Default rendering set (6 voices, 2 per accent; all local piper, CPU):

| voice_id | accent | sex | natural wpm | length_scale 130/150/180 | accent screen top-1 = label | training-data licence |
|---|---|---|---|---|---|---|
| piper-en_US-lessac-medium | en-US | F | 247 | 2.20 / 1.88 / 1.56 | 12/20 us | Blizzard 2013 Lessac (research) |
| piper-en_US-hfc_male-medium | en-US | M | 270 | 2.15 / 1.81 / 1.48 | 14/20 us | CC BY-NC-SA 4.0 |
| piper-en_GB-cori-medium | en-GB | F | 222 | 1.97 / 1.65 / 1.30 | 17/20 england | public domain (LibriVox) |
| piper-en_GB-northern_english_male-medium | en-GB | M | 262 | 2.42 / 2.05 / 1.67 | 20/20 england | CC BY-SA 4.0 |
| piper-en_US-l2arctic-medium-spk_SVBI | en-IN | F | 242 | 2.16 / 1.87 / 1.47 | 14/20 indian | CC BY-NC 4.0 (L2-ARCTIC) |
| piper-en_US-arctic-medium-spk_ksp | en-IN | M | 217 | 1.87 / 1.56 / 1.29 | 15/20 indian | CMU ARCTIC (permissive) |

Calibrated rates hit the target within +-5 wpm on the calibration sentences
(`audio/v2/voice_calibration.json`). **Primary voice** for single-condition
runs: `piper-en_US-hfc_male-medium` (lowest final WER of the two en-US voices
on the v1 scenarios: 3.6 % vs 6.6 % at 150 wpm clean).

### en-IN

* **Accent screen** (`audio/v2/accent_screen.py`, results in
  `docs/research_v2/speech_accent_screen.json`): every candidate voice reads the
  20 calibration sentences and an independent accent-ID model
  (`Jzuluaga/accent-id-commonaccent_ecapa`, SpeechBrain ECAPA, CommonAccent, 16
  English accents, MIT) classifies each render. Controls behave: US voices
  12-16/20 "us", British voices 17-20/20 "england". Indian candidates:
  CMU ARCTIC Indian speakers ksp, gka, slp 15/20 "indian", aup 14/20;
  L2-ARCTIC Hindi-L1 speakers SVBI 14/20, RRBI 10/20, TNI 7/20, ASI 3/20; the
  VCTK Indian speakers in piper's VCTK model (p248, p251, p376) 0-2/20 - the
  multi-speaker model has averaged their accent away. We use ksp (M) and SVBI
  (F), from two different models. The classifier is a proxy, not a listening
  test; a 10-minute listening check by an Indian-English speaker is still
  advisable before the paper says "Indian-accented".
* **AI4Bharat Indic Parler-TTS** (`ai4bharat/indic-parler-tts`): Apache-2.0,
  ~0.9 B parameters, supports English with named Indian speakers; fits an 8 GB
  GPU in fp16. **Not evaluated here**: the Hugging Face repo is gated
  (licence click-through; anonymous download returns 401) and no HF token is
  configured on this host. The adapter is written (`ParlerEngine` +
  `tts_worker.py --engine parler`, voices `parler-indic-spk_Thoma`,
  `parler-indic-spk_Mary`, status `needs_auth`). A second blocker on this host:
  installing parler-tts into a venv inside this deep worktree path fails on
  Windows' 260-character path limit (a transformers file path is too long), so
  create the venv at a short path, e.g. `C:\venvs\parler`, and set
  `VOICE_RAG_PARLER_PYTHON`. To enable: accept the licence on the model page,
  `huggingface-cli login`, set the voice status to `active`, then run the
  accent screen on it (extend `accent_screen.py` to the parler engine) and
  compare with ksp/SVBI.
* **Svarah** (real Indian-accented English, see 8b) is the right source for
  human en-IN speech.

### Cloud adapter

`tts_engines.AzureSpeechEngine` implements the same interface for Azure Speech
neural voices (e.g. `en-IN-NeerjaNeural`, `en-IN-PrabhatNeural`; rate via SSML
`<prosody rate>`), but refuses to construct unless `VOICE_RAG_ALLOW_PAID_TTS=1`
and `AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION` are set. It was never called. To
use it: add a registry entry with `engine: "azure"`, `status: "active"`.

## 4. Speaking rate

Rate = words / (first-word start to last-word end), from the forced alignment;
pauses inside the utterance count, leading/trailing silence does not.
Piper's rate knob is `length_scale` (duration multiplier). Each voice is
calibrated once on 20 fixed spoken-question sentences
(`voices.CALIBRATION_SENTENCES`): natural wpm at length_scale 1, then
`length_scale = natural / target`, refined once from a measured render.
Results are committed in `audio/v2/voice_calibration.json`. Per-item wpm
varies around the target (short questions vary most), as real speech does;
every trace line records `wpm_actual`.

Note: by this articulation-rate measure piper voices speak short questions at
~200-250 wpm naturally, so the 130/150/180 targets are *slower* than the voices'
defaults (length_scale ~1.4-2.3). This matches the v1 150 wpm assumption but
makes 130 wpm sound deliberate. If the paper needs a "natural" condition, add a
rate equal to each voice's natural wpm.

Engines without a duration knob (Parler) are rendered at a "moderate pace"
description and time-stretched (librosa phase vocoder) to the target.

## 5. Noise

`clean`, `babble_10dB`, `babble_5dB`. Babble is synthesised from the renders
themselves (no external corpus): 6 talker streams, each a chain of other items'
clean renders at the same rate, voices drawn from the six default voices,
random phase and 50-400 ms gaps; summed and scaled so that
`10 log10(P_speech / P_babble) = SNR`, where P_speech is the target's mean power
from first-word start to last-word end and P_babble the babble's mean power over
the file. Babble covers the lead-in and the trailing silence too, as background
talk would, which is what makes end-of-turn detection hard. Talker choice is
seeded from (item, voice, rate, noise) and recorded in the sidecar json; the
trace keeps `noise_meta` (type, SNR, talkers, seed).

## 6. Audio format and word timestamps

16 kHz mono int16 wav: 500 ms digital silence, the utterance (TTS edge silence
trimmed), 2500 ms silence; speech-active RMS levelled to -26 dBFS.

Word times come from **CTC forced alignment** of the known text
(`facebook/wav2vec2-base-960h`, 20 ms frames; digits spelled out), refined by
extending each word boundary through contiguous high-energy 10 ms frames (CTC
posteriors are peaky and cut word ends short). Accuracy check
(`--align-check-frac`, sidecar `align_check`): Whisper word timestamps on the
same audio for the words Whisper recognises correctly, and first/last word vs an
energy-based speech-activity bound.

__ALIGN__

## 7. Streaming ASR replay (trace.py)

The replay is `StreamingSTT.transcribe_stream` statement for statement with the
wall clock replaced by audio time:

* 512-sample (32 ms) blocks as `generate_mic_stream` delivers them; RMS gate
  0.003; Silero VAD (`StreamingSTT.process_vad_frame`, stateful, reset per turn)
  with threshold **0.3** - the value `audio_loop.py` passes for live turns
  (`--vad-threshold`).
* Speech audio is collected from the first speech block. Every
  `partial_interval_ms` (config: **400 ms**) of audio, the last 8 s of speech is
  transcribed (`beam_size=1`, no word timestamps, `language=en`); empty
  results are not emitted, as live.
* End of turn: `silence_hang_ms` (config: **800 ms**) of non-speech blocks.
  The turn ends at the first endpoint (as live), then the final transcript is
  `beam_size=5` with word timestamps over all speech audio.
* Whisper model: the configured `small`, but on the GPU (`float16`) instead of
  the config's CPU/int8; recorded per line in `asr_model` / `asr_settings`.
* `mic_gain` (2.0 in config, for a quiet laptop mic) is not applied: renders
  are already at a nominal level.

Clock: all times are ms from the forced-aligned start of the first word.
`compute_ms` is the measured ASR wall time; `available_ms = max(t_ms,
previous available_ms) + compute_ms` (one serial ASR worker). A consumer may use
a partial only from `available_ms`.

Deviation from live: live partials are scheduled on the wall clock while the
ASR call blocks the audio loop, so when ASR takes longer than the interval live
partials thin out; the replay schedules on audio time and carries the delay in
`available_ms` instead.

__TRACEQ__

## 8. Human speech subset

### 8a. Recording our own (record_prompts.py)

`python -m audio.v2.record_prompts record --items <items> --speaker s03 --accent en-IN --l1 Hindi --n 200`
shows one question at a time and records 16 kHz mono from the default mic;
`prepare` then trims, levels, pads, force-aligns and Whisper-checks each take
(flags likely misreads) into the render layout, after which `trace.py
--conditions human-s03__natural__clean` replays it like any voice.

Protocol:
1. **Consent**: the tool prints what is recorded and why and requires "yes";
   speakers are identified by a code only; recordings never go into git; a
   speaker may withdraw and have files deleted.
2. **Room**: quiet room, door closed, fans/AC off if possible; one mic setup
   per session, ~20-30 cm away; check the peak readout (not clipped, not
   < 0.05).
3. **Reading**: natural, as if asking a support agent; no over-articulation;
   small hesitations are fine; redo ("r") only for wrong or skipped words.
4. **Timing**: Enter, ~0.5 s pause, speak, ~1 s pause, Enter (the trailing
   silence is needed for end-of-turn detection).
5. ~200 questions per speaker in sessions of <= 30 min (the tool resumes);
   items are shuffled with `--seed` so speakers cover different items; record
   the same seed for two speakers to get paired items.

### 8b. Public human-spoken question sets

Checked through the Hugging Face API and the datasets-server row preview; nothing
was downloaded.

| set | what | size | licence / access | usable for our items? |
|---|---|---|---|---|
| **HeySQuAD human** (`yijingwu/HeySQuAD_human`, Wu et al. 2023) | SQuAD (incl. 2.0 unanswerable) questions read aloud by crowd workers; fields: audio, question, ASR `transcription`, context, answers | 76,148 questions (train 72k, validation 4,158); 14.6 GB parquet, validation 0.71 GB | CC BY 4.0, ungated | Not for d1_open/d2_support items (different questions), but usable as a **third, human-spoken QA set**: validation questions with their SQuAD contexts as the corpus. Caveats from the preview: the bundled `transcription` is an ASR output with many errors (it is not the reference - the `question` field is); at least one previewed row's audio did not match its question, so audio/question agreement must be checked (run our replay and drop items with very high final WER). Accents are not documented (mostly US crowd workers). |
| **Svarah** (`ai4bharat/Svarah`, Javed et al. 2023) | Indian-accented English: 9.6 h, 117 speakers, 65 districts, 19 states, 19 L1s; read speech and spontaneous task-oriented speech (ordering groceries, payments, pensions, passports) | ~1.1 GB | CC BY 4.0, but gated (contact-sharing agreement + HF token) | Not our questions, so it cannot give paired human readings of our items. Useful (a) to validate en-IN ASR behaviour (WER, revisions, end-of-turn) of the replay on real Indian speech vs our TTS en-IN voices, and (b) its task-oriented utterances are close to d2_support's service domain. Needs the gate accepted first. |

Recommendation: record our own ~200 questions per speaker with
`record_prompts.py` (2-4 speakers, at least 2 en-IN) on d2_support dev items -
that is the only way to get human readings *of our items*; use Svarah to check
that the TTS en-IN voices are not much easier for the ASR than real Indian
speech; treat HeySQuAD as optional external validity.

## 9. Throughput and the full run

__THROUGHPUT__

## 10. Open issues

__ISSUES__
