"""
align.py - per-word timestamps for a rendered utterance whose text is known.

Method
------
CTC forced alignment of the known reference text against a character-level
English CTC acoustic model (facebook/wav2vec2-base-960h, Apache-2.0, 20 ms
frames). Each reference word becomes a letters-only token sequence
(textnorm.align_chars, digits spelled out); a Viterbi pass through the CTC
trellis assigns every character a frame span; a word spans its first to its
last character frame, then boundaries are extended through contiguous
high-energy 10 ms frames (refine_with_energy) because CTC posteriors are
peaky and cut word ends short. Because the alignment is *forced*, every reference word
gets a time even if a recogniser would mis-hear it.

Accuracy check (cross_check)
----------------------------
The same clean audio is decoded by faster-whisper with word_timestamps=True.
For words where Whisper's normalised word equals the reference word (matched
in order), we report the signed start and end differences (whisper - forced). Whisper's DTW timestamps are themselves only ~tens of ms
accurate, so agreement within ~60 ms median means both are tracking the same
boundaries. We also check the first word start / last word end against an
energy-based speech-activity estimate.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from audio.v2.textnorm import align_chars, norm_words, reference_words

ALIGN_MODEL = "facebook/wav2vec2-base-960h"
SR = 16000


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc = Wav2Vec2Processor.from_pretrained(ALIGN_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(ALIGN_MODEL).eval()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    return proc, model


def _emissions(audio16k: np.ndarray):
    import torch

    proc, model = _model()
    inp = proc(audio16k, sampling_rate=SR, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inp.input_values).logits[0]
    return torch.log_softmax(logits, dim=-1).numpy(), audio16k.shape[0] / logits.shape[0]


def _viterbi(em: np.ndarray, tokens: list[int], blank: int):
    """Standard CTC trellis (blank/repeat), returns per-token (start_frame, end_frame)."""
    T, J = em.shape[0], len(tokens)
    S = 2 * J + 1
    ext = [blank if s % 2 == 0 else tokens[s // 2] for s in range(S)]
    NEG = -1e30
    dp = np.full((T, S), NEG, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int16)
    dp[0, 0] = em[0, blank]
    if S > 1:
        dp[0, 1] = em[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        for s in range(S):
            best, arg = prev[s], 0
            if s >= 1 and prev[s - 1] > best:
                best, arg = prev[s - 1], 1
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] and prev[s - 2] > best:
                best, arg = prev[s - 2], 2
            dp[t, s] = best + em[t, ext[s]]
            bp[t, s] = arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.empty(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = [[None, None] for _ in range(J)]
    for t, s in enumerate(path):
        if s % 2 == 1:
            j = s // 2
            if spans[j][0] is None:
                spans[j][0] = t
            spans[j][1] = t
    return spans


def force_align(audio16k: np.ndarray, text: str, refine: bool = True) -> list[dict]:
    """[{"w": surface word, "start_ms", "end_ms"}] relative to the start of `audio16k`."""
    proc, _ = _model()
    vocab = proc.tokenizer.get_vocab()
    blank = proc.tokenizer.pad_token_id
    sep = vocab.get("|")
    words = reference_words(text)
    tok_ids, owner = [], []
    for wi, w in enumerate(words):
        chars = align_chars(w).upper()
        if not chars:
            continue
        if tok_ids and sep is not None:
            tok_ids.append(sep)
            owner.append(-1)
        for ch in chars:
            if ch == " ":
                if sep is not None:
                    tok_ids.append(sep)
                    owner.append(wi)
                continue
            if ch in vocab:
                tok_ids.append(vocab[ch])
                owner.append(wi)
    em, spf = _emissions(audio16k)
    if len(tok_ids) * 2 + 1 > em.shape[0]:
        raise ValueError("audio too short for transcript")
    spans = _viterbi(em, tok_ids, blank)
    frame_ms = spf / SR * 1000.0
    out = []
    for wi, w in enumerate(words):
        fr = [spans[k] for k, o in enumerate(owner) if o == wi and spans[k][0] is not None]
        if fr:
            out.append({"w": w, "start_ms": int(round(min(a for a, _ in fr) * frame_ms)),
                        "end_ms": int(round((max(b for _, b in fr) + 1) * frame_ms))})
        else:  # word with no alignable letters: zero-length at the previous end
            t = out[-1]["end_ms"] if out else 0
            out.append({"w": w, "start_ms": t, "end_ms": t})
    return refine_with_energy(audio16k, out) if refine else out


def refine_with_energy(audio16k: np.ndarray, words: list[dict], rel_db: float = -35.0,
                       max_ext_ms: int = 300, win_ms: int = 10) -> list[dict]:
    """CTC posteriors are peaky: the last frame of a word's final character tends to
    fall before the acoustic end of the word (and similarly for starts). Extend each
    word's end forward (and start backward) through contiguous 10 ms frames whose
    energy is within rel_db of the utterance peak, never crossing the neighbouring
    word and by at most max_ext_ms."""
    n = int(SR * win_ms / 1000)
    fr = audio16k[: len(audio16k) // n * n].reshape(-1, n)
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    voiced = e > e.max() + rel_db
    nf = len(voiced)
    out = [dict(w) for w in words]
    for i, w in enumerate(out):
        if w["end_ms"] <= w["start_ms"]:
            continue
        nxt = out[i + 1]["start_ms"] if i + 1 < len(out) else nf * win_ms
        f, lim = w["end_ms"] // win_ms, min(nxt, w["end_ms"] + max_ext_ms) // win_ms
        while f < min(lim, nf) and voiced[f]:
            f += 1
        w["end_ms"] = max(w["end_ms"], f * win_ms)
        prv = out[i - 1]["end_ms"] if i > 0 else 0
        f, lim = w["start_ms"] // win_ms - 1, max(prv, w["start_ms"] - max_ext_ms) // win_ms
        while f >= max(lim, 0) and voiced[f]:
            f -= 1
        w["start_ms"] = min(w["start_ms"], (f + 1) * win_ms)
    return out


def energy_bounds(audio16k: np.ndarray, rel_db: float = -35.0, win_ms: int = 10) -> tuple[int, int]:
    """(first, last) ms where short-time energy is within rel_db of the peak."""
    n = int(SR * win_ms / 1000)
    frames = audio16k[: len(audio16k) // n * n].reshape(-1, n)
    e = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    on = np.where(e > e.max() + rel_db)[0]
    if len(on) == 0:
        return 0, 0
    return int(on[0] * win_ms), int((on[-1] + 1) * win_ms)


def cross_check(audio16k: np.ndarray, words: list[dict], whisper_model) -> dict:
    """Agreement of forced-alignment word times with Whisper word timestamps."""
    segs, _ = whisper_model.transcribe(audio16k, beam_size=5, word_timestamps=True, language="en")
    ww = [(x.word, x.start * 1000, x.end * 1000) for s in segs for x in (s.words or [])]
    ref = [(" ".join(norm_words(w["w"])), w) for w in words]
    hyp = [(" ".join(norm_words(t)), a, b) for t, a, b in ww]
    # greedy in-order matching on equal normalised tokens
    ds, de, j = [], [], 0
    for rn, w in ref:
        for k in range(j, min(j + 3, len(hyp))):
            if hyp[k][0] == rn and rn:
                ds.append(round(hyp[k][1] - w["start_ms"]))  # signed: whisper - forced
                de.append(round(hyp[k][2] - w["end_ms"]))
                j = k + 1
                break
    on, off = energy_bounds(audio16k)
    return {
        "n_matched": len(ds), "n_words": len(words),
        "start_diff_ms": ds, "end_diff_ms": de,
        "onset_vs_energy_ms": words[0]["start_ms"] - on if words else None,
        "offset_vs_energy_ms": words[-1]["end_ms"] - off if words else None,
    }
