"""
textnorm.py - text normalisation and word error rate for v2 speech QA.

Two normalisers:
  * `norm_words`   - for scoring (WER, revision rate). Uses the Whisper
                     EnglishTextNormalizer when importable (the usual choice in
                     ASR papers; spelled numbers become digits, punctuation goes,
                     British spellings are unified), else a simple fallback.
  * `align_chars`  - for CTC forced alignment: one letters-only string per
                     reference word (digits spelled out), so every reference
                     word gets a timestamp.
"""

from __future__ import annotations

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + int_to_words(rest))
    if 1100 <= n < 10000 and n % 100 != 0 and n // 100 % 10 != 0:   # 1947 -> nineteen forty seven
        return int_to_words(n // 100) + " " + int_to_words(n % 100)
    for div, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div:
            rest = n % div
            return int_to_words(n // div) + " " + name + ("" if rest == 0 else " " + int_to_words(rest))
    return str(n)


def _spell_digits(tok: str) -> str:
    def rep(m):
        s = m.group(0).replace(",", "")
        if "." in s:
            a, b = s.split(".", 1)
            return int_to_words(int(a or 0)) + " point " + " ".join(_ONES[int(c)] for c in b)
        return int_to_words(int(s))
    return re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: " " + rep(m) + " ", tok)


_SYMBOLS = {"&": " and ", "%": " percent ", "₹": " rupees ", "$": " dollars ", "+": " plus ",
            "@": " at ", "=": " equals "}


def align_chars(word: str) -> str:
    """Letters-only (plus apostrophe and space) rendering of one reference word."""
    w = word
    for k, v in _SYMBOLS.items():
        w = w.replace(k, v)
    w = _spell_digits(w)
    w = w.lower().replace("-", " ").replace("/", " ")
    w = re.sub(r"[^a-z' ]", "", w)
    return re.sub(r"\s+", " ", w).strip(" '")


def reference_words(text: str) -> list[str]:
    """The surface reference words: whitespace tokens that contain a letter or digit."""
    return [t for t in text.split() if re.search(r"[A-Za-z0-9]", t)]


try:  # Whisper's normaliser (openai-whisper); identical treatment of ref and hyp.
    from whisper.normalizers import EnglishTextNormalizer as _WN  # type: ignore

    _WHISPER_NORM = _WN()
    NORMALISER = "whisper.EnglishTextNormalizer"
except Exception:  # pragma: no cover - fallback path
    _WHISPER_NORM = None
    NORMALISER = "simple"


def norm_text(text: str) -> str:
    if _WHISPER_NORM is not None:
        return _WHISPER_NORM(text)
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_words(text: str) -> list[str]:
    return norm_text(text).split()


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of the minimum edit alignment."""
    n, m = len(ref), len(hyp)
    # dp over (cost, s, d, i)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur[j] = prev[j - 1]
                continue
            c_sub, c_del, c_ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(c_sub[0], c_del[0], c_ins[0])
            if c_sub[0] == best:
                cur[j] = (best + 1, c_sub[1] + 1, c_sub[2], c_sub[3])
            elif c_del[0] == best:
                cur[j] = (best + 1, c_del[1], c_del[2] + 1, c_del[3])
            else:
                cur[j] = (best + 1, c_ins[1], c_ins[2], c_ins[3] + 1)
        prev = cur
    _, s, d, ins = prev[m]
    return s, d, ins


def word_errors(ref_text: str, hyp_text: str) -> tuple[int, int]:
    """(errors, n_ref_words) after normalisation."""
    r, h = norm_words(ref_text), norm_words(hyp_text)
    s, d, i = edit_ops(r, h)
    return s + d + i, len(r)


def wer(ref_text: str, hyp_text: str) -> float:
    e, n = word_errors(ref_text, hyp_text)
    return e / n if n else float(e > 0)
