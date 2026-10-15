"""Prefix conditions: what the model has heard when the belief is computed.

v1 E1 computed every signal on the FULL question, but an early-timing policy
dispatches retrieval before the user finishes, so the belief it acts on is a
belief about a PREFIX. E1-v2 therefore computes signals per prefix:

  full     the complete question text            kind=full    source=text
  text     the first ceil-free floor(f * n) words kind=partial source=text
           at word fractions f (0.25, 0.5, 0.75)
  asr      every streaming-ASR partial of the     kind=partial source=asr
           selected speech traces, stamped with t_ms (speech time) and
           available_ms (when the partial text existed)
  asr final the final ASR transcript             kind=full    source=asr

Many conditions produce the same text (short questions, repeated ASR
partials), and the model's output depends only on the text, so the runner
makes one call per distinct text per (item, seed) and writes one record per
condition. See group_by_text().
"""

from __future__ import annotations

import math

from llm.experiment1_v2.common import sha256_text


def normalize_text(text: str) -> str:
    """Whitespace-normalized text. Case and punctuation are kept: the model
    sees them, so two prefixes that differ only there are different inputs."""
    return " ".join((text or "").split())


def n_prefix_words(n_words: int, frac: float) -> int:
    """Words kept at word fraction `frac`: floor(frac * n), at least 1.

    floor, not round, so a prefix never contains a word the speaker had not
    yet finished at that fraction of the word sequence."""
    if n_words <= 0:
        return 0
    return max(1, min(n_words, int(math.floor(frac * n_words + 1e-9))))


def _prefix(kind, source, text, *, word_frac, n_words, voice_id=None, noise=None,
            t_ms=None, available_ms=None, **extra) -> dict:
    record = {
        "kind": kind,
        "source": source,
        "voice_id": voice_id,
        "noise": noise,
        "t_ms": t_ms,
        "available_ms": available_ms,
        "word_frac": word_frac,
        "n_words": n_words,
        "text": text,
    }
    record.update(extra)
    return record


def text_prefixes(question: str, word_fracs=(0.25, 0.5, 0.75), include_full=True) -> list:
    words = normalize_text(question).split(" ")
    words = [w for w in words if w]
    n = len(words)
    out = []
    if include_full:
        out.append(_prefix("full", "text", normalize_text(question), word_frac=1.0, n_words=n))
    for frac in word_fracs:
        keep = n_prefix_words(n, float(frac))
        out.append(
            _prefix(
                "partial", "text", " ".join(words[:keep]),
                word_frac=float(frac), n_words=keep,
                actual_word_frac=(keep / n) if n else None,
            )
        )
    return out


def spoken_word_fraction(trace: dict, t_ms) -> float | None:
    """Fraction of the trace's (reference-aligned) words fully spoken by t_ms."""
    words = trace.get("words") or []
    if not words or t_ms is None:
        return None
    done = sum(1 for w in words if w.get("end_ms") is not None and w["end_ms"] <= t_ms)
    return done / len(words)


def asr_prefixes(trace: dict, include_final=True) -> list:
    """One prefix per non-empty ASR partial (plus the final transcript).

    available_ms defaults to t_ms when the trace does not carry it. Empty
    partials are skipped: there is nothing to condition on, so before the
    first non-empty partial the belief is simply not available (analyze.py
    reports that as reduced n, not as a guess)."""
    common = {
        "voice_id": trace.get("voice_id"),
        "noise": trace.get("noise"),
        "accent": trace.get("accent"),
        "vad_user_end_ms": trace.get("vad_user_end_ms"),
        "duration_ms": trace.get("duration_ms"),
    }
    out = []
    for partial in trace.get("partials") or []:
        text = normalize_text(partial.get("text", ""))
        if not text:
            continue
        t_ms = partial.get("t_ms")
        out.append(
            _prefix(
                "partial", "asr", text,
                word_frac=spoken_word_fraction(trace, t_ms),
                n_words=len(text.split(" ")),
                t_ms=t_ms,
                available_ms=partial.get("available_ms", t_ms),
                stable=partial.get("stable"),
                **common,
            )
        )
    final = trace.get("asr_final") or {}
    if include_final and normalize_text(final.get("text", "")):
        text = normalize_text(final["text"])
        out.append(
            _prefix(
                "full", "asr", text,
                word_frac=1.0,
                n_words=len(text.split(" ")),
                t_ms=final.get("t_ms"),
                available_ms=final.get("available_ms", final.get("t_ms")),
                **common,
            )
        )
    return out


def prefix_id(prefix: dict) -> str:
    """Stable id of one prefix CONDITION (not of its text): two conditions
    with identical text keep distinct ids, so each gets its own record."""
    return sha256_text(
        prefix["kind"], prefix["source"], prefix.get("voice_id"), prefix.get("noise"),
        prefix.get("t_ms"), prefix.get("available_ms"), prefix.get("word_frac"),
        prefix["text"],
    )[:16]


def item_prefixes(item: dict, traces=None, word_fracs=(0.25, 0.5, 0.75),
                  text=True, asr=True, include_asr_final=True) -> list:
    """All prefix conditions for one item, each tagged with its prefix_id.
    Duplicate CONDITIONS (same id) are dropped; duplicate TEXTS are kept
    here and collapsed into one call by group_by_text()."""
    out = []
    if text:
        out.extend(text_prefixes(item["question"], word_fracs))
    if asr:
        for trace in traces or []:
            out.extend(asr_prefixes(trace, include_final=include_asr_final))
    seen, unique = set(), []
    for prefix in out:
        pid = prefix_id(prefix)
        if pid in seen:
            continue
        seen.add(pid)
        unique.append({**prefix, "prefix_id": pid})
    return unique


def group_by_text(prefixes: list) -> list:
    """[(text, [prefix, ...]), ...] in first-seen order. One model call per
    group; every prefix in the group gets a record from that one call."""
    groups: dict = {}
    for prefix in prefixes:
        groups.setdefault(normalize_text(prefix["text"]), []).append(prefix)
    return list(groups.items())
