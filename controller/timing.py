"""
timing.py — Experiment 2's timing-policy knob: WHEN retrieval launches
relative to the user's speech, config-selected via controller.timing.policy
in configs/config.yaml.

This module deliberately owns only the *timing* logic — resolving which
policy is active, modeling a real elapsed-time window for the text-input
batch path, observing real asr_partial/user_end signals for the live/
synthetic-audio path, and computing the mechanism-check fields that get
logged alongside each retrieval. It does NOT decide whether to retrieve —
that rule (controller/decide.py) never changes with the timing policy, only
the moment pipeline/spine.py calls it does. Composing the two is
pipeline/spine.py's job.

Three policies:

  sentence_end      — retrieve only after user_end (baseline; the pipeline's
                       behavior before this module existed).
  immediate         — retrieve as soon as a partial transcript exists,
                       overlapping retrieval with the tail of the user's
                       speech instead of waiting for it to finish.
  fixed_early_proxy — retrieve at a fixed point in the modeled speech
                       timeline, earlier than `immediate` would react. This
                       is Experiment 2's hard policy. It is NOT a forecast of
                       a real user-hold/pause window — there's no
                       pause-prediction signal wired in. transcribe_stream()
                       (audio/stt.py) only surfaces the *final* trailing-
                       silence endpoint (user_end), never an intermediate
                       silence gap, so a genuine forecast would need editing
                       Adithi's file, which is out of scope here — this
                       policy instead dispatches at a clearly-labeled fixed
                       proxy point (see PartialSignalWatcher.repeated_partial
                       and plan_simulated_speech's pause_at_s), with a
                       structural "proxy assumption invalidated -> cancel"
                       path wired in now so swapping the proxy for a real
                       forecast/VAD signal later is a one-function change,
                       not a redesign.

Why the resulting latency difference is real, not asserted:
  Retrieval (controller_decide + rewrite_query + retrieve — all real,
  synchronous work) always used to run strictly after user_end. Making a
  policy "faster" here always means launching that *same* unmodified work
  earlier as a real asyncio task, so it races against whatever time is
  actually left before user_end. Nothing subtracts a number from a metric
  anywhere. harness/metrics.py, unmodified, computes TTFA/dead-air purely
  from the real t_ms timestamps this module and pipeline/spine.py produce —
  if a policy doesn't actually cause real overlap, the numbers won't move.

The text-input batch path (no real audio) is the one place a modeling
assumption is unavoidable: plan_simulated_speech() decides how long saying
a transcript would take (word_count / words_per_minute) and pipeline/
spine.py actually asyncio.sleep()s through it, in real wall-clock time, so
the same race applies there too. That duration is a function of the actual
scenario text, not a constant, and the outcome still depends on how that
real sleep interacts with the real retrieval latency — the assumption is
"how long is the window," never "how much faster immediate is."

The query_ready gate (Experiment 2 fix, see is_query_ready() below):
immediate/fixed_early_proxy originally triggered on the first reactive signal
available (first partial; first repeated/stable partial) with no floor on
how much of the query that signal actually contained — on short scenario
questions that's a single word ("What"/"Tell"/"Can"), so retrieval ran on
a query with no content in it. hit@3 collapsed even though TTFA improved:
fast but wrong, which fails Experiment 2's equal-accuracy condition. The
fix doesn't touch WHETHER or on WHAT the early policies retrieve relative
to sentence_end — it only delays their dispatch point until the (partial)
query has query_ready's minimum word count and at least one content word,
i.e. dispatch at the EARLIEST *meaningful* partial, not the earliest
partial. See SpeechPlan.dispatch_at_s() (text path) and
PartialSignalWatcher.first_ready_partial / repeated_ready_partial (live
path).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eventlog.log_writer import now_ms  # noqa: E402 -- read-only import, see PartialSignalWatcher

_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"

VALID_POLICIES = {"sentence_end", "immediate", "fixed_early_proxy", "forecast"}

# Mirrors configs/config.yaml's controller.timing — kept in sync the same
# way controller/decide.py's _DEFAULT_DECISION_CFG mirrors controller.decision.
_DEFAULT_TIMING_CFG = {
    "policy": "sentence_end",
    "simulated_speech": {
        "words_per_minute": 150,
        "pause_at_fraction": 0.35,
    },
    # query_ready gate (Experiment 2 fix): immediate/fixed_early_proxy must
    # not dispatch retrieval on a (partial) query this short/empty of
    # content — see is_query_ready() below and SpeechPlan.dispatch_at_s().
    "query_ready": {
        "min_words": 4,
        "require_content_word": True,
    },
    # research v2: the pre-registered forecast-timed policy. See
    # forecast_should_dispatch() below.
    "forecast": {
        "rule": "jit",                 # "jit" | "earliest"
        "remaining_quantile": "q50",   # which predictor quantile to trust
        "latency_quantile": "p50",     # which retrieval-latency quantile to hide
        "margin_ms": 0,                # extra safety margin on top of the latency
        "extra_latency_ms": 0,         # e.g. a live query rewrite before retrieval
        # the live-partial path rewrites before retrieving, so the
        # hideable latency is rewrite + search (retrieval/compose_profiles.py)
        "retrieval_profile": "rewrite_faiss",
        "predictor_path": None,        # saved RemainingSpeechPredictor JSON
        # used only when neither signals["retrieval"] nor the latency
        # profile artifact is available (E2 in-pipeline median)
        "fallback_expected_latency_ms": 672,
    },
}

# Deliberately small — the point isn't real NLP stopword filtering, just
# rejecting a partial that's *entirely* question/function words (e.g.
# "What is the", "Can you tell") so query_ready needs at least one word
# that actually carries the user's intent.
_QUERY_READY_STOPWORDS = {
    "what", "when", "where", "who", "why", "how", "which",
    "is", "are", "was", "were", "am", "be", "been",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "the", "a", "an",
    "i", "me", "my", "you", "your", "we", "our", "us",
    "it", "its", "this", "that",
    "to", "of", "in", "on", "at",
    "tell", "please",
}


def _is_content_word(word: str) -> bool:
    """A word counts as "content" once stripped of surrounding punctuation
    and lowercased, unless it's in the small stopword set above. Digits
    (e.g. a room number) always count as content."""
    core = word.strip(".,!?;:\"'").lower()
    if not core:
        return False
    if core.isdigit():
        return True
    return core not in _QUERY_READY_STOPWORDS


def is_query_ready(text: str, query_ready_cfg: Optional[dict] = None, path: Optional[str] = None) -> bool:
    """
    Whether `text` (a full or partial transcript) has enough content to
    retrieve on meaningfully — Experiment 2's query_ready gate, per
    controller.timing.query_ready. Two conditions, both required:

      - at least `min_words` words, AND
      - (if require_content_word) at least one word outside
        _QUERY_READY_STOPWORDS — so "What is the" (3+ function words)
        doesn't pass on word count alone.

    The early timing policies must dispatch at the EARLIEST partial that
    satisfies this, not the first word — see this module's docstring and
    SpeechPlan.dispatch_at_s() / PartialSignalWatcher's *_ready_partial
    futures, which are what actually apply the gate.
    """
    cfg = query_ready_cfg if query_ready_cfg is not None else load_timing_config(path)["query_ready"]
    words = text.split()
    if len(words) < int(cfg.get("min_words", 3)):
        return False
    if cfg.get("require_content_word", True) and not any(_is_content_word(w) for w in words):
        return False
    return True


def _earliest_ready_word_count(words: List[str], query_ready_cfg: dict) -> int:
    """
    Smallest prefix length of `words` for which is_query_ready() is true —
    the earliest point in the (simulated) speech timeline actually worth
    retrieving on. Falls back to the full word count (i.e. "wait for the
    whole thing") on the rare transcript that's never ready even in full
    (e.g. all stopwords) — retrieval still fires eventually, just no
    earlier than sentence_end would.
    """
    for n in range(1, len(words) + 1):
        if is_query_ready(" ".join(words[:n]), query_ready_cfg):
            return n
    return max(len(words), 1)


def _load_full_config(path: Optional[str] = None) -> dict:
    """Read the whole configs/config.yaml, not just one section — this
    module needs both controller.timing (its own) and audio.stt.
    partial_interval_ms (Adithi's, read-only, to anchor `immediate`'s
    trigger to the STT's real partial cadence instead of inventing one)."""
    import yaml  # imported here so `import controller.timing` doesn't require it

    cfg_file = Path(path) if path else Path(
        os.environ.get("VOICE_RAG_CONFIG") or _DEFAULT_CONFIG_PATH
    )
    try:
        with open(cfg_file, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def load_timing_config(path: Optional[str] = None) -> dict:
    """Read controller.timing, merged over the defaults above."""
    full = _load_full_config(path)
    timing_cfg = (full.get("controller") or {}).get("timing") or {}
    merged = {**_DEFAULT_TIMING_CFG, **timing_cfg}
    merged["simulated_speech"] = {
        **_DEFAULT_TIMING_CFG["simulated_speech"],
        **(timing_cfg.get("simulated_speech") or {}),
    }
    merged["query_ready"] = {
        **_DEFAULT_TIMING_CFG["query_ready"],
        **(timing_cfg.get("query_ready") or {}),
    }
    merged["forecast"] = {
        **_DEFAULT_TIMING_CFG["forecast"],
        **(timing_cfg.get("forecast") or {}),
    }
    return merged


def resolve_policy(path: Optional[str] = None) -> str:
    """
    Which timing policy is active right now.

    VOICE_RAG_TIMING_POLICY overrides the config file, matching the
    VOICE_RAG_LLM_BACKEND / VOICE_RAG_MODEL override convention already used
    elsewhere in configs/config.yaml — handy for running Experiment 2's
    three conditions back to back without hand-editing the file between
    runs.
    """
    policy = os.environ.get("VOICE_RAG_TIMING_POLICY") or load_timing_config(path).get(
        "policy", "sentence_end"
    )
    if policy not in VALID_POLICIES:
        print(
            f"[controller/timing] WARNING: unknown timing policy '{policy}' "
            f"— falling back to sentence_end."
        )
        policy = "sentence_end"
    return policy


def _partial_interval_s(path: Optional[str] = None) -> float:
    """audio.stt.partial_interval_ms, in seconds — read-only, so the text
    path's `immediate` trigger reuses the real STT's own partial cadence
    instead of a second, redundant number."""
    full = _load_full_config(path)
    stt_cfg = ((full.get("audio") or {}).get("stt")) or {}
    return stt_cfg.get("partial_interval_ms", 400) / 1000.0


# ---------------------------------------------------------------------------
# Text-input batch path: a real elapsed-time window to trigger inside
# ---------------------------------------------------------------------------
@dataclass
class SpeechPlan:
    """
    How long saying `transcript` would take, and when `immediate` /
    `fixed_early_proxy` would trigger inside that window — all in seconds,
    all meant to be actually asyncio.sleep()'d through by pipeline/
    spine.py, not just recorded.
    """

    total_speech_s: float
    partial_at_s: float  # immediate's raw reactive-signal trigger point
    pause_at_s: float    # fixed_early_proxy's raw fixed-proxy trigger point (a fixed fraction of the modeled speech duration, not a reactive signal; earlier than partial_at_s — see module docstring)
    ready_at_s: float         # earliest elapsed_s at which the revealed transcript satisfies query_ready (see dispatch_at_s)
    partial_interval_s: float  # audio.stt.partial_interval_ms in seconds — see dispatch_at_s's immediate branch
    words: List[str] = field(default_factory=list)

    def dispatch_at_s(self, policy: str) -> float:
        """
        Experiment 2 fix: the actual time `policy` ("immediate" or
        "fixed_early_proxy") should dispatch retrieval at, after applying
        the query_ready gate. Both are floored at ready_at_s — the
        earliest point the revealed transcript actually has enough
        content to retrieve on meaningfully (see is_query_ready()) — and
        capped at total_speech_s, so a transcript that's never "ready"
        still dispatches, at worst at the same moment sentence_end would.

        Below that shared floor, ready_at_s always dominates partial_at_s/
        pause_at_s for any min_words >= 2 (reaching two words alone takes
        longer than either raw trigger point), which would otherwise make
        immediate and fixed_early_proxy dispatch at the exact same
        instant — collapsing the very timing difference Experiment 2
        measures. So the two policies apply the shared floor differently,
        matching what each is actually modeling:

          fixed_early_proxy — a fixed proxy point in the modeled speech
          timeline, not a forecast of anything real. It acts the moment
          the query_ready floor is met at that fixed point, without
          waiting to observe anything real confirm it. Dispatches at
          exactly max(pause_at_s, ready_at_s).

          immediate — reactive. It only acts once it *observes* a partial
          actually carrying that much content, which (mirroring the real
          STT's partial_interval_ms cadence) takes at least one more
          partial-arrival tick beyond the instant the transcript
          technically becomes ready. Dispatches at
          max(partial_at_s, ready_at_s + partial_interval_s).

        This keeps fixed_early_proxy's documented "earlier than immediate"
        property real after the accuracy gate, instead of the gate erasing
        it — while never letting either policy dispatch before ready_at_s.
        """
        if policy == "fixed_early_proxy":
            raw_trigger_s = max(self.pause_at_s, self.ready_at_s)
        else:
            raw_trigger_s = max(self.partial_at_s, self.ready_at_s + self.partial_interval_s)
        return min(raw_trigger_s, self.total_speech_s)

    def text_at(self, elapsed_s: float) -> str:
        """
        A word-prefix of the transcript proportional to elapsed_s /
        total_speech_s. Stands in for "what a partial transcript would
        plausibly contain by this point" — there's no real audio to
        transcribe in the text-input path, so this is the modeling
        assumption plan_simulated_speech's docstring flags, applied.
        """
        if self.total_speech_s <= 0 or not self.words:
            return " ".join(self.words)
        frac = max(0.0, min(1.0, elapsed_s / self.total_speech_s))
        n = max(1, round(len(self.words) * frac))
        return " ".join(self.words[:n])


def plan_simulated_speech(transcript: str, path: Optional[str] = None) -> SpeechPlan:
    """
    Build the text-path's speech timeline for one transcript.

    total_speech_s comes from word_count / words_per_minute — a function of
    the actual scenario text, not a constant, so different scenarios get
    different windows and the eventual TTFA delta isn't a fixed number
    either. partial_at_s anchors to the real audio.stt.partial_interval_ms
    (capped so it can't exceed the speech window); pause_at_s is
    deliberately earlier still, capped below partial_at_s, since
    fixed_early_proxy is supposed to fire ahead of a merely-reactive
    partial (immediate) — it doesn't wait to observe anything real.
    """
    words = transcript.split()
    timing_cfg = load_timing_config(path)
    sim_cfg = timing_cfg["simulated_speech"]
    wpm = max(float(sim_cfg["words_per_minute"]), 1.0)

    total_speech_s = max(len(words), 1) / wpm * 60.0
    total_speech_s = max(total_speech_s, 0.3)  # floor: even a 1-word turn gets a real window

    partial_interval_s = _partial_interval_s(path)
    partial_at_s = min(partial_interval_s, total_speech_s * 0.9)
    pause_at_s = min(total_speech_s * float(sim_cfg["pause_at_fraction"]), partial_at_s * 0.8)

    # query_ready gate: the earliest word-prefix of the transcript that's
    # actually meaningful to retrieve on, converted to elapsed seconds via
    # the same word-count/total_speech_s proportion text_at() uses. See
    # SpeechPlan.dispatch_at_s(), which combines this with partial_at_s /
    # pause_at_s above.
    ready_word_count = _earliest_ready_word_count(words, timing_cfg["query_ready"])
    ready_at_s = (ready_word_count / max(len(words), 1)) * total_speech_s

    return SpeechPlan(
        total_speech_s=total_speech_s,
        partial_at_s=partial_at_s,
        pause_at_s=pause_at_s,
        ready_at_s=ready_at_s,
        partial_interval_s=partial_interval_s,
        words=words,
    )


# ---------------------------------------------------------------------------
# research v2: the forecast-timed policy
# ---------------------------------------------------------------------------
# The pre-registration named a "forecast-timed" policy: retrieve during the
# predicted user-still-talking window. E2 shipped fixed_early_proxy instead
# (a fixed fraction of the modeled speech, no forecast). `forecast` is the
# real thing, driven by two measured signals:
#
#   R(t)  predicted remaining speech at partial t
#         (controller/remaining_speech.py; signals["timing"]
#         ["predicted_remaining_speech_ms"])
#   L     expected retrieval latency of the active retrieval profile
#         (signals["retrieval"], from retrieval/latency_profiles_v2.json),
#         plus margin_ms and any extra pre-retrieval work
#         (extra_latency_ms, e.g. a live query rewrite)
#
# Dispatch rules (both behind the same query_ready content gate as the E2
# policies, and both falling back to sentence_end when never triggered):
#
#   jit       (default) dispatch at the first query-ready partial where
#             waiting one more partial interval would leave less predicted
#             speech than the latency to hide: R(t) - dt < L, i.e.
#             R(t) < L + dt. This is the last partial at which remaining
#             speech is still predicted to cover the retrieval latency
#             (R(t) >= L, unless it was already too late at the first
#             ready partial). It buys as much query content as possible
#             without exposing latency: the information-vs-overlap
#             trade-off (C2) resolved by the forecast.
#   earliest  dispatch at the first query-ready partial with R(t) >= L, or
#             at the first query-ready partial at all if R(t) < L there
#             (nothing left to hide by waiting). Equivalent to a gated
#             reactive policy; kept as an ablation.
#
# fixed_early_proxy is untouched (v1 reproducibility).


def forecast_should_dispatch(
    text: str,
    predicted_remaining_ms: Optional[float],
    expected_latency_ms: float,
    partial_interval_ms: float,
    forecast_cfg: Optional[dict] = None,
    query_ready_cfg: Optional[dict] = None,
) -> Tuple[bool, str]:
    """Decision for one partial. Returns (dispatch, reason)."""
    cfg = {**_DEFAULT_TIMING_CFG["forecast"], **(forecast_cfg or {})}
    qr_cfg = query_ready_cfg if query_ready_cfg is not None else _DEFAULT_TIMING_CFG["query_ready"]
    if not is_query_ready(text, qr_cfg):
        return False, "query_not_ready"
    if predicted_remaining_ms is None:
        # No forecast available: behave like the gated reactive policy.
        return True, "no_forecast_signal"
    budget = (
        float(expected_latency_ms)
        + float(cfg.get("margin_ms", 0) or 0)
        + float(cfg.get("extra_latency_ms", 0) or 0)
    )
    rule = cfg.get("rule", "jit")
    if rule == "jit":
        if predicted_remaining_ms < budget + float(partial_interval_ms):
            return True, "jit:remaining<latency+interval"
        return False, "jit:waiting_for_more_query"
    if rule == "earliest":
        return True, ("earliest:hideable" if predicted_remaining_ms >= budget else "earliest:not_hideable")
    raise ValueError(f"unknown forecast rule {rule!r}")


def lookup_prediction(predictions: Optional[List[dict]], t_ms: float, key: str = "remaining_ms") -> Optional[float]:
    """The latest prediction entry at or before t_ms (CONTRACTS §6 list)."""
    best = None
    for p in predictions or []:
        if p["t_ms"] <= t_ms + 1e-6 and (best is None or p["t_ms"] >= best["t_ms"]):
            best = p
    if best is None:
        return None
    return best.get(key, best.get("remaining_ms"))


def forecast_dispatch(
    partials: List[dict],
    predict,
    expected_latency_ms: float,
    partial_interval_ms: float,
    forecast_cfg: Optional[dict] = None,
    query_ready_cfg: Optional[dict] = None,
    onset_ms: float = 0.0,
) -> Optional[dict]:
    """
    Offline/trace version of the forecast policy: the first partial (in time
    order) at which forecast_should_dispatch() fires, or None (dispatch at
    user_end). `predict` is a callable (text, elapsed_ms) -> remaining ms,
    a precomputed signals["timing"]["predicted_remaining_speech_ms"] list,
    or None (no forecast signal).
    """
    for p in sorted(partials, key=lambda x: x["t_ms"]):
        if callable(predict):
            r = predict(p.get("text", ""), p["t_ms"] - onset_ms)
        elif predict is None:
            r = None
        else:
            r = lookup_prediction(predict, p["t_ms"])
        ok, reason = forecast_should_dispatch(
            p.get("text", ""), r, expected_latency_ms, partial_interval_ms, forecast_cfg, query_ready_cfg
        )
        if ok:
            return {"t_ms": p["t_ms"], "text": p.get("text", ""), "predicted_remaining_ms": r, "reason": reason}
    return None


def resolve_expected_latency_ms(signals: Optional[dict] = None, forecast_cfg: Optional[dict] = None) -> Tuple[float, str]:
    """
    L for the forecast policy: signals["retrieval"] if the caller passed it,
    else the measured latency-profile artifact for the configured profile,
    else the configured fallback. Returns (latency_ms, source).
    """
    cfg = {**_DEFAULT_TIMING_CFG["forecast"], **(forecast_cfg or {})}
    q = cfg.get("latency_quantile", "p50")
    r = (signals or {}).get("retrieval") or {}
    if (r.get("latency_quantiles_ms") or {}).get(q) is not None:
        return float(r["latency_quantiles_ms"][q]), f"signals.retrieval.{q}"
    if r.get("expected_latency_ms") is not None:
        return float(r["expected_latency_ms"]), "signals.retrieval.expected"
    try:
        from retrieval.profiles import retrieval_signals

        sig = retrieval_signals(cfg.get("retrieval_profile", "faiss"), expected_quantile=q)
        return float(sig["expected_latency_ms"]), f"latency_profiles_v2:{sig['latency_profile']}.{q}"
    except Exception:
        return float(cfg.get("fallback_expected_latency_ms", 672)), "config_fallback"


def load_forecast_predictor(forecast_cfg: Optional[dict] = None):
    """The configured RemainingSpeechPredictor, or None if none is configured/available."""
    cfg = {**_DEFAULT_TIMING_CFG["forecast"], **(forecast_cfg or {})}
    path = os.environ.get("VOICE_RAG_REMAINING_SPEECH_MODEL") or cfg.get("predictor_path")
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.exists():
        print(f"[controller/timing] WARNING: forecast predictor {p} not found; forecast degrades to gated reactive.")
        return None
    from controller.remaining_speech import RemainingSpeechPredictor

    return RemainingSpeechPredictor.load(p)


# ---------------------------------------------------------------------------
# Live/synthetic-audio path: observing the real asr_partial/user_end signals
# ---------------------------------------------------------------------------
class PartialSignalWatcher:
    """
    Wraps the real LogWriter passed into audio/stt.py's transcribe_stream()
    so pipeline/spine.py can react to asr_partial/user_end as they really
    happen, on the real monotonic clock, without editing audio/stt.py.

    Every log_event() call is forwarded to the real writer completely
    unchanged — this only *additionally* resolves a few asyncio futures and
    records a couple of timestamps so spine.py's early-retrieval triggers
    have something real to await. It never fabricates or delays an event.

    query_ready_cfg (controller.timing.query_ready, defaulted from config
    if not given): the same Experiment 2 gate is_query_ready() applies on
    the text path applies here too, via first_ready_partial /
    repeated_ready_partial below — a real partial resolves those futures
    only once it actually has enough content, not on the first word.
    first_partial / repeated_partial (unfiltered) are kept as-is for
    anything that wants the raw signal.
    """

    def __init__(
        self,
        inner,
        turn_id,
        query_ready_cfg: Optional[dict] = None,
        forecast: Optional[dict] = None,
    ):
        self._inner = inner
        self._turn_id = turn_id
        self._loop = asyncio.get_running_loop()
        self._query_ready_cfg = (
            query_ready_cfg if query_ready_cfg is not None else load_timing_config()["query_ready"]
        )
        self.first_partial: asyncio.Future = self._loop.create_future()
        self.repeated_partial: asyncio.Future = self._loop.create_future()
        self.first_ready_partial: asyncio.Future = self._loop.create_future()
        self.repeated_ready_partial: asyncio.Future = self._loop.create_future()
        # research v2 forecast policy (optional): dict with keys
        #   predictor (RemainingSpeechPredictor or None), expected_latency_ms,
        #   partial_interval_ms, forecast_cfg, onset_ms
        # forecast_partial resolves to {"text","t_ms","predicted_remaining_ms","reason"}.
        self._forecast = forecast
        self.forecast_partial: asyncio.Future = self._loop.create_future()
        self.user_end_ms: Optional[int] = None
        self._last_partial_text: Optional[str] = None
        self._growth_waiters: List[Tuple[str, asyncio.Future]] = []

    def log_event(self, actor, event_type, payload, turn_id):
        self._inner.log_event(actor, event_type, payload, turn_id)
        if turn_id != self._turn_id:
            return

        if event_type == "asr_partial":
            text = (payload or {}).get("text", "")

            if not self.first_partial.done():
                self.first_partial.set_result(text)

            ready = is_query_ready(text, self._query_ready_cfg)
            if ready and not self.first_ready_partial.done():
                self.first_ready_partial.set_result(text)

            # Pause proxy: two consecutive partials with identical text —
            # whisper's guess stops changing while the user is briefly
            # silent mid-utterance, since no new words arrived to
            # transcribe. Approximate, not a real silence-gap measurement;
            # see this module's docstring.
            is_repeat = (
                self._last_partial_text is not None and text == self._last_partial_text
            )
            if is_repeat and not self.repeated_partial.done():
                self.repeated_partial.set_result(text)
            if is_repeat and ready and not self.repeated_ready_partial.done():
                self.repeated_ready_partial.set_result(text)

            if self._forecast is not None and not self.forecast_partial.done():
                fc = self._forecast
                t_ms = now_ms()
                elapsed = t_ms - fc.get("onset_ms", t_ms)
                predictor = fc.get("predictor")
                key = (fc.get("forecast_cfg") or {}).get("remaining_quantile", "q50")
                r = predictor.predict(text, elapsed)[key] if predictor is not None else None
                ok, reason = forecast_should_dispatch(
                    text, r, fc["expected_latency_ms"], fc["partial_interval_ms"],
                    fc.get("forecast_cfg"), self._query_ready_cfg,
                )
                if ok:
                    self.forecast_partial.set_result(
                        {"text": text, "t_ms": t_ms, "predicted_remaining_ms": r, "reason": reason}
                    )

            # Proxy-assumption-invalidation signal: a later partial that's
            # grown past a given baseline means speech continued past the
            # point fixed_early_proxy assumed a pause — the caller uses
            # this to cancel a now-stale early retrieval. See
            # wait_for_growth().
            for baseline, fut in list(self._growth_waiters):
                if not fut.done() and text != baseline and len(text) > len(baseline):
                    fut.set_result(text)
                    self._growth_waiters.remove((baseline, fut))

            self._last_partial_text = text

        elif event_type == "user_end":
            self.user_end_ms = now_ms()

    def wait_for_growth(self, baseline_text: str) -> asyncio.Future:
        """
        A future that resolves the moment a partial arrives that's grown
        past `baseline_text` — i.e. the fixed proxy point's assumption that
        a pause was starting at baseline_text turned out to be wrong.
        Cancel it if you stop caring (e.g. the early retrieval already
        finished).
        """
        fut = self._loop.create_future()
        if baseline_text and self._last_partial_text and len(self._last_partial_text) > len(
            baseline_text
        ):
            # Growth already happened before we started watching for it.
            fut.set_result(self._last_partial_text)
        else:
            self._growth_waiters.append((baseline_text, fut))
        return fut


# ---------------------------------------------------------------------------
# Mechanism-check fields — the "hidden latency" bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class MechanismContext:
    """
    What pipeline/spine.py's retrieve() needs to log the mechanism-check
    fields honestly: which policy/trigger is active, and two getters that
    return the real user_end timestamp / real speech window the moment
    they're knowable (None before that — never guessed).
    """

    policy: str
    trigger: str
    user_end_ms_getter: Callable[[], Optional[int]]
    speech_window_ms_getter: Callable[[], Optional[Tuple[int, int]]]
    forecast_invalidated: bool = False


def mechanism_launch_fields(mechanism: MechanismContext, launch_ms: int) -> dict:
    """
    Fields for the retrieval_launched payload. `launch_offset_ms_vs_user_end`
    is negative exactly when retrieval launched during speech (before
    user_end) — the "hidden latency" the harness needs to correlate against
    outcome. It's None only when user_end genuinely hasn't happened yet at
    launch time (early triggers in the live path) — that's the honest
    answer, not a placeholder; the retrieval_result payload gets another
    chance to fill it in once more time has passed.
    """
    user_end_ms = mechanism.user_end_ms_getter()
    return {
        "timing_policy": mechanism.policy,
        "timing_trigger": mechanism.trigger,
        "user_end_known_at_launch": user_end_ms is not None,
        "launch_offset_ms_vs_user_end": (
            launch_ms - user_end_ms if user_end_ms is not None else None
        ),
    }


def mechanism_result_fields(mechanism: MechanismContext, launch_ms: int, finish_ms: int) -> dict:
    """
    Fields for the retrieval_result payload: the same offset, re-checked
    (often known by now even if it wasn't at launch), plus
    overlap_with_speech_ms — how much of [launch_ms, finish_ms] actually
    fell inside the (real or modeled) speech window — and
    forecast_invalidated for fixed_early_proxy's cancel-on-invalidation
    path (field name kept for log-schema stability; what it actually
    tracks is the fixed proxy point's assumption being invalidated by
    continued speech, not a real forecast going wrong).
    """
    user_end_ms = mechanism.user_end_ms_getter()
    window = mechanism.speech_window_ms_getter()
    overlap_ms = None
    if window is not None:
        w_start, w_end = window
        overlap_ms = max(0, min(finish_ms, w_end) - max(launch_ms, w_start))
    return {
        "timing_policy": mechanism.policy,
        "timing_trigger": mechanism.trigger,
        "user_end_known_at_finish": user_end_ms is not None,
        "launch_offset_ms_vs_user_end": (
            launch_ms - user_end_ms if user_end_ms is not None else None
        ),
        "overlap_with_speech_ms": overlap_ms,
        "forecast_invalidated": mechanism.forecast_invalidated,
    }
