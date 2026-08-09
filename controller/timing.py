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

  sentence_end    — retrieve only after user_end (baseline; the pipeline's
                     behavior before this module existed).
  immediate       — retrieve as soon as a partial transcript exists,
                     overlapping retrieval with the tail of the user's
                     speech instead of waiting for it to finish.
  predicted_pause — retrieve on a *forecast* mid-utterance pause, earlier
                     than `immediate` would react. This is Experiment 2's
                     hard policy. A real forecast needs a signal
                     audio/stt.py doesn't expose today: transcribe_stream()
                     (audio/stt.py) only surfaces the *final* trailing-
                     silence endpoint (user_end), never an intermediate
                     silence gap. Adding that would mean editing Adithi's
                     file, which is out of scope here — so this policy runs
                     on a clearly-labeled proxy instead (see
                     PartialSignalWatcher.repeated_partial and
                     plan_simulated_speech's pause_at_s), with a structural
                     "forecast invalidated -> cancel" path wired in now so
                     swapping the proxy for a real signal later is a
                     one-function change, not a redesign.

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

VALID_POLICIES = {"sentence_end", "immediate", "predicted_pause"}

# Mirrors configs/config.yaml's controller.timing — kept in sync the same
# way controller/decide.py's _DEFAULT_DECISION_CFG mirrors controller.decision.
_DEFAULT_TIMING_CFG = {
    "policy": "sentence_end",
    "simulated_speech": {
        "words_per_minute": 150,
        "pause_at_fraction": 0.35,
    },
}


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
    `predicted_pause` would trigger inside that window — all in seconds,
    all meant to be actually asyncio.sleep()'d through by pipeline/
    spine.py, not just recorded.
    """

    total_speech_s: float
    partial_at_s: float  # immediate's trigger point
    pause_at_s: float    # predicted_pause's trigger point (earlier; stub — see module docstring)
    words: List[str] = field(default_factory=list)

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
    predicted_pause is supposed to fire ahead of a merely-reactive partial.
    """
    words = transcript.split()
    timing_cfg = load_timing_config(path)
    sim_cfg = timing_cfg["simulated_speech"]
    wpm = max(float(sim_cfg["words_per_minute"]), 1.0)

    total_speech_s = max(len(words), 1) / wpm * 60.0
    total_speech_s = max(total_speech_s, 0.3)  # floor: even a 1-word turn gets a real window

    partial_at_s = min(_partial_interval_s(path), total_speech_s * 0.9)
    pause_at_s = min(total_speech_s * float(sim_cfg["pause_at_fraction"]), partial_at_s * 0.8)

    return SpeechPlan(
        total_speech_s=total_speech_s,
        partial_at_s=partial_at_s,
        pause_at_s=pause_at_s,
        words=words,
    )


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
    """

    def __init__(self, inner, turn_id):
        self._inner = inner
        self._turn_id = turn_id
        self._loop = asyncio.get_running_loop()
        self.first_partial: asyncio.Future = self._loop.create_future()
        self.repeated_partial: asyncio.Future = self._loop.create_future()
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

            # Pause proxy: two consecutive partials with identical text —
            # whisper's guess stops changing while the user is briefly
            # silent mid-utterance, since no new words arrived to
            # transcribe. Approximate, not a real silence-gap measurement;
            # see this module's docstring.
            if (
                self._last_partial_text is not None
                and text == self._last_partial_text
                and not self.repeated_partial.done()
            ):
                self.repeated_partial.set_result(text)

            # Forecast-invalidation signal: a later partial that's grown
            # past a given baseline means speech continued past the point
            # predicted_pause forecast a pause — the caller uses this to
            # cancel a now-stale early retrieval. See wait_for_growth().
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
        past `baseline_text` — i.e. the forecast a pause was starting at
        baseline_text turned out to be wrong. Cancel it if you stop caring
        (e.g. the early retrieval already finished).
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
    forecast_invalidated for predicted_pause's cancel-on-invalidation path.
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
