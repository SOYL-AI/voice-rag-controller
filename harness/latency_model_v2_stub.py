"""
latency_model_v2_stub.py — a SIMPLE STAND-IN for harness/latency_model_v2.py,
which is being written on another branch. simulator/e3v2_decide.py imports
the real module when it exists and falls back to this one otherwise; every
record produced through the stub carries ttfa_components.stub = true.

Interface (the one the real module must keep):

    LatencyModel.sample_ttfa(path, dispatch_ms, user_end_ms, profile,
                             n_prompt_tokens, rng) -> dict
      path            "skip" | "retrieve"
      dispatch_ms     retrieval dispatch time (speech-onset clock), None on skip
      user_end_ms     VAD end of the user's speech
      profile         latency profile name ("faiss" | "faiss_rerank" | "remote_api")
      n_prompt_tokens prompt length of the generation that answers
      rng             random.Random — the caller seeds it per (item, seed,
                      condition, profile) so controllers that make the same
                      decision get the same draw (common random numbers)
    returns at least {"ttfa_ms", "exposed_retrieval_ms"}

Stub model:
    ready      = user_end + eot_detection_ms
    retrieval  L ~ the profile's quantile function (controller.e3v2.latency_profiles)
    gen_start  = max(ready, dispatch + L)        (retrieve)   | ready (skip)
    ttft       = base_ttft_ms + prefill_ms_per_token * n_prompt_tokens
    ttfa       = (gen_start - user_end) + ttft + tts_first_audio_ms
All constants are placeholders, not measurements.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import latency_dist_from_quantiles, load_e3v2_config  # noqa: E402


class LatencyModel:
    BASE_TTFT_MS = 180.0
    PREFILL_MS_PER_TOKEN = 0.15
    TTS_FIRST_AUDIO_MS = 250.0

    def __init__(self, cfg: Optional[dict] = None, profiles: Optional[dict] = None):
        self.cfg = cfg if cfg is not None else load_e3v2_config()
        self.profiles = dict(self.cfg["latency_profiles"])
        self.profiles.update(profiles or {})
        self._dists = {}

    def _dist(self, profile: str):
        if profile not in self._dists:
            if profile not in self.profiles:
                raise KeyError(f"unknown latency profile {profile!r}")
            self._dists[profile] = latency_dist_from_quantiles(
                self.profiles[profile], int(self.cfg["latency_points"]), profile)
        return self._dists[profile]

    def sample_ttfa(self, path, dispatch_ms, user_end_ms, profile, n_prompt_tokens, rng) -> dict:
        eot = float(self.cfg["timing"]["eot_detection_ms"])
        ready = float(user_end_ms) + eot
        # Draw L unconditionally so the rng stream is identical whatever the
        # path — keeps common random numbers aligned across controllers.
        latency = float(rng.choice(self._dist(profile).points_ms))
        if path == "retrieve":
            if dispatch_ms is None:
                raise ValueError("retrieve path needs dispatch_ms")
            done = float(dispatch_ms) + latency
            gen_start = max(ready, done)
            exposed = gen_start - ready
        else:
            latency, gen_start, exposed = 0.0, ready, 0.0
        ttft = self.BASE_TTFT_MS + self.PREFILL_MS_PER_TOKEN * float(n_prompt_tokens or 0)
        return {
            "ttfa_ms": (gen_start - float(user_end_ms)) + ttft + self.TTS_FIRST_AUDIO_MS,
            "retrieval_ms": latency,
            "exposed_retrieval_ms": exposed,
            "eot_ms": eot,
            "ttft_ms": ttft,
            "tts_first_audio_ms": self.TTS_FIRST_AUDIO_MS,
            "stub": True,
        }
