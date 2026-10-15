"""
_objective_v2.py — the shared v2 cost model. Every priced v2 controller
(ours_v2, b9_v2, b9plus_v2) imports its quality, latency, compute and belief
terms from here and from nowhere else, so the comparison between them differs
ONLY in decision structure. Same rule as v1's _objective.py, extended to the
three channels v2 exists to test (docs/research_v2/CONTRACTS.md §0):

  C1  turn-dependent overlap. A retrieval dispatched at time t has a hiding
      window W_t = R_t + d_eot, where R_t is the PREDICTED remaining speech at
      t (signals["timing"]["predicted_remaining_speech_ms"]) and d_eot is the
      end-of-turn detection delay (config, explicit). With L the retrieval
      latency distribution of the active latency profile:

          hidden(t)  = E_L[min(L, W_t)]
          exposed(t) = E_L[max(0, L - W_t)]

      At the end-of-turn step the window is 0 by definition (generation could
      start now), so exposed = E[L].

  C2  information vs overlap. The belief b_t = P(retrieval needed | prefix
      available at t) is computed from signals["model"]["uncertainty_by_prefix"]
      through the E1-v2 dev calibration. b_t sharpens as speech continues.

  C3  partial-query quality. rho(t) is the fraction of retrieval's gain lost
      by retrieving on the prefix instead of the final transcript; a config
      table fitted on dev, defaulting to scale * (1 - f)^power of the
      estimated word fraction f.

THE CORRECTED OBJECTIVE (retrieval helps where needed and hurts where not):

    U(retrieve at t) = b_t * g+ * (1 - rho(t))
                       - (1 - b_t) * g-
                       - lambda_l * (exposed(t) + pi) [seconds]
                       - lambda_c * (1 + waste(t))
    U(skip)          = 0

pi is the NON-HIDEABLE TTFA increment of using retrieved passages (longer
prompt: extra prefill and generation). v1 measured retrieval adding
1.27-1.57 s of TTFA of which only the 0.3-0.7 s retrieval call can overlap
speech. pi comes from signals["retrieval"]["nonhideable_increment_ms"] or
the config default, and priced_latency_ms() adds it for every priced
controller identically (THEORY.md: X_k = E[pi + max(0, L - R_k)]).

g+ and g- are config (fitted on dev by E1-v2; defaults are the v1 E3
measurement +0.72 / -0.59).

CAUSALITY. Every helper here takes only quantities that a StepObs
(controller/sequential_v2.py) carries — i.e. information available at the
step's available_ms. The word fraction used by rho, waste and the belief
calibration is the CAUSAL estimate f_t = e_t / (e_t + R_t) (elapsed speech
over elapsed-plus-predicted-remaining), never the reference-transcript word
fraction, which encodes the final length and is therefore future
information.

UNITS. latency in ms inside the module, converted to seconds exactly once in
utility_retrieve(); lambda_l is quality per second, as in v1.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective import Prices  # noqa: E402 -- shared price record

_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"

# Timing policies the v2 controllers range over, ordered latest-to-earliest
# conservative first so a tie in any argmax resolves to the more
# conservative policy (same convention as v1's TIMING_ORDER).
POLICIES_V2 = ("sentence_end", "immediate", "fixed_early_proxy", "forecast")

# commit: a dispatched retrieval is always used. speculative: it may be
# discarded at end of turn (THEORY.md §6.2, Prop. 4').
ARCHITECTURES = ("commit", "speculative")

# Mirrors configs/config.yaml's controller.e3v2 block. Kept in sync the same
# way v1's _DEFAULT_E3_CFG mirrors controller.e3, so every v2 controller stays
# unit-testable without the config file.
_DEFAULT_E3V2_CFG: dict = {
    "default_prices": {"lambda_l": 1.0, "lambda_c": 0.05},
    "quality": {"g_plus": 0.72, "g_minus": 0.59},
    "calibration": {
        "from_artifact": None,
        "fallback_artifact": "controller/baselines/e1_calibration_mean_entropy.json",
        "signal": "mean_entropy_nats",
        "frac_source": "predicted",
        "inline": None,
    },
    "belief": {"prior": 0.5, "availability_lag_ms": 0},
    "timing": {
        "eot_detection_ms": 0,
        # expected speech-end -> end-of-turn-decision gap for the hiding
        # window; null = eot_detection_ms (see hiding_window_ms)
        "end_gap_ms": None,
        "default_partial_interval_ms": 300,
        "fixed_early_fraction": 0.5,
        "forecast_lead_quantile": 0.5,
        "query_ready": {"min_words": 4, "require_content_word": True},
    },
    # MEASURED (v2/pipeline; mirrors configs/config.yaml controller.e3v2):
    # bare profiles are lower bounds (no LLM rewrite); rewrite_* are the
    # whole retrieval operation on a live partial.
    "latency_profiles": {
        "faiss": {"p10": 23, "p25": 29, "p50": 41, "p75": 86, "p90": 180, "p95": 247, "p99": 423},
        "faiss_rerank": {"p10": 57, "p25": 61, "p50": 66, "p75": 71, "p90": 75, "p95": 78, "p99": 110},
        "remote_api": {"p10": 262, "p25": 264, "p50": 268, "p75": 272, "p90": 277, "p95": 281, "p99": 339},
        "rewrite_faiss": {"p10": 852, "p25": 919, "p50": 1044, "p75": 1195, "p90": 1379, "p95": 1529, "p99": 1847},
        "rewrite_faiss_rerank": {"p10": 860, "p25": 935, "p50": 1043, "p75": 1165, "p90": 1302, "p95": 1430, "p99": 1765},
        "rewrite_remote": {"p10": 1059, "p25": 1137, "p50": 1241, "p75": 1360, "p90": 1496, "p95": 1621, "p99": 1980},
    },
    "default_latency_profile": "rewrite_faiss",
    "latency_points": 41,
    # MEASURED pi (paired retrieve-minus-skip TTFA minus the exposed call,
    # real endpoint): streaming TTS 371 ms, batch TTS 855 ms. v2 uses
    # streaming TTS, so it is the default.
    "tts_mode": "streaming",
    "nonhideable_increment_ms": 371.0,
    "nonhideable_increment_ms_by_tts_mode": {"streaming": 371.0, "batch": 855.0},
    # Pipeline architecture. commit: a dispatched retrieval's passages are
    # always used. speculative: a dispatched retrieval may be discarded at
    # end of turn (compute is still spent); enables b10_speculative_v2.
    "architecture": "commit",
    "b10": {"timing": "immediate"},
    "rho": {"kind": "power", "scale": 1.0, "power": 2.0, "not_ready": 1.0, "table": None},
    "waste": {"coef": 0.1},
    "ours": {
        "lookahead_steps": 1,
        "belief_transitions_artifact": "controller/baselines/belief_transitions_v2.json",
        "allow_early_skip": False,
        "speculative": False,
    },
    "b9": {"timing_objective": "v1_full"},
    "estimates_artifact": None,
    "policy_constants_fallback": {
        "prior_belief": 0.5,
        "profiles": {},
    },
    "unpriced": {
        "b4": {"theta": 0.3},
        "b5": {"threshold": 0.5, "model": None},
        "b6": {"threshold": 1.0},
        "b8": {"trigger": "b6", "timing": "fixed_early_proxy"},
    },
    "tuning": {"budget": 21, "equal_accuracy_tolerance": 0.02},
    "phase_b": {"query_variant_order_sensitive": True},
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge — the v2 block nests three levels (e.g.
    timing.query_ready.min_words), deeper than v1's one-level merge. Unlike
    v1, an explicit null DOES override (e.g. `from_artifact: null` means
    "no artifact"), so omit a key to inherit the default."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_e3v2_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    """controller.e3v2 from configs/config.yaml, merged over the defaults,
    then over `overrides` (used by tests and the tuning harness), then over
    the estimates artifact if one is configured (tune_v2.py's output)."""
    import yaml

    cfg_file = Path(path) if path else Path(
        os.environ.get("VOICE_RAG_CONFIG") or _DEFAULT_CONFIG_PATH
    )
    try:
        with open(cfg_file, "r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        full = {}
    block = (full.get("controller") or {}).get("e3v2") or {}
    cfg = _deep_merge(_DEFAULT_E3V2_CFG, block)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    # A dev-fitted rho table in the estimates artifact applies to EVERY
    # controller (it is a shared estimate), unless config pins one itself.
    if cfg.get("estimates_artifact") and not (cfg["rho"].get("table")):
        fitted = load_estimates(cfg).get("rho_table")
        if fitted:
            cfg["rho"] = {**cfg["rho"], "table": fitted}
    q = cfg["quality"]
    # g_minus is SIGNED: positive = retrieval harms not-needed items (the v1
    # finding), negative = it helps them too (measured on d1_open ASR, where
    # the closed-book answer to a misrecognised question fails although the
    # text label says "not needed"). The objective stays well defined as long
    # as the retrieve utility increases with the belief: g+ (1-rho) + g- > 0.
    if float(q["g_plus"]) < 0 or float(q["g_plus"]) + float(q["g_minus"]) <= 0:
        raise ValueError("controller.e3v2.quality needs g_plus >= 0 and g_plus + g_minus > 0 "
                         "(the retrieve utility must increase with the belief)")
    if cfg.get("architecture") not in ARCHITECTURES:
        raise ValueError(f"controller.e3v2.architecture must be one of {ARCHITECTURES}, "
                         f"got {cfg.get('architecture')!r}")
    return cfg


def resolve_repo_path(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    path = Path(p)
    return path if path.is_absolute() else _REPO_ROOT / path


def load_estimates(cfg: dict) -> dict:
    """tune_v2.py's artifact (policy constants, tuned unpriced thresholds,
    fitted rho table), or {} when none is configured."""
    path = resolve_repo_path(cfg.get("estimates_artifact"))
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_prices_v2(prices: Optional[dict], cfg: dict) -> Prices:
    defaults = cfg["default_prices"]
    prices = prices or {}
    lam_l = prices.get("lambda_l")
    lam_c = prices.get("lambda_c")
    return Prices(
        lambda_l=float(defaults["lambda_l"] if lam_l is None else lam_l),
        lambda_c=float(defaults["lambda_c"] if lam_c is None else lam_c),
    )


@dataclass(frozen=True)
class QualityModel:
    """Asymmetric retrieval effect: +g_plus on items that need retrieval,
    -g_minus on items that do not (v1 E3 measured +0.72 / -0.59)."""

    g_plus: float
    g_minus: float


def quality_model(cfg: dict) -> QualityModel:
    q = cfg["quality"]
    return QualityModel(g_plus=float(q["g_plus"]), g_minus=float(q["g_minus"]))


# ---------------------------------------------------------------------------
# Belief calibration (E1-v2 dev artifact, prefix-aware)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeliefCalibration:
    """
    P(retrieval needed | uncertainty signal x on the prefix, word fraction f).

        logistic:  sigmoid(intercept + slope*x + frac_slope*f + interaction*x*f)
        identity:  clip01(x)   (synthetic tests only; calibrated=False)

    frac_slope/interaction let E1-v2 fit a prefix-aware calibration (an
    entropy of 0.4 nats on 30% of the question does not mean what it means on
    the full question). A v1 artifact has neither term, which reduces to the
    v1 full-transcript calibration.
    """

    signal: str
    kind: str
    intercept: float
    slope: float
    frac_slope: float
    interaction: float
    calibrated: bool
    source: str

    def apply(self, x: float, frac: float = 1.0) -> float:
        if self.kind == "identity":
            return min(1.0, max(0.0, float(x)))
        if self.kind == "logistic":
            f = min(1.0, max(0.0, float(frac)))
            z = self.intercept + self.slope * x + self.frac_slope * f + self.interaction * x * f
            z = max(-40.0, min(40.0, z))
            return 1.0 / (1.0 + math.exp(-z))
        raise ValueError(f"unknown belief calibration kind {self.kind!r}")


def load_belief_calibration(cfg: dict) -> BeliefCalibration:
    """
    Resolution: calibration.inline (tests) -> calibration.from_artifact (the
    E1-v2 dev fit) -> calibration.fallback_artifact (v1's full-transcript fit,
    no prefix terms). The requested signal is cross-checked against the
    artifact's, and a calibrated logistic must have slope > 0 — both guards
    carried over from v1's load_voi_calibration() for the same reasons.
    """
    block = dict(cfg.get("calibration") or {})
    requested = block.get("signal")
    fitted: Optional[dict] = None
    source_note = ""
    if block.get("inline"):
        fitted = dict(block["inline"])
        source_note = "inline"
    else:
        for key in ("from_artifact", "fallback_artifact"):
            path = resolve_repo_path(block.get(key))
            if path is None:
                continue
            with open(path, "r", encoding="utf-8") as fh:
                fitted = (json.load(fh) or {}).get("calibration")
            if not fitted:
                raise ValueError(f"{path} has no top-level 'calibration' object")
            source_note = key
            break
    if fitted is None:
        raise ValueError("controller.e3v2.calibration resolves to no calibration")
    fitted_signal = fitted.get("signal")
    if requested and fitted_signal and requested != fitted_signal:
        raise ValueError(
            f"e3v2 calibration mismatch: config asks for {requested!r}, the "
            f"calibration was fitted on {fitted_signal!r}"
        )
    cal = BeliefCalibration(
        signal=str(fitted_signal or requested or "mean_entropy_nats"),
        kind=str(fitted.get("kind", "logistic")),
        intercept=float(fitted.get("intercept", 0.0)),
        slope=float(fitted.get("slope", 1.0)),
        frac_slope=float(fitted.get("frac_slope", 0.0)),
        interaction=float(fitted.get("interaction", 0.0)),
        calibrated=bool(fitted.get("calibrated", False)),
        source=f"{source_note}:{fitted.get('source', '(unspecified)')}",
    )
    if cal.calibrated and cal.kind == "logistic" and cal.slope <= 0:
        raise ValueError(
            f"e3v2 belief calibration slope {cal.slope:+.4f} <= 0 inverts the "
            f"uncertainty -> need direction; refusing to load ({cal.source})"
        )
    return cal


# ---------------------------------------------------------------------------
# Latency distribution (C1)
# ---------------------------------------------------------------------------
def parse_quantiles(qdict: dict) -> List[Tuple[float, float]]:
    """Accepts keys 'p50', 'q50', '50', '0.5', 50, 0.5 -> sorted [(0.5, ms)]."""
    out = []
    for key, value in (qdict or {}).items():
        k = str(key).strip().lower()
        if k in ("mean", "expected"):
            continue
        if k[:1] in ("p", "q"):
            k = k[1:]
        try:
            q = float(k)
        except ValueError:
            continue
        if q > 1.0:
            q /= 100.0
        if not 0.0 < q < 1.0:
            continue
        out.append((q, float(value)))
    out.sort()
    return out


@dataclass(frozen=True)
class LatencyDist:
    """
    The retrieval-latency distribution of one latency profile, as M
    equal-weight points of its quantile function (piecewise-linear between
    the given quantiles, flat beyond the extreme ones — a documented
    approximation that slightly understates the tails).
    """

    profile: str
    points_ms: Tuple[float, ...]
    #: pi, the NON-HIDEABLE TTFA increment of a used retrieval (longer
    #: prompt -> extra prefill, longer generation). It is paid after the end
    #: of turn whatever the dispatch time, so no timing can hide it. See
    #: priced_latency_ms().
    nonhideable_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return sum(self.points_ms) / len(self.points_ms)

    def expected_hidden_ms(self, window_ms: float) -> float:
        w = max(0.0, float(window_ms))
        return sum(min(l, w) for l in self.points_ms) / len(self.points_ms)

    def expected_exposed_ms(self, window_ms: float) -> float:
        w = max(0.0, float(window_ms))
        return sum(max(0.0, l - w) for l in self.points_ms) / len(self.points_ms)

    def quantile(self, q: float) -> float:
        pts = sorted(self.points_ms)
        idx = min(len(pts) - 1, max(0, int(round(q * len(pts) - 0.5))))
        return pts[idx]


def latency_dist_from_quantiles(
    qdict: dict, n_points: int = 41, profile: str = "?", expected_ms: Optional[float] = None
) -> LatencyDist:
    qs = parse_quantiles(qdict)
    if not qs:
        if expected_ms is None:
            raise ValueError(f"latency profile {profile!r} has no quantiles and no expected value")
        return LatencyDist(profile=profile, points_ms=(float(expected_ms),))
    if len(qs) == 1:
        return LatencyDist(profile=profile, points_ms=(qs[0][1],))
    probs = [q for q, _ in qs]
    vals = [v for _, v in qs]
    pts = []
    for i in range(n_points):
        u = (i + 0.5) / n_points
        if u <= probs[0]:
            pts.append(vals[0])
        elif u >= probs[-1]:
            pts.append(vals[-1])
        else:
            j = bisect.bisect_right(probs, u)
            q0, q1, v0, v1 = probs[j - 1], probs[j], vals[j - 1], vals[j]
            pts.append(v0 + (v1 - v0) * (u - q0) / (q1 - q0))
    return LatencyDist(profile=profile, points_ms=tuple(pts))


def latency_dist_for(retrieval_block: Optional[dict], cfg: dict) -> LatencyDist:
    """signals["retrieval"] -> LatencyDist. Quantiles in the signal win; the
    profile's config quantiles are the fallback; a missing block means the
    configured default profile."""
    block = retrieval_block or {}
    profile = block.get("latency_profile") or cfg["default_latency_profile"]
    qdict = block.get("latency_quantiles_ms") or cfg["latency_profiles"].get(profile) or {}
    dist = latency_dist_from_quantiles(
        qdict, int(cfg["latency_points"]), profile, block.get("expected_latency_ms")
    )
    return LatencyDist(profile=dist.profile, points_ms=dist.points_ms,
                       nonhideable_ms=nonhideable_ms(block, cfg))


def nonhideable_ms(retrieval_block: Optional[dict], cfg: dict) -> float:
    """pi for this turn: signals["retrieval"]["nonhideable_increment_ms"] if
    present, else controller.e3v2.nonhideable_increment_ms (a PLACEHOLDER
    from the v1 measurement — see the config comment)."""
    value = (retrieval_block or {}).get("nonhideable_increment_ms")
    if value is None:
        value = cfg.get("nonhideable_increment_ms", 0.0)
    return max(0.0, float(value))


def priced_latency_ms(lat: LatencyDist, retrieval_exposed_ms: float) -> float:
    """
    THE latency every priced controller charges for a retrieval whose
    passages are USED: the exposed part of the retrieval call plus the
    non-hideable increment pi,

        X = E[max(0, L - W)] + pi.

    Ours-v2 passes its per-step exposure, B9-v2 the full E[L], B9+ its
    policy's dev-mean exposure, B10 the realised exposure of its speculative
    dispatch — but pi is added here, once, for all of them, so it shifts
    every controller's thresholds identically and cannot be a confound.
    """
    return float(retrieval_exposed_ms) + lat.nonhideable_ms


def hiding_window_ms(remaining_ms: Optional[float], is_final: bool, cfg: dict) -> float:
    """
    W_t: how long a retrieval dispatched now has before generation could
    start. At a partial: predicted remaining speech + the expected gap from
    the end of speech to the end-of-turn decision. At the end-of-turn step: 0.
    With no remaining-time prediction the window is taken as 0 (nothing is
    credited as hidden that the system cannot predict) — conservative, and
    the same for every controller.

    The gap is timing.end_gap_ms when set (a dev-estimated shared constant:
    on real ASR traces the VAD hang plus the final transcript's ASR time,
    since the end-of-turn step waits for both), else eot_detection_ms (the
    delay build_steps adds after the VAD end, the same quantity when the
    VAD end is the end of speech).
    """
    if is_final or remaining_ms is None:
        return 0.0
    gap = cfg["timing"].get("end_gap_ms")
    if gap is None:
        gap = cfg["timing"]["eot_detection_ms"]
    return max(0.0, float(remaining_ms)) + float(gap)


def expected_exposed_ms(lat: LatencyDist, remaining_ms: Optional[float], is_final: bool, cfg: dict) -> float:
    return lat.expected_exposed_ms(hiding_window_ms(remaining_ms, is_final, cfg))


def expected_hidden_ms(lat: LatencyDist, remaining_ms: Optional[float], is_final: bool, cfg: dict) -> float:
    return lat.expected_hidden_ms(hiding_window_ms(remaining_ms, is_final, cfg))


# ---------------------------------------------------------------------------
# Word fraction, rho (C3), waste
# ---------------------------------------------------------------------------
def estimated_frac(speech_elapsed_ms: float, remaining_ms: Optional[float], is_final: bool) -> Optional[float]:
    """Causal word-fraction estimate e/(e+R). None if R is unpredicted."""
    if is_final:
        return 1.0
    if remaining_ms is None:
        return None
    e = max(0.0, float(speech_elapsed_ms))
    r = max(0.0, float(remaining_ms))
    if e + r <= 0:
        return 0.0
    return e / (e + r)


def rho_at(frac: Optional[float], query_ready: bool, cfg: dict) -> float:
    """Fraction of retrieval's gain lost by retrieving on this prefix."""
    rcfg = cfg["rho"]
    if frac is not None and frac >= 1.0:
        return 0.0
    if not query_ready:
        return float(rcfg["not_ready"])
    if frac is None:
        # Unknown progress: treat as the start of the turn (most pessimistic
        # estimate a ready query can get).
        frac = 0.0
    table = rcfg.get("table")
    if table:
        xs = [float(p[0]) for p in table]
        ys = [float(p[1]) for p in table]
        if frac <= xs[0]:
            return min(1.0, max(0.0, ys[0]))
        if frac >= xs[-1]:
            return min(1.0, max(0.0, ys[-1]))
        j = bisect.bisect_right(xs, frac)
        x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
        return min(1.0, max(0.0, y0 + (y1 - y0) * (frac - x0) / (x1 - x0)))
    scale = float(rcfg.get("scale", 1.0))
    power = float(rcfg.get("power", 2.0))
    return min(1.0, max(0.0, scale * (1.0 - frac) ** power))


def waste_at(frac: Optional[float], is_final: bool, cfg: dict) -> float:
    """
    Expected extra compute (retrieval units) of committing a retrieval now:
    the chance a later revision of the partial forces a relaunch. PLACEHOLDER
    model coef*(1-f), 0 at end of turn — same status as v1's waste_rate, and
    to be replaced by a measured revision rate from the v2 speech traces.
    """
    if is_final:
        return 0.0
    f = 0.0 if frac is None else frac
    return max(0.0, float(cfg["waste"]["coef"]) * (1.0 - f))


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------
def utility_retrieve(
    belief: float,
    rho: float,
    exposed_ms: float,
    waste: float,
    prices: Prices,
    quality: QualityModel,
) -> float:
    """U(retrieve) = b g+ (1-rho) - (1-b) g- - lambda_l exposed_s - lambda_c (1+waste)."""
    b = min(1.0, max(0.0, float(belief)))
    return (
        b * quality.g_plus * (1.0 - rho)
        - (1.0 - b) * quality.g_minus
        - prices.lambda_l * (float(exposed_ms) / 1000.0)
        - prices.lambda_c * (1.0 + waste)
    )


def speculative_use_utility(belief_end: float, rho_dispatch: float, lat: "LatencyDist",
                            realised_window_ms: float, prices: Prices, quality: QualityModel) -> float:
    """
    Speculative architecture, end of turn: the value of USING passages that
    were dispatched realised_window_ms before generation could start, judged
    with the end-of-turn belief b_K (compute is sunk and NOT included):

        A(b_K) - lambda_l X = b_K g+ (1 - rho_d) - (1 - b_K) g-
                              - lambda_l (E[max(0, L - window)] + pi)

    Discarding is worth 0 (no quality change, no pi). THEORY.md Prop. 4'.
    """
    cost = priced_latency_ms(lat, lat.expected_exposed_ms(max(0.0, float(realised_window_ms))))
    b = min(1.0, max(0.0, float(belief_end)))
    return (b * quality.g_plus * (1.0 - rho_dispatch) - (1.0 - b) * quality.g_minus
            - prices.lambda_l * cost / 1000.0)


def belief_cut(rho: float, priced_latency_ms: float, waste: float, prices: Prices, quality: QualityModel) -> float:
    """
    The belief above which retrieving beats skipping:

        b > (cost + g-) / (g+ (1-rho) + g-),  cost = lambda_l*lat_s + lambda_c*(1+w)

    +inf when the denominator vanishes (no belief can justify retrieval).
    """
    cost = prices.lambda_l * (priced_latency_ms / 1000.0) + prices.lambda_c * (1.0 + waste)
    denom = quality.g_plus * (1.0 - rho) + quality.g_minus
    if denom <= 0:
        return float("inf")
    return (cost + quality.g_minus) / denom


@dataclass(frozen=True)
class DivergenceBandV2:
    """
    The belief interval on which a joint controller and a decomposed rival
    make different retrieve/skip calls at one dispatch point — v1's band,
    generalised to asymmetric quality, rho and per-turn exposure. Either cut
    may be the lower one; contains() is the half-open interval between them.
    DIAGNOSTIC ONLY — never feeds a decision (same rule as v1).
    """

    ours_cut: float
    rival_cut: float
    ours_priced_ms: float
    rival_priced_ms: float

    @property
    def width(self) -> float:
        return self.rival_cut - self.ours_cut

    def contains(self, belief: float) -> bool:
        lo, hi = sorted((self.ours_cut, self.rival_cut))
        return lo < belief <= hi

    def describe(self) -> str:
        return (f"band=({self.ours_cut:.3f},{self.rival_cut:.3f}] "
                f"priced {self.ours_priced_ms:.0f}ms vs {self.rival_priced_ms:.0f}ms")


def divergence_band_v2(
    prices: Prices,
    quality: QualityModel,
    ours_exposed_ms: float,
    rival_priced_ms: float,
    rho: float = 0.0,
    waste_ours: float = 0.0,
    waste_rival: float = 0.0,
) -> DivergenceBandV2:
    return DivergenceBandV2(
        ours_cut=belief_cut(rho, ours_exposed_ms, waste_ours, prices, quality),
        rival_cut=belief_cut(rho, rival_priced_ms, waste_rival, prices, quality),
        ours_priced_ms=ours_exposed_ms,
        rival_priced_ms=rival_priced_ms,
    )


# ---------------------------------------------------------------------------
# Step-level terms — the single place every priced controller reads them from
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StepTerms:
    belief: float
    frac: Optional[float]
    rho: float
    waste: float
    exposed_ms: float   # exposed part of the retrieval call, E[max(0, L - W)]
    hidden_ms: float
    belief_source: str
    cost_ms: float = 0.0  # priced latency: exposed_ms + pi (priced_latency_ms)


def belief_from_obs(obs, cfg: dict, calibration: BeliefCalibration, prior: Optional[float] = None) -> Tuple[float, str]:
    """
    Calibrated belief from the step's latest available uncertainty entry. A
    pre-calibrated 'p_needed' on the entry wins (synthetic data only); else
    the calibrated signal; else the prior (no information yet).
    """
    entry = obs.belief_entry
    prior = float(cfg["belief"]["prior"] if prior is None else prior)
    if not entry:
        return prior, "prior"
    if entry.get("p_needed") is not None:
        return min(1.0, max(0.0, float(entry["p_needed"]))), obs.belief_source
    x = entry.get(calibration.signal)
    if x is None:
        return prior, "prior(signal-missing)"
    if cfg["calibration"].get("frac_source") == "entry" and entry.get("word_frac") is not None:
        frac = float(entry["word_frac"])
    else:
        frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
        frac = 1.0 if frac is None else frac
    return calibration.apply(float(x), frac), obs.belief_source


def step_terms(obs, cfg: dict, calibration: BeliefCalibration, lat: LatencyDist, prior: Optional[float] = None) -> StepTerms:
    belief, source = belief_from_obs(obs, cfg, calibration, prior)
    frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
    exposed = expected_exposed_ms(lat, obs.remaining_ms, obs.is_final, cfg)
    return StepTerms(
        belief=belief,
        frac=frac,
        rho=rho_at(frac, obs.query_ready, cfg),
        waste=waste_at(frac, obs.is_final, cfg),
        exposed_ms=exposed,
        hidden_ms=expected_hidden_ms(lat, obs.remaining_ms, obs.is_final, cfg),
        belief_source=source,
        cost_ms=priced_latency_ms(lat, exposed),
    )


# ---------------------------------------------------------------------------
# Timing policy triggers (causal: functions of one StepObs)
# ---------------------------------------------------------------------------
def policy_fires(policy: str, obs, cfg: dict, lat: LatencyDist) -> bool:
    """
    Whether `policy` dispatches at this step. Every policy fires at the end
    of turn at the latest.

      sentence_end       only at the end-of-turn step
      immediate          first query-ready partial
      fixed_early_proxy  first query-ready partial with estimated word
                         fraction >= timing.fixed_early_fraction
      forecast           first query-ready partial whose hiding window
                         (R_t + d_eot) is <= the latency profile's
                         timing.forecast_lead_quantile quantile — i.e. wait
                         for information until retrieval would only just be
                         hidden, then go
    """
    if obs.is_final:
        return True
    if policy == "sentence_end":
        return False
    if not obs.query_ready:
        return False
    if policy == "immediate":
        return True
    if policy == "fixed_early_proxy":
        frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
        return frac is not None and frac >= float(cfg["timing"]["fixed_early_fraction"])
    if policy == "forecast":
        if obs.remaining_ms is None:
            return False
        lead = lat.quantile(float(cfg["timing"]["forecast_lead_quantile"]))
        return hiding_window_ms(obs.remaining_ms, False, cfg) <= lead
    raise ValueError(f"unknown v2 timing policy {policy!r} (expected one of {POLICIES_V2})")


# ---------------------------------------------------------------------------
# Policy constants (dev estimates used by B9 / B9+ / B8)
# ---------------------------------------------------------------------------
def estimate_policy_constants(turn_ctxs: Iterable, cfg: dict, policies: Sequence[str] = POLICIES_V2) -> dict:
    """
    Per (latency profile, policy): the mean REALISED hidden and exposed
    retrieval latency, mean rho and waste at the policy's trigger, over the
    given (dev) turns. Realised = computed with the TRUE hiding window
    end_of_turn_ms - dispatch_ms, not the predictor's, so the constants are
    the honest average a fixed policy achieves. Plus the mean end-of-turn
    belief (the prior B9's timing sub-optimizer prices risk with).
    """
    from controller.sequential_v2 import build_steps, end_of_turn_ms  # lazy: avoids an import cycle

    calibration = load_belief_calibration(cfg)
    acc: Dict[str, Dict[str, Dict[str, float]]] = {}
    beliefs: List[float] = []
    for ctx in turn_ctxs:
        lat = latency_dist_for((ctx.signals or {}).get("retrieval"), cfg)
        steps = build_steps(ctx, cfg)
        end_ms = end_of_turn_ms(ctx, cfg)
        final_belief, _ = belief_from_obs(steps[-1], cfg, calibration)
        beliefs.append(final_belief)
        prof = acc.setdefault(lat.profile, {})
        for policy in policies:
            obs = next(o for o in steps if policy_fires(policy, o, cfg, lat))
            window = 0.0 if obs.is_final else max(0.0, end_ms - obs.available_ms)
            frac = estimated_frac(obs.speech_elapsed_ms, obs.remaining_ms, obs.is_final)
            row = prof.setdefault(policy, {"hidden_ms": 0.0, "exposed_ms": 0.0, "rho": 0.0,
                                           "waste": 0.0, "early_rate": 0.0, "n": 0,
                                           "remaining_ms": 0.0, "elapsed_ms": 0.0})
            row["hidden_ms"] += lat.expected_hidden_ms(window)
            row["exposed_ms"] += lat.expected_exposed_ms(window)
            row["rho"] += rho_at(frac, obs.query_ready, cfg)
            row["waste"] += waste_at(frac, obs.is_final, cfg)
            row["early_rate"] += 0.0 if obs.is_final else 1.0
            # predicted remaining speech / elapsed speech at the trigger: the
            # scalars SPEC-theta (b11) gates on instead of this turn's values
            row["remaining_ms"] += 0.0 if (obs.is_final or obs.remaining_ms is None) else float(obs.remaining_ms)
            row["elapsed_ms"] += float(obs.speech_elapsed_ms or 0.0)
            row["n"] += 1
    for prof in acc.values():
        for row in prof.values():
            n = max(1, row["n"])
            for key in ("hidden_ms", "exposed_ms", "rho", "waste", "early_rate", "remaining_ms", "elapsed_ms"):
                row[key] /= n
    return {
        "prior_belief": (sum(beliefs) / len(beliefs)) if beliefs else float(cfg["belief"]["prior"]),
        "profiles": acc,
        "n_turns": len(beliefs),
    }


def resolve_policy_constants(cfg: dict, explicit: Optional[dict] = None) -> Tuple[dict, str]:
    """explicit -> estimates artifact's policy_constants -> config fallback."""
    if explicit:
        return explicit, "explicit"
    est = load_estimates(cfg)
    if est.get("policy_constants"):
        return est["policy_constants"], "estimates_artifact"
    return cfg["policy_constants_fallback"], "UNFITTED-fallback"


# ---------------------------------------------------------------------------
# Belief transitions (C2) — the lookahead model Ours-v2 plans with
# ---------------------------------------------------------------------------
class BeliefTransitions:
    """
    Empirical conditional distribution of the next calibrated belief given the
    current one, binned by belief and by phase (estimated word fraction).

      next[phase][bin]  -> (probs over to-bins, representative values)
      final[phase][bin] -> same, for the jump to the end-of-turn belief

    Missing cells fall back to the identity (the belief does not move) —
    which is exactly A3, so an unfitted model makes waiting worthless rather
    than inventing information.
    """

    def __init__(self, bin_edges: Sequence[float], phase_edges: Sequence[float],
                 next_table: dict, final_table: dict, provenance: Optional[dict] = None):
        self.bin_edges = [float(x) for x in bin_edges]
        self.phase_edges = [float(x) for x in phase_edges]
        self.next_table = next_table
        self.final_table = final_table
        self.provenance = provenance or {}

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges) - 1

    def bin_of(self, b: float) -> int:
        j = bisect.bisect_right(self.bin_edges, b) - 1
        return min(self.n_bins - 1, max(0, j))

    def phase_of(self, frac: Optional[float]) -> int:
        f = 0.0 if frac is None else frac
        j = bisect.bisect_right(self.phase_edges, f) - 1
        return min(len(self.phase_edges) - 2, max(0, j))

    def _dist(self, table: dict, b: float, frac: Optional[float]) -> List[Tuple[float, float]]:
        cell = (table.get(str(self.phase_of(frac))) or {}).get(str(self.bin_of(b)))
        if not cell:
            return [(1.0, b)]
        out = [(float(p), float(v)) for p, v in zip(cell["probs"], cell["values"]) if p > 0]
        return out or [(1.0, b)]

    def next_dist(self, b: float, frac: Optional[float]) -> List[Tuple[float, float]]:
        return self._dist(self.next_table, b, frac)

    def final_dist(self, b: float, frac: Optional[float]) -> List[Tuple[float, float]]:
        return self._dist(self.final_table, b, frac)

    @classmethod
    def identity(cls, n_bins: int = 10) -> "BeliefTransitions":
        return cls([i / n_bins for i in range(n_bins + 1)], [0.0, 1.0], {}, {},
                   {"kind": "identity", "note": "no fitted transitions: belief assumed constant (A3)"})

    def to_json(self) -> dict:
        return {"bin_edges": self.bin_edges, "phase_edges": self.phase_edges,
                "next": self.next_table, "final": self.final_table, "provenance": self.provenance}

    @classmethod
    def from_json(cls, blob: dict) -> "BeliefTransitions":
        return cls(blob["bin_edges"], blob["phase_edges"], blob.get("next") or {},
                   blob.get("final") or {}, blob.get("provenance") or {})


def fit_belief_transitions(
    sequences: Iterable[Sequence[Tuple[float, float]]],
    n_bins: int = 10,
    phase_edges: Sequence[float] = (0.0, 0.34, 0.67, 1.0),
    alpha: float = 1.0,
    provenance: Optional[dict] = None,
) -> BeliefTransitions:
    """
    Fit from belief sequences [(frac_k, b_k), ...] ordered by time, each
    ending at the end-of-turn belief. `alpha` pseudo-counts go to the
    identity cell (stay in the same bin with the current value), shrinking
    sparse rows toward "no information" rather than toward uniform noise.
    """
    edges = [i / n_bins for i in range(n_bins + 1)]
    probe = BeliefTransitions(edges, phase_edges, {}, {})
    counts = {"next": {}, "final": {}}
    sums = {"next": {}, "final": {}}

    def add(kind, phase, frm, to, val):
        c = counts[kind].setdefault(phase, {}).setdefault(frm, [0.0] * n_bins)
        s = sums[kind].setdefault(phase, {}).setdefault(frm, [0.0] * n_bins)
        c[to] += 1.0
        s[to] += val

    for seq in sequences:
        seq = list(seq)
        if len(seq) < 2:
            continue
        b_final = seq[-1][1]
        for k in range(len(seq) - 1):
            frac, b = seq[k]
            ph, frm = probe.phase_of(frac), probe.bin_of(b)
            nxt = seq[k + 1][1]
            add("next", ph, frm, probe.bin_of(nxt), nxt)
            add("final", ph, frm, probe.bin_of(b_final), b_final)

    tables = {}
    for kind in ("next", "final"):
        tables[kind] = {}
        for ph in range(len(phase_edges) - 1):
            for frm in range(n_bins):
                c = list(counts[kind].get(ph, {}).get(frm, [0.0] * n_bins))
                s = list(sums[kind].get(ph, {}).get(frm, [0.0] * n_bins))
                if sum(c) <= 0:
                    # No data: leave the cell out so lookups fall back to the
                    # exact identity (belief unchanged), not a bin center.
                    continue
                center = (edges[frm] + edges[frm + 1]) / 2.0
                values = [(s[i] / c[i]) if c[i] > 0 else (edges[i] + edges[i + 1]) / 2.0
                          for i in range(n_bins)]
                if alpha > 0:
                    # identity pseudo-count, valued at the bin's own mean so far
                    own = values[frm] if c[frm] > 0 else center
                    s[frm] += alpha * own
                    c[frm] += alpha
                    values[frm] = s[frm] / c[frm]
                total = sum(c)
                tables[kind].setdefault(str(ph), {})[str(frm)] = {
                    "probs": [x / total for x in c],
                    "values": values,
                    "n": int(round(total - alpha)),
                }
    return BeliefTransitions(edges, list(phase_edges), tables["next"], tables["final"],
                             provenance or {"kind": "fitted"})


def load_belief_transitions(cfg: dict) -> BeliefTransitions:
    path = resolve_repo_path(cfg["ours"].get("belief_transitions_artifact"))
    if path is None or not path.exists():
        return BeliefTransitions.identity()
    with open(path, "r", encoding="utf-8") as fh:
        return BeliefTransitions.from_json(json.load(fh))
