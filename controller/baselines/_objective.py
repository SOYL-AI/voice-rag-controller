"""
_objective.py — the scalarization objective, the value-of-information
estimate, and the latency/compute cost model that Experiment 3's
price-aware baselines (Ours, B9) share.

WHY THIS MODULE EXISTS — read this before changing anything in it.

E3's headline claim is that a UNIFIED controller (Ours) beats a
DECOMPOSED-but-price-coordinated one (B9). That claim is only meaningful
if the two differ in exactly one respect: whether the retrieve decision
and the timing decision are reasoned about jointly or separately. If Ours
also had a better uncertainty estimate, or a different objective, or
different latency constants, any measured gap would be uninterpretable —
it could be the better estimator rather than the joint reasoning.

So every ingredient that is NOT the joint-vs-decomposed distinction lives
here, in one module, and both baselines import it:

  - the objective itself            (scalarized_utility)
  - the belief -> quality estimate  (voi_estimate)
  - what retrieval costs in latency (retrieval_latency_s)
  - what timing hides / risks       (overlap_s, partial_query_risk,
                                     waste_rate)
  - where the two DISAGREE          (divergence_band) — diagnostic only,
                                     see that function; it is logged by
                                     both controllers and decides nothing

controller/baselines/ours.py and b9_decomposed.py then differ ONLY in how
they COMPOSE these — one argmax over joint (retrieve, timing) candidates
vs. two independent sub-optimizers. That is the whole experiment. See
each file's module docstring for its side of the contrast.

B8 (b8_independent.py) deliberately does NOT import this module's pricing
helpers: it's the no-price-coupling rival, so it must not read
ctx.prices at all. It reuses only voi_estimate's underlying signal
extraction, which is a belief question, not a pricing one.

---------------------------------------------------------------------------
THE OBJECTIVE
---------------------------------------------------------------------------
Per E3_BASELINE_INTERFACE.md ("The scalarization objective (B9 and Ours)")
and the handbook, every policy is scored by

    U = quality  -  lambda_l * latency  -  lambda_c * compute

UNITS MATTER, and they are the one thing easy to get silently wrong:

  quality  — dimensionless, on [0, 1]. voi_estimate()'s scale: the
             expected quality GAIN from retrieving, where 1.0 means
             "retrieval turns a certainly-wrong answer into a certainly-
             right one" and 0.0 means "retrieval adds nothing".
  latency  — SECONDS, not milliseconds. So lambda_l reads as "quality
             lost per second of added latency", which is a number a human
             can reason about and a price grid can sweep over a sane
             range (roughly 0.05 .. 5.0). Priced in milliseconds instead,
             every plausible lambda_l would round to visual zero on a
             frontier plot and the whole sweep would look degenerate. The
             measured constants below are stored in ms (that's how the
             event log records them) and converted here, once.
  compute  — retrieval units. One retrieval = 1.0. A cancelled/wasted
             early launch adds a fraction (see waste_rate).

---------------------------------------------------------------------------
WHERE THE CONSTANTS COME FROM
---------------------------------------------------------------------------
The defaults in configs/config.yaml's controller.e3 section (mirrored in
_DEFAULT_E3_CFG below, same pattern controller/decide.py and
controller/timing.py already use) are MEASURED from this repo's own
existing logs/ — 751 completed retrievals across E2's three timing
policies — not guessed:

    retrieval duration          median 672 ms   (p90 3187 ms)
    overlap, sentence_end         0 ms          (0 by construction)
    overlap, immediate           mean 50 ms     (median 0, max 1091)
    overlap, fixed_early_proxy   mean 188 ms    (median 156, max 1428)
    partial_query_risk, all three policies       0.0  (measured, see below)

Reproduce the latency constants with a scan of logs/*.jsonl over
retrieval_launched -> retrieval_result pairs and retrieval_result's
overlap_with_speech_ms (controller/timing.py's mechanism_result_fields
writes that field). Reproduce the risk constant with
harness/metrics.py's hit_at_k_report(log_dir="logs", k=3).

ONE constant remains unmeasured and is still marked a placeholder in
config: waste_rate. See its docstring below for what would ground it. It
is the one honest weak point left in this model and should be named as
such in the paper rather than presented as measured.

---------------------------------------------------------------------------
ONLY ONE COUPLING CHANNEL IS ACTIVE — read this before interpreting E3
---------------------------------------------------------------------------
The objective has two places where the timing decision can change the
value of the retrieve decision:

  LATENCY channel  exposed_latency_s(t) = retrieval_latency - overlap(t)
                   ACTIVE. Measured overlap spans 0 -> 188 ms.
  QUALITY channel  voi * (1 - partial_query_risk(t))
                   EMPIRICALLY INACTIVE. Measured risk is 0 for every
                   policy, so this factor is identically 1.

That is a measurement, not a modeling choice, and it is a clean result
rather than a limitation: it means E3's Ours-vs-B9 comparison ISOLATES the
latency channel. Whatever gap the sweep measures cannot be attributed to a
quality effect, because there is no quality effect to attribute it to.

It also makes the TIMING sub-decision nearly trivial — with risk pinned at
0, whichever policy hides the most speech wins on latency, checked only by
its waste_rate against lambda_c. The interesting coupling therefore does
NOT live in "which timing" but in "whether to retrieve at all": see
divergence_band() below and ours.py's PAPER CLAIM comment block.

The quality channel is kept parameterized rather than deleted because it
is a real effect that this configuration happens to zero out — see
partial_query_risk()'s docstring for exactly which two properties of E2's
setup are responsible, and what would make it nonzero again.

---------------------------------------------------------------------------
TUNING / FAIRNESS
---------------------------------------------------------------------------
E3_BASELINE_INTERFACE.md's fairness rule (handbook 29.3) requires every
baseline's thresholds to be SWEPT on the dev split under an equal tuning
budget, not defaulted. Nothing in this module is swept yet — the values
here are the starting point that sweep begins from. Do not report a
headline number off these defaults.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baseline import Ctx  # noqa: E402 -- see _REPO_ROOT sys.path fixup above

_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "config.yaml"

# The timing policies E3's controllers range over — controller/timing.py's
# VALID_POLICIES, ordered cheapest-to-earliest so that a tie in any argmax
# or argmin over them resolves toward the more conservative (later,
# lower-risk) policy. ours.py and b9_decomposed.py each mirror this order in
# their own _TIMING_CANDIDATES; divergence_band() reuses it so the band's
# tie-breaking matches the decisions it is annotating.
TIMING_ORDER = ("sentence_end", "immediate", "fixed_early_proxy")

# Mirrors configs/config.yaml's controller.e3 — kept in sync the same way
# controller/decide.py's _DEFAULT_DECISION_CFG mirrors controller.decision,
# so a baseline stays unit-testable without the config file present.
_DEFAULT_E3_CFG = {
    # Price point used when ctx.prices is empty. The sweep always passes
    # real prices (simulator/e3_sweep.py's --lambda-l/--lambda-c); this is
    # only so a price-aware baseline is still runnable standalone.
    "default_prices": {"lambda_l": 1.0, "lambda_c": 0.05},
    # MEASURED (see module docstring): median retrieval duration.
    "retrieval_latency_ms": 672,
    # MEASURED: mean overlap_with_speech_ms per E2 timing policy — how much
    # of a retrieval actually landed inside the user's speech window, i.e.
    # how much of its latency the user never experienced.
    "overlap_ms": {
        "sentence_end": 0,
        "immediate": 50,
        "fixed_early_proxy": 188,
    },
    # MEASURED as exactly 0 for every policy — E2's per-policy hit@3 is
    # identical (250/250 each). See partial_query_risk().
    "partial_query_risk": {
        "sentence_end": 0.0,
        "immediate": 0.0,
        "fixed_early_proxy": 0.0,
    },
    # PLACEHOLDER, not measured. See waste_rate().
    "waste_rate": {
        "sentence_end": 0.0,
        "immediate": 0.02,
        "fixed_early_proxy": 0.08,
    },
    # voi_estimate()'s entropy normalizer — see that function.
    "voi": {
        "max_entropy_nats_scale": 2.5,
        "tracker_weight": 0.35,
        "entropy_weight": 0.65,
        "floor": 0.05,
    },
}


def load_e3_config(path: Optional[str] = None) -> dict:
    """
    Read controller.e3 from configs/config.yaml, merged over the defaults
    above. Same shape and same fallback behavior as
    controller/timing.py's load_timing_config() — nested dicts are merged
    per-key so a config that overrides only one timing policy's overlap
    doesn't silently drop the other two.
    """
    import yaml  # imported here so importing this module doesn't require it

    cfg_file = Path(path) if path else Path(
        os.environ.get("VOICE_RAG_CONFIG") or _DEFAULT_CONFIG_PATH
    )
    try:
        with open(cfg_file, "r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return _deep_merge(_DEFAULT_E3_CFG, {})

    e3_cfg = (full.get("controller") or {}).get("e3") or {}
    return _deep_merge(_DEFAULT_E3_CFG, e3_cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """One-level-deep merge of `override` onto `base` — enough for the
    nested-dict shape above, and deliberately not a general recursive
    merge, which would be more machinery than this config needs."""
    merged = {}
    for key, base_value in base.items():
        override_value = override.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = {**base_value, **override_value}
        elif override_value is not None:
            merged[key] = override_value
        else:
            merged[key] = base_value
    # Keys present only in the override (forward compatibility with a
    # config that adds a knob this module doesn't know about yet).
    for key, value in override.items():
        merged.setdefault(key, value)
    return merged


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Prices:
    """
    The (lambda_l, lambda_c) price point a price-aware baseline is
    optimizing against. lambda_l is quality-per-SECOND of latency,
    lambda_c is quality-per-retrieval of compute — see this module's
    docstring on units.
    """

    lambda_l: float
    lambda_c: float


def read_prices(ctx: Ctx, cfg: Optional[dict] = None) -> Prices:
    """
    Pull the price point out of ctx.prices, falling back to config's
    default_prices for anything absent.

    Ctx.prices defaults to {} (never None) per controller/baseline.py, so
    .get() is always safe here. A missing price is a real, expected case:
    simulator/e3_sweep.py only passes prices when --lambda-l/--lambda-c
    are given, and the pipeline's non-sweep paths never pass them at all.
    Falling back rather than raising keeps Ours/B9 runnable in those
    paths; the sweep that actually produces frontier data always supplies
    both.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    defaults = cfg["default_prices"]
    lambda_l = ctx.prices.get("lambda_l")
    lambda_c = ctx.prices.get("lambda_c")
    return Prices(
        lambda_l=float(lambda_l if lambda_l is not None else defaults["lambda_l"]),
        lambda_c=float(lambda_c if lambda_c is not None else defaults["lambda_c"]),
    )


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------
def scalarized_utility(
    quality: float, latency_s: float, compute_units: float, prices: Prices
) -> float:
    """
    The one objective E3 scores every priced policy by:

        U = quality - lambda_l * latency_s - lambda_c * compute_units

    Ours and B9 both call exactly this function, with exactly the same
    price point, on exactly the same scale. Any difference in what they
    decide therefore comes from the arguments they choose to feed it —
    which is precisely the joint-vs-decomposed distinction under test,
    and nothing else.
    """
    return quality - prices.lambda_l * latency_s - prices.lambda_c * compute_units


# ---------------------------------------------------------------------------
# Belief -> expected quality gain (value of information)
# ---------------------------------------------------------------------------
def voi_estimate(ctx: Ctx, cfg: Optional[dict] = None) -> float:
    """
    Expected quality GAIN from retrieving this turn, on [0, 1] — the
    "quality" term of the objective.

    This is a belief question, not a pricing question, so it is shared
    unchanged by Ours, B9, and (via the same signals) B8. Giving Ours a
    sharper VoI estimate than B9 would confound the experiment: the
    measured gap has to come from joint reasoning, not from a better
    estimator on one side.

    Signal ladder, in order of preference, all read defensively with
    .get() per Ctx's contract ("a baseline must treat a missing key as
    'signal not available'"):

      1. model uncertainty — signals["model"]["uncertainty"]
         ["max_entropy_nats"], Experiment 1's per-token output entropy
         from the frozen closed-book generation (simulator/e3_signals.py).
         High entropy = the model is unsure answering unaided = retrieval
         is worth more. Normalized by max_entropy_nats_scale and clipped
         to [0, 1]; the scale is a config knob because nats have no
         natural ceiling.
      2. tracker uncertainty — signals["tracker"]["unknown_fraction"],
         llm/signals.py's tracker summary. Already a fraction on [0, 1].
      3. tracker_slots fallback — the raw slot dict controller/decide.py's
         live rule reads. Used when the frozen snapshot isn't present at
         all (e.g. a pipeline path that passes signals=None): intent
         unknown -> 1.0, else the unknown fraction over the slots present.

    When both (1) and (2) are available they're blended by the configured
    weights rather than one being dropped — they're different views of
    uncertainty (generation-time vs. slot-fill) and E1 validated the
    entropy family, so entropy carries the larger weight by default.

    Returns at least `floor` (default 0.05): a "retrieval is worth exactly
    nothing" estimate is never justified from these signals, and a hard
    zero would make the retrieve branch unreachable at any nonzero price,
    which would be an artifact of the estimator rather than a decision.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    voi_cfg = cfg["voi"]

    signals = ctx.signals or {}

    entropy_voi = None
    model_block = signals.get("model") or {}
    uncertainty = model_block.get("uncertainty") or {}
    max_entropy = uncertainty.get("max_entropy_nats")
    if max_entropy is not None:
        scale = max(float(voi_cfg["max_entropy_nats_scale"]), 1e-9)
        entropy_voi = _clip01(float(max_entropy) / scale)

    tracker_voi = None
    tracker_block = signals.get("tracker") or {}
    unknown_fraction = tracker_block.get("unknown_fraction")
    if unknown_fraction is not None:
        tracker_voi = _clip01(float(unknown_fraction))
    else:
        # Fallback: the raw slot dict, which pipeline/spine.py always adds
        # to Ctx.signals even when no frozen snapshot was passed.
        slots = signals.get("tracker_slots") or {}
        if slots:
            if slots.get("intent", "unknown") == "unknown":
                tracker_voi = 1.0
            else:
                # "confirmed" is a boolean status flag, not an uncertainty
                # slot — excluded for the same reason configs/config.yaml's
                # controller.decision.key_slots excludes it.
                keys = [k for k in slots if k != "confirmed"]
                unknown = sum(1 for k in keys if slots.get(k, "unknown") == "unknown")
                tracker_voi = (unknown / len(keys)) if keys else None

    floor = float(voi_cfg["floor"])
    if entropy_voi is not None and tracker_voi is not None:
        w_e = float(voi_cfg["entropy_weight"])
        w_t = float(voi_cfg["tracker_weight"])
        total = w_e + w_t
        blended = (entropy_voi * w_e + tracker_voi * w_t) / (total if total else 1.0)
        return max(floor, _clip01(blended))
    if entropy_voi is not None:
        return max(floor, entropy_voi)
    if tracker_voi is not None:
        return max(floor, tracker_voi)
    # No usable signal at all. Not the same as "retrieval is worthless" —
    # it's "we don't know", and the safe reading of "we don't know" for a
    # retrieval decision is the same one controller/decide.py takes
    # (retrieve rather than guess), so return a high VoI, not the floor.
    return 1.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# The cost model — what retrieval costs, and what timing does to that cost
# ---------------------------------------------------------------------------
def retrieval_latency_s(cfg: Optional[dict] = None) -> float:
    """
    Expected wall-clock duration of one retrieval, in seconds. MEASURED:
    median 672 ms over 751 retrievals in logs/ (see module docstring).

    This is the FULL cost of a retrieval, before any timing-dependent
    hiding — the number B9's retrieval optimizer prices against, and the
    number Ours starts from before subtracting overlap.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    return float(cfg["retrieval_latency_ms"]) / 1000.0


def overlap_s(timing: str, cfg: Optional[dict] = None) -> float:
    """
    How much of a retrieval's duration a given timing policy hides inside
    the user's speech, in seconds. MEASURED per policy from
    retrieval_result's overlap_with_speech_ms (see module docstring):
    sentence_end 0, immediate 50 ms, fixed_early_proxy 188 ms.

    This is coupling channel 1 (latency). It is the entire reason a joint
    controller can afford a retrieval a decomposed one can't: launching
    during speech means the user never experiences that portion of the
    wait. controller/timing.py's module docstring explains why this
    overlap is real work racing real time, not a number subtracted from a
    metric.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    return float(cfg["overlap_ms"].get(timing, 0)) / 1000.0


def exposed_latency_s(timing: str, cfg: Optional[dict] = None) -> float:
    """
    The latency the USER actually experiences from a retrieval launched
    under `timing`: the full retrieval duration minus whatever that timing
    hid inside speech, floored at zero.

    This is the latency term the objective should be priced on — pricing
    the full duration regardless of timing is exactly the modeling error
    that defines B9 (see b9_decomposed.py), so this function is called by
    Ours and deliberately NOT by B9's retrieval optimizer.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    return max(0.0, retrieval_latency_s(cfg) - overlap_s(timing, cfg))


def partial_query_risk(timing: str, cfg: Optional[dict] = None) -> float:
    """
    Fraction of retrieval's quality gain lost by launching early on a
    PARTIAL transcript instead of the finished one, on [0, 1].

    This is the quality coupling channel — in principle what stops "launch
    as early as possible" from being trivially optimal: early timing buys
    hidden latency but spends some retrieval quality, because the query it
    retrieves on isn't the whole question yet.

    *** MEASURED, AND THE MEASURED VALUE IS 0 FOR EVERY POLICY. ***

    Fit exactly as the formula this docstring used to propose,
    risk(t) = 1 - hit@3(t)/hit@3(sentence_end), over E2's 750 scored runs
    in simulator/e2_results/ + logs/, scored by harness/metrics.py's
    hit_at_k_report():

        policy              hit@1    hit@2    hit@3
        sentence_end        92.0%    96.0%    100.0%   (250/250)
        immediate           92.0%    96.0%    100.0%   (250/250)
        fixed_early_proxy   92.0%    96.0%    100.0%   (250/250)

    -> risk = 1 - 1.00/1.00 = 0.0, for all three.

    Reproduce:
        from harness.metrics import hit_at_k_report
        hit_at_k_report("logs", k=3)

    NOT A CEILING ARTIFACT. The obvious objection to a 100%/100% ratio is
    that hit@3 is saturated and the ratio is uninformative. It isn't: at
    k=1, where there is ample headroom (92%), all three policies miss on
    the IDENTICAL 20 (scenario, seed) pairs. Retrieval quality is identical
    PER INSTANCE across timings, not merely equal in aggregate, which is a
    much stronger statement than the hit@3 ratio alone supports.

    WHY it is zero — both causes, because naming only the second (the more
    flattering one) would overstate what E2 actually tested:

      1. E2 ran with frozen_query (simulator/e2_results/RUN_MANIFEST.md,
         commit 090ec85; pipeline/spine.py's run_retrieval_pipeline()
         skips the live rewrite when it's set). The query rewrite was
         precomputed ONCE from the full question and handed to every
         policy, so all three retrieved on a byte-identical query string.
         In this run, partial-transcript degradation was structurally
         impossible, not merely small. This is the dominant cause.
      2. controller/timing.py's query_ready gate delays dispatch to the
         earliest MEANINGFUL partial rather than firing on single-word
         prefixes ("What"/"Tell"/"Can"). This is the mechanism that would
         hold the risk down on a LIVE partial-transcript path — but E2 did
         not exercise it as a query-CONTENT effect, so E2 does not
         evidence its size.

    CONSEQUENCE FOR E3: the quality coupling channel is empirically
    inactive here, and the coupling Ours exploits runs through latency
    alone. See this module's docstring ("ONLY ONE COUPLING CHANNEL IS
    ACTIVE") and divergence_band() below.

    Kept as a parameter, and still read by both controllers, rather than
    deleted: the term is a real effect that this configuration zeroes out,
    and it would be nonzero on harder retrieval (a corpus where top-3 is
    not saturated) or on a live partial-transcript run with no frozen
    query. Everything downstream is written against the general form, so
    grounding it at a nonzero value later is a config edit and not a code
    change. What is reported in the paper, though, is the measured 0.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    return _clip01(float(cfg["partial_query_risk"].get(timing, 0.0)))


def waste_rate(timing: str, cfg: Optional[dict] = None) -> float:
    """
    Expected EXTRA compute per retrieval under `timing`, in retrieval
    units — an early launch whose premise turns out to be wrong gets
    cancelled and relaunched, so it's paid for twice.

    *** PLACEHOLDER — NOT MEASURED. *** All 751 retrievals in logs/ have
    forecast_invalidated=false, so the observed rate here is exactly 0 —
    but that's a property of the TEXT path, which has no continued-speech
    signal to invalidate against (controller/timing.py's
    PartialSignalWatcher.wait_for_growth only fires on the live/synthetic-
    audio path). A zero default would therefore encode "early launches are
    never wasted", which the text path cannot actually evidence. These
    small nonzero defaults are the honest placeholder until a live-audio
    run produces a real invalidation rate; fixed_early_proxy is set higher
    than immediate because it acts on a fixed proxy point without waiting
    to observe anything confirm it (see controller/timing.py's docstring).
    """
    cfg = cfg if cfg is not None else load_e3_config()
    return max(0.0, float(cfg["waste_rate"].get(timing, 0.0)))


def retrieval_compute_units(timing: str, cfg: Optional[dict] = None) -> float:
    """Total compute charged for retrieving under `timing`: one retrieval
    plus that timing's expected waste."""
    return 1.0 + waste_rate(timing, cfg)


# ---------------------------------------------------------------------------
# The retrieve/skip boundary — where Ours and B9 actually disagree
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DivergenceBand:
    """
    The interval of VoI values on which Ours retrieves and B9 does not.

    This is the quantitative form of E3's hypothesis. Both controllers
    retrieve iff VoI clears a threshold; the thresholds differ ONLY because
    B9 prices retrieval at full latency while Ours prices it at the latency
    a chosen timing actually exposes. Everything between the two thresholds
    is a retrieval with positive net value that decomposition cannot see.

    DIAGNOSTIC ONLY. Both ours.py and b9_decomposed.py compute this for
    their reason strings and NEITHER lets it influence a decision — see
    divergence_band() below on why that distinction is load-bearing for B9
    in particular.
    """

    timing: str              # the timing that gives Ours its cheaper threshold
    ours_cut: float          # Ours retrieves iff voi > this
    b9_cut: float            # B9 retrieves iff voi > this
    exposed_latency_s: float # what Ours pays under `timing`
    full_latency_s: float    # what B9 pays, unconditionally

    @property
    def width(self) -> float:
        """b9_cut - ours_cut. Can be <= 0 (an empty band) — see
        divergence_band()'s note on lambda_l = 0."""
        return self.b9_cut - self.ours_cut

    def contains(self, voi: float) -> bool:
        """True iff this VoI is one Ours retrieves on and B9 skips.

        The bounds match the two controllers' comparisons EXACTLY, so a
        True here is never a claim the logs can contradict: Ours retrieves
        on a strict `> ours_cut` (its argmax puts skip first on ties, see
        ours.py's _TIMING_CANDIDATES) and B9 retrieves on a strict
        `> b9_cut` (its `utility > 0.0`). So the band is half-open:
        (ours_cut, b9_cut].
        """
        return self.ours_cut < voi <= self.b9_cut

    def describe(self) -> str:
        """Compact form for a reason string / sweep log. Deliberately does
        NOT repeat the exposed-vs-full latency split — each controller logs
        the latency IT actually priced alongside this, and duplicating it
        here just makes the log line harder to read."""
        return (
            f"band=({self.ours_cut:.3f},{self.b9_cut:.3f}]@{self.timing} "
            f"w={self.width:+.3f}"
        )


def divergence_band(prices: Prices, cfg: Optional[dict] = None) -> DivergenceBand:
    """
    Compute the (ours_cut, b9_cut] retrieve/skip band at a price point.

    WHY THIS IS THE INTERESTING QUANTITY. With partial_query_risk measured
    at 0 (see that function), the timing sub-decision is nearly trivial —
    both controllers pick the highest-overlap policy at any sane price, so
    "Ours picks a better timing" is NOT where the E3 effect lives. What
    knowing the timing is cheap actually buys is a shifted retrieve/skip
    THRESHOLD, and that is what this function measures.

    Derivation. Both controllers compare a retrieve utility against
    U(skip) = 0. Ours, per candidate timing t:

        voi * (1 - risk(t)) - lambda_l * exposed(t)
                            - lambda_c * (1 + waste(t))  >  0

    which rearranges (for risk(t) < 1) to a per-timing threshold

        cut(t) = [lambda_l * exposed(t) + lambda_c * (1 + waste(t))]
                 / (1 - risk(t))

    and Ours retrieves iff voi clears the SMALLEST of those, so
    ours_cut = min_t cut(t) and `timing` is that argmin. B9's retrieval
    optimizer has no timing to condition on and prices the full schedule:

        b9_cut = lambda_l * retrieval_latency + lambda_c * (1 + waste(
                 sentence_end))

    With the measured constants (retrieval 0.672s, overlap 0.188s, risk 0)
    and lambda_c -> 0 this reduces to the headline form quoted in ours.py:

        lambda_l * 0.484  <  voi  <  lambda_l * 0.672

    i.e. a band of width lambda_l * overlap. It is PRICE-DEPENDENT by
    construction — wide at high lambda_l, vanishing at lambda_l = 0 — which
    is why E3 sweeps a grid instead of reporting one number, and what makes
    the claim falsifiable: if measured overlap collapses, so does the band.

    Note `timing` is the argmin-cut timing, which is the timing whose
    utility crosses zero first and therefore the one Ours retrieves with at
    the boundary. With risk identically 0 (the measured case) it is also
    Ours's argmax timing at EVERY voi, so the band's `timing` and the
    logged decision's timing always agree. Were risk nonzero and
    policy-dependent, the two could differ away from the boundary; the band
    is defined at the boundary, which is where it is used.

    Width can be <= 0 — at lambda_l = 0 the latency channel vanishes and
    Ours's extra waste_rate makes its cut the HIGHER one. contains() then
    returns False for every voi, which is correct: with latency free there
    is nothing for joint reasoning to exploit, and the honest log entry is
    "no divergence available at this price".

    *** DIAGNOSTIC ONLY — this must never feed a decision. *** For Ours
    that is merely good hygiene. For B9 it is load-bearing: B9 is defined
    by its retrieval optimizer having no access to timing (see
    b9_decomposed.py), and this function reads overlap_s(), so calling it
    anywhere in B9's decision path would silently convert B9 into a joint
    controller and destroy the experiment. B9 calls it only after both
    sub-optimizers have returned, purely to annotate the reason string.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    full_s = retrieval_latency_s(cfg)

    def cut(timing: str) -> float:
        risk = partial_query_risk(timing, cfg)
        cost = (
            prices.lambda_l * exposed_latency_s(timing, cfg)
            + prices.lambda_c * retrieval_compute_units(timing, cfg)
        )
        # risk == 1 would mean "early retrieval is worth literally nothing",
        # which no VoI can clear; represent that as an unreachable cut
        # rather than dividing by zero.
        return cost / (1.0 - risk) if risk < 1.0 else float("inf")

    # Same order both controllers enumerate in (cheapest-to-earliest), so a
    # tie in cut() resolves to the same conservative policy the controllers
    # themselves would pick — min() returns the first minimal element. Ties
    # are real at lambda_l = 0, where every timing's latency term is 0.
    timings = [t for t in TIMING_ORDER if t in cfg["overlap_ms"]]
    timings += [t for t in cfg["overlap_ms"] if t not in TIMING_ORDER]
    best_timing = min(timings, key=cut)

    return DivergenceBand(
        timing=best_timing,
        ours_cut=cut(best_timing),
        b9_cut=(
            prices.lambda_l * full_s
            + prices.lambda_c * retrieval_compute_units("sentence_end", cfg)
        ),
        exposed_latency_s=exposed_latency_s(best_timing, cfg),
        full_latency_s=full_s,
    )


# ---------------------------------------------------------------------------
# The memory hook — Experiment 4's slot, deliberately inert
# ---------------------------------------------------------------------------
def decide_memory(ctx: Ctx) -> Optional[str]:
    """
    The third decision (memory), not implemented.

    Returns None — "no memory operation" — for every baseline today.
    Experiment 4 is cut for this deadline per paper/preregistration.md, so
    this exists to fix the SHAPE of the eventual decision, not its
    content: Decision.memory_op already accepts it (controller/
    baseline.py), each baseline already routes through a memory hook, and
    Ours already carries a memory_op field on its joint candidates. Adding
    memory later means filling in these hooks and extending Ours's
    candidate enumeration — not restructuring any of the three
    controllers.

    Kept as a module-level function rather than a method so that all three
    baselines demonstrably share one inert implementation, and so the day
    it becomes real, the joint-vs-decomposed distinction is about how each
    baseline USES it, not about who reimplemented it.
    """
    return None
