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
  - the belief -> quality estimate  (estimate_voi, VoiCalibration)
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
ctx.prices at all. It reads the same belief signals directly, which is a
belief question, not a pricing one.

---------------------------------------------------------------------------
THE OBJECTIVE
---------------------------------------------------------------------------
Per E3_BASELINE_INTERFACE.md ("The scalarization objective (B9 and Ours)")
and the handbook, every policy is scored by

    U = quality  -  lambda_l * latency  -  lambda_c * compute

UNITS MATTER, and they are the one thing easy to get silently wrong:

  quality  — dimensionless, on [0, 1]. estimate_voi()'s scale: the
             expected quality GAIN from retrieving, read as P(retrieval is
             needed | this turn's uncertainty) * voi.utility_scale, where
             1.0 means "retrieval turns a certainly-wrong answer into a
             certainly-right one" and 0.0 means "retrieval adds nothing".
             utility_scale is 1.0 for every E3 run, so quality IS that
             probability; see the config block on what changing it costs.
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

The VoI calibration is likewise measured, but from Experiment 1 rather than
from logs/: P(retrieval needed) = sigmoid(11.2878 * mean_entropy_nats
- 3.6334), fitted on the 180 real samples of E1's Mistral-7B Modal run
(in-sample AUC 0.9260, equal to E1's own five-seed mean 0.9253 for that
signal). It lives in controller/baselines/e1_calibration_mean_entropy.json
with the source run's sha256; regenerate with
`python -m llm.experiment1.fit_calibration --signal mean_entropy_nats`.

Through 2026-08-16 this was E1's headline signal max_entropy_nats
(AUC 0.9784, artifact kept at controller/baselines/e1_calibration.json).
It ranks better and calibrates worse — being a MAX over per-token
entropies, it saturates on any open-ended answer, which made VoI
near-binary and left the derived price grid with scenarios in only 3 of 26
bands. See load_voi_calibration()'s docstring for the full argument and
simulator/e3_divergence_check.py for the measurement.

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

import json
import math
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
    "b4": {
        "threshold": 0.5,
    },

    "b6": {
        "threshold": 1.0,
    },
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
    # estimate_voi()'s belief -> P(retrieval needed) mapping. See
    # VoiCalibration and load_voi_calibration() for the field meanings and
    # for the sign guard; configs/config.yaml carries the same block with
    # the full provenance comment.
    "voi": {
        "calibration": {
            # WHICH E1 signal the estimator keys on. A config knob, not a
            # constant: which signal E3 grounds on is a decision about E1's
            # data, and it must be readable (and swappable) without editing
            # code. It is NOT the source of the coefficients — those come
            # from the artifact below — it is a REQUEST, cross-checked
            # against the artifact's own `signal` field by
            # load_voi_calibration(). Asking for one signal while pointing
            # at another signal's fit is the one silent failure this pair
            # of keys exists to make loud.
            "signal": "mean_entropy_nats",
            # THE calibration. Every coefficient lives in that artifact and
            # nowhere else — it is generated by
            # `python -m llm.experiment1.fit_calibration --signal <signal>`
            # from the committed E1 Modal result, and carries the source
            # file's sha256 next to the numbers. Deliberately NOT duplicated
            # inline here: a second copy of the coefficients is a copy that
            # can drift out of step with the run it claims to come from.
            #
            # Points at the mean_entropy fit since 2026-08-17. The
            # max_entropy fit it replaced is kept at e1_calibration.json —
            # see load_voi_calibration()'s docstring for why the swap
            # happened, and simulator/e3_grid.py for what it did to the grid.
            "from_artifact": (
                "controller/baselines/e1_calibration_mean_entropy.json"
            ),
        },
        # Quality units earned by a retrieval that is certainly needed. The
        # objective's quality term is voi = P(retrieval needed) *
        # utility_scale, so at 1.0 (the value E3 runs at) VoI IS the
        # probability and quality stays on [0, 1] — the scale this module's
        # unit docstring specifies, and the scale the price grid is
        # calibrated against. It exists as a knob only so that "how much is
        # a perfect retrieval worth?" is a stated assumption rather than an
        # invisible 1.0 buried in the estimator; changing it rescales
        # quality against latency and compute and therefore invalidates the
        # derived price grid (simulator/e3_grid.py), which is why the grid
        # artifact records the utility_scale it was derived under.
        "utility_scale": 1.0,
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
@dataclass(frozen=True)
class VoiCalibration:
    """
    The mapping from ONE Experiment 1 uncertainty signal to a VoI on [0, 1].

    E3's quality term is meant to be "the expected quality gain from
    retrieving". Experiment 1's whole contribution is that a model's own
    uncertainty on a question predicts whether retrieval was NEEDED, so the
    principled reading of VoI is a calibrated probability:

        voi = P(retrieval is needed | this turn's uncertainty signal)

    which is a quantity with units, a defensible [0, 1] range, and a fitted
    provenance — rather than a hand-chosen normalizer.

    IN USE (since 2026-08-15): the "logistic" kind, fitted on E1's real
    Modal samples and loaded from controller/baselines/e1_calibration.json
    via from_artifact. The "normalized" kind below is the interim stand-in
    it replaced; it is kept implemented, and only implemented, so that the
    sign guard and the calibrated/uncalibrated distinction stay testable and
    so a future signal without a fit has somewhere to land.

    Fields:
      signal      which key of signals["model"]["uncertainty"] to read
                  (e.g. "max_entropy_nats"). A config knob and not a
                  constant because WHICH signal E1 validates is E1's
                  result, not something this module gets to assume.
      kind        "logistic"   -> sigmoid(intercept + slope * x); this is
                                  the shape a committed E1 calibration
                                  takes, and the only kind that may set
                                  calibrated=True.
                  "normalized" -> clip01(x / normalizer); an UNCALIBRATED
                                  monotone stand-in for use before a valid
                                  E1 fit exists. See `calibrated`.
      intercept   logistic beta_0
      slope       logistic beta_1. MUST be > 0 — see load_voi_calibration().
      normalizer  "normalized" divisor. "auto" resolves to ln(top_k + 1),
                  the structural ceiling of llm/client.py's entropy (see
                  _resolve_normalizer).
      source      human-readable provenance, quoted in the paper.
      auc         the fit's reported AUC, or None if not a fit.
      calibrated  True ONLY for coefficients fitted on a valid E1 run.
                  False marks an interim mapping that must not be
                  described as a calibration anywhere.

    This record is shared UNCHANGED by Ours and B9 (both call
    estimate_voi(), which reads it), so it cannot advantage either. Only
    the joint-vs-decomposed USE of the resulting number differs.
    """

    signal: str
    kind: str
    intercept: float
    slope: float
    normalizer: float
    source: str
    auc: Optional[float]
    calibrated: bool

    def apply(self, x: float) -> float:
        """Map a raw signal value to a VoI on [0, 1]."""
        if self.kind == "logistic":
            # Guard the exponent so a wild coefficient can't raise
            # OverflowError mid-sweep; +-40 is already saturated to within
            # 5e-18 of 0/1, so the clamp is numerically invisible.
            z = max(-40.0, min(40.0, self.intercept + self.slope * float(x)))
            return 1.0 / (1.0 + math.exp(-z))
        if self.kind == "normalized":
            return _clip01(float(x) / max(self.normalizer, 1e-9))
        raise ValueError(
            f"unknown VoI calibration kind {self.kind!r} "
            f"(expected 'logistic' or 'normalized')"
        )

    def describe(self) -> str:
        """Compact provenance for a reason string / sweep log, so a reader
        of the logs can tell a fitted estimator from an interim one without
        opening the config."""
        mark = "calibrated" if self.calibrated else "INTERIM-UNCALIBRATED"
        auc = f" auc={self.auc:.3f}" if self.auc is not None else ""
        return f"voi[{self.signal}/{self.kind}/{mark}{auc}]"


def _resolve_normalizer(value, cfg_path: Optional[str] = None) -> float:
    """
    Resolve a "normalized" calibration's divisor.

    "auto" means the STRUCTURAL ceiling of the entropy signal rather than a
    chosen constant. llm/client.py's TokenLogprob.entropy_nats builds its
    distribution from the returned top-k candidates plus one "<other>"
    residual bucket, so it cannot exceed ln(top_k + 1) — 1.7918 nats at the
    repo's llm.logprobs.top_k of 5. Dividing by that maps the signal onto
    the full [0, 1] by construction, with no free parameter to tune.

    This is why top_k is load-bearing for E3 and must be frozen alongside
    the price grid: raising it raises the ceiling and rescales every VoI.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or value.strip().lower() != "auto":
        raise ValueError(
            f"voi.calibration.normalizer must be a number or 'auto', got {value!r}"
        )

    top_k = 5  # matches configs/config.yaml llm.logprobs.top_k
    try:
        import yaml

        cfg_file = Path(cfg_path) if cfg_path else Path(
            os.environ.get("VOICE_RAG_CONFIG") or _DEFAULT_CONFIG_PATH
        )
        with open(cfg_file, "r", encoding="utf-8") as fh:
            full = yaml.safe_load(fh) or {}
        configured = ((full.get("llm") or {}).get("logprobs") or {}).get("top_k")
        if configured is not None:
            top_k = int(configured)
    except (FileNotFoundError, ImportError):
        # Same fallback posture as load_e3_config(): a missing config file
        # must not make a baseline unimportable.
        pass
    return math.log(top_k + 1)


def load_voi_calibration(cfg: Optional[dict] = None) -> VoiCalibration:
    """
    Build the VoiCalibration from config, preferring a committed E1 artifact.

    Resolution order:

      1. voi.calibration.from_artifact — a path to a committed calibration
         JSON carrying a top-level "calibration" object with the same field
         names as VoiCalibration. THIS IS THE ACTIVE PATH: it points at
         controller/baselines/e1_calibration_mean_entropy.json, generated
         from the E1 Modal result by `python -m llm.experiment1.
         fit_calibration --signal mean_entropy_nats`.
      2. the inline voi.calibration block in configs/config.yaml, for
         anything the artifact does not set. It no longer carries
         coefficients — see configs/config.yaml on why they live in exactly
         one place.

    *** THE SIGNAL CROSS-CHECK. ***

    Config names the signal it WANTS (voi.calibration.signal); the artifact
    names the signal it was FITTED on. Both are set, and they must agree.

    They are two independent statements about the same choice, and the
    failure mode of keeping only one is silent in both directions. Read the
    signal only from the artifact and swapping signals means editing a
    generated file's path with nothing in config recording the intent. Read
    it only from config and a config saying "mean_entropy_nats" over an
    artifact fitted on max_entropy_nats would apply max-entropy
    coefficients to mean-entropy values — a wrong VoI on every turn, with
    no exception, no NaN and no log line out of place. The estimator would
    simply be miscalibrated, and the sweep built on it would look fine.

    So the check is an equality, and a mismatch raises. Swapping the
    signal is deliberately a TWO-line edit: point from_artifact at that
    signal's fit, and say so in `signal`.

    *** WHY THIS ESTIMATOR IS ON mean_entropy_nats AND NOT E1'S HEADLINE. ***

    E1's argmax signal is max_entropy_nats (five-seed mean AUC 0.9784), and
    E3 was grounded on it through 2026-08-16. It ranks superbly and is
    unusable as a CALIBRATION INPUT, for a reason AUC cannot see: it is a
    MAX over per-token entropies, so a single uncertain token saturates it,
    and it saturates on essentially any open-ended answer. On E3's 50
    scenarios that made VoI near-binary — 14 scenarios pinned at the 0.05
    floor, median 0.9515, nothing in between — and a price grid tiled
    across that distribution had scenarios in only 3 of its 26 bands. The
    frontier had no populated middle, which is precisely where an
    Ours-vs-B9 divergence has to live.

    mean_entropy_nats is the same measurement AVERAGED rather than maxed.
    It separates the classes nearly as well (E1 five-seed mean AUC 0.9253,
    over E1's own 0.70 success bar, so it is gated — see
    llm/experiment1/fit_calibration.py's load_e1_samples() on why the gate
    and not the argmax is the right guard) and, unlike the max, it is
    continuous: on the same 50 scenarios it spans 0.006-0.660 with a
    largest gap of 0.073 and 14 of 50 in the middle third.

    That choice was made on the SIGNAL's distribution — E1's samples and
    the frozen scenario signals — and not on any Ours-vs-B9 outcome. It
    fixes the instrument's resolution, not the answer it reports.

    *** THE SIGN GUARD, and why it exists. ***

    A calibration marked calibrated=True must have slope > 0. Every signal
    this estimator can key on is an UNCERTAINTY signal, so the direction is
    fixed by E1's hypothesis: more uncertain -> retrieval more valuable. A
    negative fitted slope does not mean "retrieval is worth less when the
    model is unsure" — it means the fit found no signal and latched onto
    noise, and installing it would silently invert the controller's premise
    while every log line still looked healthy.


    The guard was written against exactly this failure and has since been
    checked against both data sets in the repo. On the MOCK smoke run
    (llm/experiment1/results/smoke_mock.json, 180 samples) the fit gives
    slope -1.5144 at AUC 0.4453 — below chance, mean entropy HIGHER on the
    not-needed class, because MockClient's synthetic logprobs are drawn
    independently of the labels. On the REAL Modal run now in use it gives
    slope +11.7691 at AUC 0.9784. The guard turns the first case from a
    silent inversion into a loud failure and passes the second untouched.

    Uncalibrated interim mappings (calibrated=False) skip the guard,
    because they claim no fit — but "normalized" is monotone increasing by
    construction anyway.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    block = dict((cfg.get("voi") or {}).get("calibration") or {})
    # Captured BEFORE the artifact is spliced in, because the splice
    # overwrites it — this is the config's request, to be checked against
    # what the artifact was actually fitted on. See the docstring.
    requested_signal = block.get("signal")

    artifact = block.get("from_artifact")
    if artifact:
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = _REPO_ROOT / artifact_path
        try:
            with open(artifact_path, "r", encoding="utf-8") as fh:
                fitted = (json.load(fh) or {}).get("calibration")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"voi.calibration.from_artifact points at {artifact_path}, which "
                f"does not exist. Either commit the E1 result there or set "
                f"from_artifact: null to use the inline interim calibration."
            ) from error
        if not fitted:
            raise ValueError(
                f"{artifact_path} has no top-level 'calibration' object — an E1 "
                f"result used as a VoI calibration must carry the fitted "
                f"coefficients, not just the AUC table."
            )
        fitted_signal = fitted.get("signal")
        if (
            requested_signal is not None
            and fitted_signal is not None
            and requested_signal != fitted_signal
        ):
            raise ValueError(
                f"E3 VoI calibration mismatch: configs/config.yaml's "
                f"controller.e3.voi.calibration.signal asks for "
                f"{requested_signal!r}, but {artifact_path} was fitted on "
                f"{fitted_signal!r}. Applying one signal's coefficients to "
                f"another signal's values miscalibrates every VoI silently — "
                f"there is no exception and no bad-looking log line, only a "
                f"wrong number. Point from_artifact at the "
                f"{requested_signal!r} fit (python -m "
                f"llm.experiment1.fit_calibration --signal {requested_signal}), "
                f"or change `signal` to {fitted_signal!r} if the re-grounding "
                f"is what you meant."
            )
        block.update(fitted)

    calibration = VoiCalibration(
        signal=str(block.get("signal", "max_entropy_nats")),
        kind=str(block.get("kind", "normalized")),
        intercept=float(block.get("intercept", 0.0)),
        slope=float(block.get("slope", 1.0)),
        normalizer=_resolve_normalizer(block.get("normalizer", "auto")),
        source=str(block.get("source", "(unspecified)")),
        auc=(None if block.get("auc") is None else float(block["auc"])),
        calibrated=bool(block.get("calibrated", False)),
    )

    if calibration.calibrated and calibration.kind == "logistic" and calibration.slope <= 0:
        raise ValueError(
            f"E3 VoI calibration for {calibration.signal!r} has slope "
            f"{calibration.slope:+.4f} <= 0, which inverts the uncertainty -> "
            f"value-of-information direction Experiment 1 is built on (more "
            f"uncertain must mean retrieval is worth MORE). A below-chance fit "
            f"produces exactly this. Refusing to load it. Source: "
            f"{calibration.source!r}"
        )
    return calibration


def estimate_voi(ctx: Ctx, cfg: Optional[dict] = None) -> float:
    """
    Expected quality GAIN from retrieving this turn, on [0, 1] — the
    "quality" term of the objective.

        voi = P(retrieval is needed | this turn's uncertainty)
              * voi.utility_scale

    with P from the VoiCalibration above — the logistic FITTED on
    Experiment 1's 180 real Modal samples
    (controller/baselines/e1_calibration_mean_entropy.json,
     P = sigmoid(11.2878 * mean_entropy_nats - 3.6334), in-sample AUC
     0.9260, matching E1's five-seed mean 0.9253 for that signal; P = 0.5
     at 0.3219 nats). Monotone increasing in the uncertainty signal, and
    the sign guard in load_voi_calibration() enforces that it stays so: an
    unsure model has more to gain from retrieving.

    WHICH signal is read is config's call, not this function's — it comes
    from the calibration record, cross-checked against
    controller.e3.voi.calibration.signal. Through 2026-08-16 this was
    max_entropy_nats; see load_voi_calibration()'s docstring for why the
    max saturated and what the mean bought.

    This is a belief question, not a pricing question, so it is shared
    unchanged by Ours and B9. Giving Ours a sharper VoI estimate than B9
    would confound the experiment: the measured gap has to come from joint
    reasoning, not from a better estimator on one side. Both import THIS
    function; there is no second implementation to drift.

    Signal ladder, all read defensively with .get() per Ctx's contract
    ("a baseline must treat a missing key as 'signal not available'"):

      1. the calibrated signal — signals["model"]["uncertainty"][<signal>],
         from the frozen closed-book generation E3 precomputes once per
         scenario (simulator/e3_signals.py). This is the E1 path and the
         only one with a fitted mapping behind it.
      2. tracker uncertainty — signals["tracker"]["unknown_fraction"].
         FALLBACK ONLY, used when (1) is absent; it is deliberately NOT
         blended into (1) any more. Two reasons, both empirical: it is not
         the signal family E1 validates, and across E3's 50 scenarios it
         takes only the values {0.75, 1.0}, so blending it in at a fixed
         weight added a near-constant offset that compressed the VoI
         dynamic range without adding discriminative power. (In the repo's
         only labelled E1 data it scores AUC 0.4722, identical on all five
         seeds — i.e. no discrimination at all.)
      3. tracker_slots fallback — the raw slot dict controller/decide.py's
         live rule reads, for pipeline paths that pass signals=None:
         intent unknown -> 1.0, else the unknown fraction over the slots
         present.

    Every branch returns P * utility_scale, floored at `floor` — see
    _scaled() below. utility_scale is 1.0 for every E3 run, so VoI IS the
    probability and this scaling is invisible; it is applied uniformly
    anyway so that a changed scale cannot silently apply to the calibrated
    path but not the fallbacks.

    Returns at least `floor` (default 0.05): a "retrieval is worth exactly
    nothing" estimate is never justified from these signals, and a hard
    zero would make the retrieve branch unreachable at any nonzero price,
    which would be an artifact of the estimator rather than a decision.
    """
    cfg = cfg if cfg is not None else load_e3_config()
    voi_cfg = cfg["voi"]
    floor = float(voi_cfg["floor"])
    utility_scale = float(voi_cfg.get("utility_scale", 1.0))

    def _scaled(probability: float) -> float:
        """P(retrieval needed) -> VoI in the objective's quality units."""
        return max(floor, _clip01(probability) * utility_scale)

    signals = ctx.signals or {}
    uncertainty = (signals.get("model") or {}).get("uncertainty") or {}

    calibration = load_voi_calibration(cfg)
    raw = uncertainty.get(calibration.signal)
    if raw is not None:
        return _scaled(calibration.apply(float(raw)))

    tracker_block = signals.get("tracker") or {}
    unknown_fraction = tracker_block.get("unknown_fraction")
    if unknown_fraction is not None:
        return _scaled(float(unknown_fraction))

    slots = signals.get("tracker_slots") or {}
    if slots:
        if slots.get("intent", "unknown") == "unknown":
            return _scaled(1.0)
        # "confirmed" is a boolean status flag, not an uncertainty slot —
        # excluded for the same reason configs/config.yaml's
        # controller.decision.key_slots excludes it.
        keys = [k for k in slots if k != "confirmed"]
        if keys:
            unknown = sum(1 for k in keys if slots.get(k, "unknown") == "unknown")
            return _scaled(unknown / len(keys))

    # No usable signal at all. Not the same as "retrieval is worthless" —
    # it's "we don't know", and the safe reading of "we don't know" for a
    # retrieval decision is the same one controller/decide.py takes
    # (retrieve rather than guess), so return a high VoI, not the floor.
    return _scaled(1.0)


# Both price-aware baselines import this name. Kept as an alias rather than
# renamed at every call site so that the estimator's DEFINITION changing does
# not also churn ours.py/b9_decomposed.py — the point of this module is that
# there is exactly one shared estimator, whatever it is called.
voi_estimate = estimate_voi


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
