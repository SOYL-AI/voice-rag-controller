"""
test_e3_voi.py — the E3 VoI estimator's contract.

These tests exist to protect three properties that E3's headline depends on,
each of which failed silently at least once while this was being built:

  1. VoI is MONOTONE INCREASING in the uncertainty signal. The whole premise
     is "unsure model -> retrieval worth more". A below-chance fit produces
     the opposite sign and nothing downstream notices.
  2. VoI VARIES across scenarios. A constant VoI makes Ours and B9 agree on
     every turn, which is a degenerate experiment, not a null result.
  3. Ours and B9 use the SAME estimator. Any drift there confounds the
     joint-vs-decomposed comparison the paper is about.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baseline import Ctx
from controller.baselines._objective import (
    VoiCalibration,
    estimate_voi,
    load_e3_config,
    load_voi_calibration,
    voi_estimate,
)

# The committed E1 run every calibration in this repo is fitted on — pinned
# here so the several tests that reopen it cannot drift onto different files.
E1_RESULT_RELPATH = (
    "llm/experiment1/results/experiment_1_modal_mistral7b_20260811_151212.json"
)


def make_ctx(signals):
    return Ctx(
        transcript="what time does the pool close",
        signals=signals,
        elapsed_ms=0,
        user_end_known=False,
        turn_id=1,
        conversation_id="conv_voi_test",
        prices={},
    )


def active_signal():
    """
    The uncertainty signal the configured calibration is fitted on.

    Read from the calibration rather than hardcoded, because WHICH signal
    E3 grounds on is a config decision (controller.e3.voi.calibration.signal
    plus the matching artifact — see _objective.load_voi_calibration()), and
    it has already changed once: max_entropy_nats through 2026-08-16, then
    mean_entropy_nats. A test that pins the name tests the choice; these
    tests are about the estimator's PROPERTIES, which must hold for whichever
    signal is wired in. The one test that does assert the config and the
    artifact name the same signal is the cross-check test below, which is
    where that belongs.
    """
    return load_voi_calibration(load_e3_config()).signal


def entropy_ctx(nats):
    return make_ctx(
        {"model": {"available": True, "uncertainty": {active_signal(): nats}}}
    )


def responsive_nats(cfg=None):
    """
    Signal values spanning the fitted sigmoid's responsive range, in units
    of the fit itself: boundary + k/slope for k in -2..2, i.e. P sweeping
    roughly 0.12 -> 0.88.

    Scale-free on purpose. The fits are steep and their boundaries sit in
    quite different places (max_entropy: slope 11.77, P=0.5 at 1.360 nats;
    mean_entropy: slope 11.29, P=0.5 at 0.322), so a literal list of nats
    that exercises one saturates flat on the other — which is exactly how
    these tests broke when the signal was swapped.
    """
    cal = load_voi_calibration(cfg if cfg is not None else load_e3_config())
    boundary = -cal.intercept / cal.slope
    return [boundary + k / cal.slope for k in (-2, -1, 0, 1, 2)]


# ---------------------------------------------------------------------------
# 1. Monotonicity and range
# ---------------------------------------------------------------------------
def test_voi_is_monotone_increasing_in_entropy():
    cfg = load_e3_config()
    values = [estimate_voi(entropy_ctx(h), cfg) for h in responsive_nats(cfg)]
    assert values == sorted(values), values
    # And strictly increasing once clear of the floor, i.e. it actually
    # varies rather than saturating immediately.
    above_floor = [v for v in values if v > cfg["voi"]["floor"]]
    assert len(set(above_floor)) == len(above_floor)


def test_voi_stays_on_unit_interval_and_respects_floor():
    cfg = load_e3_config()
    floor = cfg["voi"]["floor"]
    for h in (-5.0, 0.0, 1.0, 50.0):
        voi = estimate_voi(entropy_ctx(h), cfg)
        assert floor <= voi <= 1.0, (h, voi)


def test_active_calibration_is_the_fitted_e1_logistic():
    """The configured calibration must be the real E1 fit, not the interim
    normalizer it replaced and not a hand-set block: kind logistic, positive
    slope, calibrated, and carrying the AUC of the E1 signal it was fitted
    on."""
    import json

    cfg = load_e3_config()
    cal = load_voi_calibration(cfg)
    assert cal.kind == "logistic"
    assert cal.calibrated is True
    assert cal.slope > 0
    assert "calibrated" in cal.describe()
    assert "INTERIM" not in cal.describe()
    # And it must be the Modal run, not the mock smoke file.
    assert "Mistral-7B" in cal.source

    # The recorded AUC must be this signal's AUC on E1's own samples, not a
    # number carried over from a different signal's fit — the way a
    # re-grounding goes wrong quietly.
    with open(_REPO_ROOT / E1_RESULT_RELPATH, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    samples = result["samples"]
    labels = [1 if s["retrieval_needed"] else 0 for s in samples]
    raw = [s["signals"][cal.signal] for s in samples]
    from llm.experiment1.run import roc_auc

    assert cal.auc == pytest.approx(roc_auc(labels, raw), abs=5e-4)
    # ... and that signal must be one E1's own gate cleared.
    assert (
        result["aggregate_auc"][cal.signal]["mean_auc"]
        >= result["thresholds"]["success_auc"]
    )


def test_config_signal_and_artifact_signal_must_agree():
    """
    Config names the signal it wants; the artifact names the signal it was
    fitted on. Applying one signal's coefficients to another signal's values
    miscalibrates every VoI with no exception and no odd-looking log line,
    so the mismatch has to raise.
    """
    cfg = load_e3_config()
    live = load_voi_calibration(cfg)
    assert cfg["voi"]["calibration"]["signal"] == live.signal

    mismatched = dict(cfg)
    mismatched["voi"] = dict(cfg["voi"])
    mismatched["voi"]["calibration"] = dict(cfg["voi"]["calibration"])
    mismatched["voi"]["calibration"]["signal"] = "one_minus_mean_confidence"
    with pytest.raises(ValueError, match="calibration mismatch"):
        load_voi_calibration(mismatched)


def test_committed_calibration_matches_a_fresh_fit_on_the_e1_result():
    """
    controller/baselines/e1_calibration.json is generated, not written by
    hand, so it can silently fall out of step with the E1 result it cites —
    someone edits a coefficient, or re-points from_artifact at a different
    run. Re-fitting from the committed E1 samples here is cheap (180 rows,
    ~8 Newton steps) and makes that drift a test failure rather than a
    footnote nobody re-derives.
    """
    from llm.experiment1.fit_calibration import build_calibration

    live = load_voi_calibration(load_e3_config())
    # Refit the SIGNAL that is actually wired in, so this stays a drift check
    # on the live calibration rather than on whichever signal happened to be
    # the default when it was written.
    fresh = build_calibration(signal=live.signal)["calibration"]
    assert live.slope == pytest.approx(fresh["slope"], rel=1e-9)
    assert live.intercept == pytest.approx(fresh["intercept"], rel=1e-9)
    assert live.signal == fresh["signal"]
    assert live.kind == fresh["kind"]


def test_fitted_sigmoid_recovers_e1_auc_on_e1_samples():
    """
    The fit's headline sanity check, run against the estimator the
    controllers actually call rather than against the fitting code: scoring
    E1's own 180 samples through estimate_voi() must rank them as well as
    E1's raw signal did. A sigmoid is monotone, so anything else means the
    wrong coefficients, the wrong sign, or the wrong signal is wired in —
    and the floor must not be high enough to flatten the ranking either,
    which is why this goes through estimate_voi() and not through
    VoiCalibration.apply().

    The target AUC is the RAW signal's AUC on those same samples rather than
    a literal, so this stays a recovery check under a re-grounding: 0.9784
    for max_entropy_nats, 0.9260 for mean_entropy_nats.
    """
    import json

    from llm.experiment1.run import roc_auc

    with open(_REPO_ROOT / E1_RESULT_RELPATH, "r", encoding="utf-8") as handle:
        samples = json.load(handle)["samples"]

    cfg = load_e3_config()
    signal = active_signal()
    labels = [1 if s["retrieval_needed"] else 0 for s in samples]
    raw = [s["signals"][signal] for s in samples]
    scores = [estimate_voi(entropy_ctx(value), cfg) for value in raw]
    assert len(samples) == 180
    assert roc_auc(labels, scores) == pytest.approx(roc_auc(labels, raw), abs=5e-4)


# ---------------------------------------------------------------------------
# 2. The sign guard
# ---------------------------------------------------------------------------
def test_calibrated_fit_with_negative_slope_is_refused():
    """The exact failure mode found in the mock E1 data: a below-chance fit
    yields slope < 0, which inverts uncertainty -> value."""
    cfg = load_e3_config()
    cfg["voi"] = dict(cfg["voi"])
    cfg["voi"]["calibration"] = {
        "signal": "max_entropy_nats",
        "kind": "logistic",
        "intercept": 2.3964,
        "slope": -1.5144,   # the mock fit
        "calibrated": True,
        "source": "mock smoke run",
        "auc": 0.4453,
    }
    with pytest.raises(ValueError, match="inverts the uncertainty"):
        load_voi_calibration(cfg)


def test_calibrated_fit_with_positive_slope_loads():
    cfg = load_e3_config()
    cfg["voi"] = dict(cfg["voi"])
    cfg["voi"]["calibration"] = {
        "signal": "max_entropy_nats",
        "kind": "logistic",
        "intercept": -2.0,
        "slope": 1.5,
        "calibrated": True,
        "source": "hypothetical valid E1 run",
        "auc": 0.9,
    }
    cal = load_voi_calibration(cfg)
    assert cal.calibrated and cal.slope > 0
    # Still monotone increasing through the logistic.
    assert cal.apply(0.5) < cal.apply(1.5)


def test_missing_artifact_fails_loudly_rather_than_falling_back():
    """A typo'd from_artifact path must not silently degrade to the interim
    mapping — that would publish an uncalibrated run as a calibrated one."""
    cfg = load_e3_config()
    cfg["voi"] = dict(cfg["voi"])
    cfg["voi"]["calibration"] = {"from_artifact": "llm/experiment1/results/nope.json"}
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_voi_calibration(cfg)


# ---------------------------------------------------------------------------
# 3. Signal ladder
# ---------------------------------------------------------------------------
def test_tracker_is_fallback_only_not_blended():
    """With the calibrated signal present, the tracker must not move the
    answer — it is no longer part of the estimate."""
    cfg = load_e3_config()
    nats = responsive_nats(cfg)[2]  # the fit's own P = 0.5 point
    with_tracker = make_ctx({
        "model": {"uncertainty": {active_signal(): nats}},
        "tracker": {"unknown_fraction": 0.0},
    })
    without = entropy_ctx(nats)
    assert estimate_voi(with_tracker, cfg) == pytest.approx(estimate_voi(without, cfg))


def test_tracker_used_when_model_signal_absent():
    cfg = load_e3_config()
    ctx = make_ctx({"tracker": {"unknown_fraction": 0.42}})
    assert estimate_voi(ctx, cfg) == pytest.approx(0.42)


def test_no_signal_at_all_returns_high_voi():
    """'We don't know' must read as 'retrieve rather than guess', matching
    controller/decide.py's live rule."""
    assert estimate_voi(make_ctx({}), load_e3_config()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Ours and B9 share one estimator
# ---------------------------------------------------------------------------
def test_ours_and_b9_import_the_same_estimator():
    from controller.baselines import b9_decomposed, ours

    assert ours.voi_estimate is b9_decomposed.voi_estimate is estimate_voi
    assert voi_estimate is estimate_voi


def test_voi_varies_across_distinct_uncertainty_inputs():
    """
    Guards the degenerate case: a constant VoI would make Ours and B9 agree
    on every scenario and there would be no frontier to measure.

    Sampled inside the fitted sigmoid's responsive range, in units of the
    fit itself (see responsive_nats()). That range is NARROW — both E1 fits
    are steep, slope ~11 per nat — so a literal list of nats picked for one
    signal saturates flat on another. That steepness is a property of the
    real fits, not a weakness of this test, and it is why
    simulator/e3_grid.py derives the price grid from the VoI distribution
    rather than from the signal's range.

    What this does NOT check is whether the SCENARIOS spread out over that
    responsive range — a signal can be responsive in principle and still put
    every real scenario at one end, which is what max_entropy_nats did (see
    _objective.load_voi_calibration()'s docstring). That is checked against
    the real 50 by simulator/e3_divergence_check.py, which needs the frozen
    signals cache and so does not belong in a unit test.
    """
    cfg = load_e3_config()
    values = {estimate_voi(entropy_ctx(h), cfg) for h in responsive_nats(cfg)}
    assert len(values) == 5


def test_utility_scale_scales_the_quality_term():
    """voi = P(retrieval needed) * utility_scale. At the configured 1.0 the
    two are the same number; this pins that the scale is actually applied,
    and applied to the tracker fallback too, so a changed scale cannot land
    on one branch of the ladder and not another."""
    cfg = load_e3_config()
    assert cfg["voi"]["utility_scale"] == 1.0

    scaled = dict(cfg)
    scaled["voi"] = {**cfg["voi"], "utility_scale": 0.5, "floor": 0.0}
    base = dict(cfg)
    base["voi"] = {**cfg["voi"], "floor": 0.0}

    assert estimate_voi(entropy_ctx(1.5), scaled) == pytest.approx(
        0.5 * estimate_voi(entropy_ctx(1.5), base)
    )
    tracker = make_ctx({"tracker": {"unknown_fraction": 0.8}})
    assert estimate_voi(tracker, scaled) == pytest.approx(0.4)
