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


def entropy_ctx(nats):
    return make_ctx({"model": {"available": True, "uncertainty": {"max_entropy_nats": nats}}})


# ---------------------------------------------------------------------------
# 1. Monotonicity and range
# ---------------------------------------------------------------------------
def test_voi_is_monotone_increasing_in_entropy():
    cfg = load_e3_config()
    values = [estimate_voi(entropy_ctx(h), cfg) for h in (0.0, 0.4, 0.8, 1.2, 1.6, 1.79)]
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
    slope, calibrated, and carrying E1's AUC."""
    cal = load_voi_calibration(load_e3_config())
    assert cal.kind == "logistic"
    assert cal.signal == "max_entropy_nats"
    assert cal.calibrated is True
    assert cal.slope > 0
    assert cal.auc == pytest.approx(0.9784, abs=5e-4)
    assert "calibrated" in cal.describe()
    assert "INTERIM" not in cal.describe()
    # And it must be the Modal run, not the mock smoke file.
    assert "Mistral-7B" in cal.source


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

    fresh = build_calibration()["calibration"]
    live = load_voi_calibration(load_e3_config())
    assert live.slope == pytest.approx(fresh["slope"], rel=1e-9)
    assert live.intercept == pytest.approx(fresh["intercept"], rel=1e-9)
    assert live.signal == fresh["signal"]
    assert live.kind == fresh["kind"]


def test_fitted_sigmoid_recovers_e1_auc_on_e1_samples():
    """
    The fit's headline sanity check, run against the estimator the
    controllers actually call rather than against the fitting code: scoring
    E1's own 180 samples through estimate_voi() must rank them as well as
    E1's raw signal did (AUC 0.978). A sigmoid is monotone, so anything
    else means the wrong coefficients, the wrong sign, or the wrong signal
    is wired in — and the floor must not be high enough to flatten the
    ranking either, which is why this goes through estimate_voi() and not
    through VoiCalibration.apply().
    """
    import json

    from llm.experiment1.run import roc_auc

    result_path = (
        _REPO_ROOT
        / "llm"
        / "experiment1"
        / "results"
        / "experiment_1_modal_mistral7b_20260811_151212.json"
    )
    with open(result_path, "r", encoding="utf-8") as handle:
        samples = json.load(handle)["samples"]

    cfg = load_e3_config()
    labels = [1 if s["retrieval_needed"] else 0 for s in samples]
    scores = [
        estimate_voi(entropy_ctx(s["signals"]["max_entropy_nats"]), cfg)
        for s in samples
    ]
    assert len(samples) == 180
    assert roc_auc(labels, scores) == pytest.approx(0.9784, abs=5e-4)


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
    with_tracker = make_ctx({
        "model": {"uncertainty": {"max_entropy_nats": 1.2}},
        "tracker": {"unknown_fraction": 0.0},
    })
    without = entropy_ctx(1.2)
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

    Sampled inside the fitted sigmoid's responsive range. That range is
    NARROW — the E1 fit is steep (slope 11.77, P = 0.5 at 1.360 nats), so
    everything below ~1.15 nats is pinned to the floor and everything above
    ~1.6 is within 0.02 of 1.0. That is a property of the real fit, not a
    weakness of this test, and it is why simulator/e3_grid.py derives the
    price grid from the VoI distribution rather than from the signal's
    range.
    """
    cfg = load_e3_config()
    values = {estimate_voi(entropy_ctx(h), cfg) for h in (1.2, 1.3, 1.4, 1.5, 1.6)}
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
