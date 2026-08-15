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

import math
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


def test_interim_normalizer_is_the_structural_entropy_ceiling():
    """voi = H / ln(top_k + 1), so a maximum-entropy token maps to 1.0."""
    cal = load_voi_calibration(load_e3_config())
    assert cal.kind == "normalized"
    assert cal.normalizer == pytest.approx(math.log(6), rel=1e-9)
    assert estimate_voi(entropy_ctx(math.log(6)), load_e3_config()) == pytest.approx(1.0)


def test_interim_calibration_is_not_labelled_as_calibrated():
    """The paper must not be able to describe the interim mapping as an E1
    fit. If someone flips this to True, they must also supply real
    coefficients — which the sign guard then checks."""
    cal = load_voi_calibration(load_e3_config())
    assert cal.calibrated is False
    assert "INTERIM" in cal.source.upper()
    assert "INTERIM-UNCALIBRATED" in cal.describe()


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
    """Guards the degenerate case: a constant VoI would make Ours and B9
    agree on every scenario and there would be no frontier to measure."""
    cfg = load_e3_config()
    values = {estimate_voi(entropy_ctx(h), cfg) for h in (0.9, 1.1, 1.3, 1.5, 1.7)}
    assert len(values) == 5
