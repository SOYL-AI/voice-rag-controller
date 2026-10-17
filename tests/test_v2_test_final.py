"""test_v2_test_final.py -- the pooled stratified cluster bootstrap and the
decision rules of analysis/v2_test_final.py on synthetic contrasts."""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis import v2_test_final as F  # noqa: E402


def _ds(name, n, shift_c, shift_s, seed, pairs=False):
    rng = np.random.default_rng(seed)
    ids = [f"{name}-{i}" for i in range(n)]
    lam = F.PRIMARY["lambda_l"][name]
    grid = {"rewrite_faiss": [{"quantile": 0.65, "lambda_l": lam, "lambda_c": 0.05}]}
    c = {i: float(shift_c + rng.normal(0, 0.1)) for i in ids}
    s = {i: float(shift_s + rng.normal(0, 0.1)) for i in ids}
    row = {"latency_profile": "rewrite_faiss", "price": f"{lam:g}|0.05",
           "d_star": {"commit": "b9plus_v2", "speculative": F.D_STAR_SPEC[name]},
           "per_item": {"delta_commit": c, "delta_speculative": s, "interaction": {i: c[i] - s[i] for i in ids}}}
    cluster = {i: (f"p{k // 2}" if pairs else i) for k, i in enumerate(ids)}
    return F.Dataset(name, {F.ASR: [row]}, {F.ASR: {"price_grid": grid}}, cluster, seed, n_boot=2000)


def test_pooled_estimate_is_the_test_weighted_mean_and_rules_follow_the_ci():
    d1 = _ds("d1_open", 400, -0.04, -0.045, 1)
    d2 = _ds("d2_support", 60, -0.01, -0.02, 2, pairs=True)
    out = F.analyse({"d1_open": d1, "d2_support": d2}, {"cells": []})
    w1, w2 = 3734 / 4232, 498 / 4232
    for k in F.KINDS:
        want = w1 * out["per_dataset"]["d1_open"]["primary"][k]["estimate"] + \
            w2 * out["per_dataset"]["d2_support"]["primary"][k]["estimate"]
        assert out["pooled"][k]["estimate"] == pytest.approx(want)
    t = out["pooled"]["confirmatory_tests"]
    assert t["H-REV"]["reject"] and not t["P1"]["reject"] and t["P1"]["p"] > 0.5
    assert not t["P3"]["pass"] and t["P3"]["reject_upper"] and not t["P3"]["reject_lower"]
    assert out["pooled"]["IUT_boundary_claim"] is False
    assert out["pooled"]["holm_individual_claims"]["H-REV"]["reject"]
    assert d2.n_clusters == 30  # pairs resampled together
    assert len(out["secondary_family"]) == 6


@pytest.mark.parametrize("reason", [None, "D3: kappa_human < 0.70"])
def test_status_tagging_keeps_d1_confirmatory(reason, tmp_path, monkeypatch):
    d1 = _ds("d1_open", 200, -0.04, -0.045, 1)
    d2 = _ds("d2_support", 40, -0.01, -0.02, 2, pairs=True)
    doc = F.analyse({"d1_open": d1, "d2_support": d2}, {"cells": []})
    doc.update({"e1_gate": {"d1_open": {}, "d2_support": {}}, "judge_checks": {"status_note": "pending"}})
    F.apply_status(doc, reason)
    assert doc["per_dataset"]["d1_open"]["status"] == "confirmatory"
    expl = "confirmatory" if reason is None else f"exploratory ({reason})"
    assert doc["per_dataset"]["d2_support"]["status"] == expl
    assert doc["pooled"]["status"] == expl and doc["secondary_family_status"] == expl
    assert doc["e1_gate"]["kill_rule_status"] == expl and doc["e1_gate"]["d1_open"]["status"] == "confirmatory"
    assert all(c["status"] == "descriptive" for c in doc["cells"] if c["dataset"] == "d1_open")
    assert all(c["status"] == ("descriptive" if reason is None else f"descriptive; exploratory ({reason})")
               for c in doc["cells"] if c["dataset"] == "d2_support")
    monkeypatch.setattr(F, "OUT_MD", tmp_path / "r.md")
    F.write_md(doc)
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    if reason is None:
        assert "EXPLORATORY" not in md
    else:
        assert md.index("## Confirmatory: d1_open") < md.index("\n# EXPLORATORY (D3")
        assert "*[EXPLORATORY (D3: KAPPA_HUMAN < 0.70)]*" in md


def test_holm_and_two_sided_p():
    h = F.holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05)
    assert h["a"]["reject"] and h["b"]["reject"] and h["c"]["reject"]
    h = F.holm({"a": 0.001, "b": 0.04, "c": 0.03}, 0.05)
    assert h["a"]["reject"] and not h["c"]["reject"] and not h["b"]["reject"]
    assert F.two_sided_p(np.array([-1.0, 1.0, 2.0, 3.0])) == pytest.approx(0.5)
