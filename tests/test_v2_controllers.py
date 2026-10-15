"""
test_v2_controllers.py — component tests for the v2 controller stack:
the shared cost model, the unpriced baselines (incl. the B4 fix), the
oracle, Phase B, the tuning harness and the belief-transition fit. All on
synthetic data; no model, ASR or judge call.
"""

import json
import math
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.e3_frontier import validate_records, validate_v2_records, select_v2_stratum
from controller.baselines._objective import Prices
from controller.baselines._objective_v2 import (
    BeliefTransitions,
    belief_cut,
    estimated_frac,
    fit_belief_transitions,
    latency_dist_from_quantiles,
    load_belief_calibration,
    load_e3v2_config,
    parse_quantiles,
    quality_model,
    rho_at,
    utility_retrieve,
    waste_at,
)
from controller.baselines.unpriced_v2 import classifier_score, min_token_prob, train_adaptive_rag_classifier
from controller.fit_belief_transitions import belief_sequences
from controller.sequential_v2 import build_steps, get_controller_v2, list_controllers_v2
from controller.tune_v2 import fit_rho_table, quantile_grid, tune_all
from simulator.e3v2_decide import (
    CellStore,
    check_split,
    load_latency_model,
    query_variant_for,
    run_phase_b,
    with_profile,
)
from simulator.e3v2_synthetic import synthetic_cfg_overrides, synthetic_dataset, synthetic_turn


def syn_cfg(**kw):
    return load_e3v2_config(overrides=synthetic_cfg_overrides(**kw))


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def test_quantile_parsing_accepts_common_spellings():
    assert parse_quantiles({"p50": 1, "q90": 2, "10": 3, "0.25": 4, "mean": 9}) == [
        (0.1, 3.0), (0.25, 4.0), (0.5, 1.0), (0.9, 2.0)]


def test_exposed_latency_is_expected_max_l_minus_window():
    deg = latency_dist_from_quantiles({"p50": 600}, 41, "x")
    assert deg.expected_exposed_ms(0) == 600
    assert deg.expected_exposed_ms(250) == 350
    assert deg.expected_exposed_ms(900) == 0
    spread = latency_dist_from_quantiles({"p10": 300, "p50": 672, "p90": 1500}, 41, "x")
    for w in (0, 200, 700, 3000):
        assert math.isclose(spread.expected_hidden_ms(w) + spread.expected_exposed_ms(w), spread.mean_ms)
    # exposure is non-increasing in the hiding window
    vals = [spread.expected_exposed_ms(w) for w in range(0, 2000, 100)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_hiding_window_includes_end_of_turn_detection():
    cfg = syn_cfg(timing={"eot_detection_ms": 200})
    turn = synthetic_turn("w", [0.5, 0.5], 0.5, interval_ms=300.0, retrieval={
        "latency_profile": "x", "latency_quantiles_ms": {"p50": 1000}})
    steps = build_steps(turn, cfg)
    # partial 0 at 300 ms, user end 900 ms -> R = 600, window = 800
    from controller.baselines._objective_v2 import expected_exposed_ms, latency_dist_for
    lat = latency_dist_for(turn.signals["retrieval"], cfg)
    assert expected_exposed_ms(lat, steps[0].remaining_ms, False, cfg) == 200
    assert steps[-1].is_final and steps[-1].available_ms == 1100  # VAD end + detection delay
    assert expected_exposed_ms(lat, 0.0, True, cfg) == 1000


def test_steps_use_partial_availability_and_trace_endpoint():
    """Real ASR traces: a partial is usable only from available_ms (ASR
    compute), predictions likewise; the end-of-turn step is the later of VAD
    end and the final transcript's availability; speech-elapsed at the end is
    the actual speech end."""
    from controller.sequential_v2 import TurnCtx, end_of_turn_ms, speech_end_ms
    cfg = syn_cfg()
    timing = {
        "speech_onset_ms": 0,
        "partials": [{"t_ms": 400, "available_ms": 700, "text": "what is the late fee"},
                     {"t_ms": 800, "available_ms": 1200, "text": "what is the late fee on my plan"}],
        "predicted_remaining_speech_ms": [{"t_ms": 400, "available_ms": 700, "remaining_ms": 900},
                                          {"t_ms": 800, "available_ms": 1200, "remaining_ms": 500}],
        "speech_end_ms": 1300, "user_end_ms": 2100,
        "asr_final": {"t_ms": 2100, "available_ms": 2600, "text": "what is the late fee on my plan"},
    }
    ctx = TurnCtx(item_id="x", signals={"timing": timing,
                                        "retrieval": {"latency_profile": "x", "latency_quantiles_ms": {"p50": 500}}})
    steps = build_steps(ctx, cfg)
    assert [s.available_ms for s in steps] == [700, 1200, 2600]
    assert steps[0].remaining_ms == pytest.approx(900 - (700 - 400))  # aged from the audio time
    assert steps[-1].speech_elapsed_ms == 1300
    assert end_of_turn_ms(ctx, cfg) == 2600 and speech_end_ms(ctx) == 1300


def test_rho_waste_and_frac():
    cfg = syn_cfg()
    assert estimated_frac(300, 900, False) == 0.25
    assert estimated_frac(300, None, False) is None
    assert estimated_frac(0, 0, True) == 1.0
    assert rho_at(0.25, True, cfg) == pytest.approx(0.5625)
    assert rho_at(0.25, False, cfg) == 1.0  # not query-ready
    assert rho_at(1.0, False, cfg) == 0.0
    tab = syn_cfg(rho={"table": [[0.2, 0.6], [0.6, 0.2], [1.0, 0.0]]})
    assert rho_at(0.4, True, tab) == pytest.approx(0.4)
    assert waste_at(0.5, False, cfg) == pytest.approx(0.05)
    assert waste_at(0.5, True, cfg) == 0.0


def test_objective_is_asymmetric_and_cut_is_consistent():
    q = quality_model(syn_cfg(quality={"g_plus": 0.72, "g_minus": 0.59}))
    p = Prices(1.0, 0.05)
    assert utility_retrieve(0.0, 0.0, 0.0, 0.0, Prices(0, 0), q) == pytest.approx(-0.59)
    assert utility_retrieve(1.0, 0.0, 0.0, 0.0, Prices(0, 0), q) == pytest.approx(0.72)
    cut = belief_cut(0.1, 500.0, 0.02, p, q)
    assert utility_retrieve(cut, 0.1, 500.0, 0.02, p, q) == pytest.approx(0.0, abs=1e-12)


def test_calibration_fallback_and_guards():
    cfg = load_e3v2_config()
    cal = load_belief_calibration(cfg)  # falls back to the committed v1 artifact
    assert cal.signal == "mean_entropy_nats" and cal.calibrated and cal.slope > 0
    assert cal.apply(0.6) > cal.apply(0.1)
    with pytest.raises(ValueError):
        load_belief_calibration(load_e3v2_config(overrides={"calibration": {"signal": "max_entropy_nats"}}))
    with pytest.raises(ValueError):
        load_belief_calibration(load_e3v2_config(overrides={"calibration": {"inline": {
            "signal": "mean_entropy_nats", "kind": "logistic", "slope": -1.0, "calibrated": True}}}))


def test_config_block_mirrors_module_defaults():
    from controller.baselines._objective_v2 import _DEFAULT_E3V2_CFG
    assert load_e3v2_config() == _DEFAULT_E3V2_CFG


def test_controllers_build_against_the_committed_config():
    cfg = load_e3v2_config()
    for name in list_controllers_v2():
        get_controller_v2(name, cfg=cfg)


def test_v1_registry_is_untouched():
    import controller.baselines  # noqa: F401
    from controller.baseline import list_baselines
    names = list_baselines()
    assert {"b0", "b4", "b5", "b6", "b8", "b9", "ours"} <= set(names)
    assert not any(n.endswith("_v2") for n in names)


# ---------------------------------------------------------------------------
# Unpriced baselines
# ---------------------------------------------------------------------------
def test_b4_flare_fix_triggers_on_low_token_probability():
    cfg = syn_cfg()
    b4 = get_controller_v2("b4_v2", cfg=cfg, theta=0.3)
    low = synthetic_turn("lo", [0.5], 0.5, final_extra={"min_token_prob": 0.1})
    high = synthetic_turn("hi", [0.5], 0.5, final_extra={"min_token_prob": 0.6})
    assert b4.decide_turn(low).retrieve and not b4.decide_turn(high).retrieve
    # v1-style signals only (no token probabilities): the documented proxy
    proxy = synthetic_turn("px", [0.5], 0.5, final_extra={"max_entropy_nats": 2.0})
    d = b4.decide_turn(proxy)
    assert d.retrieve and "proxy" in d.reason
    assert min_token_prob({"token_logprobs": [-0.1, -2.0]})[0] == pytest.approx(math.exp(-2.0))
    # and on a whole synthetic dataset it is no longer degenerate (v1: 0%)
    turns, _, _, _ = synthetic_dataset(30)
    rate = sum(b4.decide_turn(t).retrieve for t in turns.values()) / len(turns)
    assert 0.0 < rate < 1.0


def test_b5_classifier_separates_and_b6_threshold():
    texts = ["what is the fee on my northwind plan", "what is my northwind roaming charge",
             "who wrote hamlet", "what is the capital of france"]
    model = train_adaptive_rag_classifier(texts, [True, True, False, False])
    assert classifier_score(model, "northwind plan fee") > 0.5 > classifier_score(model, "capital of france")
    cfg = syn_cfg()
    b6 = get_controller_v2("b6_v2", cfg=cfg, threshold=1.0)
    t = synthetic_turn("e", [0.5], 0.5, final_extra={"max_entropy_nats": 1.4})
    d = b6.decide_turn(t)
    assert d.retrieve and d.timing_label == "sentence_end"


def test_b1_policies_dispatch_in_order():
    cfg = syn_cfg()
    turn = synthetic_turn("b1", [0.5] * 8, 0.5, interval_ms=300.0,
                          retrieval={"latency_profile": "x", "latency_quantiles_ms": {"p50": 700}})
    times = {p: get_controller_v2(f"b1_v2_{p}", cfg=cfg).decide_turn(turn).dispatch_t_ms
             for p in ("immediate", "fixed_early_proxy", "forecast", "sentence_end")}
    assert times["immediate"] <= times["fixed_early_proxy"] <= times["sentence_end"]
    assert times["immediate"] <= times["forecast"] <= times["sentence_end"]
    assert times["immediate"] == 300 and times["sentence_end"] == 2700


def test_oracle_dispatches_at_earliest_gold_hit():
    cfg = syn_cfg()
    oracle = get_controller_v2("oracle_v2", cfg=cfg)
    t = synthetic_turn("o", [0.5] * 4, 0.5, oracle={"closed_book_wrong": True,
                                                    "gold_hit_partial_indices": [2, 3]})
    d = oracle.decide_turn(t)
    assert d.dispatch_partial_index == 2 and "ORACLE" in d.reason
    t2 = synthetic_turn("o2", [0.5] * 4, 0.5, oracle={"closed_book_wrong": False})
    assert not oracle.decide_turn(t2).retrieve
    t3 = synthetic_turn("o3", [0.5] * 4, 0.5, oracle={"closed_book_wrong": True,
                                                      "gold_hit_partial_indices": []})
    assert oracle.decide_turn(t3).timing_label == "sentence_end"


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------
def test_phase_b_records_validate_and_share_outcomes():
    turns, cells, labels, items = synthetic_dataset(16)
    cfg = syn_cfg()
    store = CellStore(cells)
    gold = {it["id"]: it["gold_doc_ids"] for it in items}
    tl = list(turns.values())
    ctrls = {n: get_controller_v2(n, cfg=cfg) for n in
             ("ours_v2", "b9_v2", "b9plus_v2", "b0_v2", "b1_v2_sentence_end", "b4_v2", "oracle_v2")}
    prices = [{"lambda_l": 0.3, "lambda_c": 0.05}, {"lambda_l": 2.0, "lambda_c": 0.05}]
    recs = run_phase_b(tl, store, ctrls, prices, [11, 23], ["faiss", "remote_api"], cfg,
                       load_latency_model(cfg), "dev", labels, gold)
    validate_records(recs)
    validate_v2_records(recs)
    with pytest.raises(ValueError):
        select_v2_stratum(recs)  # two profiles must not be pooled
    assert len(select_v2_stratum(recs, "faiss")) == len(recs) // 2
    # common random numbers: same (item, seed, profile, path, dispatch) -> same TTFA
    by_key = {}
    for r in recs:
        key = (r["item_id"], r["seed"], r["latency_profile"], r["dispatch_t_ms"], r["query_variant"])
        by_key.setdefault(key, set()).add(round(r["outcome"]["ttfa_ms"], 6))
    assert all(len(v) == 1 for v in by_key.values())
    # skip records have zero exposure; retrieval records carry a variant
    for r in recs:
        if r["outcome"]["retrievals"] == 0:
            assert r["exposed_latency_ms"] == 0 and r["query_variant"] == "none"
        else:
            assert r["query_variant"].startswith("qv-")
    assert {r["baseline"] for r in recs if r["prices"] is None} == {"b0_v2", "b1_v2_sentence_end", "b4_v2", "oracle_v2"}


def test_phase_b_refuses_unmeasured_cells_and_sealed_test():
    turns, cells, labels, items = synthetic_dataset(3)
    cfg = syn_cfg()
    store = CellStore([c for c in cells if not (c["kind"] == "cell" and c["path"] == "skip")])
    with pytest.raises(KeyError):
        run_phase_b(list(turns.values()), store, {"b0_v2": get_controller_v2("b0_v2", cfg=cfg)},
                    [None], [11], ["faiss"], cfg, load_latency_model(cfg))
    with pytest.raises(SystemExit):
        check_split("test")
    check_split("dev")


def test_query_variant_order_sensitivity():
    assert query_variant_for(["a", "b"]) != query_variant_for(["b", "a"])
    assert query_variant_for(["a", "b"], False) == query_variant_for(["b", "a"], False)


# ---------------------------------------------------------------------------
# Tuning harness and transition fit
# ---------------------------------------------------------------------------
def test_tune_v2_equal_budget_and_artifact_roundtrip(tmp_path):
    turns, cells, labels, items = synthetic_dataset(30, seed=4)
    cfg = syn_cfg()
    store = CellStore(cells)
    gold = {it["id"]: it["gold_doc_ids"] for it in items}
    art = tune_all(list(turns.values()), store, [11, 23], ["faiss", "remote_api"], cfg, labels, gold, budget=9)
    for key in ("b4", "b5", "b6"):
        assert art["unpriced"][key]["n_evaluated"] <= 9
        assert len(art["unpriced"][key]["scores"]) == art["unpriced"][key]["n_evaluated"]
        assert art["unpriced"][key]["chosen"]
    assert art["unpriced"]["b8"]["chosen"]["trigger"] in ("b4", "b5", "b6")
    assert set(art["policy_constants"]["profiles"]) == {"faiss", "remote_api"}
    assert art["rho_table"][-1] == [1.0, 0.0]
    ys = [y for _, y in art["rho_table"]]
    assert all(a >= b - 1e-12 for a, b in zip(ys, ys[1:]))  # non-increasing in f
    assert art["empirical_quality"]["g_plus"] > 0
    path = tmp_path / "est.json"
    path.write_text(json.dumps(art), encoding="utf-8")
    cfg2 = load_e3v2_config(overrides={**synthetic_cfg_overrides(), "estimates_artifact": str(path)})
    assert cfg2["rho"]["table"] == art["rho_table"]
    b6 = get_controller_v2("b6_v2", cfg=cfg2)
    assert b6.trigger_params["threshold"] == art["unpriced"]["b6"]["chosen"]["threshold"]
    b9p = get_controller_v2("b9plus_v2", cfg=cfg2)
    assert b9p.constants_source == "estimates_artifact"
    b8 = get_controller_v2("b8_v2", cfg=cfg2)
    assert b8.policy == art["unpriced"]["b8"]["chosen"]["timing"]


def test_quantile_grid():
    g = quantile_grid([1, 2, 3, 4, 5], 5)
    assert g == [1, 2, 3, 4, 5]
    assert quantile_grid([1, 1, 1], 7) == [1.0]
    assert quantile_grid([], 5) == []


def test_fit_belief_transitions_from_e1v2_records():
    rng = random.Random(9)
    records = []
    for i in range(40):
        need = i % 2 == 0
        for seed in (11, 23):
            for k, frac in enumerate((0.3, 0.6, 1.0)):
                h = (0.25 + (0.25 if need else -0.15) * frac) + rng.gauss(0, 0.05)
                records.append({"item_id": f"it-{i}", "model": "m", "draw_seed": seed,
                                "prefix": {"kind": "full" if frac == 1.0 else "partial", "source": "asr",
                                           "voice_id": "v", "t_ms": 1000 * (k + 1), "word_frac": frac,
                                           "text": "..."},
                                "signals": {"mean_entropy_nats": h}})
    cal = load_belief_calibration(load_e3v2_config())
    seqs = belief_sequences(records, cal, keep={f"it-{i}" for i in range(30)})
    assert len(seqs) == 60 and all(len(s) == 3 for s in seqs)
    # records from any model other than the project's single model are dropped
    assert belief_sequences(records, cal, keep={f"it-{i}" for i in range(30)},
                            model="mistralai/Mistral-7B-Instruct-v0.3") == []
    model = fit_belief_transitions(seqs, n_bins=5)
    for table in (model.next_table, model.final_table):
        for row in (r for ph in table.values() for r in ph.values()):
            assert sum(row["probs"]) == pytest.approx(1.0)
    rt = BeliefTransitions.from_json(json.loads(json.dumps(model.to_json())))
    assert rt.next_dist(0.5, 0.3) == model.next_dist(0.5, 0.3)
    assert BeliefTransitions.identity().next_dist(0.37, 0.5) == [(1.0, 0.37)]


def test_fit_rho_table_from_cells():
    turns, cells, labels, items = synthetic_dataset(30, seed=6)
    gold = {it["id"]: it["gold_doc_ids"] for it in items}
    table = fit_rho_table(list(turns.values()), CellStore(cells), gold, labels)
    assert table is not None and table[0][1] >= table[-2][1]
