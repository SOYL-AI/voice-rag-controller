"""
test_v2_lazy.py — lazy outcome evaluation (PREREG_V2 §4.8) equals full
evaluation. A recording pass over the evaluation pipeline lists the cells
some controller lands on; evaluating on a store holding ONLY those cells
reproduces every record of the full evaluation exactly.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import load_e3v2_config
from harness.latency_model_v2_stub import LatencyModel
from simulator.e3v2_decide import CellStore, run_architectures
from simulator.e3v2_synthetic import synthetic_cfg_overrides, synthetic_dataset

NAMES = ["ours_v2", "b9_v2", "b9plus_v2", "b10_speculative_v2", "b11_speculative_gated_v2",
         "b0_v2", "b1_v2_immediate", "b1_v2_sentence_end", "b4_v2", "b6_v2", "oracle_v2"]
PRICES = [{"lambda_l": 0.3, "lambda_c": 0.05}, {"lambda_l": 1.2, "lambda_c": 0.05}]


def _run(turns, store, cfg, labels, gold):
    return run_architectures(list(turns.values()), store, NAMES, PRICES, [11, 23], ["faiss", "remote_api"], cfg,
                             LatencyModel(cfg=cfg), ("commit", "speculative"), "dev", labels, gold)


def _key(r):
    return (r["architecture"], r["baseline"], str(r["prices"]), r["latency_profile"], r["item_id"], r["seed"])


def test_lazy_equals_full():
    turns, cells, labels, items = synthetic_dataset(24, seed=12)
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(nonhideable_increment_ms=300.0))
    gold = {it["id"]: it["gold_doc_ids"] for it in items}
    full = _run(turns, CellStore(cells), cfg, labels, gold)

    record = set()
    retrieval_rows = [c for c in cells if c["kind"] == "retrieval"]
    plan_store = CellStore(retrieval_rows, recording=record)  # no outcome cells at all
    _run(turns, plan_store, cfg, labels, gold)

    all_cells = {(c["item_id"], c["condition"], c["path"], c["query_variant"]) for c in cells if c["kind"] == "cell"}
    assert record <= all_cells and len(record) < len(all_cells)  # strictly fewer cells to generate/grade

    lazy_rows = retrieval_rows + [c for c in cells if c["kind"] == "cell"
                                  and (c["item_id"], c["condition"], c["path"], c["query_variant"]) in record]
    lazy = _run(turns, CellStore(lazy_rows), cfg, labels, gold)

    assert len(lazy) == len(full)
    by_key = {_key(r): r for r in full}
    for r in lazy:
        f = by_key[_key(r)]
        assert r["outcome"] == f["outcome"]
        assert r["realized_utility"] == pytest.approx(f["realized_utility"])
        assert (r["dispatch_t_ms"], r["query_variant"], r["used"]) == (f["dispatch_t_ms"], f["query_variant"], f["used"])


def test_lazy_store_refuses_an_unplanned_cell():
    turns, cells, labels, items = synthetic_dataset(6, seed=2)
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides())
    store = CellStore([c for c in cells if c["kind"] == "retrieval"])
    with pytest.raises(KeyError):
        _run(turns, store, cfg, labels, {it["id"]: it["gold_doc_ids"] for it in items})
