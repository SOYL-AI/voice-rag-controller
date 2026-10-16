"""
test_v2_deviation_d1.py — PREREG_V2 Deviation D1 (2026-10-16). Phase A's E1
closed-book reuse no longer requires labels_<split>.json: without it no
verdicts are attached, and the reused skip cells stay unresolved (excluded
from the store) until the store is rebuilt after grading. Controller
decisions and the lazy cell plan do not depend on verdicts, so they are
identical with and without them.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import load_e3v2_config  # noqa: E402
from harness.latency_model_v2_stub import LatencyModel  # noqa: E402
from simulator.e3v2_decide import CellStore, run_architectures  # noqa: E402
from simulator.e3v2_phase_a import PhaseA  # noqa: E402
from simulator.e3v2_synthetic import synthetic_cfg_overrides, synthetic_dataset  # noqa: E402

NAMES = ["ours_v2", "b9_v2", "b9plus_v2", "b10_speculative_v2", "b11_speculative_gated_v2",
         "b0_v2", "b1_v2_immediate", "b1_v2_sentence_end", "b4_v2", "b6_v2"]
PRICES = [{"lambda_l": 0.3, "lambda_c": 0.05}, {"lambda_l": 1.2, "lambda_c": 0.05}]


def _signals(tmp_path, split):
    rows = [{"item_id": "i1", "draw_seed": s, "prefix": {"kind": "full", "source": "text"},
             "closed_book_answer": f"answer {s}", "finish_reason": "stop", "n_tokens": 5, "call_key": f"k{s}"}
            for s in (11, 23)]
    with open(tmp_path / f"signals_{split}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_e1_reuse_without_labels_attaches_no_verdicts(tmp_path):
    _signals(tmp_path, "test")
    fake = SimpleNamespace(e1dir=tmp_path, args=SimpleNamespace(split="test"), by_id={"i1": {}})
    out = PhaseA.e1_closed_book(fake)  # must not raise without labels_test.json
    assert set(out) == {("i1", 11), ("i1", 23)}
    assert all("alias_correct" not in v and "primary_k1" not in v for v in out.values())
    # a reused skip cell without a verdict is unresolved under a judge source, as before grading
    assert PhaseA._correct("judge:primary", out[("i1", 11)].get("alias_correct"),
                           out[("i1", 11)].get("primary_k1")) == (None, "unresolved")
    # with labels the verdicts are attached exactly as before
    labels = {"draws": [{"item_id": "i1", "draw_seed": 11, "condition": "closed_book", "alias_correct": True,
                         "judges": {"primary": {"samples": [{"sample_index": 0, "parsed_ok": True,
                                                             "answer_correct": False}]}}}]}
    (tmp_path / "labels_test.json").write_text(json.dumps(labels), encoding="utf-8")
    out2 = PhaseA.e1_closed_book(fake)
    assert out2[("i1", 11)]["alias_correct"] is True and out2[("i1", 11)]["primary_k1"] is False


def test_decisions_and_lazy_plan_identical_with_and_without_verdicts():
    turns, cells, labels, items = synthetic_dataset(16, seed=5)
    cfg = load_e3v2_config(overrides=synthetic_cfg_overrides(nonhideable_increment_ms=300.0))
    gold = {it["id"]: it["gold_doc_ids"] for it in items}
    retrieval = [c for c in cells if c["kind"] != "cell"]
    with_verdicts = cells
    # without verdicts: the store a pre-gate rebuild writes (unresolved skip cells excluded)
    without = retrieval + [c for c in cells if c["kind"] == "cell" and c["path"] != "skip"]

    def plan(rows):
        record = set()
        recs = run_architectures(list(turns.values()), CellStore(rows, recording=record), NAMES, PRICES,
                                 [11, 23], ["faiss", "remote_api"], cfg, LatencyModel(cfg=cfg),
                                 ("commit", "speculative"), "test", {}, gold)
        decisions = sorted((r["architecture"], r["baseline"], str(r["prices"]), r["latency_profile"], r["item_id"],
                            r["seed"], r["dispatch_t_ms"], r["query_variant"], r["used"],
                            r["outcome"]["retrievals"]) for r in recs)
        return decisions, record

    d_with, plan_with = plan(with_verdicts)
    d_without, plan_without = plan(without)
    assert d_with == d_without
    assert plan_with == plan_without and plan_with
