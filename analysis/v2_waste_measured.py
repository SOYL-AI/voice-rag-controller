"""
v2_waste_measured.py -- what the v2 waste term w is in the frozen code, and
what the existing v2 data can and cannot say about it (DESCRIPTIVE).

    PYTHONUTF8=1 python -m analysis.v2_waste_measured

THE FROZEN DEFINITION. controller/baselines/_objective_v2.py:waste_at():

    w(f) = coef * (1 - f) before the end of turn,  w = 0 at the end of turn,

f the causal word-fraction estimate e/(e + R_hat), coef =
controller.e3v2.waste.coef = 0.1 (configs/config.yaml; the frozen d1_open
bundles override quality, calibration, ours, timing and estimates_artifact,
not waste). The docstring calls it a PLACEHOLDER "to be replaced by a
measured revision rate from the v2 speech traces". It is not estimated from
any data, and it does not enter the realised utility the evaluator scores
(simulator/e3v2_decide.py charges lambda_c per dispatched retrieval); it
enters only the controllers' decisions.

WHAT IS MEASURED HERE (read-only, committed files only):

 A. Cancellation of early dispatches.
    * COMMIT architecture (the primary contrast Delta[COMMIT]): a dispatched
      retrieval is always used (controller/sequential_v2.py: cancelled_dispatches
      is set only on a SPECULATE -> DISCARD path, which commit forbids), so
      the cancellation rate of early commits is 0 by construction, not by
      measurement.
    * SPECULATIVE architecture: q_hat = fraction of SPEC (b10_speculative_v2)
      dispatches discarded at end of turn, and w_bar = mean frozen w(f) at
      those dispatches, as logged by simulator/e3v2_decide.py:boundary_contrasts
      in the dev predictions of record (docs/research_v2/v2_predictions.json)
      and the d1_open test results (docs/research_v2/v2_test_results_d1.json).
      Those files store fractions, not counts. A count is reconstructed only
      when it is exact and unique: q_hat = d/n with n <= n_items (decisions
      are seed-invariant, so the 5-seed record fraction equals the decision
      fraction); the reduced fraction p/q must reproduce the stored float
      exactly and 2q > n_items, so n = q and d = p.

 B. Revision of the partial a commit was made on (the event waste_at's
    docstring says w should measure). From the committed speech traces
    (datasets/v2/<ds>/speech/traces/<primary condition>.jsonl), with the
    frozen step construction (controller/sequential_v2.build_steps, default
    end-of-turn timing = the frozen bundles' eot_detection_ms 0) and the
    frozen query_ready gate: an early query-ready partial is REVISED if its
    word sequence (lower-cased, punctuation stripped) is not a prefix of the
    final ASR transcript's; REVISED-EXCL-LAST ignores the partial's last word
    (which may be cut mid-word). Reported over every early query-ready
    partial and at the first one (SPEC's 'immediate' trigger). The text150
    condition is revision-free by construction (its prefixes are the first
    floor(f n) words of the question). d2_support: dev split only (its test
    run is in progress elsewhere and is not read).

Writes docs/research_v2/waste_measured_v2.json.
"""

from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import load_e3v2_config  # noqa: E402
from controller.sequential_v2 import TurnCtx, build_steps  # noqa: E402
from simulator.e3v2_decide import trace_timing  # noqa: E402

OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "waste_measured_v2.json"
PRIMARY_STEM = "piper-en_US-hfc_male-medium__150wpm__clean"
PRED = _REPO_ROOT / "docs" / "research_v2" / "v2_predictions.json"
TEST_D1 = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_d1.json"
SPLITS_READ = {"d1_open": ("dev", "test"), "d2_support": ("dev",)}


def exact_count(q_hat, n_items):
    """(d, n) with d/n == q_hat exactly and n <= n_items, if unique; else None."""
    if q_hat is None:
        return None
    if q_hat == 0:
        return None  # 0 = 0/n for any n: the denominator is not recoverable
    fr = Fraction(q_hat).limit_denominator(n_items)
    if fr.numerator / fr.denominator != q_hat:
        return None
    if 2 * fr.denominator <= n_items:
        return None  # n could be any multiple of q <= n_items
    return {"discarded": fr.numerator, "spec_dispatch_decisions": fr.denominator}


def spec_rows():
    out = []
    pred = json.loads(PRED.read_text(encoding="utf-8"))
    for c in pred["cells"]:
        n = c["n_items"]["speculative"]
        out.append({"source": "v2_predictions.json (dev of record)", "split": "dev", "dataset": c["dataset"],
                    "condition": c["condition"], "profile": c["profile"], "belief_quantile": c["belief_quantile"],
                    "lambda_l": c["lambda_l"], "lambda_c": c["lambda_c"], "n_items": n,
                    "q_hat_spec_discard_rate": c["q_hat"], "w_bar_frozen_model_at_spec_dispatch": c["w_bar"],
                    "count": exact_count(c["q_hat"], n)})
    test = json.loads(TEST_D1.read_text(encoding="utf-8"))
    for c in test["cells"]:
        n = c["n_items"]["speculative"]
        cond = "text150" if c["condition"] == "text150" else c["condition"]
        lam_l, lam_c = (float(x) for x in c["price"].split("|"))
        out.append({"source": "v2_test_results_d1.json", "split": "test", "dataset": "d1_open",
                    "condition": cond, "profile": c["profile"], "belief_quantile": c["belief_quantile"],
                    "lambda_l": lam_l, "lambda_c": lam_c, "n_items": n,
                    "q_hat_spec_discard_rate": c["q_hat"], "w_bar_frozen_model_at_spec_dispatch": c["w_bar"],
                    "count": exact_count(c["q_hat"], n)})
    return out


def spec_summary(rows):
    """Per (split, dataset, condition, profile): pooled over the 4 frozen prices."""
    groups = {}
    for r in rows:
        key = (r["split"], r["dataset"], r["condition"], r["profile"])
        groups.setdefault(key, []).append(r)
    out = []
    for (split, ds, cond, prof), rs in sorted(groups.items()):
        qs = [r["q_hat_spec_discard_rate"] for r in rs]
        counts = [r["count"] for r in rs]
        pooled = None
        if all(c is not None for c in counts):
            d = sum(c["discarded"] for c in counts)
            n = sum(c["spec_dispatch_decisions"] for c in counts)
            pooled = {"discarded": d, "spec_dispatch_decisions": n, "rate": d / n}
        out.append({"split": split, "dataset": ds, "condition": cond, "profile": prof, "n_prices": len(rs),
                    "q_hat_min": min(qs), "q_hat_max": max(qs),
                    "w_bar": sorted({r["w_bar_frozen_model_at_spec_dispatch"] for r in rs}),
                    "pooled_over_prices_exact_counts": pooled})
    return out


_PUNCT = re.compile(r"[^\w']+")


def words(text):
    out = []
    for w in (text or "").split():
        w = _PUNCT.sub("", w.lower())
        if w:
            out.append(w)
    return out


def is_prefix(a, b):
    return len(a) <= len(b) and b[: len(a)] == a


def revision_rates(ds, split, cfg):
    splits = json.loads((_REPO_ROOT / "datasets" / "v2" / ds / "splits.json").read_text(encoding="utf-8"))
    keep = set(splits[split])
    path = _REPO_ROOT / "datasets" / "v2" / ds / "speech" / "traces" / f"{PRIMARY_STEM}.jsonl"
    n_turns = n_turns_with_early = 0
    all_n = all_rev = all_rev_l = 0
    first_n = first_rev = first_rev_l = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            tr = json.loads(line)
            if tr["item_id"] not in keep:
                continue
            n_turns += 1
            ctx = TurnCtx(item_id=tr["item_id"], signals={"timing": trace_timing(tr)}, condition=PRIMARY_STEM,
                          transcript_final=(tr.get("asr_final") or {}).get("text", ""))
            steps = build_steps(ctx, cfg)
            final = words(steps[-1].text)
            early = [s for s in steps if not s.is_final and s.query_ready]
            if early:
                n_turns_with_early += 1
            for j, s in enumerate(early):
                pw = words(s.text)
                rev = not is_prefix(pw, final)
                rev_l = not is_prefix(pw[:-1], final)
                all_n += 1
                all_rev += rev
                all_rev_l += rev_l
                if j == 0:
                    first_n += 1
                    first_rev += rev
                    first_rev_l += rev_l
    return {
        "dataset": ds, "split": split, "condition": PRIMARY_STEM, "trace_file": str(path.relative_to(_REPO_ROOT)),
        "n_turns": n_turns, "n_turns_with_an_early_query_ready_partial": n_turns_with_early,
        "all_early_query_ready_partials": {
            "n": all_n, "revised": all_rev, "revised_rate": all_rev / all_n if all_n else None,
            "revised_excl_last_word": all_rev_l, "revised_excl_last_word_rate": all_rev_l / all_n if all_n else None},
        "first_early_query_ready_partial": {
            "n": first_n, "revised": first_rev, "revised_rate": first_rev / first_n if first_n else None,
            "revised_excl_last_word": first_rev_l,
            "revised_excl_last_word_rate": first_rev_l / first_n if first_n else None},
    }


def main() -> int:
    cfg = load_e3v2_config()
    rows = spec_rows()
    out = {
        "_comment": "DESCRIPTIVE. Generated by python -m analysis.v2_waste_measured from committed files only. "
                    "See the module docstring for definitions.",
        "frozen_definition": {
            "function": "controller/baselines/_objective_v2.py:waste_at",
            "formula": "w(f) = coef * (1 - f) for a pre-end-of-turn commit, 0 at end of turn; f = e/(e + R_hat)",
            "coef_config": cfg["waste"]["coef"],
            "status": "PLACEHOLDER (docstring); not estimated from data; enters decisions only, not the "
                      "realised utility (lambda_c per dispatched retrieval).",
        },
        "commit_architecture_cancellations": {
            "rate": 0.0,
            "basis": "structural: under architecture 'commit' a dispatched retrieval is always used "
                     "(controller/sequential_v2.py TurnDecision.cancelled_dispatches is set only on "
                     "SPECULATE->DISCARD, which commit forbids)",
        },
        "spec_discard_rate_cells": rows,
        "spec_discard_rate_summary": spec_summary(rows),
        "partial_revision": [revision_rates(ds, split, cfg)
                             for ds, splits in SPLITS_READ.items() for split in splits],
        "partial_revision_text150": "0 by construction (prefixes are the first floor(f n) words of the question)",
    }
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for s in out["spec_discard_rate_summary"]:
        print(s)
    for r in out["partial_revision"]:
        print(r)
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
