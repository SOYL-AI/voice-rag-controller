"""
build_supplement.py — the retrieval-path supplement to the human labelling kit
(deviation D2, 2026-10-17, PI-approved, before any test answer was judged).

The original kit (build_kit.py) holds 300 d2_support CLOSED-BOOK answers only, so the
pre-registered gate criterion "|error rate on retrieve-path answers - error rate on
skip-path answers| <= 0.05" (PREREG_V2 §2.4) could not be evaluated, and the judge was
never checked on grounded answers, which are most of what it grades on test. This adds
100 d2_support DEV retrieval-path answers from E3-v2 Phase A, labelled by the same two
annotators under the same r3 rubric.

Sampling (seed 20261017): one answer per item; stratified by item_type and by the
primary judge's verdict so both verdicts are represented. Judge verdicts are never
written to the annotator sheets; they go only to supp_master_DO_NOT_SHARE.csv.

    python -m simulator.judge_validation.v2_labeling.build_supplement \
        --phase-a-dir <.../simulator/e3v2_phase_a/d2_support> --out-dir <human labelling folder>
"""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook

from simulator.judge_validation.v2_labeling import make_excel as mx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEED = 20261017
QUOTA = {"org_specific": 30, "counter_prior": 20, "ambiguous": 15, "general": 35}
SHEET_FIELDS = ["row", "case_id", "domain", "company", "conversation_before", "question",
                "reference_answer", "core_facts", "supplementary_facts", "candidate_answer",
                "verdict", "notes"]


def _facts(lines):
    return "\n".join(f"- {f}" for f in lines)


def _history(item):
    turns = item.get("history") or []
    return "\n".join(f"{'Caller' if t.get('role') == 'user' else 'Agent'}: {t.get('content', '')}"
                     for t in turns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    items = {json.loads(l)["id"]: json.loads(l)
             for l in open(REPO / "datasets/v2/d2_support/items.jsonl", encoding="utf-8")}
    dev = set(json.load(open(REPO / "datasets/v2/d2_support/splits.json"))["dev"])
    answers = {}
    for line in open(args.phase_a_dir / "answers_dev.jsonl", encoding="utf-8"):
        a = json.loads(line)
        answers[a["key"]] = a
    secondary = {}
    sec_path = args.phase_a_dir / "grades_secondary_dev.jsonl"
    if sec_path.exists():
        for line in open(sec_path, encoding="utf-8"):
            s = json.loads(line)
            secondary[s["key"]] = s.get("secondary")

    pool = defaultdict(list)  # (item_type, primary verdict) -> candidate answers
    for line in open(args.phase_a_dir / "grades_k1_dev.jsonl", encoding="utf-8"):
        g = json.loads(line)
        if g.get("condition") != "retrieval" or g["key"] not in answers or g["item_id"] not in dev:
            continue
        if g.get("primary") not in (True, False):
            continue
        it = items[g["item_id"]]
        pool[(it["item_type"], g["primary"])].append(g)

    rng = random.Random(SEED)
    chosen, used_items = [], set()
    for item_type, quota in QUOTA.items():
        per_verdict = {True: quota // 2, False: quota - quota // 2}
        for verdict, n in per_verdict.items():
            cands = sorted(pool[(item_type, verdict)], key=lambda g: g["key"])
            rng.shuffle(cands)
            for g in cands:
                if len([c for c in chosen if c[0] == (item_type, verdict)]) >= n:
                    break
                if g["item_id"] in used_items:
                    continue
                chosen.append(((item_type, verdict), g))
                used_items.add(g["item_id"])
    print(f"selected {len(chosen)} answers:",
          {f"{t}/{'correct' if v else 'incorrect'}": sum(1 for c in chosen if c[0] == (t, v))
           for t in QUOTA for v in (True, False)})

    master, sheet_rows = [], []
    for (item_type, verdict), g in chosen:
        it, a = items[g["item_id"]], answers[g["key"]]
        core = it["meta"]["core_facts"]
        supp = [f for f in it["meta"]["key_facts"] if f not in core]
        case_id = f"{g['item_id']}#s{g['seed']}#retrieval#{g['key'][:12]}"
        base = {"case_id": case_id, "domain": it["domain"], "company": it["org"],
                "conversation_before": _history(it), "question": it["question"],
                "reference_answer": it["expected_answer"], "core_facts": _facts(core),
                "supplementary_facts": _facts(supp), "candidate_answer": a["text"].strip()}
        sheet_rows.append(base)
        master.append({**base, "dataset": "d2_support", "item_id": g["item_id"],
                       "draw_seed": g["seed"], "condition": "retrieval", "path": "retrieve",
                       "item_type": item_type, "phase_a_key": g["key"],
                       "doc_ids": "|".join(a.get("doc_ids") or []),
                       "primary_r3_k1": verdict, "secondary_r3": secondary.get(g["key"], ""),
                       "partition": "holdout"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(HERE / "supp_master_DO_NOT_SHARE.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(master[0].keys()))
        w.writeheader()
        w.writerows(master)
    for who, seed in (("A", SEED + 1), ("B", SEED + 2)):
        rows = list(sheet_rows)
        random.Random(seed).shuffle(rows)
        rows = [{**r, "row": i + 1, "verdict": "", "notes": ""} for i, r in enumerate(rows)]
        csv_path = HERE / f"supp_sheet_annotator_{who}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SHEET_FIELDS)
            w.writeheader()
            w.writerows(rows)
        wb = Workbook()
        mx._instructions(wb, who)
        wb["Instructions"]["A1"] = (f"Answer-correctness labelling — SUPPLEMENT (100 answers) — Annotator {who}")
        verdict_col = mx._labeling(wb, rows)
        mx._progress(wb, verdict_col, len(rows))
        wb.active = 1
        wb.calculation.fullCalcOnLoad = True
        out = args.out_dir / f"Supplement_{who}_retrieval_labeling.xlsx"
        wb.save(out)
        print(f"wrote {out} ({len(rows)} rows)")
    digest = hashlib.sha256((HERE / "supp_master_DO_NOT_SHARE.csv").read_bytes()).hexdigest()
    print("supp master sha256:", digest)


if __name__ == "__main__":
    main()
