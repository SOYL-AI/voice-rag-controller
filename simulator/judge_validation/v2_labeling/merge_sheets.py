"""Merge the two completed annotator sheets into the judge-validation format.

    python -m simulator.judge_validation.v2_labeling.merge_sheets \
        --a sheet_annotator_A_done.csv --b sheet_annotator_B_done.csv \
        [--adjudication adjudication_done.csv] --out human_labels_v2.csv

Joins both sheets to kit_master_DO_NOT_SHARE.csv by case_id and writes the
CSV that `python -m simulator.judge_validation.score_judges --v2-labels`
reads: case_id, dataset, item_id, question, reference, answer, key_facts,
aliases, history, path, gate_subset, label_a, label_b, label_adjudicated.

Also writes adjudication_needed.csv (next to --out): every case where A and
B disagree, with both notes and an empty `verdict` column for the third
labeller. Re-run with --adjudication once it is filled.

Refuses sheets with unknown or missing case_ids or unreadable verdicts, and
prints how many verdicts are still blank, so a half-finished sheet is never
merged silently.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simulator.judge_validation.score_judges import agreement, parse_label  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_COLUMNS = ["case_id", "dataset", "item_id", "question", "reference", "answer", "key_facts",
               "aliases", "history", "path", "gate_subset", "label_a", "label_b",
               "label_adjudicated", "notes_a", "notes_b", "notes_adjudication"]


def read_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_sheet(path: Path, master_ids: set, who: str) -> dict:
    rows = read_csv(path)
    out, problems = {}, []
    for row in rows:
        cid = (row.get("case_id") or "").strip()
        if cid not in master_ids:
            problems.append(f"unknown case_id {cid!r}")
            continue
        if cid in out:
            problems.append(f"duplicate case_id {cid!r}")
            continue
        try:
            verdict = parse_label(row.get("verdict"))
        except ValueError:
            problems.append(f"{cid}: unreadable verdict {row.get('verdict')!r} "
                            "(use correct / incorrect)")
            continue
        out[cid] = {"verdict": verdict, "notes": (row.get("notes") or "").strip()}
    missing = master_ids - set(out)
    if missing:
        problems.append(f"{len(missing)} case_ids missing, e.g. {sorted(missing)[:3]}")
    if problems:
        raise ValueError(f"{who} sheet {path}: " + "; ".join(problems[:10]))
    blank = sum(v["verdict"] is None for v in out.values())
    if blank:
        print(f"WARNING: {who} left {blank} verdicts blank.")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--adjudication", type=Path)
    p.add_argument("--master", type=Path, default=HERE / "kit_master_DO_NOT_SHARE.csv")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    master = {r["case_id"]: r for r in read_csv(args.master)}
    ids = set(master)
    try:
        a = read_sheet(args.a, ids, "annotator A")
        b = read_sheet(args.b, ids, "annotator B")
        adj = {}
        if args.adjudication:
            for row in read_csv(args.adjudication):
                verdict = parse_label(row.get("verdict"))
                if verdict is not None:
                    adj[row["case_id"].strip()] = {"verdict": verdict,
                                                   "notes": (row.get("notes") or "").strip()}
    except ValueError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 1

    rows, disputes = [], []
    for cid in sorted(ids):
        m = master[cid]
        la, lb = a[cid]["verdict"], b[cid]["verdict"]
        rows.append({
            "case_id": cid, "dataset": m["dataset"], "item_id": m["item_id"],
            "question": m["question"], "reference": m["reference"], "answer": m["answer"],
            "key_facts": m["key_facts"], "aliases": m["aliases"], "history": m["history"],
            "path": m["path"], "gate_subset": m["gate_subset"],
            "label_a": "" if la is None else int(la), "label_b": "" if lb is None else int(lb),
            "label_adjudicated": "" if cid not in adj else int(adj[cid]["verdict"]),
            "notes_a": a[cid]["notes"], "notes_b": b[cid]["notes"],
            "notes_adjudication": adj.get(cid, {}).get("notes", ""),
        })
        if la is not None and lb is not None and la != lb and cid not in adj:
            disputes.append({"case_id": cid, "question": m["question"], "reference": m["reference"],
                             "key_facts": m["key_facts"], "answer": m["answer"],
                             "annotator_A": "correct" if la else "incorrect", "notes_A": a[cid]["notes"],
                             "annotator_B": "correct" if lb else "incorrect", "notes_B": b[cid]["notes"],
                             "verdict": "", "notes": ""})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    need = args.out.parent / "adjudication_needed.csv"
    with open(need, "w", encoding="utf-8-sig", newline="") as handle:
        columns = ["case_id", "question", "reference", "key_facts", "answer", "annotator_A",
                   "notes_A", "annotator_B", "notes_B", "verdict", "notes"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(disputes)
    stats = agreement([a[c]["verdict"] for c in sorted(ids)], [b[c]["verdict"] for c in sorted(ids)])
    print(f"merged {len(rows)} cases -> {args.out}")
    print(f"A vs B: {stats['agree']}/{stats['n']} agree, Cohen's kappa {stats['cohen_kappa']} "
          f"(gate requires >= 0.70 before any judge is assessed)")
    print(f"{len(disputes)} disagreement(s) still need adjudication -> {need}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
