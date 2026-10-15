"""Build the E1-v2 human labelling kit (judge-validation gate, PREREG_V2 2.4).

    python -m simulator.judge_validation.v2_labeling.build_kit \
        --labels llm/experiment1_v2/results/d2_support/mistral7b_v03/labels_dev.e1v2-r2.json

Samples N (default 300) graded dev answers with a fixed seed, stratified by
item_type x (r2 primary/secondary judge agreement), and writes:

  sheet_annotator_A.csv, sheet_annotator_B.csv
      The same answers, each sheet shuffled independently. Judge verdicts,
      item type, path and sampling stratum are NOT shown. Annotators fill
      `verdict` (correct / incorrect) and optionally `notes`.
  kit_master_DO_NOT_SHARE.csv
      One row per case: everything needed to merge and analyse, including
      the stratum, inclusion weight, r2 judge verdicts and the 100/200
      prompt-development / gate-holdout partition. Not for annotators.
  kit_manifest.json
      Seed, sizes, strata, input file hashes.

Sampling:
  * Pool: every graded draw in the labels file whose item is in the split,
    with both judges' r2 verdicts, de-duplicated so that identical answer
    texts for one item count once; at most 2 answers per item.
  * Strata: item_type x {judges agree, judges disagree}. Each item_type gets
    a share of N proportional to its share of pool ITEMS. Within a type,
    disagreements are oversampled to up to half of the type's quota (they
    are where the gate is decided); the rest come from agreements.
  * Each case carries its inclusion probability's inverse (`weight`) so
    agreement can be re-weighted to the pool.
  * Partition: cases ranked by sha256(case_id); the lowest 1/3 (100 of 300)
    are prompt_dev, the rest holdout. Fixed before any judge is re-run.

Rubric: annotators label under r3 (core facts). The sheets show the core
facts apart from the supplementary ones, and the build is refused while any
sampled item lacks meta.core_facts. The sample depends only on the labels
file and the seed, so re-running after the core-fact annotation is merged
reproduces the same 300 answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from llm.experiment1_v2 import common  # noqa: E402

HERE = Path(__file__).resolve().parent
# Annotators label under rubric r3: the sheet shows the core facts (must all
# be stated) apart from the supplementary ones (may be omitted, must not be
# contradicted).
ANNOTATION_RUBRIC = "e1v2-r3"
SHEET_COLUMNS = ["row", "case_id", "domain", "company", "conversation_before", "question",
                 "reference_answer", "core_facts", "supplementary_facts", "candidate_answer",
                 "verdict", "notes"]
MASTER_COLUMNS = ["case_id", "dataset", "item_id", "draw_seed", "condition", "path", "item_type",
                  "domain", "org", "question", "history", "reference", "key_facts", "core_facts",
                  "aliases", "answer", "stratum", "judges_agree", "primary_r2", "secondary_r2",
                  "weight", "gate_subset"]


def core_and_supplementary(item: dict) -> tuple:
    """(core facts or None if not annotated, supplementary facts)."""
    meta = item.get("meta") or {}
    key = list(meta.get("key_facts") or [])
    core = list(meta.get("core_facts") or []) or None
    return core, [f for f in key if not core or f not in core]


def history_text(history) -> str:
    lines = []
    for turn in history or []:
        role = "Caller" if str(turn.get("role", "")).lower() in ("user", "caller", "customer") else "Agent"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def case_id(draw: dict) -> str:
    return f"{draw['item_id']}#s{draw['draw_seed']}#{draw.get('condition', 'closed_book')}"


def build_pool(labels: dict, items: dict) -> list:
    pool, seen = [], set()
    for draw in labels["draws"]:
        item = items.get(draw["item_id"])
        judges = draw.get("judges") or {}
        if item is None or "primary" not in judges or "secondary" not in judges:
            continue
        p, s = judges["primary"]["answer_correct"], judges["secondary"]["answer_correct"]
        if p is None or s is None:
            continue  # no valid verdict: cannot be stratified by agreement
        answer = " ".join((draw.get("answer") or "").split())
        key = (draw["item_id"], draw.get("condition", "closed_book"), answer)
        if key in seen:
            continue
        seen.add(key)
        pool.append({"draw": draw, "item": item, "primary": p, "secondary": s,
                     "agree": p == s, "answer": answer})
    return pool


def sample(pool: list, n: int, seed: int, per_item_cap: int = 2) -> list:
    rng = random.Random(seed)
    types = sorted({c["item"]["item_type"] for c in pool})
    items_by_type = {t: {c["item"]["id"] for c in pool if c["item"]["item_type"] == t} for t in types}
    n_items = sum(len(v) for v in items_by_type.values())
    quota = {t: round(n * len(items_by_type[t]) / n_items) for t in types}
    quota[max(types, key=lambda t: quota[t])] += n - sum(quota.values())  # exact total
    chosen, per_item = [], defaultdict(int)

    def take(cands, k):
        rng.shuffle(cands)
        got = []
        for c in cands:
            if len(got) >= k:
                break
            if per_item[c["item"]["id"]] >= per_item_cap:
                continue
            per_item[c["item"]["id"]] += 1
            got.append(c)
        return got

    for c in pool:
        c["stratum"] = f"{c['item']['item_type']}|{'agree' if c['agree'] else 'disagree'}"
    for t in types:
        cands = [c for c in pool if c["item"]["item_type"] == t]
        dis = [c for c in cands if not c["agree"]]
        agr = [c for c in cands if c["agree"]]
        got = take(dis, min(len(dis), quota[t] // 2))
        got += take(agr, quota[t] - len(got))
        got += take([c for c in dis if c not in got], quota[t] - len(got))  # top up within the type
        chosen += got
    if len(chosen) < n:  # the per-item cap can leave a type short: fill from anywhere
        chosen += take([c for c in pool if c not in chosen], n - len(chosen))
    # Inverse inclusion probability per stratum, computed on the final sample.
    pool_size = defaultdict(int)
    for c in pool:
        pool_size[c["stratum"]] += 1
    sampled = defaultdict(int)
    for c in chosen:
        sampled[c["stratum"]] += 1
    for c in chosen:
        c["weight"] = pool_size[c["stratum"]] / sampled[c["stratum"]]
    return chosen


def partition(cases: list, n_prompt_dev: int) -> None:
    ranked = sorted(cases, key=lambda c: hashlib.sha256(c["case_id"].encode()).hexdigest())
    for index, c in enumerate(ranked):
        c["gate_subset"] = "prompt_dev" if index < n_prompt_dev else "holdout"


def write_csv(path: Path, columns, rows) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, required=True, help="labels_<split>.e1v2-r2.json")
    p.add_argument("--items", type=Path, help="items.jsonl (default from the labels' dataset)")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--n-prompt-dev", type=int)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--allow-missing-core", action="store_true",
                   help="testing only: show key_facts[0] as the core fact when not annotated")
    args = p.parse_args(argv)
    labels = common.load_json(args.labels)
    meta = labels["meta"]
    if meta.get("split") != "dev":
        print("REFUSED: the labelling kit is built from DEV answers only (PREREG_V2 2.4).", file=sys.stderr)
        return 3
    if meta["judges"]["primary"]["prompt_version"] == "e1v2-reasoned-strip-r1":
        print("WARNING: these labels were graded under r1; stratification uses r1 agreement.")
    items_path = args.items or common.dataset_dir(meta["dataset"]) / "items.jsonl"
    items = {i["id"]: i for i in common.load_items(items_path)}
    pool = build_pool(labels, items)
    chosen = sample(pool, args.n, args.seed)
    for c in chosen:
        c["case_id"] = case_id(c["draw"])
    partition(chosen, args.n_prompt_dev if args.n_prompt_dev is not None else round(len(chosen) / 3))
    missing_core = sorted({c["item"]["id"] for c in chosen if core_and_supplementary(c["item"])[0] is None})
    if missing_core and not args.allow_missing_core:
        print(f"REFUSED: {len(missing_core)} sampled item(s) have no meta.core_facts (e.g. "
              f"{missing_core[:3]}). Annotators label under {ANNOTATION_RUBRIC}, which needs the "
              "core facts; re-run once the core-fact annotation is merged (the same seed gives "
              "the same sample).", file=sys.stderr)
        return 2

    master = []
    for c in sorted(chosen, key=lambda c: c["case_id"]):
        item, draw = c["item"], c["draw"]
        master.append({
            "case_id": c["case_id"], "dataset": item["dataset"], "item_id": item["id"],
            "draw_seed": draw["draw_seed"], "condition": draw.get("condition", "closed_book"),
            "path": "retrieve" if draw.get("condition") == "retrieval" else "skip",
            "item_type": item.get("item_type"), "domain": item.get("domain"), "org": item.get("org"),
            "question": item["question"], "history": json.dumps(item.get("history") or []),
            "reference": item["expected_answer"],
            "key_facts": json.dumps((item.get("meta") or {}).get("key_facts") or []),
            "core_facts": json.dumps(core_and_supplementary(item)[0]
                                     or ((item.get("meta") or {}).get("key_facts") or [])[:1]),
            "aliases": json.dumps(item.get("answer_aliases") or []), "answer": c["answer"],
            "stratum": c["stratum"], "judges_agree": c["agree"],
            "primary_r2": c["primary"], "secondary_r2": c["secondary"],
            "weight": round(c["weight"], 6) if c["weight"] else None, "gate_subset": c["gate_subset"],
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "kit_master_DO_NOT_SHARE.csv", MASTER_COLUMNS, master)
    for name, offset in (("A", 1), ("B", 2)):
        rows = list(master)
        random.Random(args.seed + offset).shuffle(rows)
        sheet = [{
            "row": n + 1, "case_id": r["case_id"], "domain": r["domain"], "company": r["org"],
            "conversation_before": history_text(json.loads(r["history"])), "question": r["question"],
            "reference_answer": r["reference"],
            "core_facts": "\n".join(f"- {f}" for f in json.loads(r["core_facts"])),
            "supplementary_facts": "\n".join(f"- {f}" for f in json.loads(r["key_facts"])
                                             if f not in json.loads(r["core_facts"])),
            "candidate_answer": r["answer"], "verdict": "", "notes": "",
        } for n, r in enumerate(rows)]
        write_csv(args.out_dir / f"sheet_annotator_{name}.csv", SHEET_COLUMNS, sheet)
    strata = defaultdict(lambda: {"sampled": 0, "pool": 0})
    for c in pool:
        strata[f"{c['item']['item_type']}|{'agree' if c['agree'] else 'disagree'}"]["pool"] += 1
    for c in chosen:
        strata[c["stratum"]]["sampled"] += 1
    manifest = {
        "created_at_utc": common.now_utc(), "git_commit": common.git_commit(), "seed": args.seed,
        "n": len(chosen), "n_prompt_dev": sum(c["gate_subset"] == "prompt_dev" for c in chosen),
        "n_holdout": sum(c["gate_subset"] == "holdout" for c in chosen),
        "n_items": len({c["item"]["id"] for c in chosen}), "pool_size": len(pool),
        "strata": dict(sorted(strata.items())),
        "labels_file": common.rel(args.labels), "labels_sha256": common.sha256_file(args.labels),
        "items_sha256": common.sha256_file(items_path),
        "judge_prompt_version": meta["judges"]["primary"]["prompt_version"],
        "sampling_note": "strata use the judge agreement in the labels file above; the same "
                         "seed and labels file give the same sample",
        "annotation_rubric": ANNOTATION_RUBRIC,
        "core_facts_missing": missing_core,
        "dataset": meta["dataset"], "paths": sorted({r["path"] for r in master}),
    }
    common.atomic_write_json(args.out_dir / "kit_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "strata"}, indent=1))
    for stratum, counts in manifest["strata"].items():
        print(f"  {stratum:<28} sampled {counts['sampled']:>3} of {counts['pool']:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
