"""Convert v1 E1's 36-item dataset to the v2 item schema (smoke tests only).

    python -m llm.experiment1_v2.convert_v1 --out .tmp/v1_e1 [--all-dev] [--plumbing-traces]

Writes items.jsonl, splits.json and corpus.jsonl (from retrieval/corpus/) in
the CONTRACTS.md layout so every E1-v2 entry point can run on it. This is
plumbing data: 36 trivially separable hotel items, the reason v2 exists.

--all-dev puts every item in dev (the contract's 30% rule leaves ~11 items,
too few for a smoke run of the analysis). --plumbing-traces writes
speech/traces/synthetic-plumbing.jsonl with word timings at a fixed 150 wpm
and "partials" that are the reference words spoken so far. Those are NOT
ASR output and violate the section 4 rule on purpose; run.py refuses them
for any backend but mock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.experiment1_v2 import common  # noqa: E402

V1_DATASET = common.REPO_ROOT / "llm" / "experiment1" / "dataset.jsonl"
V1_CORPUS = common.REPO_ROOT / "retrieval" / "corpus"
DATASET = "v1_e1"
ORG = "Grand Horizon Hotel & Suites"


def convert(rows: list, all_dev: bool) -> tuple:
    items = []
    for row in rows:
        item_id = f"{DATASET}-{row['id']}"
        source = row.get("source") or ""
        gold = [Path(source).name] if source.startswith("retrieval/corpus/") else []
        items.append({
            "id": item_id,
            "dataset": DATASET,
            "domain": "hospitality",
            "org": ORG,
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "answer_aliases": [],
            "gold_doc_ids": gold,
            "item_type": "org_specific" if row["retrieval_needed"] else "general",
            "retrieval_needed_authored": bool(row["retrieval_needed"]),
            "pair_id": None,
            "popularity": None,
            "history": [],
            "split": "dev" if all_dev else common.contract_split(item_id),
            "meta": {"v1_id": row["id"], "v1_source": source},
        })
    splits = {
        "dev": [i["id"] for i in items if i["split"] == "dev"],
        "test": [i["id"] for i in items if i["split"] == "test"],
        "seed": None,
        "method": "all items in dev (smoke)" if all_dev else
                  "sha256(id) mod 1000 < 300 -> dev (CONTRACTS.md section 7)",
    }
    return items, splits


def corpus_rows() -> list:
    rows = []
    for path in sorted(V1_CORPUS.glob("*.txt")):
        rows.append({"doc_id": path.name, "dataset": DATASET, "domain": "hospitality", "org": ORG,
                     "title": path.stem, "text": path.read_text(encoding="utf-8"),
                     "source": "authored", "licence": "project-internal"})
    return rows


def plumbing_trace(item: dict, wpm: int = 150, interval_ms: int = 300) -> dict:
    words = item["question"].split()
    per_word = 60000.0 / wpm
    timed = [{"w": w, "start_ms": int(i * per_word), "end_ms": int((i + 1) * per_word)}
             for i, w in enumerate(words)]
    duration = timed[-1]["end_ms"] if timed else 0
    partials = []
    t = interval_ms
    while t <= duration + interval_ms:
        spoken = [w["w"] for w in timed if w["end_ms"] <= t]
        partials.append({"t_ms": t, "available_ms": t + 120, "text": " ".join(spoken),
                         "stable": False})
        t += interval_ms
    return {
        "item_id": item["id"], "voice_id": "synthetic-plumbing", "accent": "none",
        "wpm_target": wpm, "noise": "clean", "tts_engine": "none (synthetic timing)",
        "audio_path": None, "duration_ms": duration, "words": timed, "partials": partials,
        "asr_final": {"t_ms": duration + 350, "text": item["question"]},
        "vad_user_end_ms": duration + 200, "asr_model": "none (synthetic, NOT ASR)",
        "partial_interval_ms": interval_ms,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input", type=Path, default=V1_DATASET)
    p.add_argument("--all-dev", action="store_true")
    p.add_argument("--plumbing-traces", action="store_true")
    args = p.parse_args(argv)
    rows = common.read_jsonl(args.input, tolerate_partial_tail=False)
    items, splits = convert(rows, args.all_dev)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "items.jsonl", "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    common.atomic_write_json(args.out / "splits.json", splits)
    with open(args.out / "corpus.jsonl", "w", encoding="utf-8") as handle:
        for row in corpus_rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.plumbing_traces:
        tdir = args.out / "speech" / "traces"
        tdir.mkdir(parents=True, exist_ok=True)
        with open(tdir / "synthetic-plumbing.jsonl", "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(plumbing_trace(item)) + "\n")
    print(f"wrote {len(items)} items ({len(splits['dev'])} dev / {len(splits['test'])} test) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
