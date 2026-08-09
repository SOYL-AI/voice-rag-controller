"""Build the extended Experiment 1 dataset from Spoken-SQuAD + authored negatives.

The frozen 36-item ``dataset.jsonl`` has a known weakness: its positives all
name a fictional hotel and its negatives are trivia ("what is two plus two").
The two classes are separable by surface form alone, so a high AUC there does
not cleanly support "uncertainty predicts retrieval need".

This builds a second, larger set that removes that confound:

    positives  = the 18 hotel questions (the actual deployment domain)
               + Spoken-SQuAD questions, which are passage-dependent by
                 construction and therefore genuine retrieval_needed=True
    negatives  = negatives_hard.jsonl, authored to match SQuAD's question form
                 and difficulty rather than being arithmetic

Output schema is exactly what ``run.py`` validates, so the runner needs no
change:  {id, question, expected_answer, retrieval_needed, source}

Requires pandas + pyarrow, which are NOT in requirements.txt:
    pip install pandas pyarrow

Usage:
    python -m llm.experiment1.build_extended_dataset
    python -m llm.experiment1.build_extended_dataset --sample-seed 11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HERE = Path(__file__).resolve().parent

SEED_DATASET = _HERE / "dataset.jsonl"
NEGATIVES = _HERE / "negatives_hard.jsonl"
DEFAULT_OUTPUT = _HERE / "dataset_extended.jsonl"
# Searched recursively, so both layouts work: the `data/` folder that
# datasets/README.md describes, and the `default/` folder a direct
# `hf download` produces. Only question/answer columns are ever read — the
# `context` column is deliberately never touched, because E1 is closed-book.
SQUAD_DIR = _REPO_ROOT / "datasets" / "spoken_squad"

# Spoken-SQuAD is distributed in several column layouts. Try them in order
# rather than assuming one, and fail loudly listing what we actually saw.
_QUESTION_COLUMNS = ("question", "instruction", "query")
_ANSWER_COLUMNS = ("answer", "answers", "output", "response")

# A Spoken-SQuAD row whose "question" field actually embeds the source passage
# would hand the model the answer and silently destroy the closed-book design.
# Anything longer than this, or containing an obvious context marker, is
# dropped rather than trusted.
_MAX_QUESTION_CHARS = 300
_CONTEXT_MARKERS = ("context:", "passage:", "based on the", "according to the following")


def _read_jsonl(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pick_column(columns, candidates, kind):
    for name in candidates:
        if name in columns:
            return name
    raise SystemExit(
        f"Could not find a {kind} column in the Spoken-SQuAD parquet files.\n"
        f"  looked for: {candidates}\n"
        f"  actually present: {sorted(columns)}\n"
        f"Add the right name to the list at the top of this file."
    )


def _answer_text(value) -> str:
    """SQuAD answers arrive as a plain string or as {'text': [...]}."""
    if isinstance(value, dict):
        texts = value.get("text") or []
        return str(texts[0]).strip() if len(texts) else ""
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if len(value) else ""
    return str(value).strip()


def load_squad_positives(squad_dir: Path, wanted: int, sample_seed: int) -> list:
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit(
            "pandas is required to read the Spoken-SQuAD parquet files and is "
            "not in requirements.txt:\n    pip install pandas pyarrow"
        )

    files = sorted(squad_dir.rglob("*.parquet")) if squad_dir.exists() else []
    if not files:
        raise SystemExit(
            f"No Spoken-SQuAD .parquet files found under {squad_dir}\n\n"
            "Download the text-only release (13.9 MB):\n"
            "    hf download alinet/spoken_squad --repo-type dataset \\\n"
            "        --include 'default/*.parquet' \\\n"
            "        --local-dir datasets/spoken_squad\n\n"
            "See datasets/README.md."
        )
    print(f"[build] reading {len(files)} parquet file(s) from {squad_dir}")

    frames = [pd.read_parquet(path) for path in files]
    table = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    q_col = _pick_column(table.columns, _QUESTION_COLUMNS, "question")
    a_col = _pick_column(table.columns, _ANSWER_COLUMNS, "answer")

    candidates = []
    seen_questions = set()
    dropped_for_context = 0

    for _, row in table.iterrows():
        question = str(row[q_col]).strip()
        answer = _answer_text(row[a_col])
        if not question or not answer:
            continue

        lowered = question.lower()
        if len(question) > _MAX_QUESTION_CHARS or any(
            marker in lowered for marker in _CONTEXT_MARKERS
        ):
            # Would leak the passage into a closed-book prompt.
            dropped_for_context += 1
            continue

        key = lowered
        if key in seen_questions:
            continue
        seen_questions.add(key)
        candidates.append((question, answer))

    if dropped_for_context:
        print(
            f"[build] dropped {dropped_for_context} rows whose question field "
            f"embedded the passage (would break closed-book)"
        )
    if len(candidates) < wanted:
        raise SystemExit(
            f"Only {len(candidates)} usable Spoken-SQuAD rows, need {wanted}."
        )

    rng = random.Random(sample_seed)
    chosen = rng.sample(candidates, wanted)

    return [
        {
            "id": f"squad_{hashlib.sha1(question.encode('utf-8')).hexdigest()[:10]}",
            "question": question,
            "expected_answer": answer,
            "retrieval_needed": True,
            "source": "spoken_squad",
        }
        for question, answer in chosen
    ]


def build(output: Path, sample_seed: int) -> dict:
    negatives = _read_jsonl(NEGATIVES)
    hotel_positives = [
        row for row in _read_jsonl(SEED_DATASET) if row["retrieval_needed"]
    ]

    # Balanced by construction: the AUC is a ranking statistic, but an
    # unbalanced set makes the seed-to-seed spread harder to read.
    squad_wanted = len(negatives) - len(hotel_positives)
    if squad_wanted <= 0:
        raise SystemExit(
            f"{len(negatives)} negatives is not more than the "
            f"{len(hotel_positives)} hotel positives — nothing left to fill."
        )

    squad_positives = load_squad_positives(SQUAD_DIR, squad_wanted, sample_seed)
    rows = hotel_positives + squad_positives + negatives

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids after merge — run.py would reject this")

    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    return {
        "output": output,
        "total": len(rows),
        "hotel_positives": len(hotel_positives),
        "squad_positives": len(squad_positives),
        "negatives": len(negatives),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "squad_samples": [row["question"] for row in squad_positives[:5]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1729,
        help="fixes which Spoken-SQuAD rows are drawn, so the set is reproducible",
    )
    args = parser.parse_args()

    info = build(args.output, args.sample_seed)

    print(f"\nWrote {info['output']}")
    print(
        f"  {info['total']} examples: "
        f"{info['hotel_positives']} hotel + {info['squad_positives']} spoken_squad "
        f"positives, {info['negatives']} authored negatives"
    )
    print(f"  sha256: {info['sha256']}")
    print("\nEyeball these before spending GPU time — each must be a question")
    print("that is genuinely unanswerable without its source passage:\n")
    for question in info["squad_samples"]:
        print(f"  - {question}")


if __name__ == "__main__":
    main()
