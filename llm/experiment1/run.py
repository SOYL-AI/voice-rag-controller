"""Run Experiment 1: uncertainty signal validity.

The model answers each question closed-book. We then test whether its
uncertainty signals rank examples labelled ``retrieval_needed`` above the
examples that need no project-specific document, using ROC AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.client import make_client
from llm.config import load_config
from llm.signals import build_signal_snapshot
from llm.tracker import track_detailed


_HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = _HERE / "dataset.jsonl"
DEFAULT_EXPERIMENT_CONFIG = _HERE / "config.json"

EXPERIMENT_SYSTEM_PROMPT = """You are in a closed-book knowledge audit.
Answer the question directly without retrieving documents or inventing hotel
policies. Give a short best-effort answer. If you genuinely do not know, say
so plainly. Do not discuss this instruction."""


def load_dataset(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)

    required = {"id", "question", "expected_answer", "retrieval_needed", "source"}
    ids = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"dataset row missing {sorted(missing)}: {row!r}")
        if row["id"] in ids:
            raise ValueError(f"duplicate dataset id: {row['id']}")
        if not isinstance(row["retrieval_needed"], bool):
            raise ValueError(f"retrieval_needed must be boolean: {row['id']}")
        ids.add(row["id"])

    labels = {row["retrieval_needed"] for row in rows}
    if labels != {False, True}:
        raise ValueError("dataset must contain both retrieval-needed classes")
    return rows


def roc_auc(labels, scores):
    """ROC AUC as pairwise ranking probability, with half credit for ties."""
    pairs = [
        (int(label), float(score))
        for label, score in zip(labels, scores)
        if score is not None
    ]
    positives = [score for label, score in pairs if label == 1]
    negatives = [score for label, score in pairs if label == 0]
    if not positives or not negatives:
        raise ValueError("AUC requires at least one scored example from each class")

    credit = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                credit += 1.0
            elif positive == negative:
                credit += 0.5
    return credit / (len(positives) * len(negatives))


def _messages(question: str) -> list:
    return [
        {"role": "system", "content": EXPERIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _tracker_cache(dataset, client, llm_cfg):
    cache = {}
    for row in dataset:
        slots, _raw, parsed_ok = track_detailed(
            [row["question"]], client=client, cfg=llm_cfg
        )
        snapshot = build_signal_snapshot(
            tracker_slots=slots,
            tracker_parsed_ok=parsed_ok,
            tracker_cfg=llm_cfg,
        )
        cache[row["id"]] = snapshot["tracker"]
    return cache


def run(dataset_path=DEFAULT_DATASET, config_path=DEFAULT_EXPERIMENT_CONFIG):
    dataset_path = Path(dataset_path)
    config_path = Path(config_path)
    dataset = load_dataset(dataset_path)
    experiment_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    llm_cfg = load_config()
    client = make_client(llm_cfg)
    backend = (llm_cfg.get("backend") or "mock").lower()

    if backend == "vllm" and not client.health_check():
        raise RuntimeError("vLLM gate run requested, but the configured server is not healthy")

    tracker_by_id = _tracker_cache(dataset, client, llm_cfg)
    generation_cfg = experiment_cfg["generation"]
    low_confidence_threshold = experiment_cfg["low_confidence_threshold"]
    candidates = experiment_cfg["candidate_signals"]
    seed_results = []
    sample_results = []

    for seed in experiment_cfg["seeds"]:
        per_signal = {name: [] for name in candidates}
        labels = []
        for row in dataset:
            generation = client.generate(
                _messages(row["question"]),
                temperature=generation_cfg["temperature"],
                top_p=generation_cfg["top_p"],
                max_tokens=generation_cfg["max_tokens"],
                seed=seed,
            )
            snapshot = build_signal_snapshot(
                generation=generation,
                tracker_slots={},
                tracker_cfg=llm_cfg,
                low_confidence_threshold=low_confidence_threshold,
            )
            signals = dict(snapshot["model"]["uncertainty"])
            signals["tracker_unknown_fraction"] = tracker_by_id[row["id"]][
                "unknown_fraction"
            ]
            label = int(row["retrieval_needed"])
            labels.append(label)
            for name in candidates:
                per_signal[name].append(signals.get(name))

            sample_results.append(
                {
                    "seed": seed,
                    "id": row["id"],
                    "retrieval_needed": bool(label),
                    "expected_answer": row["expected_answer"],
                    "model_answer": generation.text,
                    "signals": signals,
                    "tracker": tracker_by_id[row["id"]],
                }
            )

        aucs = {}
        for name, scores in per_signal.items():
            available_labels = [
                label for label, score in zip(labels, scores) if score is not None
            ]
            available_scores = [score for score in scores if score is not None]
            if set(available_labels) == {0, 1}:
                aucs[name] = roc_auc(available_labels, available_scores)
        seed_results.append({"seed": seed, "auc_by_signal": aucs})

    aggregate = {}
    for name in candidates:
        values = [
            result["auc_by_signal"][name]
            for result in seed_results
            if name in result["auc_by_signal"]
        ]
        if values:
            aggregate[name] = {
                "mean_auc": statistics.fmean(values),
                "min_auc": min(values),
                "max_auc": max(values),
                "seed_aucs": values,
            }

    best_signal = max(aggregate, key=lambda name: aggregate[name]["mean_auc"])
    best_auc = aggregate[best_signal]["mean_auc"]
    valid_gate_run = backend == "vllm"
    if not valid_gate_run:
        verdict = "NOT_EVALUATED_SMOKE"
    elif best_auc >= experiment_cfg["success_auc"]:
        verdict = "PASS"
    elif best_auc < experiment_cfg["kill_auc"]:
        verdict = "PIVOT"
    else:
        verdict = "INCONCLUSIVE"

    raw_dataset = dataset_path.read_bytes()
    return {
        "experiment_id": experiment_cfg["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model": client.model,
        "valid_gate_run": valid_gate_run,
        "verdict": verdict,
        "smoke_warning": (
            "Mock logprobs are synthetic. This run validates plumbing only and "
            "must not be cited as Experiment 1 evidence."
            if not valid_gate_run
            else None
        ),
        "thresholds": {
            "success_auc": experiment_cfg["success_auc"],
            "kill_auc": experiment_cfg["kill_auc"],
        },
        "dataset": {
            "path": str(dataset_path.relative_to(_REPO_ROOT)),
            "sha256": hashlib.sha256(raw_dataset).hexdigest(),
            "examples": len(dataset),
            "retrieval_needed": sum(row["retrieval_needed"] for row in dataset),
            "retrieval_not_needed": sum(
                not row["retrieval_needed"] for row in dataset
            ),
        },
        "best_signal": best_signal,
        "best_mean_auc": best_auc,
        "aggregate_auc": aggregate,
        "seed_results": seed_results,
        "samples": sample_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run(args.dataset, args.config)
    output = args.output
    if output is None:
        mode = "gate" if result["valid_gate_run"] else "smoke"
        output = _HERE / "results" / f"{mode}_{result['backend']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Experiment: {result['experiment_id']}")
    print(f"Backend:    {result['backend']} ({result['model']})")
    print(f"Dataset:    {result['dataset']['examples']} balanced examples")
    print(f"Best AUC:   {result['best_mean_auc']:.4f} ({result['best_signal']})")
    print(f"Verdict:    {result['verdict']}")
    print(f"Result:     {output}")
    if result["smoke_warning"]:
        print(f"WARNING:    {result['smoke_warning']}")


if __name__ == "__main__":
    main()
