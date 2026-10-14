"""E1-v2 empirical retrieval-need labels (and retrieval-path gains).

The authored label says whether an item SHOULD need a document. The
empirical label says whether THIS MODEL actually gets it wrong closed-book:

    P(wrong) = fraction of the full-question closed-book draws (one per seed)
               graded incorrect
    empirical_need = P(wrong) >= 0.5          (labels.need_threshold_p_wrong)

Grading, per dataset (config.json labels.correctness_source):
  d1_open     normalized alias match against expected_answer + answer_aliases
              (primary); both LLM judges are also run and reported (secondary
              evidence and an agreement check).
  d2_support  the primary LLM judge with the item's key facts; the secondary
              judge is run on the same answers for agreement.

With retrieval records (run.py --with-retrieval) each item also gets the
grounded answers' P(wrong) and the gain  g = acc(retrieve) - acc(closed book),
which analyze.py turns into g+ (needed items) and g- (not-needed items).

Labels are per model: P(wrong) is a property of the model that answered.

  python -m llm.experiment1_v2.label --dataset d1_open --split dev --model mistral7b_v03
"""

from __future__ import annotations

import argparse
import re
import string
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.experiment1_v2 import common  # noqa: E402
from llm.experiment1_v2 import judges as judges_mod  # noqa: E402
from simulator.judge_validation.score_judges import agreement  # noqa: E402

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation + "‘’“”"})


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, punctuation to spaces, drop
    articles, collapse whitespace."""
    text = (text or "").lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def alias_match(answer: str, aliases) -> bool:
    """True if any normalized alias occurs in the normalized answer on token
    boundaries ('1999' matches 'in 1999.', not '19990')."""
    padded = f" {normalize_answer(answer)} "
    for alias in aliases:
        norm = normalize_answer(alias)
        if norm and f" {norm} " in padded:
            return True
    return False


def item_aliases(item: dict) -> list:
    return [item["expected_answer"]] + list(item.get("answer_aliases") or [])


def item_key_facts(item: dict) -> list:
    return list(item.get("key_facts") or (item.get("meta") or {}).get("key_facts") or [])


def answer_text(record: dict) -> str:
    if record.get("condition") == "retrieval":
        return record.get("grounded_answer") or ""
    return record.get("closed_book_answer") or ""


def is_full_text(record: dict) -> bool:
    prefix = record.get("prefix") or {}
    return prefix.get("kind") == "full" and prefix.get("source") == "text"


def p_wrong(values) -> dict:
    scored = [v for v in values if v is not None]
    n_wrong = sum(1 for v in scored if not v)
    return {
        "n_draws": len(values), "n_scored": len(scored), "n_wrong": n_wrong,
        "p_wrong": (n_wrong / len(scored)) if scored else None,
    }


def empirical_need(p, threshold: float):
    return None if p is None else bool(p >= threshold)


def grade_draws(records, items_by_id, judges, cfg, cache_dir, workers) -> list:
    """One row per graded draw: alias match + each judge's verdict."""
    contexts = cfg["labels"]["judge_context"]

    def one(record):
        item = items_by_id[record["item_id"]]
        answer = answer_text(record)
        row = {
            "item_id": record["item_id"], "draw_seed": record["draw_seed"],
            "condition": record.get("condition", "closed_book"), "answer": answer,
            "alias_correct": alias_match(answer, item_aliases(item)), "judges": {},
        }
        for role, judge in judges.items():
            verdict = judges_mod.grade(
                judge, item_id=item["id"], question=item["question"],
                reference=item["expected_answer"], answer=answer,
                key_facts=item_key_facts(item), aliases=item.get("answer_aliases") or [],
                context=contexts.get(item["dataset"], contexts["default"]), cache_dir=cache_dir,
            )
            row["judges"][role] = {
                "answer_correct": verdict["answer_correct"] if verdict["parsed_ok"] else None,
                "parsed_ok": verdict["parsed_ok"],
                "n_samples": verdict["n_samples"],
                "self_consistency": verdict["self_consistency"],
                "unanimous": verdict["unanimous"],
                "judge_key": verdict["judge_key"],
                "judge": verdict["judge"],
                "request_config": verdict["request_config"],
                "response_model": verdict.get("response_model"),
                "cache_hit": verdict.get("cache_hit"),
                "samples": [
                    {k: s.get(k) for k in ("sample_index", "seed", "answer_correct", "parsed_ok",
                                           "judge_raw", "response_model", "sent_params", "usage",
                                           "sample_key")}
                    for s in verdict["samples"]
                ],
            }
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, records))


def correctness(row: dict, source: str):
    if source == "alias":
        return row["alias_correct"]
    if source.startswith("judge:"):
        role = source.split(":", 1)[1]
        if role not in row["judges"]:
            raise ValueError(f"correctness source {source!r} needs the {role} judge enabled")
        return row["judges"][role]["answer_correct"]
    raise ValueError(f"unknown correctness source {source!r}")


def build_labels(items, rows, cfg, source_by_dataset, threshold, roles) -> dict:
    methods = ["alias"] + [f"judge:{r}" for r in roles]
    by_item: dict = {}
    for row in rows:
        by_item.setdefault(row["item_id"], {"closed_book": [], "retrieval": []})[
            row["condition"]].append(row)
    out = {}
    for item in items:
        source = source_by_dataset(item["dataset"])
        groups = by_item.get(item["id"], {"closed_book": [], "retrieval": []})
        entry = {
            "dataset": item["dataset"], "item_type": item.get("item_type"),
            "pair_id": item.get("pair_id"), "domain": item.get("domain"),
            "popularity": item.get("popularity"),
            "label_authored": item["retrieval_needed_authored"],
            "correctness_source": source,
        }
        for condition in ("closed_book", "retrieval"):
            draws = sorted(groups[condition], key=lambda r: r["draw_seed"])
            if not draws:
                entry[condition] = None
                continue
            summary = p_wrong([correctness(r, source) for r in draws])
            summary["p_wrong_by_method"] = {
                m: p_wrong([correctness(r, m) for r in draws])["p_wrong"] for m in methods
            }
            summary["accuracy"] = None if summary["p_wrong"] is None else 1 - summary["p_wrong"]
            summary["seeds"] = [r["draw_seed"] for r in draws]
            entry[condition] = summary
        cb = entry["closed_book"]
        entry["empirical_need"] = empirical_need(cb["p_wrong"] if cb else None, threshold)
        entry["empirical_need_by_method"] = {
            m: empirical_need(cb["p_wrong_by_method"][m], threshold) for m in methods
        } if cb else None
        ret = entry["retrieval"]
        entry["gain"] = (ret["accuracy"] - cb["accuracy"]) if (
            cb and ret and cb["accuracy"] is not None and ret["accuracy"] is not None) else None
        out[item["id"]] = entry
    return out


def agreement_report(rows, roles) -> dict:
    methods = {"alias": [r["alias_correct"] for r in rows]}
    for role in roles:
        methods[f"judge:{role}"] = [r["judges"][role]["answer_correct"] for r in rows]
    names = list(methods)
    report = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report[f"{a} vs {b}"] = agreement(methods[a], methods[b])
    report["parse_failures"] = {
        role: sum(1 for r in rows if not r["judges"][role]["parsed_ok"]) for role in roles
    }
    report["self_consistency"] = {role: self_consistency(rows, role) for role in roles}
    return report


def self_consistency(rows, role) -> dict:
    """How often a stochastic judge agrees with itself across its k samples
    on the same answer: mean share agreeing with the majority, fraction
    unanimous, and pairwise percent agreement / kappa between sample i and j.
    Also the reasoning-token usage the samples reported."""
    judged = [r["judges"][role] for r in rows]
    k = max((j["n_samples"] for j in judged), default=0)
    out = {"n_answers": len(judged), "k": k}
    if k <= 1:
        out["note"] = "single-sample judge; self-consistency not measured"
    else:
        shares = [j["self_consistency"] for j in judged if j["self_consistency"] is not None]
        out["mean_majority_share"] = sum(shares) / len(shares) if shares else None
        out["fraction_unanimous"] = sum(1 for j in judged if j["unanimous"]) / len(judged) if judged else None
        pairs = {}
        for a in range(k):
            for b in range(a + 1, k):
                pairs[f"{a}-{b}"] = agreement(
                    [j["samples"][a]["answer_correct"] if len(j["samples"]) > a else None for j in judged],
                    [j["samples"][b]["answer_correct"] if len(j["samples"]) > b else None for j in judged])
        out["pairwise_samples"] = pairs
    reasoning, completion = [], []
    for j in judged:
        for s in j["samples"]:
            usage = s.get("usage") or {}
            completion.append(usage.get("completion_tokens"))
            reasoning.append((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    reasoning = [v for v in reasoning if v is not None]
    completion = [v for v in completion if v is not None]
    out["usage"] = {
        "samples_with_usage": len(completion),
        "completion_tokens_total": sum(completion) if completion else None,
        "reasoning_tokens_total": sum(reasoning) if reasoning else None,
        "reasoning_tokens_max": max(reasoning) if reasoning else None,
    }
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--items", type=Path)
    p.add_argument("--splits", type=Path)
    p.add_argument("--split", required=True, choices=common.VALID_SPLITS)
    p.add_argument("--model", required=True, help="model key whose answers are graded")
    p.add_argument("--llm-backend", choices=("vllm", "mock"), default="vllm",
                   help="which results tree to read (results/ or results_mock/)")
    p.add_argument("--results-dir", type=Path, help="override results/<dataset>/<model_key>/")
    p.add_argument("--judges", default="primary,secondary",
                   help="comma list of judge roles, or 'none' (alias match only)")
    p.add_argument("--judge-backend", choices=("azure", "mock"), default="azure")
    p.add_argument("--judges-config", type=Path, default=common.DEFAULT_JUDGES_CONFIG)
    p.add_argument("--correctness", help="override correctness source for every dataset "
                   "(alias | judge:primary | judge:secondary)")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    p.add_argument("--no-retrieval", action="store_true", help="ignore retrieval records")
    p.add_argument("--allow-real-judge-on-mock", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        common.check_split_allowed(args.split)
    except common.SealedSplitError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    cfg = common.load_config(args.config)
    paths = common.resolve_dataset_paths(args.dataset, args.items, args.splits)
    items = common.select_split(common.load_items(paths["items"]),
                                common.load_splits(paths["splits"]), args.split)
    dataset = args.dataset or items[0]["dataset"]
    mock_tree = args.llm_backend == "mock"
    rdir = args.results_dir or common.results_dir(dataset, args.model, mock_tree)
    signals_path = rdir / f"signals_{args.split}.jsonl"
    retrieval_path = rdir / f"retrieval_{args.split}.jsonl"
    records = [r for r in common.read_jsonl(signals_path) if is_full_text(r)]
    if not records:
        print(f"no full-question closed-book records in {common.rel(signals_path)}", file=sys.stderr)
        return 1
    if not args.no_retrieval:
        records += [r for r in common.read_jsonl(retrieval_path) if is_full_text(r)]
    backends = {r.get("backend") for r in records}
    if "mock" in backends and args.judge_backend == "azure" and not args.allow_real_judge_on_mock:
        print("REFUSED: these are mock-backend answers; grading them with a paid judge is "
              "pointless. Use --judge-backend mock.", file=sys.stderr)
        return 1

    items_by_id = {i["id"]: i for i in items}
    unknown = {r["item_id"] for r in records} - set(items_by_id)
    if unknown:
        raise ValueError(f"{len(unknown)} record item ids are not in the {args.split} split, "
                         f"e.g. {sorted(unknown)[:3]}")
    roles = [] if args.judges.strip().lower() == "none" else \
        [r.strip() for r in args.judges.split(",") if r.strip()]
    judges_cfg = judges_mod.load_judges_config(args.judges_config)
    try:
        judges = judges_mod.make_judges(roles, args.judge_backend, judges_cfg)
    except judges_mod.JudgeConfigError as error:
        print(f"[label] {error}", file=sys.stderr)
        return 1
    cache_dir = args.cache_dir or rdir.parent / "judge_cache"

    def source_for(ds):
        if args.correctness:
            return args.correctness
        sources = cfg["labels"]["correctness_source"]
        return sources.get(ds, sources["default"])

    for ds in {i["dataset"] for i in items}:
        src = source_for(ds)
        if src.startswith("judge:") and src.split(":", 1)[1] not in roles:
            print(f"[label] {ds} is graded by {src}, which is not enabled (--judges {args.judges}).",
                  file=sys.stderr)
            return 1

    print(f"[label] grading {len(records)} answers ({common.rel(rdir)}) with alias match"
          + "".join(f" + {r}={judges[r].spec.identity_string()}" for r in roles))
    rows = grade_draws(records, items_by_id, judges, cfg, cache_dir, args.workers)
    threshold = float(cfg["labels"]["need_threshold_p_wrong"])
    graded_ids = {r["item_id"] for r in rows}
    labelled_items = [i for i in items if i["id"] in graded_ids]
    labels = build_labels(labelled_items, rows, cfg, source_for, threshold, roles)

    closed = [r for r in rows if r["condition"] == "closed_book"]
    grounded = [r for r in rows if r["condition"] == "retrieval"]
    output = {
        "_comment": "GENERATED by python -m llm.experiment1_v2.label. Empirical retrieval-need "
                    "labels for one model on one split; not hand-editable.",
        "meta": {
            "created_at_utc": common.now_utc(),
            "git_commit": common.git_commit(),
            "dataset": dataset,
            "split": args.split,
            "model_key": args.model,
            "model": records[0]["model"],
            "llm_backend": sorted(b for b in backends if b),
            "need_threshold_p_wrong": threshold,
            "rule": "empirical_need = P(wrong over full-question closed-book draws) >= threshold",
            "judges": {r: judges[r].identity for r in roles},
            "judge_request_config": {r: judges[r].spec.request for r in roles},
            "judge_adaptations": {r: getattr(judges[r], "adaptations", []) for r in roles},
            "judge_backend": args.judge_backend,
            "inputs": {
                "signals": common.rel(signals_path),
                "signals_sha256": common.sha256_file(signals_path),
                "retrieval": common.rel(retrieval_path) if grounded else None,
                "retrieval_sha256": common.sha256_file(retrieval_path) if grounded else None,
                "items_sha256": common.sha256_file(paths["items"]),
                "splits_sha256": common.sha256_file(paths["splits"]),
            },
            "n_items": len(labels),
            "n_closed_book_draws": len(closed),
            "n_retrieval_draws": len(grounded),
        },
        "agreement": {
            "closed_book": agreement_report(closed, roles),
            "retrieval": agreement_report(grounded, roles) if grounded else None,
        },
        "items": labels,
        "draws": rows,
    }
    out_path = rdir / f"labels_{args.split}.json"
    common.atomic_write_json(out_path, output)

    need = [v["empirical_need"] for v in labels.values() if v["empirical_need"] is not None]
    authored = [v["label_authored"] for v in labels.values() if v["empirical_need"] is not None]
    print(f"[label] {len(labels)} items; empirical need {sum(need)}/{len(need)}; "
          f"authored need {sum(authored)}/{len(authored)}")
    print(f"[label] authored vs empirical: {agreement(authored, need)}")
    for name, stats in output["agreement"]["closed_book"].items():
        if name not in ("parse_failures", "self_consistency"):
            print(f"[label] {name}: {stats['agree']}/{stats['n']} kappa={stats['cohen_kappa']}")
    print(f"[label] parse failures: {output['agreement']['closed_book']['parse_failures']}")
    for role, sc in output["agreement"]["closed_book"]["self_consistency"].items():
        if sc["k"] > 1:
            print(f"[label] {role} self-consistency (k={sc['k']}): mean majority share "
                  f"{sc['mean_majority_share']}, unanimous {sc['fraction_unanimous']}, "
                  f"reasoning tokens {sc['usage']['reasoning_tokens_total']}")
    print(f"[label] wrote {common.rel(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
