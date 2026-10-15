"""E1-v2 empirical retrieval-need labels (and retrieval-path gains).

The authored label says whether an item SHOULD need a document. The
empirical label says whether THIS MODEL actually gets it wrong closed-book:

    P(wrong) = fraction of the full-question closed-book draws (one per seed)
               graded incorrect
    empirical_need = wrong in >= 3 of the 5 draws (labels.need_rule; PREREG_V2 s1)

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
import json
import math
import re
import string
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.experiment1_v2 import common  # noqa: E402
from llm.experiment1_v2 import judges as judges_mod  # noqa: E402
from simulator.judge_validation.score_judges import agreement, fleiss_kappa  # noqa: E402

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


def item_core_facts(item: dict):
    """meta.core_facts (r3), or None when the item is not annotated (the
    rubric then falls back to the first key fact; see core_fact_warnings)."""
    facts = item.get("core_facts") or (item.get("meta") or {}).get("core_facts")
    return list(facts) if facts else None


def core_fact_warnings(items, rubric_version: str) -> list:
    """Item ids graded under a core-fact rubric without meta.core_facts."""
    from llm.experiment1_v2.rubrics import get_rubric  # noqa: PLC0415

    rubric = get_rubric(rubric_version)
    if rubric is None:
        return []
    return [i["id"] for i in items if rubric.uses_core_facts(i["dataset"]) and not item_core_facts(i)]


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


def empirical_need(n_wrong: int, n_scored: int, n_draws: int, rule: dict):
    """PREREG_V2 section 1: needed iff the closed-book answer is wrong in at
    least `min_wrong` of `of_draws` draws (3 of 5). Draws with no valid
    verdict are unscored: the label is None when they could still decide it.
    With a different number of draws (pilots) the bar is scaled
    proportionally (ceil(n_draws * 3 / 5))."""
    min_wrong = int(rule["min_wrong"])
    if n_draws != int(rule["of_draws"]):
        min_wrong = math.ceil(n_draws * rule["min_wrong"] / rule["of_draws"])
    if n_draws == 0:
        return None
    if n_wrong >= min_wrong:
        return True
    if n_wrong + (n_draws - n_scored) < min_wrong:
        return False
    return None


def draw_key(record) -> tuple:
    return (record["item_id"], record["draw_seed"], record.get("condition", "closed_book"))


def popularity_of(item):
    if item.get("popularity") is not None:
        return float(item["popularity"])
    value = (item.get("meta") or {}).get("log10_pop")
    return None if value is None else float(value)


def judge_subsample(records, items_by_id, n: int, seed: int) -> set:
    """Draw keys of a stratified random subsample for the LLM judges.

    For datasets graded by alias match (d1_open), the judges serve only as
    an agreement check, and grading every answer with k=3 gpt-6.1-sol
    samples would run into the Azure rate limit for hours. Strata are
    item_type x popularity tercile x alias verdict, allocated in proportion
    to their size with at least one draw each, seeded."""
    import random  # noqa: PLC0415

    pops = sorted(p for p in (popularity_of(items_by_id[r["item_id"]]) for r in records) if p is not None)
    cuts = [pops[len(pops) // 3], pops[2 * len(pops) // 3]] if len(pops) >= 3 else None

    def stratum(record):
        item = items_by_id[record["item_id"]]
        pop = popularity_of(item)
        tercile = None if pop is None or cuts is None else sum(pop >= c for c in cuts)
        alias = alias_match(answer_text(record), item_aliases(item))
        return (item.get("item_type"), tercile, alias)

    groups: dict = {}
    for record in sorted(records, key=draw_key):
        groups.setdefault(stratum(record), []).append(record)
    rng = random.Random(seed)
    chosen = set()
    total = len(records)
    for key in sorted(groups, key=str):
        members = groups[key]
        take = max(1, round(n * len(members) / total))
        for record in rng.sample(members, min(take, len(members))):
            chosen.add(draw_key(record))
    return chosen


def grade_draws(records, items_by_id, judges, cfg, cache_dir, workers, judge_keys=None) -> list:
    """One row per graded draw: alias match + each judge's verdict. With
    `judge_keys`, only those draws are sent to the LLM judges."""
    contexts = cfg["labels"]["judge_context"]

    def one(record):
        item = items_by_id[record["item_id"]]
        answer = answer_text(record)
        row = {
            "item_id": record["item_id"], "draw_seed": record["draw_seed"],
            "condition": record.get("condition", "closed_book"), "answer": answer,
            "alias_correct": alias_match(answer, item_aliases(item)), "judges": {},
        }
        if judge_keys is not None and draw_key(record) not in judge_keys:
            return row
        for role, judge in judges.items():
            verdict = judges_mod.grade(
                judge, item_id=item["id"], question=item["question"],
                reference=item["expected_answer"], answer=answer,
                key_facts=item_key_facts(item), aliases=item.get("answer_aliases") or [],
                context=contexts.get(item["dataset"], contexts["default"]), cache_dir=cache_dir,
                dataset=item["dataset"], history=item.get("history") or [],
                core_facts=item_core_facts(item),
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
                "n_extra_samples": verdict.get("n_extra_samples", 0),
                # The exact request body of each sample lives in its cache file
                # (judge_cache/<judge>/<sample_key[:40]>.json); its hash is here.
                "samples": [
                    {**{k: s.get(k) for k in (
                        "sample_index", "seed", "extra_sample", "answer_correct", "parsed_ok",
                        "judge_raw", "deployment", "response_model", "response_id", "finish_reason",
                        "sent_params", "usage", "sample_key", "request_body_captured")},
                     "request_body_sha256": common.sha256_text(
                         json.dumps(s.get("request_body"), sort_keys=True))}
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
            return None  # not judged (judge disabled, or outside the judge subsample)
        return row["judges"][role]["answer_correct"]
    raise ValueError(f"unknown correctness source {source!r}")


def build_labels(items, rows, cfg, source_by_dataset, rule, roles) -> dict:
    methods = ["alias"] + [f"judge:{r}" for r in roles]

    def need_of(values):
        s = p_wrong(values)
        return empirical_need(s["n_wrong"], s["n_scored"], s["n_draws"], rule)
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
            values = [correctness(r, source) for r in draws]
            summary = p_wrong(values)
            summary["p_wrong_by_method"] = {
                m: p_wrong([correctness(r, m) for r in draws])["p_wrong"] for m in methods
            }
            summary["accuracy"] = None if summary["p_wrong"] is None else 1 - summary["p_wrong"]
            # Sensitivity analysis (PREREG_V2 section 9): an answer with no
            # valid judge majority counted INCORRECT instead of excluded.
            sens = p_wrong([False if v is None else v for v in values])
            summary["p_wrong_unresolved_as_wrong"] = sens["p_wrong"]
            summary["n_unresolved"] = summary["n_draws"] - summary["n_scored"]
            summary["seeds"] = [r["draw_seed"] for r in draws]
            entry[condition] = summary
        cb = entry["closed_book"]
        if cb:
            draws = sorted(groups["closed_book"], key=lambda r: r["draw_seed"])
            values = [correctness(r, source) for r in draws]
            entry["empirical_need"] = need_of(values)
            entry["empirical_need_unresolved_as_wrong"] = need_of(
                [False if v is None else v for v in values])
            entry["empirical_need_by_method"] = {
                m: need_of([correctness(r, m) for r in draws]) for m in methods}
        else:
            entry["empirical_need"] = entry["empirical_need_unresolved_as_wrong"] = None
            entry["empirical_need_by_method"] = None
        ret = entry["retrieval"]
        entry["gain"] = (ret["accuracy"] - cb["accuracy"]) if (
            cb and ret and cb["accuracy"] is not None and ret["accuracy"] is not None) else None
        out[item["id"]] = entry
    return out


def agreement_report(rows, roles) -> dict:
    # Only draws every enabled judge graded (all of them, unless subsampled).
    rows = [r for r in rows if all(role in r["judges"] for role in roles)]
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
        out["fleiss"] = fleiss_kappa([[s["answer_correct"] for s in j["samples"][:k]] for j in judged])
        out["n_with_extra_samples"] = sum(1 for j in judged if j.get("n_samples", 0) > k)
    reasoning, completion, finish = [], [], []
    for j in judged:
        for s in j["samples"]:
            usage = s.get("usage") or {}
            completion.append(usage.get("completion_tokens"))
            reasoning.append((usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            finish.append(s.get("finish_reason"))
    reasoning = [v for v in reasoning if v is not None]
    completion = [v for v in completion if v is not None]
    known = [f for f in finish if f is not None]
    out["usage"] = {
        "samples_with_usage": len(completion),
        "completion_tokens_total": sum(completion) if completion else None,
        "completion_tokens_max": max(completion) if completion else None,
        "reasoning_tokens_total": sum(reasoning) if reasoning else None,
        "reasoning_tokens_max": max(reasoning) if reasoning else None,
        # PREREG_V2 2.2: max_completion_tokens is fixed so that < 1% truncate.
        "truncation_rate": (sum(f == "length" for f in known) / len(known)) if known else None,
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
    p.add_argument("--rubric", help="judge prompt version (default: judges.json prompt_version, "
                   "e1v2-r2; e1v2-reasoned-strip-r1 reproduces the first grading)")
    p.add_argument("--correctness", help="override correctness source for every dataset "
                   "(alias | judge:primary | judge:secondary)")
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    p.add_argument("--no-retrieval", action="store_true", help="ignore retrieval records")
    p.add_argument("--allow-real-judge-on-mock", action="store_true")
    p.add_argument("--judge-subsample", type=int,
                   help="send only a stratified random subsample of N answers to the LLM "
                   "judges (agreement check). Allowed only when correctness is alias match "
                   "(d1_open); use --judges none to skip the judges entirely")
    p.add_argument("--judge-subsample-seed", type=int, default=20261015)
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
    if ("mock" in backends and args.judge_backend == "azure" and args.judges.strip().lower() != "none"
            and not args.allow_real_judge_on_mock):
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
    if args.rubric:
        judges_cfg["prompt_version"] = args.rubric
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

    graded_items = [items_by_id[i] for i in sorted({r["item_id"] for r in records})]
    no_core = core_fact_warnings(graded_items, judges_cfg["prompt_version"]) if roles else []
    if no_core:
        print(f"[label] WARNING: {len(no_core)} item(s) have no meta.core_facts; rubric "
              f"{judges_cfg['prompt_version']} grades them against their FIRST key fact "
              f"(e.g. {no_core[:3]}).", file=sys.stderr)

    judge_keys = None
    if args.judge_subsample and roles:
        judged_sources = {source_for(ds) for ds in {i["dataset"] for i in items}}
        if any(s.startswith("judge:") for s in judged_sources):
            print("REFUSED: --judge-subsample would leave judge-graded labels incomplete; it is "
                  "only for alias-graded datasets (d1_open).", file=sys.stderr)
            return 1
        judge_keys = judge_subsample(records, items_by_id, args.judge_subsample,
                                     args.judge_subsample_seed)
    print(f"[label] grading {len(records)} answers ({common.rel(rdir)}) with alias match"
          + "".join(f" + {r}={judges[r].spec.identity_string()}" for r in roles)
          + (f" (judges on a stratified subsample of {len(judge_keys)})" if judge_keys else ""))
    try:
        rows = grade_draws(records, items_by_id, judges, cfg, cache_dir, args.workers, judge_keys)
    except judges_mod.JudgeBudgetExceeded as error:
        print(f"[label] STOPPED: {error}", file=sys.stderr)
        print("[label] No labels file was written. Every verdict graded so far is cached; "
              "re-running the same command resumes from the cache.", file=sys.stderr)
        return 4
    rule = cfg["labels"]["need_rule"]
    graded_ids = {r["item_id"] for r in rows}
    labelled_items = [i for i in items if i["id"] in graded_ids]
    labels = build_labels(labelled_items, rows, cfg, source_for, rule, roles)

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
            "need_rule": rule,
            "rule": "empirical_need = closed-book answer to the full question wrong in >= "
                    f"{rule['min_wrong']} of {rule['of_draws']} draws (PREREG_V2 section 1); "
                    "draws without a valid verdict are excluded (sensitivity: counted wrong)",
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
            "judge_subsample": None if judge_keys is None else {
                "n_requested": args.judge_subsample, "n_judged": len(judge_keys),
                "seed": args.judge_subsample_seed,
                "strata": "item_type x popularity tercile x alias verdict, proportional, >= 1 each",
            },
            "core_facts_fallback": {"n_items": len(no_core), "item_ids": no_core,
                                    "rule": "graded against key_facts[0] (no meta.core_facts)"},
            "authored_labels": ("present" if any(i.get("retrieval_needed_authored") is not None
                                                 for i in labelled_items)
                                else "not applicable (dataset has no authored labels)"),
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
    # A rubric-tagged copy, so gradings under different rubrics coexist
    # (analyze.py --labels picks one; labels_<split>.json is the latest).
    common.atomic_write_json(rdir / f"labels_{args.split}.{judges_cfg['prompt_version']}.json", output)

    need = [v["empirical_need"] for v in labels.values() if v["empirical_need"] is not None]
    # Authored labels can be null by design (d1_open): only items with both count.
    both = [(v["label_authored"], v["empirical_need"]) for v in labels.values()
            if v["empirical_need"] is not None and v["label_authored"] is not None]
    print(f"[label] {len(labels)} items; empirical need {sum(need)}/{len(need)}")
    if both:
        print(f"[label] authored need {sum(a for a, _ in both)}/{len(both)}; authored vs empirical: "
              f"{agreement([a for a, _ in both], [e for _, e in both])}")
    else:
        print("[label] authored vs empirical: not applicable (no authored labels)")
    for name, stats in output["agreement"]["closed_book"].items():
        if name not in ("parse_failures", "self_consistency") and stats["n"]:
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
