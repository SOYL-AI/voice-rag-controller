"""E1-v2 runner: uncertainty signals on PARTIAL transcripts, per model.

For every item in the chosen split, every seed, and every prefix condition
(full text, word-fraction text prefixes, and each ASR partial of the selected
speech traces), generate a closed-book answer under v1 E1's protocol and
record the signals, the answer and the measured latency (TTFT, generation
time, arrival time of token k). Output is one JSONL record per (item, seed,
prefix condition) in the CONTRACTS.md section 5 schema.

  python -m llm.experiment1_v2.run --dataset d1_open --split dev --model mistral7b_v03
  python -m llm.experiment1_v2.run --dataset d2_support --split dev --model llama31_8b \
      --with-retrieval

Properties:
  * sealed test split: --split test is refused unless
    docs/research_v2/PREREG_FROZEN exists (checked before anything loads);
  * resumable: records already in the output are skipped, a generation
    already on disk is reused for any of its missing sibling conditions, and
    failed calls go to errors_<split>.jsonl and are retried on the next run;
  * one model call per distinct prefix text per (item, seed): conditions that
    share a text share the call (reused_generation=true on the copies);
  * bounded concurrency (asyncio semaphore) with retried transient errors;
  * the endpoint URL and API key are read from the environment by
    llm/config.py and never written to a record, the manifest or a log line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.config import load_config as load_llm_config  # noqa: E402
from llm.experiment1_v2 import common  # noqa: E402
from llm.experiment1_v2.backends import GenerationError, make_backend  # noqa: E402
from llm.experiment1_v2.prefixes import group_by_text, item_prefixes  # noqa: E402
from llm.experiment1_v2.prompts import build_messages  # noqa: E402
from llm.experiment1_v2.retrieval_adapter import make_retriever  # noqa: E402
from llm.experiment1_v2.signals_v2 import compute_signals, first_k_arrival_ms  # noqa: E402

SYNTHETIC_VOICE_PREFIX = "synthetic"


@dataclass
class Unit:
    """One model call: an (item, seed, distinct text) and the prefix
    conditions that share it."""

    item: dict
    seed: int
    text: str
    prefixes: list
    messages: list
    call_key: str
    condition: str = "closed_book"
    retrieval: dict | None = None
    record_keys: list = field(default_factory=list)


def record_key(item_id, model, seed, prefix_id, condition) -> str:
    return f"{condition}|{model}|{item_id}|{seed}|{prefix_id}"


def build_units(items, traces_by_item, cfg, seeds, model, *, use_text=True, use_asr=True,
                condition="closed_book", retrieved=None) -> list:
    units = []
    for item in items:
        if condition == "retrieval":
            prefixes = item_prefixes(item, word_fracs=(), text=True, asr=False)
        else:
            prefixes = item_prefixes(
                item, traces_by_item.get(item["id"]), cfg["word_fracs"], text=use_text,
                asr=use_asr, include_asr_final=cfg["traces"].get("include_asr_final", True),
            )
        hits = (retrieved or {}).get(item["id"])
        for text, group in group_by_text(prefixes):
            passages = hits["docs"] if hits is not None else None
            messages = build_messages(item, text, cfg, passages=passages)
            for seed in seeds:
                key = common.sha256_text(condition, model, seed, json.dumps(messages, sort_keys=True))[:24]
                unit = Unit(item, seed, text, group, messages, key, condition, hits)
                unit.record_keys = [
                    record_key(item["id"], model, seed, p["prefix_id"], condition) for p in group
                ]
                units.append(unit)
    return units


def _base_record(unit: Unit, model: str, model_key: str, backend: str, split: str) -> dict:
    item = unit.item
    return {
        "item_id": item["id"],
        "dataset": item["dataset"],
        "model": model,
        "model_key": model_key,
        "backend": backend,
        "condition": unit.condition,
        "split": split,
        "draw_seed": unit.seed,
        "label_authored": item["retrieval_needed_authored"],
        "item_type": item.get("item_type"),
        "pair_id": item.get("pair_id"),
        "domain": item.get("domain"),
        "call_key": unit.call_key,
    }


def records_from_result(unit, result, cfg, model, model_key, backend, split, missing) -> list:
    gen = result.generation
    signals = compute_signals(gen, cfg["first_k"], cfg["protocol"]["low_confidence_threshold"])
    payload = {
        "signals": signals,
        "ttft_ms": result.ttft_ms,
        "gen_ms": result.gen_ms,
        "n_tokens": len(gen.tokens),
        "t_first_k_ms": first_k_arrival_ms(result.token_arrival_ms, cfg["first_k"]),
        "finish_reason": result.finish_reason,
        "served_model": result.served_model,
        "attempts": result.attempts,
        "timing_synthetic": result.timing_synthetic,
    }
    if unit.condition == "retrieval":
        payload["grounded_answer"] = gen.text
        payload["retrieval"] = {k: v for k, v in unit.retrieval.items() if k != "docs"}
        payload["retrieval"]["docs"] = [
            {"doc_id": d.get("doc_id"), "score": d.get("score"), "score_kind": d.get("score_kind")}
            for d in unit.retrieval["docs"]
        ]
    else:
        payload["closed_book_answer"] = gen.text
    return _expand(unit, payload, model, model_key, backend, split, missing, reused=False)


def _expand(unit, payload, model, model_key, backend, split, missing, reused) -> list:
    out = []
    for index, prefix in enumerate(unit.prefixes):
        key = unit.record_keys[index]
        if key not in missing:
            continue
        record = _base_record(unit, model, model_key, backend, split)
        record["prefix"] = {k: v for k, v in prefix.items() if k != "prefix_id"}
        record["prefix_id"] = prefix["prefix_id"]
        record.update(payload)
        record["reused_generation"] = bool(reused or out)
        record["created_at_utc"] = common.now_utc()
        out.append(record)
    return out


def records_from_existing(unit, existing, model, model_key, backend, split, missing) -> list:
    keep = ("signals", "ttft_ms", "gen_ms", "n_tokens", "t_first_k_ms", "finish_reason",
            "served_model", "attempts", "timing_synthetic", "closed_book_answer",
            "grounded_answer", "retrieval")
    payload = {k: existing[k] for k in keep if k in existing}
    return _expand(unit, payload, model, model_key, backend, split, missing, reused=True)


def retrieve_all(items, retriever, k) -> dict:
    out = {}
    for item in items:
        start = time.perf_counter()
        docs = retriever.search(item["question"], k)
        latency = (time.perf_counter() - start) * 1000.0
        gold = set(item.get("gold_doc_ids") or [])
        out[item["id"]] = {
            "retriever": retriever.name,
            "k": k,
            "query": item["question"],
            "latency_ms": latency,
            "gold_hit": bool(gold & {d.get("doc_id") for d in docs}) if gold else None,
            "docs": docs,
        }
    return out


async def execute(units, backend, writer, cfg, *, model, model_key, split, done_keys,
                  existing_by_call, concurrency, error_writer=None, progress_every=25) -> dict:
    gen_cfg = cfg["protocol"]["generation"]
    stats = {"calls": 0, "reused": 0, "skipped": 0, "failed": 0, "records": 0}
    todo = []
    for unit in units:
        missing = {k for k in unit.record_keys if k not in done_keys}
        if not missing:
            stats["skipped"] += 1
            continue
        existing = existing_by_call.get(unit.call_key)
        if existing is not None:
            for record in records_from_existing(unit, existing, model, model_key, backend.name,
                                                split, missing):
                writer.write(record)
                done_keys.add(record_key(record["item_id"], model, unit.seed,
                                         record["prefix_id"], unit.condition))
                stats["records"] += 1
            stats["reused"] += 1
            continue
        todo.append((unit, missing))

    total = len(todo)
    if not total:
        return stats
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    started = time.perf_counter()
    progress_every = max(progress_every, total // 20)

    async def one(unit, missing):
        async with semaphore:
            try:
                result = await backend.generate(
                    unit.messages, seed=unit.seed, temperature=gen_cfg["temperature"],
                    top_p=gen_cfg["top_p"], max_tokens=gen_cfg["max_tokens"],
                    mock_hint={"item": unit.item, "grounded": unit.condition == "retrieval"},
                )
                return unit, missing, result, None
            except GenerationError as error:
                return unit, missing, None, str(error)
            except Exception as error:  # noqa: BLE001 -- recorded, run continues
                return unit, missing, None, backend.redact(f"{type(error).__name__}: {error}")[:600]

    tasks = [asyncio.create_task(one(u, m)) for u, m in todo]
    finished = 0
    for future in asyncio.as_completed(tasks):
        unit, missing, result, error = await future
        finished += 1
        if error is not None:
            stats["failed"] += 1
            if error_writer is not None:
                error_writer.write({
                    "item_id": unit.item["id"], "draw_seed": unit.seed, "call_key": unit.call_key,
                    "condition": unit.condition, "error": error, "at_utc": common.now_utc(),
                })
        else:
            stats["calls"] += 1
            for record in records_from_result(unit, result, cfg, model, model_key, backend.name,
                                              split, missing):
                writer.write(record)
                stats["records"] += 1
            existing_by_call[unit.call_key] = record
        if finished % progress_every == 0 or finished == total:
            elapsed = time.perf_counter() - started
            rate = finished / elapsed if elapsed > 0 else 0.0
            eta = (total - finished) / rate if rate > 0 else float("nan")
            print(f"[e1v2] {finished}/{total} calls  {rate:.2f}/s  failed={stats['failed']}  "
                  f"eta={eta / 60:.1f} min", flush=True)
    return stats


def load_existing(path: Path):
    done, by_call = set(), {}
    for record in common.read_jsonl(path):
        done.add(record_key(record["item_id"], record["model"], record["draw_seed"],
                            record["prefix_id"], record.get("condition", "closed_book")))
        by_call.setdefault(record["call_key"], record)
    return done, by_call


def estimate(n_calls: int, concurrency: int, call_s: float, efficiency: float) -> float:
    """Wall seconds for n calls at a concurrency, given single-stream seconds
    per call and the fraction of linear speed-up batching delivers."""
    effective = max(1.0, concurrency * efficiency)
    return n_calls * call_s / effective


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="dataset name under datasets/v2/ (d1_open, d2_support)")
    p.add_argument("--items", type=Path, help="explicit items.jsonl (default from --dataset)")
    p.add_argument("--splits", type=Path, help="explicit splits.json")
    p.add_argument("--traces-dir", type=Path, help="speech/traces dir (default next to items)")
    p.add_argument("--split", required=True, choices=common.VALID_SPLITS)
    p.add_argument("--model", help="model key in configs/config.yaml llm.models "
                   "(default: VOICE_RAG_MODEL or active_model)")
    p.add_argument("--backend", choices=("vllm", "mock"),
                   help="default: VOICE_RAG_LLM_BACKEND or configs/config.yaml llm.backend")
    p.add_argument("--seeds", help="comma list (default: v1 E1 seeds 11,23,37,53,71)")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--timeout", type=float, help="per-request timeout seconds")
    p.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path, help="default results/<dataset>/<model_key>/ "
                   "(results_mock/ for the mock backend)")
    p.add_argument("--limit", type=int, help="first N items of the split (pilot runs)")
    p.add_argument("--no-text-prefixes", action="store_true",
                   help="only the full text (plus ASR partials)")
    p.add_argument("--no-asr", action="store_true", help="ignore speech traces")
    p.add_argument("--trace-voices", default=None, help="comma list or 'all'")
    p.add_argument("--trace-noise", default=None, help="comma list or 'all' (default: config)")
    p.add_argument("--with-retrieval", action="store_true",
                   help="also generate grounded answers for the full question (dev g+/g-)")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--retriever", default="v2",
                   help="'v2' (retrieval.search:search(query, index_dir, k)), 'lexical' "
                   "(plumbing only), or module:function")
    p.add_argument("--index-dir", type=Path)
    p.add_argument("--corpus", type=Path, help="corpus.jsonl for --retriever lexical")
    p.add_argument("--k", type=int)
    p.add_argument("--dry-run", action="store_true", help="count calls, estimate time, call nothing")
    p.add_argument("--assume-call-s", type=float, default=2.6,
                   help="single-stream seconds per call for --dry-run estimates")
    p.add_argument("--assume-efficiency", type=float, default=0.6,
                   help="batching efficiency for --dry-run estimates")
    p.add_argument("--skip-health-check", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Sealing first: nothing is loaded, built or called for a sealed split.
    try:
        common.check_split_allowed(args.split)
    except common.SealedSplitError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3

    cfg = common.load_config(args.config)
    paths = common.resolve_dataset_paths(args.dataset, args.items, args.splits, args.traces_dir)
    items_all = common.load_items(paths["items"])
    splits = common.load_splits(paths["splits"])
    items = common.select_split(items_all, splits, args.split)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print(f"no items in split {args.split!r}", file=sys.stderr)
        return 1
    dataset = args.dataset or items[0]["dataset"]

    llm_cfg = load_llm_config(model_key=args.model)
    backend_kind = (args.backend or llm_cfg.get("backend") or "mock").lower()
    model, model_key = llm_cfg["model"], llm_cfg["model_key"]
    if model_key not in cfg["eligible_models"]:
        print(f"[e1v2] WARNING: {model_key} is not in config eligible_models {cfg['eligible_models']}")
    top_k = int((llm_cfg.get("logprobs") or {}).get("top_k", 5))
    seeds = common.parse_csv_list(args.seeds, int) or cfg["protocol"]["seeds"]
    run_cfg = dict(cfg["run"])
    if args.timeout:
        run_cfg["request_timeout_s"] = args.timeout
    concurrency = args.concurrency or run_cfg["concurrency"]

    use_asr = not args.no_asr
    noise = common.parse_csv_list(args.trace_noise) if args.trace_noise else cfg["traces"]["noise"]
    voices = args.trace_voices or cfg["traces"].get("voices", "all")
    voices = "all" if voices == "all" else common.parse_csv_list(voices)
    traces = common.load_traces(paths["traces_dir"], voices, noise) if use_asr else {}
    synthetic = sorted({t.get("voice_id") for ts in traces.values() for t in ts
                        if str(t.get("voice_id", "")).startswith(SYNTHETIC_VOICE_PREFIX)})
    if synthetic and backend_kind != "mock":
        print(f"REFUSED: synthetic plumbing traces {synthetic} may only be used with the mock "
              "backend; they are not speech data.", file=sys.stderr)
        return 1

    out_dir = args.out_dir or common.results_dir(dataset, model_key, backend_kind == "mock")
    signals_path = out_dir / f"signals_{args.split}.jsonl"
    retrieval_path = out_dir / f"retrieval_{args.split}.jsonl"
    errors_path = out_dir / f"errors_{args.split}.jsonl"

    units = []
    if not args.retrieval_only:
        units = build_units(items, traces, cfg, seeds, model,
                            use_text=not args.no_text_prefixes, use_asr=use_asr)
    r_units, retrieved = [], None
    if args.with_retrieval or args.retrieval_only:
        k = args.k or cfg["retrieval"]["k"]
        index_dir = args.index_dir or (common.REPO_ROOT / cfg["retrieval"]["index_dir_template"]
                                       .format(dataset=dataset))
        corpus = args.corpus or paths["base"] / "corpus.jsonl"
        if args.dry_run:
            retrieved = {i["id"]: {"docs": []} for i in items}
        else:
            retriever = make_retriever(args.retriever, index_dir, corpus)
            retrieved = retrieve_all(items, retriever, k)
        r_units = build_units(items, {}, cfg, seeds, model, condition="retrieval",
                              retrieved=retrieved)

    done_s, by_call_s = load_existing(signals_path)
    done_r, by_call_r = load_existing(retrieval_path)
    n_cond = sum(len(u.prefixes) for u in units)
    need_s = [u for u in units if any(k not in done_s for k in u.record_keys)
              and u.call_key not in by_call_s]
    need_r = [u for u in r_units if any(k not in done_r for k in u.record_keys)
              and u.call_key not in by_call_r]
    n_calls = len(need_s) + len(need_r)
    print(f"[e1v2] dataset={dataset} split={args.split} items={len(items)} seeds={seeds} "
          f"model={model} ({model_key}) backend={backend_kind}")
    print(f"[e1v2] traces: {sum(len(v) for v in traces.values())} for {len(traces)} items; "
          f"closed-book conditions={n_cond} distinct calls={len(units)}; "
          f"retrieval calls={len(r_units)}")
    print(f"[e1v2] to do: {n_calls} calls ({len(units) + len(r_units) - n_calls} already on disk)")
    wall = estimate(n_calls, concurrency, args.assume_call_s, args.assume_efficiency)
    print(f"[e1v2] estimate at concurrency {concurrency}: ~{wall / 60:.1f} min "
          f"(assumes {args.assume_call_s}s/call single-stream, {args.assume_efficiency:.0%} "
          "batching efficiency)")
    if args.dry_run:
        return 0

    backend = make_backend(backend_kind, llm_cfg, top_k, run_cfg)
    return asyncio.run(_run_async(
        args, cfg, backend, units, r_units, model, model_key, concurrency, seeds, top_k,
        signals_path, retrieval_path, errors_path, done_s, by_call_s, done_r, by_call_r,
        paths, dataset, items, traces, noise, voices,
    ))


async def _run_async(args, cfg, backend, units, r_units, model, model_key, concurrency, seeds,
                     top_k, signals_path, retrieval_path, errors_path, done_s, by_call_s, done_r,
                     by_call_r, paths, dataset, items, traces, noise, voices) -> int:
    try:
        if backend.name == "vllm" and not args.skip_health_check:
            print("[e1v2] health check (a cold endpoint can take minutes)...", flush=True)
            try:
                served = await backend.served_models()
            except GenerationError as error:
                print(f"[e1v2] {error}", file=sys.stderr)
                return 1
            if model not in served:
                print(f"[e1v2] server serves {served}, not {model}. Set --model / "
                      "VOICE_RAG_MODEL to the served model.", file=sys.stderr)
                return 1
        started = common.now_utc()
        t0 = time.perf_counter()
        stats = {}
        with common.JsonlAppender(errors_path) as errors:
            if units:
                with common.JsonlAppender(signals_path) as writer:
                    stats["closed_book"] = await execute(
                        units, backend, writer, cfg, model=model, model_key=model_key,
                        split=args.split, done_keys=done_s, existing_by_call=by_call_s,
                        concurrency=concurrency, error_writer=errors)
            if r_units:
                with common.JsonlAppender(retrieval_path) as writer:
                    stats["retrieval"] = await execute(
                        r_units, backend, writer, cfg, model=model, model_key=model_key,
                        split=args.split, done_keys=done_r, existing_by_call=by_call_r,
                        concurrency=concurrency, error_writer=errors)
        wall_s = time.perf_counter() - t0
    finally:
        await backend.close()
    if errors_path.exists() and errors_path.stat().st_size == 0:
        errors_path.unlink()

    manifest_path = signals_path.parent / f"run_manifest_{args.split}.json"
    manifest = common.load_json(manifest_path) if manifest_path.exists() else {"runs": []}
    manifest["runs"].append({
        "experiment_id": cfg["experiment_id"],
        "started_at_utc": started,
        "wall_s": wall_s,
        "git_commit": common.git_commit(),
        "backend": backend.name,
        "model": model,
        "model_key": model_key,
        "dataset": dataset,
        "split": args.split,
        "n_items": len(items),
        "seeds": seeds,
        "protocol": {**cfg["protocol"], "top_logprobs": top_k},
        "first_k": cfg["first_k"],
        "word_fracs": cfg["word_fracs"],
        "prompts": cfg["prompts"],
        "traces": {"n_traces": sum(len(v) for v in traces.values()), "noise": noise,
                   "voices": voices},
        "concurrency": concurrency,
        "inputs": {
            "items": common.rel(paths["items"]),
            "items_sha256": common.sha256_file(paths["items"]),
            "splits": common.rel(paths["splits"]),
            "splits_sha256": common.sha256_file(paths["splits"]),
            "config_sha256": common.sha256_file(args.config),
        },
        "stats": stats,
        "endpoint": "read from VOICE_RAG_LLM_BASE_URL; deliberately not recorded",
    })
    common.atomic_write_json(manifest_path, manifest)
    failed = sum(s.get("failed", 0) for s in stats.values())
    print(f"[e1v2] done in {wall_s:.1f}s: {json.dumps(stats)}")
    print(f"[e1v2] signals:   {common.rel(signals_path)}")
    if r_units:
        print(f"[e1v2] retrieval: {common.rel(retrieval_path)}")
    if failed:
        print(f"[e1v2] {failed} call(s) failed; see {common.rel(errors_path)}. Re-run the same "
              "command to retry them.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
