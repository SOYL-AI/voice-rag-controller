"""Self-assessed retrieval need: ask the model whether it can answer.

    python -m llm.experiment1_v2.self_assess --dataset d2_support --split dev

Entropy of the closed-book answer is near chance on d2_support dev (in the
support-agent persona Mistral fabricates org policies fluently, so it is
confidently wrong). An exploratory probe instead asks the same model a
yes/no question about the prefix heard so far:

    "Can this be answered correctly from general knowledge alone, without
     {org}'s own policies, prices, plans or records? Answer with one word:
     Yes or No."

and reads the answer distribution from one token (temperature 0,
max_tokens 1, top-20 logprobs):

    self_assessed_need = P(No) = mass("no"...) / (mass("yes"...) + mass("no"...))

The wording (config.json "self_assessment", prompt_version sa-v1-frozen) is
FROZEN as written in the probe. It is not tuned on dev: with 222 dev items,
any search over wordings would overfit.

One call per distinct prefix text per item. The call is deterministic, so
seeds do not matter. The prefix conditions are the runner's own:
item_prefixes() with config word_fracs, plus every ASR partial and final
transcript of the selected traces. Each call records its own TTFT and
generation time, plus the requests in flight when it was sent. Output goes
to self_assessment_<split>.jsonl next to signals_<split>.jsonl. analyze.py
joins it to every signal record by (item_id, prefix text) as signal
`self_assessed_need`. Resumable; sealed test split as everywhere else.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.config import load_config as load_llm_config  # noqa: E402
from llm.experiment1_v2 import common  # noqa: E402
from llm.experiment1_v2.backends import GenerationError, make_backend  # noqa: E402
from llm.experiment1_v2.prefixes import group_by_text, item_prefixes, normalize_text  # noqa: E402

SIGNAL = "self_assessed_need"


def template_for(item: dict, sa_cfg: dict) -> dict:
    return sa_cfg.get(item.get("dataset")) or sa_cfg["default"]


def build_messages(item: dict, prefix_text: str, sa_cfg: dict) -> list:
    """The probe's prompt, exactly: history as 'Role: content' lines (role
    capitalised), then the caller line and the yes/no question."""
    tpl = template_for(item, sa_cfg)
    history = "".join(
        sa_cfg["history_line"].format(Role=str(t.get("role", "")).capitalize(), content=t.get("content", ""))
        for t in (item.get("history") or []) if isinstance(t, dict)
    )
    fill = {"org": item.get("org") or "the company", "domain": item.get("domain") or "service",
            "prefix": prefix_text, "history": history}
    return [{"role": "system", "content": tpl["system"].format(**fill)},
            {"role": "user", "content": tpl["user"].format(**fill)}]


def prompt_sha256(sa_cfg: dict) -> str:
    import json  # noqa: PLC0415

    keep = {k: v for k, v in sa_cfg.items() if not k.startswith("_")}
    return common.sha256_text(json.dumps(keep, sort_keys=True))


def p_no_from_top(top_logprobs) -> dict:
    """Yes/No masses over the first token's top-k, as the probe computed
    them: a token counts when its stripped lower-case text starts with
    'yes' / 'no' (so 'Not' counts as No; frozen with the wording)."""
    yes = sum(math.exp(t.logprob) for t in top_logprobs if t.token.strip().lower().startswith("yes"))
    no = sum(math.exp(t.logprob) for t in top_logprobs if t.token.strip().lower().startswith("no"))
    return {"p_no": (no / (yes + no)) if (yes + no) > 0 else None, "yes_mass": yes, "no_mass": no}


def record_key(item_id: str, sha: str, text: str) -> str:
    return f"{item_id}|{sha[:12]}|{normalize_text(text)}"


def plan(items, traces_by_item, cfg, use_text=True, use_asr=True) -> list:
    """[(item, text, [prefix, ...])]: one entry per distinct prefix text."""
    out = []
    for item in items:
        prefixes = item_prefixes(item, traces_by_item.get(item["id"]), cfg["word_fracs"],
                                 text=use_text, asr=use_asr,
                                 include_asr_final=cfg["traces"].get("include_asr_final", True))
        for text, group in group_by_text(prefixes):
            out.append((item, text, group))
    return out


async def execute(units, backend, writer, sa_cfg, model, model_key, split, concurrency) -> dict:
    sha = prompt_sha256(sa_cfg)
    req = sa_cfg["request"]
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    in_flight = [0]
    stats = {"calls": 0, "failed": 0}

    async def one(item, text, group):
        async with semaphore:
            in_flight[0] += 1
            at_send = in_flight[0]
            try:
                # Mock only: answer "No"/"Yes" by the authored label, mostly.
                hint = {"item": dict(item, expected_answer=(
                    "No" if item.get("retrieval_needed_authored") else "Yes")), "grounded": True}
                result = await backend.generate(
                    build_messages(item, text, sa_cfg), seed=0, temperature=req["temperature"],
                    top_p=1.0, max_tokens=req["max_tokens"], mock_hint=hint)
                return item, text, group, result, at_send, None
            except GenerationError as error:
                return item, text, group, None, at_send, str(error)
            finally:
                in_flight[0] -= 1

    tasks = [asyncio.create_task(one(*u)) for u in units]
    for future in asyncio.as_completed(tasks):
        item, text, group, result, at_send, error = await future
        if error is not None:
            stats["failed"] += 1
            print(f"[self_assess] {item['id']}: {error}", file=sys.stderr)
            continue
        tokens = result.generation.tokens
        masses = p_no_from_top(tokens[0].top_logprobs if tokens else [])
        writer.write({
            "item_id": item["id"], "dataset": item["dataset"], "model": model, "model_key": model_key,
            "backend": backend.name, "split": split, "signal": SIGNAL,
            "prompt_version": sa_cfg["prompt_version"], "prompt_sha256": sha,
            "key": record_key(item["id"], sha, text), "prefix_text": text,
            "prefix_ids": [p["prefix_id"] for p in group],
            SIGNAL: masses["p_no"], "yes_mass": masses["yes_mass"], "no_mass": masses["no_mass"],
            "first_token": tokens[0].token if tokens else None,
            "top_logprobs": [{"token": t.token, "logprob": t.logprob}
                             for t in (tokens[0].top_logprobs if tokens else [])],
            "ttft_ms": result.ttft_ms, "gen_ms": result.gen_ms, "in_flight_at_send": at_send,
            "timing_synthetic": result.timing_synthetic, "served_model": result.served_model,
            "created_at_utc": common.now_utc(),
        })
        stats["calls"] += 1
    return stats


def load_lookup(path: Path) -> dict:
    """{(item_id, normalized text): record} for analyze.py's join."""
    return {(r["item_id"], normalize_text(r["prefix_text"])): r for r in common.read_jsonl(path)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--items", type=Path)
    p.add_argument("--splits", type=Path)
    p.add_argument("--traces-dir", type=Path)
    p.add_argument("--split", required=True, choices=common.VALID_SPLITS)
    p.add_argument("--model")
    p.add_argument("--backend", choices=("vllm", "mock"))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--latency-probe", action="store_true",
                   help="concurrency 1, written to self_assessment_latency_<split>.jsonl: the "
                   "unqueued latency of the self-assessment call (use with --limit)")
    args = p.parse_args(argv)
    try:
        common.check_split_allowed(args.split)
    except common.SealedSplitError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    cfg = common.load_config(args.config)
    sa_cfg = cfg["self_assessment"]
    paths = common.resolve_dataset_paths(args.dataset, args.items, args.splits, args.traces_dir)
    items = common.select_split(common.load_items(paths["items"]), common.load_splits(paths["splits"]),
                                args.split)[: args.limit or None]
    dataset = args.dataset or items[0]["dataset"]
    llm_cfg = load_llm_config(model_key=args.model)
    backend_kind = (args.backend or llm_cfg.get("backend") or "mock").lower()
    traces = {} if args.no_asr else common.load_traces(
        paths["traces_dir"], cfg["traces"].get("voices", "all"), cfg["traces"]["noise"])
    if backend_kind != "mock" and any(str(t.get("voice_id", "")).startswith("synthetic")
                                      for ts in traces.values() for t in ts):
        print("REFUSED: synthetic plumbing traces are for the mock backend only.", file=sys.stderr)
        return 1
    out_dir = args.out_dir or common.results_dir(dataset, llm_cfg["model_key"], backend_kind == "mock")
    out_path = out_dir / (f"self_assessment_latency_{args.split}.jsonl" if args.latency_probe
                          else f"self_assessment_{args.split}.jsonl")
    if args.latency_probe:
        args.concurrency = 1
    sha = prompt_sha256(sa_cfg)
    done = {r["key"] for r in common.read_jsonl(out_path)}
    units = [u for u in plan(items, traces, cfg, use_asr=not args.no_asr)
             if record_key(u[0]["id"], sha, u[1]) not in done]
    print(f"[self_assess] {dataset}/{args.split}: {len(items)} items, {len(units)} calls to make "
          f"({len(done)} on disk), prompt {sa_cfg['prompt_version']} sha {sha[:12]}")
    if args.dry_run or not units:
        return 0
    backend = make_backend(backend_kind, llm_cfg, sa_cfg["request"]["top_logprobs"], cfg["run"])

    async def go():
        try:
            if backend.name == "vllm":
                served = await backend.served_models()
                if llm_cfg["model"] not in served:
                    print(f"[self_assess] server serves {served}, not {llm_cfg['model']}", file=sys.stderr)
                    return None
            with common.JsonlAppender(out_path) as writer:
                return await execute(units, backend, writer, sa_cfg, llm_cfg["model"],
                                     llm_cfg["model_key"], args.split, args.concurrency)
        finally:
            await backend.close()

    t0 = time.perf_counter()
    stats = asyncio.run(go())
    if stats is None:
        return 1
    print(f"[self_assess] {stats} in {time.perf_counter() - t0:.1f}s -> {common.rel(out_path)}")
    return 2 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
