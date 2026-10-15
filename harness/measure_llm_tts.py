"""
measure_llm_tts.py — measure the LLM- and TTS-side latency components on the
real served model (research v2), within a hard request budget.

Part A (pipeline): every v1 scenario through pipeline/spine.py at
sentence_end, for path in {skip, retrieve} x tts_mode in {batch, streaming},
with real resident piper synthesis (no playback sleeps). Cell order is
shuffled per scenario so endpoint drift spreads evenly. Gives:
  - streaming vs batch first-audio latency,
  - time to first token / first complete clause / llm_final,
  - the paired retrieve-minus-skip TTFA increment per TTS mode, and its
    non-hideable part (everything except the exposed retrieval call).

Part B (prefill sweep): direct streaming chat requests with max_tokens=1 for
a subset of questions with k in {0, 5, 10} retrieved passages, recording
TTFT and the server-reported prompt token count (stream usage), to fit
prefill/first-token latency as a function of prompt length.

Credentials are read from an env file (KEY=VALUE lines) into the process
environment and never printed or written anywhere. Logs go to
logs/v2_llm_tts/ (endpoint-free: they record the model id only).

    python -m harness.measure_llm_tts --env-file <path/to/.env> \\
        --max-requests 300 --out analysis/llm_tts_v2/measure_llm_tts.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class Budget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def take(self, n: int = 1) -> None:
        if self.used + n > self.limit:
            raise RuntimeError(f"request budget exhausted ({self.used}/{self.limit})")
        self.used += n


def load_env_file(path: str) -> list:
    """Load KEY=VALUE lines into os.environ. Returns the key names only."""
    keys = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k.startswith("export "):
            k = k[len("export "):].strip()
        os.environ[k] = v
        keys.append(k)
    return keys


def _complete(log_path: Path) -> bool:
    """A cell log that reached first audio and the final answer."""
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8")
    return '"tts_first_audio"' in text and '"llm_final"' in text


def frozen_query_for(scenario_id: int) -> str:
    """The E3 sweep's precomputed rewrite for this scenario (no request),
    so retrieval runs on the same query E2/E3 used and no live rewrite
    lands inside the timed window."""
    cells = sorted((_REPO_ROOT / "simulator" / "e3_sweep_cache" / "cells").glob(f"s{scenario_id:03d}_seed*_retrieve_*.json"))
    for c in cells:
        fq = json.loads(c.read_text(encoding="utf-8")).get("frozen_query")
        if fq:
            return fq
    raise RuntimeError(f"no frozen query cached for scenario {scenario_id}")


def run_pipeline_cell(scenario, path, mode, seed, log_dir):
    from pipeline.spine import run_conversation

    cid = f"v2llm_{mode}_{path}_s{scenario['id']:03d}"
    log_path = Path(log_dir) / f"{cid}.jsonl"
    log_path.unlink(missing_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        written = asyncio.run(
            run_conversation(
                cid,
                question=scenario["question"],
                force_retrieve=(path == "retrieve"),
                frozen_query=frozen_query_for(scenario["id"]),
                seed=seed,
                tts_mode=mode,
                simulate_playback=False,
                log_dir=str(log_dir),
            )
        )
    return Path(written)


def prefill_request(client, messages, budget: Budget) -> dict:
    """One streaming request with max_tokens=1: TTFT + server prompt tokens."""
    budget.take()
    from eventlog.log_writer import now_ms

    request = {
        "model": client.model,
        "messages": messages,
        "stream": True,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    first = None
    usage = None
    stream = client._client.chat.completions.create(**request)
    for chunk in stream:
        if chunk.choices and first is None:
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                first = (time.perf_counter() - t0) * 1000.0
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    total = (time.perf_counter() - t0) * 1000.0
    return {
        "ttft_ms": first if first is not None else total,
        "total_ms": total,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "prompt_chars": sum(len(m["content"]) for m in messages),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--mock-dry-run", action="store_true")
    ap.add_argument("--max-requests", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scenarios", type=int, default=50)
    ap.add_argument("--prefill-questions", type=int, default=20)
    ap.add_argument("--prefill-k", type=int, nargs="+", default=[0, 5, 10])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--log-dir", default="logs/v2_llm_tts")
    ap.add_argument("--out", default="analysis/llm_tts_v2/measure_llm_tts.json")
    ap.add_argument("--resume", action="store_true",
                    help="reuse complete cell logs from an interrupted run (no new request for them)")
    ap.add_argument("--already-used", type=int, default=0,
                    help="requests already spent by interrupted runs; counted against --max-requests")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    args = ap.parse_args(argv)

    if args.mock_dry_run:
        # Exercise the whole harness against the mock LLM (no endpoint).
        os.environ["VOICE_RAG_LLM_BACKEND"] = "mock"
        args.skip_prefill = True
        args.warmup = 0
    else:
        keys = load_env_file(args.env_file)
        print(f"loaded {len(keys)} env keys (values not shown)")
        os.environ["VOICE_RAG_LLM_BACKEND"] = "vllm"
    os.environ.setdefault("VOICE_RAG_MODEL", "mistral7b_v03")
    os.environ["VOICE_RAG_TIMING_POLICY"] = "sentence_end"
    os.environ.setdefault("VOICE_RAG_TORCH_THREADS", "4")
    os.environ.setdefault("VOICE_RAG_PIPER_MODE", "resident")

    scenarios = json.loads((_REPO_ROOT / "simulator" / "scenarios.json").read_text(encoding="utf-8"))
    scenarios = scenarios[: args.scenarios]
    log_dir = Path(args.log_dir)
    n_pipeline = 0
    if not args.skip_pipeline:
        for sc in scenarios:
            for path in ("skip", "retrieve"):
                for mode in ("batch", "streaming"):
                    done = log_dir / f"v2llm_{mode}_{path}_s{sc['id']:03d}.jsonl"
                    if not (args.resume and _complete(done)):
                        n_pipeline += 1
    n_prefill = 0 if args.skip_prefill else len(args.prefill_k) * args.prefill_questions
    planned = args.already_used + args.warmup + n_pipeline + n_prefill
    if planned > args.max_requests:
        raise SystemExit(f"planned {planned} requests > budget {args.max_requests}")
    budget = Budget(args.max_requests)
    budget.used = args.already_used
    print(f"planned requests: {planned} (warm-up {args.warmup}, pipeline {n_pipeline}, prefill {n_prefill})")

    from llm.client import VLLMClient, build_messages, make_client
    from llm.config import load_config

    cfg = load_config()
    client = make_client(cfg) if args.mock_dry_run else VLLMClient(cfg)
    if not args.mock_dry_run and client.model != "mistralai/Mistral-7B-Instruct-v0.3":
        raise SystemExit(f"refusing to run against {client.model}: Mistral-7B-Instruct-v0.3 is the only model")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up (cold start can take minutes); discarded.
    for i in range(args.warmup):
        budget.take()
        t0 = time.perf_counter()
        client.generate(build_messages("Is breakfast included?", []), max_tokens=8)
        print(f"warm-up {i + 1}: {(time.perf_counter() - t0):.1f} s")

    # Local warm-up (no requests): first embedding call and first piper
    # synthesis in a process pay one-off initialisation costs.
    from retrieval.search import search as _warm_search
    from audio.config import load_config as _audio_cfg
    from audio.tts import StreamingTTS

    _warm_search("warm up the embedding model", k=3)
    StreamingTTS(_audio_cfg()).synthesize_clause_to_raw_pcm("Warming up the voice.")

    record = {
        "model": client.model,
        "seed": args.seed,
        "generation": cfg.get("generation"),
        "tts": "piper resident en_US-lessac-medium (CPU), no playback sleeps",
        "torch_threads": os.environ.get("VOICE_RAG_TORCH_THREADS"),
        "pipeline_logs": [],
        "prefill": [],
    }

    if not args.skip_pipeline:
        rng = random.Random(args.seed)
        for sc in scenarios:
            cells = [(p, m) for p in ("skip", "retrieve") for m in ("batch", "streaming")]
            rng.shuffle(cells)
            for path, mode in cells:
                done_log = log_dir / f"v2llm_{mode}_{path}_s{sc['id']:03d}.jsonl"
                if args.resume and _complete(done_log):
                    lp = done_log  # measured in an earlier (interrupted) run
                else:
                    lp = None
                    for attempt in range(2):
                        budget.take()
                        try:
                            lp = run_pipeline_cell(sc, path, mode, args.seed, log_dir)
                            break
                        except Exception as exc:  # e.g. TTS out of memory on a loaded host
                            import gc

                            gc.collect()
                            print(f"cell s{sc['id']:03d} {path}/{mode} failed ({type(exc).__name__}: {exc}); "
                                  f"{'retrying' if attempt == 0 else 'giving up on this cell'}")
                    if lp is None:
                        continue
                record["pipeline_logs"].append(
                    {"scenario_id": sc["id"], "path": path, "tts_mode": mode, "log": str(lp)}
                )
            print(f"scenario {sc['id']:3d} done (requests used {budget.used})")

    if not args.skip_prefill:
        from retrieval.search import search

        rng = random.Random(args.seed + 1)
        qs = rng.sample(scenarios, min(args.prefill_questions, len(scenarios)))
        cells = [(sc, k) for sc in qs for k in args.prefill_k]
        rng.shuffle(cells)
        for sc, k in cells:
            passages = [r["text"] for r in search(sc["question"], k=k)] if k > 0 else []
            r = prefill_request(client, build_messages(sc["question"], passages), budget)
            r.update({"scenario_id": sc["id"], "k": k})
            record["prefill"].append(r)
        print(f"prefill sweep done (requests used {budget.used})")

    record["requests_used"] = budget.used
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {out}; requests used {budget.used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
