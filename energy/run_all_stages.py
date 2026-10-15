"""Run every stage of energy/bench_stages.py in its own fresh process.

One process per stage keeps peak RAM low (the host was under heavy memory
pressure from other workloads; one combined run died in mkl_malloc) and
isolates allocator/thread state between stages. Results merge into
energy/results/stages.json; stages already present are skipped unless
--redo. A failed stage is retried once and otherwise recorded as skipped.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.run_all_stages
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "stages.json"
STAGES = [
    "vad_utterance_cpu", "asr_final_cpu_int8", "asr_stream_cpu_int8",
    "asr_stream_cuda_fp16", "embed_query_cpu", "embed_query_cuda", "gpu_wake_tail",
    "faiss_search_60", "faiss_search_10k", "faiss_search_100k",
    "rerank50_minilm_cuda", "rerank50_minilm_cpu", "rerank50_bgebase_cuda",
    "tts_piper_inproc_short", "tts_piper_inproc_typical", "tts_piper_inproc_long",
    "tts_piper_subprocess_typical",
]


def done() -> set:
    if not OUT.exists():
        return set()
    return set(json.loads(OUT.read_text()).get("stages", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--target-s", type=float, default=4.0)
    ap.add_argument("--idle-s", type=float, default=3.0)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    for st in STAGES:
        if st in done() and not args.redo:
            print("have", st, flush=True)
            continue
        for attempt in (1, 2):
            cmd = [sys.executable, "-u", "-m", "energy.bench_stages", "--trials", str(args.trials),
                   "--target-s", str(args.target_s), "--idle-s", str(args.idle_s), "--only", st]
            print(">>", st, "attempt", attempt, flush=True)
            rc = subprocess.call(cmd)
            if rc == 0 and st in done():
                break
        else:
            print("FAILED", st, flush=True)


if __name__ == "__main__":
    main()
