"""Measure retriever latency profiles and write retrieval/latency_profiles_v2.json.

Query set: the 50 v1 questions in simulator/scenarios.json plus every word
prefix of each question with at least 3 words (partial transcripts are what
an early-dispatch policy actually searches with), de-duplicated in first-seen
order. The sha256 of the newline-joined query list is recorded.

Each variant runs in its own fresh Python process so torch's process-global
thread setting, CUDA context and allocator state cannot leak between
variants. Per variant: one full warm-up pass (discarded), then ``--passes``
passes over the query set, each in a different seeded shuffle, timed with
``time.perf_counter`` around the call as the caller would see it.

Variants (``name: what it isolates``):

* ``faiss`` - canonical in-process profile: bge-small on CPU, torch 4 threads.
* ``faiss_to_thread`` - the same, called through ``asyncio.to_thread`` from a
  running event loop, as pipeline/spine.py calls retrieval.
* ``faiss_v1_default_threads`` / ``faiss_v1_default_threads_to_thread`` -
  torch's default intra-op thread count (one per logical core), i.e. the
  condition the v1 pipeline ran in. Explains the E2 in-pipeline number.
* ``faiss_cuda`` - query embedding on the GPU.
* ``faiss_rerank_minilm_l6`` / ``faiss_rerank_bge_base`` (+ ``_fp16``) - FAISS
  top-50 then a cross-encoder (ms-marco-MiniLM-L6-v2 or bge-reranker-base)
  on CUDA in float32 or float16. The canonical ``faiss_rerank`` is chosen by
  the rule in :func:`choose_canonical_rerank` (quality first, then p50) and
  the reason is recorded in the artifact. Quality = hit@1/hit@3 of the
  scenario ``document`` over the retrieval_needed questions.
* ``remote_api_faiss`` / ``remote_api_faiss_rerank`` - HTTP client against
  retrieval/service.py started as a local uvicorn subprocess on 127.0.0.1.
  Transport overhead = client round trip - server handler time. Network is
  loopback only; WAN RTT must be added after a deployment is approved.
* ``remote_api_wan`` - only with ``--remote-url``: the same client against an
  already-running external endpoint.

Also an index-size sweep of the FAISS search component alone (random 384-d
vectors, IndexFlatL2, single query), sized to the memory available.

Usage::

    python -m retrieval.measure_latency                    # everything, ~10 min
    python -m retrieval.measure_latency --only faiss,faiss_cuda --passes 3
    python -m retrieval.measure_latency --skip-sweep
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = ROOT / "simulator" / "scenarios.json"
DEFAULT_OUT = ROOT / "retrieval" / "latency_profiles_v2.json"
QUERY_SOURCE = "simulator/scenarios.json questions + prefixes>=3 words"
MIN_PREFIX_WORDS = 3
MAX_SAMPLES = 2000
QUANT_PCTS = {"p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90, "p95": 95, "p99": 99}
K = 3
CANDIDATES = 50
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MINILM = "cross-encoder/ms-marco-MiniLM-L6-v2"
BGE_RERANK = "BAAI/bge-reranker-base"
# rerank variant -> (cross-encoder, dtype on CUDA)
RERANKERS = {
    "faiss_rerank_minilm_l6": (MINILM, "float32"),
    "faiss_rerank_bge_base": (BGE_RERANK, "float32"),
    "faiss_rerank_minilm_l6_fp16": (MINILM, "float16"),
    "faiss_rerank_bge_base_fp16": (BGE_RERANK, "float16"),
}
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _rerank_variant(name: str) -> dict:
    model, dtype = RERANKERS[name]
    return {
        "kind": "local", "profile": "faiss_rerank", "device": "cpu", "torch_threads": 4,
        "call": "sync", "rerank_model": model, "rerank_device": "cuda", "rerank_dtype": dtype,
        "notes": f"FAISS top-{CANDIDATES} (CPU embed, 4 threads) then {model} rescoring all "
                 f"{CANDIDATES} candidates in one batch on CUDA in {dtype}."}


VARIANTS: dict[str, dict] = {
    "faiss": {
        "kind": "local", "profile": "faiss", "device": "cpu", "torch_threads": 4, "call": "sync",
        "notes": "Canonical in-process profile: bge-small query embedding on CPU with torch "
                 "intra-op threads = 4, exact IndexFlatL2 search over the v1 hotel index."},
    "faiss_to_thread": {
        "kind": "local", "profile": "faiss", "device": "cpu", "torch_threads": 4, "call": "to_thread",
        "notes": "As faiss, but each call goes through asyncio.to_thread from a running event "
                 "loop (how pipeline/spine.py runs retrieval). Measures the thread-hop cost."},
    "faiss_v1_default_threads": {
        "kind": "local", "profile": "faiss", "device": "cpu", "torch_threads": None, "call": "sync",
        "notes": "torch default intra-op threads (one per logical core), no OMP/MKL env "
                 "overrides: the condition the v1 pipeline ran in. Isolated process with no "
                 "competing STT/LLM/TTS load, so it is a lower bound on the in-pipeline E2 cost."},
    "faiss_v1_default_threads_to_thread": {
        "kind": "local", "profile": "faiss", "device": "cpu", "torch_threads": None, "call": "to_thread",
        "notes": "v1 default threads + asyncio.to_thread: closest isolated reproduction of the "
                 "E2 in-pipeline retrieval call."},
    "faiss_cuda": {
        "kind": "local", "profile": "faiss", "device": "cuda", "torch_threads": 4, "call": "sync",
        "notes": "Query embedding on the GPU; FAISS search stays on CPU."},
    **{name: _rerank_variant(name) for name in RERANKERS},
    "remote_api_faiss": {
        "kind": "remote", "profile": "remote_api", "backend": "faiss", "device": "cpu",
        "torch_threads": 4, "call": "sync",
        "notes": "HTTP client -> retrieval/service.py (uvicorn subprocess, 127.0.0.1) running the "
                 "faiss backend. transport = client round trip - server handler time."},
    "remote_api_faiss_rerank": {
        "kind": "remote", "profile": "remote_api", "backend": "faiss_rerank", "device": "cpu",
        "torch_threads": 4, "call": "sync", "rerank_device": "cuda",
        "notes": "HTTP client -> retrieval/service.py (uvicorn subprocess, 127.0.0.1) running the "
                 "canonical faiss_rerank backend."},
    "remote_api_wan": {
        "kind": "remote_external", "profile": "remote_api", "call": "sync",
        "notes": "HTTP client against an externally deployed service (--remote-url)."},
}
DEFAULT_ORDER = ["faiss", "faiss_to_thread", "faiss_v1_default_threads",
                 "faiss_v1_default_threads_to_thread", "faiss_cuda", *RERANKERS,
                 "remote_api_faiss", "remote_api_faiss_rerank"]


# --------------------------------------------------------------------------
# Query set
# --------------------------------------------------------------------------

def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gold(sc: dict) -> Optional[str]:
    """Source document of a scenario that needs retrieval, else None."""
    if sc.get("retrieval_needed") and sc.get("document"):
        return sc["document"]
    return None


def build_query_set(scenarios: list[dict], min_words: int = MIN_PREFIX_WORDS) -> list[dict]:
    """Questions plus their word prefixes (>= min_words), de-duplicated in
    first-seen order.

    Each entry: {"query", "full": bool, "gold": str | None}. ``gold`` is the
    scenario's ``document`` when every scenario producing that string needs
    retrieval and they all share one document; otherwise None (closed-book
    questions, and generic prefixes such as "Tell me about" shared by
    questions about different documents). Only entries with a gold document
    count towards hit@k; every entry is timed.
    """
    out: list[dict] = []
    by_query: dict[str, dict] = {}
    golds: dict[str, set] = {}
    for sc in scenarios:
        words = sc["question"].split()
        for n in range(min(min_words, len(words)), len(words) + 1):
            q = " ".join(words[:n])
            golds.setdefault(q, set()).add(_gold(sc))
            if q in by_query:
                if n == len(words):  # a full question that is also another's prefix
                    by_query[q]["full"] = True
                continue
            entry = {"query": q, "full": n == len(words), "gold": None}
            by_query[q] = entry
            out.append(entry)
    for e in out:
        g = golds[e["query"]]
        e["gold"] = next(iter(g)) if len(g) == 1 else None
    return out


def query_set_sha256(queries: list[str]) -> str:
    return hashlib.sha256("\n".join(queries).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def quantiles(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {}
    return {k: round(float(np.percentile(a, p)), 3) for k, p in QUANT_PCTS.items()}


def summarize(samples: list[float], components: dict[str, list[float]],
              max_samples: int = MAX_SAMPLES, seed: int = 0) -> dict:
    a = np.asarray(samples, dtype=float)
    kept = list(a)
    if len(kept) > max_samples:
        rng = random.Random(seed)
        kept = sorted(rng.sample(range(len(a)), max_samples))
        kept = [a[i] for i in kept]
    return {
        "n": int(a.size),
        "n_samples_kept": len(kept),
        "quantiles_ms": quantiles(a),
        "mean_ms": round(float(a.mean()), 3),
        "std_ms": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 3),
        "min_ms": round(float(a.min()), 3),
        "max_ms": round(float(a.max()), 3),
        "samples_ms": [round(float(x), 2) for x in kept],
        "components_ms": {name: {**quantiles(v), "mean": round(float(np.mean(v)), 3)}
                          for name, v in components.items() if v},
    }


# --------------------------------------------------------------------------
# Host info
# --------------------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _nvidia_smi(fields: list[str]) -> dict:
    """Query nvidia-smi for GPU 0; {} when unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0 or not out.stdout.strip():
        return {}
    vals = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    res: dict = {}
    for f, v in zip(fields, vals):
        try:
            res[f] = float(v)
        except ValueError:
            res[f] = v
    return res


def host_info() -> dict:
    info = {"cpu": _cpu_name(), "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version()}
    smi = _nvidia_smi(["driver_version", "enforced.power.limit", "power.default_limit",
                       "power.max_limit", "clocks.max.sm"])
    if smi:
        info["gpu_driver"] = smi.get("driver_version")
        info["gpu_power_limit_w"] = smi.get("enforced.power.limit")
        info["gpu_power_default_limit_w"] = smi.get("power.default_limit")
        info["gpu_power_max_limit_w"] = smi.get("power.max_limit")
        info["gpu_max_sm_clock_mhz"] = smi.get("clocks.max.sm")
    try:
        import psutil

        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["torch_default_threads"] = torch.get_num_threads()
    except Exception:
        info["torch"] = None
    for mod in ("faiss", "sentence_transformers", "fastapi", "uvicorn", "httpx"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def load_snapshot() -> dict:
    """CPU and GPU load just before/after measuring, to flag contention."""
    snap: dict = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        import psutil

        snap["cpu_percent_1s"] = psutil.cpu_percent(interval=1.0)
        snap["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 2)
    except ImportError:
        pass
    smi = _nvidia_smi(["utilization.gpu", "memory.used", "memory.total", "pstate",
                       "power.draw", "enforced.power.limit", "temperature.gpu"])
    names = {"utilization.gpu": "gpu_util_percent", "memory.used": "gpu_mem_used_mb",
             "memory.total": "gpu_mem_total_mb", "pstate": "gpu_pstate",
             "power.draw": "gpu_power_draw_w", "enforced.power.limit": "gpu_power_limit_w",
             "temperature.gpu": "gpu_temp_c"}
    for k, v in smi.items():
        snap[names[k]] = v
    return snap


# --------------------------------------------------------------------------
# Worker (runs inside a fresh process per variant)
# --------------------------------------------------------------------------

def _make_worker_retriever(spec: dict, url: Optional[str]):
    from retrieval.profiles import FaissRerankRetriever, FaissRetriever, RemoteRetriever

    if spec["kind"] in ("remote", "remote_external"):
        return RemoteRetriever(url)
    if spec["profile"] == "faiss":
        return FaissRetriever(None, device=spec["device"], torch_threads=spec["torch_threads"])
    return FaissRerankRetriever(None, device=spec["device"], torch_threads=spec["torch_threads"],
                                rerank_model=spec["rerank_model"],
                                rerank_device=spec.get("rerank_device", "cuda"),
                                rerank_dtype=spec.get("rerank_dtype", "float32"),
                                candidates=CANDIDATES)


def run_worker(spec: dict, qset: list[dict], passes: int, seed: int,
               url: Optional[str]) -> dict:
    import torch

    retriever = _make_worker_retriever(spec, url)
    queries = [e["query"] for e in qset]
    to_thread = spec.get("call") == "to_thread"
    loop = asyncio.new_event_loop() if to_thread else None
    inner_threads: list[int] = []

    def call(q: str) -> list[dict]:
        if not inner_threads:
            inner_threads.append(torch.get_num_threads())
        return retriever.search(q, k=K)

    def timed(q: str) -> tuple[float, list[dict]]:
        if to_thread:
            async def go():
                t0 = time.perf_counter()
                hits = await asyncio.to_thread(call, q)
                return (time.perf_counter() - t0) * 1000.0, hits
            return loop.run_until_complete(go())
        t0 = time.perf_counter()
        hits = call(q)
        return (time.perf_counter() - t0) * 1000.0, hits

    t_load0 = time.perf_counter()
    retriever.warmup()
    load_s = time.perf_counter() - t_load0

    # Warm-up pass (discarded); also used for the quality check since the
    # ranking is deterministic. Only entries with a gold document count.
    hit1 = hit3 = n_full = 0
    phit1 = phit3 = n_prefix = 0
    for e in qset:
        _, hits = timed(e["query"])
        if not e.get("gold"):
            continue
        names = [h.get("parent_doc_id") or h.get("filename") for h in hits]
        ok1 = bool(names) and names[0] == e["gold"]
        ok3 = e["gold"] in names[:3]
        if e["full"]:
            n_full += 1
            hit1 += ok1
            hit3 += ok3
        else:
            n_prefix += 1
            phit1 += ok1
            phit3 += ok3

    totals: list[float] = []
    comps: dict[str, list[float]] = {}
    for p in range(passes):
        order = list(range(len(queries)))
        random.Random(seed * 1000 + p).shuffle(order)
        for i in order:
            ms, _ = timed(queries[i])
            totals.append(ms)
            timing = retriever.last_timing or {}
            for key, val in timing.items():
                name = key[:-3] if key.endswith("_ms") else key
                comps.setdefault(name, []).append(float(val))
            if to_thread and "total_ms" in timing:
                # event loop + to_thread hop beyond the retriever's own work
                comps.setdefault("dispatch", []).append(ms - float(timing["total_ms"]))
    comps.pop("total", None)
    if loop is not None:
        loop.close()

    desc = retriever.describe()
    if hasattr(retriever, "close"):
        retriever.close()
    return {
        "totals": totals,
        "components": comps,
        "describe": desc,
        "torch_threads_main": torch.get_num_threads(),
        "torch_threads_in_call_thread": inner_threads[0] if inner_threads else None,
        "load_and_first_query_s": round(load_s, 3),
        "quality": {"n_full_questions": n_full,
                    "hit1_full": round(hit1 / n_full, 4) if n_full else None,
                    "hit3_full": round(hit3 / n_full, 4) if n_full else None,
                    "n_prefixes": n_prefix,
                    "hit1_prefixes": round(phit1 / n_prefix, 4) if n_prefix else None,
                    "hit3_prefixes": round(phit3 / n_prefix, 4) if n_prefix else None,
                    "note": "hit@k of simulator/scenarios.json `document` over retrieval_needed "
                            "questions (full) and their unambiguous partial prefixes; ranking "
                            "from the warm-up pass"},
    }


def run_sweep_worker(sizes: list[int], dim: int, seed: int, reps: dict) -> dict:
    import faiss

    try:
        import psutil
        avail = psutil.virtual_memory().available
    except ImportError:
        avail = None
    rng = np.random.default_rng(seed)
    out = {"dim": dim, "index": "IndexFlatL2 (exact)", "nq_per_search": 1,
           "faiss_omp_threads": faiss.omp_get_max_threads(), "results": {}, "skipped": {}}
    for n in sizes:
        need = n * dim * 4
        if avail is not None and need * 1.3 + 0.5 * 2**30 > avail:
            out["skipped"][str(n)] = (f"needs ~{need / 2**30:.2f} GiB, only "
                                      f"{avail / 2**30:.2f} GiB available")
            continue
        index = faiss.IndexFlatL2(dim)
        step = 100_000
        for s in range(0, n, step):
            index.add(rng.standard_normal((min(step, n - s), dim), dtype=np.float32))
        nrep = reps.get(n, 200)
        queries = rng.standard_normal((nrep + 10, dim), dtype=np.float32)
        res = {"n_vectors": n, "index_mb": round(need / 2**20, 1)}
        for k in (3, 50):
            for q in queries[:10]:
                index.search(q.reshape(1, -1), min(k, n))
            ts = []
            for q in queries[10:]:
                t0 = time.perf_counter()
                index.search(q.reshape(1, -1), min(k, n))
                ts.append((time.perf_counter() - t0) * 1000.0)
            res[f"k{k}"] = {"n": len(ts), **quantiles(ts), "mean": round(float(np.mean(ts)), 4)}
        out["results"][str(n)] = res
        del index
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _clean_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    for v in THREAD_ENV_VARS:  # reproduce the default-thread condition faithfully
        env.pop(v, None)
    return env


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_subprocess_worker(args: list[str], timeout_s: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.json"
        cmd = [sys.executable, "-m", "retrieval.measure_latency", *args, "--worker-out", str(out_path)]
        proc = subprocess.run(cmd, cwd=ROOT, env=_clean_env(), timeout=timeout_s,
                              capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"worker failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)


class LocalServer:
    """retrieval/service.py in a uvicorn subprocess on 127.0.0.1."""

    def __init__(self, server_args: list[str]):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.args = server_args
        self.proc: Optional[subprocess.Popen] = None
        self.log = tempfile.TemporaryFile(mode="w+")

    def __enter__(self) -> "LocalServer":
        import httpx

        cmd = [sys.executable, "-X", "faulthandler", "-m", "retrieval.service", "--port", str(self.port), *self.args]
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=_clean_env(), stdout=self.log,
                                     stderr=subprocess.STDOUT)
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.log.seek(0)
                raise RuntimeError(f"server exited early (code {self.proc.returncode}):\n{self.log.read()[-4000:]}")
            try:
                r = httpx.get(f"{self.url}/healthz", timeout=2.0)
                if r.status_code == 200 and r.json().get("ready"):
                    self.health = r.json()
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError("server did not become ready in 300 s")

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


QUALITY_TOL = 0.02


def choose_canonical_rerank(profiles: dict) -> tuple[Optional[str], str]:
    """Pick the canonical faiss_rerank variant among the measured rerank
    variants, lexicographically:

    1. hit@3 on the full retrieval_needed questions: keep every variant
       within ``QUALITY_TOL`` of the best;
    2. hit@1 on the same questions: likewise;
    3. lowest p50 latency.

    A reranker that is no better but slower should not set the latency
    profile the controller plans against.
    """
    cands = [n for n in RERANKERS if n in profiles]
    if not cands:
        return None, "no rerank variant measured"

    def qual(n: str, key: str) -> float:
        return float((profiles[n].get("quality") or {}).get(key) or 0.0)

    def p50(n: str) -> float:
        return float(profiles[n]["quantiles_ms"]["p50"])

    pool = list(cands)
    for key in ("hit3_full", "hit1_full"):
        best = max(qual(n, key) for n in pool)
        pool = [n for n in pool if qual(n, key) >= best - QUALITY_TOL]
    choice = min(pool, key=p50)
    summary = "; ".join(f"{n}: hit@1={qual(n, 'hit1_full'):.2f} hit@3={qual(n, 'hit3_full'):.2f} "
                        f"p50={p50(n):.1f} ms" for n in cands)
    return choice, (f"rule: max hit@3, then max hit@1 (each within {QUALITY_TOL}), then min p50. "
                    f"Tied on quality: {', '.join(pool)}. Measured: {summary}")


def measure_variant(name: str, qset_path: Path, passes: int, seed: int,
                    canonical_rerank: Optional[str], remote_url: Optional[str]) -> dict:
    """Measure one variant in fresh processes. ``canonical_rerank`` names the
    rerank variant whose model and dtype the remote rerank server uses."""
    spec = dict(VARIANTS[name])
    wargs = ["--worker", name, "--queries-file", str(qset_path), "--passes", str(passes),
             "--seed", str(seed)]
    if spec["kind"] == "remote":
        sargs = ["--backend", spec["backend"], "--device", spec["device"],
                 "--torch-threads", str(spec["torch_threads"])]
        if spec["backend"] == "faiss_rerank":
            spec["rerank_model"], spec["rerank_dtype"] = RERANKERS[
                canonical_rerank or "faiss_rerank_minilm_l6"]
            spec["notes"] += f" Reranker: {spec['rerank_model']} ({spec['rerank_dtype']})."
            sargs += ["--rerank-model", spec["rerank_model"],
                      "--rerank-device", spec.get("rerank_device", "cuda"),
                      "--rerank-dtype", spec["rerank_dtype"],
                      "--candidates", str(CANDIDATES)]
        with LocalServer(sargs) as srv:
            raw = _run_subprocess_worker(wargs + ["--url", srv.url], timeout_s=3600)
            raw["server_describe"] = (srv.health or {}).get("backend")
    elif spec["kind"] == "remote_external":
        if not remote_url:
            raise ValueError("remote_api_wan needs --remote-url")
        raw = _run_subprocess_worker(wargs + ["--url", remote_url], timeout_s=3600)
    else:
        raw = _run_subprocess_worker(wargs, timeout_s=3600)

    entry = summarize(raw["totals"], raw["components"], seed=seed)
    desc = raw.get("server_describe") or raw.get("describe") or {}
    is_remote = spec["kind"] != "local"
    entry.update({
        "profile": spec["profile"],
        "device": desc.get("device", spec.get("device")),
        "torch_threads": desc.get("torch_threads") if is_remote else raw["torch_threads_main"],
        "torch_threads_requested": spec.get("torch_threads", "default"),
        "torch_threads_in_call_thread": None if is_remote else raw["torch_threads_in_call_thread"],
        "call": spec.get("call", "sync"),
        "k": K,
        "candidates": CANDIDATES if (spec["profile"] == "faiss_rerank"
                                     or spec.get("backend") == "faiss_rerank") else None,
        "model": EMBED_MODEL,
        "rerank_model": spec.get("rerank_model"),
        "rerank_device": desc.get("rerank_device") if spec.get("rerank_model") else None,
        "rerank_dtype": desc.get("rerank_dtype") if spec.get("rerank_model") else None,
        "score_kind": desc.get("score_kind"),
        "load_and_first_query_s": raw["load_and_first_query_s"],
        "quality": raw["quality"],
        "notes": spec["notes"],
    })
    if spec["kind"] == "remote":
        entry.update({"network": "localhost", "wan_rtt_ms": None, "server_backend": desc,
                      "wan_note": "Loopback only. Real WAN RTT must be added after a deployment "
                                  "is approved (see retrieval/modal_app.py)."})
    elif spec["kind"] == "remote_external":
        entry.update({"network": "wan", "wan_rtt_ms": None,
                      "wan_note": "transport component includes the WAN round trip."})
    if spec.get("call") == "to_thread":
        entry["notes"] += (" The 'dispatch' component = caller-observed time - retriever total "
                           "(event loop + to_thread hop).")
    return entry


def print_table(art: dict) -> None:
    rows = []
    for name, e in art["profiles"].items():
        q = e["quantiles_ms"]
        comp = e.get("components_ms", {})
        parts = []
        for c in ("embed", "search", "rerank", "server", "transport", "dispatch"):
            if c in comp:
                parts.append(f"{c}={comp[c]['p50']:.2f}")
        qual = e.get("quality") or {}
        rows.append((name, str(e.get("device")), str(e.get("torch_threads")), e["n"],
                     q["p50"], q["p90"], q["p95"], q["p99"], e["mean_ms"],
                     qual.get("hit1_full"), qual.get("hit3_full"), " ".join(parts)))
    hdr = ("variant", "device", "thr", "n", "p50", "p90", "p95", "p99", "mean", "hit@1", "hit@3",
           "components p50 (ms)")
    w = [max(len(str(r[i])) if not isinstance(r[i], float) else 8 for r in rows + [hdr]) for i in range(len(hdr))]
    w[0] = max(len(r[0]) for r in rows + [hdr])
    line = "  ".join(f"{h:<{w[i]}}" for i, h in enumerate(hdr))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            s = f"{v:.2f}" if isinstance(v, float) else str(v)
            cells.append(f"{s:<{w[i]}}")
        print("  ".join(cells))
    canon = art.get("canonical", {})
    print("\ncanonical:", ", ".join(f"{k} -> {v}" for k, v in canon.items()))
    sw = art.get("faiss_index_size_sweep")
    if sw and sw.get("results"):
        print("\nFAISS IndexFlatL2 search only, 384-d, 1 query (ms):")
        for n, r in sw["results"].items():
            print(f"  n={int(n):>9,}  k=3 p50={r['k3']['p50']:.3f} p99={r['k3']['p99']:.3f}   "
                  f"k=50 p50={r['k50']['p50']:.3f} p99={r['k50']['p99']:.3f}")
        for n, why in sw.get("skipped", {}).items():
            print(f"  n={int(n):>9,}  skipped: {why}")


def orchestrate(args) -> int:
    scenarios = load_scenarios()
    qset = build_query_set(scenarios)
    queries = [e["query"] for e in qset]
    out_path = Path(args.out)

    names = args.only.split(",") if args.only else list(DEFAULT_ORDER)
    if args.remote_url and "remote_api_wan" not in names:
        names.append("remote_api_wan")
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant {n!r}; known: {', '.join(VARIANTS)}")

    art: dict
    if args.only and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            art = json.load(f)
    else:
        art = {"schema_version": 1, "profiles": {}, "canonical": {}}
    art["schema_version"] = 1
    art["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    art["host"] = host_info()
    art["query_set"] = {"source": QUERY_SOURCE, "n_queries": len(queries),
                        "n_full_questions": sum(e["full"] for e in qset),
                        "min_prefix_words": MIN_PREFIX_WORDS, "dedup": True,
                        "sha256": query_set_sha256(queries)}
    art["corpus"] = {"index_dir": None, "n_docs": None,
                     "note": "index_dir null = the v1 hotel index (retrieval/index.faiss + "
                             "chunks.json), as make_retriever(index_dir=None) loads it"}
    host = art["host"]
    lim, dflt = host.get("gpu_power_limit_w"), host.get("gpu_power_default_limit_w")
    if isinstance(lim, float) and isinstance(dflt, float) and lim < dflt:
        host["gpu_note"] = (
            f"Enforced GPU power limit {lim:.0f} W is below the board default {dflt:.0f} W "
            "(laptop power mode; it varies at run time, see load_before/load_after). CUDA "
            "timings, especially the cross-encoders, are slower than the same GPU at its "
            "default limit.")
    art["protocol"] = {
        "warmup": "1 full pass over the query set, discarded",
        "passes": args.passes, "order": "seeded shuffle per pass", "seed": args.seed,
        "timer": "time.perf_counter around the call as the caller sees it",
        "isolation": "fresh Python process per variant; OMP/MKL/OPENBLAS thread env vars removed",
        "max_samples_kept": MAX_SAMPLES,
        "quantiles": "numpy percentile, linear interpolation, over all samples",
    }
    art["load_before"] = load_snapshot()

    with tempfile.TemporaryDirectory() as td:
        qpath = Path(td) / "queries.json"
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump(qset, f)
        for name in names:
            if name == "remote_api_faiss_rerank":
                canon, why = choose_canonical_rerank(art["profiles"])
                art["canonical"]["faiss_rerank"] = canon
                art["canonical_rerank_reason"] = why
            print(f"[measure] {name} ...", flush=True)
            t0 = time.time()
            art["profiles"][name] = measure_variant(name, qpath, args.passes, args.seed,
                                                    art["canonical"].get("faiss_rerank"),
                                                    args.remote_url)
            e = art["profiles"][name]
            print(f"[measure] {name}: n={e['n']} p50={e['quantiles_ms']['p50']:.2f} "
                  f"p90={e['quantiles_ms']['p90']:.2f} p99={e['quantiles_ms']['p99']:.2f} ms "
                  f"({time.time() - t0:.0f} s)", flush=True)

    canon, why = choose_canonical_rerank(art["profiles"])
    if canon:
        art["canonical"]["faiss_rerank"] = canon
        art["canonical_rerank_reason"] = why
    if "faiss" in art["profiles"]:
        art["canonical"]["faiss"] = "faiss"
    if "remote_api_faiss" in art["profiles"]:
        art["canonical"]["remote_api"] = "remote_api_faiss"
    if art["profiles"].get("remote_api_wan"):
        art["canonical"]["remote_api"] = "remote_api_wan"
    art["canonical_note"] = (
        "remote_api is canonically the faiss backend behind HTTP on loopback; its compute matches "
        "the faiss profile, so its extra cost is pure transport. Replace with a WAN measurement "
        "once a deployment is approved.")
    art["load_after"] = load_snapshot()

    if not args.skip_sweep:
        print("[measure] FAISS index-size sweep ...", flush=True)
        sizes = [int(s) for s in args.sweep_sizes.split(",")]
        art["faiss_index_size_sweep"] = _run_subprocess_worker(
            ["--sweep-worker", ",".join(map(str, sizes)), "--seed", str(args.seed)], timeout_s=3600)

    # n_docs from the v1 index
    try:
        import faiss
        from retrieval.profiles import V1_INDEX_PATH

        art["corpus"]["n_docs"] = int(faiss.read_index(str(V1_INDEX_PATH)).ntotal)
        art["corpus"]["name"] = "v1 hotel corpus (retrieval/corpus, retrieval/index.faiss)"
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=1)
    print(f"\nwrote {out_path}\n")
    print_table(art)
    return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m retrieval.measure_latency",
                                description="Measure retriever latency profiles.")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", help="comma-separated variant names; merges into an existing --out")
    p.add_argument("--remote-url", help="external service URL for the remote_api_wan variant")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--sweep-sizes", default="61,1000,10000,100000,1000000")
    p.add_argument("--table", action="store_true", help="print the table of an existing --out and exit")
    # internal worker modes
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--sweep-worker", help=argparse.SUPPRESS)
    p.add_argument("--queries-file", help=argparse.SUPPRESS)
    p.add_argument("--url", help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.worker:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            qset = json.load(f)
        res = run_worker(VARIANTS[args.worker], qset, args.passes, args.seed, args.url)
        with open(args.worker_out, "w", encoding="utf-8") as f:
            json.dump(res, f)
        return 0
    if args.sweep_worker:
        sizes = [int(s) for s in args.sweep_worker.split(",")]
        res = run_sweep_worker(sizes, 384, args.seed, reps={1_000_000: 50, 300_000: 100})
        with open(args.worker_out, "w", encoding="utf-8") as f:
            json.dump(res, f)
        return 0
    if args.table:
        with open(args.out, "r", encoding="utf-8") as f:
            print_table(json.load(f))
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
