"""
e3v2_phase_a.py — Phase A of the E3-v2 sweep: measure every (item, seed,
condition, path, query_variant) cell that simulator/e3v2_decide.py can land
on, once, with real generations and real grading. DEV ONLY until
docs/research_v2/PREREG_FROZEN exists.

    python -m simulator.e3v2_phase_a --dataset d2_support --split dev --dry-run
    python -m simulator.e3v2_phase_a --dataset d2_support --split dev \
        --env-file <repo>/.env --e1-results <repo>/llm/experiment1_v2/results \
        --index-dir <repo>/retrieval/indexes/d2_support --concurrency 32 --judge-workers 4

---------------------------------------------------------------------------
CONDITIONS AND DISPATCH POINTS
---------------------------------------------------------------------------
A condition is one way the question reaches the controller. Every condition
is expressed as a trace (CONTRACTS §4), so Phase A and Phase B index the same
partials:

  <trace stem>   a real streaming-ASR trace file under
                 datasets/v2/<ds>/speech/traces/<stem>.jsonl: every non-empty
                 partial (sorted by t_ms) plus the final ASR transcript.
  text150        a text timeline at 150 wpm (400 ms per word): the E1-v2 text
                 prefixes at word fractions 0.3/0.5/0.7 (llm/experiment1_v2/
                 prefixes.py, same texts as the self-assessment records) plus
                 the full question; VAD end = speech end + 800 ms hang.

Dispatch point k of a condition = partial k (index in that sorted list) or
the final transcript (partial_index null). simulator/e3v2_decide.py looks
retrievals up by exactly this index.

---------------------------------------------------------------------------
WHAT IS GENERATED (and why only this)
---------------------------------------------------------------------------
1. Query rewrite at every dispatch point, as the live pipeline does
   (retrieval/rewrite.py: build_messages([], prefix); pipeline generation
   parameters temperature 0.2, top_p 0.9, max_tokens 256; fixed seed).
   Deduplicated by prefix text: the rewrite sees nothing else.
2. Retrieval of the rewrite: bge-small + FAISS (retrieval/v2_index.py), top-3
   distinct parent documents (v1's pipeline k). CPU-local.
3. One GROUNDED answer per (item, seed, final text, query_variant), where
   query_variant = sha256 of the ordered retrieved chunk ids: the answer is
   generated at end of turn on the FINAL transcript with the retrieved
   passages, so it depends on the dispatch time only through the passages.
   E1-v2's grounded prompt (prompts.py), the draw seed, temperature 0.2,
   top_p 0.9 and max_tokens 256 (the pipeline's answer cap,
   configs/config.yaml llm.generation).
4. The SKIP path (closed-book answer on the final transcript). E1-v2's
   closed-book answers were capped at 80 tokens; 38% of d2_support's
   full-question answers hit that cap, so they are not comparable with
   256-token grounded answers. Decision: an E1 answer that ended with
   finish_reason "stop" (under 80 tokens) is reused, because the same seed
   and prompt with a 256 cap produce the same text up to that stop; an E1
   answer that hit the 80-token cap is REGENERATED at 256. A final ASR text
   that differs from the reference question gets its own closed-book answer.
5. Grading with the dataset's scorer of record: normalized alias match for
   d1_open; the r3 judges (primary k=3, secondary) for d2_support, through
   llm/experiment1_v2/label.py:grade_draws (≤ --judge-workers, cached).
   Reused E1 skip answers keep E1's verdicts from labels_dev.json.

Everything is resumable: each stage appends JSONL keyed by its inputs and
skips keys already present. --dry-run prints exact call counts for what is
known (rewrites; answers once retrieval exists) and an upper bound otherwise.

---------------------------------------------------------------------------
OUTPUT (<out>/<dataset>/)
---------------------------------------------------------------------------
  rewrites.jsonl, retrievals.jsonl, answers.jsonl, skip_answers.jsonl, grades.jsonl
  store_dev.jsonl     the Phase A store simulator/e3v2_decide.py reads:
                      {"kind":"retrieval", item_id, condition, partial_index,
                       retrieved_doc_ids, query_variant, query_text, rewrite, ...}
                      {"kind":"cell", item_id, seed, condition, path, query_variant,
                       answer_correct, answer, n_prompt_tokens, ...}
  conditions_dev.jsonl the per-(item, condition) trace used (Phase B timing)
  manifest_dev.json   provenance: git commit, inputs + sha256, parameters,
                      counts, wall time (the endpoint URL is never recorded)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1_v2 import common as e1common  # noqa: E402
from llm.experiment1_v2.prefixes import normalize_text, text_prefixes  # noqa: E402
from llm.experiment1_v2.prompts import build_messages as e1_build_messages  # noqa: E402

PREREG_FROZEN = _REPO_ROOT / "docs" / "research_v2" / "PREREG_FROZEN"
SEEDS = (11, 23, 37, 53, 71)
TEXT_CONDITION = "text150"
TEXT_MS_PER_WORD = 400.0       # 150 wpm
TEXT_VAD_HANG_MS = 800.0       # the live VAD's silence hang (SPEECH.md)
TEXT_FRACS = (0.3, 0.5, 0.7)   # E1-v2's word fractions (config.json)
REWRITE_SEED = 0
RETRIEVAL_K = 3                # v1 pipeline: retriever.search(query, k=3)
DEFAULT_OUT = _REPO_ROOT / "simulator" / "e3v2_phase_a"
MODEL_KEY = "mistral7b_v03"


def sha(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def query_variant_for(doc_ids: Sequence[str]) -> str:
    # identical to simulator/e3v2_decide.py (order-sensitive)
    return "qv-" + hashlib.sha256("\x1f".join(doc_ids).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
def text_trace(item: dict) -> dict:
    """The text150 condition as a CONTRACTS §4 trace."""
    question = normalize_text(item["question"])
    words = question.split(" ")
    n = len(words)
    end = n * TEXT_MS_PER_WORD
    partials = []
    for p in text_prefixes(question, TEXT_FRACS, include_full=False):
        t = p["n_words"] * TEXT_MS_PER_WORD
        partials.append({"t_ms": t, "available_ms": t, "text": p["text"], "word_frac": p["word_frac"],
                         "n_words": p["n_words"]})
    vad_end = end + TEXT_VAD_HANG_MS
    return {
        "item_id": item["id"], "condition": TEXT_CONDITION, "voice_id": None, "noise": None,
        "words": [{"w": w, "start_ms": i * TEXT_MS_PER_WORD, "end_ms": (i + 1) * TEXT_MS_PER_WORD}
                  for i, w in enumerate(words)],
        "duration_ms": end, "vad_user_end_ms": vad_end, "endpoint_reason": "text_timeline",
        "partials": partials,
        "asr_final": {"t_ms": vad_end, "available_ms": vad_end, "text": question},
        # the three text prefixes are 0.2 of the duration apart
        "partial_interval_ms": 0.2 * end,
        "trace_version": "text150-v1",
    }


def load_trace_conditions(dataset: str, keep: set, only: Optional[set]) -> Dict[str, Dict[str, dict]]:
    """{condition stem: {item_id: trace}} for the real speech traces."""
    tdir = _REPO_ROOT / "datasets" / "v2" / dataset / "speech" / "traces"
    out: Dict[str, Dict[str, dict]] = {}
    if not tdir.exists():
        return out
    for f in sorted(tdir.glob("*.jsonl")):
        if only is not None and f.stem not in only:
            continue
        rows = {}
        for tr in e1common.read_jsonl(f, tolerate_partial_tail=False):
            if tr["item_id"] in keep:
                tr = dict(tr, condition=f.stem)
                tr["partials"] = sorted([p for p in tr.get("partials") or [] if normalize_text(p.get("text", ""))],
                                        key=lambda p: float(p["t_ms"]))
                rows[tr["item_id"]] = tr
        out[f.stem] = rows
    return out


def dispatch_points(trace: dict) -> List[dict]:
    """[{partial_index, text}] for every partial plus the final (index None)."""
    pts = [{"partial_index": k, "text": normalize_text(p["text"])} for k, p in enumerate(trace["partials"])]
    pts.append({"partial_index": None, "text": normalize_text((trace.get("asr_final") or {}).get("text", ""))})
    return pts


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------
async def run_calls(backend, jobs: List[dict], concurrency: int, writer, label: str,
                    progress_every: int = 200) -> dict:
    """jobs: {key, messages, seed, temperature, top_p, max_tokens, meta}.
    Appends one record per settled job; failures are recorded and retried on
    the next run (their key is not written as done)."""
    sem = asyncio.Semaphore(max(1, concurrency))
    stats = Counter()
    t0 = time.perf_counter()

    async def one(job):
        async with sem:
            try:
                res = await backend.generate(job["messages"], seed=job["seed"], temperature=job["temperature"],
                                             top_p=job["top_p"], max_tokens=job["max_tokens"])
            except Exception as error:  # noqa: BLE001
                stats["failed"] += 1
                print(f"  [{label}] FAILED {job['key'][:12]}: {type(error).__name__}: "
                      f"{backend.redact(error)[:200]}", file=sys.stderr)
                return
            gen = res.generation
            writer.write({
                "key": job["key"], **job["meta"],
                "text": gen.text, "n_tokens": len(gen.tokens), "finish_reason": res.finish_reason,
                "ttft_ms": res.ttft_ms, "gen_ms": res.gen_ms, "served_model": res.served_model,
                "seed": job["seed"], "temperature": job["temperature"], "top_p": job["top_p"],
                "max_tokens": job["max_tokens"], "attempts": res.attempts,
                "prompt_chars": sum(len(m["content"]) for m in job["messages"]),
                "created_at_utc": e1common.now_utc(),
            })
            stats["done"] += 1
            if stats["done"] % progress_every == 0:
                rate = stats["done"] / max(1e-9, time.perf_counter() - t0)
                print(f"  [{label}] {stats['done']}/{len(jobs)} ({rate:.2f}/s)")

    await asyncio.gather(*(one(j) for j in jobs))
    stats["wall_s"] = time.perf_counter() - t0
    return dict(stats)


def make_backend(kind: str = "vllm"):
    from llm.config import load_config as load_llm_config
    from llm.experiment1_v2.backends import make_backend as e1_make_backend

    llm_cfg = load_llm_config(model_key=MODEL_KEY)
    top_k = int((llm_cfg.get("logprobs") or {}).get("top_k", 5))
    run_cfg = e1common.load_config()["run"]
    return llm_cfg, e1_make_backend(kind, llm_cfg, top_k, run_cfg)


def answer_params() -> dict:
    from llm.config import load_config as load_llm_config
    gen = (load_llm_config(model_key=MODEL_KEY).get("generation") or {})
    return {"temperature": float(gen.get("temperature", 0.2)), "top_p": float(gen.get("top_p", 0.9)),
            "max_tokens": int(gen.get("max_tokens", 256))}


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
class PhaseA:
    def __init__(self, args):
        self.args = args
        self.ds = args.dataset
        self.out = Path(args.out) / self.ds
        self.out.mkdir(parents=True, exist_ok=True)
        paths = e1common.resolve_dataset_paths(self.ds)
        self.items = e1common.select_split(e1common.load_items(paths["items"]),
                                           e1common.load_splits(paths["splits"]), args.split)
        if args.limit:
            self.items = self.items[: args.limit]
        self.by_id = {i["id"]: i for i in self.items}
        self.e1cfg = e1common.load_config()
        self.e1dir = Path(args.e1_results) / self.ds / MODEL_KEY
        self.params = answer_params()
        self.seeds = tuple(int(s) for s in args.seeds.split(","))
        # conditions
        wanted = None if args.conditions in ("all", "") else set(args.conditions.split(","))
        self.conditions: Dict[str, Dict[str, dict]] = {}
        if wanted is None or TEXT_CONDITION in wanted:
            self.conditions[TEXT_CONDITION] = {i["id"]: text_trace(i) for i in self.items}
        self.conditions.update(load_trace_conditions(self.ds, set(self.by_id), wanted))

    # -- paths ---------------------------------------------------------------
    def f(self, name: str) -> Path:
        return self.out / f"{name}_{self.args.split}.jsonl"

    def done_keys(self, name: str) -> set:
        path = self.f(name)
        return {r["key"] for r in e1common.read_jsonl(path)} if path.exists() else set()

    # -- E1 closed-book reuse ------------------------------------------------
    def e1_closed_book(self) -> Dict[tuple, dict]:
        """(item_id, seed) -> E1 full-question closed-book record (+ verdicts)."""
        out = {}
        sig = self.e1dir / f"signals_{self.args.split}.jsonl"
        for r in e1common.read_jsonl(sig):
            p = r.get("prefix") or {}
            if p.get("kind") == "full" and p.get("source") == "text" and r["item_id"] in self.by_id:
                out[(r["item_id"], int(r["draw_seed"]))] = {
                    "answer": r.get("closed_book_answer") or "", "finish_reason": r.get("finish_reason"),
                    "n_tokens": r.get("n_tokens"), "call_key": r.get("call_key")}
        labels = e1common.load_json(self.e1dir / f"labels_{self.args.split}.json")
        for d in labels["draws"]:
            key = (d["item_id"], int(d["draw_seed"]))
            if d.get("condition", "closed_book") == "closed_book" and key in out:
                out[key]["alias_correct"] = d.get("alias_correct")
                out[key]["judges"] = {role: {k: v.get(k) for k in ("answer_correct", "parsed_ok", "n_samples",
                                                                    "self_consistency", "unanimous", "judge_key")}
                                      for role, v in (d.get("judges") or {}).items()}
        return out

    # -- stage 1: rewrites ---------------------------------------------------
    def rewrite_jobs(self) -> List[dict]:
        from retrieval.rewrite import build_messages as rw_messages
        texts = {}
        for traces in self.conditions.values():
            for tr in traces.values():
                for pt in dispatch_points(tr):
                    if pt["text"]:
                        texts.setdefault(sha("rewrite", pt["text"]), pt["text"])
        done = self.done_keys("rewrites")
        return [{"key": k, "messages": rw_messages([], t), "seed": REWRITE_SEED,
                 "temperature": self.params["temperature"], "top_p": self.params["top_p"],
                 "max_tokens": self.params["max_tokens"], "meta": {"prefix_text": t}}
                for k, t in texts.items() if k not in done]

    def rewrites(self) -> Dict[str, str]:
        out = {}
        for r in e1common.read_jsonl(self.f("rewrites")) if self.f("rewrites").exists() else []:
            rw = " ".join((r.get("text") or "").split())
            out[r["prefix_text"]] = rw or r["prefix_text"]
        return out

    # -- stage 2: retrieval --------------------------------------------------
    def retrieve(self) -> dict:
        from retrieval.v2_index import load_index, sentence_transformer_encoder
        rewrites = self.rewrites()
        done = self.done_keys("retrievals")
        todo = sorted({rw for rw in rewrites.values() if sha("retrieval", rw) not in done})
        if not todo:
            return {"queries": 0}
        index = load_index(self.args.index_dir)
        encoder = sentence_transformer_encoder(index.meta["model"])
        t0 = time.perf_counter()
        with e1common.JsonlAppender(self.f("retrievals")) as w:
            for start in range(0, len(todo), 256):
                batch = todo[start:start + 256]
                hits = index.search_vectors(encoder(batch), RETRIEVAL_K, dedupe_parents=True)
                for q, hs in zip(batch, hits):
                    w.write({"key": sha("retrieval", q), "query": q, "k": RETRIEVAL_K,
                             "hits": [{"doc_id": h["doc_id"], "parent_doc_id": h["parent_doc_id"],
                                       "score": h["score"], "row": h["row"]} for h in hs]})
        return {"queries": len(todo), "wall_s": time.perf_counter() - t0,
                "index_meta": {k: index.meta.get(k) for k in ("model", "metric", "normalize", "corpus_sha256",
                                                              "n_docs", "max_words")}}

    def retrieval_rows(self) -> List[dict]:
        """One kind=retrieval store row per (item, condition, dispatch point)."""
        rewrites = self.rewrites()
        hits = {r["query"]: r["hits"] for r in e1common.read_jsonl(self.f("retrievals"))} \
            if self.f("retrievals").exists() else {}
        rows = []
        for cond, traces in self.conditions.items():
            for item_id, tr in traces.items():
                for pt in dispatch_points(tr):
                    rw = rewrites.get(pt["text"])
                    if rw is None or rw not in hits:
                        continue
                    ids = [h["doc_id"] for h in hits[rw]]
                    rows.append({"kind": "retrieval", "item_id": item_id, "condition": cond,
                                 "partial_index": pt["partial_index"], "query_text": pt["text"],
                                 "rewrite": rw, "retrieved_doc_ids": ids,
                                 "retrieved_parent_ids": [h["parent_doc_id"] for h in hits[rw]],
                                 "retrieved_scores": [h["score"] for h in hits[rw]],
                                 "query_variant": query_variant_for(ids)})
        return rows

    # -- stage 3: grounded answers ------------------------------------------
    def passages(self, doc_ids: Sequence[str]) -> List[dict]:
        if not hasattr(self, "_docs"):
            from retrieval.v2_index import load_index
            self._docs = {d["doc_id"]: d for d in load_index(self.args.index_dir).docs}
        return [{"doc_id": i, "text": self._docs[i]["text"]} for i in doc_ids]

    def answer_units(self) -> Dict[str, dict]:
        """key -> {item_id, seed, final_text, query_variant, doc_ids} (deduplicated)."""
        units = {}
        for row in self.retrieval_rows():
            tr = self.conditions[row["condition"]][row["item_id"]]
            final = normalize_text(tr["asr_final"]["text"])
            for seed in self.seeds:
                key = sha("answer", row["item_id"], seed, final, row["query_variant"])
                units.setdefault(key, {"item_id": row["item_id"], "seed": seed, "final_text": final,
                                       "query_variant": row["query_variant"],
                                       "doc_ids": row["retrieved_doc_ids"]})
        return units

    def answer_jobs(self) -> List[dict]:
        done = self.done_keys("answers")
        jobs = []
        for key, u in self.answer_units().items():
            if key in done:
                continue
            item = self.by_id[u["item_id"]]
            msgs = e1_build_messages(item, u["final_text"], self.e1cfg, passages=self.passages(u["doc_ids"]))
            jobs.append({"key": key, "messages": msgs, "seed": u["seed"], **self.params,
                         "meta": {k: u[k] for k in ("item_id", "seed", "final_text", "query_variant", "doc_ids")}})
        return jobs

    # -- stage 4: skip answers ----------------------------------------------
    def skip_units(self) -> Dict[str, dict]:
        e1 = self.e1_closed_book()
        units = {}
        for cond, traces in self.conditions.items():
            for item_id, tr in traces.items():
                final = normalize_text(tr["asr_final"]["text"])
                question = normalize_text(self.by_id[item_id]["question"])
                for seed in self.seeds:
                    key = sha("skip", item_id, seed, final)
                    prior = e1.get((item_id, seed)) if final == question else None
                    reuse = prior is not None and prior.get("finish_reason") == "stop"
                    units.setdefault(key, {"item_id": item_id, "seed": seed, "final_text": final,
                                           "reuse_e1": reuse, "e1": prior if reuse else None,
                                           "e1_finish_reason": (prior or {}).get("finish_reason")})
        return units

    def skip_jobs(self) -> List[dict]:
        done = self.done_keys("skip_answers")
        jobs = []
        for key, u in self.skip_units().items():
            if u["reuse_e1"] or key in done:
                continue
            msgs = e1_build_messages(self.by_id[u["item_id"]], u["final_text"], self.e1cfg)
            jobs.append({"key": key, "messages": msgs, "seed": u["seed"], **self.params,
                         "meta": {"item_id": u["item_id"], "seed": u["seed"], "final_text": u["final_text"],
                                  "regenerated_because": ("e1_truncated_at_80" if u["e1_finish_reason"] == "length"
                                                          else "final_text_differs_or_missing")}})
        return jobs

    # -- stage 5: grading ----------------------------------------------------
    def grade(self, workers: int, judge_backend: str) -> dict:
        from llm.experiment1_v2 import judges as judges_mod
        from llm.experiment1_v2.label import grade_draws

        done = self.done_keys("grades")
        todo = []
        for name, cond in (("answers", "retrieval"), ("skip_answers", "closed_book")):
            if not self.f(name).exists():
                continue
            for r in e1common.read_jsonl(self.f(name)):
                if r["key"] in done:
                    continue
                rec = {"item_id": r["item_id"], "draw_seed": r["seed"], "condition": cond,
                       "grounded_answer": r["text"], "closed_book_answer": r["text"], "_key": r["key"]}
                todo.append(rec)
        if not todo:
            return {"graded": 0}
        source = self.e1cfg["labels"]["correctness_source"].get(self.ds,
                                                                 self.e1cfg["labels"]["correctness_source"]["default"])
        roles = [] if source == "alias" else ["primary", "secondary"]
        judges = judges_mod.make_judges(roles, judge_backend) if roles else {}
        cache_dir = self.out / "judge_cache"
        t0 = time.perf_counter()
        n = 0
        with e1common.JsonlAppender(self.f("grades")) as w:
            for start in range(0, len(todo), 200):
                batch = todo[start:start + 200]
                rows = grade_draws(batch, self.by_id, judges, self.e1cfg, cache_dir, workers)
                for rec, row in zip(batch, rows):
                    w.write({"key": rec["_key"], "item_id": rec["item_id"], "seed": rec["draw_seed"],
                             "alias_correct": row["alias_correct"],
                             "judges": {role: {k: v.get(k) for k in ("answer_correct", "parsed_ok", "n_samples",
                                                                      "self_consistency", "unanimous",
                                                                      "judge_key", "response_model")}
                                        for role, v in row["judges"].items()},
                             "correctness_source": source})
                n += len(batch)
                print(f"  [grade] {n}/{len(todo)} ({n / max(1e-9, time.perf_counter() - t0):.2f}/s)")
        return {"graded": n, "wall_s": time.perf_counter() - t0, "correctness_source": source}

    # -- stage 6: the store --------------------------------------------------
    @staticmethod
    def _correct(source: str, alias, judges: dict):
        if source == "alias":
            return bool(alias), "alias"
        role = source.split(":", 1)[1]
        v = (judges.get(role) or {}).get("answer_correct")
        if v is not None:
            return bool(v), source
        sec = (judges.get("secondary") or {}).get("answer_correct")
        return (None, "unresolved") if sec is None else (bool(sec), "judge:secondary(fallback)")

    def build_store(self) -> dict:
        source = self.e1cfg["labels"]["correctness_source"].get(self.ds,
                                                                 self.e1cfg["labels"]["correctness_source"]["default"])
        grades = {r["key"]: r for r in e1common.read_jsonl(self.f("grades"))} if self.f("grades").exists() else {}
        answers = {r["key"]: r for r in e1common.read_jsonl(self.f("answers"))} if self.f("answers").exists() else {}
        skips = {r["key"]: r for r in e1common.read_jsonl(self.f("skip_answers"))} \
            if self.f("skip_answers").exists() else {}
        stats = Counter()
        rows = self.retrieval_rows()
        cells = []
        for row in rows:
            tr = self.conditions[row["condition"]][row["item_id"]]
            final = normalize_text(tr["asr_final"]["text"])
            for seed in self.seeds:
                key = sha("answer", row["item_id"], seed, final, row["query_variant"])
                a, g = answers.get(key), grades.get(key)
                if a is None or g is None:
                    stats["retrieve_cell_missing"] += 1
                    continue
                correct, how = self._correct(source, g["alias_correct"], g["judges"])
                stats[f"retrieve_{how}"] += 1
                cells.append({"kind": "cell", "item_id": row["item_id"], "seed": seed,
                              "condition": row["condition"], "path": "retrieve",
                              "query_variant": row["query_variant"], "answer_correct": correct,
                              "correctness_source": how, "alias_correct": g["alias_correct"],
                              "judges": g["judges"], "answer": a["text"], "retrieved_doc_ids": row["retrieved_doc_ids"],
                              "n_prompt_tokens": round(a["prompt_chars"] / 4.103, 1),
                              "n_tokens": a["n_tokens"], "finish_reason": a["finish_reason"],
                              "ttft_ms": a["ttft_ms"], "gen_ms": a["gen_ms"], "answer_key": key})
        for (cond, item_id), _ in {(c, i): 1 for c, ts in self.conditions.items() for i in ts}.items():
            tr = self.conditions[cond][item_id]
            final = normalize_text(tr["asr_final"]["text"])
            for seed in self.seeds:
                key = sha("skip", item_id, seed, final)
                u = self._skip_units.get(key) if hasattr(self, "_skip_units") else None
                if u is None:
                    self._skip_units = self.skip_units()
                    u = self._skip_units.get(key)
                if u["reuse_e1"]:
                    e = u["e1"]
                    correct, how = self._correct(source, e.get("alias_correct"), e.get("judges") or {})
                    text, nt, fr, origin = e["answer"], e.get("n_tokens"), e.get("finish_reason"), "e1_reused"
                    chars = None
                else:
                    a, g = skips.get(key), grades.get(key)
                    if a is None or g is None:
                        stats["skip_cell_missing"] += 1
                        continue
                    correct, how = self._correct(source, g["alias_correct"], g["judges"])
                    text, nt, fr, origin, chars = a["text"], a["n_tokens"], a["finish_reason"], "regenerated", \
                        a["prompt_chars"]
                stats[f"skip_{origin}"] += 1
                stats[f"skip_{how}"] += 1
                cells.append({"kind": "cell", "item_id": item_id, "seed": seed, "condition": cond, "path": "skip",
                              "query_variant": "none", "answer_correct": correct, "correctness_source": how,
                              "answer": text, "origin": origin, "n_tokens": nt, "finish_reason": fr,
                              "n_prompt_tokens": None if chars is None else round(chars / 4.103, 1),
                              "answer_key": key})
        unresolved = [c for c in cells if c["answer_correct"] is None]
        path = self.out / f"store_{self.args.split}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in rows + [c for c in cells if c["answer_correct"] is not None]:
                fh.write(json.dumps(r) + "\n")
        tmp.replace(path)
        with open(self.out / f"conditions_{self.args.split}.jsonl", "w", encoding="utf-8") as fh:
            for cond, traces in self.conditions.items():
                for tr in traces.values():
                    fh.write(json.dumps(tr) + "\n")
        stats["retrieval_rows"] = len(rows)
        stats["cells"] = len(cells) - len(unresolved)
        stats["cells_unresolved_excluded"] = len(unresolved)
        return dict(stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E3-v2 Phase A: real cells for the Phase B evaluator.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--conditions", default="all", help="all | text150 | <trace stem>[,...]")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--e1-results", default=str(_REPO_ROOT / "llm" / "experiment1_v2" / "results"))
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--env-file", default=None, help="load credentials from this .env (values never printed)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--judge-workers", type=int, default=4)
    ap.add_argument("--judge-backend", default="azure")
    ap.add_argument("--backend", default="vllm", help="vllm | mock (plumbing only; never data)")
    ap.add_argument("--stages", default="rewrite,retrieve,answer,skip,grade,store")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.split not in ("dev", "test"):
        raise SystemExit("--split must be dev or test")
    if args.split == "test" and not PREREG_FROZEN.exists():
        raise SystemExit("Refusing --split test: sealed until docs/research_v2/PREREG_FROZEN exists.")
    if args.judge_workers > 4:
        raise SystemExit("--judge-workers is capped at 4 (judge rate limit)")
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    os.environ.setdefault("VOICE_RAG_MODEL", MODEL_KEY)
    if args.index_dir is None:
        args.index_dir = str(_REPO_ROOT / "retrieval" / "indexes" / args.dataset)

    pa = PhaseA(args)
    stages = set(args.stages.split(","))
    manifest_path = pa.out / f"manifest_{args.split}.json"
    manifest = e1common.load_json(manifest_path) if manifest_path.exists() else {"runs": []}
    run = {"started_at_utc": e1common.now_utc(), "git_commit": e1common.git_commit(), "args": {
        k: v for k, v in vars(args).items() if k != "env_file"}, "dataset": args.dataset, "split": args.split,
        "n_items": len(pa.items), "conditions": {c: len(t) for c, t in pa.conditions.items()},
        "answer_params": pa.params, "rewrite_seed": REWRITE_SEED, "retrieval_k": RETRIEVAL_K,
        "model_key": MODEL_KEY, "endpoint": "read from VOICE_RAG_LLM_BASE_URL; deliberately not recorded",
        "inputs": {}, "stages": {}}
    for name in ("signals", "labels"):
        p = pa.e1dir / f"{name}_{args.split}.{'jsonl' if name == 'signals' else 'json'}"
        if p.exists():
            run["inputs"][str(p.name)] = e1common.sha256_file(p)
    print(f"[phase-a] {args.dataset}/{args.split}: {len(pa.items)} items, conditions "
          f"{ {c: len(t) for c, t in pa.conditions.items()} }")

    if args.dry_run:
        rw = pa.rewrite_jobs()
        n_pts = sum(len(dispatch_points(tr)) for ts in pa.conditions.values() for tr in ts.values())
        have_ret = pa.f("retrievals").exists()
        ans = pa.answer_jobs() if have_ret else None
        skip = pa.skip_jobs()
        units = pa.skip_units()
        print(f"  dispatch points: {n_pts}")
        print(f"  rewrite calls (dedup by text, not yet done): {len(rw)}")
        if ans is not None:
            print(f"  grounded answer calls (exact, after retrieval dedup): {len(ans)}")
        else:
            print(f"  grounded answer calls: <= {n_pts * len(pa.seeds)} (exact after the retrieve stage)")
        print(f"  skip answers: {len(units)} units, {sum(u['reuse_e1'] for u in units.values())} reuse E1, "
              f"{len(skip)} to generate")
        return 0

    async def gen_stage(label, jobs, name):
        if not jobs:
            return {"done": 0}
        llm_cfg, backend = make_backend(args.backend)
        try:
            served = await backend.served_models()
            if llm_cfg["model"] not in served:
                raise SystemExit(f"endpoint serves {served}, not {llm_cfg['model']}")
            with e1common.JsonlAppender(pa.f(name)) as w:
                return await run_calls(backend, jobs, args.concurrency, w, label)
        finally:
            await backend.close()

    if "rewrite" in stages:
        jobs = pa.rewrite_jobs()
        print(f"[phase-a] rewrite: {len(jobs)} calls")
        run["stages"]["rewrite"] = asyncio.run(gen_stage("rewrite", jobs, "rewrites"))
    if "retrieve" in stages:
        print("[phase-a] retrieve")
        run["stages"]["retrieve"] = pa.retrieve()
    if "answer" in stages:
        jobs = pa.answer_jobs()
        print(f"[phase-a] grounded answers: {len(jobs)} calls")
        run["stages"]["answer"] = asyncio.run(gen_stage("answer", jobs, "answers"))
    if "skip" in stages:
        jobs = pa.skip_jobs()
        print(f"[phase-a] skip answers to generate: {len(jobs)}")
        run["stages"]["skip"] = asyncio.run(gen_stage("skip", jobs, "skip_answers"))
    if "grade" in stages:
        print("[phase-a] grade")
        run["stages"]["grade"] = pa.grade(args.judge_workers, args.judge_backend)
    if "store" in stages:
        run["stages"]["store"] = pa.build_store()
        print(f"[phase-a] store: {run['stages']['store']}")
    run["finished_at_utc"] = e1common.now_utc()
    manifest["runs"].append(run)
    e1common.atomic_write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
