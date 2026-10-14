"""
latency_model_v2.py — trace-driven TTFA model for research v2.

TTFA = time from the TRUE end of user speech to the first audio sample.
It is assembled from component distributions fitted on measured logs:

    eot_detect        true speech end -> system end-of-turn decision
                      (0 on the v1 text path; VAD hang on real audio, from
                      speech traces when available)
    dispatch_overhead end-of-turn -> retrieval launch when retrieval is
                      dispatched at end of turn (ASR final + controller)
    retrieval         the retrieval call itself, per latency profile
                      (HIDEABLE: it can overlap the user's speech)
    pre_llm_overhead  max(end of turn, retrieval result) -> LLM request
    prefill (TTFT)    LLM request -> first token, as a function of prompt
                      length (residual pool + fitted slope per token)
    gen_to_tts_input  first token -> text handed to TTS: the full answer in
                      batch TTS (v1), the first complete clause in streaming
                      TTS; sampled jointly from per-run generation records
                      (n_tokens, first-clause token index, per-token time)
    tts_first_audio   TTS input -> first audio: the logged v1 value
                      ("logged_v1": the E2/E3 runs used a synthetic tone, no
                      real synthesis) or real resident piper synthesis time
                      as a function of clause length ("piper_resident")

The model separates retrieval's TTFA cost into

    HIDEABLE      the retrieval call (only its part after end of turn costs
                  latency, so early dispatch can hide it), and
    NON-HIDEABLE  the retrieve-minus-skip difference in everything after
                  the call: prefill of the passages, a longer/different
                  answer before TTS input, the first clause's synthesis.

v1 E3 showed the second dominates: paired retrieve-minus-skip TTFA is
~1.57 s (sentence_end) while the call itself is ~0.6 s. The fitted model
carries the measured paired increments per TTS mode, exported through
signals_retrieval() as signals["retrieval"] (CONTRACTS §6 plus
"nonhideable_increment_ms").

API:
    m = LatencyModel.fit(["logs/conv_*.jsonl", ...], profiles_path=...)
    m.sample_ttfa(path, dispatch_ms, user_end_ms, profile, n_prompt_tokens, rng,
                  tts_mode="batch") -> dict
    m.signals_retrieval("faiss", tts_mode="streaming") -> dict

CLI (fit + fidelity report):
    python -m harness.latency_model_v2 --e2 "logs/conv_*.jsonl" \\
        --e3 "<main checkout>/logs/e3full_*.jsonl" --v2 "logs/v2_llm_tts/*.jsonl" \\
        --prefill analysis/llm_tts_v2/measure_llm_tts.json \\
        --profiles retrieval/latency_profiles_v2.json --out-dir analysis/latency_model_v2
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harness.latency_breakdown import _read_events, parse_turn, policy_from_name  # noqa: E402

QUANTS = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
_BOUNDARY = re.compile(r"[.!?;\n]\s")
DEFAULT_CHARS_PER_TOKEN = 4.0
SYSTEM_PROMPT_CHARS = None  # filled lazily from llm.client


def quantiles(values: Sequence[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {k: None for k in QUANTS}
    return {k: float(np.quantile(v, q)) for k, q in QUANTS.items()}


def _token_text(tok: str) -> str:
    if tok.startswith("<0x") and tok.endswith(">"):
        try:
            return bytes([int(tok[3:-1], 16)]).decode("utf-8", errors="ignore")
        except ValueError:
            return ""
    return tok.replace("▁", " ")


def first_clause_token_index(tokens: List[str]) -> Optional[int]:
    """
    Index of the token at which streaming TTS would have its first complete
    clause (a boundary char followed by whitespace, as ClauseAccumulator
    detects it). None when the answer is a single clause (ready only when
    the stream ends).
    """
    text = ""
    for i, tok in enumerate(tokens):
        text += _token_text(tok)
        if _BOUNDARY.search(text.lstrip()):
            return i
    return None


def _system_prompt_chars() -> int:
    global SYSTEM_PROMPT_CHARS
    if SYSTEM_PROMPT_CHARS is None:
        try:
            from llm.client import SYSTEM_PROMPT

            SYSTEM_PROMPT_CHARS = len(SYSTEM_PROMPT)
        except Exception:
            SYSTEM_PROMPT_CHARS = 330
    return SYSTEM_PROMPT_CHARS


def prompt_chars(question: str, passages: List[str]) -> int:
    if passages:
        body = "Reference passages:\n" + "\n".join(f"- {p}" for p in passages) + f"\n\nGuest question: {question}"
    else:
        body = f"Guest question: {question}"
    return _system_prompt_chars() + len(body)


# ---------------------------------------------------------------------------
# Per-run records
# ---------------------------------------------------------------------------
def scenario_id_from_name(name: str) -> Optional[str]:
    m = re.search(r"_s(\d{3})_seed(\d+)", name)  # e3full_s001_seed11_...
    if m:
        return m.group(1)
    m = re.search(r"_s(\d+)_(\d{3})$", name)  # conv_<policy>_s<seed>_<scenario>
    if m:
        return m.group(2)
    m = re.search(r"_s(\d{3})$", name)  # v2llm_<mode>_<path>_s001
    if m:
        return m.group(1)
    return None


def seed_from_name(name: str) -> Optional[str]:
    m = re.search(r"_seed(\d+)", name)
    if m:
        return m.group(1)
    m = re.search(r"_s(\d+)_(\d{3})$", name)
    if m:
        return m.group(1)
    return None


def run_record(log_path) -> Optional[dict]:
    path = Path(log_path)
    events = _read_events(path)
    turn = parse_turn(events)
    if turn is None:
        return None
    first: Dict[str, dict] = {}
    for e in events:
        if e.get("turn_id") == turn["turn_id"]:
            first.setdefault(e["event_type"], e)
    lf = (first.get("llm_final") or {}).get("payload") or {}
    tokens = [t.get("token", "") for t in (lf.get("logprobs") or [])]
    n_tokens = lf.get("n_tokens") or len(tokens)
    gen = turn["generation_ms"]
    per_token = gen / (n_tokens - 1) if (gen is not None and n_tokens and n_tokens > 1) else None
    fci = first_clause_token_index(tokens) if tokens else None
    question = ((first.get("asr_final") or {}).get("payload") or {}).get("text", "")
    passages = ((first.get("retrieval_result") or {}).get("payload") or {}).get("passages") or []
    answer = lf.get("text", "") or ""
    first_clause_text = answer.strip()
    m = re.search(r"(?<=[.!?;\n])\s+", answer.strip())
    if m:
        first_clause_text = answer.strip()[: m.start()]
    off = turn["offsets_ms"]
    st = turn["stages_ms"]
    tts_ready = [
        e["payload"] for e in events
        if e.get("event_type") == "tts_clause_ready" and e.get("turn_id") == turn["turn_id"]
    ]
    ue_payload = (first.get("user_end") or {}).get("payload") or {}
    name = path.stem
    rec = {
        "log": path.name,
        "scenario": scenario_id_from_name(name),
        "seed": seed_from_name(name),
        "policy": policy_from_name(name) or "sentence_end",
        "path": turn["path"],
        "tts_mode": turn["tts_mode"],
        "ttfa_ms": turn["ttfa_ms"],
        "speech_end_offset_ms": ue_payload.get("speech_end_offset_ms"),
        "launch_offset_ms": off.get("retrieval_launched"),
        "retrieval_ms": turn["retrieval_ms"],
        "retrieval_exposed_ms": st["retrieval_exposed"],
        "dispatch_wait_ms": st["dispatch_wait"],
        "pre_llm_overhead_ms": st["pre_llm_overhead"],
        "skip_overhead_ms": (off.get("llm_request") or 0) if turn["path"] == "skip" else None,
        "ttft_ms": turn["ttft_ms"],
        "n_tokens": n_tokens,
        "per_token_ms": per_token,
        "first_clause_token_index": fci,
        "first_clause_words": len(first_clause_text.split()),
        "tts_stage_ms": st["tts_first_audio"],
        "prompt_chars": prompt_chars(question, passages),
        "n_passages": len(passages),
        "observed_first_clause_ms": (
            off["llm_first_clause"] - off["llm_first_token"]
            if off.get("llm_first_clause") is not None and off.get("llm_first_token") is not None
            else None
        ),
        "tts_clause0": next(
            ({"words": len(c.get("text", "").split()), "synth_ms": c.get("synth_ms")} for c in tts_ready if c.get("clause_index") == 0),
            None,
        ),
    }
    return rec


def load_records(patterns: Iterable[str], source: str) -> List[dict]:
    out = []
    for pat in patterns or []:
        for p in sorted(glob.glob(pat)):
            r = run_record(p)
            if r is not None:
                r["source"] = source
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class LatencyModel:
    """See module docstring. All times in ms."""

    def __init__(self, params: dict):
        self.p = params

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        records: List[dict],
        profiles_path: Optional[str] = None,
        prefill_sweep: Optional[List[dict]] = None,
        tts_records: Optional[List[dict]] = None,
        eot_samples_ms: Optional[List[float]] = None,
        chars_per_token: Optional[float] = None,
    ) -> "LatencyModel":
        """
        records: run_record() dicts (E2/E3/v2 logs) — the LLM, overhead and
        generation pools and the in-pipeline retrieval profile.
        prefill_sweep: [{"ttft_ms","prompt_tokens","prompt_chars","k"}]
        from harness/measure_llm_tts.py — fixes the TTFT slope per prompt
        token and the chars-per-token ratio.
        tts_records: records with tts_clause0 (real piper synthesis).
        """
        p: dict = {"version": 1}

        # chars per token
        cpt = chars_per_token
        if cpt is None and prefill_sweep:
            pts = [(s["prompt_chars"], s["prompt_tokens"]) for s in prefill_sweep if s.get("prompt_tokens")]
            if pts:
                cpt = float(np.sum([c for c, _ in pts]) / np.sum([t for _, t in pts]))
        p["chars_per_token"] = cpt or DEFAULT_CHARS_PER_TOKEN

        # TTFT slope per prompt token
        slope, slope_src = 0.0, "none"
        if prefill_sweep:
            xs = np.array([s["prompt_tokens"] for s in prefill_sweep if s.get("prompt_tokens")], float)
            ys = np.array([s["ttft_ms"] for s in prefill_sweep if s.get("prompt_tokens")], float)
            if xs.size >= 3 and np.ptp(xs) > 0:
                # robust: slope between per-k medians
                by_k = defaultdict(list)
                for s in prefill_sweep:
                    if s.get("prompt_tokens"):
                        by_k[s["k"]].append((s["prompt_tokens"], s["ttft_ms"]))
                med = sorted((np.median([a for a, _ in v]), np.median([b for _, b in v])) for v in by_k.values())
                if len(med) >= 2:
                    mx = np.array([m[0] for m in med]); my = np.array([m[1] for m in med])
                    slope = float(np.polyfit(mx, my, 1)[0])
                    slope_src = "prefill_sweep_medians"
        else:
            # from logs: retrieve vs skip TTFT medians vs estimated tokens
            r = [x for x in records if x["path"] == "retrieve" and x["ttft_ms"] is not None]
            s = [x for x in records if x["path"] == "skip" and x["ttft_ms"] is not None]
            if r and s:
                dt = np.median([x["ttft_ms"] for x in r]) - np.median([x["ttft_ms"] for x in s])
                dn = (np.median([x["prompt_chars"] for x in r]) - np.median([x["prompt_chars"] for x in s])) / p["chars_per_token"]
                if dn > 0:
                    slope = float(max(dt, 0.0) / dn)
                    slope_src = "logs_retrieve_vs_skip_medians"
        p["ttft_slope_ms_per_token"] = max(slope, 0.0)
        p["ttft_slope_source"] = slope_src

        def toks(rec):
            return rec["prompt_chars"] / p["chars_per_token"]

        p["ttft_residuals_ms"] = [
            float(x["ttft_ms"] - p["ttft_slope_ms_per_token"] * toks(x)) for x in records if x["ttft_ms"] is not None
        ]

        # overheads
        p["dispatch_overhead_ms"] = [float(x["dispatch_wait_ms"]) for x in records
                                     if x["path"] == "retrieve" and x["policy"] == "sentence_end"]
        p["pre_llm_overhead_ms"] = [float(x["pre_llm_overhead_ms"]) for x in records if x["path"] == "retrieve"]
        p["skip_overhead_ms"] = [float(x["skip_overhead_ms"]) for x in records
                                 if x["path"] == "skip" and x["skip_overhead_ms"] is not None]
        if not p["skip_overhead_ms"]:
            p["skip_overhead_ms"] = [d + o for d, o in zip(p["dispatch_overhead_ms"], p["pre_llm_overhead_ms"])] or [0.0]

        # generation records per path
        gen = defaultdict(list)
        for x in records:
            if x["per_token_ms"] is None or not x["n_tokens"]:
                continue
            gen[x["path"]].append(
                {
                    "n_tokens": int(x["n_tokens"]),
                    "per_token_ms": float(x["per_token_ms"]),
                    "first_clause_token_index": x["first_clause_token_index"],
                    "first_clause_words": int(x["first_clause_words"]),
                    "answer_words": None,
                }
            )
        p["generation"] = dict(gen)

        # logged v1 TTS stage (synthetic tone in E2/E3)
        p["tts_logged_v1_ms"] = [float(x["tts_stage_ms"]) for x in records if x["tts_mode"] == "batch"
                                 and x.get("tts_clause0") is None]

        # real piper synthesis: log(synth) = a + b log(words) + e
        tr = [x["tts_clause0"] for x in (tts_records or []) if x.get("tts_clause0") and x["tts_clause0"]["synth_ms"]]
        if len(tr) >= 5:
            w = np.log(np.maximum([t["words"] for t in tr], 1))
            y = np.log(np.maximum([t["synth_ms"] for t in tr], 1.0))
            b, a = np.polyfit(w, y, 1)
            p["tts_piper"] = {"a": float(a), "b": float(b), "residuals": (y - (a + b * w)).tolist(), "n": len(tr)}
            # playback start after synthesis: streaming/batch overhead beyond synth
        else:
            p["tts_piper"] = None

        # retrieval profiles
        p["retrieval_profiles"] = {}
        e2r = [float(x["retrieval_ms"]) for x in records if x["retrieval_ms"] is not None]
        if e2r:
            p["retrieval_profiles"]["pipeline_logged"] = {"samples_ms": e2r, "source": "launch->result in the fitted logs"}
        if profiles_path and Path(profiles_path).exists():
            art = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
            for name, prof in (art.get("profiles") or {}).items():
                if prof.get("samples_ms"):
                    p["retrieval_profiles"][name] = {
                        "samples_ms": prof["samples_ms"],
                        "source": f"{profiles_path}",
                        "wan_rtt_ms": prof.get("wan_rtt_ms"),
                    }
            for alias, name in (art.get("canonical") or {}).items():
                if name in p["retrieval_profiles"] and alias not in p["retrieval_profiles"]:
                    p["retrieval_profiles"][alias] = dict(p["retrieval_profiles"][name], alias_of=name)

        p["eot_samples_ms"] = [float(v) for v in (eot_samples_ms or [0.0])]
        p["default_tts_engine"] = "piper_resident" if p["tts_piper"] else "logged_v1"
        p["fit_sources"] = sorted({x.get("source", "?") for x in records})
        p["n_records"] = len(records)
        return cls(p)

    # -- persistence -------------------------------------------------------
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.p), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LatencyModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- sampling ----------------------------------------------------------
    @staticmethod
    def _draw(pool, rng):
        return pool[int(rng.integers(len(pool)))]

    def _tts_ms(self, words: int, engine: str, rng) -> float:
        if engine == "piper_resident":
            t = self.p.get("tts_piper")
            if not t:
                raise ValueError("model has no piper TTS fit; use tts_engine='logged_v1'")
            e = self._draw(t["residuals"], rng)
            return float(math.exp(t["a"] + t["b"] * math.log(max(words, 1)) + e))
        if engine == "logged_v1":
            return float(self._draw(self.p["tts_logged_v1_ms"] or [0.0], rng))
        if engine == "none":
            return 0.0
        raise ValueError(f"unknown tts_engine {engine!r}")

    def sample_ttfa(
        self,
        path: str,
        dispatch_ms: Optional[float],
        user_end_ms: float,
        profile: str,
        n_prompt_tokens: Optional[float],
        rng: np.random.Generator,
        tts_mode: str = "batch",
        tts_engine: Optional[str] = None,
        eot_ms: Optional[float] = None,
    ) -> dict:
        """
        One draw of the turn's latency components.

        path            "retrieve" | "skip"
        dispatch_ms     when retrieval is dispatched, on the same clock as
                        user_end_ms; None = at the end-of-turn decision
        user_end_ms     TRUE end of user speech (TTFA is measured from here)
        profile         retrieval latency profile name (see
                        self.p["retrieval_profiles"]; "pipeline_logged" is the
                        in-pipeline launch->result pool of the fitted logs)
        n_prompt_tokens prompt length for prefill; None -> estimate from the
                        path's typical prompt
        tts_mode        "batch" (TTS waits for the full answer) | "streaming"
                        (TTS starts on the first complete clause)
        """
        if path not in ("retrieve", "skip"):
            raise ValueError("path must be 'retrieve' or 'skip'")
        engine = tts_engine or self.p.get("default_tts_engine", "logged_v1")
        eot = float(eot_ms) if eot_ms is not None else float(self._draw(self.p["eot_samples_ms"], rng))
        t_eot = user_end_ms + eot

        retrieval_ms = 0.0
        exposed = 0.0
        if path == "retrieve":
            prof = self.p["retrieval_profiles"].get(profile)
            if prof is None:
                raise KeyError(f"unknown retrieval profile {profile!r}; have {sorted(self.p['retrieval_profiles'])}")
            retrieval_ms = float(self._draw(prof["samples_ms"], rng)) + float(prof.get("wan_rtt_ms") or 0.0)
            d = dispatch_ms if dispatch_ms is not None else t_eot + float(self._draw(self.p["dispatch_overhead_ms"] or [0.0], rng))
            r_done = d + retrieval_ms
            exposed = max(0.0, r_done - max(t_eot, d))
            llm_req = max(t_eot, r_done) + float(self._draw(self.p["pre_llm_overhead_ms"] or [0.0], rng))
        else:
            llm_req = t_eot + float(self._draw(self.p["skip_overhead_ms"], rng))

        if n_prompt_tokens is None:
            n_prompt_tokens = (_system_prompt_chars() + (60 if path == "skip" else 1500)) / self.p["chars_per_token"]
        ttft = max(1.0, float(self._draw(self.p["ttft_residuals_ms"], rng)) + self.p["ttft_slope_ms_per_token"] * n_prompt_tokens)
        first_tok = llm_req + ttft

        g = self._draw(self.p["generation"][path], rng)
        full_gen = (g["n_tokens"] - 1) * g["per_token_ms"]
        if tts_mode == "batch":
            gen_ms = full_gen
            # batch TTS synthesizes the first clause of the full answer
            words = g["first_clause_words"]
        elif tts_mode == "streaming":
            idx = g["first_clause_token_index"]
            gen_ms = full_gen if idx is None else idx * g["per_token_ms"]
            words = g["first_clause_words"]
        else:
            raise ValueError("tts_mode must be 'batch' or 'streaming'")
        tts = self._tts_ms(words, engine, rng)
        first_audio = first_tok + gen_ms + tts
        ttfa = first_audio - user_end_ms
        return {
            "ttfa_ms": ttfa,
            "eot_detect_ms": eot,
            "retrieval_ms": retrieval_ms,
            "retrieval_exposed_ms": exposed,
            "llm_request_ms_after_eot": llm_req - t_eot,
            "ttft_ms": ttft,
            "gen_to_tts_input_ms": gen_ms,
            "tts_first_audio_ms": tts,
            "n_prompt_tokens": n_prompt_tokens,
            "tts_mode": tts_mode,
            "tts_engine": engine,
        }

    def sample_many(self, n: int, rng, **kw) -> np.ndarray:
        return np.array([self.sample_ttfa(rng=rng, **kw)["ttfa_ms"] for _ in range(n)])

    # -- hideable / non-hideable decomposition ------------------------------
    def nonhideable_increment(
        self, tts_mode: str = "batch", tts_engine: Optional[str] = None, n: int = 4000, seed: int = 0,
        n_prompt_tokens_retrieve: Optional[float] = None, n_prompt_tokens_skip: Optional[float] = None,
    ) -> dict:
        """
        Model-based retrieve-minus-skip TTFA with the retrieval call fully
        hidden (dispatched long before speech end): the part of retrieval's
        cost that no timing policy can hide. Returns mean and quantiles of
        independent-draw differences (wider than paired differences).
        """
        rng = np.random.default_rng(seed)
        prof = next(iter(self.p["retrieval_profiles"]))
        diffs, means_r, means_s = [], [], []
        for _ in range(n):
            r = self.sample_ttfa("retrieve", -1e7, 0.0, prof, n_prompt_tokens_retrieve, rng, tts_mode, tts_engine, eot_ms=0.0)
            s = self.sample_ttfa("skip", None, 0.0, prof, n_prompt_tokens_skip, rng, tts_mode, tts_engine, eot_ms=0.0)
            diffs.append(r["ttfa_ms"] - s["ttfa_ms"])
        d = np.asarray(diffs)
        return {"mean_ms": float(d.mean()), "quantiles_ms": quantiles(d), "n_draws": n}

    def signals_retrieval(
        self, profile: str, tts_mode: str = "streaming", expected_quantile: str = "p50",
        measured_increments: Optional[dict] = None,
    ) -> dict:
        """
        signals["retrieval"] for a controller (CONTRACTS §6), extended with
        the non-hideable increment. expected_latency_ms / latency_quantiles_ms
        describe the HIDEABLE retrieval call; nonhideable_increment_ms the
        retrieve-minus-skip cost after the call, for this TTS mode (measured
        paired increments when given, else the model's estimate).
        """
        prof = self.p["retrieval_profiles"][profile]
        q = quantiles(prof["samples_ms"])
        inc = None
        src = "latency_model_v2"
        if measured_increments and tts_mode in measured_increments:
            inc = measured_increments[tts_mode]
            src = inc.get("source", "measured_paired")
        else:
            inc = self.nonhideable_increment(tts_mode)
        return {
            "latency_profile": profile,
            "expected_latency_ms": q[expected_quantile],
            "latency_quantiles_ms": q,
            "nonhideable_increment_ms": {
                "tts_mode": tts_mode,
                "mean": inc.get("mean_ms"),
                "quantiles": inc.get("quantiles_ms"),
                "source": src,
            },
        }


# ---------------------------------------------------------------------------
# Empirical paired increments (measured, not modelled)
# ---------------------------------------------------------------------------
def paired_increments(records: List[dict], key=("scenario", "seed", "policy", "tts_mode")) -> dict:
    """
    retrieve-minus-skip TTFA per matched (scenario, seed, policy, tts_mode)
    cell, plus its non-hideable part (subtracting the retrieval time exposed
    after end of turn and the dispatch wait). Grouped by (policy, tts_mode).
    """
    cells = defaultdict(dict)
    for r in records:
        cells[tuple(r.get(k) for k in key)][r["path"]] = r
    out = defaultdict(lambda: {"total": [], "nonhideable": []})
    for k, d in cells.items():
        if "retrieve" in d and "skip" in d:
            r, s = d["retrieve"], d["skip"]
            g = f"{k[2]}/{k[3]}"
            out[g]["total"].append(r["ttfa_ms"] - s["ttfa_ms"])
            out[g]["nonhideable"].append(
                r["ttfa_ms"] - r["retrieval_exposed_ms"] - r["dispatch_wait_ms"] - s["ttfa_ms"]
            )
    res = {}
    for g, v in sorted(out.items()):
        res[g] = {
            "n_pairs": len(v["total"]),
            "total_mean_ms": float(np.mean(v["total"])),
            "total_quantiles_ms": quantiles(v["total"]),
            "nonhideable_mean_ms": float(np.mean(v["nonhideable"])),
            "nonhideable_quantiles_ms": quantiles(v["nonhideable"]),
        }
    return res


# ---------------------------------------------------------------------------
# Fidelity checks
# ---------------------------------------------------------------------------
def _hash_frac(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 1000 / 1000.0


def split_by_scenario(records: List[dict], holdout_frac: float = 0.4, salt: str = "v2-latency"):
    fit, hold = [], []
    for r in records:
        (hold if _hash_frac(f"{salt}|{r['scenario']}") < holdout_frac else fit).append(r)
    return fit, hold


def ks_distance(a, b) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


def predict_record(model: LatencyModel, rec: dict, rng, draws: int, profile: str, tts_mode=None, tts_engine=None) -> np.ndarray:
    """Model draws for one observed run, using its actual dispatch time and
    prompt length (user_end at 0, eot 0 on the text path)."""
    dispatch = rec["launch_offset_ms"] if rec["path"] == "retrieve" else None
    if rec["path"] == "retrieve" and rec["policy"] == "sentence_end":
        dispatch = None  # dispatched at end of turn: sample the overhead
    n_tok = rec["prompt_chars"] / model.p["chars_per_token"]
    eot = -(rec.get("speech_end_offset_ms") or 0.0)  # user_end - speech end
    out = []
    for _ in range(draws):
        out.append(
            model.sample_ttfa(
                rec["path"], dispatch, 0.0, profile, n_tok, rng,
                tts_mode=tts_mode or rec["tts_mode"], tts_engine=tts_engine or "logged_v1", eot_ms=eot,
            )["ttfa_ms"]
        )
    return np.asarray(out)


def fidelity_distribution(model: LatencyModel, holdout: List[dict], profile="pipeline_logged", draws=40, seed=0,
                          group=("policy", "path", "tts_mode"), tts_engine="logged_v1") -> dict:
    rng = np.random.default_rng(seed)
    obs, pred = defaultdict(list), defaultdict(list)
    for r in holdout:
        g = "/".join(str(r[k]) for k in group)
        obs[g].append(r["ttfa_ms"])
        pred[g].extend(predict_record(model, r, rng, draws, profile, tts_engine=tts_engine).tolist())
    res = {}
    for g in sorted(obs):
        o, p = np.asarray(obs[g]), np.asarray(pred[g])
        res[g] = {
            "n_obs": int(o.size),
            "observed": {"p50": float(np.median(o)), "p95": float(np.quantile(o, 0.95)), "mean": float(o.mean())},
            "predicted": {"p50": float(np.median(p)), "p95": float(np.quantile(p, 0.95)), "mean": float(p.mean())},
            "ks_distance": ks_distance(o, p),
        }
    return res


def fidelity_increments(model: LatencyModel, holdout: List[dict], profile="pipeline_logged", draws=40, seed=0,
                        tts_engine="logged_v1") -> dict:
    """Observed vs predicted paired retrieve-minus-skip increments per policy."""
    rng = np.random.default_rng(seed)
    cells = defaultdict(dict)
    for r in holdout:
        cells[(r["scenario"], r["seed"], r["policy"], r["tts_mode"])][r["path"]] = r
    obs, pred_mean = defaultdict(list), defaultdict(list)
    for (sc, sd, pol, mode), d in cells.items():
        if "retrieve" not in d or "skip" not in d:
            continue
        g = f"{pol}/{mode}"
        obs[g].append(d["retrieve"]["ttfa_ms"] - d["skip"]["ttfa_ms"])
        pr = predict_record(model, d["retrieve"], rng, draws, profile, tts_engine=tts_engine)
        ps = predict_record(model, d["skip"], rng, draws, profile, tts_engine=tts_engine)
        pred_mean[g].append(float(pr.mean() - ps.mean()))
    res = {}
    for g in sorted(obs):
        o, p = np.asarray(obs[g]), np.asarray(pred_mean[g])
        res[g] = {
            "n_pairs": int(o.size),
            "observed_mean_ms": float(o.mean()),
            "observed_median_ms": float(np.median(o)),
            "predicted_mean_ms": float(p.mean()),
            "predicted_median_of_expected_ms": float(np.median(p)),
        }
    return res


def cv_folds(records: List[dict], k: int = 5, salt: str = "v2-latency-cv"):
    """Grouped K-fold by scenario: yields (fit, holdout) with every scenario
    held out exactly once."""
    scen = sorted({r["scenario"] for r in records}, key=lambda s: _hash_frac(f"{salt}|{s}"))
    fold = {s: i % k for i, s in enumerate(scen)}
    for f in range(k):
        yield [r for r in records if fold[r["scenario"]] != f], [r for r in records if fold[r["scenario"]] == f]


def cv_fidelity(records: List[dict], k: int = 5, draws: int = 40, seed: int = 0, fit_kwargs=None,
                tts_engine: str = "logged_v1", increments: bool = True) -> dict:
    """Pool out-of-fold predictions over all K folds (distributions and,
    when skip/retrieve pairs exist, paired increments)."""
    rng = np.random.default_rng(seed)
    obs, pred = defaultdict(list), defaultdict(list)
    inc_obs, inc_pred = defaultdict(list), defaultdict(list)
    for fit, hold in cv_folds(records, k):
        m = LatencyModel.fit(fit, **(fit_kwargs(fit) if callable(fit_kwargs) else (fit_kwargs or {})))
        for r in hold:
            g = f"{r['policy']}/{r['path']}/{r['tts_mode']}"
            obs[g].append(r["ttfa_ms"])
            pred[g].extend(predict_record(m, r, rng, draws, "pipeline_logged", tts_engine=tts_engine).tolist())
        if increments:
            cells = defaultdict(dict)
            for r in hold:
                cells[(r["scenario"], r["seed"], r["policy"], r["tts_mode"])][r["path"]] = r
            for (sc, sd, pol, mode), d in cells.items():
                if "retrieve" in d and "skip" in d:
                    g = f"{pol}/{mode}"
                    inc_obs[g].append(d["retrieve"]["ttfa_ms"] - d["skip"]["ttfa_ms"])
                    pr = predict_record(m, d["retrieve"], rng, draws, "pipeline_logged", tts_engine=tts_engine)
                    ps = predict_record(m, d["skip"], rng, draws, "pipeline_logged", tts_engine=tts_engine)
                    inc_pred[g].append(float(pr.mean() - ps.mean()))
    dist = {}
    for g in sorted(obs):
        o, p = np.asarray(obs[g]), np.asarray(pred[g])
        dist[g] = {
            "n_obs": int(o.size),
            "observed": {"p50": float(np.median(o)), "p95": float(np.quantile(o, 0.95)), "mean": float(o.mean())},
            "predicted": {"p50": float(np.median(p)), "p95": float(np.quantile(p, 0.95)), "mean": float(p.mean())},
            "ks_distance": ks_distance(o, p),
        }
    inc = {}
    for g in sorted(inc_obs):
        o, p = np.asarray(inc_obs[g]), np.asarray(inc_pred[g])
        inc[g] = {
            "n_pairs": int(o.size),
            "observed_mean_ms": float(o.mean()),
            "observed_median_ms": float(np.median(o)),
            "predicted_mean_ms": float(p.mean()),
            "predicted_median_of_expected_ms": float(np.median(p)),
        }
    return {"k_folds": k, "by_group": dist, "increments": inc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt_table(fid: dict) -> str:
    lines = ["| group | n | obs p50 | pred p50 | obs p95 | pred p95 | obs mean | pred mean | KS |", "|---|---|---|---|---|---|---|---|---|"]
    for g, r in fid.items():
        lines.append(
            f"| {g} | {r['n_obs']} | {r['observed']['p50']:.0f} | {r['predicted']['p50']:.0f} | "
            f"{r['observed']['p95']:.0f} | {r['predicted']['p95']:.0f} | {r['observed']['mean']:.0f} | "
            f"{r['predicted']['mean']:.0f} | {r['ks_distance']:.3f} |"
        )
    return "\n".join(lines)


def _fmt_inc(inc: dict) -> str:
    lines = ["| policy/tts | pairs | obs mean | pred mean | obs median | pred median (of expectations) |", "|---|---|---|---|---|---|"]
    for g, r in inc.items():
        lines.append(
            f"| {g} | {r['n_pairs']} | {r['observed_mean_ms']:.0f} | {r['predicted_mean_ms']:.0f} | "
            f"{r['observed_median_ms']:.0f} | {r['predicted_median_of_expected_ms']:.0f} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit latency_model_v2 and report fidelity")
    ap.add_argument("--e2", nargs="*", default=["logs/conv_*.jsonl"])
    ap.add_argument("--e3", nargs="*", default=[])
    ap.add_argument("--v2", nargs="*", default=[], help="v2 real-endpoint logs (streaming + piper)")
    ap.add_argument("--prefill", default=None, help="measure_llm_tts.json with a prefill sweep")
    ap.add_argument("--profiles", default="retrieval/latency_profiles_v2.json")
    ap.add_argument("--holdout-frac", type=float, default=0.4)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-dir", default="analysis/latency_model_v2")
    args = ap.parse_args(argv)

    e2 = load_records(args.e2, "e2")
    e3 = load_records(args.e3, "e3")
    v2 = load_records(args.v2, "v2")
    prefill = None
    if args.prefill and Path(args.prefill).exists():
        prefill = json.loads(Path(args.prefill).read_text(encoding="utf-8")).get("prefill") or None

    report: dict = {"n_records": {"e2": len(e2), "e3": len(e3), "v2": len(v2)}, "holdout_frac": args.holdout_frac}
    md = []

    # 1) E2: single 60/40 scenario hold-out (as specified) + 5-fold grouped CV
    if e2:
        fit, hold = split_by_scenario(e2, args.holdout_frac)
        m = LatencyModel.fit(fit, prefill_sweep=None)
        fid = fidelity_distribution(m, hold, draws=args.draws)
        cv2 = cv_fidelity(e2, k=args.folds, draws=args.draws, increments=False)
        report["e2_holdout"] = {"n_fit": len(fit), "n_holdout": len(hold), "by_group": fid}
        report["e2_cv"] = cv2
        md += ["## E2 hold-out (fit on 60% of scenarios, predict TTFA of the other 40%)", "", _fmt_table(fid), "",
               f"## E2 {args.folds}-fold grouped CV (every scenario predicted out-of-fold)", "",
               _fmt_table(cv2["by_group"]), ""]

    # 2) E3: paired retrieve-minus-skip increments, out-of-fold
    if e3:
        report["e3_observed_paired_increments_all"] = paired_increments(e3)
        cv3 = cv_fidelity(e3, k=args.folds, draws=args.draws)
        report["e3_cv"] = cv3
        md += [f"## E3 {args.folds}-fold grouped CV: paired retrieve-minus-skip TTFA increment", "",
               _fmt_inc(cv3["increments"]), "",
               f"## E3 {args.folds}-fold grouped CV: TTFA distributions", "", _fmt_table(cv3["by_group"]), ""]

    # 3) v2 real-endpoint runs (streaming and batch TTS, real piper)
    if v2:
        report["v2_observed_paired_increments"] = paired_increments(v2)
        cvv = cv_fidelity(
            v2, k=args.folds, draws=args.draws, tts_engine="piper_resident",
            fit_kwargs=lambda fit: {"prefill_sweep": prefill, "tts_records": fit},
        )
        report["v2_cv"] = cvv
        md += [f"## v2 real-endpoint runs, {args.folds}-fold grouped CV (real piper TTS)", "",
               _fmt_table(cvv["by_group"]), "", _fmt_inc(cvv["increments"]), ""]
        chk = [(r["observed_first_clause_ms"], r["first_clause_token_index"], r["per_token_ms"]) for r in v2
               if r["observed_first_clause_ms"] is not None and r["per_token_ms"]]
        if chk:
            err = [abs(i * pt - o) for o, i, pt in chk if i is not None]
            report["first_clause_reconstruction"] = {
                "n": len(err),
                "mae_ms": float(np.mean(err)) if err else None,
                "note": "token-index x per-token-time reconstruction of time-to-first-clause vs the logged llm_first_clause",
            }

    # Final model: everything measured, for downstream simulation
    all_recs = e2 + e3 + v2
    if all_recs:
        final_records = v2 or (e3 + e2)
        final = LatencyModel.fit(final_records, profiles_path=args.profiles, prefill_sweep=prefill,
                                 tts_records=v2 or None)
        final.p["fit_note"] = ("LLM/TTS pools from v2 real-endpoint logs when present (current endpoint epoch), "
                               "else E3+E2; retrieval profiles from the measured artifact")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final.save(out_dir / "latency_model_v2_fit.json")
        measured = {}
        src = v2 or e3
        if src:
            pi = paired_increments(src)
            for g, v in pi.items():
                pol, mode = g.split("/")
                if pol == "sentence_end":
                    measured[mode] = {"mean_ms": v["nonhideable_mean_ms"], "quantiles_ms": v["nonhideable_quantiles_ms"],
                                      "source": f"measured paired retrieve-skip, {'v2' if v2 else 'e3'} logs, {v['n_pairs']} pairs"}
        sig = {}
        for prof in final.p["retrieval_profiles"]:
            for mode in ("batch", "streaming"):
                try:
                    sig[f"{prof}/{mode}"] = final.signals_retrieval(prof, mode, measured_increments=measured)
                except Exception as exc:  # pragma: no cover - reported, not fatal
                    sig[f"{prof}/{mode}"] = {"error": str(exc)}
        (out_dir / "signals_retrieval_v2.json").write_text(json.dumps(sig, indent=1), encoding="utf-8")
        report["model_nonhideable_increment"] = {
            mode: final.nonhideable_increment(mode, tts_engine=final.p["default_tts_engine"]) for mode in ("batch", "streaming")
        }
        report["final_model"] = {"path": str(out_dir / "latency_model_v2_fit.json"),
                                 "ttft_slope_ms_per_token": final.p["ttft_slope_ms_per_token"],
                                 "ttft_slope_source": final.p["ttft_slope_source"],
                                 "chars_per_token": final.p["chars_per_token"],
                                 "default_tts_engine": final.p["default_tts_engine"],
                                 "retrieval_profiles": sorted(final.p["retrieval_profiles"])}
        out = out_dir / "fidelity_report.json"
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        (out_dir / "fidelity_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
