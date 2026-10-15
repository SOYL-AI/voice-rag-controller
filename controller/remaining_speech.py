"""
remaining_speech.py — causal predictor of remaining user speech R_t.

At a partial transcript (text, elapsed_ms since speech onset) predict how
many milliseconds of speech remain before the user stops talking
(speech end = the end of the last word, not the VAD endpoint). Output is a
small set of quantiles so a controller can be as conservative as it wants.
It feeds signals["timing"]["predicted_remaining_speech_ms"] (CONTRACTS §6)
and the `forecast` timing policy (controller/timing.py).

Causal by construction: features only use the partial text and the elapsed
time at that moment. Punctuation and case are stripped before featurizing,
because streaming ASR partials carry unreliable punctuation (Whisper
often closes a partial with "." mid-utterance).

Models (all fitted on DEV traces only; see fit()):

    prior      survival prior on total duration D from the fit set:
               R = quantile_q(D - t | D > t). Uses elapsed time only.
    rate_words survival prior on total word count N given n words heard,
               times the measured speaking rate (elapsed / n). Uses the
               word count and the user's own rate, no lexical cues.
    qr         linear quantile regression (one model per quantile) on the
               rate_words estimate plus lexical/syntactic completeness
               features: last word is a function word / determiner /
               preposition / auxiliary / wh-word, question opener type
               (wh-, auxiliary-, imperative "tell me"), content-word count,
               and a verb-less-object heuristic. Features are scaled by the
               measured per-word time so the model is rate-invariant.

External options considered (see docs/research_v2/timing_components.md):
LiveKit's turn detector cannot be used here (its licence restricts use to
the LiveKit Agents framework and forbids using its outputs to develop other
models); VAP needs two-channel dialogue audio and predicts a 2 s voice
activity window, not remaining utterance time. Pipecat Smart Turn v3
(BSD-2, audio) is a candidate EOU feature once real audio traces exist.

API:
    p = RemainingSpeechPredictor.fit(traces, model="qr")
    p.predict("what time does the", elapsed_ms=900) -> {"q10":..,"q50":..,"q90":..}
    p.signal_entries(partials) -> [{"t_ms", "remaining_ms", "remaining_q10_ms", "remaining_q90_ms"}]
    p.save(path) / RemainingSpeechPredictor.load(path)

    python -m controller.remaining_speech evaluate --traces "simulator/v1_text_traces/traces/*.jsonl" \\
        --items simulator/v1_text_traces/items.jsonl --split dev --out analysis/remaining_speech_v1_textsim.json
"""

from __future__ import annotations

import argparse
import glob
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

QUANTILES = (0.1, 0.5, 0.9)
MODELS = ("prior", "rate_words", "qr")

_PUNCT = re.compile(r"[^\w\s'-]")

WH = {"what", "when", "where", "who", "whom", "whose", "why", "how", "which"}
AUX = {
    "is", "are", "was", "were", "am", "do", "does", "did", "can", "could",
    "would", "should", "will", "shall", "may", "might", "have", "has", "had",
}
DET = {"the", "a", "an", "my", "your", "our", "their", "his", "her", "its",
       "this", "that", "these", "those", "any", "some", "each", "every", "no"}
PREP = {"to", "of", "in", "on", "at", "for", "with", "about", "from", "by",
        "into", "during", "before", "after", "near", "between", "than", "per", "as"}
CONJ = {"and", "or", "but", "if", "because", "so", "while", "whether"}
PRON = {"i", "me", "you", "we", "us", "it", "they", "them", "he", "she", "there"}
FILLERS = {"um", "uh", "erm", "like"}
_FUNCTION = WH | AUX | DET | PREP | CONJ | PRON | FILLERS | {"please", "tell", "not"}


def normalize(text: str) -> List[str]:
    return _PUNCT.sub(" ", (text or "").lower()).split()


def lexical_features(words: Sequence[str]) -> Dict[str, float]:
    n = len(words)
    last = words[-1] if words else ""
    first = words[0] if words else ""
    content = [w for w in words if w not in _FUNCTION]
    return {
        "last_det": float(last in DET),
        "last_prep": float(last in PREP),
        "last_aux": float(last in AUX),
        "last_wh": float(last in WH),
        "last_conj": float(last in CONJ),
        "last_pron": float(last in PRON),
        "last_content": float(bool(last) and last not in _FUNCTION),
        "open_wh": float(first in WH),
        "open_aux": float(first in AUX),
        "open_tell": float(first in {"tell", "please", "explain", "describe", "give", "show", "list"}),
        "n_content": float(len(content)),
        "no_content_yet": float(len(content) == 0),
        "n_words": float(n),
    }


# ---------------------------------------------------------------------------
# Training samples from traces
# ---------------------------------------------------------------------------
def speech_end_ms(trace: dict) -> float:
    words = trace.get("words") or []
    if words:
        return float(words[-1]["end_ms"])
    return float(trace.get("duration_ms") or trace.get("vad_user_end_ms") or 0)


def samples_from_trace(trace: dict, include_after_end: bool = False) -> List[dict]:
    end = speech_end_ms(trace)
    out = []
    for p in trace.get("partials") or []:
        t = float(p["t_ms"])
        if t > end and not include_after_end:
            continue
        words = normalize(p.get("text", ""))
        if not words:
            continue
        out.append(
            {
                "item_id": trace.get("item_id"),
                "voice_id": trace.get("voice_id"),
                "t_ms": t,
                "words": words,
                "n_words": len(words),
                "remaining_ms": max(0.0, end - t),
                "total_ms": end,
                "total_words": len(normalize((trace.get("asr_final") or {}).get("text", "")))
                or len(trace.get("words") or []),
            }
        )
    return out


def pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# ---------------------------------------------------------------------------
# The predictor
# ---------------------------------------------------------------------------
class RemainingSpeechPredictor:
    def __init__(self, model: str, quantiles: Sequence[float], state: dict):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.model = model
        self.quantiles = tuple(float(q) for q in quantiles)
        self.state = state

    # -- priors ------------------------------------------------------------
    @staticmethod
    def _prior_state(samples_by_trace: List[dict]) -> dict:
        durations = sorted(t["total_ms"] for t in samples_by_trace)
        words = sorted(t["total_words"] for t in samples_by_trace)
        return {"durations_ms": durations, "total_words": words}

    def _cond_quantile(self, values: Sequence[float], given: float, q: float) -> float:
        """quantile_q(V - given | V > given), falling back to the tail."""
        arr = np.asarray(values, dtype=float)
        rest = arr[arr > given] - given
        if rest.size == 0:
            return 0.0
        return float(np.quantile(rest, q))

    def _rate(self, n_words: int, elapsed_ms: float) -> float:
        prior_rate = self.state.get("prior_ms_per_word", 400.0)
        if n_words <= 0 or elapsed_ms <= 0:
            return prior_rate
        # Shrink the user's measured rate toward the fit-set rate for very
        # short prefixes, where elapsed/n is dominated by onset effects.
        k = 2.0
        own = elapsed_ms / n_words
        return (n_words * own + k * prior_rate) / (n_words + k)

    def _rate_words(self, n_words: int, elapsed_ms: float, q: float) -> float:
        rem_words = self._cond_quantile(self.state["prior"]["total_words"], n_words - 0.5, q)
        # remaining words strictly beyond the ones heard
        rem_words = max(0.0, rem_words - 0.5)
        return rem_words * self._rate(n_words, elapsed_ms)

    def _design(self, words: Sequence[str], elapsed_ms: float) -> np.ndarray:
        n = len(words)
        rate = self._rate(n, elapsed_ms)
        lex = lexical_features(words)
        base = self._rate_words(n, elapsed_ms, 0.5)
        cols = [
            base,
            rate,
            rate * lex["last_det"],
            rate * lex["last_prep"],
            rate * lex["last_aux"],
            rate * lex["last_wh"],
            rate * lex["last_conj"],
            rate * lex["last_pron"],
            rate * lex["open_wh"],
            rate * lex["open_aux"],
            rate * lex["open_tell"],
            rate * lex["no_content_yet"],
            rate * min(lex["n_content"], 6.0),
        ]
        return np.asarray(cols, dtype=float)

    FEATURES = (
        "rate_words_q50", "ms_per_word", "rate*last_det", "rate*last_prep",
        "rate*last_aux", "rate*last_wh", "rate*last_conj", "rate*last_pron",
        "rate*open_wh", "rate*open_aux", "rate*open_tell", "rate*no_content_yet",
        "rate*n_content",
    )

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        traces: Iterable[dict],
        model: str = "qr",
        quantiles: Sequence[float] = QUANTILES,
        alpha: float = 0.0,
    ) -> "RemainingSpeechPredictor":
        """
        Fit on the given traces. The CALLER is responsible for passing DEV
        traces only (the CLI enforces it via harness/v2_split.py).
        """
        traces = list(traces)
        per_trace = []
        samples = []
        for tr in traces:
            s = samples_from_trace(tr)
            if not s:
                continue
            samples.extend(s)
            per_trace.append({"total_ms": s[0]["total_ms"], "total_words": s[0]["total_words"]})
        if not samples:
            raise ValueError("no usable partials in the given traces")
        rates = [t["total_ms"] / max(t["total_words"], 1) for t in per_trace]
        state = {
            "prior": cls._prior_state(per_trace),
            "prior_ms_per_word": float(np.median(rates)),
            "n_traces": len(per_trace),
            "n_samples": len(samples),
        }
        self = cls(model, quantiles, state)
        if model == "qr":
            from sklearn.linear_model import QuantileRegressor

            X = np.vstack([self._design(s["words"], s["t_ms"]) for s in samples])
            y = np.asarray([s["remaining_ms"] for s in samples], dtype=float)
            coefs = {}
            for q in self.quantiles:
                reg = QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
                reg.fit(X, y)
                coefs[str(q)] = {"coef": reg.coef_.tolist(), "intercept": float(reg.intercept_)}
            self.state["qr"] = {"features": list(cls.FEATURES), "coefs": coefs, "alpha": alpha}
        return self

    # -- prediction --------------------------------------------------------
    def predict(self, text: str, elapsed_ms: float) -> Dict[str, float]:
        words = normalize(text) if isinstance(text, str) else list(text)
        n = len(words)
        out = {}
        for q in self.quantiles:
            if self.model == "prior":
                v = self._cond_quantile(self.state["prior"]["durations_ms"], elapsed_ms, q)
            elif self.model == "rate_words":
                v = self._rate_words(n, elapsed_ms, q)
            else:
                c = self.state["qr"]["coefs"][str(q)]
                v = float(np.dot(self._design(words, elapsed_ms), c["coef"]) + c["intercept"])
            out[f"q{int(round(q * 100)):02d}"] = max(0.0, v)
        # enforce monotone quantiles
        keys = sorted(out, key=lambda k: int(k[1:]))
        vals = np.maximum.accumulate([out[k] for k in keys])
        return {k: float(v) for k, v in zip(keys, vals)}

    def predict_ms(self, text: str, elapsed_ms: float, quantile: str = "q50") -> float:
        return self.predict(text, elapsed_ms)[quantile]

    def signal_entries(self, partials: Iterable[dict], onset_ms: float = 0.0) -> List[dict]:
        """CONTRACTS §6 signals["timing"]["predicted_remaining_speech_ms"]."""
        out = []
        for p in partials:
            t = float(p["t_ms"])
            pred = self.predict(p.get("text", ""), t - onset_ms)
            entry = {"t_ms": t, "remaining_ms": pred.get("q50")}
            for k, v in pred.items():
                if k != "q50":
                    entry[f"remaining_{k}_ms"] = v
            out.append(entry)
        return out

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"model": self.model, "quantiles": list(self.quantiles), "state": self.state}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "RemainingSpeechPredictor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["model"], d["quantiles"], d["state"])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(pred: RemainingSpeechPredictor, traces: Iterable[dict]) -> dict:
    ys, preds = [], defaultdict(list)
    bins = defaultdict(lambda: {"y": [], "q50": []})
    for tr in traces:
        for s in samples_from_trace(tr):
            p = pred.predict(s["words"], s["t_ms"])
            ys.append(s["remaining_ms"])
            for k, v in p.items():
                preds[k].append(v)
            frac = s["t_ms"] / max(s["total_ms"], 1.0)
            b = min(int(frac * 4), 3)
            bins[b]["y"].append(s["remaining_ms"])
            bins[b]["q50"].append(p["q50"])
    y = np.asarray(ys)
    res = {"n_samples": int(len(y))}
    if "q50" in preds:
        res["mae_q50_ms"] = float(np.mean(np.abs(y - np.asarray(preds["q50"]))))
        res["median_ae_q50_ms"] = float(np.median(np.abs(y - np.asarray(preds["q50"]))))
    for k, v in preds.items():
        q = int(k[1:]) / 100.0
        res[f"pinball_{k}"] = pinball(y, np.asarray(v), q)
    if "q10" in preds and "q90" in preds:
        lo, hi = np.asarray(preds["q10"]), np.asarray(preds["q90"])
        res["coverage_q10_q90"] = float(np.mean((y >= lo) & (y <= hi)))
        res["frac_below_q10"] = float(np.mean(y < lo))
        res["frac_above_q90"] = float(np.mean(y > hi))
    res["mae_q50_by_time_fraction"] = {
        f"{b * 25}-{b * 25 + 25}%": {
            "n": len(v["y"]),
            "mae_ms": float(np.mean(np.abs(np.asarray(v["y"]) - np.asarray(v["q50"])))),
            "mean_remaining_ms": float(np.mean(v["y"])),
        }
        for b, v in sorted(bins.items())
    }
    return res


def grouped_cv(traces: List[dict], model: str, n_folds: int = 5, seed: int = 0) -> dict:
    """K-fold CV grouped by item_id (all voices of an item share a fold)."""
    items = sorted({t["item_id"] for t in traces})
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    fold_of = {it: i % n_folds for i, it in enumerate(items)}
    pooled = defaultdict(list)
    per_fold = []
    for f in range(n_folds):
        train = [t for t in traces if fold_of[t["item_id"]] != f]
        test = [t for t in traces if fold_of[t["item_id"]] == f]
        p = RemainingSpeechPredictor.fit(train, model=model)
        r = evaluate(p, test)
        per_fold.append(r)
    # sample-weighted average of the scalar metrics
    total = sum(r["n_samples"] for r in per_fold)
    out = {"n_folds": n_folds, "n_samples": total}
    for key in per_fold[0]:
        if key in ("n_samples", "mae_q50_by_time_fraction"):
            continue
        out[key] = sum(r[key] * r["n_samples"] for r in per_fold) / total
    bins = defaultdict(lambda: {"n": 0, "abs": 0.0, "rem": 0.0})
    for r in per_fold:
        for b, v in r["mae_q50_by_time_fraction"].items():
            bins[b]["n"] += v["n"]
            bins[b]["abs"] += v["mae_ms"] * v["n"]
            bins[b]["rem"] += v["mean_remaining_ms"] * v["n"]
    out["mae_q50_by_time_fraction"] = {
        b: {"n": v["n"], "mae_ms": v["abs"] / v["n"], "mean_remaining_ms": v["rem"] / v["n"]}
        for b, v in sorted(bins.items())
    }
    return out


def load_traces(patterns: Iterable[str]) -> List[dict]:
    out = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, "r", encoding="utf-8") as fh:
                out.extend(json.loads(l) for l in fh if l.strip())
    return out


def _dev_traces(traces: List[dict], items_path: Optional[str], split: str) -> List[dict]:
    from harness.v2_split import check_split, filter_split, item_split, load_jsonl

    check_split(split)
    if items_path:
        keep = {it.get("id") for it in filter_split(load_jsonl(items_path), split)}
    else:
        keep = {t["item_id"] for t in traces if item_split(t["item_id"]) == split}
    return [t for t in traces if t["item_id"] in keep]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Remaining-speech predictor: fit / evaluate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--traces", nargs="+", required=True)
        sp.add_argument("--items", default=None, help="items.jsonl carrying the split field")
        sp.add_argument("--split", default="dev", choices=["dev", "test"])
        sp.add_argument("--all-items-are-dev", action="store_true",
                        help="development data with no v2 split (e.g. the simulated v1 traces)")
        sp.add_argument("--out", required=True)
        sp.add_argument("--model", default="qr", choices=MODELS)
        sp.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    traces = load_traces(args.traces)
    if not args.all_items_are_dev:
        traces = _dev_traces(traces, args.items, args.split)
    if args.cmd == "fit":
        p = RemainingSpeechPredictor.fit(traces, model=args.model)
        p.state["fit_traces"] = args.traces
        p.save(args.out)
        print(f"saved {args.model} predictor fitted on {p.state['n_traces']} traces to {args.out}")
        return 0

    report = {
        "traces": args.traces,
        "n_traces": len(traces),
        "split": "all-dev (development data)" if args.all_items_are_dev else args.split,
        "cv": "grouped by item_id",
        "models": {m: grouped_cv(traces, m, n_folds=args.folds) for m in MODELS},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'model':<11} {'MAE q50':>9} {'pin q10':>8} {'pin q50':>8} {'pin q90':>8} {'cov80':>6}")
    for m, r in report["models"].items():
        print(
            f"{m:<11} {r['mae_q50_ms']:9.0f} {r['pinball_q10']:8.1f} {r['pinball_q50']:8.1f} "
            f"{r['pinball_q90']:8.1f} {r['coverage_q10_q90']:6.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
