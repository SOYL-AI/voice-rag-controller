"""
fit_belief_transitions.py — fit ours_v2's belief-transition (lookahead)
model on DEV E1-v2 signal records (CONTRACTS.md §5).

    python -m controller.fit_belief_transitions --split dev \
        --records llm/experiment1_v2/results/*.jsonl \
        --items datasets/v2/d2_support/items.jsonl \
        --out controller/baselines/belief_transitions_v2.json

One belief SEQUENCE per (item_id, model, draw_seed, prefix source/voice):
the record's uncertainty signal on each prefix, mapped through the shared
v2 calibration (_objective_v2.load_belief_calibration), ordered by prefix
t_ms (or word_frac when t_ms is absent), ending at the full-transcript
record. The model is the binned empirical conditional distribution of the
next prefix's belief (and of the end-of-turn belief) given the current
belief and phase — see _objective_v2.fit_belief_transitions.

Fit-time phase uses the record's word_frac (the reference-transcript
fraction, available in hindsight on dev); at decision time ours_v2 uses the
causal estimate e/(e+R). The calibration's own frac feature, if any, is
applied with word_frac here for the same reason.

Only dev items are read: --split test is refused unless
docs/research_v2/PREREG_FROZEN exists, and fitting on test would be a
protocol violation even then, so it additionally requires --i-know.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import (  # noqa: E402
    BeliefCalibration,
    fit_belief_transitions,
    load_belief_calibration,
    load_e3v2_config,
)

PREREG_FROZEN = _REPO_ROOT / "docs" / "research_v2" / "PREREG_FROZEN"
# Project-wide decision: Mistral-7B-Instruct-v0.3 is the only model; the
# transition model must describe that model's beliefs and nothing else.
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def belief_sequences(records: Iterable[dict], calibration: BeliefCalibration,
                     keep: Optional[set] = None,
                     model: Optional[str] = None) -> List[List[Tuple[float, float]]]:
    groups: Dict[Tuple, List[dict]] = {}
    for rec in records:
        if keep is not None and str(rec["item_id"]) not in keep:
            continue
        if model is not None and rec.get("model") != model:
            continue
        prefix = rec.get("prefix") or {}
        key = (str(rec["item_id"]), rec.get("model"), rec.get("draw_seed"),
               prefix.get("source"), prefix.get("voice_id"))
        groups.setdefault(key, []).append(rec)
    sequences = []
    for recs in groups.values():
        def order(r):
            p = r.get("prefix") or {}
            full = 1 if p.get("kind") == "full" else 0
            return (full, float(p.get("t_ms") or 0.0), float(p.get("word_frac") or 0.0))
        seq = []
        for r in sorted(recs, key=order):
            x = (r.get("signals") or {}).get(calibration.signal)
            if x is None:
                continue
            p = r.get("prefix") or {}
            frac = 1.0 if p.get("kind") == "full" else float(p.get("word_frac", 0.0))
            seq.append((frac, calibration.apply(float(x), frac)))
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--records", nargs="+", required=True, help="E1-v2 JSONL files or globs")
    ap.add_argument("--items", type=Path, help="items.jsonl (split field) for dev filtering")
    ap.add_argument("--splits", type=Path, help="splits.json for dev filtering")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--phase-edges", default="0,0.34,0.67,1.0")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="only records from this model are used")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "controller" / "baselines" / "belief_transitions_v2.json")
    ap.add_argument("--i-know", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.split not in ("dev", "test"):
        raise SystemExit("--split must be dev or test")
    if args.split == "test" and (not PREREG_FROZEN.exists() or not args.i_know):
        raise SystemExit("Refusing to fit on the test split (CONTRACTS.md §7).")

    keep = None
    if args.splits:
        with open(args.splits, "r", encoding="utf-8") as fh:
            keep = set(json.load(fh)[args.split])
    elif args.items:
        with open(args.items, "r", encoding="utf-8") as fh:
            keep = {str(json.loads(l)["id"]) for l in fh if l.strip() and json.loads(l).get("split") == args.split}
    else:
        raise SystemExit("--items or --splits is required so test items can be excluded")

    files = sorted({Path(p) for pattern in args.records for p in glob.glob(pattern)})
    if not files:
        raise SystemExit("no E1-v2 record files matched --records")
    records = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())

    cfg = load_e3v2_config()
    calibration = load_belief_calibration(cfg)
    seqs = belief_sequences(records, calibration, keep, args.model)
    if not seqs:
        raise SystemExit("no usable belief sequences (need >= 2 prefixes per item/draw)")
    model = fit_belief_transitions(
        seqs, n_bins=args.bins, phase_edges=[float(x) for x in args.phase_edges.split(",")],
        alpha=args.alpha,
        provenance={
            "kind": "fitted",
            "generated_by": "python -m controller.fit_belief_transitions",
            "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "n_sequences": len(seqs),
            "n_records": len(records),
            "model": args.model,
            "calibration": calibration.source,
            "signal": calibration.signal,
            "alpha": args.alpha,
            "sources": [{"path": str(f), "sha256": _sha256(f)} for f in files],
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model.to_json(), fh, indent=1)
    print(f"wrote {args.out} ({len(seqs)} sequences)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
