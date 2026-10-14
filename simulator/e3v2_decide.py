"""
e3v2_decide.py — Phase B of the v2 sweep: run every v2 controller causally
over every turn and look each decision's outcome up in the Phase A cell
store. No model, ASR or judge call happens here.

    python -m simulator.e3v2_decide --split dev \
        --signals <signals.jsonl> --cells <phase_a_cells.jsonl> \
        --items datasets/v2/d2_support/items.jsonl \
        --labels <e1v2_labels.json> --profiles faiss,remote_api \
        --out simulator/e3v2_results

---------------------------------------------------------------------------
WHY ANSWERS ARE KEYED BY query_variant
---------------------------------------------------------------------------
v1's Phase A keyed answers by (scenario, seed, path) because every policy
retrieved on one frozen query. In v2 the query is the transcript prefix at
dispatch, so the answer depends on timing — but ONLY through the passages
that query retrieves. So Phase A records, per (item, condition), what each
dispatch point retrieves:

    {"kind": "retrieval", "item_id", "condition",
     "partial_index": k | null (null = final transcript),
     "retrieved_doc_ids": [...], "query_variant"?: "...", "query_text"?: "..."}

and generates ONE answer per unique passage list (query_variant =
query_variant_for(retrieved_doc_ids)):

    {"kind": "cell", "item_id", "seed", "condition",
     "path": "skip" | "retrieve", "query_variant": "none" | "qv-...",
     "answer_correct": bool, "retrieved_doc_ids": [...],
     "n_prompt_tokens"?, "gen_ms"?, "ttft_ms"?}

Two dispatch points whose queries retrieve the same passages share a cell,
and two controllers making the same decision are scored on the same
measured answer (v1's common-outcome principle, kept).

query_variant is ORDER-SENSITIVE by default (the prompt lists passages in
rank order, so a reordering is a different prompt); set
controller.e3v2.phase_b.query_variant_order_sensitive=false to key on the
set instead.

---------------------------------------------------------------------------
TTFA
---------------------------------------------------------------------------
From harness/latency_model_v2.LatencyModel.sample_ttfa(path, dispatch_ms,
user_end_ms, profile, n_prompt_tokens, rng) when that module exists, else
from harness/latency_model_v2_stub.py (records then carry
ttfa_components.stub = true). The rng is seeded per (item, seed, condition,
profile), so controllers making the same decision get the same latency draw.

---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------
<out>/e3_v2_<architecture>_<controller>_seed<seed>.json (matching analysis/e3_frontier.py's e3_*.json glob), lists of records compatible with
analysis/e3_frontier.py's validate_records (required keys present) plus the
v2 fields checked by its validate_v2_records: dispatch_t_ms, condition,
latency_profile, exposed_latency_ms, architecture.

ARCHITECTURE (controller.e3v2.architecture, --architectures): under
"commit" a dispatched retrieval is always used; under "speculative" a
controller may discard it at end of turn (b10_speculative_v2). A discarded
dispatch is scored on the skip cell and skip-path TTFA with its compute
counted (retrievals = 1, used = false). Results are written and analysed
per architecture.

SEALING: --split test is refused unless docs/research_v2/PREREG_FROZEN exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import latency_dist_from_quantiles, load_e3v2_config  # noqa: E402
from controller.sequential_v2 import (  # noqa: E402
    PRICED_V2,
    SPECULATIVE_ONLY_V2,
    SequentialController,
    TurnCtx,
    get_controller_v2,
    list_controllers_v2,
    user_end_ms,
)

PREREG_FROZEN = _REPO_ROOT / "docs" / "research_v2" / "PREREG_FROZEN"
DEFAULT_SEEDS = (11, 23, 37, 53, 71)


def load_latency_model(cfg: dict):
    """The real latency model if its branch has landed, else the stub."""
    try:
        from harness.latency_model_v2 import LatencyModel  # type: ignore
        try:
            return LatencyModel(cfg=cfg)
        except TypeError:
            return LatencyModel()
    except ImportError:
        from harness.latency_model_v2_stub import LatencyModel
        return LatencyModel(cfg=cfg)


def check_split(split: str) -> None:
    if split not in ("dev", "test"):
        raise SystemExit(f"--split must be dev or test, got {split!r}")
    if split == "test" and not PREREG_FROZEN.exists():
        raise SystemExit(
            "Refusing --split test: the test split is sealed until "
            "docs/research_v2/PREREG_FROZEN exists (CONTRACTS.md §7).")


def query_variant_for(doc_ids: Sequence[str], order_sensitive: bool = True) -> str:
    ids = list(doc_ids) if order_sensitive else sorted(set(doc_ids))
    return "qv-" + hashlib.sha256("\x1f".join(ids).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Phase A cell store
# ---------------------------------------------------------------------------
class CellStore:
    def __init__(self, records: Iterable[dict], order_sensitive: bool = True):
        self.order_sensitive = order_sensitive
        self.retrievals: Dict[Tuple[str, str, Optional[int]], dict] = {}
        self.cells: Dict[Tuple[str, int, str, str, str], dict] = {}
        for rec in records:
            kind = rec.get("kind")
            if kind == "retrieval":
                pi = rec.get("partial_index")
                key = (str(rec["item_id"]), str(rec["condition"]), None if pi is None else int(pi))
                rec = dict(rec)
                rec.setdefault("query_variant", query_variant_for(rec["retrieved_doc_ids"], order_sensitive))
                self.retrievals[key] = rec
            elif kind == "cell":
                key = (str(rec["item_id"]), int(rec["seed"]), str(rec["condition"]), str(rec["path"]),
                       str(rec.get("query_variant", "none")))
                self.cells[key] = rec
            else:
                raise ValueError(f"cell store record with unknown kind {kind!r}")

    @classmethod
    def load(cls, path: Path, order_sensitive: bool = True) -> "CellStore":
        path = Path(path)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        records = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                if f.suffix == ".json":
                    records.extend(json.load(fh))
                else:
                    records.extend(json.loads(line) for line in fh if line.strip())
        return cls(records, order_sensitive)

    def retrieval_for(self, item_id: str, condition: str, partial_index: Optional[int]) -> dict:
        key = (item_id, condition, partial_index)
        if key not in self.retrievals:
            raise KeyError(f"Phase A has no retrieval for {key} — a decision landed on an "
                           f"unmeasured dispatch point; run Phase A over every partial")
        return self.retrievals[key]

    def cell(self, item_id: str, seed: int, condition: str, path: str, variant: str) -> dict:
        key = (item_id, int(seed), condition, path, variant)
        if key not in self.cells:
            raise KeyError(f"Phase A never generated cell {key}")
        return self.cells[key]

    def gold_hit_partials(self, item_id: str, condition: str, gold_ids: Iterable[str]) -> List[int]:
        gold = set(gold_ids)
        return sorted(k[2] for k, rec in self.retrievals.items()
                      if k[0] == item_id and k[1] == condition and k[2] is not None
                      and gold & set(rec["retrieved_doc_ids"]))


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------
def with_profile(turn: TurnCtx, profile: str, cfg: dict, prices: Optional[dict] = None,
                 oracle: Optional[dict] = None) -> TurnCtx:
    q = cfg["latency_profiles"][profile]
    dist = latency_dist_from_quantiles(q, int(cfg["latency_points"]), profile)
    signals = dict(turn.signals or {})
    previous = (turn.signals or {}).get("retrieval") or {}
    signals["retrieval"] = {"latency_profile": profile, "latency_quantiles_ms": dict(q),
                            "expected_latency_ms": dist.mean_ms}
    if previous.get("nonhideable_increment_ms") is not None:
        # a per-turn measured pi survives the profile substitution
        signals["retrieval"]["nonhideable_increment_ms"] = previous["nonhideable_increment_ms"]
    return TurnCtx(item_id=turn.item_id, signals=signals, prices=dict(prices or {}),
                   condition=turn.condition, transcript_final=turn.transcript_final,
                   turn_id=turn.turn_id, conversation_id=turn.conversation_id,
                   oracle=oracle if oracle is not None else turn.oracle)


def _rng_for(item_id: str, seed: int, condition: str, profile: str) -> random.Random:
    digest = hashlib.sha256(f"{item_id}|{seed}|{condition}|{profile}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:12], 16))


def outcome_record(name: str, decision, turn: TurnCtx, seed: int, profile: str, prices: Optional[dict],
                   store: CellStore, latency_model, split: str, architecture: str = "commit") -> dict:
    """
    One scored record. A DISCARDED speculative dispatch (decision.used is
    False) is scored on the skip cell and the skip-path TTFA (the answer was
    generated without the passages and did not wait for them), but its
    compute is counted (retrievals = 1) and its passages are logged.
    """
    item, cond = turn.item_id, turn.condition
    retr = None
    if decision.retrieve:
        retr = store.retrieval_for(item, cond, decision.dispatch_partial_index)
    if decision.uses_passages:
        variant, path = retr["query_variant"], "retrieve"
    else:
        variant, path = "none", "skip"
    cell = store.cell(item, seed, cond, path, variant)
    ue = user_end_ms(turn)
    comps = latency_model.sample_ttfa(path, decision.dispatch_t_ms if path == "retrieve" else None,
                                      ue, profile, cell.get("n_prompt_tokens", 0),
                                      _rng_for(item, seed, cond, profile))
    exposed = comps.get("exposed_retrieval_ms")
    return {
        "baseline": name,
        "seed": seed,
        "scenario_id": item,
        "item_id": item,
        "split": split,
        "architecture": architecture,
        "prices": prices,
        "condition": cond,
        "latency_profile": profile,
        "dispatch_t_ms": decision.dispatch_t_ms,
        "exposed_latency_ms": 0.0 if exposed is None else float(exposed),
        "timing_label": decision.timing_label,
        "query_variant": variant,
        "used": decision.uses_passages if decision.retrieve else None,
        "outcome": {
            "answer_correct": bool(cell["answer_correct"]),
            "ttfa_ms": float(comps["ttfa_ms"]),
            "retrievals": 1 if decision.retrieve else 0,
        },
        "ttfa_components": comps,
        "decision": {
            "retrieve": decision.retrieve,
            "used": decision.used,
            "decided_t_ms": decision.decided_t_ms,
            "discarded_query_variant": (retr["query_variant"]
                                        if retr is not None and not decision.uses_passages else None),
            "timing_label": decision.timing_label,
            "dispatch_partial_index": decision.dispatch_partial_index,
            "query_text": decision.query_text,
            "reason": decision.reason,
            "retrieved_doc_ids": None if retr is None else retr["retrieved_doc_ids"],
        },
        "oracle": name == "oracle_v2",
    }


def run_phase_b(
    turns: Sequence[TurnCtx],
    store: CellStore,
    controllers: Dict[str, SequentialController],
    price_points: Sequence[dict],
    seeds: Sequence[int],
    profiles: Sequence[str],
    cfg: dict,
    latency_model,
    split: str = "dev",
    labels: Optional[Dict[str, bool]] = None,
    gold: Optional[Dict[str, List[str]]] = None,
) -> List[dict]:
    """Every controller x price (priced only) x profile x turn x seed, under
    the architecture of `cfg` (controller.e3v2.architecture). The decision
    is computed once per (controller, price, profile, turn) and the seed
    varies only the measured outcome. A speculative-only controller under
    the commit architecture is an error (run_architectures() leaves it out)."""
    architecture = cfg["architecture"]
    records: List[dict] = []
    for name, ctrl in controllers.items():
        if name in SPECULATIVE_ONLY_V2 and architecture != "speculative":
            raise ValueError(f"{name} needs architecture: speculative (got {architecture!r})")
        if ctrl.cfg.get("architecture") != architecture:
            raise ValueError(f"{name} was built for architecture {ctrl.cfg.get('architecture')!r}, "
                             f"evaluating under {architecture!r}")
        priced = name in PRICED_V2
        for prices in (price_points if priced else [None]):
            for profile in profiles:
                for turn in turns:
                    oracle = None
                    if name == "oracle_v2":
                        if labels is None or gold is None:
                            raise ValueError("oracle_v2 needs labels and gold ids")
                        oracle = {"closed_book_wrong": bool(labels[turn.item_id]),
                                  "gold_hit_partial_indices": store.gold_hit_partials(
                                      turn.item_id, turn.condition, gold.get(turn.item_id, [])),
                                  "label_source": "E1-v2 empirical closed-book label"}
                    ctx = with_profile(turn, profile, cfg, prices, oracle)
                    decision = ctrl.decide_turn(ctx)
                    for seed in seeds:
                        records.append(outcome_record(name, decision, ctx, seed, profile, prices,
                                                      store, latency_model, split, architecture))
    return records


def run_architectures(
    turns: Sequence[TurnCtx],
    store: CellStore,
    names: Sequence[str],
    price_points: Sequence[dict],
    seeds: Sequence[int],
    profiles: Sequence[str],
    cfg: dict,
    latency_model,
    architectures: Sequence[str] = ("commit", "speculative"),
    split: str = "dev",
    labels: Optional[Dict[str, bool]] = None,
    gold: Optional[Dict[str, List[str]]] = None,
    controller_params: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    """
    Phase B once per pipeline architecture, controllers rebuilt for each.
    Under commit, speculative-only controllers (b10_speculative_v2) are left
    out: a commit pipeline cannot discard a retrieval. Every other
    controller is run under both, and every record carries its
    architecture, so results are reported per architecture.
    """
    records: List[dict] = []
    for arch in architectures:
        arch_cfg = {**cfg, "architecture": arch}
        ctrls = {n: get_controller_v2(n, cfg=arch_cfg, **((controller_params or {}).get(n) or {}))
                 for n in names if arch == "speculative" or n not in SPECULATIVE_ONLY_V2}
        records.extend(run_phase_b(turns, store, ctrls, price_points, seeds, profiles, arch_cfg,
                                   latency_model, split, labels, gold))
    return records


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_items(path: Optional[Path]) -> Dict[str, dict]:
    return {} if path is None else {str(it["id"]): it for it in _read_jsonl(path)}


def split_ids(items: Dict[str, dict], splits_path: Optional[Path], split: str) -> Optional[set]:
    if splits_path is not None:
        with open(splits_path, "r", encoding="utf-8") as fh:
            return set(json.load(fh)[split])
    if items:
        return {i for i, it in items.items() if it.get("split") == split}
    return None


def trace_timing(trace: dict) -> dict:
    """CONTRACTS §4 speech trace -> the timing fields of signals["timing"]."""
    out = {"speech_onset_ms": 0, "partials": trace.get("partials") or [],
           "user_end_ms": trace.get("vad_user_end_ms"),
           "partial_interval_ms": trace.get("partial_interval_ms")}
    if trace.get("asr_final"):
        out["asr_final"] = trace["asr_final"]
    return out


def load_turns(signals_path: Path, traces_dir: Optional[Path], keep: Optional[set]) -> List[TurnCtx]:
    """signals.jsonl lines: {"item_id", "condition", "signals": {...§6...},
    "transcript_final"?}. Timing fields missing from a signals block are
    filled from <traces_dir>/<voice_id>.jsonl when condition == voice_id
    (or "<voice_id>|<noise>")."""
    traces: Dict[Tuple[str, str], dict] = {}
    if traces_dir is not None:
        for f in sorted(Path(traces_dir).glob("*.jsonl")):
            for tr in _read_jsonl(f):
                cond = tr.get("voice_id", f.stem)
                traces[(str(tr["item_id"]), cond)] = tr
                if tr.get("noise"):
                    traces[(str(tr["item_id"]), f"{cond}|{tr['noise']}")] = tr
    turns = []
    for row in _read_jsonl(signals_path):
        item, cond = str(row["item_id"]), str(row.get("condition", "default"))
        if keep is not None and item not in keep:
            continue
        signals = dict(row.get("signals") or {})
        tr = traces.get((item, cond))
        if tr is not None:
            timing = trace_timing(tr)
            timing.update(signals.get("timing") or {})
            signals["timing"] = timing
        final = row.get("transcript_final") or ((tr or {}).get("asr_final") or {}).get("text", "")
        turns.append(TurnCtx(item_id=item, signals=signals, condition=cond, transcript_final=final))
    return turns


def load_price_points(grid_path: Path) -> List[dict]:
    with open(grid_path, "r", encoding="utf-8") as fh:
        grid = json.load(fh)
    lam_c = grid.get("lambda_c", 0.05)
    values = grid.get("lambda_l") or grid.get("lambda_l_grid") or grid.get("grid")
    if values and isinstance(values[0], dict):
        return [{"lambda_l": float(v["lambda_l"]), "lambda_c": float(v.get("lambda_c", lam_c))} for v in values]
    return [{"lambda_l": float(v), "lambda_c": float(lam_c)} for v in values]


def write_records(records: List[dict], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: Dict[Tuple[str, int], List[dict]] = {}
    for r in records:
        groups.setdefault((r.get("architecture", "commit"), r["baseline"], r["seed"]), []).append(r)
    paths = []
    for (arch, name, seed), rows in sorted(groups.items()):
        p = out_dir / f"e3_v2_{arch}_{name}_seed{seed}.json"
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        tmp.replace(p)
        paths.append(p)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E3-v2 Phase B: causal decisions over cached Phase A cells.")
    ap.add_argument("--split", required=True)
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--traces-dir", type=Path)
    ap.add_argument("--items", type=Path)
    ap.add_argument("--splits", type=Path)
    ap.add_argument("--labels", type=Path, help="JSON {item_id: closed_book_wrong} from E1-v2")
    ap.add_argument("--grid", type=Path, default=_REPO_ROOT / "configs" / "e3_price_grid.json")
    ap.add_argument("--profiles", default="faiss")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--controllers", default=",".join(list_controllers_v2()))
    ap.add_argument("--estimates", type=Path, help="tune_v2.py artifact")
    ap.add_argument("--architectures", default="commit,speculative",
                    help="pipeline architectures to report (commit | speculative)")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "simulator" / "e3v2_results")
    args = ap.parse_args(argv)

    check_split(args.split)
    overrides = {"estimates_artifact": str(args.estimates)} if args.estimates else None
    cfg = load_e3v2_config(overrides=overrides)
    items = load_items(args.items)
    keep = split_ids(items, args.splits, args.split)
    turns = load_turns(args.signals, args.traces_dir, keep)
    store = CellStore.load(args.cells, bool(cfg["phase_b"]["query_variant_order_sensitive"]))
    labels = None
    if args.labels:
        with open(args.labels, "r", encoding="utf-8") as fh:
            labels = {str(k): bool(v) for k, v in json.load(fh).items()}
    gold = {i: list(it.get("gold_doc_ids") or []) for i, it in items.items()}
    names = [n for n in args.controllers.split(",") if n]
    if "oracle_v2" in names and labels is None:
        names.remove("oracle_v2")
        print("oracle_v2 skipped: no --labels")
    records = run_architectures(turns, store, names, load_price_points(args.grid),
                                [int(s) for s in args.seeds.split(",")], args.profiles.split(","),
                                cfg, load_latency_model(cfg), args.architectures.split(","),
                                args.split, labels, gold)
    for p in write_records(records, args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
