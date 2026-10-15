"""
grid_subset.py - the fixed, seeded dev subset for the speech robustness grid.

    python -m audio.v2.grid_subset            # writes datasets/v2/{d1_open,d2_support}/speech/grid_subset.json

300 dev items: 150 from d1_open stratified by popularity bin (pop_bin 0-4, plus
the multihop items as a sixth stratum; equal allocation, 25 each) and 150 from
d2_support stratified by item_type (allocation proportional to the dev counts,
largest remainder). Within a stratum the items with the lowest
sha256("speech-grid-v1|<id>") are taken, so the subset is deterministic and
independent of file order. Dev ids come from each dataset's splits.json.

Render / trace it with `--ids datasets/v2/<name>/speech/grid_subset.json`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SALT = "speech-grid-v1"


def _h(i):
    return hashlib.sha256(f"{SALT}|{i}".encode()).hexdigest()


def _load(name):
    d = REPO / "datasets" / "v2" / name
    dev = set(json.loads((d / "splits.json").read_text(encoding="utf-8"))["dev"])
    items = [json.loads(l) for l in (d / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in items if r["id"] in dev]


def _alloc(sizes: dict, n: int, equal: bool) -> dict:
    keys = sorted(sizes)
    if equal:
        q = {k: n / len(keys) for k in keys}
    else:
        tot = sum(sizes.values())
        q = {k: n * sizes[k] / tot for k in keys}
    out = {k: min(int(q[k]), sizes[k]) for k in keys}
    for k in sorted(keys, key=lambda k: -(q[k] - int(q[k]))):
        if sum(out.values()) >= n:
            break
        if out[k] < sizes[k]:
            out[k] += 1
    return out


def select(name, stratum, n, equal):
    items = _load(name)
    groups: dict = {}
    for r in items:
        groups.setdefault(stratum(r), []).append(r["id"])
    alloc = _alloc({k: len(v) for k, v in groups.items()}, n, equal)
    ids = []
    for k, v in sorted(groups.items()):
        ids += sorted(v, key=_h)[: alloc[k]]
    return sorted(ids), {str(k): alloc[k] for k in sorted(alloc)}


def main():
    specs = [
        ("d1_open", lambda r: "multihop" if r["item_type"] == "multihop" else f"pop_bin_{r['meta']['pop_bin']}",
         150, True, "pop_bin (0-4) + multihop, equal allocation"),
        ("d2_support", lambda r: r["item_type"], 150, False, "item_type, proportional allocation"),
    ]
    for name, strat, n, equal, desc in specs:
        ids, alloc = select(name, strat, n, equal)
        out = REPO / "datasets" / "v2" / name / "speech" / "grid_subset.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"dataset": name, "split": "dev", "n": len(ids), "salt": SALT,
                                   "stratified_by": desc, "allocation": alloc, "ids": ids}, indent=1),
                       encoding="utf-8")
        print(name, len(ids), alloc)


if __name__ == "__main__":
    main()
