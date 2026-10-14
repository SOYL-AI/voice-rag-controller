"""
items.py - load the questions to render from any supported items file.

Supported inputs
  * datasets/v2/<name>/items.jsonl   (CONTRACTS.md §2)  -> dataset name <name>
  * simulator/scenarios.json         (v1, a JSON list)  -> dataset name v1_scenarios,
                                      item_id "v1_scenarios-%04d" % scenario["id"]

Output root for a dataset: datasets/v2/<name>/speech/
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def dataset_name_for(path: Path) -> str:
    path = Path(path)
    if path.name == "scenarios.json":
        return "v1_scenarios"
    return path.parent.name


def speech_root(name: str) -> Path:
    return REPO_ROOT / "datasets" / "v2" / name / "speech"


def load_items(path, split: str = "all") -> list[dict]:
    """[{"item_id", "text", "split"}] in file order."""
    path = Path(path)
    out = []
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for s in data:
            out.append({"item_id": f"v1_scenarios-{int(s['id']):04d}", "text": s["question"].strip(),
                        "split": "v1"})
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out.append({"item_id": r["id"], "text": r["question"].strip(),
                                "split": r.get("split", "unknown")})
    if split != "all":
        out = [r for r in out if r["split"] == split]
    ids = [r["item_id"] for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate item ids in {path}")
    return out


def shard(items: list[dict], spec: str | None) -> list[dict]:
    """--shard k/n keeps every n-th item starting at k (0-based)."""
    if not spec:
        return items
    k, n = (int(x) for x in spec.split("/"))
    return items[k::n]
