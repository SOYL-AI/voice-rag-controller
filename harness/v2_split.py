"""
v2_split.py — the research-v2 split and sealing rules (CONTRACTS §7) as code.

    item_split(id)        sha256(id) mod 1000 < 300 -> "dev", else "test"
                          (the deterministic rule; a dataset's own items.jsonl
                          `split` field / splits.json wins when present, since
                          stratification and pair_id grouping may move items)
    check_split(split)    raise unless split is "dev", or "test" with
                          docs/research_v2/PREREG_FROZEN present
    filter_split(items, split)
                          items in `split`, honouring each item's own `split`
                          field when present

Every v2 runner takes --split dev|test and calls check_split() first.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_FROZEN = _REPO_ROOT / "docs" / "research_v2" / "PREREG_FROZEN"


class SealedSplitError(RuntimeError):
    pass


def item_split(item_id: str) -> str:
    h = int(hashlib.sha256(str(item_id).encode("utf-8")).hexdigest(), 16)
    return "dev" if h % 1000 < 300 else "test"


def check_split(split: str, frozen_marker: Optional[Path] = None) -> None:
    marker = frozen_marker if frozen_marker is not None else PREREG_FROZEN
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")
    if split == "test" and not Path(marker).exists():
        raise SealedSplitError(
            "The test split is sealed until the v2 pre-registration addendum is "
            f"frozen ({marker} does not exist). Use --split dev."
        )


def split_of(item: dict) -> str:
    return item.get("split") or item_split(item.get("id") or item.get("item_id"))


def filter_split(items: Iterable[dict], split: str) -> List[dict]:
    return [it for it in items if split_of(it) == split]


def load_jsonl(path) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
