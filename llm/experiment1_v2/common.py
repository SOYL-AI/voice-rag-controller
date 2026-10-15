"""Shared plumbing for E1-v2: paths, the sealing guard, item/split loading,
and crash-safe JSON/JSONL I/O.

Every E1-v2 entry point (run, label, analyze) calls check_split_allowed()
before it touches anything, per CONTRACTS.md section 7: the test split is
sealed until docs/research_v2/PREREG_FROZEN exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_JUDGES_CONFIG = HERE / "judges.json"
RESULTS_ROOT = HERE / "results"
# Mock-backend output goes to a sibling tree that no analysis of record reads
# and that is gitignored, the same quarantine the E3 runner uses.
MOCK_RESULTS_ROOT = HERE / "results_mock"
PREREG_FROZEN_RELPATH = Path("docs") / "research_v2" / "PREREG_FROZEN"
VALID_SPLITS = ("dev", "test")

ITEM_REQUIRED = ("id", "dataset", "question", "expected_answer", "retrieval_needed_authored")


class SealedSplitError(RuntimeError):
    """Raised when code asks for the sealed test split before the freeze."""


def check_split_allowed(split: str, repo_root: Path = None) -> None:
    """Refuse `test` unless the pre-registration addendum is frozen."""
    if split not in VALID_SPLITS:
        raise ValueError(f"--split must be one of {VALID_SPLITS}, got {split!r}")
    if split == "test":
        marker = Path(repo_root or REPO_ROOT) / PREREG_FROZEN_RELPATH
        if not marker.exists():
            raise SealedSplitError(
                f"The test split is sealed: {PREREG_FROZEN_RELPATH.as_posix()} does not "
                "exist. No generation, signal extraction, judging or tuning may touch "
                "test items until the v2 pre-registration addendum is frozen and "
                "tagged (docs/research_v2/CONTRACTS.md section 7)."
            )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")  # separator: ("ab","c") != ("a","bc")
    return digest.hexdigest()


def rel(path: Path) -> str:
    """Repo-relative POSIX path when possible (provenance), else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return None


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem[:16]}.{os.getpid()}-{threading.get_ident() % 100000}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def read_jsonl(path: Path, tolerate_partial_tail: bool = True) -> list:
    """Read a JSONL file. A crash mid-append can leave a truncated last line;
    that line is dropped (and will be redone by a resumed run) rather than
    aborting the whole read. A corrupt line anywhere else is an error."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_partial_tail and number == len(lines):
                print(f"[e1v2] dropping truncated last line of {path.name}")
                continue
            raise
    return rows


class JsonlAppender:
    """Append-only JSONL writer that flushes every record, so a killed run
    loses at most the record being written. Repairs a missing trailing
    newline (from a previous crash) before the first append."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_tail()
        self._handle = open(self.path, "a", encoding="utf-8")

    def _repair_tail(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        data = self.path.read_bytes()
        if data.endswith(b"\n"):
            return
        # Truncated tail: drop the partial line so the file stays valid JSONL.
        cut = data.rfind(b"\n")
        self.path.write_bytes(data[: cut + 1] if cut >= 0 else b"")

    def write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Items and splits (CONTRACTS.md sections 2 and 7)
# ---------------------------------------------------------------------------
def dataset_dir(name: str) -> Path:
    return REPO_ROOT / "datasets" / "v2" / name


def resolve_dataset_paths(dataset: str = None, items: Path = None, splits: Path = None,
                          traces_dir: Path = None) -> dict:
    if items is None and dataset is None:
        raise ValueError("give --dataset NAME or --items PATH")
    base = dataset_dir(dataset) if dataset else Path(items).resolve().parent
    items = Path(items) if items else base / "items.jsonl"
    splits = Path(splits) if splits else items.parent / "splits.json"
    traces_dir = Path(traces_dir) if traces_dir else items.parent / "speech" / "traces"
    return {"items": items, "splits": splits, "traces_dir": traces_dir, "base": items.parent}


def load_items(path: Path) -> list:
    rows = read_jsonl(path, tolerate_partial_tail=False)
    seen = set()
    for row in rows:
        missing = [key for key in ITEM_REQUIRED if key not in row]
        if missing:
            raise ValueError(f"{path}: item {row.get('id')!r} missing {missing}")
        if row["id"] in seen:
            raise ValueError(f"{path}: duplicate item id {row['id']!r}")
        if not isinstance(row["retrieval_needed_authored"], bool):
            raise ValueError(f"{path}: retrieval_needed_authored must be bool ({row['id']})")
        seen.add(row["id"])
    return rows


def load_splits(path: Path) -> dict:
    splits = load_json(path)
    dev, test = set(splits.get("dev") or []), set(splits.get("test") or [])
    overlap = dev & test
    if overlap:
        raise ValueError(f"{path}: {len(overlap)} ids in both dev and test")
    return splits


def select_split(items: list, splits: dict, split: str) -> list:
    """Items in `split`. Test ids are loaded only to be excluded (section 7).
    An item in neither list, or whose own `split` field disagrees with
    splits.json, is a dataset error and stops the run."""
    dev, test = set(splits.get("dev") or []), set(splits.get("test") or [])
    chosen = []
    for item in items:
        where = "dev" if item["id"] in dev else "test" if item["id"] in test else None
        if where is None:
            raise ValueError(f"item {item['id']!r} is in neither split of splits.json")
        declared = item.get("split")
        if declared in VALID_SPLITS and declared != where:
            raise ValueError(
                f"item {item['id']!r} says split={declared!r} but splits.json puts it in {where!r}"
            )
        if where == split:
            chosen.append(item)
    return chosen


def contract_split(item_id: str, group: str = None) -> str:
    """CONTRACTS.md section 7: sha256(id) mod 1000 < 300 -> dev. `group`
    (the pair_id) replaces the id so pair partners always share a split."""
    key = group or item_id
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 1000
    return "dev" if value < 300 else "test"


def load_traces(traces_dir: Path, voices="all", noise=None) -> dict:
    """{item_id: [trace, ...]} from speech/traces/*.jsonl, filtered by voice
    and noise condition. Missing directory -> {} (text-only run)."""
    traces_dir = Path(traces_dir)
    by_item: dict = {}
    if not traces_dir.exists():
        return by_item
    wanted_voices = None if voices in (None, "all") else set(voices)
    wanted_noise = None if not noise or noise == "all" else set(noise)
    for path in sorted(traces_dir.glob("*.jsonl")):
        for trace in read_jsonl(path, tolerate_partial_tail=False):
            if wanted_voices is not None and trace.get("voice_id") not in wanted_voices:
                continue
            if wanted_noise is not None and trace.get("noise") not in wanted_noise:
                continue
            by_item.setdefault(trace["item_id"], []).append(trace)
    return by_item


def load_config(path: Path = None) -> dict:
    cfg = load_json(path or DEFAULT_CONFIG)
    protocol = load_json(REPO_ROOT / cfg["protocol_source"])
    cfg["protocol"] = {
        "seeds": list(protocol["seeds"]),
        "generation": dict(protocol["generation"]),
        "low_confidence_threshold": protocol["low_confidence_threshold"],
        "source": cfg["protocol_source"],
        "source_sha256": sha256_file(REPO_ROOT / cfg["protocol_source"]),
    }
    return cfg


def parse_csv_list(value, cast=str):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v.strip()) for v in str(value).split(",") if v.strip()]


def results_dir(dataset: str, model_key: str, mock: bool) -> Path:
    root = MOCK_RESULTS_ROOT if mock else RESULTS_ROOT
    return root / dataset / model_key
