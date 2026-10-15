"""Judge spend ledger with a hard cap.

    python -m llm.experiment1_v2.spend            # print the ledger
    python -m llm.experiment1_v2.spend --json     # machine-readable

One JSON ledger is shared by every process and every git worktree. The
default path is the MAIN checkout's
llm/experiment1_v2/results/_spend_ledger.json, resolved through
`git rev-parse --git-common-dir`; override it with VOICE_RAG_JUDGE_LEDGER.
The file is gitignored. Each judge budget (judges.json "budget") has one
entry:

  gpt-6.1-sol  primary judge, both deployments. Starts from the baseline
               already spent (Azure metrics to 2026-10-16: 28,710,941 input +
               3,142,391 output tokens = $88.85) and is capped at INR 20,000
               (VOICE_RAG_JUDGE_CAP_INR).
  gpt-4o       secondary judge. Tracked and reported, no cap by default.

Cost = (input tokens x input price + output tokens x output price) / 1M,
where output tokens are the completion tokens (reasoning tokens included),
converted at INR/USD (default 88, VOICE_RAG_JUDGE_INR_PER_USD).

Before each call the judge RESERVES a conservative estimate: the prompt at
one token per 3 characters, plus the full max-completion budget. If
committed + live reservations + this estimate would exceed the cap, it
raises JudgeBudgetExceeded and makes no call. After the call it commits the
actual usage and drops the reservation; a failed call just drops it. A
reservation older than 15 minutes (a crashed process) is ignored. All
read-modify-write goes through an exclusive lock file, so parallel
processes are safe. Verdicts already graded stay in the judge cache, so a
stopped run resumes from where it stopped once the cap is raised.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.experiment1_v2 import common  # noqa: E402

LEDGER_RELPATH = Path("llm") / "experiment1_v2" / "results" / "_spend_ledger.json"
RESERVATION_TTL_S = 15 * 60
LOCK_STALE_S = 60.0


class JudgeBudgetExceeded(RuntimeError):
    """The next judge call could exceed the spend cap. No call was made."""


def main_repo_root() -> Path:
    """The main checkout (shared by all worktrees), via git's common dir."""
    try:
        out = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=common.REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        common_dir = Path(out.stdout.strip())
        if out.returncode == 0 and out.stdout.strip():
            if not common_dir.is_absolute():
                common_dir = (common.REPO_ROOT / common_dir).resolve()
            return common_dir.parent
    except Exception:  # noqa: BLE001 -- fall back to this checkout
        pass
    return common.REPO_ROOT


def default_ledger_path() -> Path:
    override = os.environ.get("VOICE_RAG_JUDGE_LEDGER")
    return Path(override) if override else main_repo_root() / LEDGER_RELPATH


@contextmanager
def file_lock(path: Path, timeout_s: float = 120.0):
    """Exclusive cross-process lock: an O_EXCL lock file next to the ledger.
    A lock older than LOCK_STALE_S (a killed process) is broken."""
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not lock {lock} within {timeout_s}s")
            time.sleep(0.02)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def cost_usd(input_tokens: float, output_tokens: float, price: dict) -> float:
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6


class SpendBudget:
    """One ledger entry (one judge budget), shared through the ledger file."""

    _process_lock = threading.Lock()

    def __init__(self, key: str, price_usd_per_1m: dict, inr_per_usd: float, cap_inr=None,
                 baseline: dict = None, ledger_path: Path = None, log_every: int = 50,
                 label: str = None):
        self.key = key
        self.price = {"input": float(price_usd_per_1m["input"]),
                      "output": float(price_usd_per_1m["output"])}
        self.inr_per_usd = float(inr_per_usd)
        self.cap_inr = None if cap_inr in (None, "", "none") else float(cap_inr)
        self.baseline = dict(baseline or {})
        self.path = Path(ledger_path) if ledger_path else default_ledger_path()
        self.log_every = max(1, int(log_every))
        self.label = label or key
        self._commits = 0

    # -- ledger I/O (call under file_lock) --------------------------------
    def _read(self) -> dict:
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
        return {"_comment": "E1-v2 judge spend ledger (llm/experiment1_v2/spend.py). "
                            "Gitignored; do not hand-edit while a grading run is active.",
                "entries": {}}

    def _write(self, ledger: dict) -> None:
        ledger["updated_at_utc"] = common.now_utc()
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def _entry(self, ledger: dict) -> dict:
        entry = ledger["entries"].get(self.key)
        if entry is None:
            entry = {"baseline": {"input_tokens": int(self.baseline.get("input_tokens", 0)),
                                  "output_tokens": int(self.baseline.get("output_tokens", 0)),
                                  "as_of": self.baseline.get("as_of"),
                                  "source": self.baseline.get("source")},
                     "tracked": {"input_tokens": 0, "output_tokens": 0, "calls": 0,
                                 "estimated_calls": 0},
                     "reservations": {}}
            ledger["entries"][self.key] = entry
        # Prices, rate and cap are refreshed from config on every write so the
        # file shows what the running code enforces.
        entry["price_usd_per_1m"] = self.price
        entry["inr_per_usd"] = self.inr_per_usd
        entry["cap_inr"] = self.cap_inr
        return entry

    def _spent_usd(self, entry: dict) -> float:
        b, t = entry["baseline"], entry["tracked"]
        return cost_usd(b["input_tokens"] + t["input_tokens"],
                        b["output_tokens"] + t["output_tokens"], self.price)

    def _reserved_usd(self, entry: dict) -> float:
        now = time.time()
        live = {k: r for k, r in entry["reservations"].items() if now - r["at"] < RESERVATION_TTL_S}
        entry["reservations"] = live
        return sum(r["usd"] for r in live.values())

    # -- the API the judge uses --------------------------------------------
    def estimate_usd(self, prompt_chars: int, max_output_tokens: int) -> float:
        return cost_usd(prompt_chars / 3.0 + 50, max_output_tokens, self.price)

    def reserve(self, estimate_usd: float) -> str:
        """Hold `estimate_usd` against the cap, or raise JudgeBudgetExceeded."""
        rid = uuid.uuid4().hex
        with self._process_lock, file_lock(self.path):
            ledger = self._read()
            entry = self._entry(ledger)
            spent, reserved = self._spent_usd(entry), self._reserved_usd(entry)
            if self.cap_inr is not None:
                projected_inr = (spent + reserved + estimate_usd) * self.inr_per_usd
                if projected_inr > self.cap_inr:
                    self._write(ledger)
                    raise JudgeBudgetExceeded(
                        f"{self.label}: spend cap reached. Spent INR {spent * self.inr_per_usd:,.0f} "
                        f"+ in flight INR {reserved * self.inr_per_usd:,.0f} + this call "
                        f"(up to INR {estimate_usd * self.inr_per_usd:,.2f}) would exceed the cap of "
                        f"INR {self.cap_inr:,.0f} (ledger {self.path}). No call was made. Graded "
                        "verdicts are cached; re-run the same command to resume once the cap is "
                        "raised (VOICE_RAG_JUDGE_CAP_INR).")
            entry["reservations"][rid] = {"usd": estimate_usd, "at": time.time(), "pid": os.getpid()}
            self._write(ledger)
        return rid

    def release(self, rid: str) -> None:
        with self._process_lock, file_lock(self.path):
            ledger = self._read()
            entry = self._entry(ledger)
            entry["reservations"].pop(rid, None)
            self._write(ledger)

    def commit(self, rid, input_tokens=None, output_tokens=None, estimate_usd=None) -> dict:
        """Record a finished call's actual usage (or, if the service reported
        none, the reservation's estimate, conservatively) and drop the
        reservation."""
        with self._process_lock, file_lock(self.path):
            ledger = self._read()
            entry = self._entry(ledger)
            reservation = entry["reservations"].pop(rid, None) if rid else None
            tracked = entry["tracked"]
            if input_tokens is None or output_tokens is None:
                usd = estimate_usd if estimate_usd is not None else (reservation or {}).get("usd", 0.0)
                # Book the estimate as output tokens at the output price (exact in USD).
                output_tokens = usd * 1e6 / self.price["output"]
                input_tokens = 0
                tracked["estimated_calls"] += 1
            tracked["input_tokens"] += int(input_tokens)
            tracked["output_tokens"] += int(round(output_tokens))
            tracked["calls"] += 1
            self._write(ledger)
            status = self._status(entry)
        self._commits += 1
        if self._commits % self.log_every == 0:
            print(f"[spend] {self.label}: INR {status['spent_inr']:,.0f} spent"
                  + (f", INR {status['headroom_inr']:,.0f} headroom of {status['cap_inr']:,.0f}"
                     if status["cap_inr"] is not None else " (no cap)"), flush=True)
        return status

    def _status(self, entry: dict) -> dict:
        spent = self._spent_usd(entry)
        reserved = self._reserved_usd(entry)
        return {
            "key": self.key, "spent_usd": spent, "spent_inr": spent * self.inr_per_usd,
            "reserved_inr": reserved * self.inr_per_usd, "cap_inr": self.cap_inr,
            "headroom_inr": None if self.cap_inr is None else
            self.cap_inr - (spent + reserved) * self.inr_per_usd,
            "calls": entry["tracked"]["calls"],
        }

    def status(self) -> dict:
        with self._process_lock, file_lock(self.path):
            ledger = self._read()
            entry = self._entry(ledger)
            self._write(ledger)
            return self._status(entry)


def budget_from_config(role: str, judges_cfg: dict, env=None, ledger_path=None):
    """The SpendBudget for a judge role (None if it has no budget block)."""
    env = os.environ if env is None else env
    block = (judges_cfg["judges"][role] or {}).get("budget")
    if not block:
        return None
    spend_cfg = judges_cfg.get("spend") or {}
    rate = float(env.get("VOICE_RAG_JUDGE_INR_PER_USD") or spend_cfg.get("inr_per_usd", 88))
    cap_env = block.get("cap_env")
    cap = env.get(cap_env) if cap_env and env.get(cap_env) is not None else block.get("cap_inr")
    return SpendBudget(
        key=block["ledger_key"], price_usd_per_1m=block["price_usd_per_1m"], inr_per_usd=rate,
        cap_inr=cap, baseline=block.get("baseline"),
        ledger_path=ledger_path or env.get("VOICE_RAG_JUDGE_LEDGER") or None,
        log_every=int(env.get("VOICE_RAG_JUDGE_SPEND_LOG_EVERY") or spend_cfg.get("log_every", 50)),
        label=f"{role} judge ({block['ledger_key']})",
    )


def bump_for_test(ledger_path: str, n: int, input_tokens: int, output_tokens: int) -> None:
    """Test helper (run in subprocesses): n reserve+commit cycles."""
    budget = SpendBudget("concurrency-test", {"input": 2.0, "output": 10.0}, 88.0,
                         cap_inr=None, ledger_path=Path(ledger_path))
    for _ in range(n):
        rid = budget.reserve(0.001)
        budget.commit(rid, input_tokens, output_tokens)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", type=Path, help="default: the main checkout's ledger")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    from llm.experiment1_v2.judges import load_judges_config  # noqa: PLC0415

    cfg = load_judges_config()
    path = args.ledger or default_ledger_path()
    rows = []
    for role in cfg["judges"]:
        budget = budget_from_config(role, cfg, ledger_path=path)
        if budget is None:
            continue
        status = budget.status()
        ledger = json.loads(path.read_text(encoding="utf-8"))
        entry = ledger["entries"][budget.key]
        rows.append({"role": role, **status, "baseline": entry["baseline"],
                     "tracked": entry["tracked"], "price_usd_per_1m": entry["price_usd_per_1m"],
                     "inr_per_usd": entry["inr_per_usd"], "in_flight_reservations": len(entry["reservations"])})
    if args.json:
        print(json.dumps({"ledger": str(path), "budgets": rows}, indent=2))
        return 0
    print(f"ledger: {path}")
    for r in rows:
        b, t = r["baseline"], r["tracked"]
        print(f"\n{r['role']} judge [{r['key']}]  price ${r['price_usd_per_1m']['input']}/"
              f"${r['price_usd_per_1m']['output']} per 1M in/out, INR {r['inr_per_usd']}/USD")
        print(f"  baseline : {b['input_tokens']:,} in + {b['output_tokens']:,} out "
              f"(as of {b.get('as_of')}; {b.get('source')})")
        print(f"  tracked  : {t['input_tokens']:,} in + {t['output_tokens']:,} out over {t['calls']:,} "
              f"calls ({t['estimated_calls']} booked at estimate)")
        print(f"  spent    : ${r['spent_usd']:,.2f} = INR {r['spent_inr']:,.0f}; in flight INR "
              f"{r['reserved_inr']:,.0f} ({r['in_flight_reservations']} reservations)")
        if r["cap_inr"] is None:
            print("  cap      : none (reported only)")
        else:
            print(f"  cap      : INR {r['cap_inr']:,.0f}; headroom INR {r['headroom_inr']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
