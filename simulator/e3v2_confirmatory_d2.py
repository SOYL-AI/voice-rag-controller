"""
e3v2_confirmatory_d2.py -- the confirmatory d2_support test sequence after the
judge-validation gate (PREREG_V2 §9 steps 8-9; notes N1, N3; deviation D1).
Frozen modules only, run as subprocesses; every stage is resumable.

    python -m simulator.e3v2_confirmatory_d2 --env-file <main .env> all
    python -m simulator.e3v2_confirmatory_d2 --env-file <main .env> <stage>

Stages (in order):
  1 labels      llm.experiment1_v2.label, d2_support test, E1 closed-book draws,
                judges primary + secondary, primary at k = 1 via
                llm/experiment1_v2/judges_test_k1.json (judges.json with
                primary samples 1; PREREG §2.2) -> labels_test.json
  2 grade       Phase A grade + store on the ORIGINAL plan (sha f59b33bc...),
                primary k = 1 (grade_primary_k1), secondary 30% stratified
  3 oracle      N3: preserve the original plan; recompute the frozen plan's
                non-oracle cells (must equal the original exactly) and the
                oracle's cells; estimate the extra cost (stop if > INR 1,000);
                write the frozen plan; generate + grade the extras; rebuild
                the store
  4 d1sub       d1_open 400-answer judge subsample (both judges, k = 1, seed
                20261017; PREREG §2.1, O6)
  5 evaluate    E1 analyze (d2 test), frozen e3v2_lazy evaluate for d2_support
                (D*[SPEC] = b10_speculative_v2, m3 = 0.016), then
                analysis.v2_test_final (pooled)
Spend guard: before every judge stage the gpt-6.1-sol ledger headroom minus
the stage's estimate must stay >= INR 1,500, else the sequence stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MAIN = Path(r"C:\Users\ryang\OneDrive\Desktop\voice-rag\voice-rag-controller")
E1 = MAIN / "llm" / "experiment1_v2" / "results"
PA = _REPO_ROOT / "simulator" / "e3v2_phase_a"
LOGS = PA / "test_logs"
D2 = PA / "d2_support"
ORIGINAL_PLAN_SHA = "f59b33bce622fb2a637da0f418d3ca3d05eae27f0bf80313a18a161e3b8d50fc"
JUDGES_K1 = _REPO_ROOT / "llm" / "experiment1_v2" / "judges_test_k1.json"
INR_PER_PRIMARY_CALL = 0.17946  # docs/research_v2/v2_test_budget.json: $0.0020394 x 88
MIN_HEADROOM = 1500.0
ORACLE_PREAPPROVED_INR = 1000.0
BUNDLES = [str(D2 / "analysis" / "bundle_asr_3e1c1ada.json"), str(D2 / "analysis" / "bundle_text150.json")]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def say(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "conf_d2_pipeline.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def headroom() -> float:
    from llm.experiment1_v2.judges import load_judges_config
    from llm.experiment1_v2.spend import budget_from_config
    return float(budget_from_config("primary", load_judges_config(JUDGES_K1)).status()["headroom_inr"])


def guard(stage: str, est_inr: float) -> None:
    h = headroom()
    say(f"spend guard [{stage}]: headroom INR {h:,.0f}, estimate INR {est_inr:,.0f}")
    if h - est_inr < MIN_HEADROOM:
        say(f"STOP [{stage}]: headroom after this stage would be below INR {MIN_HEADROOM:,.0f}")
        raise SystemExit(5)


def run(stage: str, cmd: list, env: dict) -> None:
    log = LOGS / f"conf_d2_{stage}.log"
    say(f"{stage}: running {' '.join(c if '.env' not in c else '<env-file>' for c in cmd[1:])}")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(_REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env)
    say(f"{stage}: exit {rc} after {time.time() - t0:.0f} s (log {log.name})")
    if rc != 0:
        raise SystemExit(rc)


def phase_a(args, extra):
    return [sys.executable, "-u", "-m", "simulator.e3v2_phase_a", "--dataset", "d2_support", "--split", "test",
            "--e1-results", str(E1), "--index-dir", str(MAIN / "retrieval" / "indexes" / "d2_support"),
            "--env-file", args.env_file, "--concurrency", "128", "--judge-workers", "8"] + extra


# ---------------------------------------------------------------------------
def stage_labels(args, env):
    n = sum(1 for _ in open(E1 / "d2_support" / "mistral7b_v03" / "signals_test.jsonl", encoding="utf-8"))
    guard("labels", 2490 * INR_PER_PRIMARY_CALL)
    run("labels", [sys.executable, "-u", "-m", "llm.experiment1_v2.label", "--dataset", "d2_support", "--split",
                   "test", "--model", "mistral7b_v03", "--no-retrieval", "--results-dir",
                   str(E1 / "d2_support" / "mistral7b_v03"), "--judges", "primary,secondary",
                   "--judges-config", str(JUDGES_K1), "--workers", "8"], env)
    assert (E1 / "d2_support" / "mistral7b_v03" / "labels_test.json").exists()
    say(f"labels: labels_test.json written ({n} E1 records read)")


def stage_grade(args, env):
    plan = D2 / "required_test.json"
    assert sha(plan) == ORIGINAL_PLAN_SHA, "the d2 plan is not the original (f59b33bc...)"
    guard("grade", 20625 * INR_PER_PRIMARY_CALL)
    run("grade", phase_a(args, ["--stages", "grade,store", "--required-cells", str(plan)]), env)


def _records_cells(names, record):
    from llm.experiment1_v2 import common as e1common
    from simulator.e3v2_decide import CellStore
    from simulator.e3v2_lazy import frozen_records
    store = CellStore.load(D2 / "store_test.jsonl", recording=record)
    for b in BUNDLES:
        frozen_records(e1common.load_json(Path(b)), "d2_support", "test", D2, E1, store, names)


def stage_oracle(args, env):
    from simulator.e3v2_lazy import ALL
    original = D2 / "required_test.pre_oracle.json"
    if not original.exists():
        assert sha(D2 / "required_test.json") == ORIGINAL_PLAN_SHA
        shutil.copyfile(D2 / "required_test.json", original)
    assert sha(original) == ORIGINAL_PLAN_SHA
    orig = {tuple(c) for c in json.loads(original.read_text(encoding="utf-8"))["cells"]}
    non_oracle, oracle = set(), set()
    _records_cells([n for n in ALL if n != "oracle_v2"], non_oracle)
    _records_cells(["oracle_v2"], oracle)
    if non_oracle != orig:
        say(f"STOP [oracle]: non-oracle cells differ from the original plan ({len(non_oracle ^ orig)} cells)")
        raise SystemExit(6)
    extra = oracle - orig
    say(f"oracle: non-oracle cells == original plan ({len(orig)}); oracle cells {len(oracle)}, "
        f"extra {len(extra)} ({sum(1 for c in extra if c[2] == 'retrieve')} retrieve)")
    # write the frozen plan (oracle now included) and check it is exactly original + oracle
    run("oracle_plan", [sys.executable, "-u", "-m", "simulator.e3v2_lazy", "plan", "--dataset", "d2_support",
                        "--split", "test", "--e1-results", str(E1)] + sum((["--bundle", b] for b in BUNDLES), []), env)
    new = {tuple(c) for c in json.loads((D2 / "required_test.json").read_text(encoding="utf-8"))["cells"]}
    if new != orig | oracle:
        say("STOP [oracle]: the re-run plan is not the original plan plus the oracle's cells")
        raise SystemExit(7)
    say(f"oracle: new plan sha256 {sha(D2 / 'required_test.json')} = original + oracle cells")
    # cost of the extra cells: count the answers/skips Phase A would generate
    import argparse as _ap

    from simulator.e3v2_phase_a import PhaseA
    pa = PhaseA(_ap.Namespace(dataset="d2_support", split="test", out=str(PA), conditions="text150," +
                              "piper-en_US-hfc_male-medium__150wpm__clean", seeds="11,23,37,53,71",
                              e1_results=str(E1), index_dir=str(MAIN / "retrieval" / "indexes" / "d2_support"),
                              limit=None, required_cells=str(D2 / "required_test.json")))
    n_new = len(pa.answer_jobs()) + len(pa.skip_jobs())
    est = n_new * INR_PER_PRIMARY_CALL
    say(f"oracle: {n_new} new answers to generate and grade, estimate INR {est:,.0f} "
        f"(pre-approved up to INR {ORACLE_PREAPPROVED_INR:,.0f})")
    if est > ORACLE_PREAPPROVED_INR:
        say("STOP [oracle]: above the pre-approved INR 1,000; report and wait for the PI")
        raise SystemExit(8)
    guard("oracle", est)
    run("oracle_generate_grade", phase_a(args, ["--stages", "answer,skip,grade,store", "--required-cells",
                                                str(D2 / "required_test.json")]), env)


def stage_d1sub(args, env):
    d1 = E1 / "d1_open" / "mistral7b_v03"
    if not (d1 / "labels_test.alias_only.json").exists():
        shutil.copyfile(d1 / "labels_test.json", d1 / "labels_test.alias_only.json")
    guard("d1sub", 400 * INR_PER_PRIMARY_CALL)
    run("d1sub", [sys.executable, "-u", "-m", "llm.experiment1_v2.label", "--dataset", "d1_open", "--split", "test",
                  "--model", "mistral7b_v03", "--no-retrieval", "--results-dir", str(d1),
                  "--judges", "primary,secondary", "--judges-config", str(JUDGES_K1),
                  "--judge-subsample", "400", "--judge-subsample-seed", "20261017", "--workers", "8"], env)


def stage_evaluate(args, env):
    run("e1_analyze", [sys.executable, "-u", "-m", "llm.experiment1_v2.analyze", "--dataset", "d2_support",
                       "--split", "test", "--models", "mistral7b_v03", "--results-root", str(E1 / "d2_support"),
                       "--out-dir", str(E1 / "d2_support" / "analysis_test"), "--no-figures"], env)
    run("evaluate", [sys.executable, "-u", "-m", "simulator.e3v2_lazy", "evaluate", "--dataset", "d2_support",
                     "--split", "test", "--e1-results", str(E1)] + sum((["--bundle", b] for b in BUNDLES), [])
        + ["--d-star-speculative", "b10_speculative_v2", "--m3", "0.016"], env)
    run("final_analysis", [sys.executable, "-u", "-m", "analysis.v2_test_final"]
        + ([] if args.d2_exploratory_reason is None else ["--d2-exploratory-reason", args.d2_exploratory_reason]), env)


STAGES = {"labels": stage_labels, "grade": stage_grade, "oracle": stage_oracle, "d1sub": stage_d1sub,
          "evaluate": stage_evaluate}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", required=True)
    ap.add_argument("--d2-exploratory-reason", default=None,
                    help="passed to analysis.v2_test_final (e.g. 'D3: kappa_human < 0.70'): d2_support-dependent "
                         "and pooled verdicts tagged exploratory, d1_open results confirmatory and primary")
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    args = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    env = {**os.environ, "VOICE_RAG_MODEL": "mistral7b_v03", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    if not (_REPO_ROOT / "docs" / "research_v2" / "PREREG_FROZEN").exists():
        raise SystemExit("PREREG_FROZEN missing")
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        say(f"=== stage {name}")
        STAGES[name](args, env)
    say("sequence done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
