"""Score E3's answer judge against hand labels.

    python -m simulator.judge_validation.score_judges     (needs the Azure env vars)

Grades the cached answers in simulator/e3_sweep_cache/answers that are listed
in labels.json, with the judge exactly as simulator/e3_full_sweep.py is
configured, and reports agreement with those labels.

WHY THIS EXISTS. E3's frontier plots answer accuracy against latency, so the
judge IS the y-axis. The first full sweep graded with a terse prompt that got
an 8-token budget and no room to reason, and it decided borderline cases on
surface form: an answer identical to its reference except for the words "The
Grand Horizon Hotel & Suites offers" at the front scored NO five times out of
five, and YES five out of five with that phrase removed. That is not a
property anyone would want in a measuring instrument, and it was invisible
until the verdicts were compared against labels.

MEASURED, on the 19 labels in labels.json:

    terse prompt, answer unstripped     14/19   74%   <- the first sweep
    terse prompt, answer stripped       15/19   79%
    reasoned prompt, stripped           18/19   95%   <- adopted

Those first two rows are historical: the terse prompt is gone from
e3_full_sweep.py, so this script no longer reproduces them. They are recorded
here and in the adopting commit rather than kept runnable, because keeping a
superseded prompt alive in the scorer invites grading a future sweep with it
by accident. What this script checks now is that the CURRENT configuration
still agrees with the labels.

DIRECTION OF ERROR MATTERS MORE THAN THE RATE. Every disagreement, under every
variant tested, was the judge calling a correct answer incorrect — never the
reverse. A judge that errs only that way under-credits retrieval, and so
penalises whichever baseline retrieves most, which in E3 is the joint
controller the experiment is about. So the two directions are reported
separately below and should not be collapsed into one accuracy number.

The labels are one person's reading of answer against reference under the
single rule recorded in labels.json. They are committed so the choice of judge
is auditable and overridable: change a label, re-run this, and see whether the
conclusion moves.

V2 (E1-v2) MODE. With --v2-labels FILE this scores the two E1-v2 judges
(llm/experiment1_v2/judges.py: primary GPT-6-family, secondary the v4 GPT-4o
judge above) against the team's double-labelled answers, through the same
grade() path label.py uses, and reports agreement with each human labeller,
with the adjudicated/consensus label, and between the two judges (percent
agreement and Cohen's kappa), with the two error directions kept separate:

    python -m simulator.judge_validation.score_judges --v2-labels labels_v2.csv \
        [--judges primary,secondary] [--judge-backend azure|mock] [--out report.json]

FILE is CSV or JSON ({"cases": [...]} or a list) with, per case:
    case_id, dataset, item_id, question, reference, answer,
    key_facts   (JSON list or ' | '-separated; optional),
    aliases     (JSON list or ' | '-separated; optional),
    label_a, label_b   (the two human labels: 1/0, yes/no, true/false, correct/incorrect),
    label_adjudicated  (optional; used when present, else the a/b consensus;
                        cases where a != b with no adjudication are excluded from
                        judge-vs-human scoring and counted),
    path        retrieve | skip (for the gate's path error-rate condition).

The report ends with the PREREG_V2 section 2.4 gate, on the ~2/3 holdout
(sha256(case_id) mod 3 != 0; the rest is prompt-development): human kappa >=
0.70 first; then per judge kappa >= 0.80 with lower 95% bootstrap bound >=
0.70, accuracy >= 0.90, |error(retrieve) - error(skip)| <= 0.05, and for the
k-sample judge unanimity >= 0.90 and Fleiss' kappa >= 0.80.
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

HERE = Path(__file__).resolve().parent
ANSWERS = _REPO / "simulator" / "e3_sweep_cache" / "answers"


# ---------------------------------------------------------------------------
# Agreement statistics (shared with llm/experiment1_v2/label.py)
# ---------------------------------------------------------------------------
def agreement(a, b) -> dict:
    """Percent agreement and Cohen's kappa between two boolean raters over
    the positions where both are not None. kappa is None when it is
    undefined (both raters constant and identical: expected agreement 1)."""
    pairs = [(bool(x), bool(y)) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    out = {"n": n, "agree": None, "percent_agreement": None, "cohen_kappa": None,
           "a_yes_b_no": 0, "a_no_b_yes": 0}
    if not n:
        return out
    agree = sum(x == y for x, y in pairs)
    po = agree / n
    pa, pb = sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    out.update({
        "agree": agree,
        "percent_agreement": po,
        "cohen_kappa": None if pe >= 1.0 else (po - pe) / (1 - pe),
        "a_yes_b_no": sum(x and not y for x, y in pairs),
        "a_no_b_yes": sum(y and not x for x, y in pairs),
    })
    return out


def fleiss_kappa(ratings) -> dict:
    """Fleiss' kappa for binary ratings: `ratings` is a list of per-subject
    lists of k booleans (the k judge samples on one answer). Subjects with a
    missing rating or a different k are left out."""
    rows = [r for r in ratings if r and None not in r]
    if not rows:
        return {"n": 0, "k": None, "fleiss_kappa": None}
    k = len(rows[0])
    rows = [r for r in rows if len(r) == k]
    n = len(rows)
    if k < 2:
        return {"n": n, "k": k, "fleiss_kappa": None}
    yes = [sum(bool(v) for v in r) for r in rows]
    p_i = [(y * (y - 1) + (k - y) * (k - y - 1)) / (k * (k - 1)) for y in yes]
    p_bar = sum(p_i) / n
    p_yes = sum(yes) / (n * k)
    p_e = p_yes ** 2 + (1 - p_yes) ** 2
    return {"n": n, "k": k, "fleiss_kappa": None if p_e >= 1.0 else (p_bar - p_e) / (1 - p_e),
            "observed_agreement": p_bar}


def parse_label(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "correct", "c"):
        return True
    if text in ("0", "false", "no", "n", "incorrect", "wrong", "i"):
        return False
    if text in ("", "na", "n/a", "none", "null", "skip"):
        return None
    raise ValueError(f"unrecognised label {value!r}")


def _parse_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if text.startswith("["):
        return json.loads(text)
    return [part.strip() for part in text.split(" | ") if part.strip()]


def load_v2_labels(path: Path) -> list:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["cases"] if isinstance(data, dict) else data
    cases = []
    for index, row in enumerate(rows):
        a, b = parse_label(row.get("label_a")), parse_label(row.get("label_b"))
        adjudicated = parse_label(row.get("label_adjudicated"))
        reference_label = adjudicated if adjudicated is not None else (a if a == b else None)
        cases.append({
            "case_id": row.get("case_id") or f"case-{index:04d}",
            "dataset": row.get("dataset") or "default",
            "item_id": row.get("item_id") or row.get("case_id") or f"case-{index:04d}",
            "question": row["question"],
            "reference": row["reference"],
            "answer": row["answer"],
            "key_facts": _parse_list(row.get("key_facts")),
            "aliases": _parse_list(row.get("aliases")),
            "core_facts": _parse_list(row.get("core_facts")) or None,
            "history": json.loads(row["history"]) if isinstance(row.get("history"), str)
            and row["history"].strip().startswith("[") else (row.get("history") or []),
            "label_a": a, "label_b": b, "label_adjudicated": adjudicated,
            "label": reference_label,
            "path": (row.get("path") or "").strip().lower() or None,  # retrieve | skip
        })
    explicit = {str(row.get("gate_subset") or "").strip() for row in rows}
    for case, row in zip(cases, rows):
        # The labelling kit writes an exact 100/200 partition (ranked by
        # sha256 of the answer id); older files fall back to sha256 mod 3.
        given = str(row.get("gate_subset") or "").strip()
        case["gate_subset"] = given if given in ("prompt_dev", "holdout") else gate_subset(case["case_id"])
    if explicit - {"", "prompt_dev", "holdout"}:
        raise ValueError(f"unknown gate_subset values {sorted(explicit - {'', 'prompt_dev', 'holdout'})}")
    return cases


def gate_subset(case_id: str) -> str:
    """PREREG_V2 2.4: split by sha256 of the answer id, before any judge sees
    it, into ~100 prompt-development and ~200 gate-holdout answers."""
    import hashlib  # noqa: PLC0415

    return "prompt_dev" if int(hashlib.sha256(str(case_id).encode()).hexdigest(), 16) % 3 == 0 \
        else "holdout"


GATE = {"human_kappa_min": 0.70, "kappa_min": 0.80, "kappa_lower95_min": 0.70,
        "accuracy_min": 0.90, "path_error_gap_max": 0.05, "unanimous_min": 0.90,
        "fleiss_min": 0.80}


def _kappa_lower95(judge, human, n_boot=2000, seed=0):
    import random  # noqa: PLC0415

    pairs = [(j, h) for j, h in zip(judge, human) if j is not None and h is not None]
    if len(pairs) < 10:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        k = agreement([a for a, _ in sample], [b for _, b in sample])["cohen_kappa"]
        if k is not None:
            values.append(k)
    values.sort()
    return values[int(0.025 * len(values))] if values else None


def gate_report(cases, verdicts, samples, roles) -> dict:
    """The judge-validation gate on the holdout subset, criteria as in
    PREREG_V2 2.4. A criterion that cannot be computed counts as not passed."""
    human = agreement([c["label_a"] for c in cases], [c["label_b"] for c in cases])
    hold = [i for i, c in enumerate(cases) if c["gate_subset"] == "holdout" and c["label"] is not None]
    out = {"criteria": GATE, "n_holdout_labelled": len(hold),
           "n_prompt_dev": sum(c["gate_subset"] == "prompt_dev" for c in cases),
           "human_kappa": human["cohen_kappa"],
           "human_kappa_ok": human["cohen_kappa"] is not None
           and human["cohen_kappa"] >= GATE["human_kappa_min"], "judges": {}}
    for role in roles:
        j = [verdicts[role][i] for i in hold]
        h = [cases[i]["label"] for i in hold]
        stats = agreement(j, h)
        lower = _kappa_lower95(j, h)
        gaps = {}
        for path in ("retrieve", "skip"):
            idx = [n for n, i in enumerate(hold) if cases[i]["path"] == path and j[n] is not None]
            gaps[path] = (sum(j[n] != h[n] for n in idx) / len(idx)) if idx else None
        gap = abs(gaps["retrieve"] - gaps["skip"]) if None not in gaps.values() else None
        ratings = [samples[role][i] for i in hold]
        k = max((len(r) for r in ratings), default=0)
        fleiss = fleiss_kappa([r[:k] for r in ratings]) if k > 1 else {"fleiss_kappa": None}
        unanimous = (sum(1 for r in ratings if len(r) >= k and None not in r[:k] and len(set(r[:k])) == 1)
                     / len(ratings)) if ratings and k > 1 else None
        checks = {
            "kappa": stats["cohen_kappa"] is not None and stats["cohen_kappa"] >= GATE["kappa_min"],
            "kappa_lower95": lower is not None and lower >= GATE["kappa_lower95_min"],
            "accuracy": stats["percent_agreement"] is not None
            and stats["percent_agreement"] >= GATE["accuracy_min"],
            "path_error_gap": gap is not None and gap <= GATE["path_error_gap_max"],
            "unanimous": unanimous is not None and unanimous >= GATE["unanimous_min"],
            "fleiss": fleiss["fleiss_kappa"] is not None and fleiss["fleiss_kappa"] >= GATE["fleiss_min"],
        }
        if k <= 1:  # a deterministic single-sample judge has no self-consistency to test
            checks["unanimous"] = checks["fleiss"] = None
        out["judges"][role] = {
            "kappa": stats["cohen_kappa"], "kappa_lower95": lower, "accuracy": stats["percent_agreement"],
            "n": stats["n"], "error_rate_by_path": gaps, "path_error_gap": gap,
            "fraction_unanimous": unanimous, "fleiss_kappa": fleiss["fleiss_kappa"], "checks": checks,
            "passed": out["human_kappa_ok"] and all(v for v in checks.values() if v is not None),
        }
    return out


def _versus(judge_labels, cases, field):
    report = agreement(judge_labels, [c[field] for c in cases])
    report["wrongly_incorrect"] = [c["case_id"] for c, j in zip(cases, judge_labels)
                                   if c[field] is True and j is False]
    report["wrongly_correct"] = [c["case_id"] for c, j in zip(cases, judge_labels)
                                 if c[field] is False and j is True]
    return report


def validate_v2(labels_path: Path, roles=("primary", "secondary"), backend="azure",
                cache_dir: Path = None, workers: int = 8, rubric: str = None) -> dict:
    """Score the E1-v2 judges against double-labelled human cases."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from llm.experiment1_v2 import judges as v2  # noqa: PLC0415

    cfg = json.loads((_REPO / "llm" / "experiment1_v2" / "config.json").read_text(encoding="utf-8"))
    contexts = cfg["labels"]["judge_context"]
    cases = load_v2_labels(labels_path)
    judges_cfg = v2.load_judges_config()
    if rubric:
        judges_cfg["prompt_version"] = rubric
    judges = v2.make_judges(roles, backend, judges_cfg)
    cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="judge_v2_val_"))

    def run(role, case):
        verdict = v2.grade(
            judges[role], item_id=case["item_id"], question=case["question"],
            reference=case["reference"], answer=case["answer"], key_facts=case["key_facts"],
            aliases=case["aliases"], context=contexts.get(case["dataset"], contexts["default"]),
            cache_dir=cache_dir, dataset=case["dataset"], history=case["history"],
            core_facts=case["core_facts"],
        )
        return verdict

    verdicts, samples = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for role in roles:
            full = list(pool.map(lambda c, r=role: run(r, c), cases))
            verdicts[role] = [v["answer_correct"] if v["parsed_ok"] else None for v in full]
            samples[role] = [[s["answer_correct"] for s in v["samples"]] for v in full]

    report = {
        "labels_file": str(labels_path),
        "n_cases": len(cases),
        "n_with_reference_label": sum(c["label"] is not None for c in cases),
        "n_human_disagreements_unadjudicated": sum(
            c["label_a"] is not None and c["label_b"] is not None and c["label_a"] != c["label_b"]
            and c["label_adjudicated"] is None for c in cases),
        "human_a_vs_b": agreement([c["label_a"] for c in cases], [c["label_b"] for c in cases]),
        "judges": {role: judges[role].identity for role in roles},
        "judge_parse_failures": {role: sum(v is None for v in verdicts[role]) for role in roles},
        "vs_reference": {role: _versus(verdicts[role], cases, "label") for role in roles},
        "vs_label_a": {role: _versus(verdicts[role], cases, "label_a") for role in roles},
        "vs_label_b": {role: _versus(verdicts[role], cases, "label_b") for role in roles},
        "by_dataset": {},
    }
    if len(roles) >= 2:
        report["judge_vs_judge"] = agreement(verdicts[roles[0]], verdicts[roles[1]])
    for dataset in sorted({c["dataset"] for c in cases}):
        index = [i for i, c in enumerate(cases) if c["dataset"] == dataset]
        subset = [cases[i] for i in index]
        report["by_dataset"][dataset] = {
            role: agreement([verdicts[role][i] for i in index], [c["label"] for c in subset])
            for role in roles
        }
    report["gate"] = gate_report(cases, verdicts, samples, roles)
    return report


def _print_v2(report: dict) -> None:
    def fmt(stats):
        if not stats["n"]:
            return "n=0"
        kappa = "n/a" if stats["cohen_kappa"] is None else f"{stats['cohen_kappa']:.3f}"
        return f"{stats['agree']}/{stats['n']} = {stats['percent_agreement']:.1%}  kappa {kappa}"

    print(f"cases : {report['n_cases']} ({report['n_with_reference_label']} with a reference "
          f"label; {report['n_human_disagreements_unadjudicated']} unadjudicated human splits)")
    print(f"human A vs B           : {fmt(report['human_a_vs_b'])}")
    for role, ident in report["judges"].items():
        vs = report["vs_reference"][role]
        print(f"{role:<9} {ident['deployment']}[{ident['model_id']}]@{ident['api_version']}")
        print(f"   vs reference label  : {fmt(vs)}")
        print(f"   vs labeller A / B   : {fmt(report['vs_label_a'][role])} | "
              f"{fmt(report['vs_label_b'][role])}")
        print(f"   wrongly INCORRECT   : {len(vs['wrongly_incorrect'])}   "
              f"wrongly CORRECT: {len(vs['wrongly_correct'])}   "
              f"parse failures: {report['judge_parse_failures'][role]}")
    if "judge_vs_judge" in report:
        print(f"primary vs secondary   : {fmt(report['judge_vs_judge'])}")
    gate = report["gate"]
    print(f"GATE (holdout n={gate['n_holdout_labelled']}, prompt-dev n={gate['n_prompt_dev']}): "
          f"human kappa {gate['human_kappa']} ok={gate['human_kappa_ok']}")
    for role, g in gate["judges"].items():
        print(f"   {role:<9} passed={g['passed']}  checks={g['checks']}")


def score_production(judge, scenario: dict, answer: str) -> bool:
    """Grade one answer through the production path.

    Deliberately calls e3_full_sweep.judge_answer() rather than rebuilding
    the prompt here: a validation harness that reimplements the thing it
    validates can pass while the real path is broken. The cache is a fresh
    temp directory each call so a verdict is never served from the sweep's
    cache instead of being recomputed.
    """
    # Imported here, not at module level: the E3 runner pulls in the whole
    # pipeline and the v1 FAISS index, which the v2 mode must not need.
    from simulator.e3_full_sweep import judge_answer, judge_path  # noqa: PLC0415

    tmp = Path(tempfile.mkdtemp(prefix="judge_validation_"))
    judge_path(tmp, judge.slug, "0" * 64).parent.mkdir(parents=True, exist_ok=True)
    return judge_answer(judge, scenario, answer, tmp)["answer_correct"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score answer judges against hand labels.")
    parser.add_argument("--v2-labels", type=Path,
                        help="E1-v2 human-labelled CSV/JSON; scores the v2 judges instead of E3's")
    parser.add_argument("--judges", default="primary,secondary")
    parser.add_argument("--judge-backend", choices=("azure", "mock"), default="azure")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out", type=Path, help="write the v2 report JSON here")
    parser.add_argument("--rubric", help="judge prompt version (default: judges.json, e1v2-r2)")
    args = parser.parse_args(argv)
    if args.v2_labels:
        roles = tuple(r.strip() for r in args.judges.split(",") if r.strip())
        report = validate_v2(args.v2_labels, roles, args.judge_backend, args.cache_dir,
                             args.workers, args.rubric)
        _print_v2(report)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"report: {args.out}")
        return 0
    return main_e3()


def main_e3() -> int:
    from simulator.e3_full_sweep import AzureOpenaiJudge  # noqa: PLC0415 -- see score_production

    labels = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))
    scenarios = {
        item["id"]: item
        for item in json.loads(
            (_REPO / "simulator" / "scenarios.json").read_text(encoding="utf-8")
        )
    }
    judge = AzureOpenaiJudge.from_env()
    cases = labels["cases"]

    print(f"judge : {judge.identity}")
    print(f"labels: {len(cases)} cases from {labels['source']}")
    print(f"rule  : {labels['labelling_rule'][:96]}...")
    print()

    agree = 0
    wrongly_incorrect = []  # label True, judge False -> under-credits retrieval
    wrongly_correct = []    # label False, judge True -> hides hallucination
    missing = []

    for case in cases:
        path = ANSWERS / (
            f"s{case['scenario_id']:03d}_seed{case['seed']}_{case['path']}.json"
        )
        if not path.exists():
            missing.append(path.name)
            continue
        answer = json.loads(path.read_text(encoding="utf-8"))["answer_text"]
        got = score_production(judge, scenarios[case["scenario_id"]], answer)
        tag = f"s{case['scenario_id']:03d}/{case['path']}"
        if got == case["label"]:
            agree += 1
        elif case["label"]:
            wrongly_incorrect.append(tag)
        else:
            wrongly_correct.append(tag)

    scored = len(cases) - len(missing)
    if missing:
        print(f"skipped {len(missing)} case(s) with no cached answer: "
              f"{', '.join(missing)}")
        print()
    if not scored:
        print("nothing to score — is simulator/e3_sweep_cache/answers populated?")
        return 1

    print(f"agreement with labels : {agree}/{scored} = {agree / scored:.0%}")
    print(f"wrongly INCORRECT ({len(wrongly_incorrect)}): "
          f"{', '.join(wrongly_incorrect) or '-'}")
    print(f"   ^ judge marks a correct answer wrong. Under-credits retrieval.")
    print(f"wrongly CORRECT   ({len(wrongly_correct)}): "
          f"{', '.join(wrongly_correct) or '-'}")
    print(f"   ^ judge marks a wrong answer correct. Makes skipping look free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
