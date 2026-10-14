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
                        judge-vs-human scoring and counted).
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
            "label_a": a, "label_b": b, "label_adjudicated": adjudicated,
            "label": reference_label,
        })
    return cases


def _versus(judge_labels, cases, field):
    report = agreement(judge_labels, [c[field] for c in cases])
    report["wrongly_incorrect"] = [c["case_id"] for c, j in zip(cases, judge_labels)
                                   if c[field] is True and j is False]
    report["wrongly_correct"] = [c["case_id"] for c, j in zip(cases, judge_labels)
                                 if c[field] is False and j is True]
    return report


def validate_v2(labels_path: Path, roles=("primary", "secondary"), backend="azure",
                cache_dir: Path = None, workers: int = 8) -> dict:
    """Score the E1-v2 judges against double-labelled human cases."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from llm.experiment1_v2 import judges as v2  # noqa: PLC0415

    cfg = json.loads((_REPO / "llm" / "experiment1_v2" / "config.json").read_text(encoding="utf-8"))
    contexts = cfg["labels"]["judge_context"]
    cases = load_v2_labels(labels_path)
    judges = v2.make_judges(roles, backend)
    cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="judge_v2_val_"))

    def run(role, case):
        verdict = v2.grade(
            judges[role], item_id=case["item_id"], question=case["question"],
            reference=case["reference"], answer=case["answer"], key_facts=case["key_facts"],
            aliases=case["aliases"], context=contexts.get(case["dataset"], contexts["default"]),
            cache_dir=cache_dir,
        )
        return verdict["answer_correct"] if verdict["parsed_ok"] else None

    verdicts = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for role in roles:
            verdicts[role] = list(pool.map(lambda c, r=role: run(r, c), cases))

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
    args = parser.parse_args(argv)
    if args.v2_labels:
        roles = tuple(r.strip() for r in args.judges.split(",") if r.strip())
        report = validate_v2(args.v2_labels, roles, args.judge_backend, args.cache_dir,
                             args.workers)
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
