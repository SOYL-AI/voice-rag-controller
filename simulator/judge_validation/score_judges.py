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
"""
import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simulator.e3_full_sweep import (  # noqa: E402
    AzureOpenaiJudge,
    judge_answer,
    judge_path,
)

HERE = Path(__file__).resolve().parent
ANSWERS = _REPO / "simulator" / "e3_sweep_cache" / "answers"


def score_production(judge, scenario: dict, answer: str) -> bool:
    """Grade one answer through the production path.

    Deliberately calls e3_full_sweep.judge_answer() rather than rebuilding
    the prompt here: a validation harness that reimplements the thing it
    validates can pass while the real path is broken. The cache is a fresh
    temp directory each call so a verdict is never served from the sweep's
    cache instead of being recomputed.
    """
    tmp = Path(tempfile.mkdtemp(prefix="judge_validation_"))
    judge_path(tmp, judge.slug, "0" * 64).parent.mkdir(parents=True, exist_ok=True)
    return judge_answer(judge, scenario, answer, tmp)["answer_correct"]


def main() -> int:
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
