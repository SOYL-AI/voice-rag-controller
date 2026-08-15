"""
fit_calibration.py — fit E3's VoI calibration on Experiment 1's real samples.

WHAT THIS PRODUCES, AND WHY IT IS A SEPARATE COMMITTED STEP.

Experiment 3's objective needs a quality term, and the principled reading of
that term is a calibrated probability (controller/baselines/_objective.py's
VoiCalibration):

    voi = P(retrieval is needed | this turn's uncertainty)

Experiment 1 is what earns that reading. It ran 36 labelled questions x 5
seeds = 180 closed-book generations on Mistral-7B-Instruct-v0.3 via Modal,
scored every uncertainty signal by ROC AUC, and reported max_entropy_nats as
the best signal at mean AUC 0.978 (verdict PASS, gate thresholds
success 0.70 / kill 0.60). E1 validated that the signal RANKS; it never fit
a probability, because ranking is all AUC needs.

This script fits that probability, on E1's own samples, in the one shape
_objective.py can consume:

    P(retrieval_needed) = sigmoid(slope * max_entropy_nats + intercept)

and writes the coefficients plus their provenance to
controller/baselines/e1_calibration.json, which configs/config.yaml's
controller.e3.voi.calibration.from_artifact points at.

WHY max_entropy_nats AND NOT A MULTI-SIGNAL COMBO. E1 saved no logistic
combination — its headline is a single-signal result. Fitting a combo here
would be a NEW model selection, done after seeing the data, with no held-out
split to defend it and no line in the pre-registration. Grounding on E1's
actual validated headline signal keeps the E3 quality term traceable to a
result that was gated, rather than to a model this script invented.

WHY A COMMITTED ARTIFACT RATHER THAN NUMBERS PASTED INTO CONFIG. The
artifact records the source file, its sha256, the sample count, the fit
date, and the recovered AUC, so a reader of the paper can tell which Modal
run the controller's quality term came from and re-derive it. Pasted
coefficients cannot be traced to a run and cannot be checked.

Usage:
    python -m llm.experiment1.fit_calibration            # refit + rewrite artifact
    python -m llm.experiment1.fit_calibration --check    # verify, write nothing

--check re-fits and compares against the committed artifact without touching
it; it exits nonzero on drift, so CI can assert that the committed
coefficients still match the committed E1 result.

*** THE FIT IS IN-SAMPLE, AND THE ARTIFACT SAYS SO. *** 180 samples, all of
them used. There is no held-out AUC here and this script does not pretend
otherwise: the reported AUC is a RECOVERY check (does the fitted sigmoid
preserve E1's ranking?), not evidence of generalization. That evidence is
E1's own five-seed result, which this file's provenance block cites. Note
too that the 180 samples are 36 questions x 5 seeds, so they are clustered,
not independent — the effective sample size is nearer 36 than 180, which is
why nothing downstream quotes a standard error on these coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1.run import roc_auc  # noqa: E402 -- see _REPO_ROOT fixup

# The E1 Modal run this calibration is grounded on. Pinned by path rather
# than globbed over results/ so that adding a second E1 run to the repo
# cannot silently re-ground the controller's quality term.
DEFAULT_E1_RESULT = (
    _REPO_ROOT
    / "llm"
    / "experiment1"
    / "results"
    / "experiment_1_modal_mistral7b_20260811_151212.json"
)
DEFAULT_ARTIFACT = _REPO_ROOT / "controller" / "baselines" / "e1_calibration.json"

# The signal to fit on. Not a free choice — it must be the signal E1's gate
# actually passed on, and load_e1_samples() below asserts that against the
# result file's own best_signal rather than trusting this constant.
SIGNAL = "max_entropy_nats"


def load_e1_samples(result_path: Path):
    """
    Read (x, y, metadata) out of a committed E1 result JSON.

    Refuses anything that is not a valid, passing gate run. The repo also
    contains llm/experiment1/results/smoke_mock.json, which is a
    MockClient run whose logprobs are drawn independently of the labels —
    fitting on it yields slope -1.51 at AUC 0.445 (below chance). That file
    is exactly what this guard exists to reject: a calibration fitted on
    mock data would install an INVERTED uncertainty -> value mapping while
    every log line downstream still looked healthy.
    """
    with open(result_path, "r", encoding="utf-8") as handle:
        result = json.load(handle)

    if not result.get("valid_gate_run"):
        raise ValueError(
            f"{result_path} is not a valid gate run (valid_gate_run="
            f"{result.get('valid_gate_run')!r}). A VoI calibration must be "
            f"fitted on a real, gated E1 run — see this module's docstring."
        )
    if result.get("verdict") != "PASS":
        raise ValueError(
            f"{result_path} has verdict {result.get('verdict')!r}, not PASS. "
            f"Refusing to ground E3's quality term on a signal E1's own gate "
            f"did not clear."
        )
    if result.get("best_signal") != SIGNAL:
        raise ValueError(
            f"{result_path}'s best_signal is {result.get('best_signal')!r}, "
            f"but this script fits {SIGNAL!r}. Fitting a signal other than "
            f"the one E1 validated would make E3's quality term untraceable "
            f"to E1's result — update SIGNAL deliberately, or use the run "
            f"whose headline signal this is."
        )

    samples = result.get("samples") or []
    values, labels, seeds = [], [], []
    for sample in samples:
        raw = (sample.get("signals") or {}).get(SIGNAL)
        label = sample.get("retrieval_needed")
        # Both are required per-sample: a sample missing either cannot
        # contribute to the fit, and silently dropping it would change n
        # without changing the recorded provenance.
        if raw is None or label is None:
            raise ValueError(
                f"E1 sample {sample.get('id')!r} (seed {sample.get('seed')!r}) "
                f"is missing {SIGNAL!r} or retrieval_needed; cannot fit."
            )
        values.append(float(raw))
        labels.append(1.0 if label else 0.0)
        seeds.append(sample.get("seed"))

    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    if y.min() == y.max():
        raise ValueError("E1 samples contain only one class; cannot fit.")

    return x, y, seeds, result


def fit_logistic(x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-12):
    """
    Maximum-likelihood 1-D logistic regression by Newton-Raphson (IRLS),
    returning (slope, intercept) for P(y=1) = sigmoid(slope * x + intercept).

    Hand-rolled on numpy rather than pulled from scikit-learn for two
    reasons: numpy is already the repo's only numeric dependency
    (requirements.txt), and sklearn's LogisticRegression applies L2
    regularization BY DEFAULT (C=1.0), which would shrink the coefficients
    of a 2-parameter fit by an amount nobody reading the artifact would
    know about. This is the unregularized MLE, which is what the artifact
    claims it is.

    Newton converges in under 10 iterations on this problem. The tiny ridge
    added to the Hessian is a numerical guard against a singular solve if
    the classes ever became perfectly separable (they are not here — the
    fit's in-sample accuracy is 90.6%), not a regularizer: at 1e-10 it
    changes the coefficients far below the precision the artifact records.
    """
    design = np.column_stack([x, np.ones_like(x)])
    beta = np.zeros(2)
    for iteration in range(max_iter):
        probability = _sigmoid(design @ beta)
        weights = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (y - probability)
        hessian = design.T @ (design * weights[:, None])
        step = np.linalg.solve(hessian + 1e-10 * np.eye(2), gradient)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError(
            f"logistic fit did not converge in {max_iter} Newton iterations"
        )
    return float(beta[0]), float(beta[1]), iteration + 1


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def build_calibration(result_path: Path = DEFAULT_E1_RESULT) -> dict:
    """
    Fit on `result_path` and return the artifact dict, including the
    top-level "calibration" object _objective.py's load_voi_calibration()
    reads via from_artifact.
    """
    x, y, seeds, result = load_e1_samples(result_path)
    slope, intercept, iterations = fit_logistic(x, y)

    probability = _sigmoid(slope * x + intercept)
    # A sigmoid is strictly monotone in x, so this AUC must equal the raw
    # signal's AUC exactly. Computing both and recording both is the point:
    # a mismatch would mean the fit is not the monotone map it claims to be
    # (a sign flip is the way that happens), and it is a free check.
    auc_fitted = roc_auc(y, probability)
    auc_signal = roc_auc(y, x)

    per_seed = {}
    for seed in sorted({s for s in seeds if s is not None}):
        index = [i for i, s in enumerate(seeds) if s == seed]
        per_seed[str(seed)] = roc_auc(y[index], probability[index])

    log_loss = float(
        -np.mean(
            y * np.log(np.clip(probability, 1e-15, 1.0))
            + (1.0 - y) * np.log(np.clip(1.0 - probability, 1e-15, 1.0))
        )
    )

    return {
        "_comment": (
            "GENERATED by `python -m llm.experiment1.fit_calibration` — do not "
            "hand-edit. This is Experiment 3's VoI calibration: the mapping "
            "from Experiment 1's validated uncertainty signal to "
            "P(retrieval is needed), consumed by "
            "controller/baselines/_objective.py via configs/config.yaml's "
            "controller.e3.voi.calibration.from_artifact."
        ),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source_file": str(result_path.relative_to(_REPO_ROOT)).replace("\\", "/"),
            "source_sha256": _sha256(result_path),
            "experiment_id": result.get("experiment_id"),
            "e1_run_created_at_utc": result.get("created_at_utc"),
            "backend": result.get("backend"),
            "model": result.get("model"),
            "verdict": result.get("verdict"),
            "e1_best_signal": result.get("best_signal"),
            "e1_best_mean_auc": result.get("best_mean_auc"),
            "e1_thresholds": result.get("thresholds"),
            "dataset": result.get("dataset"),
            "n_samples": int(x.size),
            "n_retrieval_needed": int(y.sum()),
            "n_retrieval_not_needed": int((1.0 - y).sum()),
            "n_questions": (result.get("dataset") or {}).get("examples"),
            "n_seeds": len({s for s in seeds if s is not None}),
            "clustering_note": (
                "The 180 samples are 36 questions x 5 seeds, so they are "
                "clustered rather than independent; the effective sample size "
                "is nearer 36 than 180. No standard error is quoted on these "
                "coefficients for that reason."
            ),
            "fit_method": (
                "unregularized maximum-likelihood 1-D logistic regression, "
                "Newton-Raphson/IRLS on numpy "
                "(llm/experiment1/fit_calibration.py:fit_logistic)"
            ),
            "newton_iterations": iterations,
            "in_sample_only": True,
        },
        "diagnostics": {
            "auc_fitted_probability": auc_fitted,
            "auc_raw_signal": auc_signal,
            "auc_matches_raw_signal": abs(auc_fitted - auc_signal) < 1e-12,
            "auc_per_seed": per_seed,
            "log_loss": log_loss,
            "accuracy_at_0_5": float(np.mean((probability >= 0.5) == (y == 1.0))),
            "decision_boundary_nats": float(-intercept / slope),
            "signal_mean_retrieval_needed": float(x[y == 1.0].mean()),
            "signal_mean_retrieval_not_needed": float(x[y == 0.0].mean()),
            "signal_min": float(x.min()),
            "signal_max": float(x.max()),
            "note": (
                "auc_fitted_probability is a RECOVERY check, not held-out "
                "evidence: the sigmoid is monotone, so it must reproduce the "
                "raw signal's ranking exactly. Generalization evidence is E1's "
                "own five-seed result (e1_best_mean_auc above)."
            ),
        },
        # The block _objective.py's load_voi_calibration() splices over
        # config. Field names must match VoiCalibration's.
        "calibration": {
            "signal": SIGNAL,
            "kind": "logistic",
            "slope": slope,
            "intercept": intercept,
            "calibrated": True,
            "auc": auc_fitted,
            "source": (
                f"Fitted on Experiment 1 Modal run "
                f"{result.get('model')} ({result.get('backend')}), "
                f"{result.get('created_at_utc')}, verdict {result.get('verdict')}: "
                f"P(retrieval_needed) = sigmoid({slope:.4f} * {SIGNAL} "
                f"{intercept:+.4f}) on n={int(x.size)} samples, in-sample "
                f"AUC {auc_fitted:.4f} (E1 five-seed mean AUC "
                f"{result.get('best_mean_auc'):.4f}). "
                f"Regenerate: python -m llm.experiment1.fit_calibration"
            ),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit E3's VoI calibration — P(retrieval_needed) = "
            "sigmoid(slope * max_entropy_nats + intercept) — on Experiment 1's "
            "real samples, and write controller/baselines/e1_calibration.json."
        )
    )
    parser.add_argument(
        "--result",
        default=str(DEFAULT_E1_RESULT),
        help="Committed E1 result JSON to fit on (default: the Modal run).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_ARTIFACT),
        help="Where to write the calibration artifact.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-fit and compare against the committed artifact without "
        "writing it. Exits nonzero if the coefficients have drifted.",
    )
    args = parser.parse_args()

    artifact = build_calibration(Path(args.result))
    fitted = artifact["calibration"]

    print(f"E1 result : {artifact['provenance']['source_file']}")
    print(f"            {artifact['provenance']['model']} "
          f"({artifact['provenance']['backend']}), "
          f"verdict {artifact['provenance']['verdict']}")
    print(f"n samples : {artifact['provenance']['n_samples']} "
          f"({artifact['provenance']['n_retrieval_needed']} needed / "
          f"{artifact['provenance']['n_retrieval_not_needed']} not needed, "
          f"{artifact['provenance']['n_questions']} questions x "
          f"{artifact['provenance']['n_seeds']} seeds)")
    print()
    print(f"  P(retrieval_needed) = sigmoid({fitted['slope']:.4f} * "
          f"{SIGNAL} {fitted['intercept']:+.4f})")
    print()
    print(f"slope a   : {fitted['slope']:+.4f}   "
          f"({'OK, > 0' if fitted['slope'] > 0 else 'FAIL, <= 0'} — higher "
          f"entropy must mean retrieval is worth more)")
    print(f"intercept b: {fitted['intercept']:+.4f}")
    print(f"boundary  : {artifact['diagnostics']['decision_boundary_nats']:.4f} nats "
          f"(P = 0.5)")
    print(f"in-sample AUC : {artifact['diagnostics']['auc_fitted_probability']:.4f} "
          f"(raw signal {artifact['diagnostics']['auc_raw_signal']:.4f}, "
          f"E1 five-seed mean {artifact['provenance']['e1_best_mean_auc']:.4f})")
    print(f"log loss / acc: {artifact['diagnostics']['log_loss']:.4f} / "
          f"{artifact['diagnostics']['accuracy_at_0_5']:.4f}")

    if fitted["slope"] <= 0:
        print(
            "\nREFUSING TO WRITE: a slope <= 0 inverts the uncertainty -> "
            "value-of-information direction E1 is built on.",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.out)
    if args.check:
        try:
            with open(out_path, "r", encoding="utf-8") as handle:
                committed = (json.load(handle) or {}).get("calibration") or {}
        except FileNotFoundError:
            print(f"\nCHECK FAILED: {out_path} does not exist.", file=sys.stderr)
            return 1
        drift = [
            key
            for key in ("signal", "kind", "slope", "intercept", "calibrated")
            if committed.get(key) != fitted.get(key)
        ]
        if drift:
            print(
                f"\nCHECK FAILED: committed artifact differs from a fresh fit "
                f"on {args.result} in {drift}.",
                file=sys.stderr,
            )
            return 1
        print(f"\nCHECK OK: {out_path} matches a fresh fit.")
        return 0

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    print(f"\nwrote {out_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
