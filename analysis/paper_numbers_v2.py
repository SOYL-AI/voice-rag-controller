"""
paper_numbers_v2.py -- small OFFLINE computations for the v2 paper that are
not in docs/research_v2/v2_test_results_final.json. No model, judge, GPU or
cloud call: every number is read or recomputed from files already on disk.

    PYTHONUTF8=1 VOICE_RAG_MODEL=mistral7b_v03 .venv-e2/Scripts/python.exe -m analysis.paper_numbers_v2 \
        --phase-a <confirmatory-run checkout>/simulator/e3v2_phase_a

Writes docs/research_v2/paper_numbers_v2.json. Contents:
  * e1_latency_dev: median closed-book draft latency (TTFT, full generation)
    and median self-assessment probe latency, isolated dev timing runs
    (llm/experiment1_v2/results/<ds>/mistral7b_v03/{latency,self_assessment_latency}_dev.jsonl;
    80 calls each, timing_synthetic = false).
  * e1_test_signals: test AUC of self_assessed_need, mean_entropy_nats and
    max_entropy_nats on the full question (empirical label), and the text
    prefix curves of self_assessed_need and mean_entropy_nats, copied from the
    frozen analyze.py output (analysis_test.json).
  * e1_test_extra: per dataset, the DEV-fitted operational calibration applied
    to test (ECE, 10 bins; boundary mass within +-0.15 of b*), and the
    self_assessed_need AUC on the primary-ASR condition at the latest partial
    with spoken word fraction <= f (f = 1.0: ASR final transcript), with 95%
    percentile bootstrap CIs (items for d1_open, pair_id clusters for
    d2_support; B = 2,000). Same rule as analysis/v2_test_d1.py:asr_curve.
  * remaining_speech_dev: out-of-fold (5 folds, the dev-analysis fold rule)
    error of the remaining-speech predictor on the dev primary-ASR traces.
  * primary_cell_diagnostics_test: q-hat (SPEC discard rate) and
    lambda_c (1 + w-bar) q-hat at the primary cell, from contrasts_test.json.
  * realised_precision: bootstrap SE of the primary contrasts at the realised
    test N and the implied 80%-power MDE (2.8016 x SE), next to the dev
    prediction (PREREG_V2 Sec. 6.1).

d2_support E1 numbers use judge-graded empirical labels and are EXPLORATORY
(PREREG_V2 deviation D3); d1_open uses alias-graded labels.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1_v2 import common as e1common  # noqa: E402
from llm.experiment1_v2.prefixes import normalize_text  # noqa: E402
from llm.experiment1_v2.stats import ece as ece_fn  # noqa: E402

E1 = _REPO_ROOT / "llm" / "experiment1_v2" / "results"
DEFAULT_PA = _REPO_ROOT / "simulator" / "e3v2_phase_a"  # pass --phase-a for the confirmatory-run copy
FINAL = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_final.json"
PRED = _REPO_ROOT / "docs" / "research_v2" / "v2_predictions.json"
POWER = _REPO_ROOT / "docs" / "research_v2" / "v2_power_dev.json"
OUT = _REPO_ROOT / "docs" / "research_v2" / "paper_numbers_v2.json"
ASR = "piper-en_US-hfc_male-medium__150wpm__clean"
DATASETS = ("d1_open", "d2_support")
STATUS = {"d1_open": "pre-registered per-dataset analysis (unaffected by D3)",
          "d2_support": "exploratory (D3: judge agreed only moderately with the human labels)"}
B_AUC = 2000
SEED = 20261018
Z80 = 1.959963984540054 + 0.8416212335729143


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = y.size - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(y, s, clusters, seed=SEED, n_boot=B_AUC):
    """Percentile CI of AUC under cluster resampling."""
    rng = np.random.default_rng(seed)
    uc = sorted(set(clusters))
    cidx = defaultdict(list)
    for k, c in enumerate(clusters):
        cidx[c].append(k)
    groups = [np.array(cidx[c]) for c in uc]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        k = np.concatenate([groups[j] for j in pick])
        if 0 < y[k].sum() < k.size:
            vals.append(auc(y[k], s[k]))
    q = np.percentile(vals, [2.5, 97.5])
    return [float(q[0]), float(q[1])]


# ---------------------------------------------------------------------------
def e1_latency_dev() -> dict:
    out = {}
    for ds in DATASETS:
        d = {}
        for name, fn in (("closed_book_draft", "latency_dev.jsonl"),
                         ("self_assessment_probe", "self_assessment_latency_dev.jsonl")):
            rs = [r for r in e1common.read_jsonl(E1 / ds / "mistral7b_v03" / fn) if not r.get("timing_synthetic")]
            d[name] = {"n_calls": len(rs), "n_items": len({r["item_id"] for r in rs}),
                       "ttft_ms_median": statistics.median(float(r["ttft_ms"]) for r in rs),
                       "gen_ms_median": statistics.median(float(r["gen_ms"]) for r in rs),
                       "source": str((E1 / ds / "mistral7b_v03" / fn).relative_to(_REPO_ROOT))}
        out[ds] = d
    return out


def e1_test_signals() -> dict:
    out = {}
    for ds in DATASETS:
        m = e1common.load_json(E1 / ds / "analysis_test" / "analysis_test.json")["models"]["mistral7b_v03"]
        sig = {}
        for s in ("self_assessed_need", "mean_entropy_nats", "max_entropy_nats"):
            e = m["auc_full"][s]["empirical"]
            sig[s] = {"auc": e["auc"], "ci95": e["ci95"], "n_items": e["n_items"], "n_pos": e["n_pos"]}
        curves = {}
        for s in ("self_assessed_need", "mean_entropy_nats"):
            curves[s] = {str(r["word_frac"]): {"auc": r["empirical"]["auc"], "ci95": r["empirical"]["ci95"]}
                         for r in m["prefix_curve_text"][s]}
        out[ds] = {"auc_full_question": sig, "text_prefix_curve": curves,
                   "bootstrap_B_analyze_py": e1common.load_json(E1 / ds / "analysis_test" / "analysis_test.json")["bootstrap"],
                   "status": STATUS[ds],
                   "source": str((E1 / ds / "analysis_test" / "analysis_test.json").relative_to(_REPO_ROOT))}
    return out


def e1_test_extra(pa: Path) -> dict:
    from simulator.e3v2_dev_analysis import prefix_frac
    out = {}
    for ds in DATASETS:
        cal_path = _REPO_ROOT / "controller" / "baselines" / f"e3v2_calibration_{ds}_mistral7b_v03_self_assessed_need_logit.json"
        cal = e1common.load_json(cal_path)["calibration"]
        labels = {i: v.get("empirical_need") for i, v in
                  e1common.load_json(E1 / ds / "mistral7b_v03" / "labels_test.json")["items"].items()}
        y_of = {i: int(bool(v)) for i, v in labels.items() if v is not None}
        items = {it["id"]: it for it in e1common.load_items(e1common.resolve_dataset_paths(ds)["items"])}
        cluster = {i: (items[i].get("pair_id") or i) if ds == "d2_support" else i for i in y_of}
        sa = {}
        for r in e1common.read_jsonl(E1 / ds / "mistral7b_v03" / "self_assessment_test.jsonl"):
            sa[(r["item_id"], normalize_text(r["prefix_text"]))] = float(r["self_assessed_need"])

        def belief(v, f):
            p = min(1 - 1e-4, max(1e-4, v))
            x = math.log(p / (1 - p))
            z = cal["intercept"] + cal["slope"] * x + cal["frac_slope"] * f + cal["interaction"] * x * f
            return 1 / (1 + math.exp(-max(-40, min(40, z))))

        trs = {}
        for line in open(pa / ds / "conditions_test.jsonl", encoding="utf-8"):
            tr = json.loads(line)
            if tr["condition"] == ASR:
                trs[tr["item_id"]] = tr
        bundle = e1common.load_json(pa / ds / "analysis" / "bundle_asr_3e1c1ada.json")
        g = bundle["overrides"]["quality"]
        b_star = g["g_minus"] / (g["g_plus"] + g["g_minus"])
        calib = {}
        for name, text_of in (("full_text_question", lambda i: normalize_text(items[i]["question"])),
                              ("primary_asr_final", lambda i: normalize_text(trs[i]["asr_final"]["text"]) if i in trs else None)):
            ys, ps = [], []
            for i, y in y_of.items():
                t = text_of(i)
                v = sa.get((i, t)) if t else None
                if v is None:
                    continue
                ys.append(y)
                ps.append(belief(v, 1.0))
            ys, ps = np.array(ys), np.array(ps)
            calib[name] = {"n_items": int(ys.size), "base_rate": float(ys.mean()), "mean_belief": float(ps.mean()),
                           "ece_10_bins_dev_calibration": ece_fn(ys, ps, 10),
                           "boundary_mass": {"b_star": b_star, "half_width": 0.15,
                                             "fraction": float(np.mean(np.abs(ps - b_star) <= 0.15))}}
        curve = {}
        for f in (0.3, 0.5, 0.7, 1.0):
            ids, ss = [], []
            for i, tr in trs.items():
                if i not in y_of:
                    continue
                if f == 1.0:
                    v = sa.get((i, normalize_text(tr["asr_final"]["text"])))
                else:
                    v = None
                    for k, p in enumerate(tr["partials"]):
                        if not normalize_text(p.get("text", "")):
                            continue
                        if prefix_frac(tr, k) <= f:
                            v = sa.get((i, normalize_text(p["text"])), v)
                if v is not None:
                    ids.append(i)
                    ss.append(v)
            y = np.array([y_of[i] for i in ids])
            s = np.array(ss)
            curve[str(f)] = {"n_items": int(y.size), "auc": auc(y, s),
                             "ci95": boot_auc(y, s, [cluster[i] for i in ids], seed=SEED + int(f * 10))}
        out[ds] = {"dev_calibration_on_test": calib, "asr_prefix_curve_self_assessed_need": curve,
                   "bootstrap": {"B": B_AUC, "unit": "pair_id cluster" if ds == "d2_support" else "item"},
                   "calibration_artifact": str(cal_path.relative_to(_REPO_ROOT)),
                   "status": STATUS[ds]}
    return out


def remaining_speech_dev(pa: Path) -> dict:
    from controller.remaining_speech import RemainingSpeechPredictor, samples_from_trace
    from simulator.e3v2_dev_analysis import fold_of
    out = {}
    for ds in DATASETS:
        trs = [json.loads(l) for l in open(pa / ds / "conditions_dev.jsonl", encoding="utf-8")]
        trs = [t for t in trs if t["condition"] == ASR]
        y, q10, q50, q90 = [], [], [], []
        for k in range(5):
            train = [t for t in trs if fold_of(t["item_id"]) != k]
            test = [t for t in trs if fold_of(t["item_id"]) == k]
            if not test:
                continue
            pred = RemainingSpeechPredictor.fit(train, model="qr")
            for t in test:
                for s in samples_from_trace(t):
                    p = pred.predict(s["words"], s["t_ms"])
                    y.append(s["remaining_ms"])
                    q10.append(p["q10"])
                    q50.append(p["q50"])
                    q90.append(p["q90"])
        y, q10, q50, q90 = map(np.asarray, (y, q10, q50, q90))
        ok = (y >= 200.0) & (q50 > 0)
        le = np.log(q50[ok] / y[ok])
        out[ds] = {"n_traces": len(trs), "n_samples": int(y.size),
                   "mae_q50_ms": float(np.mean(np.abs(y - q50))),
                   "median_ae_q50_ms": float(np.median(np.abs(y - q50))),
                   "coverage_q10_q90": float(np.mean((y >= q10) & (y <= q90))),
                   "log_error_sd": float(np.std(le, ddof=1)),
                   "log_error_definition": "ln(q50 / true remaining), samples with true remaining >= 200 ms (n = %d)" % int(ok.sum()),
                   "cv": "5-fold out-of-fold, fold rule simulator.e3v2_dev_analysis.fold_of, qr model",
                   "source": "conditions_dev.jsonl (primary ASR condition) under --phase-a"}
    return out


def primary_cell_diagnostics(pa: Path) -> dict:
    out = {}
    target = {"d1_open": "0.173057|0.05", "d2_support": "0.215782|0.05"}
    for ds in DATASETS:
        doc = e1common.load_json(pa / ds / "contrasts_test.json")
        for c in doc["contrasts"][ASR]:
            if c["latency_profile"] == "rewrite_faiss" and c["price"] == target[ds]:
                out[ds] = {"q_hat": c.get("q_hat"), "w_bar": c.get("w_bar"),
                           "lambda_c_1_plus_wbar_q_hat": c.get("m_c_qhat"), "price": c["price"],
                           "status": STATUS[ds]}
    return out


def realised_precision() -> dict:
    fin = e1common.load_json(FINAL)
    out = {}
    blocks = {"d1_open": fin["per_dataset"]["d1_open"]["primary"],
              "d2_support": fin["per_dataset"]["d2_support"]["primary"], "pooled": fin["pooled"]}
    for name, b in blocks.items():
        out[name] = {k: {"boot_se": b[k]["boot_se"], "mde80": Z80 * b[k]["boot_se"]}
                     for k in ("delta_commit", "delta_speculative", "interaction")}
    out["n_test"] = {"d1_open": fin["per_dataset"]["d1_open"]["n_items"],
                     "d2_support": fin["per_dataset"]["d2_support"]["n_items"],
                     "d2_support_clusters": fin["per_dataset"]["d2_support"]["n_clusters"]}
    out["dev_prediction_prereg_6_1"] = {"se_commit_at_test_N": 0.0055, "mde_commit": 0.0155,
                                        "se_interaction_at_test_N": 0.0024, "mde_interaction": 0.0068}
    return out


def dev_predictions_primary() -> dict:
    p = e1common.load_json(PRED)
    out = {}
    for c in p["cells"]:
        if c["condition"] == ASR and c["profile"] == "rewrite_faiss" and abs(c["belief_quantile"] - 0.65) < 1e-9:
            out[c["dataset"]] = {k: c[k] for k in ("delta_commit", "se_commit", "delta_speculative", "se_speculative",
                                                   "interaction", "se_interaction", "d_star")}
    for c in p["pooled_primary"]:
        if c["profile"] == "rewrite_faiss" and abs(c["belief_quantile"] - 0.65) < 1e-9:
            out["pooled"] = {k: c[k] for k in ("delta_commit", "se_commit", "delta_speculative", "se_speculative",
                                               "interaction", "se_interaction")}
    out["g_dev"] = {ds: p["inputs"][ds]["g"] for ds in DATASETS}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", default=str(DEFAULT_PA))
    a = ap.parse_args(argv)
    pa = Path(a.phase_a)
    doc = {
        "_comment": "Offline paper numbers for v2 (no model/judge calls). d2_support values that use "
                    "judge-graded labels are exploratory (PREREG_V2 D3).",
        "generated_by": "python -m analysis.paper_numbers_v2",
        "generated_at_utc": e1common.now_utc(),
        "phase_a_dir": "simulator/e3v2_phase_a of the frozen confirmatory run (read only)",
        "e1_latency_dev": e1_latency_dev(),
        "e1_test_signals": e1_test_signals(),
        "e1_test_extra": e1_test_extra(pa),
        "remaining_speech_dev": remaining_speech_dev(pa),
        "primary_cell_diagnostics_test": primary_cell_diagnostics(pa),
        "realised_precision": realised_precision(),
        "dev_predictions_primary_cell": dev_predictions_primary(),
    }
    e1common.atomic_write_json(OUT, doc)
    print(json.dumps({k: doc[k] for k in ("e1_latency_dev", "e1_test_extra", "remaining_speech_dev",
                                          "primary_cell_diagnostics_test", "realised_precision")}, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
