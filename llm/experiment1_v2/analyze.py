"""E1-v2 analysis: does the belief at DISPATCH time predict retrieval need?

Per model and per signal, on the dev split only:

  1. AUC vs the authored label and vs the empirical label (label.py), on the
     full question: mean of per-seed AUCs (v1's statistic) and the
     item-level AUC (signal averaged over seeds) with a cluster-bootstrap CI
     (clusters = pair_id where items come in matched pairs, else item).
  2. The belief-at-dispatch curves: AUC vs text-prefix word fraction, and AUC
     vs time on the speech traces, under two alignments:
        transcript   the latest ASR partial available by time t
        belief_ready the latest partial whose BELIEF was computed by t
                     (partial available_ms + the measured latency of that
                     signal: t_first_k_ms[k] for first-k entropy, gen_ms for
                     whole-answer signals)
     both on absolute ms from speech onset and relative to the user's turn end.
  3. AUC vs first-k tokens next to the measured latency of each k.
  4. Calibration fitted on DEV only (logistic, isotonic); ECE, Brier, log loss
     and reliability from grouped k-fold cross-validation on dev; transfer of
     the full-question calibration to partial prefixes. The test split is
     sealed and is evaluated later, once, with the frozen artifact.
  5. g+ (gain where retrieval is needed) and g- (harm where it is not), with
     cluster-bootstrap CIs, for the corrected objective
        Q = b * g+ - (1 - b) * g-,  break-even belief b* = g- / (g+ + g-).
  6. The surface-form leak baseline (datasets/v2/leak_test_d2.py output),
     when present, next to the signal AUCs.

Writes analysis_<split>.json, figures, and per-model calibration artifacts in
the format of controller/baselines/e1_calibration_mean_entropy.json.

  python -m llm.experiment1_v2.analyze --dataset d1_open --split dev
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.experiment1_v2 import common, stats  # noqa: E402
from llm.experiment1_v2.prefixes import normalize_text  # noqa: E402
from llm.experiment1_v2.self_assess import SIGNAL as SA_SIGNAL  # noqa: E402
from llm.experiment1_v2.signals_v2 import flatten_signals  # noqa: E402

FULL = ("text", 1.0)
LEAK_CANDIDATES = (
    "datasets/v2/d2_support/leak_test_results.json",
    "datasets/v2/d2_support/leak_test_d2.json",
    "datasets/v2/leak_test_d2_results.json",
    "datasets/v2/leak_test_d2.json",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model_inputs(rdir: Path, split: str, labels_name: str = None) -> dict:
    signals_path = rdir / f"signals_{split}.jsonl"
    labels_path = rdir / (labels_name or f"labels_{split}.json")
    records = [r for r in common.read_jsonl(signals_path) if r.get("condition", "closed_book")
               == "closed_book"]
    labels = common.load_json(labels_path) if labels_path.exists() else None
    for record in records:
        record["_flat"] = flatten_signals(record["signals"])
    probe = common.read_jsonl(rdir / f"latency_{split}.jsonl")
    # Self-assessed need (self_assess.py): one deterministic call per distinct
    # prefix text, joined to every record with that (item, text).
    sa_path = rdir / f"self_assessment_{split}.jsonl"
    sa_rows = common.read_jsonl(sa_path)
    sa_latency = latency_model([], sa_rows) if sa_rows else None
    if sa_rows:
        lookup = {(r["item_id"], normalize_text(r["prefix_text"])): r for r in sa_rows}
        for record in records:
            hit = lookup.get((record["item_id"], normalize_text(record["prefix"]["text"])))
            if hit is not None and hit.get(SA_SIGNAL) is not None:
                record["_flat"][SA_SIGNAL] = float(hit[SA_SIGNAL])
                record["_sa_gen_ms"] = (sa_latency["gen_ms_median"]
                                        if sa_latency and sa_latency["valid_latency_estimate"]
                                        else hit.get("gen_ms"))
    return {"records": records, "labels": labels, "signals_path": signals_path,
            "labels_path": labels_path if labels else None,
            "latency": latency_model(probe, records),
            "self_assessment_path": sa_path if sa_rows else None,
            "self_assessment_latency": sa_latency}


def latency_model(probe_records, run_records) -> dict:
    """How long the belief takes to obtain, from calls that were NOT queued.

    Preference: a --latency-probe run (concurrency 1, warm server); else the
    main run's calls sent with nothing else in flight; else every call, which
    includes server queueing and is flagged as not a latency estimate.
    Latency of token k is modelled as ttft + (k - 1) * per-token interval,
    with medians over calls, so it can be applied to any record's length."""
    def calls(records):
        return list({r.get("call_key") or r.get("key"): r for r in records
                     if r.get("ttft_ms") is not None}.values())

    probe = calls(probe_records)
    solo = [r for r in calls(run_records) if r.get("in_flight_at_send") == 1]
    if probe:
        chosen, source, valid = probe, "latency probe (concurrency 1, warm)", True
    elif len(solo) >= 5:
        chosen, source, valid = solo, "main run, calls with in_flight_at_send == 1", True
    else:
        chosen, source = calls(run_records), "main run, ALL calls (includes server queueing)"
        valid = False
    if not chosen:
        return None
    ttft = [r["ttft_ms"] for r in chosen]
    per_token = [(r["gen_ms"] - r["ttft_ms"]) / (r["n_tokens"] - 1) for r in chosen
                 if r.get("gen_ms") is not None and (r.get("n_tokens") or 0) > 1]
    return {
        "source": source, "valid_latency_estimate": valid, "n_calls": len(chosen),
        "timing_synthetic": any(r.get("timing_synthetic") for r in chosen),
        "ttft_ms_median": float(np.median(ttft)), "ttft_ms_p90": float(np.percentile(ttft, 90)),
        "per_token_ms_median": float(np.median(per_token)) if per_token else 0.0,
        "gen_ms_median": float(np.median([r["gen_ms"] for r in chosen])),
    }


def label_maps(items, labels) -> dict:
    # Authored labels are null by design on d1_open (CONTRACTS.md): null items
    # are left out, and with none at all the "authored" label does not exist,
    # so every authored-label path (AUC, agreement, calibration) is skipped
    # and reported as not applicable. Never bool(None) -> False.
    authored = {i["id"]: bool(i["retrieval_needed_authored"]) for i in items
                if i.get("retrieval_needed_authored") is not None}
    out = {"authored": authored} if authored else {}
    if labels:
        out["empirical"] = {k: v["empirical_need"] for k, v in labels["items"].items()
                            if v.get("empirical_need") is not None}
    return out


def cluster_of(items) -> dict:
    return {i["id"]: (i.get("pair_id") or i["id"]) for i in items}


def condition_of(record):
    prefix = record["prefix"]
    if prefix["source"] == "text":
        return ("text", float(prefix["word_frac"]))
    return ("asr", prefix.get("voice_id"), prefix.get("noise"))


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------
def item_scores(records, signal) -> dict:
    """{item_id: mean over seeds of the signal}."""
    acc: dict = {}
    for record in records:
        value = record["_flat"].get(signal)
        if value is not None and math.isfinite(value):
            acc.setdefault(record["item_id"], []).append(value)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def item_auc(scores: dict, label_map: dict, clusters: dict, n_boot: int, with_ci=True) -> dict:
    ids = [i for i in scores if i in label_map]
    y = np.array([label_map[i] for i in ids], dtype=float)
    s = np.array([scores[i] for i in ids], dtype=float)
    out = {"n_items": len(ids), "n_pos": int(y.sum()) if len(ids) else 0,
           "auc": stats.auc(y, s) if len(ids) else None, "ci95": None}
    if with_ci and out["auc"] is not None and n_boot:
        out["ci95"] = stats.cluster_bootstrap(
            lambda idx: stats.auc(y[idx], s[idx]), [clusters.get(i, i) for i in ids], n_boot)
    return out


def per_seed_auc(records, signal, label_map) -> dict:
    by_seed: dict = {}
    for record in records:
        value = record["_flat"].get(signal)
        if value is None or record["item_id"] not in label_map:
            continue
        by_seed.setdefault(record["draw_seed"], ([], []))
        by_seed[record["draw_seed"]][0].append(label_map[record["item_id"]])
        by_seed[record["draw_seed"]][1].append(value)
    aucs = {str(s): stats.auc(y, v) for s, (y, v) in sorted(by_seed.items())}
    values = [a for a in aucs.values() if a is not None]
    return {"seed_aucs": aucs, "mean_auc": float(np.mean(values)) if values else None,
            "min_auc": min(values) if values else None, "max_auc": max(values) if values else None}


# ---------------------------------------------------------------------------
# Belief-at-dispatch on speech traces
# ---------------------------------------------------------------------------
def signal_latency_ms(record, signal, lat=None):
    """Time from request start until `signal` can be computed for this
    record: token min(k, n) for first-k entropy, the last token otherwise.
    With a latency model the time is predicted from unqueued calls; without
    one it is the record's own (possibly queued) measurement."""
    n = record.get("n_tokens") or 0
    if signal == SA_SIGNAL:  # its own one-token call
        return record.get("_sa_gen_ms")
    if signal.startswith("first_k_mean_entropy_nats@"):
        k = int(signal.split("@", 1)[1])
        if lat:
            return lat["ttft_ms_median"] + (max(1, min(k, n)) - 1) * lat["per_token_ms_median"]
        value = (record.get("t_first_k_ms") or {}).get(str(k))
        return value if value is not None else record.get("gen_ms")
    if lat:
        return lat["ttft_ms_median"] + max(0, n - 1) * lat["per_token_ms_median"]
    return record.get("gen_ms")


def asr_curves(records, signal, label_maps_, clusters, grid_ms, rel_grid, n_boot, lat=None) -> dict:
    """AUC of the belief available at time t, per trace condition."""
    out = {}
    asr = [r for r in records if r["prefix"]["source"] == "asr" and r["prefix"]["kind"] == "partial"]
    conditions = sorted({(r["prefix"].get("voice_id"), r["prefix"].get("noise")) for r in asr},
                        key=str)
    for voice, noise in conditions:
        rows = [r for r in asr if (r["prefix"].get("voice_id"), r["prefix"].get("noise"))
                == (voice, noise)]
        # per (item, seed): sorted list of (transcript_time, ready_time, value)
        series: dict = {}
        end_ms: dict = {}
        for r in rows:
            value = r["_flat"].get(signal)
            if value is None:
                continue
            avail = r["prefix"].get("available_ms")
            avail = r["prefix"].get("t_ms") if avail is None else avail
            latency = signal_latency_ms(r, signal, lat) or 0.0
            series.setdefault((r["item_id"], r["draw_seed"]), []).append(
                (avail, avail + latency, value))
            if r["prefix"].get("vad_user_end_ms") is not None:
                end_ms[r["item_id"]] = r["prefix"]["vad_user_end_ms"]
        if not series:
            continue
        horizon = max(t for pts in series.values() for _, t, _ in pts) + grid_ms
        abs_grid = list(np.arange(grid_ms, horizon + grid_ms, grid_ms))
        entry = {}
        for alignment, pos in (("transcript", 0), ("belief_ready", 1)):
            def belief_at(t_of_item):
                acc: dict = {}
                for (item_id, _seed), pts in series.items():
                    t = t_of_item(item_id)
                    if t is None:
                        continue
                    ready = [p for p in pts if p[pos] <= t]
                    if ready:
                        acc.setdefault(item_id, []).append(max(ready, key=lambda p: p[pos])[2])
                return {k: float(np.mean(v)) for k, v in acc.items()}

            curve_abs, curve_rel = [], []
            for t in abs_grid:
                scores = belief_at(lambda _i, t=t: t)
                point = {"t_ms": float(t), "n_items": len(scores)}
                for name, lm in label_maps_.items():
                    point[f"auc_{name}"] = item_auc(scores, lm, clusters, 0, with_ci=False)["auc"]
                curve_abs.append(point)
            for r_frac in rel_grid:
                scores = belief_at(lambda i, r=r_frac: (end_ms.get(i) * r) if end_ms.get(i) else None)
                point = {"rel_turn": float(r_frac), "n_items": len(scores)}
                for name, lm in label_maps_.items():
                    res = item_auc(scores, lm, clusters, n_boot)
                    point[f"auc_{name}"] = res["auc"]
                    point[f"ci95_{name}"] = res["ci95"]
                curve_rel.append(point)
            entry[alignment] = {"absolute": curve_abs, "relative_to_turn_end": curve_rel}
        out[f"{voice}|{noise}"] = entry
    return out


def asr_word_frac_curve(records, signal, label_maps_, clusters, grid, n_boot, c2_threshold) -> dict:
    """PREREG_V2 H-E1v2 prefix curve: AUC at word_frac in {0.3, 0.5, 0.7, 1.0}
    on ASR partials, per trace condition. The belief at fraction f is the
    ASR partial (or the final transcript, f = 1.0) with the largest spoken-
    word fraction <= f. Adds the paired cluster bootstrap of
    AUC(max f) - AUC(min f) > 0 and the C2-availability flag AUC(0.5) >= 0.65."""
    out = {}
    asr = [r for r in records if r["prefix"]["source"] == "asr"
           and r["prefix"].get("word_frac") is not None]
    conditions = sorted({(r["prefix"].get("voice_id"), r["prefix"].get("noise")) for r in asr}, key=str)
    for voice, noise in conditions:
        rows = [r for r in asr if (r["prefix"].get("voice_id"), r["prefix"].get("noise")) == (voice, noise)]
        scores_at = {}
        for f in grid:
            best: dict = {}
            for r in rows:
                value = r["_flat"].get(signal)
                wf = r["prefix"]["word_frac"]
                if value is None or wf > f + 1e-9:
                    continue
                key = (r["item_id"], r["draw_seed"])
                if key not in best or wf > best[key][0]:
                    best[key] = (wf, value)
            acc: dict = {}
            for (item_id, _seed), (_wf, value) in best.items():
                acc.setdefault(item_id, []).append(value)
            scores_at[f] = {k: float(np.mean(v)) for k, v in acc.items()}
        entry = {"curve": [], "paired_diff": {}, "c2_available": {}}
        for f in grid:
            point = {"word_frac": f, "n_items": len(scores_at[f])}
            for name, lm in label_maps_.items():
                point[name] = item_auc(scores_at[f], lm, clusters, n_boot)
            entry["curve"].append(point)
        lo_f, hi_f = min(grid), max(grid)
        for name, lm in label_maps_.items():
            ids = [i for i in scores_at[hi_f] if i in scores_at[lo_f] and i in lm]
            y = np.array([lm[i] for i in ids], float)
            hi = np.array([scores_at[hi_f][i] for i in ids])
            lo = np.array([scores_at[lo_f][i] for i in ids])

            def diff(idx):
                a, b = stats.auc(y[idx], hi[idx]), stats.auc(y[idx], lo[idx])
                return None if a is None or b is None else a - b

            point = diff(np.arange(len(ids))) if ids else None
            boot = []
            if ids and n_boot:
                rng = np.random.default_rng(1)
                cl = np.array([clusters.get(i, i) for i in ids])
                uniq = np.unique(cl)
                members = {c: np.flatnonzero(cl == c) for c in uniq}
                for _ in range(int(n_boot)):
                    idx = np.concatenate([members[c] for c in rng.choice(uniq, len(uniq))])
                    d = diff(idx)
                    if d is not None:
                        boot.append(d)
            entry["paired_diff"][name] = {
                "from": lo_f, "to": hi_f, "n_items": len(ids), "auc_diff": point,
                "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
                "p_one_sided_diff_le_0": float(np.mean(np.array(boot) <= 0)) if boot else None,
            }
            mid = next((p[name]["auc"] for p in entry["curve"] if abs(p["word_frac"] - 0.5) < 1e-9), None)
            entry["c2_available"][name] = None if mid is None else bool(mid >= c2_threshold)
        out[f"{voice}|{noise}"] = entry
    return out


def boundary_mass(beliefs, b_star, half_width=0.15):
    """PREREG_V2: fraction of beliefs within +-0.15 of b* = g-/(g+ + g-)."""
    if b_star is None or beliefs is None or len(beliefs) == 0:
        return None
    beliefs = np.asarray(beliefs, float)
    return {"b_star": b_star, "half_width": half_width, "n": int(beliefs.size),
            "fraction": float(np.mean(np.abs(beliefs - b_star) <= half_width)),
            "prediction": ">= 0.15 (v1 had 0.08)"}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_block(records, signal, label_map, clusters, cfg_a) -> dict | None:
    full = [r for r in records if condition_of(r) == FULL and r["item_id"] in label_map
            and r["_flat"].get(signal) is not None]
    if len(full) < 10:
        return None
    x = np.array([r["_flat"][signal] for r in full])
    y = np.array([float(label_map[r["item_id"]]) for r in full])
    if len(np.unique(y)) < 2:
        return None
    groups = [clusters.get(r["item_id"], r["item_id"]) for r in full]
    cv, oof = stats.cross_validate(x, y, groups, cfg_a["cv_folds"], cfg_a["ece_bins"])
    slope, intercept, info = stats.fit_logistic_safe(x, y)
    iso = stats.fit_isotonic(x, y)
    p_full = stats.sigmoid(slope * x + intercept)
    transfer = {}
    for frac in sorted({condition_of(r)[1] for r in records if condition_of(r)[0] == "text"}):
        rows = [r for r in records if condition_of(r) == ("text", frac)
                and r["item_id"] in label_map and r["_flat"].get(signal) is not None]
        if not rows:
            continue
        xs = np.array([r["_flat"][signal] for r in rows])
        ys = np.array([float(label_map[r["item_id"]]) for r in rows])
        ps = stats.sigmoid(slope * xs + intercept)
        transfer[str(frac)] = {"n": len(rows), "ece_of_full_fit": stats.ece(ys, ps, cfg_a["ece_bins"]),
                               "mean_predicted": float(ps.mean()), "base_rate": float(ys.mean())}
    return {
        "signal": signal, "n_samples": len(full), "n_items": len({r["item_id"] for r in full}),
        "n_seeds": len({r["draw_seed"] for r in full}), "base_rate": float(y.mean()),
        "logistic": {"slope": slope, "intercept": intercept, "fit": info,
                     "in_sample_auc": stats.auc(y, p_full),
                     "decision_boundary": float(-intercept / slope) if slope else None},
        "isotonic": {"thresholds": iso[0].tolist(), "values": iso[1].tolist()},
        "cross_validation": cv,
        "transfer_to_prefixes": transfer,
        # Beliefs for the boundary-mass check; popped before writing JSON.
        "_beliefs_oof": [float(v) for v in oof["logistic"] if not np.isnan(v)],
        "_beliefs_in_sample": [float(v) for v in p_full],
    }


def _prefix_frac(record):
    prefix = record["prefix"]
    if prefix["kind"] == "full":
        return 1.0
    if prefix["source"] == "text":
        return prefix.get("actual_word_frac", prefix.get("word_frac"))
    return prefix.get("word_frac")


def prefix_calibration_block(records, signal, label_map, clusters, cfg_a) -> dict | None:
    """Prefix-aware calibration for the v2 controllers
    (controller/baselines/_objective_v2.py:BeliefCalibration):
        P(need | x, f) = sigmoid(intercept + slope*x + frac_slope*f + interaction*x*f)
    fitted on dev over every prefix condition of one source: ASR partials and
    final transcripts when traces exist (f = spoken-word fraction at the
    partial's t_ms), else the text prefixes (f = prefix word fraction).
    Grouped k-fold CV gives out-of-fold ECE overall and per f band."""
    source = "asr" if any(r["prefix"]["source"] == "asr" for r in records) else "text"
    rows = [r for r in records if r["prefix"]["source"] == source and r["item_id"] in label_map
            and r["_flat"].get(signal) is not None and _prefix_frac(r) is not None]
    if len(rows) < 20:
        return None
    x = np.array([r["_flat"][signal] for r in rows])
    f = np.array([float(_prefix_frac(r)) for r in rows])
    y = np.array([float(label_map[r["item_id"]]) for r in rows])
    if len(np.unique(y)) < 2:
        return None
    groups = [clusters.get(r["item_id"], r["item_id"]) for r in rows]
    folds = stats.group_folds(groups, cfg_a["cv_folds"])
    oof = np.full(len(rows), np.nan)
    for k in range(cfg_a["cv_folds"]):
        test = folds == k
        if not test.any() or len(np.unique(y[~test])) < 2:
            continue
        model = stats.fit_prefix_logistic(x[~test], f[~test], y[~test])
        oof[test] = stats.predict_prefix_logistic(model, x[test], f[test])
    mask = ~np.isnan(oof)
    bands = []
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)):
        m = mask & (f >= lo) & (f < hi)
        bands.append({"f_lo": lo, "f_hi": min(hi, 1.0), "n": int(m.sum()),
                      "ece": stats.ece(y[m], oof[m], cfg_a["ece_bins"]) if m.any() else None,
                      "auc": stats.auc(y[m], oof[m]) if m.any() else None})
    model = stats.fit_prefix_logistic(x, f, y)
    return {
        "signal": signal, "source": source, "n_samples": len(rows),
        "n_items": len({r["item_id"] for r in rows}), "n_seeds": len({r["draw_seed"] for r in rows}),
        "base_rate": float(y.mean()), "model": model,
        "cv": {"n": int(mask.sum()),
               "ece": stats.ece(y[mask], oof[mask], cfg_a["ece_bins"]) if mask.any() else None,
               "auc": stats.auc(y[mask], oof[mask]) if mask.any() else None,
               "brier": stats.brier(y[mask], oof[mask]) if mask.any() else None,
               "by_frac_band": bands},
        "frac_definition": ("spoken-word fraction of the trace's reference word timings at the "
                            "partial's t_ms (1.0 for the final transcript)" if source == "asr" else
                            "fraction of question words in the text prefix (1.0 for the full text)"),
    }


def write_prefix_artifact(path, pcal, model_info, label_kind, inputs, dataset, split) -> dict:
    m = pcal["model"]
    artifact = {
        "_comment": (
            "GENERATED by `python -m llm.experiment1_v2.analyze` - do not hand-edit. E1-v2 "
            "PREFIX-AWARE calibration for the v2 controllers: P(retrieval needed | signal x on "
            "the prefix, word fraction f) = sigmoid(intercept + slope*x + frac_slope*f + "
            "interaction*x*f), fitted on the DEV split only. Read by "
            "controller/baselines/_objective_v2.py:load_belief_calibration() via "
            "controller.e3v2.calibration.from_artifact."),
        "fitted_at_utc": common.now_utc(),
        "provenance": {
            "experiment_id": "e1-v2-signal-validity", "git_commit": common.git_commit(),
            "dataset": dataset, "split": split, "backend": model_info["backend"],
            "model": model_info["model"], "model_key": model_info["model_key"],
            "fitted_signal": pcal["signal"], "label": label_kind, "prefix_source": pcal["source"],
            "frac_definition": pcal["frac_definition"], "inputs": inputs,
            "n_samples": pcal["n_samples"], "n_items": pcal["n_items"], "n_seeds": pcal["n_seeds"],
            "base_rate": pcal["base_rate"], "fit_method": f"IRLS, {m['method']}",
            "in_sample_only": False,
            "evaluation": "grouped k-fold CV on dev; test evaluation pending (sealed)",
        },
        "diagnostics": {"cv": pcal["cv"]},
        "calibration": {
            "signal": pcal["signal"], "kind": "logistic",
            "intercept": m["intercept"], "slope": m["slope"],
            "frac_slope": m["frac_slope"], "interaction": m["interaction"],
            "calibrated": True, "auc": pcal["cv"]["auc"],
            "source": (f"E1-v2 dev prefix-aware fit on {dataset}, {model_info['model']}: "
                       f"sigmoid({m['intercept']:+.4f} {m['slope']:+.4f}*x {m['frac_slope']:+.4f}*f "
                       f"{m['interaction']:+.4f}*x*f), x={pcal['signal']}, {pcal['source']} prefixes, "
                       f"n={pcal['n_samples']}; CV AUC {pcal['cv']['auc']}, ECE {pcal['cv']['ece']}"),
        },
    }
    common.atomic_write_json(path, artifact)
    return artifact


def write_artifact(path, cal, model_info, label_kind, inputs, dataset, split) -> dict:
    logistic = cal["logistic"]
    cv = cal["cross_validation"].get("logistic") or {}
    artifact = {
        "_comment": (
            "GENERATED by `python -m llm.experiment1_v2.analyze` - do not hand-edit. E1-v2 "
            "calibration: P(retrieval is needed) from one uncertainty signal on the FULL "
            "question, fitted on the DEV split only. Same 'calibration' block as "
            "controller/baselines/e1_calibration_mean_entropy.json, so "
            "controller/baselines/_objective.py:load_voi_calibration() can load it via "
            "from_artifact. The 'isotonic' block is extra and ignored by that loader."
        ),
        "fitted_at_utc": common.now_utc(),
        "provenance": {
            "experiment_id": "e1-v2-signal-validity",
            "git_commit": common.git_commit(),
            "dataset": dataset,
            "split": split,
            "backend": model_info["backend"],
            "model": model_info["model"],
            "model_key": model_info["model_key"],
            "fitted_signal": cal["signal"],
            "label": label_kind,
            "prefix_condition": "full question text",
            "inputs": inputs,
            "n_samples": cal["n_samples"],
            "n_items": cal["n_items"],
            "n_seeds": cal["n_seeds"],
            "base_rate": cal["base_rate"],
            "clustering_note": (
                f"{cal['n_samples']} samples are {cal['n_items']} items x {cal['n_seeds']} seeds; "
                "clustered, so the effective sample size is nearer the item count. Cross-"
                "validation folds are grouped by pair_id/item so no item or pair straddles folds."),
            "fit_method": f"1-D logistic, {logistic['fit']['method']} "
                          "(llm/experiment1_v2/stats.py:fit_logistic_safe)",
            "in_sample_only": False,
            "evaluation": "grouped k-fold cross-validation on dev (diagnostics.cv_*); test "
                          "evaluation pending (sealed until PREREG_FROZEN)",
        },
        "diagnostics": {
            "auc_in_sample": logistic["in_sample_auc"],
            "decision_boundary": logistic["decision_boundary"],
            "cv_logistic": {k: v for k, v in cv.items() if k != "reliability"},
            "cv_isotonic": {k: v for k, v in (cal["cross_validation"].get("isotonic") or {}).items()
                            if k != "reliability"},
            "transfer_to_prefixes": cal["transfer_to_prefixes"],
        },
        "calibration": {
            "signal": cal["signal"],
            "kind": "logistic",
            "slope": logistic["slope"],
            "intercept": logistic["intercept"],
            "calibrated": True,
            "auc": cv.get("auc", logistic["in_sample_auc"]),
            "source": (
                f"E1-v2 dev fit on {dataset}, {model_info['model']} ({model_info['backend']}): "
                f"P(retrieval_needed[{label_kind}]) = sigmoid({logistic['slope']:.4f} * "
                f"{cal['signal']} {logistic['intercept']:+.4f}) on n={cal['n_samples']} draws; "
                f"cross-validated AUC {cv.get('auc') if cv.get('auc') is None else round(cv['auc'], 4)}, "
                f"ECE {cv.get('ece') if cv.get('ece') is None else round(cv['ece'], 4)}. "
                f"Regenerate: python -m llm.experiment1_v2.analyze --dataset {dataset} --split {split}"
            ),
        },
        "isotonic": {"signal": cal["signal"], **cal["isotonic"]},
    }
    common.atomic_write_json(path, artifact)
    return artifact


# ---------------------------------------------------------------------------
# Gains and leak baseline
# ---------------------------------------------------------------------------
def gains(labels, clusters, n_boot) -> dict | None:
    if not labels:
        return None
    rows = [(k, v) for k, v in labels["items"].items() if v.get("gain") is not None]
    if not rows:
        return None
    out = {}
    for kind in ("authored", "empirical"):
        need = [(k, v["gain"], v["label_authored"] if kind == "authored" else v["empirical_need"])
                for k, v in rows]
        need = [t for t in need if t[2] is not None]
        if not need:
            continue
        ids = [k for k, _, _ in need]
        g = np.array([x for _, x, _ in need], float)
        n = np.array([bool(z) for _, _, z in need])
        cl = [clusters.get(i, i) for i in ids]

        def g_plus(idx):
            m = n[idx]
            return float(g[idx][m].mean()) if m.any() else None

        def g_minus(idx):
            m = ~n[idx]
            return float(-g[idx][~n[idx]].mean()) if m.any() else None

        everything = np.arange(len(ids))
        gp, gm = g_plus(everything), g_minus(everything)
        out[kind] = {
            "n_needed": int(n.sum()), "n_not_needed": int((~n).sum()),
            "g_plus": gp, "g_plus_ci95": stats.cluster_bootstrap(g_plus, cl, n_boot),
            "g_minus": gm, "g_minus_ci95": stats.cluster_bootstrap(g_minus, cl, n_boot),
            "base_rate_b": float(n.mean()),
            # Defined only when retrieval both helps where needed and hurts
            # where not; otherwise "always retrieve" or "never" dominates.
            "break_even_belief": (gm / (gp + gm)) if (gp is not None and gm is not None
                                                       and gp > 0 and gm > 0) else None,
            "Q_at_base_rate": (n.mean() * gp - (1 - n.mean()) * gm) if (gp is not None and gm
                                                                         is not None) else None,
        }
    return out


def leak_baseline(path, label_maps_, items_dev) -> dict | None:
    if path is None:
        for candidate in LEAK_CANDIDATES:
            if (common.REPO_ROOT / candidate).exists():
                path = common.REPO_ROOT / candidate
                break
    if path is None or not Path(path).exists():
        return None
    data = common.load_json(path)
    out = {"path": common.rel(path), "sha256": common.sha256_file(path)}
    scores = None
    if isinstance(data, dict) and isinstance(data.get("scores"), dict):
        scores = data["scores"]
    elif isinstance(data, list) and data and isinstance(data[0], dict) and "score" in data[0]:
        scores = {d.get("item_id") or d.get("id"): d["score"] for d in data}
    if scores:
        dev = {i["id"] for i in items_dev}
        s = {k: float(v) for k, v in scores.items() if k in dev}
        out["dev_auc"] = {name: stats.auc([lm[k] for k in s if k in lm], [s[k] for k in s if k in lm])
                          for name, lm in label_maps_.items()}
        out["n_dev_items_scored"] = len(s)
    elif isinstance(data, dict):
        out["reported"] = {k: v for k, v in data.items()
                           if "auc" in k.lower() and isinstance(v, (int, float))}
        if not out["reported"]:
            out["unparsed_keys"] = sorted(data)[:50]
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
INK, MUTED = "#0b0b0b", "#52514e"


def _style(plt):
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.markersize": 7, "savefig.dpi": 160,
    })


def make_figures(analysis: dict, out_dir: Path, label_kind: str) -> list:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    _style(plt)
    written = []
    models = list(analysis["models"])
    colors = {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}
    signals = analysis["headline_signals"]

    # 1. AUC vs word fraction, one panel per headline signal, one line per model.
    fig, axes = plt.subplots(1, len(signals), figsize=(4.6 * len(signals), 4.0), sharey=True,
                             squeeze=False)
    drawn = False
    for ax, signal in zip(axes[0], signals):
        for m in models:
            curve = analysis["models"][m]["prefix_curve_text"].get(signal) or []
            pts = [(c["word_frac"], c[label_kind]) for c in curve if c.get(label_kind)
                   and c[label_kind]["auc"] is not None]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1]["auc"] for p in pts]
            lo = [p[1]["ci95"][0] if p[1]["ci95"] else p[1]["auc"] for p in pts]
            hi = [p[1]["ci95"][1] if p[1]["ci95"] else p[1]["auc"] for p in pts]
            ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.15, linewidth=0)
            ax.plot(xs, ys, marker="o", color=colors[m], label=m)
            drawn = True
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlabel(f"Words heard (fraction)\n{signal}")
        ax.set_xticks([0.3, 0.5, 0.7, 1.0])
    axes[0][0].set_ylabel(f"AUC vs {label_kind} need")
    if drawn:
        if len(models) > 1:
            axes[0][-1].legend(frameon=False, loc="lower right")
        path = out_dir / f"fig_auc_vs_word_fraction_{label_kind}.png"
        fig.tight_layout()
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    # 2. AUC vs time relative to turn end, transcript vs belief-ready.
    signal = signals[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    drawn = False
    for ax, alignment, xlabel in (
        (axes[0], "transcript", "Time / user turn end\n(latest ASR partial)"),
        (axes[1], "belief_ready", "Time / user turn end\n(belief computed by then)"),
    ):
        for m in models:
            for cond, entry in (analysis["models"][m]["asr_curves"].get(signal) or {}).items():
                pts = [p for p in entry[alignment]["relative_to_turn_end"]
                       if p.get(f"auc_{label_kind}") is not None]
                if pts:
                    label = m if len(analysis["models"][m]["asr_curves"][signal]) == 1 else f"{m} {cond}"
                    ax.plot([p["rel_turn"] for p in pts], [p[f"auc_{label_kind}"] for p in pts],
                            marker="o", color=colors[m], label=label)
                    for p in pts:  # items with a belief by then: early points rest on few
                        ax.annotate(f"n={p['n_items']}", (p["rel_turn"], p[f"auc_{label_kind}"]),
                                    textcoords="offset points", xytext=(0, -15), ha="center",
                                    fontsize=9, color=MUTED)
                    drawn = True
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(f"AUC vs {label_kind} need")
    fig.text(0.01, 0.01, signal, fontsize=10, color=MUTED)
    if drawn:
        if len(models) > 1:
            axes[1].legend(frameon=False, loc="lower right")
        path = out_dir / f"fig_auc_vs_asr_time_{label_kind}.png"
        fig.tight_layout()
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    # 3. AUC vs measured latency of the first-k signal.
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    drawn = False
    for m in models:
        rows = [r for r in analysis["models"][m]["first_k"] if r.get(label_kind)
                and r[label_kind]["auc"] is not None and r["latency_ms"]["median"] is not None]
        if not rows:
            continue
        xs = [r["latency_ms"]["median"] for r in rows]
        ys = [r[label_kind]["auc"] for r in rows]
        ax.plot(xs, ys, marker="o", color=colors[m], label=m)
        for r, x, y in zip(rows, xs, ys):
            ax.annotate(f"k={r['k']}", (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=10, color=INK)
        drawn = True
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("Median time to signal (ms from request)")
    ax.set_ylabel(f"AUC vs {label_kind} need")
    if drawn:
        if len(models) > 1:
            ax.legend(frameon=False, loc="lower right")
        path = out_dir / f"fig_auc_vs_first_k_latency_{label_kind}.png"
        fig.tight_layout()
        fig.savefig(path)
        written.append(path)
    plt.close(fig)

    # 4. Reliability diagrams (out-of-fold on dev), one per model.
    for m in models:
        cal = analysis["models"][m].get("calibration")
        if not cal:
            continue
        fig, ax = plt.subplots(figsize=(4.8, 4.6))
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
        for color, kind in ((SERIES[0], "logistic"), (SERIES[1], "isotonic")):
            block = cal["cross_validation"].get(kind)
            if not block:
                continue
            pts = [b for b in block["reliability"] if b["n"]]
            ax.plot([b["mean_predicted"] for b in pts], [b["observed_rate"] for b in pts],
                    marker="o", color=color, label=f"{kind} (ECE {block['ece']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Predicted P(need) [{cal['signal']}]")
        ax.set_ylabel("Observed need rate")
        ax.legend(frameon=False, loc="upper left")
        path = out_dir / f"fig_reliability_{m}.png"
        fig.tight_layout()
        fig.savefig(path)
        written.append(path)
        plt.close(fig)

    # 5. g+ and g- per model.
    gain_rows = [(m, analysis["models"][m]["gains"][label_kind]) for m in models
                 if (analysis["models"][m].get("gains") or {}).get(label_kind)]
    if gain_rows:
        fig, ax = plt.subplots(figsize=(1.6 + 1.8 * len(gain_rows), 4.0))
        width = 0.36
        for j, (name, key) in enumerate((("g+ (needed)", "g_plus"), ("g- (not needed)", "g_minus"))):
            xs = np.arange(len(gain_rows)) + (j - 0.5) * width
            vals = [r[key] if r[key] is not None else 0.0 for _, r in gain_rows]
            err = [[v - (r[f"{key}_ci95"][0] if r[f"{key}_ci95"] else v) for v, (_, r) in
                    zip(vals, gain_rows)],
                   [(r[f"{key}_ci95"][1] if r[f"{key}_ci95"] else v) - v for v, (_, r) in
                    zip(vals, gain_rows)]]
            ax.bar(xs, vals, width=width - 0.04, color=SERIES[j], label=name, yerr=err,
                   ecolor=INK, capsize=3)
        ax.set_xticks(np.arange(len(gain_rows)))
        ax.set_xticklabels([m for m, _ in gain_rows])
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_ylabel(f"Accuracy change ({label_kind} need)")
        ax.legend(frameon=False)
        path = out_dir / f"fig_gain_{label_kind}.png"
        fig.tight_layout()
        fig.savefig(path)
        written.append(path)
        plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze_model(inp, items, cfg_a, headline, n_boot) -> dict:
    records = inp["records"]
    lmaps = label_maps(items, inp["labels"])
    clusters = cluster_of(items)
    all_signals = sorted({k for r in records for k in r["_flat"]})
    full = [r for r in records if condition_of(r) == FULL]
    result = {
        "model": records[0]["model"], "backend": records[0]["backend"],
        "model_key": records[0]["model_key"],
        "n_records": len(records), "n_items": len({r["item_id"] for r in records}),
        "seeds": sorted({r["draw_seed"] for r in records}),
        "labels_available": sorted(lmaps),
        "timing_synthetic": any(r.get("timing_synthetic") for r in records),
        "auc_full": {}, "prefix_curve_text": {}, "asr_curves": {}, "first_k": [],
    }
    lat_model = inp.get("latency")
    # Only an unqueued estimate may stand in for "time to obtain the belief";
    # otherwise each record's own measurement is used and flagged.
    result["latency_model"] = lat_model
    if lat_model and not lat_model["valid_latency_estimate"]:
        print(f"[analyze] WARNING {result['model_key']}: no latency probe and too few unqueued "
              "calls; belief-ready timings include server queueing. Run run.py --latency-probe.")
        lat_model = None
    if "authored" not in lmaps:
        result["authored_vs_empirical"] = "not applicable (dataset has no authored labels)"
    if "empirical" in lmaps and "authored" in lmaps:
        both = [i for i in lmaps["empirical"] if i in lmaps["authored"]]
        from simulator.judge_validation.score_judges import agreement  # noqa: PLC0415
        result["authored_vs_empirical"] = agreement(
            [lmaps["authored"][i] for i in both], [lmaps["empirical"][i] for i in both])
    if "empirical" in lmaps:
        result["empirical_base_rate"] = (float(np.mean(list(lmaps["empirical"].values())))
                                         if lmaps["empirical"] else None)

    for signal in all_signals:
        entry = {}
        for name, lm in lmaps.items():
            entry[name] = {
                **item_auc(item_scores(full, signal), lm, clusters, n_boot if signal in headline else 0,
                           with_ci=signal in headline),
                "per_seed": per_seed_auc(full, signal, lm),
            }
        result["auc_full"][signal] = entry

    fracs = sorted({condition_of(r)[1] for r in records if condition_of(r)[0] == "text"})
    for signal in headline:
        curve = []
        for frac in fracs:
            rows = [r for r in records if condition_of(r) == ("text", frac)]
            point = {"word_frac": frac}
            for name, lm in lmaps.items():
                point[name] = item_auc(item_scores(rows, signal), lm, clusters, n_boot)
            curve.append(point)
        result["prefix_curve_text"][signal] = curve
        result.setdefault("asr_word_frac_curve", {})[signal] = asr_word_frac_curve(
            records, signal, lmaps, clusters, cfg_a.get("asr_word_frac_grid", [0.3, 0.5, 0.7, 1.0]),
            n_boot, cfg_a.get("c2_auc_threshold", 0.65))
        result["asr_curves"][signal] = asr_curves(
            records, signal, lmaps, clusters, cfg_a["asr_time_grid_ms"],
            cfg_a["asr_relative_grid"], n_boot, lat_model)

    ks = sorted({int(k.split("@")[1]) for k in all_signals if k.startswith("first_k_mean_entropy_nats@")})
    for k in ks + ["all"]:
        signal = "mean_entropy_nats" if k == "all" else f"first_k_mean_entropy_nats@{k}"
        lat = [signal_latency_ms(r, signal, lat_model) for r in full]
        lat = [x for x in lat if x is not None]
        row = {"k": k, "signal": signal,
               "latency_ms": {"median": float(np.median(lat)) if lat else None,
                              "p90": float(np.percentile(lat, 90)) if lat else None},
               "latency_source": (lat_model or {}).get("source")}
        for name, lm in lmaps.items():
            row[name] = item_auc(item_scores(full, signal), lm, clusters, n_boot)
        result["first_k"].append(row)
    if SA_SIGNAL in all_signals:
        # Self-assessment is its own one-token call, not a prefix of the answer.
        lat = [r["_sa_gen_ms"] for r in full if r.get("_sa_gen_ms") is not None]
        sa_lat = inp.get("self_assessment_latency") or {}
        row = {"k": "self", "signal": SA_SIGNAL,
               "latency_ms": {"median": float(np.median(lat)) if lat else None,
                              "p90": float(np.percentile(lat, 90)) if lat else None},
               "latency_source": ("self-assessment calls, " + sa_lat.get("source", "")
                                  + ("" if sa_lat.get("valid_latency_estimate") else
                                     " (includes queueing: not a latency estimate)"))}
        for name, lm in lmaps.items():
            row[name] = item_auc(item_scores(full, SA_SIGNAL), lm, clusters, n_boot)
        result["first_k"].append(row)
        result["self_assessment_latency"] = sa_lat or None
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--items", type=Path)
    p.add_argument("--splits", type=Path)
    p.add_argument("--split", required=True, choices=common.VALID_SPLITS)
    p.add_argument("--models", help="comma list of model keys (default: every model with results)")
    p.add_argument("--llm-backend", choices=("vllm", "mock"), default="vllm")
    p.add_argument("--results-root", type=Path, help="override results/<dataset>/")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    p.add_argument("--bootstrap", type=int)
    p.add_argument("--calibration-signal")
    p.add_argument("--calibration-label", choices=("empirical", "authored"))
    p.add_argument("--leak-results", type=Path)
    p.add_argument("--labels", help="labels file name inside each model dir (default labels_<split>.json; e.g. labels_dev.e1v2-r2.json)")
    p.add_argument("--controller-artifact", action="store_true",
                   help="also write each calibration artifact to controller/baselines/ "
                   "(refused for mock-backend results)")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        common.check_split_allowed(args.split)
    except common.SealedSplitError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    cfg = common.load_config(args.config)
    cfg_a = dict(cfg["analysis"])
    n_boot = args.bootstrap if args.bootstrap is not None else cfg_a["bootstrap"]
    paths = common.resolve_dataset_paths(args.dataset, args.items, args.splits)
    items = common.select_split(common.load_items(paths["items"]),
                                common.load_splits(paths["splits"]), args.split)
    dataset = args.dataset or items[0]["dataset"]
    root = args.results_root or ((common.MOCK_RESULTS_ROOT if args.llm_backend == "mock"
                                  else common.RESULTS_ROOT) / dataset)
    model_keys = common.parse_csv_list(args.models) or sorted(
        d.name for d in root.iterdir() if (d / f"signals_{args.split}.jsonl").exists()) \
        if root.exists() else []
    if not model_keys:
        print(f"no results under {common.rel(root)}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or root / f"analysis_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    headline = cfg_a["headline_signals"]
    cal_signals = common.parse_csv_list(args.calibration_signal) or cfg_a.get(
        "calibration_signals") or [cfg_a["calibration_signal"]]
    cal_label = args.calibration_label or cfg_a["calibration_label"]

    analysis = {"dataset": dataset, "split": args.split, "created_at_utc": common.now_utc(),
                "git_commit": common.git_commit(), "headline_signals": headline,
                "bootstrap": n_boot, "models": {}, "artifacts": {}}
    lmaps_any = None
    for key in model_keys:
        inp = load_model_inputs(root / key, args.split, args.labels)
        if not inp["records"]:
            print(f"[analyze] {key}: no closed-book records, skipped")
            continue
        unknown = {r["item_id"] for r in inp["records"]} - {i["id"] for i in items}
        if unknown:
            raise ValueError(f"{key}: records for {len(unknown)} items outside the {args.split} split")
        result = analyze_model(inp, items, cfg_a, headline, n_boot)
        lmaps = label_maps(items, inp["labels"])
        lmaps_any = lmaps_any or lmaps
        # Requested label if present; else empirical; authored only as a last
        # resort (before labelling) and never when it does not exist (d1_open).
        label_kind = next((k for k in (cal_label, "empirical", "authored") if k in lmaps), None)
        result["calibration_label"] = label_kind
        result["labels_not_applicable"] = sorted({"authored", "empirical"} - set(lmaps))
        if label_kind is None:
            print(f"[analyze] {key}: no labels available; calibration skipped.")
        result["gains"] = gains(inp["labels"], cluster_of(items), n_boot)
        b_star = ((result["gains"] or {}).get(label_kind) or {}).get("break_even_belief")
        result["calibrations"] = {}
        inputs = {
            "signals": common.rel(inp["signals_path"]),
            "signals_sha256": common.sha256_file(inp["signals_path"]),
            "labels": common.rel(inp["labels_path"]) if inp["labels_path"] else None,
            "labels_sha256": common.sha256_file(inp["labels_path"]) if inp["labels_path"] else None,
            "items": common.rel(paths["items"]),
            "items_sha256": common.sha256_file(paths["items"]),
            "splits": common.rel(paths["splits"]),
            "splits_sha256": common.sha256_file(paths["splits"]),
        }
        if inp.get("self_assessment_path"):
            inputs["self_assessment"] = common.rel(inp["self_assessment_path"])
            inputs["self_assessment_sha256"] = common.sha256_file(inp["self_assessment_path"])
        # (artifact name, writer, block, effective slopes that must be > 0)
        candidates = []
        for signal in (cal_signals if label_kind else []):
            cal = calibration_block(inp["records"], signal, lmaps[label_kind], cluster_of(items), cfg_a)
            pcal = prefix_calibration_block(inp["records"], signal, lmaps[label_kind],
                                            cluster_of(items), cfg_a)
            bmass = None
            if cal:
                bmass = {"label": label_kind,
                         "dev_out_of_fold": boundary_mass(cal.pop("_beliefs_oof"), b_star),
                         "dev_in_sample": boundary_mass(cal.pop("_beliefs_in_sample"), b_star)}
                candidates.append((f"calibration_{dataset}_{key}_{signal}.json", write_artifact, cal,
                                   [cal["logistic"]["slope"]]))
            if pcal:
                m = pcal["model"]
                candidates.append((f"calibration_prefix_{dataset}_{key}_{signal}.json",
                                   write_prefix_artifact, pcal,
                                   [m["slope"], m["slope"] + m["interaction"]]))  # d logit/dx, f=0 and 1
            result["calibrations"][signal] = {"full": cal, "prefix": pcal, "boundary_mass": bmass}
        first = result["calibrations"].get(cal_signals[0], {})
        result["calibration"] = first.get("full")          # first signal, for the figures
        result["prefix_calibration"] = first.get("prefix")
        result["boundary_mass"] = first.get("boundary_mass")
        analysis["models"][key] = result
        analysis["artifacts"][key] = []
        for name, writer, block, slopes in candidates:
            if min(slopes) <= 0:
                print(f"[analyze] {key}: {name}: signal slope {min(slopes):+.4f} <= 0; refusing to "
                      "write it (it would invert the belief).")
                continue
            path = out_dir / name
            writer(path, block, result, label_kind, inputs, dataset, args.split)
            analysis["artifacts"][key].append(common.rel(path))
            if args.controller_artifact:
                if result["backend"] == "mock":
                    print("[analyze] REFUSED --controller-artifact for mock-backend results.")
                else:
                    cpath = common.REPO_ROOT / "controller" / "baselines" / f"e1v2_{name}"
                    writer(cpath, block, result, label_kind, inputs, dataset, args.split)
                    analysis["artifacts"][key].append(common.rel(cpath))
    if not analysis["models"]:
        return 1
    analysis["leak_baseline"] = leak_baseline(args.leak_results, lmaps_any, items)
    figures = []
    if not args.no_figures:
        for kind in ("empirical", "authored"):
            if any(kind in m["labels_available"] for m in analysis["models"].values()):
                figures += make_figures(analysis, out_dir, kind)
    analysis["figures"] = [common.rel(f) for f in figures]
    out_path = out_dir / f"analysis_{args.split}.json"
    common.atomic_write_json(out_path, analysis)

    for key, m in analysis["models"].items():
        print(f"[analyze] {key} ({m['backend']}): {m['n_items']} items, seeds {m['seeds']}")
        for signal in headline:
            row = m["auc_full"].get(signal, {})
            cells = []
            for kind in ("authored", "empirical"):
                if kind in row and row[kind]["auc"] is not None:
                    ci = row[kind]["ci95"]
                    cells.append(f"{kind} {row[kind]['auc']:.3f}"
                                 + (f" [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
            print(f"    {signal:<34} " + "   ".join(cells))
        for signal, cals in m.get("calibrations", {}).items():
            full = cals["full"]
            if full and full["cross_validation"].get("logistic"):
                cvl = full["cross_validation"]["logistic"]
                print(f"    calibration ({m['calibration_label']}, {signal}): CV AUC {cvl['auc']:.3f} "
                      f"ECE {cvl['ece']:.3f}")
            if cals["prefix"]:
                pc = cals["prefix"]
                print(f"    prefix calibration ({signal}, {pc['source']}): CV AUC {pc['cv']['auc']} "
                      f"ECE {pc['cv']['ece']}")
        for signal, conds in (m.get("asr_word_frac_curve") or {}).items():
            for cond, entry in conds.items():
                print(f"    ASR word_frac curve {signal} [{cond}]: c2_available {entry['c2_available']}")
        for kind, g in (m.get("gains") or {}).items():
            print(f"    gains ({kind}): g+ {g['g_plus']} g- {g['g_minus']} b* {g['break_even_belief']}"
                  "   -> configs/config.yaml controller.e3v2.quality")
        if m.get("boundary_mass") and m["boundary_mass"]["dev_out_of_fold"]:
            print(f"    boundary mass (OOF): {m['boundary_mass']['dev_out_of_fold']['fraction']:.3f}")
    print(f"[analyze] wrote {common.rel(out_path)} and {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
