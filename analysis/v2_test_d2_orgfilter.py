"""
v2_test_d2_orgfilter.py -- EXPLORATORY / post hoc / not pre-registered
(PREREG_V2 note N2). d2_support TEST contrasts under PER-COMPANY retrieval
(simulator/e3v2_orgfilter.py), next to the frozen shared-index retriever.
Nothing here enters any pre-registered test or verdict.

    PYTHONUTF8=1 python -m analysis.v2_test_d2_orgfilter

Inputs:
  * simulator/e3v2_phase_a/orgfilter/d2_support/contrasts_test.json
        python -m simulator.e3v2_lazy evaluate --dataset d2_support --split test
            --phase-a simulator/e3v2_phase_a/orgfilter --bundle <asr> --bundle <text150>
            --d-star-speculative b10_speculative_v2 --m3 0.016
  * the confirmatory simulator/e3v2_phase_a/d2_support/contrasts_test.json when
    present (frozen retriever, for the side-by-side), and both stores
    (own-company retrieval rates, grounded accuracy).

Statistics as PREREG_V2 §4.1 (paired cluster bootstrap over d2_support
pair_id clusters, B = 10,000, the same resamples for every contrast,
percentile CIs); every p-value is descriptive.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.experiment1_v2 import common as e1common  # noqa: E402

DS = "d2_support"
FROZEN = _REPO_ROOT / "simulator" / "e3v2_phase_a" / DS
ORGF = _REPO_ROOT / "simulator" / "e3v2_phase_a" / "orgfilter" / DS
OUT_JSON = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_d2_orgfilter_EXPLORATORY.json"
OUT_MD = _REPO_ROOT / "docs" / "research_v2" / "v2_test_results_d2_orgfilter_EXPLORATORY.md"
LABEL = "EXPLORATORY / post hoc / not pre-registered"
B, SEED, M3 = 10_000, 20261017, 0.016
PRIMARY = {"condition": "piper-en_US-hfc_male-medium__150wpm__clean", "profile": "rewrite_faiss", "quantile": 0.65,
           "lambda_l": 0.2157823714003309, "lambda_c": 0.05}


def cluster_counts(universe, cluster_of):
    clusters = sorted({cluster_of.get(i, i) for i in universe})
    cidx = {c: k for k, c in enumerate(clusters)}
    item_c = np.array([cidx[cluster_of.get(i, i)] for i in universe])
    rng = np.random.default_rng(SEED)
    counts = np.zeros((B, len(universe)))
    for b in range(B):
        cc = np.bincount(rng.integers(0, len(clusters), size=len(clusters)), minlength=len(clusters))
        counts[b] = cc[item_c]
    return counts


def boot(counts, uidx, values: dict):
    x, m = np.zeros(counts.shape[1]), np.zeros(counts.shape[1])
    for i, v in values.items():
        x[uidx[i]], m[uidx[i]] = float(v), 1.0
    tot = counts @ m
    means = (counts @ x) / np.where(tot > 0, tot, np.nan)
    q = np.nanpercentile(means, [2.5, 97.5])
    return {"estimate": float(np.mean(list(values.values()))), "ci95": [float(q[0]), float(q[1])],
            "boot_se": float(np.nanstd(means, ddof=1))}, means


def block(counts, uidx, per_item):
    c, dc = boot(counts, uidx, per_item["delta_commit"])
    s, ds = boot(counts, uidx, per_item["delta_speculative"])
    i, it = boot(counts, uidx, per_item["interaction"])
    return {"n_items": {k: len(per_item[v]) for k, v in (("commit", "delta_commit"),
                                                         ("speculative", "delta_speculative"),
                                                         ("paired", "interaction"))},
            "delta_commit": c, "delta_speculative": s, "interaction": i,
            "descriptive_one_sided_p": {"P1_delta_C_le_0": float(np.nanmean(dc <= 0)),
                                        "H-REV_delta_C_ge_0": float(np.nanmean(dc >= 0)),
                                        "P2_interaction_le_0": float(np.nanmean(it <= 0)),
                                        "P3_upper_delta_S_ge_m3": float(np.nanmean(ds >= M3)),
                                        "P3_lower_delta_S_le_minus_m3": float(np.nanmean(ds <= -M3))}}


def store_stats(root: Path) -> dict:
    items = {it["id"]: it for it in e1common.load_items(e1common.resolve_dataset_paths(DS)["items"])}
    docs = {}
    idx_dir = Path(r"C:\Users\ryang\OneDrive\Desktop\voice-rag\voice-rag-controller\retrieval\indexes") / DS
    for line in open(idx_dir / "docs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
    top1, anyown, n = 0, 0, 0
    acc = defaultdict(list)
    for line in open(root / "store_test.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] == "retrieval":
            org = items[r["item_id"]]["org"]
            own = [docs[d]["org"] == org for d in r["retrieved_doc_ids"]]
            n, top1, anyown = n + 1, top1 + own[0], anyown + any(own)
        elif r["kind"] == "cell" and r["answer_correct"] is not None:
            acc[(r["path"], items[r["item_id"]].get("item_type"))].append(float(r["answer_correct"]))
    by_path = defaultdict(list)
    for (p, _), v in acc.items():
        by_path[p] += v
    return {"dispatch_points": n, "top1_own_company": top1 / max(1, n), "any_own_company_top3": anyown / max(1, n),
            "accuracy_by_path": {p: {"n_cells": len(v), "accuracy": float(np.mean(v))} for p, v in by_path.items()},
            "accuracy_by_path_and_item_type": {f"{p}|{t}": {"n_cells": len(v), "accuracy": float(np.mean(v))}
                                               for (p, t), v in sorted(acc.items())}}


def analyse(root: Path, cluster_of) -> dict:
    doc = e1common.load_json(root / "contrasts_test.json")
    bundles = {e1common.load_json(Path(p))["condition"]: e1common.load_json(Path(p)) for p in doc["bundles"]}
    universe = sorted({i for rows in doc["contrasts"].values() for r in rows
                       for k in ("delta_commit", "delta_speculative") for i in r["per_item"][k]})
    uidx = {i: k for k, i in enumerate(universe)}
    counts = cluster_counts(universe, cluster_of)
    cells = []
    for cond, rows in doc["contrasts"].items():
        grid = bundles[cond]["price_grid"]
        for r in rows:
            q = next((g["quantile"] for g in grid[r["latency_profile"]]
                      if f"{g['lambda_l']:g}|{g['lambda_c']:g}" == r["price"]), None)
            cells.append({"label": LABEL, "condition": cond, "profile": r["latency_profile"], "belief_quantile": q,
                          "price": r["price"], "d_star": r["d_star"], **block(counts, uidx, r["per_item"])})
    return {"contrasts_sha256": e1common.sha256_file(root / "contrasts_test.json"), "cells": cells,
            "n_items": len(universe), "n_clusters": len({cluster_of.get(i, i) for i in universe})}


def main() -> int:
    items = {it["id"]: it for it in e1common.load_items(e1common.resolve_dataset_paths(DS)["items"])}
    cluster_of = {i: (it.get("pair_id") or i) for i, it in items.items()}
    out = {"_comment": LABEL + " (PREREG_V2 note N2). d2_support test under per-company retrieval; it does not "
                               "enter any pre-registered test or verdict.",
           "label": LABEL, "generated_by": "python -m analysis.v2_test_d2_orgfilter",
           "generated_at_utc": e1common.now_utc(), "git_commit": e1common.git_commit(),
           "bootstrap": {"B": B, "seed": SEED, "unit": "pair_id cluster"}, "m3": M3, "primary_cell": PRIMARY,
           "retrieval": {"org_filtered": store_stats(ORGF), "frozen_shared_index": store_stats(FROZEN)},
           "org_filtered": analyse(ORGF, cluster_of)}
    if (FROZEN / "contrasts_test.json").exists():
        out["frozen_shared_index_same_statistics"] = analyse(FROZEN, cluster_of)
    for k in ("org_filtered", "frozen_shared_index_same_statistics"):
        if k in out:
            out[k]["primary"] = next((c for c in out[k]["cells"] if c["condition"] == PRIMARY["condition"]
                                      and c["profile"] == PRIMARY["profile"]
                                      and c["belief_quantile"] == PRIMARY["quantile"]), None)
    e1common.atomic_write_json(OUT_JSON, out)
    write_md(out)
    print(json.dumps({k: out[k].get("primary") for k in ("org_filtered", "frozen_shared_index_same_statistics")
                      if k in out}, indent=1)[:3000])
    return 0


def f4(x):
    return f"{x:+.4f}"


def write_md(out):
    L = [f"# {LABEL}: d2_support test with per-company retrieval", "",
         f"**{LABEL}** (PREREG_V2 note N2). The confirmatory analysis uses the frozen shared-index retriever and is "
         "unaffected. Paired cluster bootstrap over pair_id clusters, B = 10,000; all p-values are descriptive.", "",
         f"## Retrieval and grounded accuracy ({LABEL})", "",
         "| retriever | top-1 own company | any own company in top-3 | grounded (retrieve-path) accuracy | skip-path accuracy |",
         "|---|---|---|---|---|"]
    for name, k in (("frozen shared index", "frozen_shared_index"), ("per-company filter", "org_filtered")):
        s = out["retrieval"][k]
        ap = s["accuracy_by_path"]
        L.append(f"| {name} | {s['top1_own_company']:.3f} | {s['any_own_company_top3']:.3f} | "
                 f"{ap.get('retrieve', {}).get('accuracy', float('nan')):.3f} | "
                 f"{ap.get('skip', {}).get('accuracy', float('nan')):.3f} |")
    for k, title in (("org_filtered", "per-company filter"), ("frozen_shared_index_same_statistics",
                                                             "frozen shared index (same statistics)")):
        if k not in out:
            continue
        L += ["", f"## Contrasts, {title} ({LABEL})", "",
              "| condition | profile | q | Δ[COMMIT] [95% CI] | Δ[SPEC] [95% CI] | interaction [95% CI] |",
              "|---|---|---|---|---|---|"]
        for c in sorted(out[k]["cells"], key=lambda c: (c["condition"], c["profile"], c["belief_quantile"] or 0)):
            cond = "ASR" if c["condition"] == PRIMARY["condition"] else c["condition"]
            L.append(f"| {cond} | {c['profile']} | {c['belief_quantile']} | " + " | ".join(
                f"{f4(c[x]['estimate'])} [{f4(c[x]['ci95'][0])}, {f4(c[x]['ci95'][1])}]"
                for x in ("delta_commit", "delta_speculative", "interaction")) + " |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
