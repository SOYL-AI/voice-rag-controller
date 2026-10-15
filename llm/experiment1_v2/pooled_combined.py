"""Combined belief, per dataset and pooled across datasets (dev only).

    python -m llm.experiment1_v2.pooled_combined --datasets d1_open,d2_support \
        --labels d2_support=labels_dev.e1v2-r3.json --out pooled_combined_dev.json

For each combined signal (analyze.COMBINED: combined_v1 = mean entropy +
logit(self_assessed_need); combined_fast = first-16-token entropy + the
same), it compares two out-of-fold fits, both with item/pair-grouped CV:

  per_dataset   a combiner fitted within each dataset
  pooled        one combiner fitted on all datasets' full-question records

AUC is item-level (signal averaged over seeds) against the EMPIRICAL label,
with a cluster-bootstrap CI. It is reported per dataset and over the pooled
items, on the full question and at each text-prefix word fraction, next to
the single inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.experiment1_v2 import analyze, common  # noqa: E402

INPUTS = ["mean_entropy_nats", "first_k_mean_entropy_nats@16", analyze.SA_SIGNAL]


def _rename(records, suffix):
    for r in records:
        for name in analyze.COMBINED:
            if name in r["_flat"]:
                r["_flat"][f"{name}_{suffix}"] = r["_flat"].pop(name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="d1_open,d2_support")
    p.add_argument("--split", default="dev", choices=common.VALID_SPLITS)
    p.add_argument("--model", default="mistral7b_v03")
    p.add_argument("--results-root", type=Path, default=common.RESULTS_ROOT)
    p.add_argument("--labels", action="append", default=[],
                   help="DATASET=labels file name (default labels_<split>.json)")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        common.check_split_allowed(args.split)
    except common.SealedSplitError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 3
    label_files = dict(entry.split("=", 1) for entry in args.labels)
    data, all_records, all_labels, all_clusters = {}, [], {}, {}
    for ds in common.parse_csv_list(args.datasets):
        paths = common.resolve_dataset_paths(ds)
        items = common.select_split(common.load_items(paths["items"]),
                                    common.load_splits(paths["splits"]), args.split)
        inp = analyze.load_model_inputs(args.results_root / ds / args.model, args.split,
                                        label_files.get(ds))
        lmap = analyze.label_maps(items, inp["labels"]).get("empirical")
        if not lmap:
            print(f"[pooled] {ds}: no empirical labels, skipped")
            continue
        clusters = analyze.cluster_of(items)
        combiners = analyze.add_combined_signals(inp["records"], lmap, clusters, args.folds)
        _rename(inp["records"], "per_dataset")
        data[ds] = {"records": inp["records"], "labels": lmap, "clusters": clusters,
                    "combiners_per_dataset": combiners,
                    "labels_file": common.rel(inp["labels_path"]) if inp["labels_path"] else None}
        all_records += inp["records"]
        all_labels.update(lmap)
        all_clusters.update(clusters)
    pooled_combiners = analyze.add_combined_signals(all_records, all_labels, all_clusters, args.folds)
    _rename(all_records, "pooled")

    signals = INPUTS + [f"{n}_{s}" for n in analyze.COMBINED for s in ("per_dataset", "pooled")]
    fracs = sorted({analyze.condition_of(r)[1] for r in all_records
                    if analyze.condition_of(r)[0] == "text"})

    def table(records, lmap, clusters):
        out = {}
        for signal in signals:
            out[signal] = {}
            for frac in fracs:
                rows = [r for r in records if analyze.condition_of(r) == ("text", frac)]
                res = analyze.item_auc(analyze.item_scores(rows, signal), lmap, clusters, args.bootstrap)
                out[signal][str(frac)] = {"auc": res["auc"], "ci95": res["ci95"], "n_items": res["n_items"]}
        return out

    report = {"split": args.split, "model": args.model, "label": "empirical",
              "datasets": {}, "pooled": None, "pooled_combiners": pooled_combiners,
              "created_at_utc": common.now_utc(), "git_commit": common.git_commit()}
    for ds, d in data.items():
        report["datasets"][ds] = {"labels_file": d["labels_file"], "n_items": len(d["labels"]),
                                  "base_rate": sum(d["labels"].values()) / len(d["labels"]),
                                  "combiners_per_dataset": d["combiners_per_dataset"],
                                  "auc": table(d["records"], d["labels"], d["clusters"])}
    report["pooled"] = {"n_items": len(all_labels), "auc": table(all_records, all_labels, all_clusters)}
    common.atomic_write_json(args.out, report)

    def show(name, tab):
        print(f"== {name}")
        for signal in signals:
            cells = []
            for frac in fracs:
                c = tab[signal][str(frac)]
                if c["auc"] is not None:
                    ci = c["ci95"]
                    cells.append(f"{frac}: {c['auc']:.3f}" + (f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""))
            print(f"   {signal:36s} " + " | ".join(cells))

    for ds in data:
        show(f"{ds} (n={report['datasets'][ds]['n_items']}, base rate "
             f"{report['datasets'][ds]['base_rate']:.3f})", report["datasets"][ds]["auc"])
    show(f"pooled (n={len(all_labels)})", report["pooled"]["auc"])
    print(f"[pooled] wrote {common.rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
