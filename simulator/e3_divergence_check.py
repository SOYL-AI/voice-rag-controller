"""
e3_divergence_check.py — does the derived price grid actually resolve an
Ours-vs-B9 disagreement, and at how many prices?

    python -m simulator.e3_divergence_check

---------------------------------------------------------------------------
THE QUESTION THIS ANSWERS, AND WHY IT IS NOT THE ONE e3_grid.py ANSWERS
---------------------------------------------------------------------------
simulator/e3_grid.py reports, per price, how many scenarios fall inside the
PREDICTED divergence band — an interval computed in closed form from the
price and the measured latency constants (controller/baselines/
_objective.divergence_band()). That is algebra about the controllers, not
the controllers.

This module runs the actual controllers. For every price in the grid and
every one of the 50 frozen scenario snapshots it constructs the Ctx
pipeline/spine.py would hand a baseline, calls controller.baselines.ours
and controller.baselines.b9_decomposed's decide(), and compares the two
Decisions. A "divergence" here is an OBSERVED disagreement about whether to
retrieve, read off two real decisions, not a VoI compared against an
interval.

Both numbers are reported side by side, because their AGREEMENT is itself
the check: the band is derived from the same primitives the controllers
use, so if predicted and observed ever came apart, one of the two is
wrong — either the band algebra has drifted from the controllers'
arithmetic, or a controller is not optimizing the objective it claims to.
Reporting only the prediction would hide that; reporting only the
observation would leave the grid's own diagnostic unchecked.

---------------------------------------------------------------------------
WHAT A HEALTHY RESULT LOOKS LIKE
---------------------------------------------------------------------------
Not "a large total number of divergences" — that would be maximized by a
degenerate grid parked at one price where half the scenarios sit. What E3
needs is divergence spread across MANY prices, because the frontier is a
curve and each price is one point on it. A grid where only two prices
resolve any disagreement gives a two-point frontier no matter how many
scenarios diverge at those two prices.

So the headline number here is `prices_with_divergence`: how many grid
points produce at least one observed Ours != B9 retrieve decision. That is
the number of frontier points that can differ at all.

This is a DIAGNOSTIC, not a result. It says the instrument can resolve the
effect; it says nothing about which controller is better, which is what the
scored sweep measures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# import-for-side-effect: registers every baseline (see
# controller/baselines/__init__.py) so get_baseline() can find them.
import controller.baselines  # noqa: F401,E402
from controller.baseline import Ctx, get_baseline  # noqa: E402
from controller.baselines._objective import (  # noqa: E402
    Prices,
    divergence_band,
    estimate_voi,
    load_e3_config,
    load_voi_calibration,
)

SIGNALS_CACHE = Path(__file__).parent / "e3_scenario_signals.json"
GRID_FILE = _REPO_ROOT / "configs" / "e3_price_grid.json"


def scenario_contexts(cache: dict, prices: dict) -> list:
    """
    One Ctx per cached scenario at a given price point — the same shape
    simulator/e3_grid.py's voi_distribution() builds and the same one
    pipeline/spine.py hands a baseline, so the decisions here are the
    decisions a sweep would log.
    """
    contexts = []
    for entry in cache["signals"]:
        contexts.append(
            Ctx(
                transcript=entry["question"],
                signals={
                    "model": {
                        "available": True,
                        "uncertainty": entry["model_uncertainty"],
                    },
                    "tracker": entry["tracker"],
                },
                elapsed_ms=0,
                user_end_known=True,
                turn_id=1,
                conversation_id=f"e3_divergence_{entry['scenario_id']:03d}",
                prices=prices,
            )
        )
    return contexts


def check_price(cache: dict, cfg: dict, lambda_l: float, lambda_c: float) -> dict:
    """
    Run Ours and B9 over every scenario at one price and return the
    observed disagreement, alongside the band's prediction for the same
    price.

    `ours_only` is the divergence E3 is about: Ours retrieves where B9
    skips, because Ours prices the retrieval at the latency its chosen
    timing actually exposes. `b9_only` is the opposite disagreement, which
    the band predicts only at prices where its width is negative (Ours's
    extra waste_rate outweighing its latency saving); it is counted rather
    than assumed absent, because assuming it is exactly the way a sign
    error would go unnoticed.
    """
    ours = get_baseline("ours")
    b9 = get_baseline("b9")
    band = divergence_band(Prices(lambda_l=lambda_l, lambda_c=lambda_c), cfg)

    contexts = scenario_contexts(cache, {"lambda_l": lambda_l, "lambda_c": lambda_c})
    ours_only, b9_only, both, neither, predicted, disagreeing = 0, 0, 0, 0, 0, []
    for ctx in contexts:
        voi = estimate_voi(ctx, cfg)
        ours_retrieves = ours.decide(ctx).retrieve
        b9_retrieves = b9.decide(ctx).retrieve
        if band.contains(voi):
            predicted += 1
        if ours_retrieves and not b9_retrieves:
            ours_only += 1
            disagreeing.append(ctx.conversation_id)
        elif b9_retrieves and not ours_retrieves:
            b9_only += 1
            disagreeing.append(ctx.conversation_id)
        elif ours_retrieves:
            both += 1
        else:
            neither += 1

    return {
        "lambda_l": lambda_l,
        "lambda_c": lambda_c,
        "band": {
            "ours_cut": band.ours_cut,
            "b9_cut": band.b9_cut,
            "width": band.width,
            "timing": band.timing,
        },
        "n_predicted_in_band": predicted,
        "n_ours_retrieves_b9_skips": ours_only,
        "n_b9_retrieves_ours_skips": b9_only,
        "n_both_retrieve": both,
        "n_neither_retrieves": neither,
        "n_observed_divergence": ours_only + b9_only,
        "prediction_matches_observation": predicted == ours_only + b9_only,
        "diverging_scenarios": disagreeing,
    }


def voi_summary(cache: dict, cfg: dict) -> dict:
    """
    The VoI distribution the grid has to resolve, plus the two shape
    statistics that AUC cannot see.

    `largest_gap` and `middle_third` are the point of this block. A signal
    can rank perfectly (high AUC) and still take only two values, and a
    two-valued VoI cannot produce a frontier with a middle no matter how
    the prices are chosen. These are the numbers that distinguish "the
    estimator discriminates" from "the estimator resolves".
    """
    contexts = scenario_contexts(cache, {})
    values = sorted(estimate_voi(ctx, cfg) for ctx in contexts)
    low, high = values[0], values[-1]
    span = high - low
    gaps = [b - a for a, b in zip(values, values[1:])]
    middle = [v for v in values if low + span / 3 <= v <= low + 2 * span / 3]
    floor = float(cfg["voi"]["floor"])
    return {
        "n": len(values),
        "min": low,
        "max": high,
        "spread": span,
        "mean": sum(values) / len(values),
        "median": values[len(values) // 2],
        "largest_gap": max(gaps),
        "n_middle_third": len(middle),
        "n_at_floor": sum(1 for v in values if v <= floor + 1e-12),
        "floor": floor,
        "values": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ours and B9 over the 50 frozen scenario snapshots at every "
            "price in the derived grid and count, per price, how many "
            "scenarios the two controllers actually disagree about."
        )
    )
    parser.add_argument(
        "--grid",
        default=str(GRID_FILE),
        help="Grid artifact to read lambda_l/lambda_c from.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the full per-price report as JSON.",
    )
    args = parser.parse_args()

    cfg = load_e3_config()
    calibration = load_voi_calibration(cfg)
    with open(SIGNALS_CACHE, "r", encoding="utf-8") as handle:
        cache = json.load(handle)
    with open(args.grid, "r", encoding="utf-8") as handle:
        grid_artifact = json.load(handle)

    grid = grid_artifact["grid"]["lambda_l_grid"]
    lambda_c = float(grid_artifact["grid"]["lambda_c"])

    print("=" * 78)
    print("E3 DIVERGENCE CHECK — observed Ours-vs-B9 disagreement per grid price")
    print("=" * 78)
    print(f"calibration : {calibration.describe()}")
    print(f"signals     : backend={cache['backend']} "
          f"model={cache.get('active_model')} "
          f"({cache['n_scenarios']} scenarios, "
          f"real={cache['backend'] != 'mock'})")
    print(f"grid        : {len(grid)} prices, lambda_c {lambda_c}")

    summary = voi_summary(cache, cfg)
    print()
    print(f"(a) VoI DISTRIBUTION over {summary['n']} scenarios")
    print(f"    min {summary['min']:.4f}   max {summary['max']:.4f}   "
          f"spread {summary['spread']:.4f}   mean {summary['mean']:.4f}   "
          f"median {summary['median']:.4f}")
    print(f"    largest gap {summary['largest_gap']:.4f}   "
          f"middle third {summary['n_middle_third']}/{summary['n']}   "
          f"at floor {summary['n_at_floor']}/{summary['n']}")

    rows = [check_price(cache, cfg, lambda_l, lambda_c) for lambda_l in grid]

    print()
    print("(b) PER-PRICE DIVERGENCE — ours and b9 decide() over all 50 scenarios")
    print("    lambda_l      band (ours_cut, b9_cut]   predicted  observed  "
          "ours_only  b9_only  both  neither")
    for row in rows:
        flag = "" if row["prediction_matches_observation"] else "   <-- MISMATCH"
        print(f"    {row['lambda_l']:9.4f}  ({row['band']['ours_cut']:.4f}, "
              f"{row['band']['b9_cut']:.4f}]  "
              f"{row['n_predicted_in_band']:9d}  "
              f"{row['n_observed_divergence']:8d}  "
              f"{row['n_ours_retrieves_b9_skips']:9d}  "
              f"{row['n_b9_retrieves_ours_skips']:7d}  "
              f"{row['n_both_retrieve']:4d}  {row['n_neither_retrieves']:7d}{flag}")

    with_divergence = [r for r in rows if r["n_observed_divergence"] > 0]
    mismatches = [r for r in rows if not r["prediction_matches_observation"]]
    total = sum(r["n_observed_divergence"] for r in rows)

    print()
    print(f"    prices with >= 1 observed divergence : "
          f"{len(with_divergence)}/{len(rows)}")
    print(f"    total (price, scenario) divergences  : {total}")
    print(f"    band prediction == observation       : "
          f"{len(rows) - len(mismatches)}/{len(rows)} prices"
          + ("" if not mismatches else "   *** MISMATCHES PRESENT ***"))

    report = {
        "calibration": {
            "signal": calibration.signal,
            "slope": calibration.slope,
            "intercept": calibration.intercept,
            "auc": calibration.auc,
            "source": calibration.source,
        },
        "signals": {
            "backend": cache["backend"],
            "active_model": cache.get("active_model"),
            "n_scenarios": cache["n_scenarios"],
            "are_real_model_signals": cache["backend"] != "mock",
        },
        "voi_distribution": summary,
        "lambda_c": lambda_c,
        "per_price": rows,
        "n_prices": len(rows),
        "n_prices_with_divergence": len(with_divergence),
        "total_divergences": total,
        "n_prices_band_prediction_mismatch": len(mismatches),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
