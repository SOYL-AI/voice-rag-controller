"""
e3_grid.py — derive E3's price grid from the VoI distribution it has to
resolve. ONE COMMAND regenerates everything:

    python -m simulator.e3_grid

That prints the VoI distribution across the 50 scenarios and the derived
lambda_l grid, and writes configs/e3_price_grid.json. Nothing else in the
repo needs to be re-run by hand afterwards; the grid artifact records every
input it was derived from, so a stale one is detectable rather than merely
suspected.

---------------------------------------------------------------------------
WHY A GRID HAS TO BE DERIVED AND NOT CHOSEN
---------------------------------------------------------------------------
E3 compares a joint controller (Ours) against a decomposed, price-coordinated
one (B9). They differ on exactly one thing: at a given price, Ours prices a
retrieval at the latency its chosen timing actually EXPOSES, while B9 prices
the full retrieval. So they disagree about a turn iff that turn's VoI falls
between their two retrieve/skip thresholds — the divergence band of
controller/baselines/_objective.py:

    ours_cut(l) = l * exposed(t*) + l_c * (1 + waste(t*))
    b9_cut(l)   = l * full        + l_c * (1 + waste(sentence_end))
    band        = (ours_cut, b9_cut]        width = l * overlap - l_c * dwaste

Both cuts slide with the price l = lambda_l. A price grid whose bands all sit
ABOVE the observed VoI values measures nothing (neither controller ever
retrieves); one whose bands all sit BELOW measures nothing either (both always
retrieve). Either way the sweep returns "no difference", and that null is an
artifact of the grid, not a result about joint reasoning. Hand-picking round
numbers (0.1, 0.5, 1, 5) gives no reason to believe the bands land anywhere
near the VoI values the estimator actually produces.

---------------------------------------------------------------------------
THE BAND-TILING FORMULA
---------------------------------------------------------------------------
So the grid is derived from the VoI distribution, by TILING: choose prices so
that consecutive divergence bands are edge-to-edge, covering the observed VoI
range once, with no gaps and no overlap. Band k+1 starts where band k stops:

    ours_cut(l_{k+1}) = b9_cut(l_k)

    =>  l_{k+1} = [ full * l_k + l_c * (w_sentence_end - w_t*) ] / exposed(t*)

an affine recurrence whose slope full/exposed = 0.672/0.484 = 1.388 makes the
grid roughly geometric — dense where bands are narrow, sparse where they are
wide, which is exactly the spacing that spends sweep budget evenly over the
question being asked.

Endpoints, both read off the data rather than chosen:

  START  the smallest price whose band top edge reaches the lowest REACHABLE
         observed VoI:  b9_cut(l_1) = v_lo.
  STOP   the first price whose band bottom edge has passed the highest
         observed VoI: ours_cut(l_K) >= v_hi. That band can contain no
         scenario, so it and everything above it are dropped.

"Reachable" is load-bearing. ours_cut(l) >= l_c * (1 + waste(t*)) for every
l >= 0, so a scenario whose VoI is below that compute floor is one NEITHER
controller ever retrieves, at any price. Such scenarios cannot be tiled over
and are reported separately rather than dragging the grid down to prices where
nothing happens.

There is also a degenerate price below which the band inverts (width < 0,
Ours's extra waste outweighing its latency saving):

    l_deg = l_c * (waste(t*) - waste(sentence_end)) / overlap

l_deg is exactly the fixed point of the tiling recurrence — a zero-width band
tiles to itself forever — so the derivation starts strictly above it and says
so if the data would have put it lower.

--refine N subdivides each tiling step into N geometric sub-steps. It does not
change what the grid covers, only the resolution inside it; a frontier plot
usually wants more than the handful of points bare tiling produces. N=1 is the
pure formula.

---------------------------------------------------------------------------
WHERE THE VoI VALUES COME FROM — AND THE CAVEAT THAT TRAVELS WITH THEM
---------------------------------------------------------------------------
Per scenario, this runs simulator/e3_signals.py's precompute_signals() (the
same frozen closed-book snapshot every E3 baseline is handed) and then the
SHARED estimator, controller/baselines/_objective.estimate_voi(). It does not
reimplement the estimator — a second copy of the belief -> VoI mapping is
exactly the drift the shared-objective design exists to prevent.

*** THE SIGNALS ARE ONLY AS REAL AS THE BACKEND THAT PRODUCED THEM. *** With
configs/config.yaml's llm.backend = mock, max_entropy_nats comes from
MockClient's synthetic logprobs, which are seeded on the prompt (so this is
deterministic and reproducible) but say nothing about what the real model is
uncertain about. The CALIBRATION is real — it is fitted on E1's 180 real Modal
samples — but a real calibration applied to mock signals still yields a mock
VoI distribution, and therefore a mock grid. The signals cache and the grid
artifact both record the backend and model they were produced under, and the
printed report says so out loud. Re-run this one command against the real
backend before any headline number is reported off the grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baseline import Ctx  # noqa: E402
from controller.baselines._objective import (  # noqa: E402
    Prices,
    divergence_band,
    estimate_voi,
    load_e3_config,
    load_voi_calibration,
    overlap_s,
    retrieval_compute_units,
    retrieval_latency_s,
    exposed_latency_s,
)
from harness.stats import percentile  # noqa: E402
from llm.client import make_client  # noqa: E402
from llm.config import load_config as load_llm_config  # noqa: E402
from simulator.e3_signals import (  # noqa: E402
    PRECOMPUTE_PROTOCOL_VERSION,
    load_e1_protocol,
    precompute_signals,
)

SCENARIOS_FILE = Path(__file__).parent / "scenarios.json"
SIGNALS_CACHE = Path(__file__).parent / "e3_scenario_signals.json"
GRID_FILE = _REPO_ROOT / "configs" / "e3_price_grid.json"

PERCENTILES = (0, 5, 10, 25, 50, 75, 90, 95, 100)


# ---------------------------------------------------------------------------
# Step 1 — the frozen per-scenario signals
# ---------------------------------------------------------------------------
def load_or_build_signals(refresh: bool = False) -> dict:
    """
    Return the frozen belief-signal snapshot per scenario, from the cache
    at simulator/e3_scenario_signals.json, building it if absent or if
    `refresh`.

    CACHED ON PURPOSE, and committed. Deriving the grid must not require a
    model call: the grid is an input to every sweep, so re-deriving it has
    to be cheap and byte-reproducible on a laptop with no GPU. The cache
    also pins WHICH signals the committed grid was derived from — with the
    snapshot thrown away, a grid and a sweep could silently disagree about
    what the model believed.

    The cache records the backend and model. Changing either (mock -> the
    real vLLM/Modal backend) invalidates it, so --refresh is required to
    regenerate rather than the cache being topped up in place: a cache
    holding half mock and half real signals would be undetectable
    downstream.

    It records the PRECOMPUTE PROTOCOL for the same reason, and invalidates
    on it too. Signals averaged over E1's five seeds are not comparable to
    the single unseeded draw the pre-v2 cache holds (simulator/e3_signals.py's
    docstring), and reusing the old file would derive a grid from signals no
    current code path can reproduce.
    """
    llm_cfg = load_llm_config()
    backend = llm_cfg.get("backend")
    model = llm_cfg.get("active_model")
    protocol = load_e1_protocol()

    if SIGNALS_CACHE.exists() and not refresh:
        with open(SIGNALS_CACHE, "r", encoding="utf-8") as handle:
            cache = json.load(handle)
        cached_protocol = cache.get("precompute") or {}
        if cache.get("backend") != backend or cache.get("active_model") != model:
            print(
                f"signals cache was built under backend={cache.get('backend')!r} / "
                f"model={cache.get('active_model')!r} but the config now says "
                f"backend={backend!r} / model={model!r} — rebuilding.",
            )
        elif (
            cached_protocol.get("protocol_version") != PRECOMPUTE_PROTOCOL_VERSION
            or cached_protocol.get("seeds") != list(protocol["seeds"])
            or cached_protocol.get("temperature") != protocol["temperature"]
            or cached_protocol.get("top_p") != protocol["top_p"]
            or cached_protocol.get("max_tokens") != protocol["max_tokens"]
        ):
            print(
                f"signals cache was built under precompute protocol "
                f"v{cached_protocol.get('protocol_version')} "
                f"(seeds={cached_protocol.get('seeds')}, "
                f"temperature={cached_protocol.get('temperature')}, "
                f"max_tokens={cached_protocol.get('max_tokens')}) but the current "
                f"protocol is v{PRECOMPUTE_PROTOCOL_VERSION} "
                f"(seeds={protocol['seeds']}, "
                f"temperature={protocol['temperature']}, "
                f"max_tokens={protocol['max_tokens']}) — rebuilding.",
            )
        else:
            return cache

    with open(SCENARIOS_FILE, "r", encoding="utf-8") as handle:
        scenarios = json.load(handle)

    print(f"precomputing belief signals for {len(scenarios)} scenarios "
          f"(backend={backend}, model={model}) ...")
    client = make_client(llm_cfg)
    entries = []
    precompute_provenance = None
    for scenario in scenarios:
        snapshot = precompute_signals(scenario["question"], client=client, cfg=llm_cfg)
        provenance = snapshot["precompute"]
        if precompute_provenance is None:
            # Identical for every scenario (it is the protocol, not the
            # result), so it is stored once at the cache level rather than
            # repeated 50 times.
            precompute_provenance = {
                key: value
                for key, value in provenance.items()
                if key not in {"per_draw_uncertainty", "token_counts_per_draw",
                               "n_draws_by_signal"}
            }
        entries.append(
            {
                "scenario_id": scenario["id"],
                "question": scenario["question"],
                "retrieval_needed": scenario.get("retrieval_needed"),
                # Only the two blocks estimate_voi()'s signal ladder reads.
                # Storing the whole snapshot would bloat the cache with
                # generated text that no consumer of this file looks at.
                "model_uncertainty": snapshot["model"]["uncertainty"],
                "tracker": snapshot["tracker"],
                # The draws behind the means above. Not read by the grid, and
                # kept anyway: they are the only record of how much spread
                # each mean was taken over, which is what a reader checking
                # that these signals are stable has to look at. Numbers, not
                # generated text, so this stays small.
                "model_uncertainty_per_draw": provenance["per_draw_uncertainty"],
                "n_draws_by_signal": provenance["n_draws_by_signal"],
            }
        )

    cache = {
        "_comment": (
            "GENERATED by `python -m simulator.e3_grid --refresh-signals` — the "
            "frozen closed-book belief signals E3's price grid is derived from. "
            "Same snapshot simulator/e3_signals.py hands every baseline. Each "
            "model_uncertainty is a mean over the E1-matched draws recorded in "
            "\"precompute\" below."
        ),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "active_model": model,
        "precompute": precompute_provenance,
        "n_scenarios": len(entries),
        "scenarios_file": "simulator/scenarios.json",
        "signals": entries,
    }
    with open(SIGNALS_CACHE, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)
        handle.write("\n")
    print(f"wrote {SIGNALS_CACHE.relative_to(_REPO_ROOT)}")
    return cache


# ---------------------------------------------------------------------------
# Step 2 — the VoI distribution
# ---------------------------------------------------------------------------
def voi_distribution(cache: dict, cfg: dict) -> list:
    """
    Run the SHARED estimator over every cached scenario snapshot and return
    one record per scenario, sorted by VoI.

    Builds the same Ctx shape pipeline/spine.py hands a baseline, so the
    number here is the number Ours and B9 will see, not an approximation
    of it.
    """
    signal = load_voi_calibration(cfg).signal
    records = []
    for entry in cache["signals"]:
        ctx = Ctx(
            transcript=entry["question"],
            signals={
                "model": {"available": True, "uncertainty": entry["model_uncertainty"]},
                "tracker": entry["tracker"],
            },
            elapsed_ms=0,
            user_end_known=True,
            turn_id=1,
            conversation_id=f"e3_grid_{entry['scenario_id']:03d}",
            prices={},
        )
        records.append(
            {
                "scenario_id": entry["scenario_id"],
                "signal": entry["model_uncertainty"].get(signal),
                "voi": estimate_voi(ctx, cfg),
                "retrieval_needed": entry.get("retrieval_needed"),
            }
        )
    return sorted(records, key=lambda r: r["voi"])


def summarize(records: list, cfg: dict) -> dict:
    """Distribution summary, plus the two counts that decide whether a grid
    can be tiled at all (floored scenarios and, later, unreachable ones)."""
    values = [r["voi"] for r in records]
    floor = float(cfg["voi"]["floor"])
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
        "mean": sum(values) / len(values),
        "percentiles": {f"p{p}": percentile(values, p) for p in PERCENTILES},
        "n_at_floor": sum(1 for v in values if v <= floor + 1e-12),
        "floor": floor,
        "utility_scale": float(cfg["voi"].get("utility_scale", 1.0)),
    }


# ---------------------------------------------------------------------------
# Step 3 — the band-tiling grid
# ---------------------------------------------------------------------------
def derive_grid(records: list, cfg: dict, lambda_c: float, refine: int = 1) -> dict:
    """
    Tile divergence bands across the observed VoI range and return the
    lambda_l grid plus every quantity the derivation used.

    The cut formulas are NOT reimplemented here: the affine coefficients
    come from _objective's own primitives (retrieval_latency_s, overlap_s,
    retrieval_compute_units), and every derived price is checked against
    _objective.divergence_band() at that price before it goes in the grid.
    If this file's algebra ever drifts from the controllers' arithmetic,
    the assertion fires instead of the grid quietly describing bands the
    controllers do not have.
    """
    full_s = retrieval_latency_s(cfg)
    # The argmin timing, and so the exposed latency Ours actually pays, is
    # in principle price-dependent. Take it from divergence_band() at a
    # price in the working range rather than assuming, and re-check it at
    # every grid point below.
    probe = divergence_band(Prices(lambda_l=1.0, lambda_c=lambda_c), cfg)
    timing = probe.timing
    exposed = exposed_latency_s(timing, cfg)
    w_ours = retrieval_compute_units(timing, cfg)
    w_b9 = retrieval_compute_units("sentence_end", cfg)
    overlap = overlap_s(timing, cfg)

    def ours_cut(lambda_l: float) -> float:
        return lambda_l * exposed + lambda_c * w_ours

    def b9_cut(lambda_l: float) -> float:
        return lambda_l * full_s + lambda_c * w_b9

    # Below this price the band inverts (Ours's extra waste costs more than
    # its latency saving buys) and the tiling recurrence stalls on it as a
    # fixed point. Strictly excluded from the grid.
    lambda_degenerate = (
        lambda_c * (w_ours - w_b9) / overlap if overlap > 0 else float("inf")
    )

    values = [r["voi"] for r in records]
    voi_max = max(values)
    # Reachable = above the compute floor of ours_cut, which no price can
    # get under. Scenarios below it are never retrieved by either
    # controller at any price; they are reported, not tiled over.
    compute_floor = lambda_c * w_ours
    reachable = [v for v in values if v > compute_floor]
    unreachable = [v for v in values if v <= compute_floor]

    grid, bands, notes = [], [], []
    if not reachable:
        notes.append(
            f"NO GRID: every scenario's VoI is at or below the compute floor "
            f"lambda_c * (1 + waste) = {compute_floor:.4f}, so neither "
            f"controller retrieves at any price. Lower lambda_c or raise the "
            f"VoI floor before sweeping."
        )
    else:
        v_lo = min(reachable)
        # START: band top edge lands on the lowest reachable VoI.
        lambda_l = (v_lo - lambda_c * w_b9) / full_s
        if lambda_l <= lambda_degenerate:
            notes.append(
                f"start price from the data ({lambda_l:.6f}) was at or below the "
                f"degenerate price {lambda_degenerate:.6f}, where the band has "
                f"non-positive width; started just above it instead."
            )
            lambda_l = lambda_degenerate * 1.000001

        step = 0
        # STOP: the first band whose bottom edge has passed the top of the
        # distribution contains nothing, so it is not emitted.
        while ours_cut(lambda_l) < voi_max and step < 10_000:
            band = divergence_band(Prices(lambda_l=lambda_l, lambda_c=lambda_c), cfg)
            assert band.timing == timing, (
                f"argmin timing changed from {timing!r} to {band.timing!r} at "
                f"lambda_l={lambda_l}; the tiling recurrence assumes one timing "
                f"across the grid"
            )
            assert abs(band.ours_cut - ours_cut(lambda_l)) < 1e-9, (
                "this module's ours_cut disagrees with _objective.divergence_band"
            )
            assert abs(band.b9_cut - b9_cut(lambda_l)) < 1e-9, (
                "this module's b9_cut disagrees with _objective.divergence_band"
            )
            grid.append(lambda_l)
            bands.append(
                {
                    "lambda_l": lambda_l,
                    "ours_cut": band.ours_cut,
                    "b9_cut": band.b9_cut,
                    "width": band.width,
                    "timing": band.timing,
                    "n_scenarios_in_band": sum(1 for v in values if band.contains(v)),
                }
            )
            # The tiling recurrence.
            lambda_l = (full_s * lambda_l + lambda_c * (w_b9 - w_ours)) / exposed
            step += 1

    covered = (
        (bands[0]["ours_cut"], bands[-1]["b9_cut"]) if bands else (None, None)
    )
    in_band = sum(b["n_scenarios_in_band"] for b in bands)

    refined = _refine(grid, refine)
    return {
        "lambda_l_grid": refined,
        "lambda_l_tiling_points": grid,
        "lambda_c": lambda_c,
        "refine": refine,
        "bands": bands,
        "covered_voi_range": {"low": covered[0], "high": covered[1]},
        "n_scenarios_in_some_band": in_band,
        "n_scenarios_unreachable_at_any_price": len(unreachable),
        "unreachable_voi_max": max(unreachable) if unreachable else None,
        "compute_floor": compute_floor,
        "lambda_degenerate": lambda_degenerate,
        "constants": {
            "timing": timing,
            "retrieval_latency_s": full_s,
            "exposed_latency_s": exposed,
            "overlap_s": overlap,
            "compute_units_ours": w_ours,
            "compute_units_b9": w_b9,
            "tiling_ratio": full_s / exposed if exposed > 0 else None,
        },
        "notes": notes,
    }


def _refine(grid: list, refine: int) -> list:
    """
    Insert refine-1 points inside each tiling INTERVAL, geometrically
    spaced within that interval.

    Per-interval and not one global ratio: the tiling recurrence is affine,
    not purely geometric (l_{k+1} = 1.3884 * l_k - lambda_c * dwaste /
    exposed), so its step ratio drifts — 1.197 at the bottom of the current
    grid, 1.381 at the top. A single ratio taken from the first pair would
    bunch the added points at one end.

    Nothing is added beyond the last tiling point: by construction that
    band's lower edge is already past the highest observed VoI, so prices
    above it have empty bands and would spend sweep budget on turns where
    the two controllers cannot disagree. Coverage is therefore unchanged;
    only resolution inside it goes up.
    """
    if refine <= 1 or len(grid) < 2:
        return list(grid)
    out = []
    for lower, upper in zip(grid, grid[1:]):
        out.append(lower)
        ratio = (upper / lower) ** (1.0 / refine)
        out.extend(lower * ratio ** sub for sub in range(1, refine))
    out.append(grid[-1])
    return out


# ---------------------------------------------------------------------------
# Report + artifact
# ---------------------------------------------------------------------------
def _print_report(cache, cfg, summary, grid) -> None:
    calibration = load_voi_calibration(cfg)
    print()
    print("=" * 72)
    print("E3 PRICE GRID")
    print("=" * 72)
    print(f"calibration : {calibration.describe()}")
    print(f"              {calibration.source}")
    print(f"signals     : backend={cache['backend']} model={cache['active_model']} "
          f"({cache['n_scenarios']} scenarios, built {cache['built_at_utc'][:19]}Z)")
    precompute = cache.get("precompute") or {}
    if precompute:
        print(f"              protocol v{precompute.get('protocol_version')} "
              f"{precompute.get('protocol')}: mean over "
              f"{precompute.get('n_draws')} draws at temperature "
              f"{precompute.get('temperature')}, seeds {precompute.get('seeds')}, "
              f"max_tokens {precompute.get('max_tokens')} "
              f"(from {precompute.get('protocol_source')})")
    if cache["backend"] == "mock":
        print("              *** MOCK SIGNALS — the calibration is real, the")
        print("              signals it is applied to are synthetic. Re-run")
        print("              against the real backend before reporting. ***")

    print()
    print(f"VoI DISTRIBUTION over {summary['n']} scenarios "
          f"(floor {summary['floor']}, utility_scale {summary['utility_scale']})")
    print(f"  min {summary['min']:.4f}   max {summary['max']:.4f}   "
          f"spread {summary['spread']:.4f}   mean {summary['mean']:.4f}")
    print("  " + "  ".join(
        f"{name} {value:.4f}" for name, value in summary["percentiles"].items()
    ))
    print(f"  at floor: {summary['n_at_floor']}/{summary['n']} scenarios")

    print()
    print(f"BAND TILING at lambda_c = {grid['lambda_c']} "
          f"(timing {grid['constants']['timing']}, ratio "
          f"{grid['constants']['tiling_ratio']:.4f} per step, refine "
          f"{grid['refine']})")
    for band in grid["bands"]:
        print(f"  lambda_l {band['lambda_l']:9.4f}  band "
              f"({band['ours_cut']:.4f}, {band['b9_cut']:.4f}]  "
              f"width {band['width']:+.4f}  "
              f"scenarios {band['n_scenarios_in_band']:3d}")
    low, high = grid["covered_voi_range"]["low"], grid["covered_voi_range"]["high"]
    if low is not None:
        print(f"  covers VoI ({low:.4f}, {high:.4f}] — "
              f"{grid['n_scenarios_in_some_band']}/{summary['n']} scenarios land "
              f"in some band")
    print(f"  unreachable at any price (VoI <= compute floor "
          f"{grid['compute_floor']:.4f}): "
          f"{grid['n_scenarios_unreachable_at_any_price']}/{summary['n']}")
    for note in grid["notes"]:
        print(f"  NOTE: {note}")

    print()
    print(f"lambda_l grid ({len(grid['lambda_l_grid'])} points):")
    print("  " + ", ".join(f"{value:.4f}" for value in grid["lambda_l_grid"]))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive E3's lambda_l price grid by tiling divergence bands across "
            "the VoI distribution of the 50 scenarios. Regenerates "
            "configs/e3_price_grid.json."
        )
    )
    parser.add_argument(
        "--lambda-c",
        type=float,
        default=None,
        help="Compute price the grid is derived at (default: "
        "controller.e3.default_prices.lambda_c from config).",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=1,
        help="Sub-steps per tiling step. 1 (default) is the pure formula; "
        "higher values raise resolution inside the same covered range.",
    )
    parser.add_argument(
        "--refresh-signals",
        action="store_true",
        help="Recompute the per-scenario belief signals instead of using the "
        "committed cache. Needed after changing the LLM backend/model.",
    )
    parser.add_argument(
        "--out",
        default=str(GRID_FILE),
        help="Where to write the grid artifact.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the report without writing the grid artifact.",
    )
    args = parser.parse_args()

    cfg = load_e3_config()
    lambda_c = (
        args.lambda_c
        if args.lambda_c is not None
        else float(cfg["default_prices"]["lambda_c"])
    )

    cache = load_or_build_signals(refresh=args.refresh_signals)
    records = voi_distribution(cache, cfg)
    summary = summarize(records, cfg)
    grid = derive_grid(records, cfg, lambda_c=lambda_c, refine=args.refine)

    _print_report(cache, cfg, summary, grid)

    calibration = load_voi_calibration(cfg)
    artifact = {
        "_comment": (
            "GENERATED by `python -m simulator.e3_grid` — do not hand-edit. "
            "E3's lambda_l price grid, derived by tiling divergence bands "
            "across the VoI distribution of simulator/scenarios.json. See that "
            "module's docstring for the formula and its endpoints."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": (
            f"python -m simulator.e3_grid --lambda-c {lambda_c} "
            f"--refine {args.refine}"
        ),
        "calibration": {
            "signal": calibration.signal,
            "kind": calibration.kind,
            "slope": calibration.slope,
            "intercept": calibration.intercept,
            "calibrated": calibration.calibrated,
            "auc": calibration.auc,
            "source": calibration.source,
        },
        "signals": {
            "cache": "simulator/e3_scenario_signals.json",
            "backend": cache["backend"],
            "active_model": cache["active_model"],
            "built_at_utc": cache["built_at_utc"],
            "n_scenarios": cache["n_scenarios"],
            "are_real_model_signals": cache["backend"] != "mock",
            # The sampling protocol behind those signals. A grid derived from
            # one unseeded draw and one derived from the E1-matched mean are
            # different artifacts; this is what tells them apart.
            "precompute": cache.get("precompute"),
        },
        "voi_distribution": summary,
        "voi_per_scenario": records,
        "grid": grid,
    }

    if args.dry_run:
        print("(--dry-run: nothing written)")
        return 0

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
