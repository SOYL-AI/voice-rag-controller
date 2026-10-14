"""
E3 Frontier Analysis
====================

Loads the provisional E3 result schema and produces:

1. Aggregated operating points:
      quality  -> mean answer_correct
      latency  -> p95 TTFA
      compute  -> mean retrievals

2. Bootstrap 95% confidence bands over scenarios.

3. Frontier figures:
      - accuracy vs p95 TTFA
      - accuracy vs compute

4. Pre-registered Ours vs B9 dominance analysis.

Baselines:
    B0 / B4
        Unpriced fixed heuristics.
        One operating point per seed.

    B9 / Ours
        Price-based policies.
        One operating point per seed + (lambda_l, lambda_c).

Provisional schema:

{
    "baseline": "ours",
    "seed": 11,
    "scenario_id": "...",
    "prices": {
        "lambda_l": 0.533,
        "lambda_c": 0.050
    },
    "outcome": {
        "answer_correct": true,
        "ttfa_ms": 3395.3,
        "retrievals": 6
    }
}

For B0/B4:
    "prices": null

Usage:

    python -m analysis.e3_frontier e3_sample_result

Expected output files:

    analysis/
        e3_accuracy_vs_p95_ttfa.png
        e3_accuracy_vs_compute.png
        e3_dominance.json
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt


# ==========================================================================
# Configuration
# ==========================================================================

BOOTSTRAP_ITERATIONS = 2000

# Fixed seed so results are reproducible.
BOOTSTRAP_SEED = 12345

# The preregistered requirement is dominance in >= 3 of 5 seeds.
REQUIRED_DOMINANT_SEEDS = 3

# A single operating point is not treated as a "region".
# Therefore a contiguous region requires at least 2 adjacent operating points.
MIN_CONTIGUOUS_POINTS = 2

EXPECTED_SEEDS = {11, 23, 37, 53, 71}

FRONTIER_BASELINES = [
    "b0",
    "b4",
    "b9",
    "ours",
]


# ==========================================================================
# Percentile
# ==========================================================================

def percentile(values, p):
    """
    Compute a percentile using linear interpolation.
    """

    if not values:
        return float("nan")

    values = sorted(float(value) for value in values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * (p / 100.0)

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower]
        + fraction * (
            values[upper] - values[lower]
        )
    )


# ==========================================================================
# Load result files
# ==========================================================================

def load_result_files(results_dir):
    """
    Load all E3 JSON result files.

    Expected files include:

        e3_b0_seed*.json
        e3_b4_seed*.json
        e3_b9_seed*.json
        e3_ours_seed*.json

    The provisional sample set may also use a different naming convention,
    so all e3_*.json files are loaded.
    """

    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    files = sorted(
        results_dir.glob("e3_*.json")
    )

    if not files:
        raise FileNotFoundError(
            f"No E3 result files found in {results_dir}"
        )

    records = []

    for path in files:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                f"{path} does not contain a JSON list"
            )

        for row in data:

            if not isinstance(row, dict):
                raise ValueError(
                    f"{path} contains a non-object record"
                )

            row["_source_file"] = path.name

            records.append(row)

    return records


# ==========================================================================
# Validate records
# ==========================================================================

def validate_records(records):
    """
    Validate the provisional E3 schema.

    The schema is intentionally kept flexible because the manager said
    the real-sweep field names may still be locked later.
    """

    required_fields = {
        "baseline",
        "seed",
        "scenario_id",
        "prices",
        "outcome",
    }

    required_outcome_fields = {
        "answer_correct",
        "ttfa_ms",
        "retrievals",
    }

    for index, row in enumerate(records):

        if not isinstance(row, dict):
            raise ValueError(
                f"Record {index} is not an object."
            )

        missing = required_fields - row.keys()

        if missing:
            raise ValueError(
                f"Record {index} is missing fields: "
                f"{sorted(missing)}"
            )

        outcome = row["outcome"]

        if not isinstance(outcome, dict):
            raise ValueError(
                f"Record {index}: outcome must be an object."
            )

        missing_outcome = (
            required_outcome_fields - outcome.keys()
        )

        if missing_outcome:
            raise ValueError(
                f"Record {index} is missing outcome fields: "
                f"{sorted(missing_outcome)}"
            )

        prices = row["prices"]

        if prices is None:
            continue

        if not isinstance(prices, dict):
            raise ValueError(
                f"Record {index}: prices must be "
                f"an object or null."
            )

        if (
            "lambda_l" not in prices
            or "lambda_c" not in prices
        ):
            raise ValueError(
                f"Record {index}: priced record is missing "
                f"lambda_l or lambda_c."
            )


# ==========================================================================
# v2 schema extension (simulator/e3v2_decide.py)
# ==========================================================================

# Extra fields every v2 record carries on top of the v1 schema above. The
# v1 validator tolerates extras, so v2 records pass it unchanged; this adds
# the v2-specific checks.
V2_RECORD_FIELDS = {
    "dispatch_t_ms",       # int ms on the speech-onset clock, or null (skip)
    "condition",           # voice/noise condition of the speech trace
    "latency_profile",     # faiss | faiss_rerank | remote_api
    "exposed_latency_ms",  # retrieval latency the user waited for (0 on skip)
    "architecture",        # commit | speculative (a speculative dispatch may be discarded)
}


def is_v2_records(records):
    return bool(records) and all(
        isinstance(row, dict) and "latency_profile" in row
        for row in records
    )


def validate_v2_records(records):
    """
    v1 validation plus the v2 fields. A retrieving record must carry a
    dispatch time; a skipping one must not, and its exposed latency is 0.
    """

    validate_records(records)

    for index, row in enumerate(records):

        missing = V2_RECORD_FIELDS - row.keys()

        if missing:
            raise ValueError(
                f"v2 record {index} is missing fields: "
                f"{sorted(missing)}"
            )

        retrieved = int(row["outcome"]["retrievals"]) > 0

        if retrieved and row["dispatch_t_ms"] is None:
            raise ValueError(
                f"v2 record {index} retrieved but has no dispatch_t_ms."
            )

        if not retrieved and row["dispatch_t_ms"] is not None:
            raise ValueError(
                f"v2 record {index} skipped but has a dispatch_t_ms."
            )

        if row["architecture"] not in ("commit", "speculative"):
            raise ValueError(
                f"v2 record {index} has unknown architecture "
                f"{row['architecture']!r}."
            )

        if row.get("used") is False and row["architecture"] != "speculative":
            raise ValueError(
                f"v2 record {index} discarded a retrieval under "
                f"the commit architecture."
            )

        if float(row["exposed_latency_ms"]) < 0:
            raise ValueError(
                f"v2 record {index} has negative exposed_latency_ms."
            )


def select_v2_stratum(records, latency_profile=None, condition=None,
                      architecture=None):
    """
    The v1 aggregation groups by (baseline, seed, prices) only, so v2
    records must be fed ONE latency profile (and optionally one condition)
    at a time — pooling profiles would average operating points that live
    on different latency scales. Raises if several profiles remain.
    """

    rows = [
        row for row in records
        if (latency_profile is None or row["latency_profile"] == latency_profile)
        and (condition is None or row["condition"] == condition)
        and (
            architecture is None
            or row.get("architecture", "commit") == architecture
        )
    ]

    profiles = sorted({row["latency_profile"] for row in rows})

    if len(profiles) > 1:
        raise ValueError(
            f"v2 records span latency profiles {profiles}; "
            f"pass --v2-profile to analyse one at a time."
        )

    architectures = sorted({row.get("architecture", "commit") for row in rows})

    if len(architectures) > 1:
        raise ValueError(
            f"v2 records span architectures {architectures}; "
            f"pass --v2-architecture to analyse one at a time."
        )

    return rows


# ==========================================================================
# Bootstrap helper
# ==========================================================================

def bootstrap_metrics(
    scenarios,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=BOOTSTRAP_SEED,
):
    """
    Bootstrap over scenarios.

    For each bootstrap sample we calculate:

        accuracy
        p95 TTFA
        mean retrievals

    Returns 95% confidence intervals for all three metrics.
    """

    if not scenarios:
        return {
            "accuracy": {
                "low": float("nan"),
                "median": float("nan"),
                "high": float("nan"),
            },
            "p95_ttfa_ms": {
                "low": float("nan"),
                "median": float("nan"),
                "high": float("nan"),
            },
            "compute": {
                "low": float("nan"),
                "median": float("nan"),
                "high": float("nan"),
            },
        }

    rng = random.Random(seed)

    n = len(scenarios)

    accuracy_samples = []
    ttfa_samples = []
    compute_samples = []

    for _ in range(iterations):

        sample = [
            scenarios[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        accuracy = (
            sum(
                1
                for scenario in sample
                if scenario["answer_correct"]
            )
            / n
        )

        ttfa_values = [
            scenario["ttfa_ms"]
            for scenario in sample
        ]

        retrieval_values = [
            scenario["retrievals"]
            for scenario in sample
        ]

        p95_ttfa = percentile(
            ttfa_values,
            95,
        )

        compute = (
            sum(retrieval_values)
            / n
        )

        accuracy_samples.append(
            accuracy
        )

        ttfa_samples.append(
            p95_ttfa
        )

        compute_samples.append(
            compute
        )

    return {
        "accuracy": {
            "low": percentile(
                accuracy_samples,
                2.5,
            ),
            "median": percentile(
                accuracy_samples,
                50,
            ),
            "high": percentile(
                accuracy_samples,
                97.5,
            ),
        },
        "p95_ttfa_ms": {
            "low": percentile(
                ttfa_samples,
                2.5,
            ),
            "median": percentile(
                ttfa_samples,
                50,
            ),
            "high": percentile(
                ttfa_samples,
                97.5,
            ),
        },
        "compute": {
            "low": percentile(
                compute_samples,
                2.5,
            ),
            "median": percentile(
                compute_samples,
                50,
            ),
            "high": percentile(
                compute_samples,
                97.5,
            ),
        },
    }


# ==========================================================================
# Aggregate frontier points
# ==========================================================================

def aggregate_frontier(records):
    """
    Aggregate scenario records into operating points.

    B0 / B4:
        baseline + seed

    B9 / Ours:
        baseline + seed + lambda_l + lambda_c

    Each operating point contains:

        accuracy
        p95_ttfa_ms
        compute

    Plus bootstrap 95% confidence intervals.
    """

    groups = {}

    for row in records:

        baseline = str(
            row["baseline"]
        ).lower()

        seed = int(
            row["seed"]
        )

        prices = row.get("prices")

        if prices is None:

            lambda_l = None
            lambda_c = None

        else:

            lambda_l = float(
                prices["lambda_l"]
            )

            lambda_c = float(
                prices["lambda_c"]
            )

        outcome = row["outcome"]

        key = (
            baseline,
            seed,
            lambda_l,
            lambda_c,
        )

        if key not in groups:

            groups[key] = {
                "baseline": baseline,
                "seed": seed,
                "lambda_l": lambda_l,
                "lambda_c": lambda_c,
                "scenarios": [],
            }

        groups[key]["scenarios"].append(
            {
                "scenario_id": row["scenario_id"],

                "answer_correct": bool(
                    outcome["answer_correct"]
                ),

                "ttfa_ms": float(
                    outcome["ttfa_ms"]
                ),

                "retrievals": float(
                    outcome["retrievals"]
                ),
            }
        )

    aggregated = []

    for group in groups.values():

        scenarios = group["scenarios"]

        if not scenarios:
            continue

        # --------------------------------------------------------------
        # Point estimates
        # --------------------------------------------------------------

        accuracy = (
            sum(
                1
                for scenario in scenarios
                if scenario["answer_correct"]
            )
            / len(scenarios)
        )

        ttfa_values = [
            scenario["ttfa_ms"]
            for scenario in scenarios
        ]

        p95_ttfa_ms = percentile(
            ttfa_values,
            95,
        )

        retrieval_values = [
            scenario["retrievals"]
            for scenario in scenarios
        ]

        compute = (
            sum(retrieval_values)
            / len(retrieval_values)
        )

        # --------------------------------------------------------------
        # Bootstrap
        # --------------------------------------------------------------

        bootstrap_seed = (
            BOOTSTRAP_SEED
            + group["seed"]
            + int(
                round(
                    (group["lambda_l"] or 0)
                    * 100000
                )
            )
            + int(
                round(
                    (group["lambda_c"] or 0)
                    * 100000
                )
            )
        )

        bootstrap = bootstrap_metrics(
            scenarios,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=bootstrap_seed,
        )

        # --------------------------------------------------------------
        # Store
        # --------------------------------------------------------------

        aggregated.append(
            {
                "baseline": group["baseline"],
                "seed": group["seed"],

                "lambda_l": group["lambda_l"],
                "lambda_c": group["lambda_c"],

                "n_scenarios": len(
                    scenarios
                ),

                "accuracy": accuracy,

                "accuracy_ci_low": bootstrap[
                    "accuracy"
                ]["low"],

                "accuracy_ci_high": bootstrap[
                    "accuracy"
                ]["high"],

                "p95_ttfa_ms": p95_ttfa_ms,

                "p95_ttfa_ci_low": bootstrap[
                    "p95_ttfa_ms"
                ]["low"],

                "p95_ttfa_ci_high": bootstrap[
                    "p95_ttfa_ms"
                ]["high"],

                "compute": compute,

                "compute_ci_low": bootstrap[
                    "compute"
                ]["low"],

                "compute_ci_high": bootstrap[
                    "compute"
                ]["high"],

                # Retain raw scenarios for traceability.
                "scenarios": scenarios,
            }
        )

    return sorted(
        aggregated,
        key=lambda row: (
            row["baseline"],
            row["seed"],
            row["lambda_l"] is not None,
            (
                row["lambda_l"]
                if row["lambda_l"] is not None
                else 0
            ),
            (
                row["lambda_c"]
                if row["lambda_c"] is not None
                else 0
            ),
        ),
    )


# ==========================================================================
# Print summary
# ==========================================================================

def print_summary(aggregated):

    print()
    print("=" * 70)
    print("E3 FRONTIER AGGREGATION")
    print("=" * 70)

    if not aggregated:

        print(
            "No operating points found."
        )

        return

    baselines = sorted(
        set(
            row["baseline"]
            for row in aggregated
        )
    )

    print(
        f"Baselines: {', '.join(baselines)}"
    )

    print()

    print(
        f"Total operating points: "
        f"{len(aggregated)}"
    )

    for baseline in baselines:

        rows = [
            row
            for row in aggregated
            if row["baseline"] == baseline
        ]

        seeds = sorted(
            set(
                row["seed"]
                for row in rows
            )
        )

        print()
        print("-" * 70)
        print(baseline)
        print("-" * 70)

        print(
            f"Seeds: {seeds}"
        )

        print(
            f"Operating points: "
            f"{len(rows)}"
        )

        for seed in seeds:

            seed_rows = [
                row
                for row in rows
                if row["seed"] == seed
            ]

            print()
            print(
                f"  Seed {seed}: "
                f"{len(seed_rows)} operating point(s)"
            )

            for row in seed_rows:

                if (
                    row["lambda_l"] is None
                    and row["lambda_c"] is None
                ):

                    price_text = (
                        "unpriced"
                    )

                else:

                    price_text = (
                        f"lambda_l="
                        f"{row['lambda_l']:.3f}, "
                        f"lambda_c="
                        f"{row['lambda_c']:.3f}"
                    )

                print(
                    f"    {price_text} | "
                    f"n={row['n_scenarios']} | "
                    f"accuracy="
                    f"{row['accuracy']:.3f} "
                    f"["
                    f"{row['accuracy_ci_low']:.3f}, "
                    f"{row['accuracy_ci_high']:.3f}"
                    f"] | "
                    f"p95 TTFA="
                    f"{row['p95_ttfa_ms']:.1f} ms | "
                    f"compute="
                    f"{row['compute']:.3f}"
                )


# ==========================================================================
# Frontier plotting
# ==========================================================================

def generate_frontier_plots(
    aggregated,
    output_dir="analysis",
    baselines=None,
):
    """
    Generate:

        1. Accuracy vs p95 TTFA
        2. Accuracy vs compute

    Error bars represent bootstrap 95% confidence intervals.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Accuracy vs p95 TTFA
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(11, 7)
    )

    for baseline in (baselines or FRONTIER_BASELINES):

        rows = [
            row
            for row in aggregated
            if row["baseline"] == baseline
        ]

        if not rows:
            continue

        # For priced baselines, sort by lambda_l.
        # For fixed baselines, this simply gives stable ordering.
        rows = sorted(
            rows,
            key=lambda row: (
                row["seed"],
                row["lambda_l"]
                if row["lambda_l"] is not None
                else -float("inf"),
            ),
        )

        # Average across seeds for each price point.
        price_groups = {}

        for row in rows:

            key = (
                row["lambda_l"],
                row["lambda_c"],
            )

            price_groups.setdefault(
                key,
                []
            ).append(row)

        frontier_rows = []

        for price, price_rows in price_groups.items():

            accuracy_values = [
                row["accuracy"]
                for row in price_rows
            ]

            accuracy_low_values = [
                row["accuracy_ci_low"]
                for row in price_rows
            ]

            accuracy_high_values = [
                row["accuracy_ci_high"]
                for row in price_rows
            ]

            ttfa_values = [
                row["p95_ttfa_ms"]
                for row in price_rows
            ]

            ttfa_low_values = [
                row["p95_ttfa_ci_low"]
                for row in price_rows
            ]

            ttfa_high_values = [
                row["p95_ttfa_ci_high"]
                for row in price_rows
            ]

            frontier_rows.append(
                {
                    "lambda_l": price[0],
                    "lambda_c": price[1],

                    "accuracy": (
                        sum(accuracy_values)
                        / len(accuracy_values)
                    ),

                    "accuracy_low": (
                        sum(accuracy_low_values)
                        / len(
                            accuracy_low_values
                        )
                    ),

                    "accuracy_high": (
                        sum(accuracy_high_values)
                        / len(
                            accuracy_high_values
                        )
                    ),

                    "ttfa": (
                        sum(ttfa_values)
                        / len(ttfa_values)
                    ),

                    "ttfa_low": (
                        sum(ttfa_low_values)
                        / len(
                            ttfa_low_values
                        )
                    ),

                    "ttfa_high": (
                        sum(ttfa_high_values)
                        / len(
                            ttfa_high_values
                        )
                    ),
                }
            )

        frontier_rows = sorted(
            frontier_rows,
            key=lambda row: (
                row["ttfa"]
            ),
        )

        if not frontier_rows:
            continue

        x = [
            row["ttfa"]
            for row in frontier_rows
        ]

        y = [
            row["accuracy"]
            for row in frontier_rows
        ]

        x_low = [
            row["ttfa"] - row["ttfa_low"]
            for row in frontier_rows
        ]

        x_high = [
            row["ttfa_high"] - row["ttfa"]
            for row in frontier_rows
        ]

        y_low = [
            row["accuracy"]
            - row["accuracy_low"]
            for row in frontier_rows
        ]

        y_high = [
            row["accuracy_high"]
            - row["accuracy"]
            for row in frontier_rows
        ]

        plt.errorbar(
            x,
            y,
            xerr=[
                x_low,
                x_high,
            ],
            yerr=[
                y_low,
                y_high,
            ],
            marker="o",
            capsize=3,
            linewidth=1,
            label=baseline,
        )

    plt.xlabel(
        "p95 TTFA (ms)"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "E3 Accuracy vs p95 TTFA Frontier"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    ttfa_path = (
        output_dir
        / "e3_accuracy_vs_p95_ttfa.png"
    )

    plt.savefig(
        ttfa_path,
        dpi=200,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Accuracy vs Compute
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(11, 7)
    )

    for baseline in (baselines or FRONTIER_BASELINES):

        rows = [
            row
            for row in aggregated
            if row["baseline"] == baseline
        ]

        if not rows:
            continue

        price_groups = {}

        for row in rows:

            key = (
                row["lambda_l"],
                row["lambda_c"],
            )

            price_groups.setdefault(
                key,
                []
            ).append(row)

        frontier_rows = []

        for price, price_rows in price_groups.items():

            accuracy_values = [
                row["accuracy"]
                for row in price_rows
            ]

            accuracy_low_values = [
                row["accuracy_ci_low"]
                for row in price_rows
            ]

            accuracy_high_values = [
                row["accuracy_ci_high"]
                for row in price_rows
            ]

            compute_values = [
                row["compute"]
                for row in price_rows
            ]

            compute_low_values = [
                row["compute_ci_low"]
                for row in price_rows
            ]

            compute_high_values = [
                row["compute_ci_high"]
                for row in price_rows
            ]

            frontier_rows.append(
                {
                    "lambda_l": price[0],
                    "lambda_c": price[1],

                    "accuracy": (
                        sum(accuracy_values)
                        / len(accuracy_values)
                    ),

                    "accuracy_low": (
                        sum(accuracy_low_values)
                        / len(
                            accuracy_low_values
                        )
                    ),

                    "accuracy_high": (
                        sum(accuracy_high_values)
                        / len(
                            accuracy_high_values
                        )
                    ),

                    "compute": (
                        sum(compute_values)
                        / len(compute_values)
                    ),

                    "compute_low": (
                        sum(compute_low_values)
                        / len(
                            compute_low_values
                        )
                    ),

                    "compute_high": (
                        sum(compute_high_values)
                        / len(
                            compute_high_values
                        )
                    ),
                }
            )

        frontier_rows = sorted(
            frontier_rows,
            key=lambda row: (
                row["compute"]
            ),
        )

        if not frontier_rows:
            continue

        x = [
            row["compute"]
            for row in frontier_rows
        ]

        y = [
            row["accuracy"]
            for row in frontier_rows
        ]

        x_low = [
            row["compute"]
            - row["compute_low"]
            for row in frontier_rows
        ]

        x_high = [
            row["compute_high"]
            - row["compute"]
            for row in frontier_rows
        ]

        y_low = [
            row["accuracy"]
            - row["accuracy_low"]
            for row in frontier_rows
        ]

        y_high = [
            row["accuracy_high"]
            - row["accuracy"]
            for row in frontier_rows
        ]

        plt.errorbar(
            x,
            y,
            xerr=[
                x_low,
                x_high,
            ],
            yerr=[
                y_low,
                y_high,
            ],
            marker="o",
            capsize=3,
            linewidth=1,
            label=baseline,
        )

    plt.xlabel(
        "Mean Retrievals / Compute"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "E3 Accuracy vs Compute Frontier"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    compute_path = (
        output_dir
        / "e3_accuracy_vs_compute.png"
    )

    plt.savefig(
        compute_path,
        dpi=200,
    )

    plt.close()

    print()
    print("=" * 70)
    print("E3 FRONTIER PLOTS GENERATED")
    print("=" * 70)

    print(
        ttfa_path
    )

    print(
        compute_path
    )


# ==========================================================================
# Dominance helpers
# ==========================================================================

def same_operating_point(
    ours,
    b9,
):
    """
    Match Ours and B9 at the same price operating point.
    """

    if (
        ours["lambda_l"] is None
        or ours["lambda_c"] is None
        or b9["lambda_l"] is None
        or b9["lambda_c"] is None
    ):
        return False

    return (
        abs(
            ours["lambda_l"]
            - b9["lambda_l"]
        ) < 1e-12
        and
        abs(
            ours["lambda_c"]
            - b9["lambda_c"]
        ) < 1e-12
    )


def is_strictly_dominant(
    ours,
    b9,
):
    """
    Conservative strict dominance test.

    Ours must have:

        accuracy lower CI > B9 accuracy upper CI

    AND

        p95 TTFA upper CI <= B9 p95 TTFA lower CI

    AND

        compute upper CI <= B9 compute lower CI

    The non-overlapping accuracy bands provide the strict quality
    requirement. Latency and compute must be no worse.
    """

    accuracy_dominates = (
        ours["accuracy_ci_low"]
        > b9["accuracy_ci_high"]
    )

    latency_not_worse = (
        ours["p95_ttfa_ci_high"]
        <= b9["p95_ttfa_ci_low"]
    )

    compute_not_worse = (
        ours["compute_ci_high"]
        <= b9["compute_ci_low"]
    )

    return (
        accuracy_dominates
        and latency_not_worse
        and compute_not_worse
    )


def longest_contiguous_run(indices):
    """
    Return the longest contiguous run in a sorted list of indices.
    """

    if not indices:
        return 0

    indices = sorted(
        set(indices)
    )

    longest = 1
    current = 1

    for index in range(
        1,
        len(indices),
    ):

        if (
            indices[index]
            == indices[index - 1] + 1
        ):

            current += 1

            longest = max(
                longest,
                current,
            )

        else:

            current = 1

    return longest


# ==========================================================================
# Dominance test
# ==========================================================================

def dominance_test(
    aggregated,
    output_dir="analysis",
    ours="ours",
    rival="b9",
):
    """
    Pre-registered dominance test of `ours` against `rival` (default: the
    v1 pair Ours vs B9; v2 compares ours_v2 against b9plus_v2,
    b10_speculative_v2 or b9_v2). The output keeps the key "ours" for the
    challenger and uses the rival's baseline name as its key, so the
    default v1 output is unchanged.

    Pre-registered Ours vs B9 dominance test.

    For each seed:

      1. Match Ours and B9 at the same price operating points.
      2. Test strict dominance.
      3. Find contiguous runs of dominated operating points.
      4. Require a contiguous region of >= 2 points.
      5. Count seeds satisfying the criterion.

    Overall preregistered criterion:

        Ours strictly dominates B9 in >= 3 of 5 seeds.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = sorted(
        set(
            row["seed"]
            for row in aggregated
            if row["baseline"]
            in {rival, ours}
        )
    )

    per_seed_results = []

    print()
    print("=" * 70)
    print(f"E3 DOMINANCE TEST: {ours} vs {rival}")
    print("=" * 70)

    for seed in seeds:

        ours_rows = sorted(
            [
                row
                for row in aggregated
                if (
                    row["baseline"]
                    == ours
                    and row["seed"]
                    == seed
                )
            ],
            key=lambda row: (
                row["lambda_l"],
                row["lambda_c"],
            ),
        )

        b9_rows = [
            row
            for row in aggregated
            if (
                row["baseline"]
                == rival
                and row["seed"]
                == seed
            )
        ]

        dominated_indices = []
        matched_points = []

        for ours_index, ours in enumerate(
            ours_rows
        ):

            matching_b9 = None

            for b9 in b9_rows:

                if same_operating_point(
                    ours,
                    b9,
                ):

                    matching_b9 = b9
                    break

            if matching_b9 is None:
                continue

            dominated = is_strictly_dominant(
                ours,
                matching_b9,
            )

            matched_points.append(
                {
                    "index": ours_index,
                    "lambda_l": ours[
                        "lambda_l"
                    ],
                    "lambda_c": ours[
                        "lambda_c"
                    ],
                    "dominated": dominated,
                    "ours": {
                        "accuracy": ours[
                            "accuracy"
                        ],
                        "accuracy_ci_low": ours[
                            "accuracy_ci_low"
                        ],
                        "accuracy_ci_high": ours[
                            "accuracy_ci_high"
                        ],
                        "p95_ttfa_ms": ours[
                            "p95_ttfa_ms"
                        ],
                        "p95_ttfa_ci_low": ours[
                            "p95_ttfa_ci_low"
                        ],
                        "p95_ttfa_ci_high": ours[
                            "p95_ttfa_ci_high"
                        ],
                        "compute": ours[
                            "compute"
                        ],
                        "compute_ci_low": ours[
                            "compute_ci_low"
                        ],
                        "compute_ci_high": ours[
                            "compute_ci_high"
                        ],
                    },
                    rival: {
                        "accuracy": matching_b9[
                            "accuracy"
                        ],
                        "accuracy_ci_low": matching_b9[
                            "accuracy_ci_low"
                        ],
                        "accuracy_ci_high": matching_b9[
                            "accuracy_ci_high"
                        ],
                        "p95_ttfa_ms": matching_b9[
                            "p95_ttfa_ms"
                        ],
                        "p95_ttfa_ci_low": matching_b9[
                            "p95_ttfa_ci_low"
                        ],
                        "p95_ttfa_ci_high": matching_b9[
                            "p95_ttfa_ci_high"
                        ],
                        "compute": matching_b9[
                            "compute"
                        ],
                        "compute_ci_low": matching_b9[
                            "compute_ci_low"
                        ],
                        "compute_ci_high": matching_b9[
                            "compute_ci_high"
                        ],
                    },
                }
            )

            if dominated:
                dominated_indices.append(
                    ours_index
                )

        longest_run = longest_contiguous_run(
            dominated_indices
        )

        contiguous_region = (
            longest_run
            >= MIN_CONTIGUOUS_POINTS
        )

        verdict = (
            "STRICT DOMINANCE"
            if contiguous_region
            else "NO STRICT DOMINANCE"
        )

        result = {
            "seed": seed,
            "matched_operating_points": len(
                matched_points
            ),
            "dominated_operating_points": len(
                dominated_indices
            ),
            "dominated_indices": dominated_indices,
            "longest_contiguous_run": longest_run,
            "contiguous_region": contiguous_region,
            "verdict": verdict,
            "operating_points": matched_points,
        }

        per_seed_results.append(
            result
        )

        print()
        print(
            f"Seed {seed}: {verdict}"
        )

        print(
            f"  Matched operating points: "
            f"{len(matched_points)}"
        )

        print(
            f"  Dominated points: "
            f"{len(dominated_indices)}"
        )

        print(
            f"  Longest contiguous run: "
            f"{longest_run}"
        )

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------

    dominance_count = sum(
        1
        for result in per_seed_results
        if result["contiguous_region"]
    )

    total_expected_seeds = len(
        EXPECTED_SEEDS
    )

    if (
        dominance_count
        >= REQUIRED_DOMINANT_SEEDS
    ):

        overall_verdict = (
            f"{ours} strictly dominates {rival} "
            f"over a contiguous region in "
            f"{dominance_count}/{total_expected_seeds} "
            f"seeds — PREREGISTERED CRITERION MET."
        )

    else:

        overall_verdict = (
            f"{ours} does NOT meet the "
            "preregistered dominance criterion: "
            f"{dominance_count}/{total_expected_seeds} "
            f"seeds satisfy contiguous strict dominance "
            f"(requires >= "
            f"{REQUIRED_DOMINANT_SEEDS})."
        )

    print()
    print("-" * 70)
    print(
        f"Seeds with strict dominance: "
        f"{dominance_count}/{total_expected_seeds}"
    )

    print(
        f"Required: "
        f">= {REQUIRED_DOMINANT_SEEDS}/"
        f"{total_expected_seeds}"
    )

    print()
    print(
        f"VERDICT: {overall_verdict}"
    )

    # ------------------------------------------------------------------
    # Save machine-readable verdict
    # ------------------------------------------------------------------

    output = {
        "criterion": {
            "required_dominant_seeds":
                REQUIRED_DOMINANT_SEEDS,
            "expected_seeds":
                sorted(EXPECTED_SEEDS),
            "minimum_contiguous_points":
                MIN_CONTIGUOUS_POINTS,
            "bootstrap_iterations":
                BOOTSTRAP_ITERATIONS,
            "description": (
                "Ours must have a strictly higher "
                "accuracy with non-overlapping 95% "
                "bootstrap bands while having no "
                f"higher p95 TTFA or compute than {rival} "
                "at matched price operating points, "
                "over a contiguous region, in at "
                "least 3 of 5 seeds."
            ),
        },

        "compared": {"ours": ours, "rival": rival},

        "seeds_with_strict_dominance":
            dominance_count,

        "overall_criterion_met":
            dominance_count
            >= REQUIRED_DOMINANT_SEEDS,

        "verdict":
            overall_verdict,

        "per_seed":
            per_seed_results,
    }

    dominance_path = (
        output_dir
        / (
            "e3_dominance.json"
            if (ours, rival) == ("ours", "b9")
            else f"e3_dominance_{ours}_vs_{rival}.json"
        )
    )

    with open(
        dominance_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print(
        f"Saved dominance result: "
        f"{dominance_path}"
    )

    return output


# ==========================================================================
# Main
# ==========================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "E3 frontier analysis: aggregate "
            "quality/latency/compute, bootstrap "
            "confidence bands, generate frontier "
            "figures, and test Ours vs B9 dominance."
        )
    )

    parser.add_argument(
        "results_dir",
        help=(
            "Directory containing e3_*.json "
            "result files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="analysis",
        help=(
            "Directory for generated figures "
            "and dominance JSON."
        ),
    )

    parser.add_argument(
        "--ours",
        default="ours",
        help="challenger baseline for the dominance test (v2: ours_v2).",
    )

    parser.add_argument(
        "--rival",
        default="b9",
        help=(
            "rival baseline for the dominance test "
            "(v2: b9plus_v2, b10_speculative_v2 or b9_v2)."
        ),
    )

    parser.add_argument(
        "--baselines",
        default=None,
        help="comma-separated baselines to plot (default: FRONTIER_BASELINES).",
    )

    parser.add_argument(
        "--v2-architecture",
        default=None,
        help="v2 records only: commit | speculative.",
    )

    parser.add_argument(
        "--v2-profile",
        default=None,
        help="v2 records only: the latency profile to analyse.",
    )

    parser.add_argument(
        "--v2-condition",
        default=None,
        help="v2 records only: restrict to one speech condition.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    records = load_result_files(
        args.results_dir
    )

    print(
        f"Loaded {len(records)} scenario records."
    )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    if is_v2_records(records):
        validate_v2_records(records)
        records = select_v2_stratum(
            records, args.v2_profile, args.v2_condition,
            args.v2_architecture,
        )
        print(f"v2 stratum: {len(records)} records")
    else:
        validate_records(
            records
        )

    print(
        "Schema validation: OK"
    )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    aggregated = aggregate_frontier(
        records
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print_summary(
        aggregated
    )

    # ------------------------------------------------------------------
    # Frontier figures
    # ------------------------------------------------------------------

    generate_frontier_plots(
        aggregated,
        output_dir=args.output_dir,
        baselines=(
            args.baselines.split(",")
            if args.baselines
            else None
        ),
    )

    # ------------------------------------------------------------------
    # Dominance test
    # ------------------------------------------------------------------

    dominance_test(
        aggregated,
        output_dir=args.output_dir,
        ours=args.ours,
        rival=args.rival,
    )


if __name__ == "__main__":
    main()