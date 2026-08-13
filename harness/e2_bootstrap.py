"""
Experiment 2: paired bootstrap significance test for TTFA across timing policies.

Compares the two early-retrieval policies (`immediate`, `fixed_early_proxy`) against
the baseline (`sentence_end`) on Time To First Audio (TTFA), using a paired bootstrap
resampled over SCENARIOS -- not over individual runs. Accuracy (hit@3) is 100% and
equal across all three policies in this dataset, so TTFA is the only thing that can
differ, and comparing it is valid.

Why scenarios, not runs, are the resampling unit
--------------------------------------------------
Each scenario is run 5 times (seeds 0-4) per policy. Those 5 runs are NOT independent
draws of "how does this policy perform" -- they're 5 noisy measurements of the same
underlying (scenario, policy) latency, differing only in generation seed. Treating
all 750 runs as 750 independent samples (the naive approach) would pretend we have
5x more independent evidence than we do, and would let a single easy/hard scenario's
5 seeds dominate the resampling distribution. The paired bootstrap handbook recipe
here instead:
  1. Collapses each (scenario, policy)'s 5 seeds to one robust number (median).
  2. Resamples whole SCENARIOS (with replacement), each carrying its one number.
This treats "a scenario" as the unit of independent evidence, which is what it is --
the 50 scenarios are the actual independent draws in this experiment design.

Usage
-----
    python -m harness.e2_bootstrap logs/
    python -m harness.e2_bootstrap logs/ --seed 42 --iterations 10000
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse the exact conversation_id parser Experiment 2's hit@k analysis already uses --
# do not reinvent policy/seed/scenario_id parsing. See harness/metrics.py for the
# conv_<policy>_s<seed>_<scenario_id> naming this expects (batch_runner.py's
# run_sweep() naming); warmup_*.jsonl files and anything else that doesn't match
# conv_... simply parse to None and get skipped below.
from harness.metrics import parse_conversation_id

POLICIES = ("sentence_end", "immediate", "fixed_early_proxy")
BASELINE = "sentence_end"
EARLY_POLICIES = ("immediate", "fixed_early_proxy")


# ======================================================================
# Step 1: robust per-log TTFA extraction
# ======================================================================
#
# TTFA is defined exactly as in harness/metrics.py:
#     TTFA = tts_first_audio.t_ms - user_end.t_ms
#
# The naive way to compute this per log is "the last user_end event and the last
# tts_first_audio event, subtracted" -- but that's an overwrite-as-you-go approach
# that silently produces nonsense if a log ever contains events from more than one
# run (e.g. append-mode logging picking up a leftover pre-run smoke-test event).
# metrics.py's dead_air computation solves this by *pairing*: walk events in
# chronological order, and whenever a tts_first_audio is seen while a user_end is
# still "pending" (unpaired), that closes one (user_end, tts_first_audio) pair. This
# function does the same pairing, across the whole log (these are single-turn
# conversations, so a file-wide sort reproduces what per-turn pairing in metrics.py
# would do), and -- per this analysis's spec -- takes the LAST complete pair as the
# log's TTFA. Taking the last pair (not the first, not a sum) is what makes this
# immune to stale earlier events: even if a smoke-test's user_end/tts_first_audio
# pair sits earlier in the sorted stream, the real run's pair still closes last and
# wins.
def extract_ttfa(log_path):
    """
    Returns the log's TTFA in ms (float), or None if no complete
    (user_end -> tts_first_audio) pair exists in the file at all.
    """
    events = []
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    # Sort by t_ms first -- required both for correct pairing order and because
    # append-mode logging does not guarantee events arrive in timestamp order.
    events.sort(key=lambda event: event["t_ms"])

    pending_user_end_t = None
    last_pair_ttfa = None

    for event in events:
        event_type = event["event_type"]

        if event_type == "user_end":
            pending_user_end_t = event["t_ms"]

        elif event_type == "tts_first_audio" and pending_user_end_t is not None:
            last_pair_ttfa = event["t_ms"] - pending_user_end_t
            pending_user_end_t = None  # pair closed; wait for the next user_end

    return last_pair_ttfa


# ======================================================================
# Step 2: load all logs, keyed by (policy, seed, scenario_id)
# ======================================================================
def load_all_ttfa(log_dir):
    """
    Returns:
      raw[policy][scenario_id] = [ttfa_seed_a, ttfa_seed_b, ...]   (one per seed run)
    Only includes files whose conversation_id parses to one of the three known
    E2 policies with a seed (skips warmup_*.jsonl and anything else non-conforming).
    """
    raw = defaultdict(lambda: defaultdict(list))
    skipped_unparseable = []
    skipped_no_ttfa = []

    for log_path in sorted(Path(log_dir).glob("*.jsonl")):
        conversation_id = log_path.stem
        parsed = parse_conversation_id(conversation_id)

        if parsed is None or parsed["policy"] not in POLICIES or parsed["seed"] is None:
            skipped_unparseable.append(conversation_id)
            continue

        ttfa = extract_ttfa(log_path)

        if ttfa is None:
            skipped_no_ttfa.append(conversation_id)
            continue

        raw[parsed["policy"]][parsed["scenario_id"]].append(ttfa)

    return raw, skipped_unparseable, skipped_no_ttfa


# ======================================================================
# Step 3: collapse seeds -> one median TTFA per (policy, scenario), then
# take the 3-way scenario intersection
# ======================================================================
def collapse_to_scenario_medians(raw):
    """
    raw[policy][scenario_id] = [ttfa, ...] (per seed)
    Returns median[policy][scenario_id] = median across that scenario's seeds,
    and the sorted list of scenario_ids present under ALL THREE policies.
    """
    median = defaultdict(dict)

    for policy in POLICIES:
        for scenario_id, ttfa_list in raw[policy].items():
            median[policy][scenario_id] = float(np.median(ttfa_list))

    scenario_sets = [set(median[policy].keys()) for policy in POLICIES]
    intersection = sorted(set.intersection(*scenario_sets))

    return median, intersection


# ======================================================================
# Step 4: paired bootstrap over scenarios (the significance test)
# ======================================================================
def paired_bootstrap(diffs, rng, iterations):
    """
    diffs: 1D array of per-scenario paired differences (early_policy - baseline).

    Resamples the N scenarios WITH replacement `iterations` times; each iteration's
    statistic is the mean of that resampled set of paired differences. The observed
    statistic is the mean of the *actual* (unresampled) diffs. Returns
    (observed_mean, ci_low, ci_high, bootstrap_means).
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    observed_mean = float(np.mean(diffs))

    # rng.integers draws scenario INDICES with replacement -- this is what makes the
    # resampling unit "a scenario" rather than "a run": every resampled index pulls
    # along that scenario's single already-collapsed (median-over-seeds) diff value.
    resample_indices = rng.integers(0, n, size=(iterations, n))
    bootstrap_means = diffs[resample_indices].mean(axis=1)

    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    return observed_mean, float(ci_low), float(ci_high), bootstrap_means


# ======================================================================
# Step 5: pre-registered p95 per policy, with a scenario-CLUSTER bootstrap CI
# ======================================================================
def p95_with_cluster_bootstrap(raw_policy, scenario_ids, rng, iterations):
    """
    raw_policy: {scenario_id: [ttfa, ...]} for one policy (all seeds, un-collapsed).
    scenario_ids: the intersection scenario list to restrict to.

    p95 is a per-RUN statistic (computed over all runs pooled together), but the
    resampling unit must still be the scenario, per this analysis's spec -- a plain
    resample-the-750-runs bootstrap would again overcount each scenario's 5 seeds as
    independent evidence. This uses a cluster/block bootstrap: each iteration
    resamples scenario_ids WITH replacement, and for every scenario drawn, pulls in
    ALL of that scenario's raw per-seed TTFA values (the whole cluster travels
    together). p95 is then computed on the pooled bag of values from that iteration's
    resampled scenarios.
    """
    scenario_ids = list(scenario_ids)
    n = len(scenario_ids)

    # Observed p95: all real runs across the (intersection) scenarios, pooled once.
    all_values = np.concatenate([np.asarray(raw_policy[sid]) for sid in scenario_ids])
    observed_p95 = float(np.percentile(all_values, 95))

    # Precompute each scenario's array once so the loop below just does lookups.
    per_scenario_arrays = [np.asarray(raw_policy[sid]) for sid in scenario_ids]

    bootstrap_p95s = np.empty(iterations)
    resample_indices = rng.integers(0, n, size=(iterations, n))

    for i in range(iterations):
        pooled = np.concatenate([per_scenario_arrays[j] for j in resample_indices[i]])
        bootstrap_p95s[i] = np.percentile(pooled, 95)

    ci_low, ci_high = np.percentile(bootstrap_p95s, [2.5, 97.5])

    return observed_p95, float(ci_low), float(ci_high)


# ======================================================================
# Report
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance test for E2 TTFA across timing policies."
    )
    parser.add_argument("log_dir", help="Directory of conv_<policy>_s<seed>_<scenario>.jsonl logs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--iterations", type=int, default=10000, help="Bootstrap iterations (default: 10000)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("E2 PAIRED BOOTSTRAP SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    print(f"log_dir     : {args.log_dir}")
    print(f"rng seed    : {args.seed}")
    print(f"iterations  : {args.iterations}")

    # --- Step 1+2: load & extract -------------------------------------------------
    raw, skipped_unparseable, skipped_no_ttfa = load_all_ttfa(args.log_dir)

    total_runs = sum(len(v) for policy in raw.values() for v in policy.values())
    print(f"\nLoaded {total_runs} runs with a complete TTFA pair across {len(POLICIES)} policies.")
    if skipped_unparseable:
        print(f"Skipped (not a conv_<policy>_s<seed>_<scenario> log, e.g. warmup files): "
              f"{len(skipped_unparseable)}")
    if skipped_no_ttfa:
        print(f"Skipped (no complete user_end -> tts_first_audio pair found): "
              f"{len(skipped_no_ttfa)} -> {skipped_no_ttfa}")

    for policy in POLICIES:
        n_scenarios = len(raw[policy])
        n_runs = sum(len(v) for v in raw[policy].values())
        print(f"  {policy:20s}: {n_scenarios} scenarios, {n_runs} runs")

    # --- Step 3: collapse to scenario medians + intersection ----------------------
    median, intersection = collapse_to_scenario_medians(raw)

    print(f"\nScenario intersection across all 3 policies: {len(intersection)} scenarios")
    for policy in POLICIES:
        missing = set(median[policy].keys()) - set(intersection)
        if missing:
            print(f"  NOTE: {policy} has scenarios outside the intersection, excluded: {sorted(missing)}")
        # Sanity check: flag any (policy, scenario) collapsed from fewer than 5 seeds.
        thin = [sid for sid in intersection if len(raw[policy][sid]) < 5]
        if thin:
            print(f"  WARNING: {policy} has scenarios collapsed from <5 seeds: {thin}")

    if len(intersection) == 0:
        print("\nNo scenarios present under all three policies -- cannot compute paired diffs.")
        sys.exit(1)

    # --- Step 4: paired bootstrap, per early policy vs baseline --------------------
    print("\n" + "=" * 70)
    print("PAIRED BOOTSTRAP: per-scenario median-over-seeds TTFA difference")
    print("(early policy - sentence_end); negative = early policy is FASTER")
    print("=" * 70)

    for early_policy in EARLY_POLICIES:
        diffs = np.array([
            median[early_policy][sid] - median[BASELINE][sid]
            for sid in intersection
        ])

        observed_mean, ci_low, ci_high, _ = paired_bootstrap(diffs, rng, args.iterations)

        significant = not (ci_low <= 0.0 <= ci_high)
        direction = "FASTER" if observed_mean < 0 else "SLOWER"

        print(f"\n{early_policy} vs {BASELINE}  (n={len(diffs)} scenarios)")
        print(f"  observed mean paired diff : {observed_mean:+.2f} ms  ({direction} than {BASELINE})")
        print(f"  95% bootstrap CI           : ({ci_low:+.2f}, {ci_high:+.2f}) ms")
        print(f"  significant (CI excludes 0): {significant}")

    # --- Step 5: pre-registered p95, with cluster-bootstrap CI ---------------------
    print("\n" + "=" * 70)
    print("PRE-REGISTERED P95 TTFA (over all runs, per policy)")
    print("=" * 70)
    print(
        "NOTE: p95 is a tail statistic. With only 50 independent scenarios (each\n"
        "contributing up to 5 correlated seed-runs), its bootstrap CI is inherently\n"
        "noisier than the paired-mean test above. Treat the paired difference test\n"
        "as the reliable significance signal for E2's claim; p95 here is reported\n"
        "for completeness/pre-registration, not as the primary evidence."
    )

    for policy in POLICIES:
        observed_p95, ci_low, ci_high = p95_with_cluster_bootstrap(
            raw[policy], intersection, rng, args.iterations
        )
        print(f"\n{policy}")
        print(f"  p95 TTFA                  : {observed_p95:.2f} ms")
        print(f"  95% bootstrap CI (cluster) : ({ci_low:.2f}, {ci_high:.2f}) ms")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
