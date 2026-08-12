import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from harness.stats import percentile, bootstrap_confidence_interval


def main():

    if len(sys.argv) != 2:
        print("Usage: python metrics.py <log_file>")
        return

    log_file = sys.argv[1]

    turns = defaultdict(list)

    try:
        with open(log_file, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                event = json.loads(line)
                turns[event["turn_id"]].append(event)

    except FileNotFoundError:
        print(f"Error: {log_file} not found.")
        return

    ttfa_list = []
    dead_air_list = []

    print("=" * 50)
    print("VOICE RAG METRICS REPORT")
    print("=" * 50)

    for turn_id in sorted(turns):

        user_end = None
        first_audio = None

        # Sort events by timestamp. This also matters for dead air below:
        # spine.py always logs turn_id=1 and the log file is opened in
        # append mode, so a single turn_id bucket can hold events from
        # several separate runs. Sorting by t_ms at least puts each run's
        # own events back in chronological order before we try to pair
        # them up.
        events = sorted(
            turns[turn_id],
            key=lambda event: event["t_ms"]
        )

        for event in events:

            if event["event_type"] == "user_end":
                user_end = event["t_ms"]

            elif event["event_type"] == "tts_first_audio":
                first_audio = event["t_ms"]

        if user_end is not None and first_audio is not None:

            ttfa = first_audio - user_end
            ttfa_list.append(ttfa)

            print(f"Turn {turn_id}: TTFA = {ttfa} ms")

        # --- Dead air --------------------------------------------------
        # Dead air is the silence the caller actually *hears*: stretches
        # where no audio is playing. It is NOT the whole turn span — most
        # of a turn is system-working time (transcribing, retrieving,
        # generating) that happens while no one expects sound yet, and
        # summing every gap between every event (the old, buggy approach)
        # counted all of that as "silence," inflating the number.
        #
        # A silent gap only runs from an audio boundary to the next audio
        # boundary: from user_end (the user stopped talking) — or, once
        # multi-audio turns exist, from the end of the previous audio
        # output — until the next tts_first_audio (audio starts playing
        # again). So we walk the turn's events in order and pair each
        # user_end with the very next tts_first_audio that follows it,
        # summing one gap per pair, instead of summing every consecutive
        # event-to-event gap in the turn.
        #
        # Pairing (rather than blindly summing) also makes this robust to
        # a log file that has accumulated multiple runs under the same
        # turn_id: each run's user_end only ever pairs with its own run's
        # tts_first_audio, so old runs can't get summed together into a
        # huge nonsense number.
        #
        # Today's pipeline generates exactly one audio output per turn,
        # so there's only ever one such pair, and dead air ends up equal
        # to TTFA (both are tts_first_audio - user_end) — that's expected
        # and correct, not a bug. Once multi-turn/multi-audio conversations
        # exist, this loop will naturally sum multiple gaps, and dead air
        # will diverge from (and become more informative than) TTFA.
        dead_air = 0
        pending_user_end = None
        found_gap = False

        for event in events:

            if event["event_type"] == "user_end":
                pending_user_end = event["t_ms"]

            elif event["event_type"] == "tts_first_audio" and pending_user_end is not None:
                dead_air += event["t_ms"] - pending_user_end
                pending_user_end = None
                found_gap = True

        if found_gap:

            dead_air_list.append(dead_air)

            print(f"Turn {turn_id}: Dead Air = {dead_air} ms")

    print("\n" + "-" * 50)
    print("TTFA SUMMARY")
    print("-" * 50)

    if ttfa_list:

        average_ttfa = sum(ttfa_list) / len(ttfa_list)
        p95_ttfa = percentile(ttfa_list, 95)
        ci_ttfa = bootstrap_confidence_interval(ttfa_list)

        print(f"Average TTFA : {average_ttfa:.2f} ms")
        print(f"P95 TTFA     : {p95_ttfa:.2f} ms")

        if ci_ttfa:
            print(
                f"95% CI       : "
                f"({ci_ttfa[0]:.2f}, {ci_ttfa[1]:.2f}) ms"
            )

    else:
        print("No TTFA values found.")

    print("\n" + "-" * 50)
    print("DEAD AIR SUMMARY")
    print("-" * 50)

    if dead_air_list:

        average_dead_air = sum(dead_air_list) / len(dead_air_list)
        p95_dead_air = percentile(dead_air_list, 95)
        ci_dead_air = bootstrap_confidence_interval(dead_air_list)

        print(f"Average Dead Air : {average_dead_air:.2f} ms")
        print(f"P95 Dead Air     : {p95_dead_air:.2f} ms")

        if ci_dead_air:
            print(
                f"95% CI           : "
                f"({ci_dead_air[0]:.2f}, {ci_dead_air[1]:.2f}) ms"
            )

    else:
        print("No Dead Air values found.")


if __name__ == "__main__":
    main()


# ============================================================
# ADDITIONAL METRICS
# IMPORTANT:
# Everything above this section is the existing code.
# ============================================================

def additional_metrics(log_file):

    turns = defaultdict(list)

    try:
        with open(log_file, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                event = json.loads(line)
                turns[event["turn_id"]].append(event)

    except FileNotFoundError:
        print(f"Error: {log_file} not found.")
        return

    retrieval_latency_list = []
    llm_ttft_list = []
    llm_generation_list = []
    total_response_list = []

    retrieval_count = 0
    total_turns = 0

    for turn_id in sorted(turns):

        events = sorted(
            turns[turn_id],
            key=lambda event: event["t_ms"]
        )

        user_end = None
        retrieval_start = None
        retrieval_end = None
        llm_first_token = None
        llm_final = None
        tts_first_audio = None

        for event in events:

            event_type = event["event_type"]
            timestamp = event["t_ms"]

            if event_type == "user_end":
                user_end = timestamp
                total_turns += 1

            elif event_type == "retrieval_launched":
                retrieval_start = timestamp
                retrieval_count += 1

            elif event_type == "retrieval_result":
                retrieval_end = timestamp

            elif event_type == "llm_first_token":
                llm_first_token = timestamp

            elif event_type == "llm_final":
                llm_final = timestamp

            elif event_type == "tts_first_audio":
                tts_first_audio = timestamp

        # --------------------------------------------------------
        # Retrieval latency
        # --------------------------------------------------------

        if retrieval_start is not None and retrieval_end is not None:

            retrieval_latency = retrieval_end - retrieval_start

            retrieval_latency_list.append(
                retrieval_latency
            )

        # --------------------------------------------------------
        # LLM Time To First Token
        # --------------------------------------------------------

        if llm_first_token is not None and user_end is not None:

            llm_ttft = llm_first_token - user_end

            llm_ttft_list.append(
                llm_ttft
            )

        # --------------------------------------------------------
        # LLM generation time
        # --------------------------------------------------------

        if llm_first_token is not None and llm_final is not None:

            llm_generation = llm_final - llm_first_token

            llm_generation_list.append(
                llm_generation
            )

        # --------------------------------------------------------
        # Total response latency
        # --------------------------------------------------------

        if user_end is not None and tts_first_audio is not None:

            total_response = tts_first_audio - user_end

            total_response_list.append(
                total_response
            )

    # ============================================================
    # ADDITIONAL METRICS REPORT
    # ============================================================

    print("\n" + "=" * 50)
    print("ADDITIONAL VOICE RAG METRICS")
    print("=" * 50)

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------

    print("\n" + "-" * 50)
    print("RETRIEVAL METRICS")
    print("-" * 50)

    if retrieval_latency_list:

        average_retrieval = (
            sum(retrieval_latency_list)
            / len(retrieval_latency_list)
        )

        p95_retrieval = percentile(
            retrieval_latency_list,
            95
        )

        print(
            f"Retrieval Count : "
            f"{len(retrieval_latency_list)}"
        )

        print(
            f"Average Retrieval Latency : "
            f"{average_retrieval:.2f} ms"
        )

        print(
            f"P95 Retrieval Latency     : "
            f"{p95_retrieval:.2f} ms"
        )

    else:

        print("No retrieval latency values found.")

    # ------------------------------------------------------------
    # LLM Time To First Token
    # ------------------------------------------------------------

    print("\n" + "-" * 50)
    print("LLM TTFT SUMMARY")
    print("-" * 50)

    if llm_ttft_list:

        average_llm_ttft = (
            sum(llm_ttft_list)
            / len(llm_ttft_list)
        )

        p95_llm_ttft = percentile(
            llm_ttft_list,
            95
        )

        print(
            f"Average LLM TTFT : "
            f"{average_llm_ttft:.2f} ms"
        )

        print(
            f"P95 LLM TTFT     : "
            f"{p95_llm_ttft:.2f} ms"
        )

    else:

        print("No LLM TTFT values found.")

    # ------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------

    print("\n" + "-" * 50)
    print("LLM GENERATION SUMMARY")
    print("-" * 50)

    if llm_generation_list:

        average_llm_generation = (
            sum(llm_generation_list)
            / len(llm_generation_list)
        )

        p95_llm_generation = percentile(
            llm_generation_list,
            95
        )

        print(
            f"Average LLM Generation : "
            f"{average_llm_generation:.2f} ms"
        )

        print(
            f"P95 LLM Generation     : "
            f"{p95_llm_generation:.2f} ms"
        )

    else:

        print("No LLM generation values found.")

    # ------------------------------------------------------------
    # Total response latency
    # ------------------------------------------------------------

    print("\n" + "-" * 50)
    print("TOTAL RESPONSE LATENCY")
    print("-" * 50)

    if total_response_list:

        average_total_response = (
            sum(total_response_list)
            / len(total_response_list)
        )

        p95_total_response = percentile(
            total_response_list,
            95
        )

        print(
            f"Average Response Latency : "
            f"{average_total_response:.2f} ms"
        )

        print(
            f"P95 Response Latency     : "
            f"{p95_total_response:.2f} ms"
        )

    else:

        print("No total response latency values found.")

    # ------------------------------------------------------------
    # Retrieval rate
    # ------------------------------------------------------------

    print("\n" + "-" * 50)
    print("RETRIEVAL RATE")
    print("-" * 50)

    if total_turns > 0:

        retrieval_rate = (
            retrieval_count / total_turns
        ) * 100

        print(
            f"Total Turns      : {total_turns}"
        )

        print(
            f"Retrieval Calls  : {retrieval_count}"
        )

        print(
            f"Retrieval Rate   : {retrieval_rate:.2f}%"
        )

    else:

        print("No turns found.")


# ============================================================
# RUN ADDITIONAL METRICS
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) == 2:
        additional_metrics(sys.argv[1])


# ============================================================
# HIT@K METRICS (Experiment 2)
# IMPORTANT:
# Everything above this section is the existing code.
# ============================================================
#
# Retrieval hit@k: did the scenario's gold source document actually show
# up in the top-k retrieved passages? Experiment 2 compares timing
# policies (sentence_end/immediate/predicted_pause) on the claim that
# retrieving earlier doesn't cost accuracy -- but pipeline/spine.py's
# retrieval_result event only logged "passages" (raw text) and "scores"
# (FAISS distances), neither of which says WHICH document a passage came
# from, so hit@k wasn't computable from the log at all. spine.py's
# retrieve() now also logs "filenames" (retrieval/search.py already
# returns this per hit) -- this section reads that back out and joins it
# against each scenario's gold `source` field from simulator/scenarios.json
# to compute hit@3.

# simulator/batch_runner.py's run_sweep() names each conversation
# f"conv_{policy}_s{seed}_{scenario['id']:03d}" (its --single mode instead
# uses plain f"conv_{scenario['id']:03d}", no policy/seed). This regex
# parses both: policy/seed are None when the "_s<seed>_" segment isn't
# there at all.
_CONV_ID_RE = re.compile(r"^conv_(?:(?P<policy>.+)_s(?P<seed>\d+)_)?(?P<scenario_id>\d+)$")

DEFAULT_SCENARIOS_FILE = Path(__file__).resolve().parent.parent / "simulator" / "scenarios.json"


def parse_conversation_id(conversation_id):
    """
    Parses a batch-runner conversation_id into {policy, seed, scenario_id},
    or None if it doesn't match the expected conv_... naming at all (e.g.
    Week 1's hand-run conv_001 IS matched -- scenario_id=1, policy=None,
    seed=None -- it's only conversation_ids with no numeric id suffix at
    all that return None).
    """
    match = _CONV_ID_RE.match(conversation_id)
    if not match:
        return None

    seed = match.group("seed")

    return {
        "policy": match.group("policy"),
        "seed": int(seed) if seed is not None else None,
        "scenario_id": int(match.group("scenario_id")),
    }


def load_gold_sources(scenarios_file=DEFAULT_SCENARIOS_FILE):
    """
    Returns {scenario_id: source_document} from simulator/scenarios.json --
    the gold document hit@k checks retrieval against, e.g.
    {1: "01_checkin_policy.txt", ...}.
    """
    with open(scenarios_file, "r", encoding="utf-8") as file:
        scenarios = json.load(file)

    return {scenario["id"]: scenario["source"] for scenario in scenarios}


def hit_at_k_per_turn(log_path, gold_source, k=3):
    """
    Reads one conversation log and returns {turn_id: hit_bool} for every
    turn that actually ran a retrieval. A turn with no retrieval_result
    event at all (the controller decided to skip retrieval) is simply
    absent from the result -- hit@k isn't defined when nothing was
    retrieved.

    A turn can have more than one retrieval_result event -- predicted_pause
    can launch an early retrieval, discard it when the forecast it was
    based on turns out wrong (see controller/timing.py's
    wait_for_growth()/forecast_invalidated), and retrieve again on the
    real transcript. Only one retrieval's passages ever actually reach
    generate_answer() per turn (pipeline/spine.py's run_conversation()
    awaits exactly one retrieval path per turn), and that's always the
    one launched last: a discarded early attempt's retrieval_result, if it
    even finishes before being cancelled, is logged before the fallback
    retrieval it's replaced by, never after. So among a turn's
    retrieval_result events, the one with the latest t_ms is the one hit@k
    should score.
    """
    events_by_turn = defaultdict(list)

    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            event = json.loads(line)

            if event["event_type"] == "retrieval_result":
                events_by_turn[event["turn_id"]].append(event)

    hits = {}

    for turn_id, events in events_by_turn.items():
        events.sort(key=lambda event: event["t_ms"])
        used = events[-1]
        filenames = used["payload"].get("filenames") or []
        hits[turn_id] = gold_source in filenames[:k]

    return hits


def hit_at_k_report(log_dir, scenarios_file=DEFAULT_SCENARIOS_FILE, k=3):
    """
    Walks every *.jsonl log in log_dir, joins each conversation_id to its
    scenario's gold source document (via parse_conversation_id() +
    load_gold_sources()), and reports the hit@k rate -- overall and broken
    out per timing policy, so Experiment 2's "equal accuracy across
    policies" claim is something this prints instead of something asserted.

    Logs that don't match the conv_... naming, or whose scenario_id has no
    entry in scenarios_file, or that never ran a retrieval at all, are
    counted and reported separately rather than silently dropped.
    """
    gold_sources = load_gold_sources(scenarios_file)

    per_policy_hits = defaultdict(list)  # policy -> [bool, ...]
    skipped_unparseable = []
    skipped_no_gold = []
    skipped_no_retrieval = []

    for log_path in sorted(Path(log_dir).glob("*.jsonl")):
        conversation_id = log_path.stem
        parsed = parse_conversation_id(conversation_id)

        if parsed is None:
            skipped_unparseable.append(conversation_id)
            continue

        gold_source = gold_sources.get(parsed["scenario_id"])

        if gold_source is None:
            skipped_no_gold.append(conversation_id)
            continue

        turns = hit_at_k_per_turn(log_path, gold_source, k=k)

        if not turns:
            skipped_no_retrieval.append(conversation_id)
            continue

        policy_key = parsed["policy"] or "(no policy in conversation_id)"
        per_policy_hits[policy_key].extend(turns.values())

    print("\n" + "=" * 50)
    print(f"HIT@{k} REPORT")
    print("=" * 50)

    if not per_policy_hits:
        print("No scored conversations found.")
    else:
        all_hits = []

        for policy in sorted(per_policy_hits):
            hits = per_policy_hits[policy]
            rate = 100 * sum(hits) / len(hits)
            all_hits.extend(hits)
            print(f"{policy:24s} hit@{k} = {rate:6.2f}%  ({sum(hits)}/{len(hits)})")

        overall_rate = 100 * sum(all_hits) / len(all_hits)
        print("-" * 50)
        print(f"{'overall':24s} hit@{k} = {overall_rate:6.2f}%  ({sum(all_hits)}/{len(all_hits)})")

    if skipped_unparseable:
        print(f"\nSkipped (conversation_id didn't match conv_... naming): {len(skipped_unparseable)}")
    if skipped_no_gold:
        print(f"Skipped (scenario_id has no entry in {scenarios_file}): {len(skipped_no_gold)}")
    if skipped_no_retrieval:
        print(f"Skipped (no retrieval happened this conversation): {len(skipped_no_retrieval)}")

    return per_policy_hits


if __name__ == "__main__":

    if len(sys.argv) >= 3 and sys.argv[1] == "--hit3":
        log_dir = sys.argv[2]
        scenarios_file = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_SCENARIOS_FILE
        hit_at_k_report(log_dir, scenarios_file=scenarios_file, k=3)