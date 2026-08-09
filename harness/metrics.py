import json
import sys
from collections import defaultdict

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