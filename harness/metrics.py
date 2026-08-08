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
