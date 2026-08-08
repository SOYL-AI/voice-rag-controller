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

        # Sort events by timestamp
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


        dead_air = 0

        for i in range(1, len(events)):

            previous_time = events[i - 1]["t_ms"]
            current_time = events[i]["t_ms"]

            gap = current_time - previous_time

            if gap > 0:
                dead_air += gap

        if len(events) >= 2:

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