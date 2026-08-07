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

        for event in turns[turn_id]:

            if event["event_type"] == "user_end":
                user_end = event["t_ms"]

            elif event["event_type"] == "tts_first_audio":
                first_audio = event["t_ms"]

        if user_end is not None and first_audio is not None:

            
            ttfa = first_audio - user_end
            ttfa_list.append(ttfa)

            
            dead_air = first_audio - user_end
            dead_air_list.append(dead_air)

            print(
                f"Turn {turn_id}: "
                f"TTFA = {ttfa} ms | "
                f"Dead Air = {dead_air} ms"
            )

    if not ttfa_list:
        print("No TTFA values found.")
        return


    avg_ttfa = sum(ttfa_list) / len(ttfa_list)
    p95_ttfa = percentile(ttfa_list, 95)
    ci_ttfa = bootstrap_confidence_interval(ttfa_list)

    print("\n" + "-" * 50)
    print("TTFA SUMMARY")
    print("-" * 50)
    print(f"Average TTFA : {avg_ttfa:.2f} ms")
    print(f"P95 TTFA     : {p95_ttfa:.2f} ms")

    if ci_ttfa:
        print(f"95% CI       : ({ci_ttfa[0]:.2f}, {ci_ttfa[1]:.2f}) ms")

    avg_dead_air = sum(dead_air_list) / len(dead_air_list)
    p95_dead_air = percentile(dead_air_list, 95)
    ci_dead_air = bootstrap_confidence_interval(dead_air_list)

    print("\n" + "-" * 50)
    print("DEAD AIR SUMMARY")
    print("-" * 50)
    print(f"Average Dead Air : {avg_dead_air:.2f} ms")
    print(f"P95 Dead Air     : {p95_dead_air:.2f} ms")

    if ci_dead_air:
        print(f"95% CI           : ({ci_dead_air[0]:.2f}, {ci_dead_air[1]:.2f}) ms")


if __name__ == "__main__":
    main()