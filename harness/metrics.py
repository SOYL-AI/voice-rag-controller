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

    print("=" * 45)
    print("VOICE RAG TTFA REPORT")
    print("=" * 45)

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

            print(f"Turn {turn_id}: TTFA = {ttfa} ms")

    if not ttfa_list:
        print("No TTFA values found.")
        return

    average = sum(ttfa_list) / len(ttfa_list)
    p95 = percentile(ttfa_list, 95)
    ci = bootstrap_confidence_interval(ttfa_list)

    print("\n" + "-" * 45)
    print(f"Average TTFA : {average:.2f} ms")
    print(f"P95 TTFA     : {p95:.2f} ms")

    if ci:
        print(f"95% CI       : ({ci[0]:.2f}, {ci[1]:.2f}) ms")


if __name__ == "__main__":
    main()
