"""Block until the host has at least N GB of available RAM (or time out).

The measurement host is shared with other workloads that at times exhaust
RAM and the commit limit; starting a benchmark then fails while loading DLLs
or models. Exit code 0 = memory available, 1 = timed out.

    python -m energy.wait_mem --gb 3 --timeout-min 120
"""

import argparse
import sys
import time

import psutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=3.0)
    ap.add_argument("--timeout-min", type=float, default=120)
    a = ap.parse_args()
    end = time.time() + a.timeout_min * 60
    while time.time() < end:
        avail = psutil.virtual_memory().available / 1e9
        if avail >= a.gb:
            print(f"available {avail:.1f} GB", flush=True)
            return 0
        time.sleep(20)
    print("timed out waiting for memory", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
