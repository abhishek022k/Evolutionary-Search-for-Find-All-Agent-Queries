"""Show labeling progress. Read-only, so it is safe during a run.

Once:         .venv/bin/python scripts/progress.py
Live view:    .venv/bin/python scripts/progress.py --watch
"""
import argparse
import datetime
import json
import os
import time

CONFIG = json.load(open("config.json"))
# From the config, so labelling a different question does not silently show the old run.
PATH = CONFIG["labels_path"]
QUESTION = CONFIG["question_id"]
TOTAL = CONFIG["n_filings"]
WIDTH = 40


def snapshot():
    labeled, failed, spent = {}, set(), 0.0
    if os.path.exists(PATH):
        for line in open(PATH):
            row = json.loads(line)
            spent += row.get("cost", 0)
            if "labels" in row:
                labeled[row["filing_id"]] = row["labels"][QUESTION]["answer"]
                failed.discard(row["filing_id"])
            else:
                failed.add(row["filing_id"])
    return len(labeled), sum(labeled.values()), len(failed - labeled.keys()), spent


def render():
    done, yes, failed, spent = snapshot()
    filled = int(WIDTH * done / TOTAL)
    bar = "#" * filled + "-" * (WIDTH - filled)
    rate = 100 * yes / done if done else 0
    print(f"[{bar}] {done}/{TOTAL} labeled ({100 * done / TOTAL:.1f}%)")
    print(f"Yes found: {yes} ({rate:.1f}% of labeled) | Waiting to retry: {failed} | "
          f"Cost so far: ${spent:.2f}")
    if os.path.exists(PATH):
        last = datetime.datetime.fromtimestamp(os.path.getmtime(PATH))
        print(f"Last saved: {last:%H:%M:%S}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        render()
        if not args.watch:
            break
        time.sleep(30)
        print()


if __name__ == "__main__":
    main()
