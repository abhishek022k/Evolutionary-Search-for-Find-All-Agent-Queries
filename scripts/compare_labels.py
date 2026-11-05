"""Compare two labels files on the filings both labeled successfully.

Prints counts only. Disagreements, with quotes, go to the output file, so the
person writing Generation 0 queries can avoid seeing them.

Usage: .venv/bin/python scripts/compare_labels.py <labels A> <labels B> <output file>
"""
import json
import sys


def load(path):
    rows = {}
    for line in open(path):
        row = json.loads(line)
        if "labels" in row:
            rows[row["filing_id"]] = row
    return rows


path_a, path_b, out_path = sys.argv[1:4]
a, b = load(path_a), load(path_b)
shared = sorted(set(a) & set(b))
print(f"labeled: A {len(a)}, B {len(b)}, both {len(shared)}")

with open(out_path, "w") as out:
    for qid in a[shared[0]]["labels"]:
        both_yes = only_a = only_b = both_no = 0
        out.write(f"\n=== {qid} ===\n")
        for fid in shared:
            la, lb = a[fid]["labels"][qid], b[fid]["labels"][qid]
            if la["answer"] and lb["answer"]:
                both_yes += 1
            elif not la["answer"] and not lb["answer"]:
                both_no += 1
            else:
                yes = la if la["answer"] else lb
                only_a += la["answer"]
                only_b += lb["answer"]
                who = "A" if la["answer"] else "B"
                # Same marker summarize_labels.py uses: an unverified quote was not
                # found in the filing, so it must not look confirmed to a human reader.
                mark = "" if yes.get("quote_verified") else " [QUOTE NOT FOUND]"
                out.write(f"{fid} yes from {who}{mark}: {yes['quote'][:300]}\n")
        n = len(shared)
        any_yes = both_yes + only_a + only_b
        overlap = both_yes / any_yes if any_yes else 0
        print(f"{qid}: both yes {both_yes}, only A {only_a}, only B {only_b}, both no "
              f"{both_no}; A yes rate {100 * (both_yes + only_a) / n:.1f}%, B yes rate "
              f"{100 * (both_yes + only_b) / n:.1f}%; of filings either called yes, "
              f"both agreed on {100 * overlap:.0f}%")
print(f"Disagreements saved to {out_path}")
