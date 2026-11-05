"""Summarize a labels file: rarity per question, cost, and yes quotes to review.

Usage: .venv/bin/python scripts/summarize_labels.py data/processed/<labels file>
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from label import FILINGS_PATH, normalize  # noqa: E402

N_FULL = 3000
ITEM_7_SHARE_TO_SWITCH = 0.80  # agreed rule: at or above this, use Item 7 only
MIN_YES_TO_DECIDE = 5


def quote_section(quote, filing):
    needle = normalize(quote)
    in_1a = needle in normalize(filing["item_1a"])
    in_7 = needle in normalize(filing["item_7"])
    if in_1a and in_7:
        return "both"
    return "item_7" if in_7 else "item_1a" if in_1a else "not_found"


latest = {}
for line in open(sys.argv[1]):
    row = json.loads(line)
    previous = latest.get(row["filing_id"])
    if previous is None or "labels" in row or "labels" not in previous:
        latest[row["filing_id"]] = row
rows = list(latest.values())
good = [r for r in rows if "labels" in r]
total_cost = sum(json.loads(line).get("cost", 0) for line in open(sys.argv[1]))

print(f"filings: {len(rows)}, labeled: {len(good)}, failed: {len(rows) - len(good)}")
errors = {}
for r in rows:
    if "labels" not in r:
        errors[r.get("error", "unknown")] = errors.get(r.get("error", "unknown"), 0) + 1
if errors:
    print("failure types:", errors)
if good:
    avg_in = sum(r["usage"]["input_tokens"] for r in good) / len(good)
    avg_out = sum(r["usage"]["output_tokens"] for r in good) / len(good)
    print(f"average tokens per filing: {avg_in:,.0f} in, {avg_out:,.0f} out")
    print(f"spent: ${total_cost:.2f}; a full {N_FULL}-filing run at this rate: "
          f"${total_cost / len(good) * N_FULL:.2f}")

needed = {r["filing_id"] for r in good}
filings = {}
for line in open(FILINGS_PATH):
    f = json.loads(line)
    if f["filing_id"] in needed:
        filings[f["filing_id"]] = f

questions = good[0]["labels"].keys() if good else []
for qid in questions:
    yes = [r for r in good if r["labels"][qid]["answer"]]
    n, k = len(good), len(yes)
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    bad = sum(1 for r in yes if not r["labels"][qid]["quote_verified"])
    print(f"\n{qid}: {k}/{n} yes ({100 * p:.1f}%), 95% range "
          f"{100 * max(0, centre - half):.1f}-{100 * (centre + half):.1f}%, "
          f"quotes not found in filing: {bad}")

    sections = {"item_7": 0, "item_1a": 0, "both": 0, "not_found": 0}
    for r in yes:
        sections[quote_section(r["labels"][qid]["quote"], filings[r["filing_id"]])] += 1
    usable = k - sections["not_found"]
    # A quote present in both sections would survive an Item 7 only collection.
    share = (sections["item_7"] + sections["both"]) / usable if usable else 0
    if usable < MIN_YES_TO_DECIDE:
        verdict = f"keep both sections (fewer than {MIN_YES_TO_DECIDE} usable yes quotes)"
    elif share >= ITEM_7_SHARE_TO_SWITCH:
        verdict = "switch to Item 7 only"
    else:
        verdict = "keep both sections"
    print(f"  quote sections: Item 7 {sections['item_7']}, Item 1A {sections['item_1a']}, "
          f"both {sections['both']}, not found {sections['not_found']}; "
          f"available in Item 7: {100 * share:.0f}%; if this question wins: {verdict}")

    for r in yes:
        mark = "" if r["labels"][qid]["quote_verified"] else " [QUOTE NOT FOUND]"
        print(f"  {r['filing_id']}{mark}: {r['labels'][qid]['quote'][:220]}")
