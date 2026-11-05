"""Read-only check: what are the very short Item 1A sections, and what period
do these filings cover? Writes nothing."""
import collections
import glob
import json
import random
import re

rows = []
for path in sorted(glob.glob("data/raw/2020_*.jsonl")):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "cik": r["cik"],
                "a": r.get("section_1A") or "",
                "s7": (r.get("section_7") or ""),
                "s9a": (r.get("section_9A") or ""),
            })

for r in rows:
    r["n"] = len(r["a"].split())

bands = [(1, 49), (50, 249), (250, 999), (1000, 10**9)]
for lo, hi in bands:
    print(f"Item 1A {lo}-{hi} words: {sum(1 for r in rows if lo <= r['n'] <= hi)}")

rng = random.Random(0)
stubs = [r for r in rows if 1 <= r["n"] < 50]
phrases = collections.Counter(" ".join(r["a"].lower().split())[:70] for r in stubs)
print("\ncommonest stub openings:")
for p, c in phrases.most_common(8):
    print(f"  {c:5d}  {p}")

print("\nexamples, 50-999 words:")
mid = [r for r in rows if 50 <= r["n"] < 1000]
for r in rng.sample(mid, min(5, len(mid))):
    print(f"  [{r['n']} words] {' '.join(r['a'].split())[:170]}")

print("\nsize of filings with 1000+ words:")
big = sorted(r["n"] for r in rows if r["n"] >= 1000)
print("  count", len(big), "median", big[len(big) // 2], "mean", sum(big) // len(big),
      "p90", big[9 * len(big) // 10])
print("  3,000 of them at the mean is about",
      round(3000 * sum(big) / len(big) / 1e6, 1), "million words")

# Period check: which fiscal year-end does each filing's internal-control
# statement (Item 9A) or MD&A (Item 7) report on?
fy = collections.Counter()
usable = [r for r in rows if r["n"] >= 1000]
for r in usable:
    text = r["s9a"] + " " + r["s7"][:5000]
    dates = re.findall(
        r"(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2},\s+(20\d{2})", text, re.I)
    if not dates:
        fy["no date found"] += 1
    else:
        fy[collections.Counter(dates).most_common(1)[0][0]] += 1
print("\nmost-mentioned year in Item 9A / start of Item 7:", fy.most_common(6))
print("usable filings mentioning COVID:",
      sum(1 for r in usable if re.search(r"covid|coronavirus", r["a"], re.I)), "of", len(usable))
