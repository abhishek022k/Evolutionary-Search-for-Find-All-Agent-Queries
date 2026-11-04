"""Read-only look at the raw EDGAR-CORPUS 2020 files. Writes nothing."""
import collections
import glob
import json
import re
import statistics

FILES = sorted(glob.glob("data/raw/2020_*.jsonl"))

total = 0
keys = None
year_values = collections.Counter()
filename_suffix = collections.Counter()
ciks = collections.Counter()
words_1a = []
empty_1a = 0
first = None
covid_mentions = 0
order_sample = []

for path in FILES:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if first is None:
                first = r
                keys = list(r.keys())
            year_values[str(r.get("year"))] += 1
            m = re.search(r"_(\d{4})", str(r.get("filename", "")))
            filename_suffix[m.group(1) if m else "none"] += 1
            ciks[str(r.get("cik"))] += 1
            text = r.get("section_1A") or ""
            n = len(text.split())
            words_1a.append(n)
            if n == 0:
                empty_1a += 1
            if re.search(r"covid|coronavirus", text, re.I):
                covid_mentions += 1
            if total <= 8:
                order_sample.append(r.get("cik"))

words_1a.sort()
print("files:", FILES)
print("total filings:", total)
print("fields:", keys)
print("year field:", year_values.most_common(5))
print("year in filename:", filename_suffix.most_common(5))
print("unique companies:", len(ciks), "| companies with >1 filing:",
      sum(1 for c in ciks.values() if c > 1))
print("first 8 CIKs in file order:", order_sample)
print("Item 1A empty:", empty_1a)
for cutoff in (1, 50, 100, 250, 500, 1000):
    print(f"  Item 1A >= {cutoff} words: {sum(1 for w in words_1a if w >= cutoff)}")
nonempty = [w for w in words_1a if w > 0]
print("Item 1A words, non-empty: median", statistics.median(nonempty),
      "p10", nonempty[len(nonempty) // 10],
      "p90", nonempty[9 * len(nonempty) // 10], "max", nonempty[-1])
print("total Item 1A words (all filings):", sum(words_1a))
print("filings whose Item 1A mentions COVID/coronavirus:", covid_mentions)
print("example:", {k: (str(v)[:70]) for k, v in first.items() if k in
                   ("filename", "cik", "year")})
print("example 1A start:", (first.get("section_1A") or "")[:300].replace("\n", " "))
