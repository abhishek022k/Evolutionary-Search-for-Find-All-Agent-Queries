"""Read-only, free check: which 10-K sections carry 'COVID raised demand'
language, and how large are they? Word counts only; no labels. Writes nothing."""
import glob
import json
import re
import statistics

COVID = re.compile(r"covid|coronavirus|pandemic", re.I)
UP = (r"(increas\w*|higher|heighten\w*|surg\w*|strong\w*|elevated|greater|"
      r"accelerat\w*|record|unprecedented|growth in|rise in|spike\w*)")
DEMAND = re.compile(
    rf"\b{UP}\s+(?:\w+\s+){{0,3}}?(demand|orders|sales of|adoption|usage|utilization)\b"
    rf"|\b(demand|orders|adoption|usage)\b\s+(?:\w+\s+){{0,3}}?{UP}\b"
    rf"|\bbenefit(?:ed|ted)\s+from\b",
    re.I)
SECTIONS = ["section_1A", "section_7", "section_1"]


def hit(text, size=500, step=250):
    if not COVID.search(text):
        return False
    for i in range(0, max(1, len(text) - size + 1), step):
        w = text[i:i + size]
        if COVID.search(w) and DEMAND.search(w):
            return True
    return False


n = 0
hits = {s: 0 for s in SECTIONS}
only = {s: 0 for s in SECTIONS}
any_hit = 0
words = {s: [] for s in SECTIONS}
missing = {s: 0 for s in SECTIONS}

for path in sorted(glob.glob("data/raw/2020_*.jsonl")):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if len((r.get("section_1A") or "").split()) < 250:
                continue
            n += 1
            found = []
            for s in SECTIONS:
                t = r.get(s) or ""
                c = len(t.split())
                words[s].append(c)
                missing[s] += c < 250
                if t and hit(t):
                    hits[s] += 1
                    found.append(s)
            any_hit += bool(found)
            if len(found) == 1:
                only[found[0]] += 1

print("filings:", n)
for s in SECTIONS:
    w = words[s]
    print(f"{s}: rough hits {hits[s]} ({100 * hits[s] / n:.1f}%), only here {only[s]}, "
          f"under 250 words {missing[s]}, mean words {sum(w) // n}, "
          f"median {int(statistics.median(w))}")
print(f"rough hits in any of the three: {any_hit} ({100 * any_hit / n:.1f}%)")
