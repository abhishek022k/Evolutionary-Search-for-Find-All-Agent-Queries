"""Step 4: select the fixed collection and the pilot subset.

Keeps 2020 filings whose Item 1A has at least MIN_1A_WORDS words (shorter ones
are mostly placeholders such as "Omitted"), then draws a seeded random sample.
Filings are sorted by CIK first so the result does not depend on file order.

Run from the project root:  .venv/bin/python scripts/select_filings.py
"""
import glob
import json
import random
import statistics

CONFIG = json.load(open("config.json"))
SEED = CONFIG["seed"]
N_FILINGS = CONFIG["n_filings"]
N_PILOT = CONFIG["n_pilot"]
MIN_1A_WORDS = CONFIG["min_1a_words"]

rows = []
total = 0
for path in sorted(glob.glob("data/raw/2020_*.jsonl")):
    with open(path) as f:
        for line in f:
            total += 1
            r = json.loads(line)
            item_1a = r.get("section_1A") or ""
            if len(item_1a.split()) < MIN_1A_WORDS:
                continue
            rows.append({
                "filing_id": f"{r['cik']}_2020",
                "cik": r["cik"],
                "item_1a": item_1a,
                "item_7": r.get("section_7") or "",
            })

rows.sort(key=lambda r: int(r["cik"]))
assert len({r["filing_id"] for r in rows}) == len(rows), "duplicate filing ids"

rng = random.Random(SEED)
sample = rng.sample(rows, N_FILINGS)
sample.sort(key=lambda r: int(r["cik"]))
pilot_ids = sorted(
    random.Random(SEED + 1).sample([r["filing_id"] for r in sample], N_PILOT))

with open("data/processed/filings.jsonl", "w") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")
with open("data/processed/pilot_ids.txt", "w") as f:
    f.write("\n".join(pilot_ids) + "\n")

words = [len(r["item_1a"].split()) + len(r["item_7"].split()) for r in sample]
print(f"raw filings: {total}; passed filter: {len(rows)}; sampled: {len(sample)}; "
      f"pilot: {len(pilot_ids)}")
print(f"words per filing (1A + 7): mean {sum(words) // len(words)}, "
      f"median {int(statistics.median(words))}, max {max(words)}")
print(f"total words: {sum(words) / 1e6:.1f} million")
print(f"filings over 70,000 words (could pass 100K tokens): "
      f"{sum(1 for w in words if w > 70000)}")
print(f"filings with Item 7 under 250 words: "
      f"{sum(1 for r in sample if len(r['item_7'].split()) < 250)}")
