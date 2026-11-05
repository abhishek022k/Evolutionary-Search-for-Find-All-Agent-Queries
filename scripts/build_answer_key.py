"""Build the answer key that judge() reads. Makes no API calls.

Run from the project root:  .venv/bin/python scripts/build_answer_key.py

Writes one line per filing: filing_id, relevant (yes or no), and needs_review.
needs_review marks yeses whose quote could not be found in the filing. They count
as yes until a person checks them.
"""
import json

QUESTION = "stay_home_boom"
FILINGS = "data/processed/filings.jsonl"
LABELS = f"data/processed/labels_full_{QUESTION}_claude-haiku-5-5.jsonl"
OUT = f"data/shared/answer_key_{QUESTION}.jsonl"

filing_ids = [json.loads(line)["filing_id"] for line in open(FILINGS)]
if len(filing_ids) != len(set(filing_ids)):
    raise SystemExit("filings.jsonl has repeated filing IDs")

# A retried filing has an error row and a later labeled row, so keep the labeled one.
labels = {}
for line in open(LABELS):
    row = json.loads(line)
    if "labels" in row:
        labels[row["filing_id"]] = row["labels"][QUESTION]

missing = set(filing_ids) - labels.keys()
extra = labels.keys() - set(filing_ids)
if missing or extra:
    raise SystemExit(f"labels do not match filings: {len(missing)} missing, {len(extra)} extra")

with open(OUT, "w") as f:
    for filing_id in sorted(filing_ids):
        label = labels[filing_id]
        needs_review = bool(label["answer"]) and not label.get("quote_verified", False)
        f.write(json.dumps({"filing_id": filing_id, "relevant": bool(label["answer"]),
                            "needs_review": needs_review}) + "\n")

yes = sum(bool(l["answer"]) for l in labels.values())
review = sum(bool(l["answer"]) and not l.get("quote_verified", False) for l in labels.values())
print(f"wrote {len(filing_ids)} filings to {OUT}: {yes} yes, {review} need review")
