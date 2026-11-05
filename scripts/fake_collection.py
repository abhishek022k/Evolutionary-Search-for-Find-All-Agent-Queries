"""Invented filings and labels in A's exact file formats, for tests and plumbing.

These labels are fake. Never report numbers from them.

Writes fake labels for the real pilot filings, for testing the full pipeline:
  .venv/bin/python scripts/fake_collection.py --pilot-labels
The file name deliberately does not match label.py's labels_*.jsonl budget glob.
"""
import argparse
import json
import os
import re

PILOT_LABELS_PATH = "data/processed/fake_labels_pilot.jsonl"
FAKE_QUESTION = "fake_question"
FAKE_RULE = re.compile(r"curbside|telehealth", re.I)

FILLER = ("the company operates in a competitive market and its results may vary from "
          "period to period because of general business conditions and regulation ")


def filler(n_words, tag):
    words = (FILLER * (n_words // 20 + 1)).split()[:n_words]
    return " ".join(words) + f" {tag}"


# Overlap is 50 words and chunks are 300, so a chunk boundary falls at word 300.
BOUNDARY_PHRASE = "quarterly marmalade reconciliation"
FILINGS = [
    {"filing_id": "1_2020", "cik": "1",
     "item_1a": filler(290, "alpha") + f" {BOUNDARY_PHRASE} " + filler(290, "alpha"),
     "item_7": filler(100, "alpha")},
    {"filing_id": "2_2020", "cik": "2",
     "item_1a": filler(280, "beta") + " zygomorphic zygomorphic tractors",
     "item_7": filler(100, "beta")},
    {"filing_id": "3_2020", "cik": "3",
     "item_1a": filler(280, "gamma") + " The marmalade was reconciled quarterly.",
     "item_7": ""},
    {"filing_id": "4_2020", "cik": "4",
     "item_1a": "Focus on OMITTED text with a very short section.", "item_7": ""},
    {"filing_id": "5_2020", "cik": "5", "item_1a": "", "item_7": ""},
    {"filing_id": "6_2020", "cik": "6",
     "item_1a": "We sell bicycle helmets to families.",
     "item_7": "Bicycle helmets bicycle helmets bicycle helmets remain our main product."},
] + [
    {"filing_id": f"{i}_2020", "cik": str(i), "item_1a": filler(400, f"tag{i}"),
     "item_7": filler(350, f"other{i}")}
    for i in range(7, 25)
]


def label_row(filing_id, answer):
    return {"filing_id": filing_id, "model": "fake", "usage": {}, "cost": 0.0,
            "questions": [FAKE_QUESTION],
            "labels": {FAKE_QUESTION: {"answer": answer, "quote": "fake" if answer else "",
                                       "quote_verified": True if answer else None}}}


def write_fake_collection(directory):
    """Writes filings.jsonl and labels.jsonl. Filing 24_2020 has only a failed label."""
    filings_path = os.path.join(directory, "filings.jsonl")
    labels_path = os.path.join(directory, "labels.jsonl")
    with open(filings_path, "w") as f:
        for filing in FILINGS:
            f.write(json.dumps(filing) + "\n")
    with open(labels_path, "w") as f:
        for filing in FILINGS:
            fid = filing["filing_id"]
            if fid == "24_2020":
                f.write(json.dumps({"filing_id": fid, "error": "http_429"}) + "\n")
            else:
                f.write(json.dumps(label_row(fid, fid in ("1_2020", "2_2020", "6_2020"))) + "\n")
    return filings_path, labels_path


def write_pilot_labels():
    wanted = set(open("data/processed/pilot_ids.txt").read().split())
    yes = total = 0
    with open(PILOT_LABELS_PATH, "w") as out:
        for line in open("data/processed/filings.jsonl"):
            filing = json.loads(line)
            if filing["filing_id"] in wanted:
                answer = bool(FAKE_RULE.search(filing["item_1a"] + " " + filing["item_7"]))
                out.write(json.dumps(label_row(filing["filing_id"], answer)) + "\n")
                yes += answer
                total += 1
    print(f"{PILOT_LABELS_PATH}: {yes}/{total} fake yes. Plumbing only; never report these.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-labels", action="store_true", required=True)
    p.parse_args()
    write_pilot_labels()
