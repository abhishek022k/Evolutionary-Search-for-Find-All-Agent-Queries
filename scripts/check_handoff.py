"""Check A's labels and Generation 0 queries before a real run. Sends nothing anywhere.

Prints counts only, never quotes, so the Generation 0 author can run it safely.

Run from the project root:  .venv/bin/python scripts/check_handoff.py
Exits with an error if anything would break or invalidate a run.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label import load_done  # noqa: E402

BRANCHES = {"bm25", "embedding", "exact"}
MIN_PREVALENCE, MAX_PREVALENCE = 0.01, 0.05


def check(filings_path, labels_path, question_id, gen0_path):
    """Returns (problems, warnings, info). Problems block a run; warnings need a look."""
    problems, warnings, info = [], [], []
    filing_ids = [json.loads(line)["filing_id"] for line in open(filings_path)]
    if len(set(filing_ids)) != len(filing_ids):
        problems.append("filings file has duplicate filing ids")
    wanted = set(filing_ids)

    if not labels_path or not question_id:
        problems.append("labels_path and question_id must both be set")
    elif not os.path.exists(labels_path):
        problems.append(f"labels file not found: {labels_path}")
    else:
        rows = load_done(labels_path)
        extra = set(rows) - wanted
        if extra:
            problems.append(f"{len(extra)} labeled filings are not in the collection "
                            f"(different filings file?)")
        labeled = {fid: r for fid, r in rows.items() if "labels" in r and fid in wanted}
        missing_question = [fid for fid, r in labeled.items() if question_id not in r["labels"]]
        if missing_question:
            problems.append(f"{len(missing_question)} rows have no label for {question_id!r}")
        else:
            unlabeled = wanted - set(labeled)
            if unlabeled:
                problems.append(f"{len(unlabeled)} of {len(wanted)} filings have no successful "
                                f"label; judging one would stop the run. Rerun label.py.")
            yes = [r for r in labeled.values() if r["labels"][question_id]["answer"]]
            unverified = sum(1 for r in yes if not r["labels"][question_id]["quote_verified"])
            prevalence = len(yes) / len(labeled) if labeled else 0
            info.append(f"labeled {len(labeled)} of {len(wanted)}; {len(yes)} relevant "
                        f"({100 * prevalence:.1f}%); {unverified} yes rows with unverified quotes")
            if not MIN_PREVALENCE <= prevalence <= MAX_PREVALENCE:
                warnings.append(f"prevalence {100 * prevalence:.1f}% is outside the planned 1-5%")
            if unverified:
                warnings.append(f"{unverified} yes rows have quotes not found in the filing; "
                                f"they count as relevant unless A decides otherwise")

    if not os.path.exists(gen0_path):
        problems.append(f"Generation 0 file not found: {gen0_path}")
    else:
        queries = json.load(open(gen0_path))
        bad = [q for q in queries if q.get("branch") not in BRANCHES or not q.get("query")
               or not q.get("query_id")]
        ids = [q.get("query_id") for q in queries]
        if not queries:
            problems.append("Generation 0 file has no queries")
        if bad:
            problems.append(f"{len(bad)} Generation 0 queries need query_id, query, and a branch "
                            f"from {sorted(BRANCHES)}")
        if len(set(ids)) != len(ids):
            problems.append("Generation 0 query ids are not unique")
        branches = sorted({q.get("branch") for q in queries} & BRANCHES)
        info.append(f"{len(queries)} Generation 0 queries on branches: {branches}")
    return problems, warnings, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--gen0", default="data/processed/gen0_queries.json")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    problems, warnings, info = check(
        cfg["filings_path"], cfg["labels_path"], cfg["question_id"], args.gen0)
    for line in info:
        print("info:", line)
    for line in warnings:
        print("WARNING:", line)
    for line in problems:
        print("PROBLEM:", line)
    if problems:
        sys.exit(1)
    print("handoff OK")


if __name__ == "__main__":
    main()
