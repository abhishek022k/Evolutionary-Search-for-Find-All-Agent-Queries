"""Baseline: Generation 0 hybrid search, spending the whole judgment budget.

Runs every Generation 0 query, fuses the filing rankings with reciprocal rank fusion,
and judges down the fused list until the budget or the list runs out. Writes the same
logs as an evolution run, so the same evaluation reads both.

Gen 0 file: a JSON list of {"query_id", "branch", "query"}.

Usage: .venv/bin/python baseline.py --budget 1000
Pilot plumbing test, with fake labels:
  .venv/bin/python baseline.py --pilot --budget 50 --labels data/processed/fake_labels_pilot.jsonl \
      --question fake_question --gen0 <file>
"""
import argparse
import json
import os
import time

from search import SearchEngine, first_rank_per_filing
from judge import BudgetExhausted, Judge, load_labels
from ledger import Ledger

GEN0_PATH = "data/processed/gen0_queries.json"
PILOT_IDS_PATH = "data/processed/pilot_ids.txt"
RRF_K = 60


def rrf_order(ranked_filings):
    """Fuse {query_id: [filing_id, ...]} lists, best first. Ties sort by filing id.

    Returns [(filing_id, query_id)], where query_id gave the filing its best position.
    """
    scores, best = {}, {}
    for query_id, filings in sorted(ranked_filings.items()):
        for position, filing_id in enumerate(filings, start=1):
            scores[filing_id] = scores.get(filing_id, 0.0) + 1.0 / (RRF_K + position)
            if filing_id not in best or position < best[filing_id][0]:
                best[filing_id] = (position, query_id)
    order = sorted(scores, key=lambda fid: (-scores[fid], fid))
    return [(fid, best[fid][1]) for fid in order]


def run_baseline(engine, queries, labels, filing_ids, budget, log_dir, k=None):
    judge = Judge(labels, budget)
    ledger = Ledger(filing_ids, log_dir=log_dir)
    ranked = {}
    for q in queries:
        hits = engine.search(q["branch"], q["query"], k)
        ledger.observe(hits, q["query_id"], 0)
        ranked[q["query_id"]] = list(first_rank_per_filing(hits))
    try:
        for filing_id, query_id in rrf_order(ranked):
            relevant = judge.judge(filing_id)
            ledger.record_judgment(filing_id, relevant, query_id, 0, "baseline")
    except BudgetExhausted:
        pass
    ledger.close()
    return judge, ledger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--budget", type=int)
    p.add_argument("--gen0", default=GEN0_PATH)
    p.add_argument("--labels")
    p.add_argument("--question")
    p.add_argument("--pilot", action="store_true", help="Use index/pilot and the pilot filings.")
    p.add_argument("--run-id")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    labels_path = args.labels or cfg["labels_path"]
    question = args.question or cfg["question_id"]
    if not labels_path or not question:
        raise SystemExit("Set labels_path and question_id in config.json, or pass "
                         "--labels and --question.")
    if not os.path.exists(args.gen0):
        raise SystemExit(f"{args.gen0} is missing. It is a JSON list of "
                         '{"query_id", "branch", "query"}.')

    subset = set(open(PILOT_IDS_PATH).read().split()) if args.pilot else None
    index_dir = "index/pilot" if args.pilot else cfg["index_dir"]
    engine = SearchEngine(cfg, index_dir=index_dir, subset_ids=subset)
    filing_ids = [json.loads(line)["filing_id"] for line in open(cfg["filings_path"])]
    if subset is not None:
        filing_ids = [fid for fid in filing_ids if fid in subset]

    labels = load_labels(labels_path, question)
    budget = args.budget if args.budget is not None else cfg["judgment_budget"]
    run_id = args.run_id or f"baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("eval", "runs", run_id)
    queries = json.load(open(args.gen0))
    judge, ledger = run_baseline(engine, queries, labels, filing_ids, budget, log_dir)

    in_scope = sum(labels.get(fid, False) for fid in filing_ids)
    found = len(ledger.found())
    print(f"{run_id}: {judge.used}/{budget} judgments, {found} found of {in_scope} labeled "
          f"relevant (recall {found / in_scope if in_scope else 0:.3f}); {ledger.counts()}")
    print(f"logs in {log_dir}")


if __name__ == "__main__":
    main()
