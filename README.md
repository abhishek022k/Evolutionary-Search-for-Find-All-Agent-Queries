# Evolutionary-Search-for-Find-All-Agent-Queries
Course Project for CS839 - Next Generation Systems

## Retrieval setup

Needs Python 3.10–3.14 (`faiss-cpu` does not support 3.15). Run everything from the repo root.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_raw.py       # about 2 GB
.venv/bin/python scripts/select_filings.py     # same seed, same 3,000 filings
.venv/bin/python scripts/build_index.py --pilot   # 200 filings, a few minutes
.venv/bin/python scripts/fake_collection.py --pilot-labels   # fake labels for testing only
```

`scripts/build_index.py` without `--pilot` builds the full index. It takes about an hour on an M3 laptop and resumes if interrupted.

**Import `search` or `embedders` before anything that imports `torch`.** On macOS, loading faiss after torch segfaults. These modules raise an error if the order is wrong.

## Using search, the judge and the ledger

```python
from search import SearchEngine, first_rank_per_filing   # first, before torch
from judge import BudgetExhausted, Judge, load_labels
from ledger import Ledger
import json

cfg = json.load(open("config.json"))
pilot = set(open("data/processed/pilot_ids.txt").read().split())
engine = SearchEngine(cfg, index_dir="index/pilot", subset_ids=pilot)   # full index: SearchEngine(cfg)
labels = load_labels("data/processed/fake_labels_pilot.jsonl", "fake_question")
judge = Judge(labels, budget=300)
ledger = Ledger(sorted(pilot), log_dir="eval/runs/my_run")   # one per run; the folder must be new

hits = engine.search("bm25", "work from home", k=2000)   # [(chunk_id, filing_id, rank), ...]
ledger.observe(hits, "q1", generation=0)                 # returns overlap counts
for filing_id in ledger.top_unjudged(hits, n=50):
    relevant = judge.judge(filing_id)                    # raises BudgetExhausted at the limit
    ledger.record_judgment(filing_id, relevant, "q1", 0, via="search")
```

- **Branches:** `bm25`, `embedding`, `exact`. `k` counts chunks, ranks start at 1, and a branch can return fewer than `k`.
- **For amplify:** `engine.search_by_vector(vector, k)`, `engine.get_vectors(chunk_ids)`, `engine.get_text(chunk_id)`, `engine.chunks_of(filing_id)`.
- **Ledger pools:** `found()`, `rejected()`, `seen_unjudged()`, `untouched()`, `near_miss(lo, hi)`, `counts()`.
- **Sampling:** pass `draw_prob` to `record_judgment`, and it marks the filing as sampled.
- **Breeding costs:** `judge.CostTracker`, kept separate from judgments.
- **Logs:** `judgments.jsonl` holds the judgment order for recall curves, and `Ledger.replay()` rebuilds a run from its logs.

## Handing over labels

Set `labels_path` and `question_id` in `config.json`, and save the Generation 0 queries as `data/processed/gen0_queries.json`, a list of `{"query_id", "branch", "query"}`. Then:

```
.venv/bin/python scripts/check_handoff.py
.venv/bin/python baseline.py --budget 1000
```

No index rebuild is needed unless `sections` changes, and then only `scripts/build_index.py` needs rerunning, with no re-embedding.

## Tests

No network or model download needed:

```
.venv/bin/python scripts/test_judge_ledger.py
.venv/bin/python scripts/test_search.py
.venv/bin/python scripts/test_baseline.py
```
