# Evolutionary-Search-for-Find-All-Agent-Queries
Course Project for CS839 - Next Generation Systems

## Retrieval status

Search, the ledger, the simulated judge and the baseline work end to end on the real collection, tested so far with fake labels only.

| Still needed | From |
|---|---|
| Full labels file and final question | A |
| Generation 0 queries | A |
| Confirmation that `filings.jsonl` matches (see below) | A |
| Replacing the mock `search()` with these modules | C |

## Retrieval setup

Needs Python 3.10–3.14 (`faiss-cpu` does not support 3.15). Run everything from the repo root.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_raw.py       # about 2 GB
.venv/bin/python scripts/select_filings.py     # same seed, same 3,000 filings
.venv/bin/python scripts/build_index.py --pilot   # 200 filings, about 4 minutes
.venv/bin/python scripts/fake_collection.py --pilot-labels   # fake labels for testing only
```

`scripts/build_index.py` without `--pilot` builds the full index. It took about 85 minutes on an M3 laptop, nearly all of it embedding, and resumes if interrupted.

To check you have the same filings, run `shasum -a 256 data/processed/filings.jsonl`. It should print `9800a8c9b9c5fc6e9606e9dd48f2d27d6c262109bf21f7989f9c4d3f7d723386`. This has not yet been compared with A's original file. All 37 filing IDs in A's pilot reports are in this selection, which checks the IDs but not the text.

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

For the full index, `search(branch, query, k=2000)` from `search.py` is the agreed one-line call.

- **Branches:** `bm25`, `embedding`, `exact`. `k` counts chunks, ranks start at 1, and a branch can return fewer than `k`.
- **For amplify:** `engine.search_by_vector(vector, k)`, `engine.get_vectors(chunk_ids)`, `engine.get_text(chunk_id)`, `engine.chunks_of(filing_id)`.
- **Ledger pools:** `found()`, `rejected()`, `seen_unjudged()`, `untouched()`, `near_miss(lo, hi)`, `counts()`.
- **Sampling:** pass `draw_prob` to `record_judgment`, and it marks the filing as sampled.
- **Breeding costs:** `judge.CostTracker`, kept separate from judgments.
- **Logs:** `judgments.jsonl` holds the judgment order for recall curves, and `Ledger.replay()` rebuilds a run from its logs.

A filing with no successful label raises `UnlabeledFiling` when judged. It never counts as not relevant.

## Handing over labels

Set `labels_path` and `question_id` in `config.json`, and save the Generation 0 queries as `data/processed/gen0_queries.json`, a list of `{"query_id", "branch", "query"}`. Then:

```
.venv/bin/python scripts/check_handoff.py
.venv/bin/python baseline.py --budget 1000
```

`check_handoff.py` prints counts only, never quotes, so the Generation 0 author can run it. Whoever writes those queries should not open `eval/results/`, which contains quotes from filings labeled relevant.

Labels never touch the index. Changing `sections` or `index_type` only needs `scripts/build_index.py` rerun, which took about 30 seconds with no re-embedding.

## Changes from the POC plan

**Built this way:**

| POC plan | What we did | Why |
|---|---|---|
| HNSW for embedding search | Exact flat index (`index_type: flat`) | HNSW was not faster at this size and `k` (7 ms vs. 5 ms) and missed up to 2.1% of results. With exact search, any misses come from the embedding, not the index. |
| Item 1A only, about 3,000 filings | Item 1A and Item 7, chunked separately | Matches A's collection. `sections` can switch to Item 7 only. |
| Small `sentence-transformers` model | `BAAI/bge-small-en-v1.5` | Its limit is 512 tokens. `all-MiniLM-L6-v2` stops at 256 and would cut off most 300-word chunks. |
| Chunks of about 300 words | 300 words with 50 overlapping | Phrases are not lost at chunk boundaries. |
| `diskcache` for the ledger and judgments | In-memory ledger with JSONL logs | The simulated judge costs nothing to repeat. Caching can wait for the real judge. |
| `index/` holds indexing code | Code is in `scripts/` and the repo root | `index/` is gitignored, so code there would never be committed. |

**Added to the agreed interfaces, without changing their signatures:**
- `record_judgment` is separate from `judge(filing_id)`, which cannot record the query or generation. The loop is `observe` → `top_unjudged` → `judge` → `record_judgment`.
- `Sampled` is a flag plus `draw_prob`, not a state, because a sampled filing is also found or rejected.
- `near_miss()` applies the 51–2,000 range, while `seen_unjudged()` returns every seen, unjudged filing. A top-50 filing left unjudged when the budget runs out is only in `seen_unjudged()`, so we recommend sampling from that pool.

**Proposed, for the team to confirm:**
- **`k` counts chunks, not filings.** A filing-level top 1,000 is a third of the collection.
- **Report unreachable mass at several depths** (top 100, 250, 500 and 1,000 chunks). Even in chunks, one query reaches many filings (see below), so a single cutoff could make unreachable mass look small. This only changes the evaluation script.
- **The baseline spends the whole budget.** Generation 0 at 50 judgments per query only uses 250–400, so `baseline.py` judges down a reciprocal rank fusion of the Generation 0 rankings until the budget ends.

## Measurements

Full index: 3,000 filings and 270,029 chunks (median 84 per filing, maximum 394). In a sample of 2,000 chunks, the median is 366 tokens and 0.1% exceed the 512-token limit.

Filings reached per query, using diagnostic queries rather than Generation 0 queries, on BM25 and embedding search:

| `k` (chunks) | Filings reached | Share of 3,000 |
|---|---|---|
| 1,000 | 502–792 | 17–26% |
| 2,000 | 829–1,349 | 28–45% |

| Latency per query | Time |
|---|---|
| BM25 | 2–6 ms |
| Embedding search, including embedding the query | 13–18 ms; first call about 90 ms |
| Exact phrase | 250–820 ms on the HNSW run; first call about 12 s, most likely from normalizing all the text once |

HNSW comparison, from the run before switching (`hnsw_m` 32, `ef_search` 2500), against exact search at `k`=2000 over 26 queries: mean recall 0.997, minimum 0.979, median latency 7 ms vs. 5 ms. To measure it again, set `index_type` to `hnsw`, rerun `build_index.py`, then run `scripts/check_index.py`.

Paraphrase check, where each query has no exact-phrase match: chunks mentioning the target topic among the top 10 embedding results were 10/10 for cyber breaches, 9/10 for interest rates and 3/10 for telehealth. Telehealth is usually a passing mention inside a 300-word chunk.

Regenerate these with `.venv/bin/python scripts/check_index.py` (add `--pilot` for the 200-filing index).

## Tests

No network or model download needed:

```
.venv/bin/python scripts/test_judge_ledger.py
.venv/bin/python scripts/test_search.py
.venv/bin/python scripts/test_baseline.py
```
