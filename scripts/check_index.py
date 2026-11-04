"""Measure a built index: size, truncation, latency, HNSW recall, filings per query.

Diagnostic queries are the question definitions from questions.py plus a few unrelated
ones. They are not Generation 0 queries, and no labels are read.

Pilot:  .venv/bin/python scripts/check_index.py --pilot
Full:   .venv/bin/python scripts/check_index.py
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search import SearchEngine, first_rank_per_filing  # noqa: E402
import faiss  # noqa: E402
import numpy as np  # noqa: E402
from build_index import PILOT_IDS_PATH, sections_tag  # noqa: E402
from questions import QUESTIONS  # noqa: E402

GENERIC = ["interest rate risk on variable rate debt",
           "cybersecurity breach of customer data",
           "dependence on a single supplier"]
EXACT = ["pandemic", "stay-at-home", "supply chain", "interest rate"]
PARAPHRASES = [
    ("seeing a doctor over video instead of visiting in person",
     re.compile(r"telehealth|telemedicine|virtual visit", re.I)),
    ("hackers stealing private customer records",
     re.compile(r"cyber|data breach|security breach", re.I)),
    ("borrowing costs going up when the central bank tightens",
     re.compile(r"interest rate", re.I)),
]
TOKEN_SAMPLE = 2000
AMPLIFY_SAMPLE = 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    subset = set(open(PILOT_IDS_PATH).read().split()) if args.pilot else None
    index_dir = "index/pilot" if args.pilot else cfg["index_dir"]
    start = time.time()
    engine = SearchEngine(cfg, index_dir=index_dir, subset_ids=subset)
    print(f"loaded {index_dir} in {time.time() - start:.1f}s (without the embedding model)")

    n_filings = len(engine._filing_chunks)
    per_filing = sorted(len(v) for v in engine._filing_chunks.values())
    print(f"filings with chunks: {n_filings}; chunks: {engine.chunk_count()}; "
          f"per filing median {statistics.median(per_filing):.0f}, max {per_filing[-1]}")
    print(f"sections: {sections_tag(cfg['sections'])}")

    rng = random.Random(0)
    sample = rng.sample(engine._texts, min(TOKEN_SAMPLE, engine.chunk_count()))
    counts = engine.embedder.token_counts(sample)
    limit = engine.embedder.model.max_seq_length
    print(f"tokens per chunk (sample of {len(sample)}): median {statistics.median(counts):.0f}, "
          f"max {max(counts)}; over the {limit}-token limit: "
          f"{100 * sum(c > limit for c in counts) / len(counts):.1f}%")

    text_queries = [q["question"] for q in QUESTIONS.values()] + GENERIC
    k_small, k_large = 1000, cfg["search_k"]
    print(f"\nfilings per query out of {n_filings} (k counts chunks)")
    for branch, queries in (("bm25", text_queries), ("embedding", text_queries),
                            ("exact", EXACT)):
        for query in queries:
            t = time.time()
            hits = engine.search(branch, query, k_large)
            ms = (time.time() - t) * 1000
            at_small = len(first_rank_per_filing([h for h in hits if h[2] <= k_small]))
            at_large = len(first_rank_per_filing(hits))
            print(f"  {branch:9s} {len(hits):5d} chunks, {at_small:5d} filings at k={k_small}, "
                  f"{at_large:5d} at k={k_large}, {ms:7.0f} ms  {query[:48]}")

    # Paraphrase check: the query shares no word with the target, so only meaning connects them.
    print("\nparaphrase check (top 10 embedding hits)")
    for query, target in PARAPHRASES:
        exact_hits = len(engine.search("exact", query, k_large))
        top = engine.search("embedding", query, 10)
        with_target = sum(target.search(engine.get_text(h[0])) is not None for h in top)
        print(f"  {with_target}/10 mention {target.pattern!r}; exact hits for the query: "
              f"{exact_hits}  ({query})")

    # Recall against an exact search over the same vectors, from text and amplify queries.
    rows = np.load(os.path.join(index_dir, sections_tag(cfg["sections"]), "rows.npy"))
    vectors = np.asarray(
        np.load(os.path.join(index_dir, "embeddings.npy"), mmap_mode="r")[rows], dtype="float32")
    flat = faiss.IndexFlatIP(vectors.shape[1])
    flat.add(vectors)
    query_vectors = list(engine.embedder.encode_queries(text_queries))
    seed_positions = rng.sample(range(len(vectors)), min(AMPLIFY_SAMPLE, len(vectors)))
    query_vectors += [vectors[i] for i in seed_positions]

    recalls, ann_ms, flat_ms = [], [], []
    for query in query_vectors:
        t = time.time()
        ann = {h[0] for h in engine.search_by_vector(query, k_large)}
        ann_ms.append((time.time() - t) * 1000)
        t = time.time()
        _, ids = flat.search(query.reshape(1, -1), min(k_large, len(vectors)))
        flat_ms.append((time.time() - t) * 1000)
        exact = {engine._chunk_ids[int(i)] for i in ids[0] if i >= 0}
        recalls.append(len(ann & exact) / len(exact))
    print(f"\n{cfg['index_type']} recall@{k_large} vs. exact, {len(query_vectors)} queries: "
          f"mean {statistics.mean(recalls):.3f}, min {min(recalls):.3f}")
    print(f"latency: {cfg['index_type']} median {statistics.median(ann_ms):.0f} ms, "
          f"exact flat median {statistics.median(flat_ms):.0f} ms")


if __name__ == "__main__":
    main()
