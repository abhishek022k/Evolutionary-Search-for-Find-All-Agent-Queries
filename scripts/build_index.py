"""Build the search indexes. Embedding resumes if interrupted.

Writes to <index_dir>:
  chunks.jsonl, embeddings.npy, base_manifest.json   every section, embedded once
  <sections>/ rows.npy, bm25/, vectors.faiss, manifest.json   one per section choice

Changing "sections" in the config only rebuilds <sections>/, with no re-embedding.

Full collection:  .venv/bin/python scripts/build_index.py
Pilot filings:    .venv/bin/python scripts/build_index.py --pilot
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedders import make_embedder  # noqa: E402  (loads faiss before torch)
import faiss  # noqa: E402
import numpy as np  # noqa: E402
from chunk import chunk_filing, read_filings  # noqa: E402

EMBED_BLOCK = 2048
PILOT_IDS_PATH = "data/processed/pilot_ids.txt"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sections_tag(sections):
    return "+".join(sorted(sections))


def base_settings(cfg, subset_ids):
    subset = None
    if subset_ids is not None:
        subset = hashlib.sha256("\n".join(sorted(subset_ids)).encode()).hexdigest()
    return {"filings_sha256": file_hash(cfg["filings_path"]), "subset_sha256": subset,
            "chunk_words": cfg["chunk_words"], "chunk_overlap": cfg["chunk_overlap"],
            "embedding_model": cfg["embedding_model"]}


def index_settings(cfg):
    return {"sections": sorted(cfg["sections"]), "index_type": cfg["index_type"],
            "hnsw_m": cfg["hnsw_m"], "ef_construction": cfg["ef_construction"]}


def read_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def build_base(cfg, index_dir, subset_ids, embedder):
    """Chunks and embeddings for every section. Reuses what is already there."""
    os.makedirs(index_dir, exist_ok=True)
    settings = base_settings(cfg, subset_ids)
    manifest_path = os.path.join(index_dir, "base_manifest.json")
    chunks_path = os.path.join(index_dir, "chunks.jsonl")
    vectors_path = os.path.join(index_dir, "embeddings.npy")
    progress_path = os.path.join(index_dir, "embeddings.progress")

    existing = read_json(manifest_path)
    if existing is not None and existing["settings"] != settings:
        raise SystemExit(f"{index_dir} was built from different inputs or settings. "
                         f"Delete it or use another index_dir.")

    if existing is None:
        texts, n_filings = [], 0
        tmp_path = chunks_path + ".tmp"
        with open(tmp_path, "w") as out:
            for filing in read_filings(cfg["filings_path"], subset_ids):
                n_filings += 1
                for chunk in chunk_filing(filing, cfg["chunk_words"], cfg["chunk_overlap"]):
                    out.write(json.dumps(chunk) + "\n")
                    texts.append(chunk["text"])
        os.replace(tmp_path, chunks_path)
        if os.path.exists(progress_path):
            os.remove(progress_path)
        vectors = np.lib.format.open_memmap(
            vectors_path, mode="w+", dtype="float32", shape=(len(texts), embedder.dim))
        write_json(manifest_path, {"settings": settings, "n_filings": n_filings,
                                   "n_chunks": len(texts), "dim": embedder.dim,
                                   "complete": False})
        done = 0
    else:
        if existing["complete"]:
            print(f"{index_dir}: {existing['n_chunks']} chunks already embedded")
            return existing
        texts = [json.loads(line)["text"] for line in open(chunks_path)]
        vectors = np.lib.format.open_memmap(vectors_path, mode="r+")
        done = int(open(progress_path).read()) if os.path.exists(progress_path) else 0
        print(f"resuming embedding at {done}/{len(texts)}")

    start, start_done = time.time(), done
    while done < len(texts):
        block = texts[done:done + EMBED_BLOCK]
        vectors[done:done + len(block)] = embedder.encode_chunks(block)
        vectors.flush()
        done += len(block)
        with open(progress_path, "w") as f:
            f.write(str(done))
        rate = (done - start_done) / max(time.time() - start, 1e-9)
        print(f"  embedded {done}/{len(texts)}, {rate:.0f} chunks/s, "
              f"{(len(texts) - done) / max(rate, 1e-9) / 60:.0f} min left", flush=True)

    manifest = read_json(manifest_path)
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    return manifest


def build_indexes(cfg, index_dir):
    """BM25 and vector indexes for the configured sections."""
    import bm25s
    import Stemmer

    out_dir = os.path.join(index_dir, sections_tag(cfg["sections"]))
    os.makedirs(out_dir, exist_ok=True)
    wanted = set(cfg["sections"])
    rows, texts = [], []
    for row, line in enumerate(open(os.path.join(index_dir, "chunks.jsonl"))):
        chunk = json.loads(line)
        if chunk["section"] in wanted:
            rows.append(row)
            texts.append(chunk["text"])
    rows = np.array(rows, dtype="int64")
    np.save(os.path.join(out_dir, "rows.npy"), rows)

    start = time.time()
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"),
                            show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    retriever.save(os.path.join(out_dir, "bm25"))
    print(f"BM25: {len(texts)} chunks in {time.time() - start:.0f}s")

    start = time.time()
    all_vectors = np.load(os.path.join(index_dir, "embeddings.npy"), mmap_mode="r")
    vectors = np.asarray(all_vectors[rows], dtype="float32")
    if cfg["index_type"] == "hnsw":
        index = faiss.IndexHNSWFlat(vectors.shape[1], cfg["hnsw_m"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg["ef_construction"]
    elif cfg["index_type"] == "flat":
        index = faiss.IndexFlatIP(vectors.shape[1])
    else:
        raise SystemExit(f"unknown index_type {cfg['index_type']!r}")
    index.add(vectors)
    faiss.write_index(index, os.path.join(out_dir, "vectors.faiss"))
    print(f"{cfg['index_type']}: {len(vectors)} vectors in {time.time() - start:.0f}s")
    write_json(os.path.join(out_dir, "manifest.json"), {"settings": index_settings(cfg)})


def build(cfg, index_dir, subset_ids=None, embedder=None):
    embedder = embedder or make_embedder(cfg["embedding_model"])
    manifest = build_base(cfg, index_dir, subset_ids, embedder)
    build_indexes(cfg, index_dir)
    print(f"built {index_dir}: {manifest['n_filings']} filings, {manifest['n_chunks']} chunks")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--pilot", action="store_true", help="Only the pilot filings, in index/pilot.")
    p.add_argument("--embed-only", action="store_true", help="Stop after embedding.")
    args = p.parse_args()

    cfg = json.load(open(args.config))
    subset = set(open(PILOT_IDS_PATH).read().split()) if args.pilot else None
    index_dir = "index/pilot" if args.pilot else cfg["index_dir"]
    if args.embed_only:
        build_base(cfg, index_dir, subset, make_embedder(cfg["embedding_model"]))
    else:
        build(cfg, index_dir, subset)


if __name__ == "__main__":
    main()
