"""The search API. Import this before anything that imports torch (see embedders.py).

    from search import search
    hits = search("bm25", "work from home demand")   # [(chunk_id, filing_id, rank), ...]

k counts chunks. Ranks start at 1. Branches: bm25, embedding, exact. A branch can
return fewer than k: BM25 drops documents that match no query word, and exact
returns only real matches.

The first call loads the index named in config.json. Use SearchEngine directly to
load a different one, such as index/pilot.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from embedders import make_embedder  # noqa: E402  (loads faiss before torch)
import faiss  # noqa: E402
import numpy as np  # noqa: E402
from build_index import base_settings, index_settings, sections_tag  # noqa: E402
from label import normalize  # noqa: E402
from ledger import first_rank_per_filing  # noqa: E402,F401  (part of this module's API)

BRANCHES = ("bm25", "embedding", "exact")


class StaleIndex(Exception):
    """The index on disk does not match the current filings or settings."""


class SearchEngine:
    def __init__(self, cfg, index_dir=None, subset_ids=None, embedder=None):
        import bm25s
        import Stemmer

        self.cfg = cfg
        self.index_dir = index_dir or cfg["index_dir"]
        out_dir = os.path.join(self.index_dir, sections_tag(cfg["sections"]))
        self._check_manifests(out_dir, subset_ids)

        self._rows = np.load(os.path.join(out_dir, "rows.npy"))
        keep = set(self._rows.tolist())
        self._chunk_ids, self._filing_ids, self._texts = [], [], []
        self._position = {}
        self._filing_chunks = {}
        for row, line in enumerate(open(os.path.join(self.index_dir, "chunks.jsonl"))):
            if row not in keep:
                continue
            chunk = json.loads(line)
            position = len(self._chunk_ids)
            self._chunk_ids.append(chunk["chunk_id"])
            self._filing_ids.append(chunk["filing_id"])
            self._texts.append(chunk["text"])
            self._position[chunk["chunk_id"]] = position
            self._filing_chunks.setdefault(chunk["filing_id"], []).append(chunk["chunk_id"])
        self._normalized = None

        self._stemmer = Stemmer.Stemmer("english")
        self._bm25 = bm25s.BM25.load(os.path.join(out_dir, "bm25"))
        self._tokenize = bm25s.tokenize
        self._vectors = faiss.read_index(os.path.join(out_dir, "vectors.faiss"))
        self._all_embeddings = np.load(
            os.path.join(self.index_dir, "embeddings.npy"), mmap_mode="r")
        self._embedder = embedder

    def _check_manifests(self, out_dir, subset_ids):
        base = _read_manifest(os.path.join(self.index_dir, "base_manifest.json"))
        if not base.get("complete"):
            raise StaleIndex(f"{self.index_dir}: embedding did not finish; rerun build_index.py")
        expected = base_settings(self.cfg, subset_ids)
        if base["settings"] != expected:
            raise StaleIndex(f"{self.index_dir} does not match the current inputs: "
                             f"{_differences(base['settings'], expected)}")
        built = _read_manifest(os.path.join(out_dir, "manifest.json"))
        if built["settings"] != index_settings(self.cfg):
            raise StaleIndex(f"{out_dir} does not match the current settings: "
                             f"{_differences(built['settings'], index_settings(self.cfg))}")

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = make_embedder(self.cfg["embedding_model"])
        return self._embedder

    def search(self, branch, query, k=None):
        k = self.cfg["search_k"] if k is None else k
        if branch == "bm25":
            return self._search_bm25(query, k)
        if branch == "embedding":
            return self.search_by_vector(self.embedder.encode_queries([query])[0], k)
        if branch == "exact":
            return self._search_exact(query, k)
        raise ValueError(f"unknown branch {branch!r}; use one of {BRANCHES}")

    def _hits(self, positions):
        return [(self._chunk_ids[p], self._filing_ids[p], rank)
                for rank, p in enumerate(positions, start=1)]

    def _search_bm25(self, query, k):
        tokens = self._tokenize(query, stopwords="en", stemmer=self._stemmer,
                                return_ids=False, show_progress=False)[0]
        known = [t for t in tokens if t in self._bm25.vocab_dict]
        if not known:
            return []
        k = min(k, len(self._chunk_ids))
        ids, scores = self._bm25.retrieve([known], k=k, show_progress=False)
        # Documents matching no query word come back with score 0. Ties sort by position.
        pairs = [(-float(s), int(i)) for i, s in zip(ids[0], scores[0]) if s > 0]
        return self._hits([i for _, i in sorted(pairs)])

    def _search_exact(self, query, k):
        needle = normalize(query)
        if not needle:
            return []
        if self._normalized is None:
            self._normalized = [normalize(t) for t in self._texts]
        # A cheap substring check first; word boundaries stop "us" matching "focus".
        pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
        matches = []
        for position, text in enumerate(self._normalized):
            if needle in text:
                count = len(pattern.findall(text))
                if count:
                    matches.append((-count, self._chunk_ids[position], position))
        return self._hits([p for _, _, p in sorted(matches)[:k]])

    def search_by_vector(self, vector, k=None):
        """Nearest chunks to a vector. For amplify, pass a vector from get_vectors()."""
        k = self.cfg["search_k"] if k is None else k
        k = min(k, len(self._chunk_ids))
        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        if self.cfg["index_type"] == "hnsw":
            params = faiss.SearchParametersHNSW(efSearch=max(self.cfg["ef_search"], k))
            _, ids = self._vectors.search(query, k, params=params)
        else:
            _, ids = self._vectors.search(query, k)
        return self._hits([int(i) for i in ids[0] if i >= 0])

    def get_vectors(self, chunk_ids):
        rows = self._rows[[self._position[c] for c in chunk_ids]]
        return np.asarray(self._all_embeddings[rows], dtype="float32")

    def get_text(self, chunk_id):
        return self._texts[self._position[chunk_id]]

    def chunks_of(self, filing_id):
        """This filing's chunk ids. Empty if it has no text in the indexed sections."""
        return list(self._filing_chunks.get(filing_id, []))

    def chunk_count(self):
        return len(self._chunk_ids)


def _read_manifest(path):
    if not os.path.exists(path):
        raise StaleIndex(f"{path} is missing; run scripts/build_index.py")
    return json.load(open(path))


def _differences(built, expected):
    return {k: {"built": built.get(k), "now": expected.get(k)}
            for k in expected if built.get(k) != expected.get(k)}


_engine = None


def get_engine(config_path="config.json"):
    global _engine
    if _engine is None:
        _engine = SearchEngine(json.load(open(config_path)))
    return _engine


def search(branch, query, k=None):
    """The agreed search(branch, query, k=2000), using the index in config.json."""
    return get_engine().search(branch, query, k)
