"""Text embedders. Import this (or search.py) before anything that imports torch.

On macOS, loading faiss after torch segfaults on the first HNSW call (two OpenMP
runtimes in one process). Loading faiss first works, so we check the order here
and fail with a clear message.
"""
import hashlib
import re
import sys

if "torch" in sys.modules and "faiss" not in sys.modules:
    raise ImportError("Import search or embedders before torch or sentence_transformers: "
                      "loading faiss after torch segfaults on macOS.")

import faiss  # noqa: E402,F401
import numpy as np  # noqa: E402

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class HashEmbedder:
    """Deterministic word-overlap vectors for tests. No model download, no meaning."""
    name = "hash"

    def __init__(self, dim=4096):  # large, so different words rarely share a bucket
        self.dim = dim

    def encode_chunks(self, texts):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(word.encode()).digest()
                vectors[i, int.from_bytes(digest[:4], "little") % self.dim] += 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    encode_queries = encode_chunks


class BgeEmbedder:
    def __init__(self, name, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.batch_size = batch_size
        self.model = SentenceTransformer(name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True).astype("float32")

    def encode_chunks(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode([BGE_QUERY_PREFIX + t for t in texts])

    def token_counts(self, texts):
        """Token lengths before truncation, to measure how much gets cut off."""
        ids = self.model.tokenizer(texts, truncation=False, add_special_tokens=True,
                                   verbose=False)["input_ids"]
        return [len(x) for x in ids]


def make_embedder(name):
    return HashEmbedder() if name == "hash" else BgeEmbedder(name)
