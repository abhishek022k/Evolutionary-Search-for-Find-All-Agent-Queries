"""Split filings into overlapping word chunks. Each section is chunked on its own.

Run through build_index.py, which saves the result as chunks.jsonl.
"""
import json

ALL_SECTIONS = ["item_1a", "item_7"]


def chunk_text(text, chunk_words, overlap):
    """Word windows with whitespace collapsed. The last window may be short."""
    if not 0 <= overlap < chunk_words:
        raise ValueError(f"need 0 <= overlap < chunk_words, got {overlap}, {chunk_words}")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
        start += chunk_words - overlap
    return chunks


def chunk_filing(filing, chunk_words, overlap):
    for section in ALL_SECTIONS:
        for i, text in enumerate(chunk_text(filing.get(section) or "", chunk_words, overlap)):
            yield {"chunk_id": f"{filing['filing_id']}#{section}#{i:04d}",
                   "filing_id": filing["filing_id"], "section": section, "text": text}


def read_filings(path, only_ids=None):
    with open(path) as f:
        for line in f:
            filing = json.loads(line)
            if only_ids is None or filing["filing_id"] in only_ids:
                yield filing
