# Shared data

`filings.jsonl.xz` holds the final 3,000 filings (2020 10-Ks) used by the project.
Each line has `filing_id`, `cik`, `item_1a` and `item_7`.

Unpack it from the project root (works anywhere Python 3 runs):

```bash
mkdir -p data/processed
python3 -c "import lzma, shutil; shutil.copyfileobj(lzma.open('data/shared/filings.jsonl.xz'), open('data/processed/filings.jsonl', 'wb'))"
```

The result is 422 MB. Its SHA-256 is
`9800a8c9b9c5fc6e9606e9dd48f2d27d6c262109bf21f7989f9c4d3f7d723386`.

The search index must use the same text as the labels: Item 1A plus Item 7.
