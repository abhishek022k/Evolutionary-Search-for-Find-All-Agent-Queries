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

## Labels (do not open if you are writing Generation 0 queries)

- `answer_key_stay_home_boom.jsonl` is what `judge()` should read. Each line has
  `filing_id`, `relevant` and `needs_review`. Filings with `needs_review` have a yes
  whose quote was not found in the filing, and they count as yes until checked.
- `data/processed/labels_full_stay_home_boom_claude-haiku-5-5.jsonl` is the full label
  file with supporting quotes.

Rebuild the key after corrections with `scripts/build_answer_key.py`.
