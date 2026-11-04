"""Step 1: download the raw EDGAR-CORPUS 2020 files into data/raw/.

Fetches the JSONL files directly, which skips the dataset's loading script.
Reruns skip files already present. select_filings.py reads the result.

Run from the project root:  .venv/bin/python scripts/download_raw.py
"""
import os
import shutil

from huggingface_hub import hf_hub_download

REPO = "eloukas/edgar-corpus"
SPLITS = ["train", "validate", "test"]

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
for split in SPLITS:
    out_path = f"data/raw/2020_{split}.jsonl"
    if os.path.exists(out_path):
        print(f"already have {out_path}")
        continue
    cached = hf_hub_download(REPO, f"2020/{split}.jsonl", repo_type="dataset")
    shutil.copyfile(cached, out_path)
    print(f"saved {out_path}: {os.path.getsize(out_path) / 1e6:.0f} MB")
