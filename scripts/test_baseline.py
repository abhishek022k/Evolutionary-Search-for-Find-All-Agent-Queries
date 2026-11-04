"""Tests baseline.py and check_handoff.py on invented filings. No network, no model.

Run from the project root:  .venv/bin/python scripts/test_baseline.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search import SearchEngine  # noqa: E402
from embedders import HashEmbedder  # noqa: E402
from baseline import rrf_order, run_baseline  # noqa: E402
import build_index  # noqa: E402
from check_handoff import check  # noqa: E402
from fake_collection import FAKE_QUESTION, FILINGS, label_row, write_fake_collection  # noqa: E402
from judge import UnlabeledFiling, load_labels  # noqa: E402
from ledger import Ledger  # noqa: E402

QUERIES = [{"query_id": "q1", "branch": "exact", "query": "bicycle helmets"},
           {"query_id": "q2", "branch": "bm25", "query": "zygomorphic tractors"},
           {"query_id": "q3", "branch": "embedding", "query": "competitive market regulation"}]
ALL_IDS = [f["filing_id"] for f in FILINGS]


def raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


def write_rows(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_engine(d):
    filings_path, labels_path = write_fake_collection(d)
    cfg = {"filings_path": filings_path, "sections": ["item_1a", "item_7"], "chunk_words": 300,
           "chunk_overlap": 50, "embedding_model": "hash", "index_dir": os.path.join(d, "i"),
           "index_type": "hnsw", "hnsw_m": 16, "ef_construction": 80, "ef_search": 200,
           "search_k": 2000}
    with contextlib.redirect_stdout(io.StringIO()):
        build_index.build(cfg, cfg["index_dir"], embedder=HashEmbedder())
    return SearchEngine(cfg, embedder=HashEmbedder()), labels_path


def test_rrf_rewards_agreement_and_breaks_ties_by_id():
    order = rrf_order({"q1": ["a", "b"], "q2": ["c", "b"]})
    assert [fid for fid, _ in order] == ["b", "a", "c"]  # b is in both lists
    assert dict(order)["c"] == "q2"
    assert [fid for fid, _ in rrf_order({"q1": ["z"], "q2": ["y"]})] == ["y", "z"]


def test_baseline_stays_on_budget_and_logs_in_order():
    with tempfile.TemporaryDirectory() as d:
        engine, labels_path = make_engine(d)
        labels = load_labels(labels_path, FAKE_QUESTION)
        log_dir = os.path.join(d, "run")
        judge, ledger = run_baseline(engine, QUERIES[:2], labels, ALL_IDS, 2, log_dir)
        assert judge.used == 2
        assert set(ledger.found()) == {"6_2020", "2_2020"}  # the two planted answers
        rows = [json.loads(line) for line in open(os.path.join(log_dir, "judgments.jsonl"))]
        assert [r["seq"] for r in rows] == [1, 2]
        assert all(r["via"] == "baseline" and r["generation"] == 0 for r in rows)
        assert Ledger.replay(ALL_IDS, log_dir).snapshot() == ledger.snapshot()


def test_baseline_can_spend_past_fifty_per_query_and_stops_when_lists_end():
    with tempfile.TemporaryDirectory() as d:
        engine, labels_path = make_engine(d)
        labels = {fid: False for fid in ALL_IDS}
        judge, ledger = run_baseline(engine, QUERIES, labels, ALL_IDS, 1000,
                                     os.path.join(d, "run"))
        # Every filing with chunks is reachable; 5_2020 has none, so it stays untouched.
        assert judge.used == len(ALL_IDS) - 1
        assert ledger.untouched() == ["5_2020"]


def test_baseline_stops_loudly_on_a_filing_with_no_label():
    with tempfile.TemporaryDirectory() as d:
        engine, labels_path = make_engine(d)
        labels = load_labels(labels_path, FAKE_QUESTION)  # 24_2020 only has a failed row
        assert raises(UnlabeledFiling, run_baseline, engine, QUERIES, labels, ALL_IDS, 1000,
                      os.path.join(d, "run"))


def good_setup(d):
    filings_path, _ = write_fake_collection(d)
    labels_path = os.path.join(d, "labels_ok.jsonl")
    write_rows(labels_path, [label_row(fid, fid == "1_2020") for fid in ALL_IDS])
    gen0_path = os.path.join(d, "gen0.json")
    json.dump(QUERIES, open(gen0_path, "w"))
    return filings_path, labels_path, gen0_path


def test_handoff_passes_good_files():
    with tempfile.TemporaryDirectory() as d:
        filings, labels, gen0 = good_setup(d)
        problems, warnings, info = check(filings, labels, FAKE_QUESTION, gen0)
        assert problems == [] and warnings == [], (problems, warnings)
        assert any("1 relevant" in line for line in info)


def test_handoff_catches_failed_label_rows():
    with tempfile.TemporaryDirectory() as d:
        filings, labels, gen0 = good_setup(d)
        with open(labels, "a") as f:
            f.write(json.dumps({"filing_id": "3_2020", "error": "http_429"}) + "\n")
        problems, _, _ = check(filings, labels, FAKE_QUESTION, gen0)
        assert any("no successful label" in p for p in problems), problems


def test_handoff_catches_mismatched_filings_and_wrong_question():
    with tempfile.TemporaryDirectory() as d:
        filings, labels, gen0 = good_setup(d)
        with open(labels, "a") as f:
            f.write(json.dumps(label_row("777_2020", False)) + "\n")
        problems, _, _ = check(filings, labels, FAKE_QUESTION, gen0)
        assert any("not in the collection" in p for p in problems), problems
        problems, _, _ = check(filings, labels, "other_question", gen0)
        assert any("no label for" in p for p in problems), problems
        problems, _, _ = check(filings, None, None, gen0)
        assert any("must both be set" in p for p in problems), problems


def test_handoff_catches_bad_gen0_and_warns_on_prevalence_and_quotes():
    with tempfile.TemporaryDirectory() as d:
        filings, labels, gen0 = good_setup(d)
        json.dump([{"query_id": "q1", "branch": "grep", "query": "x"},
                   {"query_id": "q1", "branch": "bm25", "query": ""}], open(gen0, "w"))
        problems, _, _ = check(filings, labels, FAKE_QUESTION, gen0)
        assert any("need query_id" in p for p in problems), problems
        assert any("not unique" in p for p in problems), problems
        assert any("not found" in p for p in check(filings, labels, FAKE_QUESTION,
                                                   os.path.join(d, "nope.json"))[0])

        json.dump(QUERIES, open(gen0, "w"))
        row = label_row("1_2020", True)
        row["labels"][FAKE_QUESTION]["quote_verified"] = False
        rows = [row] + [label_row(fid, True) for fid in ALL_IDS if fid != "1_2020"]
        write_rows(labels, rows)
        _, warnings, _ = check(filings, labels, FAKE_QUESTION, gen0)
        assert any("outside the planned" in w for w in warnings), warnings
        assert any("not found in the filing" in w for w in warnings), warnings


for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn()
        print("passed:", name)
print("all tests passed")
