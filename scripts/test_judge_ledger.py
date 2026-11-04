"""Tests judge.py and ledger.py on invented data. No network, no API calls.

Run from the project root:  .venv/bin/python scripts/test_judge_ledger.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from judge import BudgetExhausted, CostTracker, Judge, UnlabeledFiling, load_labels  # noqa: E402
from ledger import Ledger, first_rank_per_filing  # noqa: E402

IDS = ["a", "b", "c", "d", "e"]
LABELS = {"a": True, "b": False, "c": True, "d": False}  # e has no label


def raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


def hits(*filing_ranks):
    return [(f"{fid}#item_1a#{rank:04d}", fid, rank) for fid, rank in filing_ranks]


def test_repeat_judgment_is_free():
    j = Judge(LABELS, budget=2)
    assert j.judge("a") is True
    assert j.judge("a") is True
    assert j.used == 1 and j.remaining == 1


def test_budget_stops_exactly_at_limit():
    j = Judge(LABELS, budget=2)
    j.judge("a")
    j.judge("b")
    assert j.used == 2
    assert raises(BudgetExhausted, j.judge, "c")
    assert j.used == 2
    assert j.judge("a") is True  # already paid for, so still answered


def test_missing_label_raises_and_costs_nothing():
    j = Judge(LABELS, budget=2)
    assert raises(UnlabeledFiling, j.judge, "e")
    assert j.used == 0


def test_load_labels_reads_label_py_format():
    rows = [
        {"filing_id": "a", "error": "http_429"},
        {"filing_id": "a", "cost": 0.1, "labels": {
            "q1": {"answer": True, "quote": "x", "quote_verified": True}}},
        {"filing_id": "b", "cost": 0.1, "labels": {
            "q1": {"answer": False, "quote": "", "quote_verified": None}}},
        {"filing_id": "c", "error": "http_503"},
        # A yes with an unverified quote still counts as labeled.
        {"filing_id": "d", "cost": 0.1, "labels": {
            "q1": {"answer": True, "quote": "y", "quote_verified": False}}},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "labels.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        labels = load_labels(path, "q1")
        assert labels == {"a": True, "b": False, "d": True}  # latest row wins; c failed
        assert raises(ValueError, load_labels, path, "other_question")


def test_cost_tracker_prices_known_models_only():
    c = CostTracker()
    c.add(1_000_000, 0, "gemini-2.5-flash-lite")
    c.add(5, 5, "some-unpriced-model")
    assert abs(c.dollars - 0.10) < 1e-9
    assert c.calls == 2 and c.unpriced_calls == 1 and c.input_tokens == 1_000_005


def test_first_rank_per_filing_keeps_best_in_rank_order():
    ranks = first_rank_per_filing(hits(("b", 5), ("a", 2), ("b", 1)))
    assert list(ranks.items()) == [("b", 1), ("a", 2)]


def test_states_move_correctly_and_each_filing_has_one():
    led = Ledger(IDS)
    assert led.state("a") == "untouched"
    stats = led.observe(hits(("a", 1), ("b", 2), ("a", 3)), "q1", 0)
    assert stats["filings"] == 2 and stats["new_seen"] == 2
    assert led.state("a") == "seen_unjudged"
    led.record_judgment("a", True, "q1", 0, "search")
    led.record_judgment("b", False, "q1", 0, "search", reason="off topic")
    assert led.state("a") == "found" and led.state("b") == "rejected"
    assert led.found() == ["a"] and led.rejected() == ["b"]
    assert led.counts() == {"found": 1, "rejected": 1, "seen_unjudged": 0, "untouched": 3}
    assert sum(led.counts().values()) == len(IDS)


def test_observing_a_judged_filing_does_not_change_its_state():
    led = Ledger(IDS)
    led.observe(hits(("a", 9)), "q1", 0)
    led.record_judgment("a", True, "q1", 0, "search")
    stats = led.observe(hits(("a", 1)), "q2", 1)
    assert stats["known_found"] == 1 and stats["new_seen"] == 0
    assert led.state("a") == "found"


def test_cannot_judge_twice_or_use_unknown_filings():
    led = Ledger(IDS)
    led.record_judgment("a", True, "q1", 0, "search")
    assert raises(ValueError, led.record_judgment, "a", False, "q2", 1, "search")
    assert led.state("a") == "found"
    assert raises(KeyError, led.record_judgment, "zzz", True, "q1", 0, "search")
    assert raises(KeyError, led.observe, hits(("zzz", 1)), "q1", 0)
    assert raises(KeyError, led.state, "zzz")


def test_best_rank_never_worsens():
    led = Ledger(IDS)
    led.observe(hits(("a", 10)), "q1", 0)
    led.observe(hits(("a", 500)), "q2", 0)
    assert led.best_rank("a") == 10
    led.observe(hits(("a", 3)), "q3", 1)
    assert led.best_rank("a") == 3


def test_top_unjudged_skips_judged_and_dedupes():
    led = Ledger(IDS)
    h = hits(("a", 1), ("a", 2), ("b", 3), ("c", 4), ("d", 5))
    led.observe(h, "q1", 0)
    led.record_judgment("b", False, "q1", 0, "search")
    assert led.top_unjudged(h, n=2) == ["a", "c"]
    assert led.top_unjudged(h, n=50) == ["a", "c", "d"]


def test_pools_include_leftover_top_ranks_and_exclude_judged():
    led = Ledger(IDS)
    led.observe(hits(("a", 3), ("b", 60), ("c", 2000), ("d", 2001)), "q1", 0)
    assert led.seen_unjudged() == {"a": 3, "b": 60, "c": 2000, "d": 2001}
    assert led.near_miss() == {"b": 60, "c": 2000}
    assert led.untouched() == ["e"]
    led.record_judgment("b", False, "q1", 0, "search")
    assert "b" not in led.near_miss() and "b" not in led.seen_unjudged()


def test_sampled_is_a_flag_with_its_probability():
    led = Ledger(IDS)
    led.record_judgment("e", True, "sample", 2, "sample", draw_prob=0.01)
    assert led.state("e") == "found"
    row = led.judgment("e")
    assert row["sampled"] is True and row["draw_prob"] == 0.01
    assert row["best_rank"] is None


def test_replay_reproduces_state_and_log_order():
    with tempfile.TemporaryDirectory() as d:
        log_dir = os.path.join(d, "run")
        led = Ledger(IDS, log_dir=log_dir)
        led.observe(hits(("a", 4), ("b", 70)), "q1", 0)
        led.record_judgment("a", True, "q1", 0, "search")
        led.observe(hits(("b", 8), ("c", 900)), "q2", 1)
        led.record_judgment("b", False, "q2", 1, "amplify", reason="off topic")
        led.record_judgment("e", True, "s", 1, "sample", draw_prob=0.5)
        led.close()
        assert Ledger.replay(IDS, log_dir).snapshot() == led.snapshot()
        rows = [json.loads(line) for line in open(os.path.join(log_dir, "judgments.jsonl"))]
        assert [r["seq"] for r in rows] == [1, 2, 3]
        assert rows[1]["via"] == "amplify" and rows[1]["best_rank"] == 8
        # A second run must not append to, or overwrite, an existing log.
        assert raises(FileExistsError, Ledger, IDS, log_dir=log_dir)


for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn()
        print("passed:", name)
print("all tests passed")
