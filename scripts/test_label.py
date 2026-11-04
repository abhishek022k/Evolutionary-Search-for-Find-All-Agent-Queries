"""Tests label.py with a fake AI. Makes no API calls and costs nothing.

Run from the project root:  .venv/bin/python scripts/test_label.py
"""
import asyncio
import json
import os
import sys
import tempfile
import types

from google.genai import errors as gerrors
from google.genai import types as gtypes

sys.path.insert(0, os.path.dirname(__file__))
import label  # noqa: E402

FILINGS = [
    {"filing_id": f"{i}_2020", "cik": str(i),
     "item_1a": "We sell   bicycles.\nSales of bicycles surged during lockdowns.",
     "item_7": "Revenue grew."}
    for i in range(1, 7)
]
QIDS = ["stay_home_boom", "panic_buying"]


class FakeMessages:
    def __init__(self, script):
        self.script = script
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"]["effort"] == "low"
        schema = kwargs["output_config"]["format"]["schema"]
        assert schema["required"] == QIDS
        text, stop = self.script(self.calls)
        return types.SimpleNamespace(
            usage=types.SimpleNamespace(input_tokens=30000, output_tokens=100),
            stop_reason=stop,
            content=[types.SimpleNamespace(type="text", text=text)])


def answer(quote):
    return json.dumps({
        "stay_home_boom": {"answer": bool(quote), "quote": quote},
        "panic_buying": {"answer": False, "quote": ""},
    })


def args(**over):
    base = dict(questions=QIDS, model="claude-haiku-5-5", concurrency=2,
                ramp_seconds=0, max_spend=100.0)
    base.update(over)
    return types.SimpleNamespace(**base)


def rows(path):
    return [json.loads(line) for line in open(path)]


def test_quote_check():
    f = FILINGS[0]
    assert label.quote_found("sales of  BICYCLES surged", f)  # spacing and case ignored
    assert not label.quote_found("we invented this sentence", f)
    assert not label.quote_found("", f)


def test_cost_uses_long_prompt_rate():
    short = label.cost_of({"input_tokens": 100_000, "output_tokens": 0}, "claude-haiku-5-5")
    long = label.cost_of({"input_tokens": 100_001, "output_tokens": 0}, "claude-haiku-5-5")
    assert abs(short - 0.01) < 1e-9
    assert abs(long - 0.0500005) < 1e-9


def test_prompt_keeps_whole_filing_and_both_sections():
    prompt = label.build_prompt(FILINGS[0], QIDS)
    assert FILINGS[0]["item_1a"] in prompt and FILINGS[0]["item_7"] in prompt


def test_run_labels_flags_bad_quote_and_resumes():
    def script(call):
        if call == 1:
            return answer("an invented quote"), "end_turn"
        if call == 2:
            return "not json", "end_turn"
        if call == 3:
            return "", "max_tokens"
        return answer("Sales of bicycles surged during lockdowns."), "end_turn"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.jsonl")
        fake = FakeMessages(script)
        client = types.SimpleNamespace(messages=fake)
        asyncio.run(label.run(args(concurrency=1), client, FILINGS, path))
        first = rows(path)
        assert len(first) == 6
        assert first[0]["labels"]["stay_home_boom"]["quote_verified"] is False
        assert first[1]["error"] == "bad_json"
        assert first[2]["error"] == "stop_reason:max_tokens"
        assert first[3]["labels"]["stay_home_boom"]["quote_verified"] is True

        # A rerun retries only the two failures.
        before = fake.calls
        asyncio.run(label.run(args(concurrency=1), client, FILINGS, path))
        assert fake.calls - before == 2
        done = label.load_done(path)
        assert all("labels" in done[f["filing_id"]] for f in FILINGS)


def test_spending_limit_stops_early():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.jsonl")
        fake = FakeMessages(lambda c: (answer(""), "end_turn"))
        client = types.SimpleNamespace(messages=fake)
        # Each fake call costs about $0.00305, so a $0.005 limit stops after two.
        asyncio.run(label.run(args(concurrency=1, max_spend=0.005), client, FILINGS, path))
        assert fake.calls == 2


class FakeGeminiModels:
    def __init__(self, fail_first=0, finish=None, text=None):
        self.calls = 0
        self.fail_first = fail_first
        self.finish = finish or gtypes.FinishReason.STOP
        self.text = text or answer("Sales of bicycles surged during lockdowns.")

    async def generate_content(self, model, contents, config):
        self.calls += 1
        if model.startswith("gemini-3"):
            assert config.thinking_config.thinking_level == gtypes.ThinkingLevel.MINIMAL
        else:
            assert config.thinking_config is None
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema["required"] == QIDS
        assert FILINGS[0]["item_7"] in contents
        if self.calls <= self.fail_first:
            raise gerrors.ClientError(
                429, {"error": {"message": "slow down", "status": "RESOURCE_EXHAUSTED"}})
        return types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=30000, candidates_token_count=100,
                thoughts_token_count=20),
            candidates=[types.SimpleNamespace(finish_reason=self.finish)],
            text=self.text)


def gemini_client(models):
    return types.SimpleNamespace(aio=types.SimpleNamespace(models=models))


def label_gemini(models, model="gemini-3.6-flash"):
    async def no_sleep(_):
        return None

    real_sleep = label.asyncio.sleep
    label.asyncio.sleep = no_sleep
    try:
        return asyncio.run(label.label_one(
            gemini_client(models), FILINGS[0], QIDS, model))
    finally:
        label.asyncio.sleep = real_sleep


def test_flash_lite_sends_no_thinking_setting_and_uses_its_price():
    row = label_gemini(FakeGeminiModels(), model="gemini-2.5-flash-lite")
    assert "labels" in row
    assert abs(row["cost"] - (30000 * 0.10 + 120 * 0.40) / 1e6) < 1e-9


def test_total_budget_counts_other_runs():
    real_glob, real_budget = label.LABELS_GLOB, label.CONFIG["total_budget"]
    with tempfile.TemporaryDirectory() as d:
        label.LABELS_GLOB = os.path.join(d, "labels_*.jsonl")
        label.CONFIG["total_budget"] = 0.01
        try:
            with open(os.path.join(d, "labels_old.jsonl"), "w") as f:
                f.write(json.dumps({"filing_id": "x", "cost": 0.004}) + "\n")
            path = os.path.join(d, "labels_new.jsonl")
            fake = FakeMessages(lambda c: (answer(""), "end_turn"))
            client = types.SimpleNamespace(messages=fake)
            # $0.006 is left. At about $0.00305 a call, the second call reaches it.
            asyncio.run(label.run(args(concurrency=1), client, FILINGS, path))
            assert fake.calls == 2

            # Already at the limit: a rerun sends nothing.
            label.CONFIG["total_budget"] = 0.004
            asyncio.run(label.run(args(concurrency=1), client, FILINGS, path))
            assert fake.calls == 2
        finally:
            label.LABELS_GLOB, label.CONFIG["total_budget"] = real_glob, real_budget


def test_gemini_success_counts_thinking_cost():
    row = label_gemini(FakeGeminiModels())
    assert row["labels"]["stay_home_boom"]["quote_verified"] is True
    assert row["usage"]["output_tokens"] == 120 and row["usage"]["thought_tokens"] == 20
    assert abs(row["cost"] - (30000 * 0.75 + 120 * 3.75) / 1e6) < 1e-9


def test_gemini_retries_rate_limits_then_gives_up():
    models = FakeGeminiModels(fail_first=2)
    assert "labels" in label_gemini(models) and models.calls == 3
    models = FakeGeminiModels(fail_first=99)
    row = label_gemini(models)
    assert row["error"] == "http_429" and models.calls == label.GEMINI_ATTEMPTS


def test_gemini_cut_off_answer_is_an_error():
    row = label_gemini(FakeGeminiModels(finish=gtypes.FinishReason.MAX_TOKENS))
    assert row["error"].startswith("stop_reason:") and "labels" not in row


def test_recheck_selects_only_yes_filings():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "labels.jsonl")
        rows_out = [
            {"filing_id": "1_2020", "labels": {"q": {"answer": True}}},
            {"filing_id": "2_2020", "labels": {"q": {"answer": False}}},
            {"filing_id": "3_2020", "error": "http_429"},
            {"filing_id": "4_2020", "error": "http_429"},
            {"filing_id": "4_2020", "labels": {"q": {"answer": True}}},
        ]
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows_out)
        assert label.yes_filing_ids(path, "q") == {"1_2020", "4_2020"}


def test_stops_after_repeated_failures():
    filings = [dict(FILINGS[0], filing_id=f"{i}_2020") for i in range(30)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.jsonl")
        fake = FakeMessages(lambda c: ("not json", "end_turn"))
        client = types.SimpleNamespace(messages=fake)
        asyncio.run(label.run(args(concurrency=1), client, filings, path))
        assert fake.calls == label.MAX_FAILURES_IN_A_ROW


for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn()
        print("passed:", name)
print("all tests passed")
