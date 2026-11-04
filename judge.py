"""Simulated judge: ground-truth labels behind a budget. One new filing = one unit.

Labels stay private to Judge, so search code can only learn one by paying for it.
Run or evaluation code that needs the totals calls load_labels() directly.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from label import PRICES, cost_of, load_done  # noqa: E402


class BudgetExhausted(Exception):
    """The run has spent all its judgments."""


class UnlabeledFiling(Exception):
    """No successful label exists. Never treat this as "not relevant"."""


def load_labels(path, question_id):
    """Read one of label.py's output files into {filing_id: bool}.

    Failed filings are left out, so judging one raises UnlabeledFiling.
    """
    labels = {}
    for filing_id, row in load_done(path).items():
        if "labels" not in row:
            continue
        if question_id not in row["labels"]:
            raise ValueError(f"{path}: {filing_id} has no label for {question_id!r}; "
                             f"it has {sorted(row['labels'])}")
        labels[filing_id] = bool(row["labels"][question_id]["answer"])
    return labels


class Judge:
    def __init__(self, labels, budget):
        self._labels = labels
        self._answers = {}
        self.budget = budget
        self.used = 0

    @property
    def remaining(self):
        return self.budget - self.used

    def judge(self, filing_id):
        """The agreed judge(filing_id). A repeat call is cached and costs nothing."""
        if filing_id in self._answers:
            return self._answers[filing_id]
        if filing_id not in self._labels:
            raise UnlabeledFiling(filing_id)
        if self.used >= self.budget:
            raise BudgetExhausted(f"all {self.budget} judgments spent")
        self.used += 1
        self._answers[filing_id] = self._labels[filing_id]
        return self._answers[filing_id]


class CostTracker:
    """LLM breeding costs, reported separately from judgments."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.dollars = 0.0
        self.unpriced_calls = 0

    def add(self, input_tokens, output_tokens, model):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if model in PRICES:
            self.dollars += cost_of(
                {"input_tokens": input_tokens, "output_tokens": output_tokens}, model)
        else:
            self.unpriced_calls += 1

    def summary(self):
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "dollars": self.dollars,
                "unpriced_calls": self.unpriced_calls}
