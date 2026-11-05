"""Per-run record of every filing: found, rejected, seen_unjudged or untouched.

Each filing is in exactly one state. Sampling is a flag on top of that, since a
sampled filing is also found or rejected.

The loop is: observe(hits) -> top_unjudged(hits) -> judge() -> record_judgment().

With log_dir set, writes judgments.jsonl (the order of judgments, which recall
curves read) and observations.jsonl. Ledger.replay() rebuilds the state from them.
"""
import json
import os

FOUND = "found"
REJECTED = "rejected"
SEEN_UNJUDGED = "seen_unjudged"
UNTOUCHED = "untouched"


class Ledger:
    def __init__(self, filing_ids, log_dir=None):
        self._state = {fid: UNTOUCHED for fid in filing_ids}
        self._best_rank = {}
        self._judgments = {}
        self._seq = 0
        self._judgment_log = self._observation_log = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._judgment_log = open(os.path.join(log_dir, "judgments.jsonl"), "x")
            self._observation_log = open(os.path.join(log_dir, "observations.jsonl"), "x")

    def _check_known(self, filing_id):
        if filing_id not in self._state:
            raise KeyError(f"{filing_id} is not in this collection")

    def observe(self, hits, query_id, generation):
        """Record a query's results. Returns the overlap counts for its filings."""
        ranks = first_rank_per_filing(hits)
        stats = {"filings": len(ranks), "new_seen": 0, "known_found": 0,
                 "known_rejected": 0, "already_seen": 0}
        for filing_id, rank in ranks.items():
            self._check_known(filing_id)
            state = self._state[filing_id]
            if state == UNTOUCHED:
                self._state[filing_id] = SEEN_UNJUDGED
                stats["new_seen"] += 1
            elif state == FOUND:
                stats["known_found"] += 1
            elif state == REJECTED:
                stats["known_rejected"] += 1
            else:
                stats["already_seen"] += 1
            if rank < self._best_rank.get(filing_id, float("inf")):
                self._best_rank[filing_id] = rank
        if self._observation_log:
            self._observation_log.write(json.dumps(
                {"query_id": query_id, "generation": generation, "ranks": ranks}) + "\n")
            self._observation_log.flush()
        return stats

    def top_unjudged(self, hits, n=50):
        """The first n distinct unjudged filings in these hits, best rank first."""
        out = []
        for filing_id in first_rank_per_filing(hits):
            self._check_known(filing_id)
            if self._state[filing_id] not in (FOUND, REJECTED):
                out.append(filing_id)
                if len(out) == n:
                    break
        return out

    def record_judgment(self, filing_id, relevant, query_id, generation, via,
                        reason=None, draw_prob=None):
        self._check_known(filing_id)
        if filing_id in self._judgments:
            raise ValueError(f"{filing_id} was already judged")
        self._state[filing_id] = FOUND if relevant else REJECTED
        self._seq += 1
        row = {"seq": self._seq, "filing_id": filing_id, "relevant": bool(relevant),
               "query_id": query_id, "generation": generation, "via": via,
               "best_rank": self._best_rank.get(filing_id),
               "sampled": draw_prob is not None}
        if reason is not None:
            row["reason"] = reason
        if draw_prob is not None:
            row["draw_prob"] = draw_prob
        self._judgments[filing_id] = row
        if self._judgment_log:
            self._judgment_log.write(json.dumps(row) + "\n")
            self._judgment_log.flush()

    def state(self, filing_id):
        self._check_known(filing_id)
        return self._state[filing_id]

    def judgment(self, filing_id):
        """How a judged filing was reached, or None if it has not been judged."""
        return self._judgments.get(filing_id)

    def best_rank(self, filing_id):
        return self._best_rank.get(filing_id)

    def _in_state(self, state):
        return sorted(fid for fid, s in self._state.items() if s == state)

    def found(self):
        return self._in_state(FOUND)

    def rejected(self):
        return self._in_state(REJECTED)

    def untouched(self):
        return self._in_state(UNTOUCHED)

    def seen_unjudged(self):
        """{filing_id: best rank}. Can include ranks under 51 if budget ran out."""
        return {fid: self._best_rank[fid] for fid in self._in_state(SEEN_UNJUDGED)}

    def near_miss(self, lo=51, hi=2000):
        return {fid: rank for fid, rank in self.seen_unjudged().items() if lo <= rank <= hi}

    def counts(self):
        counts = {FOUND: 0, REJECTED: 0, SEEN_UNJUDGED: 0, UNTOUCHED: 0}
        for state in self._state.values():
            counts[state] += 1
        return counts

    def snapshot(self):
        """Everything replay must reproduce."""
        return {"state": dict(self._state), "best_rank": dict(self._best_rank),
                "judgments": {fid: dict(row) for fid, row in self._judgments.items()}}

    def close(self):
        for log in (self._judgment_log, self._observation_log):
            if log:
                log.close()

    @classmethod
    def replay(cls, filing_ids, log_dir):
        """Rebuild a ledger from its logs, without writing to them."""
        ledger = cls(filing_ids)
        for line in open(os.path.join(log_dir, "observations.jsonl")):
            row = json.loads(line)
            hits = [(None, fid, rank) for fid, rank in row["ranks"].items()]
            ledger.observe(hits, row["query_id"], row["generation"])
        for line in open(os.path.join(log_dir, "judgments.jsonl")):
            row = json.loads(line)
            ledger._state[row["filing_id"]] = FOUND if row["relevant"] else REJECTED
            ledger._judgments[row["filing_id"]] = row
            ledger._seq = row["seq"]
        return ledger


def first_rank_per_filing(hits):
    """{filing_id: its best rank in these hits}, in rank order."""
    ranks = {}
    for _chunk_id, filing_id, rank in sorted(hits, key=lambda hit: hit[2]):
        ranks.setdefault(filing_id, rank)
    return ranks
