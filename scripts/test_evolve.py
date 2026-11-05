"""Offline tests for the evolution loop. No index, no torch, no API key, no spend.

A FakeEngine stands in for SearchEngine, so these run before the real index exists.

Run from the project root:  .venv/bin/python scripts/test_evolve.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from breeder import BRANCHES, StubBreeder, _clean, build_prompt  # noqa: E402
from evolve import EvolutionRun, best_chunk_per_filing  # noqa: E402
from ledger import Ledger  # noqa: E402

N_FILINGS = 120
WORDS = ["home", "demand", "consumer", "outdoor", "bicycle", "pandemic", "retail",
         "supply", "revenue", "segment", "hobby", "lockdown"]


class FakeEngine:
    """Deterministic stand-in. Relevant filings cluster in index space so that
    vector amplify has something real to find."""

    def __init__(self, n=N_FILINGS):
        self.filing_ids = [f"F{i:04d}" for i in range(n)]
        self.cfg = {"search_k": 200, "index_type": "flat"}
        self._text, self._chunks, self._position = {}, {}, {}
        position = 0
        for i, filing_id in enumerate(self.filing_ids):
            chunk_ids = []
            for section in ("item_1a", "item_7"):
                chunk_id = f"{filing_id}#{section}#0000"
                # Word choice is a function of the index, so neighbours share wording.
                words = [WORDS[(i + j) % len(WORDS)] for j in range(5)]
                self._text[chunk_id] = " ".join(words * 12)
                self._position[chunk_id] = position
                chunk_ids.append(chunk_id)
                position += 1
            self._chunks[filing_id] = chunk_ids

    # -- the SearchEngine surface evolve.py uses --------------------------
    def search(self, branch, query, k=None):
        k = k or 200
        terms = set(query.lower().split())
        scored = []
        for chunk_id, text in self._text.items():
            body = set(text.split())
            if branch == "exact":
                hit = query.lower().strip() in text
                score = 1 if hit else 0
            else:
                score = len(terms & body)
            if score:
                scored.append((-score, self._position[chunk_id], chunk_id))
        scored.sort()
        return [(cid, cid.split("#")[0], rank)
                for rank, (_s, _p, cid) in enumerate(scored[:k], start=1)]

    def search_by_vector(self, vector, k=None):
        k = k or 200
        centre = int(vector[0])
        order = sorted(self._position.items(), key=lambda kv: abs(kv[1] - centre))
        return [(cid, cid.split("#")[0], rank)
                for rank, (cid, _p) in enumerate(order[:k], start=1)]

    def get_vectors(self, chunk_ids):
        return [[float(self._position[c])] for c in chunk_ids]

    def get_text(self, chunk_id):
        return self._text[chunk_id]

    def chunks_of(self, filing_id):
        return list(self._chunks.get(filing_id, []))


def make_setup(depth=10, amplify_cap=10, population=6):
    engine = FakeEngine()
    # Every third filing is relevant, so finds are plentiful enough to drive amplify.
    labels = {fid: (i % 3 == 0) for i, fid in enumerate(engine.filing_ids)}
    cfg = {"search_k": 200, "judge_depth": depth, "amplify_cap": amplify_cap,
           "amplify_k": 50, "elites_per_branch": 1, "seed": 7, "max_generations": 6}
    gen0 = [{"query_id": f"g0q{i}", "branch": BRANCHES[i % len(BRANCHES)],
             "query": WORDS[i % len(WORDS)] + " demand"} for i in range(population)]
    return engine, labels, cfg, gen0


def run_once(tmp, budget=120, **kw):
    engine, labels, cfg, gen0 = make_setup(**kw)
    run = EvolutionRun(engine, labels, engine.filing_ids, cfg, StubBreeder(seed=1),
                       tmp, budget, "test question", seed=7, verbose=False)
    try:
        run.run(gen0)
    finally:
        run.finish()
    return run, labels, gen0


def fresh_dir():
    path = tempfile.mkdtemp(prefix="evolve_test_")
    shutil.rmtree(path)
    return path


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    return bool(condition)


def main():
    ok = True
    print("evolution loop, offline")

    # 1. budget is never exceeded, and every judgment is logged exactly once
    tmp = fresh_dir()
    run, labels, gen0 = run_once(tmp, budget=120)
    rows = [json.loads(l) for l in open(os.path.join(tmp, "judgments.jsonl"))]
    ok &= check("budget respected", run.judge.used <= 120,
                f"used {run.judge.used}/120")
    ok &= check("one log row per judgment", len(rows) == run.judge.used,
                f"{len(rows)} rows vs {run.judge.used} judgments")
    ok &= check("no filing judged twice",
                len({r["filing_id"] for r in rows}) == len(rows))

    # 2. recall accounting matches the labels
    found = set(run.ledger.found())
    ok &= check("every found filing is really relevant",
                all(labels[f] for f in found), f"{len(found)} found")
    ok &= check("rejected filings are really irrelevant",
                not any(labels[f] for f in run.ledger.rejected()))

    # 3. elite selection: one per branch, deterministic
    gens = [json.loads(l) for l in open(os.path.join(tmp, "generations.jsonl"))]
    sized = [g for g in gens if g.get("elites")]
    branch_of = {q["query_id"]: q["branch"] for g in gens for q in g["population"]}
    one_per_branch = all(
        len(g["elites"]) == len({branch_of[e] for e in g["elites"]}) for g in sized)
    ok &= check("elites are one per branch", one_per_branch)
    ok &= check("population size stays constant",
                len({len(g["population"]) for g in gens}) == 1,
                str(sorted({len(g["population"]) for g in gens})))

    # 4. amplify respects its cap
    amp = [g["amplify"] for g in gens if g.get("amplify")]
    ok &= check("amplify within cap",
                all(a["judgments_spent"] <= 10 for a in amp),
                f"max {max((a['judgments_spent'] for a in amp), default=0)}")
    ok &= check("amplify actually ran", any(a["by_vector_seeds"] > 0 for a in amp))

    # 5. logs replay to the same state
    replayed = Ledger.replay([f"F{i:04d}" for i in range(N_FILINGS)], tmp)
    ok &= check("replay reproduces the found set", set(replayed.found()) == found)

    # 6. same seed, same run
    tmp2 = fresh_dir()
    run2, _, _ = run_once(tmp2, budget=120)
    ok &= check("deterministic under a fixed seed",
                set(run2.ledger.found()) == found and run2.judge.used == run.judge.used)

    # 7. a tiny budget stops cleanly mid-generation
    tmp3 = fresh_dir()
    run3, _, _ = run_once(tmp3, budget=7)
    ok &= check("tiny budget stops cleanly",
                run3.judge.used == 7 and run3.exhausted, f"used {run3.judge.used}")

    # 8. breeder plumbing
    print("breeder")
    stub = StubBreeder(seed=1)
    context = {"question": "q", "elites": [{"branch": "bm25", "query": "home demand",
                                            "new_found": 2, "judgments_spent": 10}],
               "dropped": [], "found_excerpts": ["some text"], "rejected_excerpts": []}
    out = stub.breed(context, 3)
    ok &= check("stub returns the requested count", len(out) == 3)
    ok &= check("stub uses valid branches", all(q["branch"] in BRANCHES for q in out))
    ok &= check("malformed queries are dropped",
                _clean([{"branch": "nope", "query": "x"}, {"branch": "bm25", "query": ""},
                        {"branch": "bm25", "query": "ok"}], 5) ==
                [{"branch": "bm25", "query": "ok", "rationale": ""}])
    prompt = build_prompt(context, 3)
    ok &= check("prompt carries grep coaching", "SHORT IDIOMATIC" in prompt)
    ok &= check("prompt names the count", "exactly 3" in prompt)

    # 9. helper
    hits = [("c2", "F2", 5), ("c1", "F1", 1), ("c3", "F1", 9)]
    ok &= check("best_chunk_per_filing picks the best rank",
                best_chunk_per_filing(hits) == {"F1": "c1", "F2": "c2"})

    for path in (tmp, tmp2, tmp3):
        shutil.rmtree(path, ignore_errors=True)
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
