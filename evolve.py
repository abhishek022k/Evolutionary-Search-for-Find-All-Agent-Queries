"""Evolutionary search for "find all": a population of queries, bred each generation.

One generation:
  1. search   every query runs, we judge its top unjudged filings
  2. amplify  confirmed finds seed more search, capped separately
  3. score    fitness = NEW relevant filings per judgment spent
  4. select   keep the best query per branch (the elites)
  5. breed    the LLM writes replacements from the elites and the newest finds

Keeping an elite is not wasteful: top_unjudged skips anything already judged, so a
re-run elite moves deeper into its own ranking and its fitness decays as it exhausts
its neighbourhood. The population prunes dead directions without an explicit rule.

This module takes the engine as an argument and never imports search itself, so the
tests run with no index and no torch. numpy is used only for one dot product.
"""
import json
import math
import os
import random
import re
from collections import Counter

import numpy

from judge import BudgetExhausted, Judge
from ledger import Ledger, first_rank_per_filing

STOPWORDS = set("""
a all also an and any are as at be been being both business businesses but by can companies
company could did do does dont each few for from had has have he her here his how i if in
include included includes including is it its just may might more most must no not now of on
only operation operations or other our own result results s same she should so some such t
than that the their then there these they this those to too us very was we were what when
where which who whom why will with would year years you your
""".split())

# A hyphen or apostrophe must be followed by more letters, so "covid-" tokenizes as
# "covid". Trailing punctuation inside a phrase makes exact-match queries miss.
TOKEN = re.compile(r"[a-z]+(?:['\-][a-z]+)*")


def lift(found_count, n_found, bg_df, bg_docs):
    """How much more common this wording is among the filings we found than in the
    collection at large. 1.0 means no better than chance.

    Counting recurrence alone picks boilerplate, because "financial condition"
    recurs in every filing we found and in every filing we did not. Weighting a
    count by rarity does not fix it either: a phrase in 26 of 31 found filings
    still outscores a rare one. A ratio of rates does, and needs no magic cap.
    """
    return (found_count / n_found) / ((bg_df + 1) / bg_docs)


def phrases_in(words, sizes=(2, 3, 4)):
    """The distinct 2 to 4 word phrases in one document, ignoring ones that start
    or end on a stopword. A set, so a phrase counts once per document."""
    out = set()
    for size in sizes:
        for i in range(len(words) - size + 1):
            gram = words[i:i + size]
            if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                continue
            out.add(" ".join(gram))
    return out


# --------------------------------------------------------------------------- queries


class Query:
    __slots__ = ("query_id", "branch", "text", "born", "origin", "parents", "rationale")

    def __init__(self, query_id, branch, text, born, origin, parents=(), rationale=""):
        self.query_id = query_id
        self.branch = branch
        self.text = text
        self.born = born
        self.origin = origin
        self.parents = list(parents)
        self.rationale = rationale

    def as_dict(self):
        return {"query_id": self.query_id, "branch": self.branch, "query": self.text,
                "born": self.born, "origin": self.origin, "parents": self.parents,
                "rationale": self.rationale}


def best_chunk_per_filing(hits):
    """{filing_id: the chunk id that gave it its best rank}. Seeds vector amplify."""
    out = {}
    for chunk_id, filing_id, _rank in sorted(hits, key=lambda h: h[2]):
        out.setdefault(filing_id, chunk_id)
    return out


# ------------------------------------------------------------------------- the run


class EvolutionRun:
    def __init__(self, engine, labels, filing_ids, cfg, breeder, log_dir, budget,
                 question_text, seed=0, verbose=True):
        self.engine = engine
        self.cfg = cfg
        self.breeder = breeder
        self.question = question_text
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.filing_ids = list(filing_ids)

        self.judge = Judge(labels, budget)
        self.ledger = Ledger(self.filing_ids, log_dir=log_dir)
        self.log_dir = log_dir
        self.exhausted = False

        self.best_chunk = {}          # filing_id -> chunk id, for amplify seeding
        self._qvec = None             # the embedded question, computed once
        self._stub = None             # lazy StubBreeder, tops up a short generation
        self._bg_df = None            # background frequencies, computed once
        self._bg_phrase_df = None
        self._bg_docs = 0
        self.generations = []
        self._query_log = open(os.path.join(log_dir, "queries.jsonl"), "x")
        self._generation_log = open(os.path.join(log_dir, "generations.jsonl"), "x")

    # -- small helpers ------------------------------------------------------

    def _say(self, message):
        if self.verbose:
            print(message, flush=True)

    def _observe(self, hits, query_id, generation, record=True):
        """record=False for amplify's vector searches: they start from a filing we
        already know is relevant, so their ranks say nothing about what a written
        query reached. Letting them set best_rank would make unreached filings look
        nearly reached, which is what near_miss() pools are built from."""
        if record:
            self.ledger.observe(hits, query_id, generation)
        for filing_id, chunk_id in best_chunk_per_filing(hits).items():
            self.best_chunk.setdefault(filing_id, chunk_id)

    def _judge_all(self, candidates, query_id, generation, via, result):
        """Judge each candidate, updating result in place. Raises BudgetExhausted."""
        for filing_id in candidates:
            relevant = self.judge.judge(filing_id)
            self.ledger.record_judgment(filing_id, relevant, query_id, generation, via)
            result["judgments_spent"] += 1
            if relevant:
                result["new_found"] += 1
                result["found_ids"].append(filing_id)

    def _question_vector(self):
        if self._qvec is None:
            self._qvec = numpy.asarray(
                self.engine.embedder.encode_queries([self.question])[0], dtype="float32")
        return self._qvec

    def _filing_text(self, filing_id):
        """The one chunk of this filing closest in meaning to the question.

        Not the chunk the query matched. That is circular: it hands back the words we
        already searched with, which is why mining it produced "risks and uncertainties
        described". Measured on generation 0, the chunk holding the labeller's evidence
        quote was the top-ranked chunk for 1 filing out of 31.

        shortcut: one chunk, so evidence spread across chunks is missed. Revisit when a
        real LLM judge reports which snippet it approved.
        """
        chunk_ids = self.engine.chunks_of(filing_id)
        if not chunk_ids:
            return ""
        sims = numpy.asarray(self.engine.get_vectors(chunk_ids)) @ self._question_vector()
        return self.engine.get_text(chunk_ids[int(numpy.argmax(sims))])

    # -- step 1: search -----------------------------------------------------

    def run_population(self, population, generation, results):
        """Each query searches and judges its top unjudged filings.

        Fills `results` in place, so a BudgetExhausted part-way through still leaves
        the caller with everything spent up to that point.
        """
        depth = self.cfg["judge_depth"]
        # Rotate the order so no query is systematically penalised for going last:
        # whoever is processed first spends the judgment on a shared filing.
        shift = generation % max(len(population), 1)
        order = population[shift:] + population[:shift]
        for query in order:
            result = {"query_id": query.query_id, "branch": query.branch,
                      "query": query.text, "generation": generation,
                      "origin": query.origin, "parents": query.parents,
                      "judgments_spent": 0, "new_found": 0, "found_ids": [],
                      "hits": 0, "filings_reached": 0}
            results[query.query_id] = result
            hits = self.engine.search(query.branch, query.text, self.cfg["search_k"])
            result["hits"] = len(hits)
            result["filings_reached"] = len(first_rank_per_filing(hits))
            if not hits:
                continue
            self._observe(hits, query.query_id, generation)
            candidates = self.ledger.top_unjudged(hits, depth)
            self._judge_all(candidates, query.query_id, generation, "search", result)
        return results

    @staticmethod
    def found_in(results):
        return [f for r in results.values() for f in r["found_ids"]]

    # -- step 2: amplify ----------------------------------------------------

    def _background_df(self, sample_size=300):
        """Document frequencies over a random sample, so 'distinctive' means something.

        Phrases need this as much as words do: wording like "financial condition"
        recurs in every filing we find and in every filing we do not.
        """
        if self._bg_df is not None:
            return self._bg_df, self._bg_phrase_df, self._bg_docs
        sample = self.rng.sample(self.filing_ids, min(sample_size, len(self.filing_ids)))
        df, phrase_df = Counter(), Counter()
        docs = 0
        for filing_id in sample:
            text = self._filing_text(filing_id)   # same rule as the foreground
            if not text:
                continue
            docs += 1
            words = TOKEN.findall(text.lower())
            df.update(set(words) - STOPWORDS)
            phrase_df.update(phrases_in(words))
        self._bg_df, self._bg_phrase_df, self._bg_docs = df, phrase_df, max(docs, 1)
        return self._bg_df, self._bg_phrase_df, self._bg_docs

    def word_queries(self, found_ids, max_terms=8):
        """Distinctive terms and repeated phrases from confirmed finds -> new queries."""
        texts = [self._filing_text(f) for f in found_ids]
        texts = [t.lower() for t in texts if t]
        if len(texts) < 2:
            return []
        fg = Counter()
        for text in texts:
            fg.update(set(TOKEN.findall(text)) - STOPWORDS)
        bg_df, bg_phrase_df, bg_docs = self._background_df()
        # With enough found filings, insist on three before believing a word matters,
        # so two coincidentally similar filings cannot set the next query.
        floor = 3 if len(texts) >= 10 else 2
        scored = {term: lift(count, len(texts), bg_df.get(term, 0), bg_docs)
                  for term, count in fg.items() if count >= floor}
        terms = [t for t, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))][:max_terms]

        # Phrases that recur across found filings AND are rare elsewhere are what the
        # exact branch is good at. Without the background term, boilerplate wins:
        # "financial condition" recurs in every filing in the corpus.
        phrase_docs = Counter()
        for text in texts:
            phrase_docs.update(phrases_in(TOKEN.findall(text)))
        phrase_scored = {p: lift(c, len(texts), bg_phrase_df.get(p, 0), bg_docs)
                         for p, c in phrase_docs.items() if c >= floor}
        phrases = [p for p, _ in sorted(phrase_scored.items(),
                                        key=lambda kv: (-kv[1], kv[0]))][:2]

        out = []
        if terms:
            out.append(("bm25", " ".join(terms)))
        for phrase in phrases:
            out.append(("exact", phrase))
        return out

    @staticmethod
    def new_amplify_result(generation, n_finds):
        return {"query_id": f"amp-g{generation}", "branch": "amplify",
                "query": f"amplify from {n_finds} finds", "generation": generation,
                "origin": "amplify", "parents": [], "judgments_spent": 0,
                "new_found": 0, "found_ids": [], "hits": 0, "filings_reached": 0,
                "by_vector_seeds": 0, "word_queries": []}

    def amplify(self, found_ids, generation, result):
        """Seed more search from confirmed finds. Capped, never joins the population.

        Fills `result` in place for the same reason run_population does.
        """
        cap = self.cfg["amplify_cap"]
        if not found_ids or cap <= 0:
            return result

        merged = {}

        def absorb(hits):
            for filing_id, rank in first_rank_per_filing(hits).items():
                if rank < merged.get(filing_id, math.inf):
                    merged[filing_id] = rank

        # (a) more like this, by meaning: seed from each find's best chunk
        amp_k = max(self.cfg.get("amplify_k", 200), cap)
        for filing_id in found_ids:
            chunk_id = self.best_chunk.get(filing_id)
            if chunk_id is None:
                chunks = self.engine.chunks_of(filing_id)
                if not chunks:
                    continue
                chunk_id = chunks[0]
            vector = self.engine.get_vectors([chunk_id])[0]
            hits = self.engine.search_by_vector(vector, k=amp_k)
            result["by_vector_seeds"] += 1
            result["hits"] += len(hits)
            self._observe(hits, result["query_id"] + "-vec", generation, record=False)
            absorb(hits)

        # (b) more like this, by words: distinctive terms and repeated phrases
        for branch, text in self.word_queries(found_ids):
            hits = self.engine.search(branch, text, amp_k)
            result["word_queries"].append({"branch": branch, "query": text,
                                           "hits": len(hits)})
            if not hits:
                continue
            result["hits"] += len(hits)
            self._observe(hits, f"{result['query_id']}-{branch}", generation)
            absorb(hits)

        result["filings_reached"] = len(merged)
        order = [f for f, _ in sorted(merged.items(), key=lambda kv: (kv[1], kv[0]))]
        candidates = self.ledger.top_unjudged(
            [(None, f, i + 1) for i, f in enumerate(order)], cap)
        self._judge_all(candidates, result["query_id"], generation, "amplify", result)
        return result

    # -- steps 3 and 4: score and select ------------------------------------

    @staticmethod
    def fitness(result):
        """New relevant filings per judgment. Re-finding known filings is worth nothing."""
        spent = result["judgments_spent"]
        return (result["new_found"] / spent) if spent else 0.0

    def select_elites(self, population, results):
        """Best query per branch. Deterministic: fitness, then new finds, then id."""
        keep_per_branch = self.cfg.get("elites_per_branch", 1)
        by_branch = {}
        for query in population:
            by_branch.setdefault(query.branch, []).append(query)
        elites = []
        for branch in sorted(by_branch):
            # A query that spent nothing has no unjudged results left. It is exhausted,
            # and since every exhausted query scores 0 it would win its branch on the
            # tiebreak and hold the slot forever. Drop it and let the breeder have the
            # slot instead: run() sizes the next batch from how many elites survive.
            live = [q for q in by_branch[branch]
                    if results[q.query_id]["judgments_spent"] > 0]
            ranked = sorted(
                live,
                key=lambda q: (-self.fitness(results[q.query_id]),
                               -results[q.query_id]["new_found"], q.query_id))
            elites.extend(ranked[:keep_per_branch])
        elite_ids = {q.query_id for q in elites}
        dropped = [q for q in population if q.query_id not in elite_ids]
        return elites, dropped

    # -- step 5: breed ------------------------------------------------------

    # Wide enough for a whole chunk: the relevant wording is rarely near the start.
    def _excerpts(self, filing_ids, limit, width=2000):
        out = []
        for filing_id in filing_ids[:limit]:
            text = " ".join(self._filing_text(filing_id).split())
            if text:
                out.append(text[:width])
        return out

    def build_context(self, elites, dropped, results, found_this_gen):
        def summarize(queries):
            return [{"branch": q.branch, "query": q.text,
                     "new_found": results[q.query_id]["new_found"],
                     "judgments_spent": results[q.query_id]["judgments_spent"]}
                    for q in queries]

        rejected = [f for f in self.ledger.rejected()]
        self.rng.shuffle(rejected)
        return {
            "question": self.question,
            "elites": summarize(elites),
            "dropped": summarize(dropped),
            "found_excerpts": self._excerpts(found_this_gen or self.ledger.found(), 6),
            "rejected_excerpts": self._excerpts(rejected, 3),
        }

    # -- the driver ---------------------------------------------------------

    def run(self, gen0_queries, max_generations=None):
        population = [Query(q["query_id"], q["branch"], q["query"], 0, "gen0")
                      for q in gen0_queries]
        size = len(population)
        max_generations = max_generations or self.cfg.get("max_generations", 20)
        counter = 0

        for generation in range(max_generations):
            started = self.judge.used
            results = {}
            amp = self.new_amplify_result(generation, 0)
            try:
                self.run_population(population, generation, results)
                amp["query"] = f"amplify from {len(self.found_in(results))} finds"
                self.amplify(self.found_in(results), generation, amp)
            except BudgetExhausted:
                self.exhausted = True
            found_now = self.found_in(results) + amp["found_ids"]

            for result in results.values():
                result["fitness"] = self.fitness(result)
                self._query_log.write(json.dumps(result) + "\n")
            amp["fitness"] = self.fitness(amp)
            self._query_log.write(json.dumps(amp) + "\n")
            self._query_log.flush()

            spent = self.judge.used - started
            found_total = len(self.ledger.found())
            summary = {
                "generation": generation,
                "population": [q.as_dict() for q in population],
                "judgments_this_generation": spent,
                "judgments_cumulative": self.judge.used,
                "found_cumulative": found_total,
                "new_found_this_generation": sum(r["new_found"] for r in results.values())
                + amp["new_found"],
                "amplify": {k: amp[k] for k in
                            ("judgments_spent", "new_found", "by_vector_seeds",
                             "word_queries")},
                "exhausted": self.exhausted,
            }
            self._say(f"  gen {generation}: {spent} judgments, "
                      f"+{summary['new_found_this_generation']} new, "
                      f"{found_total} found total")

            if self.exhausted:
                summary["elites"] = []
                self._generation_log.write(json.dumps(summary) + "\n")
                self._generation_log.flush()
                self.generations.append(summary)
                break

            elites, dropped = self.select_elites(population, results)
            summary["elites"] = [q.query_id for q in elites]
            summary["dropped"] = [q.query_id for q in dropped]
            self._generation_log.write(json.dumps(summary) + "\n")
            self._generation_log.flush()
            self.generations.append(summary)

            n_new = size - len(elites)
            context = self.build_context(elites, dropped, results, found_now)
            children = self.breeder.breed(context, n_new) if n_new else []
            if len(children) < n_new:  # keep the population size stable
                children += self._filler().breed(context, n_new - len(children))
            new_queries = []
            for child in children[:n_new]:
                counter += 1
                new_queries.append(Query(
                    f"g{generation + 1}q{counter}", child["branch"], child["query"],
                    generation + 1, "bred", [q.query_id for q in elites],
                    child.get("rationale", "")))
            population = elites + new_queries

        return self

    def _filler(self):
        """Tops the population back up when a breeder returns fewer queries than asked."""
        if self._stub is None:
            from breeder import StubBreeder
            self._stub = StubBreeder(seed=self.cfg.get("seed", 0))
        return self._stub

    def finish(self):
        self.ledger.close()
        self._query_log.close()
        self._generation_log.close()

