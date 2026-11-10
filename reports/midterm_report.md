# Evolutionary Search for "Find All" Agent Queries

**Midterm Report — CS 839, Next Generation Data Systems**

---

## 1. The problem

Most retrieval systems are built for *find me one*. A user asks a question, the system returns a ranked list, and the user reads until satisfied. Precision at the top of the list is what matters, and nobody minds much if a relevant document sits at rank 400.

A different class of question is *find me all of them*. Which of our 3,000 portfolio companies disclosed a particular risk? Which contracts contain a clause with this effect? Which patients in the registry match these criteria? Here a missed document is not a minor inconvenience — it is the failure mode. Recall is the metric, and the user cannot tell from the output whether the answer is complete.

Today these questions get handled at two extremes. At one end, an agent runs a few searches and returns what it found; the user accepts it because they have no way to check. At the other end — e-discovery, regulatory review — organisations pay humans or LLMs to read every document in the collection, at a cost that scales linearly with collection size and rules the approach out for most purposes.

Our project targets the middle: **how do you get high recall at a fraction of the cost of exhaustive review?**

The unit of cost throughout this work is a **judgment** — one decision about whether one document is relevant. In deployment this is an LLM reading a document and answering yes or no, and it dominates the cost of the whole system; retrieval, in comparison, is nearly free. So the experimental question is sharp: *given a fixed number of judgments, which search procedure finds the most relevant documents?*

## 2. How we got here: from the proposal to the current design

The design changed twice in response to the professor's feedback, and both changes are the substance of our progress this term. We record them honestly, including the critiques.

### 2.1 The proposal: capture-recapture (PondFish v1)

The original proposal attacked the *stopping* problem rather than the *finding* problem. The idea was borrowed from ecology: to estimate a fish population you capture a sample, tag it, release it, and capture a second sample; the overlap between the two tells you the size of the whole. Translated to retrieval, we would run several independent search "probes," measure how much their results overlapped, and use the overlap to estimate how many relevant documents remained undiscovered. When the estimate said we were nearly done, we would stop.

**Critique (round 1):** the probes are just hybrid search. They do not drive the agent into regions of the collection it has not touched. BM25 and hybrid retrieval cannot fix the underlying problem, which is that a cluster of relevant documents may be unreachable from any query the agent is likely to write.

We accepted this. Reviewing our own design, exploration was accidental: a probe would only reveal a blind spot if it happened to stumble into one. The machinery measured coverage well and improved it not at all.

### 2.2 Second design: Discover, Amplify, Estimate (v2)

We made exploration explicit. The collection was partitioned into regions, and the system deliberately sampled from two pools it had not covered: the **never-retrieved pool** (documents that appeared in no query's top-2,000) and the **near-miss tail** (documents that ranked 51–2,000 for some query but were never judged). A hit from either pool would seed the next round of searching.

**Critique (round 2):** if the embedding is not good enough for the index to reach a relevant cluster, random sampling over those same regions will not reach it either. Sampling is weak at finding rare items.

The arithmetic supports this. With roughly 15 relevant documents hiding among 18,000 untouched ones, 300 random samples expect about 0.25 hits. As a primary discovery mechanism, sampling does not pay for itself.

The accompanying suggestion became the current design: **make it an evolutionary algorithm, where the LLM writes each generation of queries from the previous generation's results.** The reasoning is one sentence — *we cannot change the embedding, but we can change the query.*

### 2.3 Current design: evolutionary search (v3)

If an index cannot reach a relevant cluster from the queries we happen to have, the fix is not a better index but better queries — and the documents already found are the best available evidence about what those queries should say. A found filing that talks about *recreational boating demand* tells us a vocabulary exists that none of our starting queries used.

So the system treats the query set as a **population** that evolves. Fitness is **novelty**: new relevant documents per judgment spent. Re-finding something already in the ledger scores zero, which is what stops the population from converging on one good query and mining it forever.

**One disagreement remains open.** We argued that sampling still has two jobs evolution cannot do: supplying *random immigrants* (the standard evolutionary-algorithm remedy for getting stuck in a local optimum, since evolution can only move outward from where it has already been), and providing a final validation sample to bound what is still missing. The professor was skeptical but approved keeping sampling as one branch. **We did not build this for the midterm.** It is the single cleanest experiment left, and §7 describes it as our next step.

## 3. What we scoped for the midterm, and what we cut

The POC had to be buildable in days on a student budget, so we cut aggressively and recorded the cuts rather than quietly dropping them.

**Built:** three retrieval branches, novelty fitness, per-branch elite selection, an LLM breeder, amplification from confirmed finds, a simulated judge, four static baselines, and a breeder ablation across four models.

**Deliberately deferred:**

| Deferred | Why |
|---|---|
| Capture-recapture estimation | The finding problem has to work before the stopping problem is worth solving. |
| The stopping rule | Same reason. We run to a fixed budget instead. |
| The sampling branch / random immigrants | The open disagreement in §2.3. It needs a controlled ablation, not a feature flag. |
| A real LLM judge | Would cost more than our budget and add noise to every comparison. See §5.3. |
| Multi-hop "find all" | Out of scope for a single-question POC. |

Our total spend to date is approximately **$20**, against a self-imposed cap of $50. About $15.60 of that was the one-time cost of labelling the collection.

## 4. Method

### 4.1 Branches

Each query belongs to one of three retrieval branches, chosen because they fail in *different* ways:

- **Exact phrase** — substring matching, grep-like. Finds idiomatic wording with near-perfect precision; finds nothing at all if the phrase is one word off.
- **BM25** — lexical ranking over the collection (`bm25s`). Robust to word order, blind to vocabulary it has never seen in the query.
- **Embeddings** — dense retrieval over a flat FAISS index using `BAAI/bge-small-en-v1.5` (384 dimensions). Catches paraphrase, but blurs the distinction between related and relevant.

Keeping one elite per branch is a deliberate quality-diversity decision: without it, the embedding branch's higher raw yield would crowd the other two out of the population within two generations, and we would lose the failure-mode diversity that motivated the design.

### 4.2 One generation

1. **Run** every query in the population, retrieving to depth 2,000.
2. **Judge** the top 25 *previously unjudged* documents per query. Each judgment costs one unit of budget.
3. **Record** every outcome in a ledger — found, rejected, seen-but-unjudged, untouched — so no document is ever judged twice.
4. **Amplify** from confirmed finds: take the chunk of a found document closest in meaning to the question, and derive new queries from it, both by nearest-neighbour vector search and by extracting distinctive terms and phrases. Capped at 25 judgments per generation.
5. **Select** the highest-novelty live query per branch as that branch's elite. A query whose budget is exhausted cannot keep its slot.
6. **Breed.** The LLM receives the elites, their novelty scores, the text of what they found, and coaching about each branch's failure modes, and writes the next generation.

### 4.3 The leakage rule

This one is a correctness constraint, not a preference. The six Generation 0 queries were written **before anyone looked at a single label.**

Nothing inside the search loop may read the answer key or the labeller's supporting quotes. The breeder sees only documents the judge has already confirmed — which is exactly what it would see in deployment. Without this rule, every number in §6 would be meaningless.

A clarification on those seeds, since they are the one hand-written component left in the loop. The only input the system genuinely requires from a human is **the question itself**; the seeds are a convenience, not a design commitment. Nothing stops the breeder from writing Generation 0 as well, from the question text alone — doing so reads no labels and so does not violate the rule above. We wrote them by hand for the POC to hold the starting point identical across every arm, which is what makes the baselines and the evolution runs comparable at all. Generating them is listed in §7.3.

## 5. Experimental setup

### 5.1 Collection and question

| | |
|---|---|
| Collection | 3,000 SEC 10-K filings from 2020, Items 1A and 7 |
| Chunking | 300 words, 50-word overlap → 270,029 chunks |
| Question | `stay_home_boom` — did the filer report increased demand because people stayed home during COVID-19? |
| Relevant | **129 filings (4.30%)** |
| Judgment budget | 1,000 — **33% of the cost of exhaustive review** |
| Population | 6 queries, 1 elite retained per branch |
| Judge depth | top 25 unjudged documents per query |

Items 1A (Risk Factors) and 7 (MD&A) are where a company narrates what actually happened to its business, which is where evidence for this kind of question lives. Restricting to them also keeps a filing inside an LLM context window.

All three team members verified they were working from a byte-identical collection: `filings.jsonl` matched by SHA-256 across all three machines, and the chunk count matched independently.

### 5.2 The answer key, and its known weakness

Every filing was labelled by a single cheap model (`claude-haiku-5-5`) against a rubric written before labelling began. These are **silver-standard** labels, in the sense of UMBRELA [8]: model-generated judgments standing in for human assessors.

A 200-filing pilot cross-checked that labeller against a second model from a different family (Gemini). The result is worth stating plainly:

| | Positives | Rate |
|---|---:|---:|
| Claude Haiku | 7 / 200 | 3.5% |
| Gemini | 12 / 163 | 7.4% |

On the 163 filings both labelled, there were **7 disagreements, all in one direction** — Gemini said yes where Claude said no, and never the reverse. Positive-class agreement is therefore 5/12, about **42%**. Our labeller is the more conservative of the two by roughly a factor of two on the positive class.

So 129 is best read as a **floor**, not a count. We return to the consequences in §7.2. For context, UMBRELA reports Cohen's κ of roughly 0.42–0.50 against human assessors [8], so our figure is not an outlier for this class of method — which is a statement about how hard the problem is, not a defence.

### 5.3 The judge

For the midterm the judge is a **ground-truth lookup**: asking it about a filing returns that filing's label and costs one unit of budget. This is a simulation, chosen deliberately. It makes the budget the only experimental variable, so differences between arms cannot be attributed to a noisy judge, and it costs nothing, which let us run 22 full experiments inside our budget. The price is that our recall figures are an **upper bound** on what the same system would achieve with a real LLM judge.

### 5.4 Baselines

Four static baselines, each spending the same 1,000 judgments on the *same six Generation 0 queries* with no evolution. Each fuses its queries' rankings with reciprocal rank fusion and judges down the fused list. They use no LLM and no randomness, so each is a single deterministic run.

## 6. Results

### 6.1 Headline

Relevant filings found after 1,000 judgments, out of 129:

| Arm | Found | Recall |
|---|---:|---:|
| Baseline: exact phrase only | 10 | 7.8% |
| Baseline: embeddings only | 96 | 74.4% |
| Baseline: BM25 only | 101 | 78.3% |
| Baseline: hybrid, all three fused | 101 | 78.3% |
| **Evolution, Haiku breeder** (6 runs) | **107.0** | **82.9%** |
| **Evolution, Sonnet breeder** (5 runs) | **110.8** | **85.9%** |

The Haiku runs predate two later fixes to the amplification step. A matched pair of Sonnet runs across that same change differed by 0.6 filings (p = 0.83), so we treat the two code versions as comparable, but the Haiku row is not a strict like-for-like.

The comparison that matters is not the means but the ranges. **Our worst evolution run found 103. The best baseline found 101.** Of ten runs at the current code version, nine finished above the hybrid baseline, one level with it, and none below. The best single run found 115 of 129 (89.1%).

The exact-phrase baseline exhausted itself after 13 judgments — its six queries match no further filings — which is the clearest possible illustration of why a static query set is a dead end.

Each evolution run found 12–20 relevant filings the hybrid baseline missed, while missing 5–13 that it found. The two approaches are not simply ranked; they reach different parts of the collection.

### 6.2 Recall against budget

![Relevant filings found against judgments spent. Both evolution lines rise above every static baseline after roughly 270 judgments, ending at 110.8 with the Sonnet breeder and 107.0 with Haiku, against 101 for the hybrid baseline. The shaded band is the range across the five Sonnet seeds.](figures/recall_vs_budget.png)

**Figure 1.** Relevant filings found against judgments spent, out of 129. Shaded band is the range across the five Sonnet seeds. Drawn from the same run logs as every table below.

The same data at selected budgets:

| Arm | 100 | 300 | 500 | 750 | 1,000 |
|---|---:|---:|---:|---:|---:|
| Baseline: hybrid | **35** | 58 | 73 | 86 | 101 |
| Baseline: BM25 | 25 | 47 | 66 | 87 | 101 |
| Baseline: embeddings | 27 | 52 | 71 | 85 | 96 |
| Evolution, Sonnet | 22 | 59.8 | 83.0 | 100.4 | **110.8** |

**The hybrid baseline wins early.** At 100 judgments it leads 35 to 22, because it fuses all six seed queries immediately while we are still spending our first generation on them one at a time. We overtake near 300 judgments and the gap widens from there. We report this because a method that only ever wins would be the more suspicious result: evolution costs something up front and pays back later, which is what the curve shows.

### 6.3 Where the finds come from

Totalled over ten runs:

| Source | Finds | Share | Yield per 100 judgments |
|---|---:|---:|---:|
| Bred by the LLM | 614 | **56.5%** | 9.6 |
| Generation 0 (hand-written seeds — see §4.3) | 379 | 34.9% | 20.4 |
| Amplify (from confirmed finds) | 94 | 8.6% | 5.4 |

Over half of everything found came from queries no human wrote. The hand-written seeds have the higher per-judgment yield, which is expected — they are mining the densest part of the collection first — but they run out, and the bred queries are what keep the curve climbing.

By branch, the yields are strikingly uneven:

| Branch | Finds | Judgments | Yield per 100 |
|---|---:|---:|---:|
| Exact phrase | 76 | 151 | **50.3** |
| BM25 | 461 | 3,912 | 11.8 |
| Embeddings | 456 | 4,190 | 10.9 |
| Amplify | 94 | 1,747 | 5.4 |

The exact-phrase branch is four times more efficient than anything else *when the LLM writes a good phrase for it* — and it contributes only 7% of total finds because good phrases are rare. This is the quality-diversity argument made concrete: a branch that is usually idle but occasionally spectacular is worth a population slot, and a pure fitness-maximising selection rule would have deleted it.

### 6.4 What evolution actually discovers

One run (Sonnet breeder, 115/129) generation by generation, with the category each new query opened:

| Gen | New finds | Running total | Category the bred query opened |
|---:|---:|---:|---|
| 0 | +31 | 31 | *(hand-written seeds — see §4.3)* |
| 1 | +15 | 46 | gardening, baking, puzzles, crafts, pets |
| 2 | +22 | 68 | boating, RVs, bikes, fitness equipment / streaming, video games |
| 3 | +21 | 89 | home improvement — paint, flooring, furniture |
| 4 | +11 | 100 | swimming pools, hot tubs, patio grills |
| 5 | +7 | 107 | games, puzzles, toys, books |
| 6 | +8 | 115 | at-home food, beverages, groceries |

Every one of these is a distinct consumer vertical, and **not one of those words appears in anything we wrote.** The seeds said "covid, hobby, lifestyle, increase." The system worked out that the question is really about a dozen separate industries and went after them one at a time. This is the mechanism the design was built to produce, and it is visible in the logs rather than inferred from an aggregate.

The elite mechanism is also visible. One embedding query bred in generation 2 holds its branch slot through generations 3, 5 and 6, returning 6, then 5, then 2, then 2 new finds as it digs deeper into its own ranking, and then loses the slot when it runs dry. No rule tells it to stop; novelty fitness simply stops rewarding it.

### 6.5 The breeder barely matters

We ran the same loop with four different breeder models:

| Breeder | Runs | Found (mean) | SD | Range | Cost per run |
|---|---:|---:|---:|---:|---:|
| Haiku 5.5 | 6 | 107.0 | 2.6 | 104–111 | $0.0037 |
| Sonnet 5.5 | 5 | 110.8 | 5.0 | 103–115 | $0.1171 |
| Opus 5.5 | 5 | 106.6 | 5.3 | 101–114 | $0.2280 |

Sonnet leads Opus by 4.2 finds, but a Welch t-test gives p = 0.233 — **not significant** at n = 5 — and seed-to-seed standard deviation (5.0–5.3) is larger than the gap between the models. We looked for a structural explanation in the logs and did not find one, so we report it as what it is: a difference we cannot distinguish from noise.

The useful finding is the negative one. Across roughly a **60× price range**, every breeder lands between 0.83 and 0.86 recall. The expensive part of this system is the judgments, not the breeding, and the breeding does not need a frontier model.

### 6.6 Saturation

One run at a 1,500-judgment budget reached **121 of 129 (93.8%)** — but its last new find arrived at judgment 1,066, and four further generations produced nothing. Pooled across every run we have executed, only **3 of the 129 relevant filings were never found by anything.**

So the system saturates well before it exhausts its budget. It stops because it runs out of ideas, not money. We do not yet know whether the remaining filings are unreachable in principle or merely need a kind of query we have not learned to generate.

## 7. Discussion, limitations, and next steps

### 7.1 What we think we have shown

Evolving the query set beats every static baseline at equal judgment budget, with no overlap between the two distributions, and it does so by discovering vocabulary the humans who wrote the seeds did not think of. The mechanism is not mysterious — it is readable in the generation logs as a sequence of consumer verticals. The cost of the evolutionary component is negligible compared with the cost of judging.

### 7.2 Limitations

**The judge is simulated.** Ground-truth lookup never errs and answers at the filing level regardless of which chunk was retrieved. Our recall figures are an upper bound on a real deployment.

**The answer key is silver standard and biased toward under-labelling.** Per §5.2, our labeller found positives at roughly half the rate of a second model and never disagreed in the other direction. So 129 is a floor; true recall against a complete key would be lower, and a filing we correctly found but the key marks negative is scored against us twice over. Soboroff [9] makes the general version of this argument: an LLM-generated key sets that LLM as a ceiling on measured performance.

*What this does not undermine:* every arm is scored against the same key, so a key that misses positives shifts all arms together. The comparative claim — worst evolution run above best baseline — is robust to this bias even though the absolute recall figures are not.

**n = 1 at the task level.** One question, one collection, one domain. We have five or six seeds per configuration, which tells us about run-to-run variance and nothing about generalisation.

**The collection is small enough that the problem is not yet real.** Exhaustive review of 3,000 filings is affordable. The regime this method is designed for — where exhaustive review is out of the question — is precisely the regime we have not tested.

**No stopping rule.** We run to a fixed budget. A deployed system does not know when it is done, which was the original proposal's entire subject and remains unsolved.

### 7.3 Next steps

1. **The sampling ablation** — evolution alone versus evolution plus a random-immigrant branch, at equal budget, with lineage tracking. This settles the open disagreement from §2.3 with data instead of argument, and §6.6 suggests it is the right experiment: a saturated population is exactly the condition random immigrants exist to address.
2. **Upgrade the answer key.** Re-label the ~130 positives plus a sample of negatives with a stronger model, or two-of-three voting across families. This is cheap because positives are few, and it bounds the §7.2 bias directly.
3. **A real LLM judge**, to measure how far the simulated figures overstate things — and to supply something the lookup cannot: which passage the judge actually relied on, which would let amplify mine the right chunk instead of guessing.
4. **Let the breeder write Generation 0.** Per §4.3 the seeds are the last hand-written piece, and removing them reduces the human input to the question alone. It also tests something we currently cannot see: whether our hand-written starting point helped or quietly constrained the search, since every run so far has begun from the same six queries.
5. **A second question and a larger collection**, to say anything at all about generalisation.
6. **Return to stopping.** Saturation is detectable in our logs after the fact; the question is whether it is detectable at the time, without knowing the answer.

### 7.4 Relation to prior work

The closest prior art is **AlphaEvolve** [1], which runs an evolutionary loop where an LLM proposes edits to a program and automated evaluators score the result. The structure is the same; the object under evolution is not. AlphaEvolve evolves programs against a numeric objective that can be computed for free. We evolve *queries* against novelty, where every evaluation costs a judgment, which makes budget allocation rather than mutation quality the central design problem.

Our fitness function is novelty search [2] rather than fitness maximisation, and per-branch elites are a quality-diversity archive in the sense of MAP-Elites [3]. Amplification is Rocchio relevance feedback [4] with an LLM in place of the term-weighting update. The budgeted high-recall setting, and the expectation that recall plateaus at 80–90%, come from the e-discovery literature on continuous active learning [5, 6]; our §6.6 saturation at 93.8% is consistent with it. Stopping rules with statistical guarantees [7] are the obvious next thing to adopt, and the "elusion illusion" [10] is the standing warning about why recall estimated from a sample misleads.

---

## References

1. Novikov, A. et al. *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* arXiv:2506.13131, 2025.
2. Lehman, J. and Stanley, K.O. *Abandoning Objectives: Evolution Through the Search for Novelty Alone.* Evolutionary Computation, 2011.
3. Mouret, J.-B. and Clune, J. *Illuminating search spaces by mapping elites.* arXiv:1504.04909, 2015.
4. Rocchio, J.J. *Relevance Feedback in Information Retrieval.* In The SMART Retrieval System, 1971.
5. Cormack, G.V. and Grossman, M.R. *Scalability of Continuous Active Learning for Reliable High-Recall Text Classification.* CIKM 2016. doi:10.1145/2983323.2983776
6. Robertson, S. and Zaragoza, H. *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 2009.
7. Lewis, D.D., Yang, E. and Frieder, O. *Certifying One-Phase Technology-Assisted Reviews.* arXiv:2108.12746, 2021.
8. Upadhyay, S., Pradeep, R., Thakur, N., Craswell, N. and Lin, J. *UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor.* arXiv:2406.06519, 2024.
9. Soboroff, I. *Don't Use LLMs to Make Relevance Judgments.* Information Retrieval Research 1(1), 29–46, 2025. arXiv:2409.15133
10. ACEDS. *The Elusion Illusion and the AI Revolution.* ACEDS blog.
11. Loukas, L. et al. *EDGAR-CORPUS: Billions of Tokens Make The World Go Round.* ECONLP 2021.
12. Xiao, S. et al. *C-Pack: Packed Resources For General Chinese Embeddings.* arXiv:2309.07597, 2023. (`BAAI/bge-small-en-v1.5`)
