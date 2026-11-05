# Evolutionary Search for "Find All": Results

_Generated from 10 runs in `eval/runs`. Every number is read from the run logs._

## Setup

| | |
|---|---|
| Question | `stay_home_boom` |
| Collection | 3,000 SEC 10-K filings (2020), Items 1A + 7 |
| Relevant | **129** (4.30%) |
| Judgment budget | 1,000 (33% of exhaustive review) |
| Population | 6 queries, 1 elite kept per branch |
| Judge depth | top 25 unjudged filings per query |
| Retrieval depth | 2,000 chunks |
| Amplify cap | 25 judgments per generation |
| Branches | exact phrase, BM25, embeddings (BAAI/bge-small-en-v1.5) |
| Judge | ground-truth lookup, 1 judgment = 1 filing |

Every run starts from the same six hand-written Generation 0 queries and the same judgment budget, which is what makes the arms comparable.

## All runs

| Run | Breeder | Seed | Found | Recall | Generations | Breeding cost |
|---|---|---|---:|---:|---:|---:|
| `v4_opus_s4001` | opus 5.5 | 4001 | 104 | 0.806 | 8 | $0.2203 |
| `v4_opus_s4002` | opus 5.5 | 4002 | 101 | 0.783 | 8 | $0.2260 |
| `v4_opus_s4003` | opus 5.5 | 4003 | 104 | 0.806 | 8 | $0.2249 |
| `v4_opus_s4004` | opus 5.5 | 4004 | 114 | 0.884 | 8 | $0.2345 |
| `v4_opus_s4005` | opus 5.5 | 4005 | 110 | 0.853 | 8 | $0.2345 |
| `v4_sonnet_s4001` | sonnet 5.5 | 4001 | 103 | 0.798 | 8 | $0.1209 |
| `v4_sonnet_s4002` | sonnet 5.5 | 4002 | 115 | 0.891 | 7 | $0.1056 |
| `v4_sonnet_s4003` | sonnet 5.5 | 4003 | 112 | 0.868 | 8 | $0.1187 |
| `v4_sonnet_s4004` | sonnet 5.5 | 4004 | 109 | 0.845 | 8 | $0.1198 |
| `v4_sonnet_s4005` | sonnet 5.5 | 4005 | 115 | 0.891 | 8 | $0.1204 |

## Evolution against the baselines

Relevant filings found after the same number of judgments, out of 129. Each baseline runs Generation 0 queries with no evolving: it merges their rankings with reciprocal rank fusion and judges down the merged list. Baselines use no LLM and no randomness, so each is a single run. Evolution rows show the mean, then the range across seeds.

| Arm | 100 | 200 | 300 | 500 | 750 | 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline: all six queries (hybrid) | 35 | 48 | 58 | 73 | 86 | 101 |
| Baseline: BM25 queries only | 25 | 35 | 47 | 66 | 87 | 101 |
| Baseline: embedding queries only | 27 | 42 | 52 | 71 | 85 | 96 |
| Baseline: exact-phrase queries only | 10 | 10 | 10 | 10 | 10 | 10 |
| Evolution: opus 5.5 | 22.0 (22-22) | 49.2 (48-50) | 62.2 (60-65) | 81.8 (78-85) | 96.4 (93-101) | 106.6 (101-114) |
| Evolution: sonnet 5.5 | 22.0 (22-22) | 44.6 (40-47) | 59.8 (58-63) | 83.0 (79-88) | 100.4 (93-106) | 110.8 (103-115) |

At 1,000 judgments the hybrid baseline finds 101. Of 10 evolution runs, 9 finished above it, 1 level with it and 0 below it. Each evolution run found 12-20 relevant filings the hybrid baseline missed, and missed 5-13 that it found.

Baseline: exact-phrase queries only stopped after 13 judgments because its queries match no more filings.

## Breeder comparison

| Breeder | Runs | Found (mean) | SD | Range | Recall (mean) | Cost/run |
|---|---:|---:|---:|---:|---:|---:|
| opus 5.5 | 5 | **106.6** | 5.3 | 101-114 | 0.826 | $0.2280 |
| sonnet 5.5 | 5 | **110.8** | 5.0 | 103-115 | 0.859 | $0.1171 |

Difference in means: **+4.2 finds** (+3.3 recall points), sonnet 5.5 over opus 5.5. Welch t-test: t=1.29, p=0.233, **not significant** at n=5 and 5. The ranges overlap: opus 5.5's best run (114) reaches sonnet 5.5's worst (103).

## How the search evolved

New relevant filings found in each generation, averaged over seeds. Generation 0 is the hand-written queries; everything after is bred.

| Breeder | gen 0 | gen 1 | gen 2 | gen 3 | gen 4 | gen 5 | gen 6 | gen 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opus 5.5 | 31.4 | 21.6 | 16.4 | 14.2 | 8.2 | 7.2 | 5.0 | 2.6 |
| sonnet 5.5 | 31.4 | 18.4 | 19.6 | 17.0 | 10.4 | 6.6 | 6.2 | 1.5 |

## Recall against judgments spent

**opus 5.5** (mean of 5 runs)

```
  100 judgments  ██████                              22.0 / 129   recall 0.171
  200 judgments  █████████████                       49.2 / 129   recall 0.381
  300 judgments  ████████████████                    62.2 / 129   recall 0.482
  400 judgments  ███████████████████                 70.8 / 129   recall 0.549
  500 judgments  ██████████████████████              81.8 / 129   recall 0.634
  600 judgments  ████████████████████████            90.0 / 129   recall 0.698
  700 judgments  █████████████████████████           94.2 / 129   recall 0.730
  800 judgments  ██████████████████████████          99.8 / 129   recall 0.774
  900 judgments  ███████████████████████████        102.8 / 129   recall 0.797
 1000 judgments  ████████████████████████████       106.6 / 129   recall 0.826
```

**sonnet 5.5** (mean of 5 runs)

```
  100 judgments  ██████                              22.0 / 129   recall 0.171
  200 judgments  ████████████                        44.6 / 129   recall 0.346
  300 judgments  ████████████████                    59.8 / 129   recall 0.464
  400 judgments  ███████████████████                 71.8 / 129   recall 0.557
  500 judgments  ██████████████████████              83.0 / 129   recall 0.643
  600 judgments  ████████████████████████            92.8 / 129   recall 0.719
  700 judgments  ██████████████████████████          97.6 / 129   recall 0.757
  800 judgments  ███████████████████████████        102.2 / 129   recall 0.792
  900 judgments  ████████████████████████████       105.6 / 129   recall 0.819
 1000 judgments  █████████████████████████████      110.8 / 129   recall 0.859
```

## Where the finds came from

Totalled over all 10 runs.

| Source | Finds | Share | Judgments | Yield per 100 |
|---|---:|---:|---:|---:|
| Bred by the LLM | 614 | 56.5% | 6398 | 9.6 |
| Generation 0 (hand-written) | 379 | 34.9% | 1855 | 20.4 |
| Amplify (from confirmed finds) | 94 | 8.6% | 1747 | 5.4 |

## Yield per branch

| Branch | Finds | Judgments | Yield per 100 |
|---|---:|---:|---:|
| `amplify` | 94 | 1747 | 5.4 |
| `bm25` | 461 | 3912 | 11.8 |
| `embedding` | 456 | 4190 | 10.9 |
| `exact` | 76 | 151 | 50.3 |

## The story of one run

`v4_sonnet_s4002`, breeder sonnet 5.5, **115 of 129 found (recall 0.891)**.

Each generation's most productive queries. `*` marks a query the LLM wrote; the rest are survivors from earlier generations.

**Generation 0** &nbsp; 133 judgments, +31 new, 31 found so far

```
  [bm25     ]  8 new / 25 judged   covid, hobby, lifestyle, increase
  [embedding]  7 new / 25 judged   increased revenue due to COVID related consumer habits
  [embedding]  6 new / 25 judged   staying at home led to more orders
```

**Generation 1** &nbsp; 125 judgments, +15 new, 46 found so far

```
  [embedding]  5 new / 25 judged   increased revenue due to COVID related consumer habits
* [bm25     ]  5 new / 25 judged   consumers pandemic gardening baking cooking puzzles crafts pets 
* [embedding]  2 new / 25 judged   with fewer options for travel and entertainment, families spent 
```

**Generation 2** &nbsp; 150 judgments, +22 new, 68 found so far

```
* [bm25     ]  8 new / 25 judged   pandemic homebound consumers boating RVs bikes fitness equipment
* [embedding]  6 new / 25 judged   people spending more time at home during the pandemic streamed m
  [embedding]  4 new / 25 judged   increased revenue due to COVID related consumer habits
```

**Generation 3** &nbsp; 150 judgments, +21 new, 89 found so far

```
* [bm25     ]  7 new / 25 judged   home improvement projects homeowners pandemic increased demand p
* [embedding]  5 new / 25 judged   people spending more time at home during the pandemic streamed m
* [bm25     ]  4 new / 25 judged   pandemic homebound consumers boating RVs bikes fitness equipment
```

**Generation 4** &nbsp; 150 judgments, +11 new, 100 found so far

```
* [bm25     ]  4 new / 25 judged   swimming pools backyard hot tubs grills patio consumers staying 
* [embedding]  3 new / 25 judged   consumers cooking and baking at home more during lockdowns drove
* [embedding]  3 new / 25 judged   people spending more time at home during the pandemic streamed m
```

**Generation 5** &nbsp; 150 judgments, +7 new, 107 found so far

```
* [embedding]  2 new / 25 judged   people spending more time at home during the pandemic streamed m
* [embedding]  2 new / 25 judged   with consumers confined at home, demand for our games, puzzles, 
* [embedding]  1 new / 25 judged   customers worked and studied from home so individual households 
```

**Generation 6** &nbsp; 142 judgments, +8 new, 115 found so far

```
* [embedding]  3 new / 25 judged   increased at-home consumption of our food, beverages, snacks and
* [embedding]  2 new / 25 judged   people spending more time at home during the pandemic streamed m
* [bm25     ]  2 new / 25 judged   at-home fitness workouts demand surged gyms closed consumers hom
```

## Caveats

- The judge is a ground-truth lookup, so it never makes a mistake and it answers at the filing level whatever chunk was retrieved. Recall here is an **upper bound** on what a real LLM judge would achieve.
- Labels are silver standard: one cheap model labelled every filing, and 3 relevant rows have quotes that could not be verified in the filing.
- The breeder is an LLM with no seed, so repeat runs differ even at a fixed `--seed`, which only controls the sampling inside the loop.
- Generation 0 is identical across every run by construction, which is why gen 0 finds the same number of filings every time.
