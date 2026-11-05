# Evolutionary Search for "Find All": Results

_Generated from 2 runs in `eval/runs_wide`. Every number is read from the run logs._

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
| `wide_opus_s3001` | opus 5.5 | 3001 | 109 | 0.845 | 8 | $0.2275 |
| `wide_sonnet_s3001` | sonnet 5.5 | 3001 | 116 | 0.899 | 8 | $0.1187 |

## Breeder comparison

| Breeder | Runs | Found (mean) | SD | Range | Recall (mean) | Cost/run |
|---|---:|---:|---:|---:|---:|---:|
| opus 5.5 | 1 | **109.0** | - | 109-109 | 0.845 | $0.2275 |
| sonnet 5.5 | 1 | **116.0** | - | 116-116 | 0.899 | $0.1187 |

Difference in means: **+7.0 finds** (+5.4 recall points), sonnet 5.5 over opus 5.5. Welch t-test: t=nan, p=nan, **not significant** at n=1 and 1.

## How the search evolved

New relevant filings found in each generation, averaged over seeds. Generation 0 is the hand-written queries; everything after is bred.

| Breeder | gen 0 | gen 1 | gen 2 | gen 3 | gen 4 | gen 5 | gen 6 | gen 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opus 5.5 | 31.0 | 21.0 | 22.0 | 21.0 | 2.0 | 8.0 | 3.0 | 1.0 |
| sonnet 5.5 | 31.0 | 25.0 | 20.0 | 14.0 | 17.0 | 3.0 | 5.0 | 1.0 |

## Recall against judgments spent

**opus 5.5** (mean of 1 runs)

```
  100 judgments  ██████                              22.0 / 129   recall 0.171
  200 judgments  ████████████                        46.0 / 129   recall 0.357
  300 judgments  ████████████████                    61.0 / 129   recall 0.473
  400 judgments  ███████████████████                 73.0 / 129   recall 0.566
  500 judgments  ███████████████████████             89.0 / 129   recall 0.690
  600 judgments  █████████████████████████           96.0 / 129   recall 0.744
  700 judgments  ██████████████████████████          99.0 / 129   recall 0.767
  800 judgments  ███████████████████████████        104.0 / 129   recall 0.806
  900 judgments  ████████████████████████████       107.0 / 129   recall 0.829
 1000 judgments  █████████████████████████████      109.0 / 129   recall 0.845
```

**sonnet 5.5** (mean of 1 runs)

```
  100 judgments  ██████                              22.0 / 129   recall 0.171
  200 judgments  █████████████                       49.0 / 129   recall 0.380
  300 judgments  ██████████████████                  69.0 / 129   recall 0.535
  400 judgments  ████████████████████                74.0 / 129   recall 0.574
  500 judgments  ███████████████████████             88.0 / 129   recall 0.682
  600 judgments  ███████████████████████████        101.0 / 129   recall 0.783
  700 judgments  ████████████████████████████       107.0 / 129   recall 0.829
  800 judgments  █████████████████████████████      110.0 / 129   recall 0.853
  900 judgments  ██████████████████████████████     112.0 / 129   recall 0.868
 1000 judgments  ███████████████████████████████    116.0 / 129   recall 0.899
```

## Where the finds came from

Totalled over all 2 runs.

| Source | Finds | Share | Judgments | Yield per 100 |
|---|---:|---:|---:|---:|
| Bred by the LLM | 131 | 58.2% | 1309 | 10.0 |
| Generation 0 (hand-written) | 73 | 32.4% | 341 | 21.4 |
| Amplify (from confirmed finds) | 21 | 9.3% | 350 | 6.0 |

## Yield per branch

| Branch | Finds | Judgments | Yield per 100 |
|---|---:|---:|---:|
| `amplify` | 21 | 350 | 6.0 |
| `bm25` | 98 | 750 | 13.1 |
| `embedding` | 93 | 879 | 10.6 |
| `exact` | 13 | 21 | 61.9 |

## The story of one run

`wide_sonnet_s3001`, breeder sonnet 5.5, **116 of 129 found (recall 0.899)**.

Each generation's most productive queries. `*` marks a query the LLM wrote; the rest are survivors from earlier generations.

**Generation 0** &nbsp; 133 judgments, +31 new, 31 found so far

```
  [bm25     ]  8 new / 25 judged   covid, hobby, lifestyle, increase
  [embedding]  7 new / 25 judged   increased revenue due to COVID related consumer habits
  [embedding]  6 new / 25 judged   staying at home led to more orders
```

**Generation 1** &nbsp; 125 judgments, +25 new, 56 found so far

```
* [bm25     ] 10 new / 25 judged   homebound consumers pandemic participation outdoor recreation in
  [embedding]  5 new / 25 judged   increased revenue due to COVID related consumer habits
* [embedding]  5 new / 25 judged   consumers picked up new hobbies like baking, gardening, crafting
```

**Generation 2** &nbsp; 153 judgments, +20 new, 76 found so far

```
* [bm25     ]  8 new / 25 judged   pandemic consumers cooking at home more demand pantry groceries 
* [embedding]  7 new / 25 judged   with families staying home during lockdowns, demand for our vide
* [exact    ]  1 new /  3 judged   home improvement projects
```

**Generation 3** &nbsp; 125 judgments, +14 new, 90 found so far

```
* [bm25     ]  7 new / 25 judged   consumers invested in home fitness equipment, furniture, applian
* [embedding]  4 new / 25 judged   with families staying home during lockdowns, demand for our vide
* [embedding]  3 new / 25 judged   increased usage of our broadband, internet and data services as 
```

**Generation 4** &nbsp; 150 judgments, +17 new, 107 found so far

```
* [embedding]  5 new / 25 judged   tank exchange and residential energy usage rose because people w
* [embedding]  5 new / 25 judged   demand for outdoor recreation products such as boats, RVs, campe
* [bm25     ]  3 new / 25 judged   consumers invested in home fitness equipment, furniture, applian
```

**Generation 5** &nbsp; 150 judgments, +3 new, 110 found so far

```
* [bm25     ]  1 new / 25 judged   consumers invested in home fitness equipment, furniture, applian
* [embedding]  1 new / 25 judged   demand for outdoor recreation products such as boats, RVs, campe
* [embedding]  1 new / 25 judged   consumers cooking and eating at home more during the pandemic dr
```

**Generation 6** &nbsp; 125 judgments, +5 new, 115 found so far

```
* [embedding]  2 new / 25 judged   subscribers streamed more video, played more video games, and us
* [bm25     ]  1 new / 25 judged   consumers invested in home fitness equipment, furniture, applian
* [bm25     ]  1 new / 25 judged   pets adopted pandemic households spending more time with pets in
```

**Generation 7** &nbsp; 39 judgments, +1 new, 116 found so far

```
* [embedding]  1 new / 25 judged   subscribers streamed more video, played more video games, and us
```

## Caveats

- The judge is a ground-truth lookup, so it never makes a mistake and it answers at the filing level whatever chunk was retrieved. Recall here is an **upper bound** on what a real LLM judge would achieve.
- Labels are silver standard: one cheap model labelled every filing, and 3 relevant rows have quotes that could not be verified in the filing.
- The breeder is an LLM with no seed, so repeat runs differ even at a fixed `--seed`, which only controls the sampling inside the loop.
- Generation 0 is identical across every run by construction, which is why gen 0 finds the same number of filings every time.
