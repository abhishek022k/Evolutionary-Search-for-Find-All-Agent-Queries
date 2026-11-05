# Evolutionary Search for "Find All": Results

_Generated from 12 runs in `eval/runs`. Every number is read from the run logs._

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
| `evolution_1000` | haiku 5.5 | 2020 | 105 | 0.814 | 8 | $0.0033 |
| `haiku_s2021` | haiku 5.5 | 2021 | 106 | 0.822 | 8 | $0.0040 |
| `haiku_s2022` | haiku 5.5 | 2022 | 109 | 0.845 | 8 | $0.0036 |
| `haiku_s2023` | haiku 5.5 | 2023 | 104 | 0.806 | 8 | $0.0040 |
| `haiku_s2024` | haiku 5.5 | 2024 | 111 | 0.860 | 8 | $0.0037 |
| `haiku_s2025` | haiku 5.5 | 2025 | 107 | 0.829 | 8 | $0.0033 |
| `sonnet_s2020` | sonnet 5.5 | 2020 | 114 | 0.884 | 8 | _unpriced_ |
| `sonnet_s2021` | sonnet 5.5 | 2021 | 108 | 0.837 | 8 | $0.0573 |
| `sonnet_s2022` | sonnet 5.5 | 2022 | 116 | 0.899 | 8 | $0.0587 |
| `sonnet_s2023` | sonnet 5.5 | 2023 | 109 | 0.845 | 8 | $0.0583 |
| `sonnet_s2024` | sonnet 5.5 | 2024 | 110 | 0.853 | 8 | $0.0610 |
| `sonnet_s2025` | sonnet 5.5 | 2025 | 112 | 0.868 | 8 | $0.0604 |

## Breeder comparison

| Breeder | Runs | Found (mean) | SD | Range | Recall (mean) | Cost/run |
|---|---:|---:|---:|---:|---:|---:|
| haiku 5.5 | 6 | **107.0** | 2.6 | 104-111 | 0.829 | $0.0037 |
| sonnet 5.5 | 6 | **111.5** | 3.1 | 108-116 | 0.864 | $0.0591 |

Difference in means: **+4.5 finds** (+3.5 recall points), sonnet 5.5 over haiku 5.5. Welch t-test: t=2.73, p=0.022, **significant** at n=6 and 6. The ranges overlap: haiku 5.5's best run (111) reaches sonnet 5.5's worst (108).

## How the search evolved

New relevant filings found in each generation, averaged over seeds. Generation 0 is the hand-written queries; everything after is bred.

| Breeder | gen 0 | gen 1 | gen 2 | gen 3 | gen 4 | gen 5 | gen 6 | gen 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| haiku 5.5 | 31.0 | 18.5 | 16.5 | 12.0 | 10.8 | 9.0 | 6.0 | 3.2 |
| sonnet 5.5 | 31.0 | 20.0 | 21.7 | 13.0 | 10.7 | 7.0 | 5.0 | 3.2 |

## Recall against judgments spent

**haiku 5.5** (mean of 6 runs)

```
  100 judgments  ██████                              22.0 / 129   recall 0.171
  200 judgments  ████████████                        44.3 / 129   recall 0.344
  300 judgments  ███████████████                     57.2 / 129   recall 0.443
  400 judgments  ██████████████████                  67.3 / 129   recall 0.522
  500 judgments  ████████████████████                76.3 / 129   recall 0.592
  600 judgments  ██████████████████████              83.7 / 129   recall 0.649
  700 judgments  ████████████████████████            92.7 / 129   recall 0.718
  800 judgments  ██████████████████████████          98.0 / 129   recall 0.760
  900 judgments  ███████████████████████████        102.7 / 129   recall 0.796
 1000 judgments  ████████████████████████████       107.0 / 129   recall 0.829
```

**sonnet 5.5** (mean of 6 runs)

```
  100 judgments  ██████                              22.0 / 129   recall 0.171
  200 judgments  ████████████                        46.3 / 129   recall 0.359
  300 judgments  ████████████████                    62.2 / 129   recall 0.482
  400 judgments  ███████████████████                 73.5 / 129   recall 0.570
  500 judgments  ██████████████████████              85.0 / 129   recall 0.659
  600 judgments  ████████████████████████            90.8 / 129   recall 0.704
  700 judgments  ██████████████████████████          97.7 / 129   recall 0.757
  800 judgments  ███████████████████████████        103.0 / 129   recall 0.798
  900 judgments  ████████████████████████████       106.7 / 129   recall 0.827
 1000 judgments  █████████████████████████████      111.5 / 129   recall 0.864
```

## Where the finds came from

Totalled over all 12 runs.

| Source | Finds | Share | Judgments | Yield per 100 |
|---|---:|---:|---:|---:|
| Bred by the LLM | 756 | 57.7% | 7774 | 9.7 |
| Generation 0 (hand-written) | 446 | 34.0% | 2171 | 20.5 |
| Amplify (from confirmed finds) | 109 | 8.3% | 2055 | 5.3 |

## Yield per branch

| Branch | Finds | Judgments | Yield per 100 |
|---|---:|---:|---:|
| `amplify` | 109 | 2055 | 5.3 |
| `bm25` | 540 | 4703 | 11.5 |
| `embedding` | 563 | 5003 | 11.3 |
| `exact` | 99 | 239 | 41.4 |

## The story of one run

`sonnet_s2022`, breeder sonnet 5.5, **116 of 129 found (recall 0.899)**.

Each generation's most productive queries. `*` marks a query the LLM wrote; the rest are survivors from earlier generations.

**Generation 0** &nbsp; 133 judgments, +31 new, 31 found so far

```
  [bm25     ]  8 new / 25 judged   covid, hobby, lifestyle, increase
  [embedding]  7 new / 25 judged   increased revenue due to COVID related consumer habits
  [embedding]  6 new / 25 judged   staying at home led to more orders
```

**Generation 1** &nbsp; 125 judgments, +18 new, 49 found so far

```
* [bm25     ]  6 new / 25 judged   pandemic homebound consumers gardening baking cooking crafts puz
  [embedding]  5 new / 25 judged   increased revenue due to COVID related consumer habits
* [embedding]  5 new / 25 judged   more people exercising and enjoying outdoor recreation during th
```

**Generation 2** &nbsp; 151 judgments, +23 new, 72 found so far

```
* [embedding]  9 new / 25 judged   with consumers staying home, streaming, video gaming, and online
* [embedding]  4 new / 25 judged   home improvement and DIY projects surged as homeowners invested 
* [bm25     ]  3 new / 25 judged   pandemic homebound consumers gardening baking cooking crafts puz
```

**Generation 3** &nbsp; 125 judgments, +17 new, 89 found so far

```
* [bm25     ]  6 new / 25 judged   quarantine consumers online learning telehealth home fitness sub
* [bm25     ]  6 new / 25 judged   pandemic homebound consumers gardening baking cooking crafts puz
* [embedding]  3 new / 25 judged   with consumers staying home, streaming, video gaming, and online
```

**Generation 4** &nbsp; 151 judgments, +10 new, 99 found so far

```
* [bm25     ]  5 new / 25 judged   pandemic boating RV camping fishing golf outdoor recreation part
* [bm25     ]  3 new / 25 judged   pandemic homebound consumers gardening baking cooking crafts puz
* [embedding]  2 new / 25 judged   with consumers staying home, streaming, video gaming, and online
```

**Generation 5** &nbsp; 125 judgments, +3 new, 102 found so far

```
* [embedding]  1 new / 25 judged   people exercising at home during the pandemic bought fitness equ
* [embedding]  1 new / 25 judged   with consumers staying home, streaming, video gaming, and online
* [bm25     ]  1 new / 25 judged   lockdowns consumers cooking at home grocery food consumption inc
```

**Generation 6** &nbsp; 155 judgments, +13 new, 115 found so far

```
* [embedding]  4 new / 25 judged   with consumers staying home, streaming, video gaming, and online
* [embedding]  3 new / 25 judged   our pet food and pet supplies sales grew as people stuck at home
* [bm25     ]  3 new / 25 judged   pandemic consumers home bound purchased furniture appliances bac
```

**Generation 7** &nbsp; 35 judgments, +1 new, 116 found so far

```
* [embedding]  1 new / 25 judged   with consumers staying home, streaming, video gaming, and online
```

## Caveats

- The judge is a ground-truth lookup, so it never makes a mistake and it answers at the filing level whatever chunk was retrieved. Recall here is an **upper bound** on what a real LLM judge would achieve.
- Labels are silver standard: one cheap model labelled every filing, and 3 relevant rows have quotes that could not be verified in the filing.
- The breeder is an LLM with no seed, so repeat runs differ even at a fixed `--seed`, which only controls the sampling inside the loop.
- Generation 0 is identical across every run by construction, which is why gen 0 finds the same number of filings every time.
