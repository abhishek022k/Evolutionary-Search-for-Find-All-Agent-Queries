"""Build eval/RESULTS.md from every run in eval/runs/. Reads logs only, runs nothing.

  python eval/report.py                      # all full-collection evolution runs
  python eval/report.py --out eval/RESULTS.md

Regenerate it whenever a run finishes; the numbers always come from the logs, so the
file can never drift from what was actually measured.

Prints queries and counts, never filing text, so the output is safe to share.
"""
import argparse
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
from qualitative import origin_of, read_json, read_jsonl, report_branches  # noqa: E402

BAR = "█"
CHECKPOINTS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)


def load_runs(runs_dir):
    """Every finished full-collection evolution run, newest config wins ties."""
    out = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        meta = read_json(os.path.join(path, "run_meta.json"))
        summary = read_json(os.path.join(path, "summary.json"))
        if not meta or not summary or meta.get("arm") != "evolution" or meta.get("pilot"):
            continue
        out.append({
            "name": name, "meta": meta, "summary": summary,
            "generations": read_jsonl(os.path.join(path, "generations.jsonl")),
            "queries": read_jsonl(os.path.join(path, "queries.jsonl")),
            "judgments": read_jsonl(os.path.join(path, "judgments.jsonl")),
        })
    return out


def short(model):
    return model.replace("claude-", "").replace("-5-5", " 5.5")


def curve(judgments, relevant_total):
    found, by_seq = 0, {}
    for row in sorted(judgments, key=lambda r: r["seq"]):
        found += bool(row["relevant"])
        by_seq[row["seq"]] = found
    return by_seq


def bar(value, biggest, width=34):
    return BAR * max(1, round(width * value / biggest)) if value else ""


def fmt_stats(values):
    if len(values) == 1:
        return f"{values[0]:.1f}", "-"
    return f"{st.mean(values):.1f}", f"{st.stdev(values):.1f}"


def section_setup(runs, lines):
    meta = runs[0]["meta"]
    cfg = meta["config"]
    lines += [
        "## Setup", "",
        "| | |", "|---|---|",
        f"| Question | `{meta['question_id']}` |",
        f"| Collection | {meta['filings_in_scope']:,} SEC 10-K filings (2020), "
        f"Items 1A + 7 |",
        f"| Relevant | **{meta['relevant_in_scope']}** ({100 * meta['relevant_in_scope'] / meta['filings_in_scope']:.2f}%) |",
        f"| Judgment budget | {meta['budget']:,} ({100 * meta['budget'] / meta['filings_in_scope']:.0f}% of exhaustive review) |",
        f"| Population | {meta['population_size']} queries, "
        f"{cfg['elites_per_branch']} elite kept per branch |",
        f"| Judge depth | top {cfg['judge_depth']} unjudged filings per query |",
        f"| Retrieval depth | {cfg['search_k']:,} chunks |",
        f"| Amplify cap | {cfg['amplify_cap']} judgments per generation |",
        f"| Branches | exact phrase, BM25, embeddings ({cfg['embedding_model']}) |",
        "| Judge | ground-truth lookup, 1 judgment = 1 filing |",
        "",
        "Every run starts from the same six hand-written Generation 0 queries and the "
        "same judgment budget, which is what makes the arms comparable.",
        "",
    ]


def section_results(runs, lines):
    lines += ["## All runs", "",
              "| Run | Breeder | Seed | Found | Recall | Generations | Breeding cost |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in sorted(runs, key=lambda r: (r["meta"]["breeder"], r["meta"]["seed"])):
        s, m = r["summary"], r["meta"]
        cost = s["breeding_cost"]["dollars"]
        cost_text = f"${cost:.4f}" if cost else "_unpriced_"
        lines += [f"| `{r['name']}` | {short(m['breeder'])} | {m['seed']} | "
                  f"{s['found']} | {s['recall']:.3f} | {s['generations']} | {cost_text} |"]
    lines.append("")


def section_breeder(runs, lines):
    by_model = defaultdict(list)
    for r in runs:
        by_model[r["meta"]["breeder"]].append(r)
    if len(by_model) < 2:
        return
    total = runs[0]["meta"]["relevant_in_scope"]
    lines += ["## Breeder comparison", "",
              "| Breeder | Runs | Found (mean) | SD | Range | Recall (mean) | Cost/run |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in sorted(by_model):
        group = by_model[model]
        found = [r["summary"]["found"] for r in group]
        costs = [r["summary"]["breeding_cost"]["dollars"] for r in group
                 if r["summary"]["breeding_cost"]["dollars"]]
        mean, sd = fmt_stats(found)
        cost_cell = f"${st.mean(costs):.4f}" if costs else "-"
        lines += [f"| {short(model)} | {len(group)} | **{mean}** | {sd} | "
                  f"{min(found)}-{max(found)} | {st.mean(found) / total:.3f} | "
                  f"{cost_cell} |"]
    lines.append("")

    models = sorted(by_model)
    if len(models) == 2:
        a, b = (by_model[m] for m in models)
        fa = [r["summary"]["found"] for r in a]
        fb = [r["summary"]["found"] for r in b]
        diff = st.mean(fb) - st.mean(fa)
        note = (f"Difference in means: **{diff:+.1f} finds** "
                f"({100 * diff / total:+.1f} recall points), "
                f"{short(models[1])} over {short(models[0])}.")
        try:
            from scipy import stats
            t, p = stats.ttest_ind(fb, fa, equal_var=False)
            verdict = "**significant**" if p < 0.05 else "**not significant**"
            note += (f" Welch t-test: t={t:.2f}, p={p:.3f}, {verdict} at n="
                     f"{len(fa)} and {len(fb)}.")
        except Exception:
            pass
        if max(fa) >= min(fb):
            note += (f" The ranges overlap: {short(models[0])}'s best run ({max(fa)}) "
                     f"reaches {short(models[1])}'s worst ({min(fb)}).")
        lines += [note, ""]


def section_generations(runs, lines):
    by_model = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for g in r["generations"]:
            by_model[r["meta"]["breeder"]][g["generation"]].append(
                g["new_found_this_generation"])
    lines += ["## How the search evolved", "",
              "New relevant filings found in each generation, averaged over seeds. "
              "Generation 0 is the hand-written queries; everything after is bred.", ""]
    all_gens = sorted({g for m in by_model.values() for g in m})
    header = "| Breeder | " + " | ".join(f"gen {g}" for g in all_gens) + " |"
    lines += [header, "|---|" + "---:|" * len(all_gens)]
    for model in sorted(by_model):
        cells = []
        for g in all_gens:
            vals = by_model[model].get(g)
            cells.append(f"{st.mean(vals):.1f}" if vals else "-")
        lines += [f"| {short(model)} | " + " | ".join(cells) + " |"]
    lines.append("")


def section_curve(runs, lines):
    total = runs[0]["meta"]["relevant_in_scope"]
    by_model = defaultdict(list)
    for r in runs:
        by_model[r["meta"]["breeder"]].append(curve(r["judgments"], total))
    lines += ["## Recall against judgments spent", ""]
    for model in sorted(by_model):
        curves = by_model[model]
        lines += [f"**{short(model)}** (mean of {len(curves)} runs)", "", "```"]
        for n in CHECKPOINTS:
            vals = []
            for c in curves:
                vals.append(c.get(n) or max((v for k, v in c.items() if k <= n), default=0))
            mean = st.mean(vals)
            lines += [f"{n:>5} judgments  {bar(mean, total):<34} "
                      f"{mean:5.1f} / {total}   recall {mean / total:.3f}"]
        lines += ["```", ""]


def section_sources(runs, lines):
    agg = defaultdict(lambda: [0, 0])
    branches = defaultdict(lambda: [0, 0])
    for r in runs:
        origins = origin_of(r["queries"])
        for row in r["judgments"]:
            origin = origins.get(row["query_id"]) or "gen0"
            agg[origin][0] += bool(row["relevant"])
            agg[origin][1] += 1
        for branch, (found, spent, _n) in report_branches(r["queries"]).items():
            branches[branch][0] += found
            branches[branch][1] += spent
    n = len(runs)
    total_found = sum(v[0] for v in agg.values())
    lines += ["## Where the finds came from", "",
              f"Totalled over all {n} runs.", "",
              "| Source | Finds | Share | Judgments | Yield per 100 |",
              "|---|---:|---:|---:|---:|"]
    label = {"gen0": "Generation 0 (hand-written)", "bred": "Bred by the LLM",
             "amplify": "Amplify (from confirmed finds)"}
    for origin in sorted(agg, key=lambda o: -agg[o][0]):
        found, spent = agg[origin]
        lines += [f"| {label.get(origin, origin)} | {found} | "
                  f"{100 * found / total_found:.1f}% | {spent} | "
                  f"{100 * found / spent if spent else 0:.1f} |"]
    lines += ["", "## Yield per branch", "",
              "| Branch | Finds | Judgments | Yield per 100 |", "|---|---:|---:|---:|"]
    for branch in sorted(branches):
        found, spent = branches[branch]
        lines += [f"| `{branch}` | {found} | {spent} | "
                  f"{100 * found / spent if spent else 0:.1f} |"]
    lines.append("")


def section_story(runs, lines, top=3):
    """The single best run, generation by generation: the slide-9 narrative."""
    best = max(runs, key=lambda r: r["summary"]["found"])
    s, m = best["summary"], best["meta"]
    lines += ["## The story of one run", "",
              f"`{best['name']}`, breeder {short(m['breeder'])}, "
              f"**{s['found']} of {m['relevant_in_scope']} found "
              f"(recall {s['recall']:.3f})**.", "",
              "Each generation's most productive queries. `*` marks a query the LLM "
              "wrote; the rest are survivors from earlier generations.", ""]
    by_gen = defaultdict(list)
    for q in best["queries"]:
        by_gen[q["generation"]].append(q)
    for g in sorted(by_gen):
        summary = next((x for x in best["generations"] if x["generation"] == g), {})
        rows = [q for q in by_gen[g] if q["origin"] != "amplify" and q["new_found"] > 0]
        rows.sort(key=lambda q: -q["new_found"])
        if not rows:
            continue
        lines += [f"**Generation {g}** &nbsp; "
                  f"{summary.get('judgments_this_generation', 0)} judgments, "
                  f"+{summary.get('new_found_this_generation', 0)} new, "
                  f"{summary.get('found_cumulative', 0)} found so far", "", "```"]
        for q in rows[:top]:
            star = "*" if q["origin"] == "bred" else " "
            lines += [f"{star} [{q['branch']:9}] {q['new_found']:2d} new / "
                      f"{q['judgments_spent']:2d} judged   {q['query'][:64]}"]
        lines += ["```", ""]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="eval/runs")
    p.add_argument("--out", default="eval/RESULTS.md")
    args = p.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no finished evolution runs in {args.runs_dir}")

    lines = ["# Evolutionary Search for \"Find All\": Results", "",
             f"_Generated from {len(runs)} run"
             f"{'s' if len(runs) != 1 else ''} in `{args.runs_dir}`. "
             f"Every number is read from the run logs._", ""]
    section_setup(runs, lines)
    section_results(runs, lines)
    section_breeder(runs, lines)
    section_generations(runs, lines)
    section_curve(runs, lines)
    section_sources(runs, lines)
    section_story(runs, lines)
    lines += ["## Caveats", "",
              "- The judge is a ground-truth lookup, so it never makes a mistake and it "
              "answers at the filing level whatever chunk was retrieved. Recall here is "
              "an **upper bound** on what a real LLM judge would achieve.",
              "- Labels are silver standard: one cheap model labelled every filing, and "
              "3 relevant rows have quotes that could not be verified in the filing.",
              "- The breeder is an LLM with no seed, so repeat runs differ even at a "
              "fixed `--seed`, which only controls the sampling inside the loop.",
              "- Generation 0 is identical across every run by construction, which is "
              "why gen 0 finds the same number of filings every time.", ""]

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out} from {len(runs)} runs")


if __name__ == "__main__":
    main()
