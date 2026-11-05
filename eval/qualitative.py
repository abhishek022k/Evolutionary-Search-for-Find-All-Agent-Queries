"""Turn a run directory into the numbers and the story the slides need.

  python eval/qualitative.py eval/runs/<run_id> [more run dirs ...]

Per run it writes curve.csv (judgments, found, recall) next to the logs, and prints:

  * where the finds came from: the original queries, bred queries, or amplify
  * yield per branch, as new finds per 100 judgments
  * the generation-by-generation query trace (slide 9)

With more than one run directory it also prints recall at matched judgment counts,
which is the generation-0 vs evolution comparison (slide 8).

Prints queries and counts only, never filing text, so the output is safe to paste
into slides or share with whoever is writing queries.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

CHECKPOINTS = (100, 200, 300, 500, 750, 1000, 1500)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def total_relevant(run_dir, args):
    """How many relevant filings were reachable in this run's scope."""
    meta = read_json(os.path.join(run_dir, "run_meta.json"))
    if meta.get("relevant_in_scope"):
        return meta["relevant_in_scope"], meta
    # Baseline runs do not write run_meta.json, so fall back to the labels. Say so:
    # falling back to the wrong answer key gives a wrong denominator and therefore a
    # wrong recall, which looks perfectly plausible on a plot.
    from judge import load_labels
    cfg = read_json(args.config)
    labels_path = args.labels or cfg.get("labels_path")
    question = args.question or cfg.get("question_id")
    if not labels_path or not question:
        return 0, meta
    print(f"  NOTE: {run_dir} has no run_meta.json, so recall is measured against\n"
          f"        {labels_path} / {question!r}"
          f"{' on the pilot subset' if (args.pilot or meta.get('pilot')) else ''}."
          f" Pass --labels/--question/--pilot if that is not the right answer key.")
    labels = load_labels(labels_path, question)
    filing_ids = [json.loads(l)["filing_id"] for l in open(cfg["filings_path"])]
    if meta.get("pilot") or args.pilot:
        pilot = set(open("data/processed/pilot_ids.txt").read().split())
        filing_ids = [f for f in filing_ids if f in pilot]
    return sum(bool(labels.get(f)) for f in filing_ids), meta


def write_curve(run_dir, judgments, relevant_total):
    """One row per judgment: the recall curve, straight from the judgment order."""
    path = os.path.join(run_dir, "curve.csv")
    found = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judgments", "found", "recall"])
        writer.writerow([0, 0, 0.0])
        for row in sorted(judgments, key=lambda r: r["seq"]):
            found += bool(row["relevant"])
            writer.writerow([row["seq"], found,
                             round(found / relevant_total, 6) if relevant_total else 0.0])
    return path


def recall_at(judgments, relevant_total, checkpoints):
    """Recall after the first n judgments, for each n. The curve, sampled."""
    out, found, by_seq = {}, 0, {}
    for row in sorted(judgments, key=lambda r: r["seq"]):
        found += bool(row["relevant"])
        by_seq[row["seq"]] = found
    last = max(by_seq) if by_seq else 0
    for n in checkpoints:
        if n > last:
            continue
        hits = by_seq.get(n) or max((v for k, v in by_seq.items() if k <= n), default=0)
        out[n] = (hits, hits / relevant_total if relevant_total else 0.0)
    return out


def origin_of(queries):
    """{query_id: gen0 | bred | amplify}. Baseline runs have no queries.jsonl."""
    origins = {}
    for row in queries:
        origins[row["query_id"]] = row["origin"]
    return origins


def report_sources(judgments, origins):
    """Which kind of query actually produced the relevant filings."""
    counts = defaultdict(lambda: [0, 0])  # origin -> [found, judgments]
    for row in judgments:
        origin = origins.get(row["query_id"])
        if origin is None:
            origin = "amplify" if str(row.get("via")) == "amplify" else "gen0"
        counts[origin][0] += bool(row["relevant"])
        counts[origin][1] += 1
    return counts


def report_branches(queries):
    """Yield per branch, which is the 'is this branch pulling its weight' number."""
    stats = defaultdict(lambda: [0, 0, 0])  # branch -> [new_found, judgments, queries]
    for row in queries:
        branch = row["branch"]
        stats[branch][0] += row["new_found"]
        stats[branch][1] += row["judgments_spent"]
        stats[branch][2] += 1
    return stats


def print_trace(queries, generations, limit_per_gen=None):
    """The generation-by-generation story for slide 9."""
    by_gen = defaultdict(list)
    for row in queries:
        by_gen[row["generation"]].append(row)
    elites_of = {g["generation"]: set(g.get("elites") or []) for g in generations}

    for generation in sorted(by_gen):
        summary = next((g for g in generations if g["generation"] == generation), {})
        header = (f"GENERATION {generation}  "
                  f"{summary.get('judgments_this_generation', 0)} judgments, "
                  f"+{summary.get('new_found_this_generation', 0)} new, "
                  f"{summary.get('found_cumulative', 0)} found so far")
        print("\n" + header)
        print("-" * len(header))
        rows = [r for r in by_gen[generation] if r["origin"] != "amplify"]
        rows.sort(key=lambda r: (-r["new_found"], r["query_id"]))
        for row in rows[:limit_per_gen]:
            kept = row["query_id"] in elites_of.get(generation, set())
            tag = "NEW " if row["origin"] == "bred" else ("keep" if kept else "    ")
            print(f"  {tag} [{row['branch']:9}] {row['new_found']:2d} new / "
                  f"{row['judgments_spent']:2d} judged   {row['query'][:72]!r}")
            if row["origin"] == "bred" and row.get("rationale"):
                print(f"           why: {row['rationale'][:88]}")
        amp = summary.get("amplify")
        if amp and amp.get("judgments_spent"):
            words = ", ".join(f"[{w['branch']}] {w['query'][:34]!r}"
                              for w in (amp.get("word_queries") or []))
            print(f"  amplify  {amp['new_found']} new / {amp['judgments_spent']} judged"
                  f"   ({amp.get('by_vector_seeds', 0)} vector seeds)")
            if words:
                print(f"           words: {words}")


def analyse(run_dir, args):
    judgments = read_jsonl(os.path.join(run_dir, "judgments.jsonl"))
    queries = read_jsonl(os.path.join(run_dir, "queries.jsonl"))
    generations = read_jsonl(os.path.join(run_dir, "generations.jsonl"))
    relevant_total, meta = total_relevant(run_dir, args)
    if not judgments:
        print(f"{run_dir}: no judgments.jsonl, skipping")
        return None

    found = sum(bool(r["relevant"]) for r in judgments)
    name = os.path.basename(run_dir.rstrip("/"))
    print("=" * 78)
    print(f"{name}   {meta.get('arm', 'baseline')}   breeder={meta.get('breeder', '-')}")
    print("=" * 78)
    print(f"  {len(judgments)} judgments, {found} of {relevant_total} relevant found "
          f"(recall {found / relevant_total:.3f})" if relevant_total else
          f"  {len(judgments)} judgments, {found} found")

    curve = write_curve(run_dir, judgments, relevant_total)
    print(f"  curve written to {curve}")

    if queries:
        print("\n  WHERE THE FINDS CAME FROM")
        sources = report_sources(judgments, origin_of(queries))
        for origin in sorted(sources, key=lambda o: -sources[o][0]):
            hits, spent = sources[origin]
            share = 100 * hits / found if found else 0
            print(f"    {origin:9} {hits:3d} finds ({share:4.1f}%) in {spent:4d} judgments"
                  f"   {100 * hits / spent if spent else 0:5.1f} per 100")

        print("\n  YIELD PER BRANCH")
        for branch, (new_found, spent, n) in sorted(report_branches(queries).items()):
            print(f"    {branch:9} {new_found:3d} finds in {spent:4d} judgments across "
                  f"{n:2d} query-runs   {100 * new_found / spent if spent else 0:5.1f} per 100")

        if not args.no_trace:
            print("\n  QUERY TRACE")
            print_trace(queries, generations, args.limit)

    return {"name": name, "judgments": judgments, "relevant_total": relevant_total,
            "found": found}


def compare(runs):
    print("\n" + "=" * 78)
    print("RECALL AT MATCHED JUDGMENT BUDGETS")
    print("=" * 78)
    shortest = min(len(r["judgments"]) for r in runs)
    # Always include the largest budget every run actually reached, so a short run
    # still gets compared at its own end point rather than only at round numbers.
    checkpoints = sorted({c for c in CHECKPOINTS if c <= shortest} | {shortest})
    if not checkpoints:
        print("  runs do not overlap in budget; nothing to compare")
        return
    width = max(len(r["name"]) for r in runs) + 2
    print(f"  {'judgments':>10} " + "".join(f"{r['name']:>{width}}" for r in runs))
    tables = [recall_at(r["judgments"], r["relevant_total"], checkpoints) for r in runs]
    for n in checkpoints:
        cells = []
        for table in tables:
            hits, recall = table.get(n, (0, 0.0))
            cells.append(f"{hits:3d} ({recall:.3f})".rjust(width))
        print(f"  {n:>10} " + "".join(cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--config", default="config.json")
    p.add_argument("--labels")
    p.add_argument("--question")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--limit", type=int, help="Show only the top N queries per generation.")
    p.add_argument("--no-trace", action="store_true")
    args = p.parse_args()

    runs = [r for r in (analyse(d, args) for d in args.run_dirs) if r]
    if len(runs) > 1:
        compare(runs)


if __name__ == "__main__":
    main()
