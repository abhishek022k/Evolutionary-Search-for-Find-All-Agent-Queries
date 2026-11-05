"""Run the evolution arm and write a run directory the evaluation can read.

Full run:    .venv/bin/python run_evolution.py --budget 1000
Offline test: .venv/bin/python run_evolution.py --pilot --stub --budget 60 \
                  --labels data/processed/fake_labels_pilot.jsonl --question fake_question \
                  --gen0 data/processed/gen0_pilot.json

--stub uses the deterministic breeder: no API key, no spend.

Every run records the git commit, the sha256 of the filings and labels it used, and the
full config, so any number on a slide traces back to the exact inputs that produced it.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from search import SearchEngine  # noqa: F401  (must precede torch; see embedders.py)
from judge import CostTracker, load_labels
from breeder import make_breeder
from evolve import EvolutionRun

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from build_index import file_hash  # noqa: E402
from questions import QUESTIONS  # noqa: E402

GEN0_PATH = "data/processed/gen0_queries.json"
PILOT_IDS_PATH = "data/processed/pilot_ids.txt"


def question_text(question_id):
    """The task definition the Generation 0 author worked from. Not the answer key."""
    q = QUESTIONS.get(question_id)
    if q is None:
        # A typo here would breed against a meaningless prompt and still look fine
        # in the logs, so make it loud rather than silent.
        print(f"WARNING: {question_id!r} is not in questions.py. The breeder will see "
              f"only the bare id, with no definition of what counts as relevant. "
              f"Known: {', '.join(sorted(QUESTIONS))}")
        return question_id
    return (f"{q['question']}\n\n"
            f"Counts as YES when: {q['yes']}\n\n"
            f"Counts as NO when: {q['no']}")


def load_env(path=".env"):
    """Same .env handling as label.py. See .env.example for the names."""
    if not os.path.exists(path):
        return
    for line in open(path):
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "GEMINI_API_KEY"):
            if line.startswith(name + "="):
                os.environ[name] = line.split("=", 1)[1].strip()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--budget", type=int)
    p.add_argument("--gen0", default=GEN0_PATH)
    p.add_argument("--labels")
    p.add_argument("--question")
    p.add_argument("--pilot", action="store_true", help="Use index/pilot and the pilot filings.")
    p.add_argument("--stub", action="store_true", help="Deterministic breeder, no API calls.")
    p.add_argument("--max-generations", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--judge-depth", type=int, help="Override judge_depth, for tests and sweeps.")
    p.add_argument("--amplify-cap", type=int, help="Override amplify_cap.")
    p.add_argument("--run-id")
    args = p.parse_args()

    load_env()
    if not args.stub and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Put it in .env (see "
                         ".env.example), or pass --stub to run without a breeder.")

    cfg = json.load(open(args.config))
    if args.judge_depth is not None:
        cfg["judge_depth"] = args.judge_depth
    if args.amplify_cap is not None:
        cfg["amplify_cap"] = args.amplify_cap
    labels_path = args.labels or cfg["labels_path"]
    question = args.question or cfg["question_id"]
    if not labels_path or not question:
        raise SystemExit("Set labels_path and question_id in config.json, or pass "
                         "--labels and --question.")
    if not os.path.exists(args.gen0):
        raise SystemExit(f"{args.gen0} is missing. It is a JSON list of "
                         '{"query_id", "branch", "query"}.')

    subset = set(open(PILOT_IDS_PATH).read().split()) if args.pilot else None
    index_dir = "index/pilot" if args.pilot else cfg["index_dir"]
    engine = SearchEngine(cfg, index_dir=index_dir, subset_ids=subset)

    filing_ids = [json.loads(line)["filing_id"] for line in open(cfg["filings_path"])]
    if subset is not None:
        filing_ids = [fid for fid in filing_ids if fid in subset]

    labels = load_labels(labels_path, question)
    gen0 = json.load(open(args.gen0))
    budget = args.budget if args.budget is not None else cfg["judgment_budget"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)

    run_id = args.run_id or f"evolution_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join("eval", "runs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    cost = CostTracker()
    breeder = make_breeder(cfg, cost, stub=args.stub)

    meta = {
        "run_id": run_id, "arm": "evolution", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(), "seed": seed, "budget": budget,
        "question_id": question, "labels_path": labels_path,
        "labels_sha256": file_hash(labels_path),
        "filings_path": cfg["filings_path"], "filings_sha256": file_hash(cfg["filings_path"]),
        "gen0_path": args.gen0, "gen0_sha256": file_hash(args.gen0),
        "population_size": len(gen0), "pilot": args.pilot, "index_dir": index_dir,
        "breeder": breeder.name, "stub_breeder": args.stub,
        "filings_in_scope": len(filing_ids),
        "relevant_in_scope": sum(bool(labels.get(f)) for f in filing_ids),
        "config": cfg,
    }
    with open(os.path.join(log_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{run_id}: {len(filing_ids)} filings, {meta['relevant_in_scope']} relevant, "
          f"budget {budget}, population {len(gen0)}, breeder {breeder.name}")

    run = EvolutionRun(engine, labels, filing_ids, cfg, breeder, log_dir, budget,
                       question_text(question), seed=seed)
    try:
        run.run(gen0, max_generations=args.max_generations)
    finally:
        run.finish()

    found = len(run.ledger.found())
    in_scope = meta["relevant_in_scope"]
    summary = {
        "generations": len(run.generations),
        "judgments_used": run.judge.used,
        "found": found,
        "relevant_in_scope": in_scope,
        "recall": (found / in_scope) if in_scope else 0.0,
        "ledger": run.ledger.counts(),
        "breeding_cost": cost.summary(),
        "breeder_fallbacks": getattr(breeder, "fallbacks_used", 0),
    }
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{run_id}: {run.judge.used}/{budget} judgments over {len(run.generations)} "
          f"generations, {found} found of {in_scope} (recall {summary['recall']:.3f})")
    print(f"breeding: {cost.summary()['calls']} calls, "
          f"${cost.summary()['dollars']:.4f}, {summary['breeder_fallbacks']} fallbacks")
    print(f"logs in {log_dir}")


if __name__ == "__main__":
    main()
