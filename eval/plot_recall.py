"""Draw recall against judgments spent, for the report's Figure 1.

    .venv/bin/python eval/plot_recall.py

Reads judgments.jsonl straight from the run directories, so this figure and the
tables in eval/RESULTS*.md come from the same logs. Run it from the repo root.

The two evolution arms come from different commits: the Sonnet runs are v4
(dd4d793), the Haiku runs predate two amplify fixes. A matched pair of Sonnet
runs across that change differed by 0.6 filings (p=0.83), which is why they are
drawn together here, but it is why report.py refuses to average them.
"""
import argparse
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RELEVANT, BUDGET = 129, 1000
SONNET = [f"v4_sonnet_s400{i}" for i in range(1, 6)]
HAIKU = ["evolution_1000"] + [f"haiku_s202{i}" for i in range(1, 6)]

# Chart surface and ink. Series hues are slots 1-3 of the default categorical
# palette, validated for colour-vision deficiency across all pairs; the aqua
# sits below 3:1 against this surface, which the table in the report relieves.
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
G1, G2, G3, GRID = "#8a8a85", "#a8a8a1", "#c2c2bb", "#e6e6e1"


def series(runs_dir, run):
    """Cumulative relevant found after each judgment, carried forward if the run
    stopped early -- the exact-phrase baseline stops after 13."""
    path = os.path.join(runs_dir, run, "judgments.jsonl")
    rows = sorted((json.loads(line) for line in open(path)), key=lambda r: r["seq"])
    found, by_seq = 0, {}
    for row in rows:
        found += bool(row["relevant"])
        by_seq[row["seq"]] = found
    out, last = [], 0
    for n in range(BUDGET + 1):
        last = by_seq.get(n, last)
        out.append(last)
    return out


def mean_band(runs_dir, runs):
    cs = [series(runs_dir, r) for r in runs]
    return ([st.mean(c[n] for c in cs) for n in range(BUDGET + 1)],
            [min(c[n] for c in cs) for n in range(BUDGET + 1)],
            [max(c[n] for c in cs) for n in range(BUDGET + 1)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="eval/runs")
    p.add_argument("--out", default="reports/figures/recall_vs_budget.png")
    args = p.parse_args()

    sonnet, lo, hi = mean_band(args.runs_dir, SONNET)
    haiku, _, _ = mean_band(args.runs_dir, HAIKU)
    base = {k: series(args.runs_dir, f"baseline_{k}")
            for k in ("grep", "faiss", "bm25", "hybrid")}

    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = range(BUDGET + 1)

    ax.axhline(RELEVANT, color=GRID, lw=1.2, ls=(0, (4, 4)), zorder=1)
    ax.text(BUDGET - 6, RELEVANT + 2.0, f"all {RELEVANT} relevant", fontsize=7.5,
            color=INK2, va="bottom", ha="right")

    ax.fill_between(x, lo, hi, color=BLUE, alpha=0.11, lw=0, zorder=2)
    ax.plot(x, base["grep"], color=G3, lw=1.4, ls=(0, (1, 2.5)),
            label="Baseline: exact phrase", zorder=3)
    ax.plot(x, base["faiss"], color=G2, lw=1.4, ls=(0, (5, 3)),
            label="Baseline: embeddings", zorder=4)
    ax.plot(x, base["bm25"], color=G1, lw=1.5, label="Baseline: BM25", zorder=5)
    ax.plot(x, base["hybrid"], color=ORANGE, lw=2.0,
            label="Baseline: hybrid (all three)", zorder=6)
    ax.plot(x, haiku, color=AQUA, lw=2.0, ls=(0, (6, 2.5)),
            label=f"PokéSearch · Haiku breeder (mean of {len(HAIKU)})", zorder=7)
    ax.plot(x, sonnet, color=BLUE, lw=2.4,
            label=f"PokéSearch · Sonnet breeder (mean of {len(SONNET)}, band = range)",
            zorder=8)

    ax.set_xlim(0, BUDGET)
    ax.set_ylim(0, 136)
    ax.set_xlabel("judgments spent", fontsize=9, color=INK2, labelpad=7)
    ax.set_ylabel("relevant filings found", fontsize=9, color=INK2, labelpad=7)
    ax.set_xticks(range(0, BUDGET + 1, 200))
    ax.set_yticks(range(0, 131, 25))
    ax.tick_params(colors=INK2, labelsize=8.5, length=0, pad=5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    leg = ax.legend(loc="lower right", fontsize=8, frameon=False, labelspacing=0.5,
                    handlelength=2.4, borderpad=0, bbox_to_anchor=(0.995, 0.015))
    for t in leg.get_texts():
        t.set_color(INK)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"at {BUDGET}: sonnet {sonnet[-1]:.1f}, haiku {haiku[-1]:.1f}, "
          + ", ".join(f"{k} {v[-1]}" for k, v in base.items()))


if __name__ == "__main__":
    main()
