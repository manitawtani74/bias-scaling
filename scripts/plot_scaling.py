"""Plot BBQ bias/accuracy scaling across Qwen2.5 sizes from the metrics CSVs.

For each size and mode the script aggregates over seeds: it reads every
`results/qwen{tag}_{mode}_cuda_seed*_metrics.csv` and plots the mean of
`accuracy_disambig`, `bias_score_disambig`, and `bias_score_ambig` with error
bars (± sample standard deviation across seeds). If no seed-tagged files exist
yet for a size+mode it falls back to the legacy single-seed
`qwen{tag}_{mode}_cuda_metrics.csv`, and with a single seed the error bar is
simply zero-length.

Missing sizes are skipped rather than erroring, so the text figure fills in the
7B point automatically once its CSV lands.

Run from anywhere:  python scripts/plot_scaling.py
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))

# (filename tag, display label, size in billions of params for the log x-axis)
SIZES = [("05b", "0.5B", 0.5), ("15b", "1.5B", 1.5), ("3b", "3B", 3.0), ("7b", "7B", 7.0)]

# (metrics column, legend label, marker, colour)
SERIES = [
    ("accuracy_disambig", "Accuracy (disambiguated)", "o", "#1b9e77"),
    ("bias_score_disambig", "Bias (disambiguated)", "s", "#7570b3"),
    ("bias_score_ambig", "Bias (ambiguous)", "^", "#d95f02"),
]


def overall_row(path):
    df = pd.read_csv(path)
    return df[df["group"] == "overall"].iloc[0]


def seed_metrics_paths(mode, tag):
    """Metrics CSVs for a size+mode.

    Prefer the seed-tagged files (one per seed); fall back to the legacy
    non-seed file if no seed-tagged ones exist, so plots still render before
    the multi-seed sweep has run.
    """
    seeded = sorted(
        glob.glob(os.path.join(RESULTS, f"qwen{tag}_{mode}_cuda_seed*_metrics.csv"))
    )
    if seeded:
        return seeded
    legacy = os.path.join(RESULTS, f"qwen{tag}_{mode}_cuda_metrics.csv")
    return [legacy] if os.path.exists(legacy) else []


def collect(mode):
    """Per present size: (x, label, {col: (mean, std)}, n_seeds, n_items)."""
    points = []
    for tag, label, x in SIZES:
        paths = seed_metrics_paths(mode, tag)
        if not paths:
            print(f"[{mode}] {label}: no metrics CSV, skipping")
            continue
        rows = [overall_row(p) for p in paths]
        stats = {}
        for col, *_ in SERIES:
            vals = np.array([float(r[col]) for r in rows], dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            stats[col] = (mean, std)
        n_seeds = len(paths)
        print(f"[{mode}] {label}: {n_seeds} seed(s)")
        points.append((x, label, stats, n_seeds, int(rows[0]["n_items"])))
    return points


def _seed_note(points):
    counts = sorted({n for _, _, _, n, _ in points})
    if counts == [1]:
        return "single seed"
    if len(counts) == 1:
        return f"mean ± SD, {counts[0]} seeds"
    return f"mean ± SD, {min(counts)}–{max(counts)} seeds"


def make_figure(mode, out_png):
    points = collect(mode)
    if not points:
        print(f"[{mode}] no metrics CSVs found; not writing {out_png}")
        return False

    xs = [p[0] for p in points]
    labels = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, name, marker, color in SERIES:
        means = [p[2][col][0] for p in points]
        stds = [p[2][col][1] for p in points]
        ax.errorbar(
            xs, means, yerr=stds, marker=marker, color=color, label=name,
            linewidth=2, capsize=3,
        )
    ax.axhline(0, color="0.7", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.set_xlabel("Qwen2.5 model size (log scale)")
    ax.set_ylabel("score")

    per_cat = points[0][4] // 11  # 11 BBQ categories
    ax.set_title(
        f"BBQ scaling — {mode} scoring (sample={per_cat}/category, {_seed_note(points)})"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[{mode}] wrote {out_png} from sizes {labels}")
    return True


def main():
    make_figure("letterperm", os.path.join(RESULTS, "scaling_letterperm.png"))
    make_figure("text", os.path.join(RESULTS, "scaling_text.png"))


if __name__ == "__main__":
    main()
