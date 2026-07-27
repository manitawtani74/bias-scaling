"""Plot BBQ bias/accuracy scaling across Qwen2.5 sizes from the metrics CSVs.

Reads results/qwen{tag}_{mode}_cuda_metrics.csv for each size and plots the
overall `accuracy_disambig`, `bias_score_disambig`, and `bias_score_ambig`
against model size on a log x-axis.

Missing sizes are skipped rather than erroring, so:
  - the letterperm figure uses whatever of 0.5B/1.5B/3B/7B are present, and
  - the text figure fills in the 7B point automatically once its CSV lands.

Run from anywhere:  python scripts/plot_scaling.py
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
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


def make_figure(mode, out_png):
    xs, rows, labels = [], [], []
    for tag, label, x in SIZES:
        path = os.path.join(RESULTS, f"qwen{tag}_{mode}_cuda_metrics.csv")
        if not os.path.exists(path):
            print(f"[{mode}] {label}: metrics CSV absent, skipping")
            continue
        xs.append(x)
        rows.append(overall_row(path))
        labels.append(label)

    if not xs:
        print(f"[{mode}] no metrics CSVs found; not writing {out_png}")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, name, marker, color in SERIES:
        ys = [float(r[col]) for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=2, label=name)
    ax.axhline(0, color="0.7", linewidth=0.8, zorder=0)

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.minorticks_off()
    ax.set_xlabel("Qwen2.5 model size (log scale)")
    ax.set_ylabel("score")

    per_cat = int(rows[0]["n_items"]) // 11  # 11 BBQ categories
    ax.set_title(f"BBQ scaling — {mode} scoring (sample={per_cat}/category, seed 0)")
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
