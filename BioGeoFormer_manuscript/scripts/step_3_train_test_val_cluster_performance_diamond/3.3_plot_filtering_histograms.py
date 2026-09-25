# Histograms of each test/val sequence's DIAMOND identity to train, stacked by outcome:
# kept, removed by the pathway rule (3.2), or removed as leakage (3.1).
# "removed_mmseqs" in the input is the DIAMOND leakage removal (legacy label).

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = REPO_ROOT / "BGF_clustering" / "cleaned"
OUT_DIR = REPO_ROOT / "results" / "figures" / "filtering_histograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [10, 20, 30, 40, 50, 60, 70, 80, 90]

COLORS = {"kept": "#009e73", "removed_pathway": "#e69f00", "removed_mmseqs": "#999999"}
LABELS = {"kept": "Kept", "removed_pathway": "Removed (pathway rule)",
          "removed_mmseqs": "Removed (DIAMOND leakage)"}
STATUS_ORDER = ["kept", "removed_pathway", "removed_mmseqs"]

BIN_WIDTH = 5
bin_edges = np.arange(0, 100 + BIN_WIDTH, BIN_WIDTH)
bin_centers = bin_edges[:-1] + BIN_WIDTH / 2
NOHIT_X = -7.5


def make_plot(csv_name, set_label, out_name):
    df = pd.read_csv(BASE / csv_name)
    fig, axes = plt.subplots(3, 3, figsize=(17, 12.5))

    for ax, t in zip(axes.flat, SPLITS):
        sub = df[df["threshold"] == t]
        hit_sub = sub[sub["has_hit"]]
        nohit_sub = sub[~sub["has_hit"]]

        bottom = np.zeros(len(bin_centers))
        for status in STATUS_ORDER:
            vals = hit_sub[hit_sub["status"] == status]["pident"].values
            counts, _ = np.histogram(vals, bins=bin_edges)
            ax.bar(bin_centers, counts, width=BIN_WIDTH * 0.95, bottom=bottom,
                   color=COLORS[status], align="center")
            bottom += counts

        nohit_bottom = 0
        for status in STATUS_ORDER:
            n = (nohit_sub["status"] == status).sum()
            if n > 0:
                ax.bar(NOHIT_X, n, width=BIN_WIDTH * 0.7, bottom=nohit_bottom,
                       color=COLORS[status], align="center")
                nohit_bottom += n

        # explicit headroom so the tallest stacked bar is never flush with / clipped by the panel top
        ax.set_ylim(0, max(bottom.max(), nohit_bottom) * 1.08)
        ax.axvline(-BIN_WIDTH / 2, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(t, color="black", linestyle="--", alpha=0.7, linewidth=1.2)
        ax.set_xlim(NOHIT_X - BIN_WIDTH, 100)
        ax.set_xticks([NOHIT_X, 0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(["no hit", "0", "20", "40", "60", "80", "100"], fontsize=8)
        ax.set_title(f"split {t}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("DIAMOND %identity to train (top hit)")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)

    handles = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in STATUS_ORDER]
    # legend nudged above the top edge (y > 1) so it clears the title
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.975),
               fontsize=10, ncol=3, frameon=False)
    fig.suptitle(f"DIAMOND %identity to train, by filtering outcome ({set_label})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_name)


make_plot("similarity_status_3way.csv", "test set", "similarity_histograms_before_after_with_nohit.png")
make_plot("similarity_status_3way_val.csv", "validation set", "similarity_histograms_before_after_with_nohit_val.png")
