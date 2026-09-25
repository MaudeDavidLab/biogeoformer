"""Three-panel figure (A-C): macro AUPRC, macro precision on covered classes, and n per split."""

# Run from this script's directory: reads pr_curve_auc_micro_macro.csv and
# precision_covered.csv from the CWD, writes plots/publication_summary_abc_macro.*

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

STEP8_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STEP8_DIR / "8.2_similarity_binned_eval"))
from run_evaluation import SPLITS

COLORS = {"bgf": "#0072b2", "hmm": "#009e73", "diamond": "#e66100"}
METHOD_LABELS = {"bgf": "BGF", "hmm": "HMM", "diamond": "DIAMOND"}

REPO_ROOT = STEP8_DIR.parent.parent
METRICS_FULL_MATRIX_CSV = REPO_ROOT / "results" / "tables" / "similarity_binned_eval" / "metrics_full_matrix.csv"


def draw_line_panel(ax, df, value_col, ylabel, label):
    split_labels = [f"{s}%" for s in SPLITS]
    for method, color in COLORS.items():
        m_df = df[df["method"] == method].set_index("split").reindex(SPLITS)
        ax.plot(split_labels, m_df[value_col], marker="o", markersize=6, linewidth=2,
                 color=color, label=METHOD_LABELS[method], solid_capstyle="round")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_n_panel(ax, df, label):
    split_labels = [f"{s}%" for s in SPLITS]
    x = np.arange(len(SPLITS))
    width = 0.25
    for i, (method, color) in enumerate(COLORS.items()):
        m_df = df[df["method"] == method].set_index("split").reindex(SPLITS)
        ax.bar(x + (i - 1) * width, m_df["n"], width=width, color=color, label=METHOD_LABELS[method])
    ax.set_ylabel("n (sequences)", fontsize=10)
    ax.set_ylim(0, df["n"].max() * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, rotation=45, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


if __name__ == "__main__":
    auc_df = pd.read_csv("pr_curve_auc_micro_macro.csv")
    precision_df = pd.read_csv("precision_covered.csv")
    n_df = pd.read_csv(METRICS_FULL_MATRIX_CSV)[["split", "method", "n"]]

    fig = plt.figure(figsize=(20, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_line_panel(ax_a, auc_df, "auc_macro", "Area under precision-recall curve (macro)", "A")
    draw_line_panel(ax_b, precision_df, "precision_covered_macro", "Precision, macro (covered classes only)", "B")
    draw_n_panel(ax_c, n_df, "C")

    handles, labels = ax_a.get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.05), frameon=False)

    for ext in ("png", "pdf", "svg"):
        out_path = f"plots/publication_summary_abc_macro.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)
