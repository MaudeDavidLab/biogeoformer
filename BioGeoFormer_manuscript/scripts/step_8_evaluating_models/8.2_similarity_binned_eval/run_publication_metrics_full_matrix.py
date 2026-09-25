"""Accuracy, MCC and weighted F1/precision/recall per split for BGF, HMM and DIAMOND, each
prediction being the argmax of the method's full per-class score matrix (no hit counts
as wrong). Writes metrics_full_matrix.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "8.1_perclass_pr"))
from run_diamond_perclass_micro_pr import (
    load_split_matrix as diamond_load_split_matrix,
    patch_matrix_with_combined_top1,
    sanitize,
)
from run_hmm_perclass_micro_pr import load_split_matrix as hmm_load_split_matrix
from run_method_comparison import build_comparison_frame, load_hmm_all
from run_evaluation import PLOTS_DIR, SPLITS, METRICS, METRIC_TITLES, compute_metrics

NO_HIT = "NO_HIT"
COLORS = {"bgf": "#0072b2", "hmm": "#009e73", "diamond": "#e66100"}
METHOD_LABELS = {"bgf": "BGF", "hmm": "HMM", "diamond": "DIAMOND"}


def _panel_title(metric):
    return METRIC_TITLES[metric].split(" (")[0]


def reconstruct_predictions(query_ids, classes, matrix, full_ids):
    """argmax per query -> {plain id: predicted class}, NO_HIT for any
    full-population id absent from query_ids entirely."""
    pred = {}
    argmax_idx = matrix.argmax(axis=1)
    classes_arr = np.array(classes)
    for qid, idx in zip(query_ids, argmax_idx):
        plain_id = qid.split("|")[0]
        pred[plain_id] = classes_arr[idx]
    return pd.Series({fid: pred.get(fid, NO_HIT) for fid in full_ids})


def compute_all_metrics(split: int, hmm_all) -> dict:
    df = build_comparison_frame(split, hmm_all).set_index("id")
    true_label = df["true_label"].map(sanitize)
    full_ids = df.index

    bgf_pred = df["predicted_label_bgf"].map(sanitize)

    d_query_ids, d_classes, d_matrix, d_evalue_matrix, d_true_label = diamond_load_split_matrix(split)
    d_matrix, d_evalue_matrix = patch_matrix_with_combined_top1(split, d_query_ids, d_classes, d_matrix, d_evalue_matrix)
    diamond_pred = reconstruct_predictions(d_query_ids, d_classes, d_matrix, full_ids)

    h_query_ids, h_classes, h_matrix, h_evalue_matrix, h_true_label = hmm_load_split_matrix(split)
    hmm_pred = reconstruct_predictions(h_query_ids, h_classes, h_matrix, full_ids)

    rows = {}
    for method, pred in [("bgf", bgf_pred), ("hmm", hmm_pred), ("diamond", diamond_pred)]:
        m = compute_metrics(true_label, pred.reindex(full_ids))
        rows[method] = m
    return rows


def draw_metric_panel(ax, df, metric, show_ylabel):
    split_labels = [f"{s}%" for s in SPLITS]
    for method, color in COLORS.items():
        m_df = df[df["method"] == method].set_index("split").reindex(SPLITS)
        ax.plot(
            split_labels, m_df[metric], marker="o", markersize=6, linewidth=2,
            color=color, label=METHOD_LABELS[method], solid_capstyle="round",
        )
    ax.set_title(_panel_title(metric), fontsize=10, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if show_ylabel:
        ax.set_ylabel("Score", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_n_panel(ax, df):
    split_labels = [f"{s}%" for s in SPLITS]
    x = np.arange(len(SPLITS))
    width = 0.25
    for i, (method, color) in enumerate(COLORS.items()):
        m_df = df[df["method"] == method].set_index("split").reindex(SPLITS)
        ax.bar(x + (i - 1) * width, m_df["n"], width=width, color=color, label=METHOD_LABELS[method])
    ax.set_title("n (sequences)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, df["n"].max() * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, rotation=45, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_figure(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 6, figsize=(20, 4.5))

    handles_labels = None
    for col, metric in enumerate(METRICS):
        draw_metric_panel(axes[col], df, metric, show_ylabel=(col == 0))
        if handles_labels is None:
            handles_labels = axes[col].get_legend_handles_labels()

    draw_n_panel(axes[5], df)
    fig.subplots_adjust(wspace=0.35)

    fig.legend(*handles_labels, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.08), frameon=False)
    return fig


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading HMM prediction detail (one-time)...")
    hmm_all = load_hmm_all()

    rows = []
    for split in SPLITS:
        print(f"\nsplit {split}:")
        split_metrics = compute_all_metrics(split, hmm_all)
        for method, m in split_metrics.items():
            rows.append({"split": split, "method": method, **m})
            print(f"  {method}: accuracy={m['accuracy']:.3f} MCC={m['MCC']:.3f} "
                  f"F1={m['F1']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f} n={m['n']}")

    df = pd.DataFrame(rows)
    csv_path = PLOTS_DIR.parent / "metrics_full_matrix.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    fig = build_figure(df)
    for ext in ("png", "pdf", "svg"):
        out_path = PLOTS_DIR / f"publication_methods_comparison_full_matrix.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)
