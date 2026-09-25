"""As run_publication_metrics_full_matrix.py, but predictions below each method's confidence
threshold (BGF per-split threshold; HMM/DIAMOND E-value > 1e-5) become NO_HIT.
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
from run_evaluation import PLOTS_DIR, SPLITS, METRICS, METRIC_TITLES, compute_metrics, load_thresholds

NO_HIT = "NO_HIT"
CONFIDENT_EVALUE = 1e-5
COLORS = {"bgf": "#0072b2", "hmm": "#009e73", "diamond": "#e66100"}
METHOD_LABELS = {"bgf": "BGF", "hmm": "HMM", "diamond": "DIAMOND"}


def _panel_title(metric):
    return METRIC_TITLES[metric].split(" (")[0]


def reconstruct_predictions_thresholded(query_ids, classes, matrix, evalue_matrix, full_ids, evalue_threshold):
    """Argmax prediction per query; NO_HIT if its E-value fails the threshold or the query has no hits."""
    argmax_idx = matrix.argmax(axis=1)
    classes_arr = np.array(classes)
    rows = np.arange(len(query_ids))
    confident = evalue_matrix[rows, argmax_idx] <= evalue_threshold

    pred = {}
    for qid, idx, conf in zip(query_ids, argmax_idx, confident):
        plain_id = qid.split("|")[0]
        pred[plain_id] = classes_arr[idx] if conf else NO_HIT
    return pd.Series({fid: pred.get(fid, NO_HIT) for fid in full_ids})


def compute_all_metrics(split: int, hmm_all, bgf_threshold: float) -> dict:
    df = build_comparison_frame(split, hmm_all).set_index("id")
    true_label = df["true_label"].map(sanitize)
    full_ids = df.index

    bgf_raw_pred = df["predicted_label_bgf"].map(sanitize)
    bgf_confident = df["confidence_bgf"] >= bgf_threshold
    bgf_pred = bgf_raw_pred.where(bgf_confident, NO_HIT)

    d_query_ids, d_classes, d_matrix, d_evalue_matrix, d_true_label = diamond_load_split_matrix(split)
    d_matrix, d_evalue_matrix = patch_matrix_with_combined_top1(split, d_query_ids, d_classes, d_matrix, d_evalue_matrix)
    diamond_pred = reconstruct_predictions_thresholded(
        d_query_ids, d_classes, d_matrix, d_evalue_matrix, full_ids, CONFIDENT_EVALUE
    )

    h_query_ids, h_classes, h_matrix, h_evalue_matrix, h_true_label = hmm_load_split_matrix(split)
    hmm_pred = reconstruct_predictions_thresholded(
        h_query_ids, h_classes, h_matrix, h_evalue_matrix, full_ids, CONFIDENT_EVALUE
    )

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
    bgf_thresholds = load_thresholds()

    rows = []
    for split in SPLITS:
        print(f"\nsplit {split}:")
        split_metrics = compute_all_metrics(split, hmm_all, bgf_thresholds[split])
        for method, m in split_metrics.items():
            rows.append({"split": split, "method": method, **m})
            print(f"  {method}: accuracy={m['accuracy']:.3f} MCC={m['MCC']:.3f} "
                  f"F1={m['F1']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f} n={m['n']}")

    df = pd.DataFrame(rows)
    csv_path = PLOTS_DIR.parent / "metrics_full_matrix_thresholded.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    fig = build_figure(df)
    for ext in ("png", "pdf", "svg"):
        out_path = PLOTS_DIR / f"publication_methods_comparison_full_matrix_thresholded.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)
