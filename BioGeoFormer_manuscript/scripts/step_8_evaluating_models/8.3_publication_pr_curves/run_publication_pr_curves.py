"""Micro and macro precision-recall curves for BGF, HMM and DIAMOND per split, AUC by split,
and covered-class precision. Writes plots/, pr_curve_auc_micro_macro.csv and
precision_covered.csv in this folder.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

STEP8_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STEP8_DIR / "8.1_perclass_pr"))
sys.path.insert(0, str(STEP8_DIR / "8.2_similarity_binned_eval"))
from run_diamond_perclass_micro_pr import (
    load_split_matrix as diamond_load_split_matrix,
    patch_matrix_with_combined_top1,
    micro_pr_curve as diamond_micro_pr_curve,
    macro_pr_curve as diamond_macro_pr_curve,
    evalue_operating_point as diamond_evalue_operating_point,
)
from run_hmm_perclass_micro_pr import (
    load_split_matrix as hmm_load_split_matrix,
    micro_pr_curve as hmm_micro_pr_curve,
    macro_pr_curve as hmm_macro_pr_curve,
    evalue_operating_point as hmm_evalue_operating_point,
)
from run_bgf_perclass_micro_pr import (
    micro_pr_curve as bgf_micro_pr_curve,
    macro_pr_curve as bgf_macro_pr_curve,
    prob_operating_point as bgf_prob_operating_point,
)
from run_publication_metrics_full_matrix_thresholded import (
    reconstruct_predictions_thresholded, sanitize, NO_HIT, CONFIDENT_EVALUE,
)
from run_publication_precision_covered import covered_class_precision
from run_method_comparison import build_comparison_frame, load_hmm_all
from run_evaluation import SPLITS, load_thresholds

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR = Path(__file__).resolve().parent

COLORS = {"bgf": "#0072b2", "hmm": "#009e73", "diamond": "#e66100"}
METHOD_LABELS = {"bgf": "BGF", "hmm": "HMM", "diamond": "DIAMOND"}


def plot_curve_grid(all_data: dict, operating_points, out_stem: str):
    """operating_points=None draws no operating-point dots (used for the macro grid)."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
    handles_labels = None

    for ax, split in zip(axes.flat, SPLITS):
        for method, color in COLORS.items():
            recall, precision = all_data[split][method]
            ax.plot(recall, precision, color=color, linewidth=2, label=METHOD_LABELS[method])
            if operating_points is not None:
                op_recall, op_precision = operating_points[split][method]
                ax.scatter([op_recall], [op_precision], color=color, s=60, zorder=5,
                           edgecolor="black", linewidth=0.8)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
        ax.set_title(f"{split}% split", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Recall", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Precision", fontsize=9)

    fig.legend(*handles_labels, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.03), frameon=False)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out_path = PLOTS_DIR / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def plot_auc_trend(auc_df: pd.DataFrame, value_col: str, out_stem: str):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    split_labels = [f"{s}%" for s in SPLITS]
    for method, color in COLORS.items():
        m_df = auc_df[auc_df["method"] == method].set_index("split").reindex(SPLITS)
        ax.plot(split_labels, m_df[value_col], marker="o", markersize=6, linewidth=2,
                 color=color, label=METHOD_LABELS[method], solid_capstyle="round")
    ax.set_xlabel("Graph-part split")
    ax.set_ylabel("Area under precision-recall curve")
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out_path = PLOTS_DIR / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def plot_precision_covered(df: pd.DataFrame, value_col: str, ylabel: str, out_stem: str):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    split_labels = [f"{s}%" for s in SPLITS]
    for method, color in COLORS.items():
        m_df = df[df["method"] == method].set_index("split").reindex(SPLITS)
        ax.plot(split_labels, m_df[value_col], marker="o", markersize=6, linewidth=2,
                 color=color, label=METHOD_LABELS[method], solid_capstyle="round")
    ax.set_xlabel("Graph-part split")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out_path = PLOTS_DIR / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading HMM prediction detail (one-time, for covered-classes precision)...")
    hmm_all = load_hmm_all()
    bgf_thresholds = load_thresholds()

    micro_grid, macro_grid = {}, {}
    operating_points = {}
    auc_rows = []
    precision_rows = []
    for split in SPLITS:
        print(f"\nsplit {split}:")

        print("  diamond...")
        d_query_ids, d_classes, d_matrix, d_evalue_matrix, d_true_label = diamond_load_split_matrix(split)
        d_matrix, d_evalue_matrix = patch_matrix_with_combined_top1(split, d_query_ids, d_classes, d_matrix, d_evalue_matrix)
        d_recall_mi, d_precision_mi, d_auc_mi = diamond_micro_pr_curve(d_matrix, d_true_label, d_classes)
        d_recall_ma, d_precision_ma, d_auc_ma = diamond_macro_pr_curve(d_matrix, d_true_label, d_classes)
        d_op = diamond_evalue_operating_point(d_evalue_matrix, d_true_label, d_classes)

        print("  hmm...")
        h_query_ids, h_classes, h_matrix, h_evalue_matrix, h_true_label = hmm_load_split_matrix(split)
        h_recall_mi, h_precision_mi, h_auc_mi = hmm_micro_pr_curve(h_matrix, h_true_label, h_classes)
        h_recall_ma, h_precision_ma, h_auc_ma = hmm_macro_pr_curve(h_matrix, h_true_label, h_classes)
        h_op = hmm_evalue_operating_point(h_evalue_matrix, h_true_label, h_classes)

        print("  bgf...")
        b_recall_mi, b_precision_mi, b_auc_mi = bgf_micro_pr_curve(split)
        b_recall_ma, b_precision_ma, b_auc_ma = bgf_macro_pr_curve(split)
        b_op = bgf_prob_operating_point(split, bgf_thresholds[split])

        micro_grid[split] = {
            "diamond": (d_recall_mi, d_precision_mi),
            "hmm": (h_recall_mi, h_precision_mi),
            "bgf": (b_recall_mi, b_precision_mi),
        }
        macro_grid[split] = {
            "diamond": (d_recall_ma, d_precision_ma),
            "hmm": (h_recall_ma, h_precision_ma),
            "bgf": (b_recall_ma, b_precision_ma),
        }
        # Macro grid gets no operating-point dots: point and curve average on different axes.
        operating_points[split] = {"diamond": d_op, "hmm": h_op, "bgf": b_op}
        auc_rows.extend([
            {"split": split, "method": "diamond", "auc_micro": d_auc_mi, "auc_macro": d_auc_ma},
            {"split": split, "method": "hmm", "auc_micro": h_auc_mi, "auc_macro": h_auc_ma},
            {"split": split, "method": "bgf", "auc_micro": b_auc_mi, "auc_macro": b_auc_ma},
        ])
        print(f"  AUC micro -- diamond={d_auc_mi:.3f}  hmm={h_auc_mi:.3f}  bgf={b_auc_mi:.3f}")
        print(f"  AUC macro -- diamond={d_auc_ma:.3f}  hmm={h_auc_ma:.3f}  bgf={b_auc_ma:.3f}")

        print("  covered-classes precision...")
        cf_df = build_comparison_frame(split, hmm_all).set_index("id")
        true_label = cf_df["true_label"].map(sanitize)
        full_ids = cf_df.index

        bgf_raw_pred = cf_df["predicted_label_bgf"].map(sanitize)
        bgf_confident = cf_df["confidence_bgf"] >= bgf_thresholds[split]
        bgf_pred = bgf_raw_pred.where(bgf_confident, NO_HIT).reindex(full_ids)

        diamond_pred = reconstruct_predictions_thresholded(
            d_query_ids, d_classes, d_matrix, d_evalue_matrix, full_ids, CONFIDENT_EVALUE
        ).reindex(full_ids)
        hmm_pred = reconstruct_predictions_thresholded(
            h_query_ids, h_classes, h_matrix, h_evalue_matrix, full_ids, CONFIDENT_EVALUE
        ).reindex(full_ids)

        for method, pred in [("bgf", bgf_pred), ("hmm", hmm_pred), ("diamond", diamond_pred)]:
            p = covered_class_precision(true_label, pred)
            precision_rows.append({
                "split": split, "method": method,
                "precision_covered": p["precision"],
                "precision_covered_macro": p["precision_macro"],
            })
        print(f"  precision (covered) weighted -- "
              f"bgf={precision_rows[-3]['precision_covered']:.3f} "
              f"hmm={precision_rows[-2]['precision_covered']:.3f} "
              f"diamond={precision_rows[-1]['precision_covered']:.3f}")

    plot_curve_grid(micro_grid, operating_points, "publication_pr_curves_micro")
    plot_curve_grid(macro_grid, None, "publication_pr_curves_macro")

    auc_df = pd.DataFrame(auc_rows)
    plot_auc_trend(auc_df, "auc_micro", "publication_pr_auc_micro")
    plot_auc_trend(auc_df, "auc_macro", "publication_pr_auc_macro")
    auc_csv_path = OUT_DIR / "pr_curve_auc_micro_macro.csv"
    auc_df.to_csv(auc_csv_path, index=False)
    print(f"\nWrote {auc_csv_path}")

    precision_df = pd.DataFrame(precision_rows)
    plot_precision_covered(
        precision_df, "precision_covered", "Precision, weighted (covered classes only)",
        "publication_precision_covered_weighted",
    )
    plot_precision_covered(
        precision_df, "precision_covered_macro", "Precision, macro (covered classes only)",
        "publication_precision_covered_macro",
    )
    precision_csv_path = OUT_DIR / "precision_covered.csv"
    precision_df.to_csv(precision_csv_path, index=False)
    print(f"Wrote {precision_csv_path}")
