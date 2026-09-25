"""Weighted and macro precision of thresholded predictions, restricted to the classes each
method predicted at least once. Writes precision_covered.csv and figures.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "8.1_perclass_pr"))
from run_publication_metrics_full_matrix_thresholded import (
    diamond_load_split_matrix, patch_matrix_with_combined_top1,
    hmm_load_split_matrix, reconstruct_predictions_thresholded,
    sanitize, NO_HIT, CONFIDENT_EVALUE,
)
from run_method_comparison import build_comparison_frame, load_hmm_all
from run_evaluation import PLOTS_DIR, SPLITS, compute_metrics, load_thresholds

COLORS = {"bgf": "#0072b2", "hmm": "#009e73", "diamond": "#e66100"}
METHOD_LABELS = {"bgf": "BGF", "hmm": "HMM", "diamond": "DIAMOND"}


def covered_class_precision(true_label: pd.Series, pred: pd.Series) -> dict:
    """Weighted and macro precision over covered classes only."""
    covered = sorted(pred[pred != NO_HIT].unique())
    if not covered:
        return {"precision": 0.0, "precision_macro": 0.0}
    mask = true_label.isin(covered)
    m = compute_metrics(true_label[mask], pred[mask])
    return {"precision": m["precision"], "precision_macro": m["precision_macro"]}


def compute_predictions(split: int, hmm_all, bgf_threshold: float):
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

    return true_label, {"bgf": bgf_pred.reindex(full_ids), "hmm": hmm_pred.reindex(full_ids), "diamond": diamond_pred.reindex(full_ids)}


def build_figure(df: pd.DataFrame, value_col: str, ylabel: str):
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
    return fig


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading HMM prediction detail (one-time)...")
    hmm_all = load_hmm_all()
    bgf_thresholds = load_thresholds()

    rows = []
    for split in SPLITS:
        print(f"\nsplit {split}:")
        true_label, preds = compute_predictions(split, hmm_all, bgf_thresholds[split])
        for method, pred in preds.items():
            p = covered_class_precision(true_label, pred)
            rows.append({
                "split": split, "method": method,
                "precision_covered": p["precision"],
                "precision_covered_macro": p["precision_macro"],
            })
            print(f"  {method}: precision (covered classes) weighted={p['precision']:.3f} macro={p['precision_macro']:.3f}")

    df = pd.DataFrame(rows)
    csv_path = PLOTS_DIR.parent / "precision_covered.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    for value_col, ylabel, suffix in [
        ("precision_covered", "Precision, weighted (covered classes only)", "weighted"),
        ("precision_covered_macro", "Precision, macro (covered classes only)", "macro"),
    ]:
        fig = build_figure(df, value_col, ylabel)
        for ext in ("png", "pdf", "svg"):
            out_path = PLOTS_DIR / f"publication_precision_covered_{suffix}.{ext}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved {out_path}")
        plt.close(fig)
