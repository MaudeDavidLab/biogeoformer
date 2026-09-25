"""Micro- and macro-averaged precision-recall curves for BGF from its full per-class
softmax probabilities (test_set_inference_bgf/sim_<N>/test_probs_long.parquet).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "8.2_similarity_binned_eval"))
from run_evaluation import PLOTS_DIR, SPLITS, REPO_ROOT, load_thresholds

BGF_INFERENCE_DIR = REPO_ROOT / "test_set_inference_bgf"
N_CURVE_POINTS = 3000
BGF_COLOR = "#0072b2"


def micro_pr_curve(split: int):
    path = BGF_INFERENCE_DIR / f"sim_{split}" / "test_probs_long.parquet"
    df = pd.read_parquet(path, columns=["prob", "is_true"])

    precision, recall, _ = precision_recall_curve(df["is_true"].to_numpy(dtype=bool), df["prob"].to_numpy())
    auc = -np.trapz(precision, recall)   # sklearn returns recall in DECREASING order; trapz needs the sign flipped
    n = len(recall)
    if n > N_CURVE_POINTS:
        idx = np.unique(np.linspace(0, n - 1, N_CURVE_POINTS).astype(int))
    else:
        idx = slice(None)
    return recall[idx], precision[idx], auc


def macro_pr_curve(split: int, n_grid=200):
    """Macro PR curve: per-class one-vs-rest curves on a shared recall grid, averaged across classes."""
    path = BGF_INFERENCE_DIR / f"sim_{split}" / "test_probs_long.parquet"
    df = pd.read_parquet(path, columns=["true_label", "cycle", "prob"])

    recall_grid = np.linspace(0, 1, n_grid)
    interpolated = []
    for cls, group in df.groupby("cycle"):
        y_true_c = (group["true_label"] == cls).to_numpy(dtype=bool)
        if y_true_c.sum() == 0:
            continue
        precision_c, recall_c, _ = precision_recall_curve(y_true_c, group["prob"].to_numpy())
        interpolated.append(np.interp(recall_grid, recall_c[::-1], precision_c[::-1]))
    mean_precision = np.mean(interpolated, axis=0)
    auc = np.trapz(mean_precision, recall_grid)
    return recall_grid, mean_precision, auc


def prob_operating_point(split: int, threshold: float):
    """(recall, precision) at BGF's per-split confidence threshold."""
    path = BGF_INFERENCE_DIR / f"sim_{split}" / "test_probs_long.parquet"
    df = pd.read_parquet(path, columns=["prob", "is_true"])
    mask = df["prob"].to_numpy() >= threshold
    is_true = df["is_true"].to_numpy(dtype=bool)
    tp = int((mask & is_true).sum())
    fp = int((mask & ~is_true).sum())
    total_true = int(is_true.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_true if total_true > 0 else 0.0
    return recall, precision


def macro_prob_operating_point(split: int, threshold: float):
    """Macro-native operating point -- see macro_evalue_operating_point()
    in run_diamond_perclass_micro_pr.py for the full rationale (per-class
    precision/recall at the same threshold, then unweighted mean)."""
    path = BGF_INFERENCE_DIR / f"sim_{split}" / "test_probs_long.parquet"
    df = pd.read_parquet(path, columns=["true_label", "cycle", "prob"])

    precisions, recalls = [], []
    for cls, group in df.groupby("cycle"):
        y_true_c = (group["true_label"] == cls).to_numpy(dtype=bool)
        total_true_c = int(y_true_c.sum())
        if total_true_c == 0:
            continue
        mask_c = group["prob"].to_numpy() >= threshold
        tp = int((mask_c & y_true_c).sum())
        fp = int((mask_c & ~y_true_c).sum())
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / total_true_c)
    return float(np.mean(recalls)), float(np.mean(precisions))


def plot_grid(all_data: dict):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

    for ax, split in zip(axes.flat, SPLITS):
        recall, precision = all_data[split]
        ax.plot(recall, precision, color=BGF_COLOR, linewidth=2)
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

    fig.suptitle("BGF precision-recall (per-class, micro-averaged)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = PLOTS_DIR / f"publication_pr_bgf_micro.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds()

    grid_data = {}
    for split in SPLITS:
        print(f"split {split}: loading BGF long-format probs...")
        recall, precision, auc = micro_pr_curve(split)
        op_recall, op_precision = prob_operating_point(split, thresholds[split])
        grid_data[split] = (recall, precision)
        print(f"  max recall: {recall.max():.3f} (AUC={auc:.3f})  |  "
              f"threshold={thresholds[split]} operating point: recall={op_recall:.3f} precision={op_precision:.3f}")

    plot_grid(grid_data)
