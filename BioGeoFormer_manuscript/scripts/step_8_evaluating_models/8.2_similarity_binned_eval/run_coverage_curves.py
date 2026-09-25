import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from run_evaluation import OUT_DIR, PLOTS_DIR, SPLITS, load_thresholds
from run_method_comparison import build_comparison_frame, load_hmm_all, CONFIDENT_EVALUE, NO_HIT

COLORS = {"bgf": "#0072b2", "hmm": "#009e73", "diamond": "#e66100"}
METHOD_LABELS = {"bgf": "BGF", "hmm": "HMM", "diamond": "DIAMOND"}
N_CURVE_POINTS = 300  # downsample the (up to 152k-row) sorted sequence to this many plotted points


def method_sorted_predictions(df: pd.DataFrame, method: str):
    """Return (correct, predicted_label) arrays sorted from most to least confident."""
    if method == "bgf":
        pred_col, score_col, ascending = "predicted_label_bgf", "confidence_bgf", False
    elif method == "hmm":
        pred_col, score_col, ascending = "predicted_label_hmm", "evalue_hmm", True
    elif method == "diamond":
        pred_col, score_col, ascending = "predicted_label_diamond", "evalue_diamond", True
    else:
        raise ValueError(method)

    d = df[["true_label", pred_col, score_col]].copy()
    d["correct"] = d["true_label"] == d[pred_col]
    is_no_hit = d[pred_col] == NO_HIT
    # Sort real hits by their score (best first); NO_HIT rows always last.
    real = d[~is_no_hit].sort_values(score_col, ascending=ascending)
    no_hit = d[is_no_hit]
    ordered = pd.concat([real, no_hit], ignore_index=True)
    return ordered["correct"].to_numpy(), ordered[pred_col].to_numpy()


def confident_operating_point(df: pd.DataFrame, method: str, bgf_threshold: float):
    """Return (coverage, accuracy, recall, n_classes) for the existing
    single-threshold "confidence_filtered" condition, to mark on the curves."""
    if method == "bgf":
        mask = df["confidence_bgf"] >= bgf_threshold
        pred_col = "predicted_label_bgf"
    elif method == "hmm":
        mask = df["evalue_hmm"] <= CONFIDENT_EVALUE
        pred_col = "predicted_label_hmm"
    else:
        mask = df["evalue_diamond"] <= CONFIDENT_EVALUE
        pred_col = "predicted_label_diamond"

    n_total = len(df)
    n_accepted = mask.sum()
    if n_accepted == 0:
        return None
    n_correct = (df.loc[mask, "true_label"] == df.loc[mask, pred_col]).sum()
    coverage = n_accepted / n_total
    accuracy = n_correct / n_accepted
    recall = n_correct / n_total
    n_classes = df.loc[mask, pred_col][df.loc[mask, pred_col] != NO_HIT].nunique()
    return coverage, accuracy, recall, n_classes


def class_coverage_points(predicted_label: np.ndarray):
    """From a best-first-sorted predicted-label array (NO_HIT excluded from
    the running class count -- it isn't a real class), return downsampled
    (coverage, n_classes) arrays."""
    n = len(predicted_label)
    seen = set()
    n_classes = np.empty(n, dtype=int)
    for i, label in enumerate(predicted_label):
        if label != NO_HIT:
            seen.add(label)
        n_classes[i] = len(seen)
    k = np.arange(1, n + 1)
    coverage = k / n

    if n > N_CURVE_POINTS:
        idx = np.unique(np.linspace(0, n - 1, N_CURVE_POINTS).astype(int))
    else:
        idx = np.arange(n)
    return coverage[idx], n_classes[idx]


def curve_points(correct: np.ndarray):
    """From a best-first-sorted correctness array, return downsampled
    (coverage, accuracy, recall) arrays."""
    n = len(correct)
    cum_correct = np.cumsum(correct)
    k = np.arange(1, n + 1)
    coverage = k / n
    accuracy = cum_correct / k
    recall = cum_correct / n

    if n > N_CURVE_POINTS:
        idx = np.unique(np.linspace(0, n - 1, N_CURVE_POINTS).astype(int))
    else:
        idx = np.arange(n)
    return coverage[idx], accuracy[idx], recall[idx]


def plot_accuracy_coverage(split: int, curves: dict, operating_points: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, color in COLORS.items():
        coverage, accuracy, _ = curves[method]
        ax.plot(coverage, accuracy, color=color, linewidth=2, label=METHOD_LABELS[method])
        op = operating_points.get(method)
        if op is not None:
            ax.scatter([op[0]], [op[1]], color=color, s=70, zorder=5, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Coverage (fraction of test set accepted)")
    ax.set_ylabel("Accuracy of accepted predictions")
    ax.set_title(f"Accuracy-coverage curve — {split}% split", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{split}_accuracy_coverage.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_precision_recall(split: int, curves: dict, operating_points: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, color in COLORS.items():
        _, accuracy, recall = curves[method]
        ax.plot(recall, accuracy, color=color, linewidth=2, label=METHOD_LABELS[method])
        op = operating_points.get(method)
        if op is not None:
            ax.scatter([op[2]], [op[1]], color=color, s=70, zorder=5, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Recall (fraction of split both accepted and correct)")
    ax.set_ylabel("Precision (accuracy of accepted predictions)")
    ax.set_title(f"Precision-recall curve — {split}% split", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{split}_precision_recall.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_class_coverage(split: int, class_curves: dict, operating_points: dict, n_true_classes: int):
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, color in COLORS.items():
        coverage, n_classes = class_curves[method]
        ax.plot(coverage, n_classes, color=color, linewidth=2, label=METHOD_LABELS[method])
        op = operating_points.get(method)
        if op is not None:
            ax.scatter([op[0]], [op[3]], color=color, s=70, zorder=5, edgecolor="black", linewidth=0.8)
    ax.axhline(n_true_classes, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(0.99, n_true_classes, f" {n_true_classes} true classes", fontsize=8, color="gray",
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Coverage (fraction of test set accepted)")
    ax.set_ylabel("Cumulative distinct classes predicted")
    ax.set_title(f"Class-coverage curve — {split}% split", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_true_classes * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{split}_class_coverage.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    print("Loading HMM prediction detail (one-time)...")
    hmm_all = load_hmm_all()

    summary_rows = []
    for split in SPLITS:
        df = build_comparison_frame(split, hmm_all)
        bgf_threshold = thresholds[split]
        n_true_classes = df["true_label"].nunique()

        curves = {}
        class_curves = {}
        operating_points = {}
        for method in ["bgf", "hmm", "diamond"]:
            correct, predicted_label = method_sorted_predictions(df, method)
            curves[method] = curve_points(correct)
            class_curves[method] = class_coverage_points(predicted_label)
            op = confident_operating_point(df, method, bgf_threshold)
            operating_points[method] = op
            if op is not None:
                summary_rows.append({
                    "split": split, "method": method, "n_true_classes": n_true_classes,
                    "operating_coverage": op[0], "operating_accuracy": op[1],
                    "operating_recall": op[2], "operating_n_classes": op[3],
                })

        plot_accuracy_coverage(split, curves, operating_points)
        plot_precision_recall(split, curves, operating_points)
        plot_class_coverage(split, class_curves, operating_points, n_true_classes)
        print(f"split {split}: done")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "coverage_curve_operating_points.csv", index=False)
    print(f"\nWrote coverage_curve_operating_points.csv")
