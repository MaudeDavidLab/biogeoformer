"""Share of each method's confidence-filtered predictions assigned to each class, per split."""

import pandas as pd
import matplotlib.pyplot as plt

from run_evaluation import OUT_DIR, PLOTS_DIR, SPLITS, load_thresholds
from run_method_comparison import build_comparison_frame, load_hmm_all, method_mask_and_predictions, NO_HIT

COLORS = {"bgf": "#0072b2", "hmm": "#009e73", "diamond": "#e66100"}
METHOD_LABELS = {"bgf": "BGF", "hmm": "HMM", "diamond": "DIAMOND"}


def class_breakdown_for_split(df: pd.DataFrame, split: int, bgf_threshold: float) -> pd.DataFrame:
    rows = []
    for method in ["bgf", "hmm", "diamond"]:
        mask, pred = method_mask_and_predictions(df, method, "confidence_filtered", bgf_threshold)
        accepted = pred[mask]
        accepted = accepted[accepted != NO_HIT]
        n_confident = len(accepted)
        counts = accepted.value_counts()
        for cls, n in counts.items():
            rows.append({
                "split": split, "method": method, "predicted_class": cls,
                "n": int(n), "pct_of_confident": n / n_confident if n_confident else 0.0,
            })
    return pd.DataFrame(rows)


def plot_split(split: int, breakdown: pd.DataFrame):
    split_df = breakdown[breakdown["split"] == split]
    # Order classes by BGF's share descending (falls back to max share across
    # methods for classes BGF never predicts confidently) -- puts the most
    # BGF-concentrated classes at the top, which is the pattern this plot exists to show.
    bgf_share = split_df[split_df["method"] == "bgf"].set_index("predicted_class")["pct_of_confident"]
    all_classes = split_df["predicted_class"].unique()
    max_share = split_df.groupby("predicted_class")["pct_of_confident"].max()
    order = sorted(all_classes, key=lambda c: (-bgf_share.get(c, -1), -max_share[c]))

    fig_height = max(4, 0.28 * len(order))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    y = range(len(order))
    bar_height = 0.25
    for i, (method, color) in enumerate(COLORS.items()):
        m_df = split_df[split_df["method"] == method].set_index("predicted_class")["pct_of_confident"]
        vals = [m_df.get(c, 0.0) for c in order]
        offsets = [yi + (i - 1) * bar_height for yi in y]
        ax.barh(offsets, vals, height=bar_height, color=color, label=METHOD_LABELS[method])

    ax.set_yticks(list(y))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of that method's confidence-filtered predictions")
    ax.set_title(f"Confident-prediction class breakdown — {split}% split", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    out_path = PLOTS_DIR / f"{split}_class_breakdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    print("Loading HMM prediction detail (one-time)...")
    hmm_all = load_hmm_all()

    all_rows = []
    for split in SPLITS:
        df = build_comparison_frame(split, hmm_all)
        bgf_threshold = thresholds[split]
        breakdown = class_breakdown_for_split(df, split, bgf_threshold)
        all_rows.append(breakdown)
        plot_split(split, breakdown)
        print(f"split {split}: done")

    full = pd.concat(all_rows, ignore_index=True).sort_values(["split", "method", "n"], ascending=[True, True, False])
    full.to_csv(OUT_DIR / "class_breakdown_confident.csv", index=False)
    print(f"\nWrote class_breakdown_confident.csv ({len(full)} rows)")
