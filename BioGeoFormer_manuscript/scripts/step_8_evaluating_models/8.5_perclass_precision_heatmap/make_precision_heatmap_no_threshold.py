from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

splits = [10, 20, 30, 40, 50, 60, 70, 80, 90]

REPO_ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = REPO_ROOT / "test_set_annotations"
FIG_DIR = REPO_ROOT / "results" / "figures" / "perclass_precision_heatmap"
TABLE_DIR = REPO_ROOT / "results" / "tables" / "perclass_precision_heatmap"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

results = []
all_classes = set()

# --------------------------------------------------
# 2. Calculate per-pathway precision
#    WITHOUT confidence thresholding
# --------------------------------------------------

for s in splits:

    df = pd.read_csv(PRED_DIR / f"test_predictions_{s}.csv")

    # No confidence filtering: use ALL predictions

    all_classes.update(df["true_label"].dropna())
    all_classes.update(df["predicted_label"].dropna())

    # Identify correct predictions
    df["correct"] = (
        df["true_label"] == df["predicted_label"]
    ).astype(int)

    # Group by PREDICTED pathway to calculate precision
    stats = (
        df.groupby("predicted_label")
        .agg(
            precision=("correct", "mean"),
            n=("correct", "size")
        )
        .reset_index()
    )

    stats["identity"] = s
    results.append(stats)

results = pd.concat(results, ignore_index=True)

# --------------------------------------------------
# 3. Build matrices
# --------------------------------------------------

all_classes = sorted(all_classes)

precision_matrix = results.pivot(
    index="predicted_label",
    columns="identity",
    values="precision"
).reindex(
    index=all_classes,
    columns=splits
)

n_matrix = results.pivot(
    index="predicted_label",
    columns="identity",
    values="n"
).reindex(
    index=all_classes,
    columns=splits
)

# --------------------------------------------------
# 4. Add annotations
# --------------------------------------------------

annotations = precision_matrix.copy().astype(object)

for pathway in precision_matrix.index:

    for s in splits:

        p = precision_matrix.loc[pathway, s]
        n = n_matrix.loc[pathway, s]

        if pd.isna(p):
            annotations.loc[pathway, s] = ""

        elif n < 5:
            annotations.loc[pathway, s] = f"{p:.2f}*"

        else:
            annotations.loc[pathway, s] = f"{p:.2f}"

# --------------------------------------------------
# 5. Plot heatmap
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 14))

cmap = plt.cm.Blues.copy()
cmap.set_bad("#E0E0E0")

sns.heatmap(
    precision_matrix,
    cmap=cmap,
    vmin=0,
    vmax=1,
    annot=annotations,
    fmt="",
    annot_kws={"fontsize": 7},
    linewidths=0.4,
    linecolor="white",
    cbar_kws={
        "label": "Precision",
        "shrink": 0.6
    },
    ax=ax
)

ax.set_title(
    "BGF pathway-specific precision without confidence thresholding",
    fontsize=14,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Sequence identity (%)", fontsize=12)
ax.set_ylabel("Predicted pathway", fontsize=12)

ax.set_xticklabels(
    [f"{s}%" for s in splits],
    rotation=0
)

ax.tick_params(axis="y", labelsize=9)

fig.text(
    0.12,
    0.02,
    "Gray: no predictions. *: fewer than five predictions.",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

# --------------------------------------------------
# 6. Save and display
# --------------------------------------------------

plt.savefig(
    FIG_DIR / "BGF_precision_heatmap_no_threshold.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    FIG_DIR / "BGF_precision_heatmap_no_threshold.pdf",
    bbox_inches="tight"
)

plt.savefig(
    FIG_DIR / "BGF_precision_heatmap_no_threshold.svg",
    bbox_inches="tight"
)


# --------------------------------------------------
# 7. Export underlying data
# --------------------------------------------------

precision_matrix.to_csv(
    TABLE_DIR / "BGF_precision_matrix_no_threshold.csv"
)

n_matrix.to_csv(
    TABLE_DIR / "BGF_prediction_counts_no_threshold.csv"
)
