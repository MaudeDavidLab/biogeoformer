"""
Fig 4c: above-threshold MCC of BGF test predictions by split and by mmseqs
identity to the nearest training sequence (mmseqs_identity_to_train/maxid_<N>.tsv,
no coverage or identity cutoff). Bins with n < 30 are gray.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[3]
PRED_DIR = REPO_ROOT / "test_set_annotations"
MMSEQS_DIR = REPO_ROOT / "mmseqs_identity_to_train"
FULL_DB_CSV = REPO_ROOT / "BioGeoFormer_db" / "BioGeoFormer_db.csv"
OUT_DIR = REPO_ROOT / "results" / "tables" / "fig4c_mmseqs_identity"
PLOTS_DIR = REPO_ROOT / "results" / "figures" / "fig4c_mmseqs_identity"

SPLITS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
BIN_EDGES = [i / 10 for i in range(11)]
BIN_LABELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
MIN_N = 30


def load_maxid(split: int) -> pd.DataFrame:
    maxid = pd.read_csv(
        MMSEQS_DIR / f"maxid_{split}.tsv", sep="\t", header=None,
        names=["query_id", "train_id", "identity", "one_minus_identity"],
    )
    maxid["accession"] = maxid["query_id"].str.split("|").str[0]
    return maxid


def build_accession_to_seq(accessions: set) -> dict:
    """Stream the 744MB database once, keeping only the accessions needed."""
    d = {}
    with open(FULL_DB_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_idx, seq_idx = header.index("id"), header.index("sequence")
        for row in reader:
            if row[id_idx] in accessions:
                d[row[id_idx]] = row[seq_idx]
    return d


def load_split(split: int, maxid: pd.DataFrame, acc_to_seq: dict) -> pd.DataFrame:
    preds = pd.read_csv(PRED_DIR / f"test_predictions_{split}.csv")
    maxid = maxid.assign(sequence=maxid["accession"].map(acc_to_seq))
    n_unmapped = maxid["sequence"].isna().sum()
    if n_unmapped:
        print(f"  split {split}: WARNING {n_unmapped} query accessions not found in BioGeoFormer_db.csv")
        maxid = maxid.dropna(subset=["sequence"])
    n_dup = maxid["sequence"].duplicated().sum()
    if n_dup:
        print(f"  split {split}: WARNING {n_dup} duplicate sequences in maxid file, keeping first")
        maxid = maxid.drop_duplicates(subset="sequence", keep="first")

    merged = preds.merge(maxid[["sequence", "identity"]], on="sequence", how="left", validate="one_to_one")
    n_missing = merged["identity"].isna().sum()
    if n_missing:
        print(f"  split {split}: WARNING {n_missing}/{len(merged)} predictions have no mmseqs identity match")

    merged["bin"] = pd.cut(merged["identity"], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True, right=False)
    merged.loc[merged["identity"] == 1.0, "bin"] = BIN_LABELS[-1]
    return merged


def compute_metrics(y_true, y_pred) -> dict:
    labels = sorted(pd.unique(y_true))
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred) if len(labels) > 1 else np.nan,
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
    }


def evaluate_all_splits() -> pd.DataFrame:
    thresholds = pd.read_csv(PRED_DIR / "confidence_thresholds.csv")
    thresholds = dict(zip(thresholds["similarity"], thresholds["confidence"]))

    maxids = {split: load_maxid(split) for split in SPLITS}
    accessions = set().union(*(m["accession"] for m in maxids.values()))
    print(f"Looking up {len(accessions)} accessions in {FULL_DB_CSV.name}...")
    acc_to_seq = build_accession_to_seq(accessions)

    rows = []
    for split in SPLITS:
        threshold = thresholds[split]
        df = load_split(split, maxids[split], acc_to_seq)
        conditions = {
            "all": df,
            "above_threshold": df[df["confidence"] >= threshold],
            "below_threshold": df[df["confidence"] < threshold],
        }
        for condition, subset in conditions.items():
            for bin_label in BIN_LABELS:
                bin_subset = subset[subset["bin"] == bin_label]
                if len(bin_subset) == 0:
                    continue
                m = compute_metrics(bin_subset["true_label"], bin_subset["predicted_label"])
                rows.append({"split": split, "threshold": threshold, "condition": condition, "bin": bin_label, **m})
        print(f"split {split}: done (threshold={threshold}, n={len(df)})")
    return pd.DataFrame(rows)


def plot_heatmap(by_bin: pd.DataFrame, condition: str = "above_threshold"):
    sub = by_bin[(by_bin["condition"] == condition) & (by_bin["n"] >= MIN_N)]
    mcc_mat = sub.pivot(index="split", columns="bin", values="MCC").reindex(index=sorted(SPLITS, reverse=True), columns=BIN_LABELS)
    n_mat = sub.pivot(index="split", columns="bin", values="n").reindex(index=mcc_mat.index, columns=BIN_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="#e0e0e0")
    im = ax.imshow(np.ma.masked_invalid(mcc_mat.values.astype(float)), aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(BIN_LABELS)))
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(mcc_mat.index)))
    ax.set_yticklabels([f"{s}%" for s in mcc_mat.index])
    ax.set_xlabel("identity to nearest train sequence")
    ax.set_ylabel("split (% model)")

    for i in range(mcc_mat.shape[0]):
        for j in range(mcc_mat.shape[1]):
            v = mcc_mat.values[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if v < 0.6 else "black"
            ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")
            ax.text(j, i + 0.22, f"n={int(n_mat.values[i, j])}", ha="center", va="center", fontsize=6.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MCC")
    ax.set_xticks(np.arange(-0.5, len(BIN_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mcc_mat.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        out_path = PLOTS_DIR / f"mcc_{condition}_heatmap.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    by_bin = evaluate_all_splits()
    by_bin.to_csv(OUT_DIR / "metrics_by_bin.csv", index=False)
    print(f"Wrote {len(by_bin)} rows to {OUT_DIR / 'metrics_by_bin.csv'}")
    plot_heatmap(by_bin)
