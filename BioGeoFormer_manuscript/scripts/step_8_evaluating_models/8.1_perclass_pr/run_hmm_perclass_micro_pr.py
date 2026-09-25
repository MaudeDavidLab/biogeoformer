"""Micro- and macro-averaged precision-recall curves for HMM from the per-class scores in
the hmmscan tblout output.
"""

import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "8.2_similarity_binned_eval"))
from run_evaluation import PLOTS_DIR, SPLITS, REPO_ROOT

HMM_DIR = REPO_ROOT / "hmm_runs"
TBLOUT_DIR = HMM_DIR / "scan_results"
BESTHIT_DIR = HMM_DIR / "scan_results_max_besthit"
N_CURVE_POINTS = 3000
HMM_COLOR = "#009e73"

TBLOUT_COLS = [
    "target_name", "target_accession", "query_name", "query_accession",
    "evalue_full", "score_full", "bias_full",
    "evalue_best1dom", "score_best1dom", "bias_best1dom",
    "exp", "reg", "clu", "ov", "env", "dom", "rep", "inc", "description",
]


def target_to_class(target_name: str) -> str:
    """"bgf_train_10__assnitred__nasA" -> "assnitred" (class names in this
    file are already underscore-safe, unlike DIAMOND's raw sseqid -- no
    extra sanitize() needed)."""
    return target_name.split("__", 2)[1]


def sanitize(name: str) -> str:
    """Normalize a class name for filenames/matching; must match run_split_train_by_class.py's sanitize()."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def load_split_matrix(split: int):
    """Return (query_ids, classes, score_matrix, evalue_matrix, true_label)."""
    tblout_path = TBLOUT_DIR / f"bgf_test_{split}.tblout"
    d = pd.read_csv(tblout_path, comment="#", sep=r"\s+", header=None, names=TBLOUT_COLS)
    d["class"] = d["target_name"].map(target_to_class)

    classes = sorted(d["class"].unique())
    class_pos = {cls: j for j, cls in enumerate(classes)}

    # best (max score) row per (query, class) -- a class can have multiple gene profiles;
    # sort descending so groupby().first() keeps the winning row's evalue too
    best = (d.sort_values("score_full", ascending=False)
             .groupby(["query_name", "class"])
             .first()[["score_full", "evalue_full"]]
             .reset_index())

    query_ids_tblout = sorted(best["query_name"].unique())

    besthit_path = BESTHIT_DIR / f"bgf_test_{split}.besthit.tsv"
    besthit = None
    if besthit_path.exists():
        besthit = pd.read_csv(besthit_path, sep="\t", header=None,
                               names=["query_name", "target_name", "evalue", "score"])
        besthit["class"] = besthit["target_name"].map(target_to_class)

    all_ids = set(query_ids_tblout)
    if besthit is not None:
        all_ids.update(besthit["query_name"])

    query_ids = np.array(sorted(all_ids))
    true_label = np.array([sanitize(qid.split("|")[2]) for qid in query_ids])
    id_pos = {qid: i for i, qid in enumerate(query_ids)}

    matrix = np.zeros((len(query_ids), len(classes)), dtype=np.float32)
    evalue_matrix = np.full((len(query_ids), len(classes)), np.inf, dtype=np.float32)
    for qname, cls, score, evalue in zip(best["query_name"], best["class"], best["score_full"], best["evalue_full"]):
        matrix[id_pos[qname], class_pos[cls]] = score
        evalue_matrix[id_pos[qname], class_pos[cls]] = evalue

    patched = 0
    if besthit is not None:
        tblout_ids = set(query_ids_tblout)
        for qname, cls, score, evalue in zip(besthit["query_name"], besthit["class"], besthit["score"], besthit["evalue"]):
            if qname in tblout_ids:
                continue   # already has full per-class data from the tblout, don't override
            i, j = id_pos[qname], class_pos.get(cls)
            if j is None:
                continue
            if score > matrix[i, j]:
                matrix[i, j] = score
                evalue_matrix[i, j] = evalue
            patched += 1

    coverage = len(all_ids) if besthit is None else len(query_ids_tblout)
    print(f"  {len(query_ids_tblout)} queries with tblout coverage"
          + (f", +{patched} patched in from --max besthit ({len(all_ids) - len(query_ids_tblout)} were entirely absent)" if besthit is not None else ""))

    return query_ids, classes, matrix, evalue_matrix, true_label


def evalue_operating_point(evalue_matrix, true_label, classes, threshold=1e-5):
    """(recall, precision) at the fixed E<=threshold "confident" cutoff --
    a direct mask on evalue_matrix, matching the same convention used
    throughout this project (CONFIDENT_EVALUE in run_method_comparison.py)."""
    y_true = (true_label[:, None] == np.array(classes)[None, :]).ravel()
    mask = evalue_matrix.ravel() <= threshold
    tp = int((mask & y_true).sum())
    fp = int((mask & ~y_true).sum())
    total_true = int(y_true.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_true if total_true > 0 else 0.0
    return recall, precision


def macro_evalue_operating_point(evalue_matrix, true_label, classes, threshold=1e-5):
    """Macro-native operating point -- see the matching function in
    run_diamond_perclass_micro_pr.py for the full rationale (per-class
    precision/recall at the same threshold, then unweighted mean)."""
    classes_arr = np.array(classes)
    precisions, recalls = [], []
    for j, cls in enumerate(classes_arr):
        y_true_c = true_label == cls
        total_true_c = int(y_true_c.sum())
        if total_true_c == 0:
            continue
        mask_c = evalue_matrix[:, j] <= threshold
        tp = int((mask_c & y_true_c).sum())
        fp = int((mask_c & ~y_true_c).sum())
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / total_true_c)
    return float(np.mean(recalls)), float(np.mean(precisions))


def micro_pr_curve(matrix, true_label, classes):
    y_true = (true_label[:, None] == np.array(classes)[None, :]).ravel()
    scores = matrix.ravel()
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc = -np.trapz(precision, recall)   # sklearn returns recall in DECREASING order; trapz needs the sign flipped
    n = len(recall)
    if n > N_CURVE_POINTS:
        idx = np.unique(np.linspace(0, n - 1, N_CURVE_POINTS).astype(int))
    else:
        idx = np.arange(n)
    return recall[idx], precision[idx], auc


def macro_pr_curve(matrix, true_label, classes, n_grid=200):
    """Macro PR curve: per-class one-vs-rest curves on a shared recall grid, averaged across classes."""
    recall_grid = np.linspace(0, 1, n_grid)
    classes_arr = np.array(classes)
    interpolated = []
    for j, cls in enumerate(classes_arr):
        y_true_c = true_label == cls
        if y_true_c.sum() == 0:
            continue
        precision_c, recall_c, _ = precision_recall_curve(y_true_c, matrix[:, j])
        interpolated.append(np.interp(recall_grid, recall_c[::-1], precision_c[::-1]))
    mean_precision = np.mean(interpolated, axis=0)
    auc = np.trapz(mean_precision, recall_grid)
    return recall_grid, mean_precision, auc


def plot_grid(all_data: dict):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

    for ax, split in zip(axes.flat, SPLITS):
        recall, precision = all_data[split]
        ax.plot(recall, precision, color=HMM_COLOR, linewidth=2)
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

    fig.suptitle("HMM precision-recall (per-class, micro-averaged)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = PLOTS_DIR / f"publication_pr_hmm_micro.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    grid_data = {}
    for split in SPLITS:
        print(f"\nsplit {split}: loading HMM per-class score matrix...")
        query_ids, classes, matrix, evalue_matrix, true_label = load_split_matrix(split)
        print(f"  {len(query_ids)} queries x {len(classes)} classes")

        recall, precision, auc = micro_pr_curve(matrix, true_label, classes)
        op_recall, op_precision = evalue_operating_point(evalue_matrix, true_label, classes)
        grid_data[split] = (recall, precision)
        print(f"  max recall: {recall.max():.3f} (AUC={auc:.3f})  |  E<=1e-5 operating point: recall={op_recall:.3f} precision={op_precision:.3f}")

    plot_grid(grid_data)
