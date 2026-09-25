"""Micro- and macro-averaged precision-recall curves for DIAMOND from the per-class database
searches (graphpart_diamondrun/perclass_bench/<N>/bench_test_<N>_vs_<class>.tsv).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "8.2_similarity_binned_eval"))
from run_evaluation import PLOTS_DIR, SPLITS, DIAMOND_DIR
from run_method_comparison import build_comparison_frame, load_hmm_all
from run_coverage_curves import method_sorted_predictions, curve_points

PERCLASS_BENCH_DIR = DIAMOND_DIR / "perclass_bench"
N_CURVE_POINTS = 3000
DIAMOND_COLOR = "#e66100"
OLD_CURVE_COLOR = "#999999"


def sanitize(name: str) -> str:
    """Normalize a class name for filenames/matching; must match run_split_train_by_class.py's sanitize()."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def load_split_matrix(split: int):
    """Return (query_ids, classes, bitscore_matrix, evalue_matrix, true_label)."""
    split_dir = PERCLASS_BENCH_DIR / str(split)
    class_files = sorted(split_dir.glob(f"bench_test_{split}_vs_*.tsv"))
    if not class_files:
        raise FileNotFoundError(f"no per-class bench files found under {split_dir}")

    classes = [f.stem.replace(f"bench_test_{split}_vs_", "") for f in class_files]

    per_class_scores = {}   # class -> DataFrame(qseqid -> bitscore, evalue), best row per query
    all_ids = set()
    for cls, f in zip(classes, class_files):
        d = pd.read_csv(f, sep="\t", header=None,
                         names=["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"])
        if d.empty:
            per_class_scores[cls] = pd.DataFrame(columns=["bitscore", "evalue"])
            continue
        # max-target-seqs 1 -> at most one row per query already, but guard anyway;
        # sort by bitscore descending so groupby().first() keeps the best row's evalue too
        g = d.sort_values("bitscore", ascending=False).groupby("qseqid").first()[["bitscore", "evalue"]]
        per_class_scores[cls] = g
        all_ids.update(g.index)

    query_ids = np.array(sorted(all_ids))
    true_label = np.array([sanitize(qid.split("|")[2]) for qid in query_ids])

    matrix = np.zeros((len(query_ids), len(classes)), dtype=np.float32)
    evalue_matrix = np.full((len(query_ids), len(classes)), np.inf, dtype=np.float32)
    id_pos = {qid: i for i, qid in enumerate(query_ids)}
    for j, cls in enumerate(classes):
        g = per_class_scores[cls]
        if g.empty:
            continue
        rows = [id_pos[qid] for qid in g.index]
        matrix[rows, j] = g["bitscore"].to_numpy(dtype=np.float32)
        evalue_matrix[rows, j] = g["evalue"].to_numpy(dtype=np.float32)

    return query_ids, classes, matrix, evalue_matrix, true_label


def patch_matrix_with_combined_top1(split: int, query_ids, classes, matrix, evalue_matrix):
    """Merge each query's combined-database top hit into the per-class matrix
    (keeping the higher bitscore, with its E-value).
    """
    id_pos = {qid: i for i, qid in enumerate(query_ids)}
    class_pos = {cls: j for j, cls in enumerate(classes)}

    combined = pd.read_csv(
        DIAMOND_DIR / f"bench_test_{split}.tsv", sep="\t", header=None,
        names=["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"],
    )
    combined["predicted_class"] = combined["sseqid"].str.split("|").str[2].map(sanitize)

    patched, skipped_no_row, skipped_unknown_class = 0, 0, 0
    for qseqid, pred_class, bitscore, evalue in zip(
        combined["qseqid"], combined["predicted_class"], combined["bitscore"], combined["evalue"]
    ):
        i = id_pos.get(qseqid)
        j = class_pos.get(pred_class)
        if i is None:
            skipped_no_row += 1
            continue
        if j is None:
            skipped_unknown_class += 1
            continue
        if bitscore > matrix[i, j]:
            matrix[i, j] = bitscore
            evalue_matrix[i, j] = evalue
        patched += 1

    print(f"  patched {patched} query x class cells from the combined-DB run "
          f"({skipped_no_row} not in per-class query set, {skipped_unknown_class} unknown class)")
    return matrix, evalue_matrix


def evalue_operating_point(evalue_matrix, true_label, classes, threshold=1e-5):
    """(recall, precision) at the E-value <= threshold cutoff."""
    y_true = (true_label[:, None] == np.array(classes)[None, :]).ravel()
    mask = evalue_matrix.ravel() <= threshold
    tp = int((mask & y_true).sum())
    fp = int((mask & ~y_true).sum())
    total_true = int(y_true.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_true if total_true > 0 else 0.0
    return recall, precision


def macro_evalue_operating_point(evalue_matrix, true_label, classes, threshold=1e-5):
    """Macro (recall, precision) at the E-value cutoff, averaged across classes."""
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


def sanity_check_top1(split: int, query_ids, classes, matrix, true_label, hmm_all):
    """Compare reconstructed (argmax bitscore) top-1 vs. the original
    single-combined-DB DIAMOND top-1 prediction, on the shared query set."""
    reconstructed_pred = np.array(classes)[matrix.argmax(axis=1)]
    reconstructed = dict(zip(query_ids, reconstructed_pred))

    df = build_comparison_frame(split, hmm_all)
    df = df.set_index("id")
    # query_ids embed the id as the first "|"-delimited field
    ids_only = [qid.split("|")[0] for qid in query_ids]

    matched, agree = 0, 0
    for qid, sid, recon_label in zip(query_ids, ids_only, reconstructed_pred):
        if sid not in df.index:
            continue
        orig_label = sanitize(df.loc[sid, "predicted_label_diamond"])
        matched += 1
        if orig_label == recon_label:
            agree += 1

    pct = 100 * agree / matched if matched else float("nan")
    print(f"split {split}: top-1 agreement (reconstructed vs. original combined-DB) "
          f"= {agree}/{matched} ({pct:.1f}%)")
    return pct


def micro_pr_curve(matrix, true_label, classes):
    """Micro-averaged PR curve and AUC over all (query, class) pairs."""
    n_queries, n_classes = matrix.shape
    y_true = (true_label[:, None] == np.array(classes)[None, :]).ravel()
    scores = matrix.ravel()

    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc = -np.trapz(precision, recall)   # sklearn returns recall in DECREASING order; trapz needs the sign flipped
    # sklearn returns these sorted by increasing recall (decreasing threshold);
    # downsample for plotting
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
        # sklearn returns recall DECREASING; reverse to ascending for np.interp's xp requirement
        interpolated.append(np.interp(recall_grid, recall_c[::-1], precision_c[::-1]))
    mean_precision = np.mean(interpolated, axis=0)
    auc = np.trapz(mean_precision, recall_grid)   # recall_grid is already ascending, no sign flip needed
    return recall_grid, mean_precision, auc


def old_style_curve(split: int, hmm_all):
    """Original top-1-only DIAMOND curve (recall capped at plain accuracy),
    for comparison -- reuses run_coverage_curves.py's logic exactly."""
    df = build_comparison_frame(split, hmm_all)
    correct, _ = method_sorted_predictions(df, "diamond")
    coverage, accuracy, recall = curve_points(correct)
    return recall, accuracy


def plot_grid(all_data: dict):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

    for ax, split in zip(axes.flat, SPLITS):
        recall_new, precision_new, recall_old, precision_old = all_data[split]
        ax.plot(recall_new, precision_new, color=DIAMOND_COLOR, linewidth=2)
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

    fig.suptitle(
        "DIAMOND precision-recall (per-class DB, micro-averaged)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = PLOTS_DIR / f"publication_pr_diamond_micro.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading HMM prediction detail (one-time, needed for the original-curve comparison)...")
    hmm_all = load_hmm_all()

    grid_data = {}
    for split in SPLITS:
        print(f"\nsplit {split}: loading per-class bitscore matrix...")
        query_ids, classes, matrix, evalue_matrix, true_label = load_split_matrix(split)
        print(f"  {len(query_ids)} queries x {len(classes)} classes")

        print("  before patch:")
        sanity_check_top1(split, query_ids, classes, matrix, true_label, hmm_all)
        matrix, evalue_matrix = patch_matrix_with_combined_top1(split, query_ids, classes, matrix, evalue_matrix)
        print("  after patch:")
        sanity_check_top1(split, query_ids, classes, matrix, true_label, hmm_all)

        recall_new, precision_new, auc_new = micro_pr_curve(matrix, true_label, classes)
        op_recall, op_precision = evalue_operating_point(evalue_matrix, true_label, classes)
        print(f"  E<=1e-5 operating point: recall={op_recall:.3f} precision={op_precision:.3f}")
        recall_old, precision_old = old_style_curve(split, hmm_all)
        grid_data[split] = (recall_new, precision_new, recall_old, precision_old)
        print(f"  new curve max recall: {recall_new.max():.3f} (AUC={auc_new:.3f})  |  old curve max recall: {recall_old.max():.3f}")

    plot_grid(grid_data)
