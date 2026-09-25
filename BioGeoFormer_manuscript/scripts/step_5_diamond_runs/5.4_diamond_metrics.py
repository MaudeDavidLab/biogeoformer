"""Accuracy, MCC and weighted F1/precision/recall for the DIAMOND baseline on val and test
sets (prediction = best hit's pathway; no hit counts as wrong).
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIAMOND_DIR = REPO_ROOT / "graphpart_diamondrun"
SPLIT_DIR   = REPO_ROOT / "BGF_clustering" / "train_test_val_final"
SPLITS      = [10, 20, 30, 40, 50, 60, 70, 80, 90]
NO_HIT_LABEL = "__NO_HIT__"


def load_true_labels(csv_path: Path) -> pd.Series:
    """Return a Series indexed by sequence id, values are cycle labels."""
    df = pd.read_csv(csv_path)
    # Both val and test CSVs have 'id' and 'cycle' columns
    return df.set_index("id")["cycle"]


def load_diamond_predictions(tsv_path: Path) -> dict:
    """Return {query_id: pathway of the best-hit subject}."""
    preds = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            query_fields   = parts[0].split("|")
            subject_fields = parts[1].split("|")
            query_id    = query_fields[0]
            pred_label  = subject_fields[2] if len(subject_fields) >= 3 else NO_HIT_LABEL
            # Keep only first (best) hit per query
            if query_id not in preds:
                preds[query_id] = pred_label
    return preds


def compute_metrics(y_true, y_pred, labels):
    acc  = accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    return {"accuracy": acc, "MCC": mcc, "F1": f1, "precision": prec, "recall": rec}


def evaluate_split(split: int, split_type: str) -> dict:
    """split_type: 'val' or 'test'"""
    csv_path = SPLIT_DIR / f"bgf_{split_type}_{split}.csv"
    tsv_path = DIAMOND_DIR / f"bench_{split_type}_{split}.tsv"

    true_labels = load_true_labels(csv_path)
    diamond_preds = load_diamond_predictions(tsv_path)

    # Build aligned arrays; missing = NO_HIT
    y_true = []
    y_pred = []
    n_missing = 0
    for seq_id, true_label in true_labels.items():
        y_true.append(true_label)
        pred = diamond_preds.get(seq_id, NO_HIT_LABEL)
        y_pred.append(pred)
        if pred == NO_HIT_LABEL:
            n_missing += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Labels = only the real classes (NO_HIT treated as wrong pred, not a valid class)
    labels = sorted(true_labels.unique().tolist())

    metrics = compute_metrics(y_true, y_pred, labels)
    metrics["n_total"]   = len(y_true)
    metrics["n_missing"] = n_missing
    metrics["split"]     = split
    metrics["set"]       = split_type
    return metrics


rows = []
for split in SPLITS:
    for split_type in ("val", "test"):
        tsv_path = DIAMOND_DIR / f"bench_{split_type}_{split}.tsv"
        csv_path = SPLIT_DIR  / f"bgf_{split_type}_{split}.csv"
        if not tsv_path.exists() or not csv_path.exists():
            print(f"Skipping {split_type}_{split}: file not found")
            continue
        result = evaluate_split(split, split_type)
        rows.append(result)
        print(
            f"[{split_type:4s} {split:3d}%]  "
            f"acc={result['accuracy']:.4f}  MCC={result['MCC']:.4f}  "
            f"F1={result['F1']:.4f}  prec={result['precision']:.4f}  "
            f"rec={result['recall']:.4f}  "
            f"missing={result['n_missing']}/{result['n_total']}"
        )

df = pd.DataFrame(rows)
col_order = ["set", "split", "accuracy", "MCC", "F1", "precision", "recall",
             "n_total", "n_missing"]
df = df[col_order]
out_path = REPO_ROOT / "results" / "tables" / "diamond_metrics.csv"
df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")
print("\nFull table:")
print(df.to_string(index=False))
