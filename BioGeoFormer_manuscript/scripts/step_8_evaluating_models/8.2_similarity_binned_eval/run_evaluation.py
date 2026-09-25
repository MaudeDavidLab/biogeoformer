"""
Shared paths, BGF prediction loading and metric helpers for step 8.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PRED_DIR = REPO_ROOT / "test_set_annotations"
GRAPHPART_DIR = REPO_ROOT / "BGF_clustering" / "train_test_val_final"
DIAMOND_DIR = REPO_ROOT / "graphpart_diamondrun"
OUT_DIR = REPO_ROOT / "results" / "tables" / "similarity_binned_eval"
PLOTS_DIR = REPO_ROOT / "results" / "figures" / "similarity_binned_eval"

SPLITS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
METRICS = ["accuracy", "MCC", "F1", "precision", "recall"]  # support-weighted
METRIC_TITLES = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy (macro)",
    "MCC": "MCC",
    "F1": "F1 (weighted)", "F1_macro": "F1 (macro)",
    "precision": "Precision (weighted)", "precision_macro": "Precision (macro)",
    "recall": "Recall (weighted)", "recall_macro": "Recall (macro)",
}


def load_bgf_predictions(split: int) -> pd.DataFrame:
    """BGF test predictions for one split, joined to GraphPart ids by sequence.
    Columns include id, sequence, true_label, predicted_label, confidence."""
    preds = pd.read_csv(PRED_DIR / f"test_predictions_{split}.csv")
    truth = pd.read_csv(GRAPHPART_DIR / f"bgf_test_{split}.csv")[["id", "sequence"]]
    merged = preds.merge(truth, on="sequence", how="left", validate="one_to_one")
    if merged["id"].isna().any():
        n_missing = merged["id"].isna().sum()
        raise ValueError(f"split {split}: {n_missing} predictions did not match any graphpart sequence")
    return merged


def compute_metrics(y_true, y_pred) -> dict:
    """Support-weighted accuracy/F1/precision/recall and MCC, plus macro
    variants (suffixed _macro; balanced_accuracy == macro recall)."""
    labels = sorted(pd.unique(y_true))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "balanced_accuracy": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "n": len(y_true),
    }


def load_thresholds() -> dict:
    df = pd.read_csv(PRED_DIR / "confidence_thresholds.csv")
    return dict(zip(df["similarity"], df["confidence"]))
