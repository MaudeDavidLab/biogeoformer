"""Shared helpers: per-split BGF/HMM/DIAMOND predictions and the prediction conditions."""

import pandas as pd

from run_evaluation import (
    REPO_ROOT, DIAMOND_DIR, SPLITS,
    load_bgf_predictions, compute_metrics,
)

HMM_DETAIL_CSV = REPO_ROOT / "hmm_runs" / "pathway_prediction_detail.csv"
CONFIDENT_EVALUE = 1e-5  # "reasonable confidence" cutoff for HMM and DIAMOND best-hit E-values
NO_HIT = "NO_HIT"

# HMM "all" condition: --max runs for splits 10-40, default hmmscan for 50-90.
HMM_ALL_METHOD = {s: ("no_threshold" if s <= 40 else "thresholded") for s in SPLITS}


def load_hmm_all() -> pd.DataFrame:
    df = pd.read_csv(HMM_DETAIL_CSV, low_memory=False)
    return df[df["kind"] == "test"]


def hmm_predictions_for_split(hmm_all: pd.DataFrame, split: int) -> pd.DataFrame:
    method = HMM_ALL_METHOD[split]
    sub = hmm_all[(hmm_all["split"] == split) & (hmm_all["method"] == method)]
    out = sub[["id", "predicted_pathway", "evalue_full"]].rename(
        columns={"predicted_pathway": "predicted_label_hmm", "evalue_full": "evalue_hmm"}
    )
    return out


def diamond_predictions_for_split(split: int) -> pd.DataFrame:
    d = pd.read_csv(
        DIAMOND_DIR / f"bench_test_{split}.tsv", sep="\t", header=None,
        names=["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"],
    )
    d["id"] = d["qseqid"].str.split("|").str[0]
    d["predicted_label_diamond"] = d["sseqid"].str.split("|").str[2]
    return d[["id", "predicted_label_diamond", "evalue"]].rename(columns={"evalue": "evalue_diamond"})


def build_comparison_frame(split: int, hmm_all: pd.DataFrame) -> pd.DataFrame:
    base = load_bgf_predictions(split)
    base = base.rename(columns={"predicted_label": "predicted_label_bgf", "confidence": "confidence_bgf"})
    base = base[["id", "true_label", "predicted_label_bgf", "confidence_bgf"]]

    hmm = hmm_predictions_for_split(hmm_all, split)
    diamond = diamond_predictions_for_split(split)

    df = base.merge(hmm, on="id", how="left").merge(diamond, on="id", how="left")
    # ids with no HMM/diamond row at all (shouldn't happen for HMM -- detail
    # covers every query -- but diamond has real coverage gaps) get NO_HIT.
    df["predicted_label_hmm"] = df["predicted_label_hmm"].fillna(NO_HIT)
    df["predicted_label_diamond"] = df["predicted_label_diamond"].fillna(NO_HIT)
    return df


def method_mask_and_predictions(df: pd.DataFrame, method: str, condition: str, bgf_threshold: float):
    """(mask, predictions) for one method and condition: "all" keeps every row,
    "confidence_filtered" keeps confident rows only, "thresholded" keeps every row
    but sets non-confident predictions to NO_HIT.
    """
    always = pd.Series(True, index=df.index)
    if method == "bgf":
        raw_pred, confident = df["predicted_label_bgf"], df["confidence_bgf"] >= bgf_threshold
    elif method == "hmm":
        raw_pred, confident = df["predicted_label_hmm"], df["evalue_hmm"] <= CONFIDENT_EVALUE
    elif method == "diamond":
        raw_pred, confident = df["predicted_label_diamond"], df["evalue_diamond"] <= CONFIDENT_EVALUE
    else:
        raise ValueError(method)

    if condition == "all":
        return always, raw_pred
    if condition == "confidence_filtered":
        return confident, raw_pred
    if condition == "thresholded":
        return always, raw_pred.where(confident, NO_HIT)
    raise ValueError(condition)


def covered_class_precision(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Weighted and macro precision restricted to classes the method predicted at least once."""
    covered = sorted(y_pred[y_pred != NO_HIT].unique())
    if not covered:
        return {"precision_covered": 0.0, "precision_macro_covered": 0.0}
    mask = y_true.isin(covered)
    m = compute_metrics(y_true[mask], y_pred[mask])
    return {"precision_covered": m["precision"], "precision_macro_covered": m["precision_macro"]}
