"""
ml_bench.py
===========

A clean, reproducible baseline that uses ESM-2 checkpoints as a *frozen feature
extractor* and benchmarks classical ML classifiers (Random Forest, linear/RBF
SVM, logistic regression, kNN) for predicting the metabolic-cycle label of the
protein sequences in ``data/train``.

Data layout (see ``data/train/bgf_train_<id>.csv``)::

    id, gene, cycle, sequence, cluster

The ``cycle`` column is the classification target; the integer ``<id>`` is the
sequence-identity clustering threshold used to build the train/val/test splits
(10..90).  Label-name -> index maps live in ``data/cyc_id_<id>_label_map.json``.

Pipeline
--------
1.  Obtain mean-pooled ESM-2 embeddings for the train split and an evaluation
    split (val or test).  Pre-computed embeddings in ``data/embeddings`` are
    reused automatically; otherwise they are extracted (batched, on GPU if
    available) and cached to parquet so a run is only paid for once.
2.  Fit each requested classifier on the train embeddings.
3.  Evaluate on the held-out split and report accuracy, balanced accuracy,
    macro/weighted F1, MCC and (where probabilities are available) top-k
    accuracy.
4.  Persist fitted models (``--models-dir``) and a tidy results table
    (``--results-csv``).

Examples
--------
Reuse the existing 650M mid-layer embeddings, evaluate RF + linear SVM on val::

    python ml_bench.py --id 30 --esm-ckpt facebook/esm2_t33_650M_UR50D \
        --embedding-col embedding_mid --eval-split val \
        --models rf linsvm logreg

Extract embeddings from scratch with the small 8M model, then benchmark::

    python ml_bench.py --id 10 --esm-ckpt facebook/esm2_t6_8M_UR50D \
        --layers 3 6 --embedding-col embedding_last --models rf linsvm knn
"""

import argparse
import json
import os
import time
import rmm
from rmm.allocators.torch import rmm_torch_allocator


import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    top_k_accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import torch

# INITIALIZE GH200 MEMORY MANAGEMENT
# 1. Configure RMM for GH200 Unified Memory
# 'managed_memory=True' is the critical flag that allows the GPU to 
# transparently access the Grace CPU's huge RAM (480GB+).

initial_size = 54 * 1024**3

rmm.reinitialize(
    pool_allocator=True,
    managed_memory=True,  # <--- MUST ENABLE FOR GH200 OVERSUBSCRIPTION
    initial_pool_size=initial_size, # Let it grow dynamically
)

# 2. Hot-swap the PyTorch allocator
# From this point on, every .to('cuda') or torch.tensor(..., device='cuda')
# uses RMM instead of the default PyTorch caching allocator.
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

print(f"Current Allocator: {torch.cuda.memory.get_allocator_backend()}")


# --------------------------------------------------------------------------- #
# Paths / conventions
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

LABEL_COL_CSV = "cycle"        # label column in the raw CSVs
LABEL_COL_EMB = "label"        # label column in the embedding parquet files
ID_COL = "id"

# Short tags appended to cached embedding filenames, keyed by ESM-2 checkpoint.
# These mirror the names already present in data/embeddings (e.g. *_650.parquet,
# the un-suffixed 8M files).  Unknown checkpoints fall back to a sanitised tag.
CKPT_SUFFIX = {
    "facebook/esm2_t6_8M_UR50D": "",
    "facebook/esm2_t12_35M_UR50D": "_35",
    "facebook/esm2_t30_150M_UR50D": "_150",
    "facebook/esm2_t33_650M_UR50D": "_650",
}


def ckpt_suffix(ckpt):
    """Return the embedding-file suffix for an ESM-2 checkpoint."""
    if ckpt in CKPT_SUFFIX:
        return CKPT_SUFFIX[ckpt]
    return "_" + ckpt.split("/")[-1].replace("/", "_")


# --------------------------------------------------------------------------- #
# Embeddings: load cached parquet or extract with a frozen ESM-2 backbone
# --------------------------------------------------------------------------- #
def embedding_path(split, sim_id, suffix):
    """Path to the (possibly cached) embedding parquet for a split."""
    return os.path.join(
        DATA_DIR, "embeddings", f"bgf_{split}_{sim_id}_embeddings{suffix}.parquet"
    )


def csv_path(split, sim_id):
    return os.path.join(DATA_DIR, split, f"bgf_{split}_{sim_id}.csv")


@np.errstate(all="ignore")
def extract_embeddings(csv_file, ckpt, layers, batch_size, max_len):
    """
    Mean-pool ESM-2 hidden states into one vector per sequence.

    Returns a DataFrame with columns: ids, label, embedding_mid, embedding_last
    (matching the schema produced by ``embed.py``).  ``layers`` is a two-element
    ``[mid, last]`` list of hidden-state indices.  Special tokens (BOS/EOS) and
    padding are excluded from the mean via the attention mask.
    """
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[embed] extracting from {csv_file} with {ckpt} on {device}")

    df = pd.read_csv(csv_file)
    seqs = df["sequence"].astype(str).tolist()
    ids = df[ID_COL].tolist()
    labels = df[LABEL_COL_CSV].tolist()

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = EsmModel.from_pretrained(ckpt, torch_dtype=torch.float).to(device).eval()

    mid_idx, last_idx = layers

    # Sort by length so each padded batch is as tight as possible, then restore
    # the original order at the end (keeps embeddings aligned with ids/labels).
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    emb_mid = [None] * len(seqs)
    emb_last = [None] * len(seqs)

    from tqdm import tqdm

    for start in tqdm(range(0, len(order), batch_size), desc="embedding", unit="batch"):
        idxs = order[start : start + batch_size]
        batch_seqs = [seqs[i] for i in idxs]
        enc = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        with torch.no_grad():
            hidden = model(**enc, output_hidden_states=True).hidden_states

        # Residue mask: attention mask minus BOS (first) and EOS (last real) token.
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0
        last_real = enc["attention_mask"].sum(dim=1) - 1
        mask[torch.arange(mask.size(0)), last_real] = 0
        mask = mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)

        for layer_idx, store in ((mid_idx, emb_mid), (last_idx, emb_last)):
            pooled = (hidden[layer_idx] * mask).sum(dim=1) / denom
            pooled = pooled.cpu().numpy()
            for row, i in enumerate(idxs):
                store[i] = pooled[row]

    return pd.DataFrame(
        {
            "ids": ids,
            "label": labels,
            "embedding_mid": [v.tolist() for v in emb_mid],
            "embedding_last": [v.tolist() for v in emb_last],
        }
    )


def get_embeddings(split, sim_id, ckpt, layers, batch_size, max_len, force):
    """Load cached embeddings if present, otherwise extract and cache them."""
    out_path = embedding_path(split, sim_id, ckpt_suffix(ckpt))
    if os.path.exists(out_path) and not force:
        print(f"[embed] reusing cached embeddings: {out_path}")
        return pd.read_parquet(out_path)

    src_csv = csv_path(split, sim_id)
    if not os.path.exists(src_csv):
        raise FileNotFoundError(
            f"No cached embeddings ({out_path}) and no source CSV ({src_csv})."
        )
    df = extract_embeddings(src_csv, ckpt, layers, batch_size, max_len)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), out_path)
    print(f"[embed] cached embeddings to {out_path}")
    return df


def stack_xy(df, embedding_col):
    """Return (X, y) as a float32 matrix and a 1-D label array."""
    X = np.asarray(df[embedding_col].tolist(), dtype=np.float32)
    y = np.asarray(df[LABEL_COL_EMB].tolist())
    return X, y


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
def build_classifiers(names, seed, n_jobs, rbf_max_samples):
    """
    Construct the requested estimators.

    Linear models are wrapped in a StandardScaler pipeline; tree models are
    scale-invariant and use the raw features.  RBF-SVM is O(n^2) so it is only
    sensible on a subsample -- that subsampling is handled by the caller and the
    cap is surfaced here purely for the run log.
    """
    factory = {
        "rf": lambda: RandomForestClassifier(
            n_estimators=300,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
        "linsvm": lambda: make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=5000),
        ),
        "rbfsvm": lambda: make_pipeline(
            StandardScaler(),
            SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed,
            ),
        ),
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                n_jobs=n_jobs,
                class_weight="balanced",
                #multi_class="multinomial",
            ),
        ),
        "knn": lambda: KNeighborsClassifier(
            n_neighbors=25, metric="cosine", n_jobs=n_jobs
        ),
    }
    unknown = set(names) - set(factory)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}. Choose from {sorted(factory)}.")
    return {name: factory[name]() for name in names}


def evaluate(name, clf, X_eval, y_eval, classes):
    """Predict and compute the metric bundle for one fitted classifier."""
    y_pred = clf.predict(X_eval)
    metrics = {
        "model": name,
        "n_eval": len(y_eval),
        "accuracy": accuracy_score(y_eval, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_eval, y_pred),
        "f1_macro": f1_score(y_eval, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_eval, y_pred),
        "top3_acc": np.nan,
        "top5_acc": np.nan,
    }

    # Top-k needs calibrated scores; only some estimators expose predict_proba.
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_eval)
        for k in (3, 5):
            if proba.shape[1] > k:
                metrics[f"top{k}_acc"] = top_k_accuracy_score(
                    y_eval, proba, k=k, labels=classes
                )
    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=2))
    rng = np.random.RandomState(args.seed)

    os.makedirs(args.models_dir, exist_ok=True)
    suffix = ckpt_suffix(args.esm_ckpt)

    # --- features ---------------------------------------------------------- #
    train_df = get_embeddings(
        "train", args.id, args.esm_ckpt, args.layers,
        args.batch_size, args.max_len, args.force_embed,
    )
    eval_df = get_embeddings(
        args.eval_split, args.id, args.esm_ckpt, args.layers,
        args.batch_size, args.max_len, args.force_embed,
    )

    X_train, y_train = stack_xy(train_df, args.embedding_col)
    X_eval, y_eval = stack_xy(eval_df, args.embedding_col)

    # Evaluate only on classes seen during training (others are unpredictable).
    seen = set(np.unique(y_train))
    keep = np.array([lbl in seen for lbl in y_eval])
    if not keep.all():
        dropped = (~keep).sum()
        print(f"[data] dropping {dropped} eval rows whose label is absent from train")
        X_eval, y_eval = X_eval[keep], y_eval[keep]

    if args.max_train > 0 and len(X_train) > args.max_train:
        sel = rng.choice(len(X_train), args.max_train, replace=False)
        X_train, y_train = X_train[sel], y_train[sel]
        print(f"[data] subsampled train to {args.max_train} rows")

    classes = np.unique(y_train)
    print(
        f"[data] features={X_train.shape[1]}  classes={len(classes)}  "
        f"train={len(X_train)}  eval({args.eval_split})={len(X_eval)}"
    )

    # --- fit + evaluate ---------------------------------------------------- #
    classifiers = build_classifiers(
        args.models, args.seed, args.n_jobs, args.rbf_max_samples
    )
    results = []
    for name, clf in classifiers.items():
        Xf, yf = X_train, y_train
        # RBF-SVM does not scale to hundreds of thousands of points; cap it.
        if name == "rbfsvm" and len(Xf) > args.rbf_max_samples:
            sel = rng.choice(len(Xf), args.rbf_max_samples, replace=False)
            Xf, yf = Xf[sel], yf[sel]
            print(f"[{name}] subsampling train to {args.rbf_max_samples} for RBF kernel")

        print(f"[{name}] fitting on {len(Xf)} examples ...")
        t0 = time.time()
        clf.fit(Xf, yf)
        fit_s = time.time() - t0

        metrics = evaluate(name, clf, X_eval, y_eval, classes)
        metrics["fit_seconds"] = round(fit_s, 1)
        metrics["n_train"] = len(Xf)
        results.append(metrics)
        print(
            f"[{name}] acc={metrics['accuracy']:.3f} "
            f"bal_acc={metrics['balanced_accuracy']:.3f} "
            f"f1_macro={metrics['f1_macro']:.3f} mcc={metrics['mcc']:.3f} "
            f"top5={metrics['top5_acc']:.3f}  ({fit_s:.0f}s)"
        )

        model_file = os.path.join(
            args.models_dir,
            f"{name}_bgf_{args.id}_{args.embedding_col}_esm{suffix or '8M'}.joblib",
        )
        joblib.dump(clf, model_file)
        print(f"[{name}] saved -> {model_file}")

    # --- report ------------------------------------------------------------ #
    res_df = pd.DataFrame(results)
    res_df.insert(0, "sim_id", args.id)
    res_df.insert(1, "esm_ckpt", args.esm_ckpt)
    res_df.insert(2, "embedding_col", args.embedding_col)
    res_df.insert(3, "eval_split", args.eval_split)

    cols = [
        "sim_id", "esm_ckpt", "embedding_col", "eval_split", "model",
        "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc",
        "top3_acc", "top5_acc", "n_train", "n_eval", "fit_seconds",
    ]
    res_df = res_df[cols]

    print("\n=== results ===")
    print(res_df.to_string(index=False))

    # Append to the results CSV so repeated runs (different id / ckpt) accumulate.
    header = not os.path.exists(args.results_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    res_df.to_csv(args.results_csv, mode="a", header=header, index=False)
    print(f"\n[results] appended to {args.results_csv}")


def parse_args():
    p = argparse.ArgumentParser(
        description="ESM-2 feature-extractor baseline (RF / SVM / etc.) for "
        "biogeochemical cycle classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--id", type=int, default=30,
                   help="Sequence-identity clustering threshold (10..90).")
    p.add_argument("--esm-ckpt", type=str, default="facebook/esm2_t33_650M_UR50D",
                   help="ESM-2 checkpoint used as the frozen feature extractor.")
    p.add_argument("--layers", nargs=2, type=int, default=[18, 33],
                   metavar=("MID", "LAST"),
                   help="Hidden-state layer indices [mid, last] when extracting "
                        "embeddings (ignored if cached embeddings are reused).")
    p.add_argument("--embedding-col", choices=["embedding_mid", "embedding_last"],
                   default="embedding_mid", help="Which pooled layer to classify on.")
    p.add_argument("--eval-split", choices=["val", "test"], default="val",
                   help="Held-out split to evaluate on.")
    p.add_argument("--models", nargs="+",
                   default=["rf", "linsvm", "logreg"],
                   help="Classifiers to run: rf linsvm rbfsvm logreg knn.")
    p.add_argument("--models-dir", type=str,
                   default=os.path.normpath(os.path.join(HERE, "..", "models")))
    p.add_argument("--results-csv", type=str,
                   default=os.path.normpath(os.path.join(HERE, "..", "results", "ml_bench_results.csv")))
    p.add_argument("--max-train", type=int, default=-1,
                   help="Cap training rows for all models (<=0 = use all).")
    p.add_argument("--rbf-max-samples", type=int, default=20000,
                   help="Subsample cap for the (quadratic) RBF-SVM.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size when extracting embeddings.")
    p.add_argument("--max-len", type=int, default=1024,
                   help="Max sequence length when extracting embeddings.")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-embed", action="store_true",
                   help="Re-extract embeddings even if a cached parquet exists.")
    return p.parse_args()


if __name__ == "__main__":
    main()
