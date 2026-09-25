"""Split each training set (BGF_clustering/train_test_val_final/bgf_train_<N>.csv) into one
FASTA per pathway class, for per-class DIAMOND databases (train_by_class/<N>/<class>.fasta).
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "8.2_similarity_binned_eval"))
from run_evaluation import GRAPHPART_DIR, SPLITS

OUT_DIR = GRAPHPART_DIR / "train_by_class"


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def write_fasta(path, ids, sequences):
    with open(path, "w") as f:
        for seq_id, seq in zip(ids, sequences):
            f.write(f">{seq_id}\n{seq}\n")


if __name__ == "__main__":
    summary_rows = []
    for split in SPLITS:
        df = pd.read_csv(GRAPHPART_DIR / f"bgf_train_{split}.csv", usecols=["id", "cycle", "sequence"])
        split_dir = OUT_DIR / str(split)
        split_dir.mkdir(parents=True, exist_ok=True)

        for cls, group in df.groupby("cycle"):
            out_path = split_dir / f"{sanitize(cls)}.fasta"
            write_fasta(out_path, group["id"], group["sequence"])
            summary_rows.append({"split": split, "class": cls, "n_sequences": len(group)})

        n_classes = df["cycle"].nunique()
        print(f"split {split}: {n_classes} classes, {len(df)} total train sequences -> {split_dir}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "class_split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary_df.groupby("split")["n_sequences"].agg(["count", "min", "median", "max"]))
