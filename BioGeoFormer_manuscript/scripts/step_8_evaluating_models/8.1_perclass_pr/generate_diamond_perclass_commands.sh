#!/bin/bash
# Writes DIAMOND makedb and blastp commands for one database per pathway class per split
# (--max-target-seqs 1). Expects BASE_DIR/train_by_class/<N>/<class>.fasta and
# BASE_DIR/bgf_test_<N>.fasta. Set BASE_DIR, run, then submit with the two array scripts.

set -euo pipefail

BASE_DIR="/nfs5/PHARM/David_Lab/JACOB/graphpart_diamondrun"
THREADS=4   # per-task diamond threads -- matches --cpus-per-task in run_diamond_makedb_array.slurm,
            # kept low (not 32) so many array tasks run concurrently within the 128-thread/400G budget
            # rather than one task hogging a big slice of it for a lightweight per-class makedb

DB_FILE="perclass_diamond_makedb.txt"
RUN_FILE="perclass_diamond_run.txt"
: > "$DB_FILE"
: > "$RUN_FILE"

splits=(10 20 30 40 50 60 70 80 90)

# --- Step 1: makedb commands, one per (split, class) ---
for T in "${splits[@]}"; do
    CLASS_DIR="$BASE_DIR/train_by_class/$T"
    for FASTA in "$CLASS_DIR"/*.fasta; do
        CLASS=$(basename "$FASTA" .fasta)
        DB="$BASE_DIR/perclass_dbs/$T/${CLASS}_db"
        mkdir -p "$BASE_DIR/perclass_dbs/$T"
        CMD="diamond makedb --in $FASTA -d $DB -p $THREADS"
        echo "$CMD" >> "$DB_FILE"
    done
done
echo "Diamond makedb commands created at: $DB_FILE"

# --- Step 2: blastp commands, one per (split, class), test set only ---
# (add SPLIT in (val test) here too if you also want per-class val scores)
for T in "${splits[@]}"; do
    if [ "$T" -le 30 ]; then
        SENS="--ultra-sensitive"
    elif [ "$T" -le 50 ]; then
        SENS="--sensitive"
    else
        SENS="--fast"
    fi

    QUERY="$BASE_DIR/bgf_test_${T}.fasta"
    CLASS_DIR="$BASE_DIR/train_by_class/$T"
    OUT_DIR="$BASE_DIR/perclass_bench/$T"
    mkdir -p "$OUT_DIR"

    for FASTA in "$CLASS_DIR"/*.fasta; do
        CLASS=$(basename "$FASTA" .fasta)
        DB="$BASE_DIR/perclass_dbs/$T/${CLASS}_db.dmnd"
        OUTPUT="$OUT_DIR/bench_test_${T}_vs_${CLASS}.tsv"
        CMD="diamond blastp -d $DB -q $QUERY -o $OUTPUT $SENS --max-target-seqs 1 --evalue 100000 --outfmt 6 qseqid sseqid pident length evalue bitscore -p $THREADS"
        echo "$CMD" >> "$RUN_FILE"
    done
done
echo "Benchmark blastp commands created at: $RUN_FILE"

n_makedb=$(wc -l < "$DB_FILE")
n_run=$(wc -l < "$RUN_FILE")
echo ""
echo "$n_makedb makedb commands, $n_run blastp commands generated."
