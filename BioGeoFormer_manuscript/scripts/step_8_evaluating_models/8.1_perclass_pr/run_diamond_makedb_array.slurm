#!/bin/bash
#SBATCH --job-name=diamond_makedb
#SBATCH --account=zoidberg
#SBATCH --partition=zoidberg
#SBATCH --cpus-per-task=4               # matches -p in generate_diamond_perclass_commands.sh's THREADS
#SBATCH --mem=4G                        # covers even the largest class (~162k seqs, "purine") comfortably
#SBATCH --time=00:30:00
#SBATCH --output=logs/makedb_%A_%a.out
#SBATCH --error=logs/makedb_%A_%a.err
#SBATCH --array=1-292                   # <-- EDIT: set to `wc -l < perclass_diamond_makedb.txt` (count varies by BASE_DIR)

# One per-class DIAMOND makedb per array task (line N of perclass_diamond_makedb.txt).
# Usage: sbatch --array=1-$(wc -l < perclass_diamond_makedb.txt)%32 run_diamond_makedb_array.slurm

set -euo pipefail

COMMANDS_FILE="perclass_diamond_makedb.txt"

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")
if [ -z "$CMD" ]; then
    echo "No command found for array task $SLURM_ARRAY_TASK_ID in $COMMANDS_FILE" >&2
    exit 1
fi

echo "[task $SLURM_ARRAY_TASK_ID] $CMD"
eval "$CMD"
