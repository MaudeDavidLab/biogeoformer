#!/bin/bash
#SBATCH --job-name=diamond_blastp
#SBATCH --account=zoidberg
#SBATCH --partition=zoidberg
#SBATCH --cpus-per-task=4               # matches -p in generate_diamond_perclass_commands.sh's THREADS
#SBATCH --mem=8G                        # blastp needs more headroom than makedb, esp. --ultra-sensitive on the larger classes
#SBATCH --time=04:00:00                 # rough estimate -- --ultra-sensitive (splits <=30%) is much slower than --fast (splits >50%); adjust if tasks time out
#SBATCH --output=logs/blastp_%A_%a.out
#SBATCH --error=logs/blastp_%A_%a.err
#SBATCH --array=1-292                   # <-- EDIT: set to `wc -l < perclass_diamond_run.txt` (should also be 292, but confirm)

# One per-class DIAMOND blastp per array task (line N of perclass_diamond_run.txt); run after makedb.
# Usage: sbatch --array=1-$(wc -l < perclass_diamond_run.txt)%32 run_diamond_blastp_array.slurm

set -euo pipefail

COMMANDS_FILE="perclass_diamond_run.txt"

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMANDS_FILE")
if [ -z "$CMD" ]; then
    echo "No command found for array task $SLURM_ARRAY_TASK_ID in $COMMANDS_FILE" >&2
    exit 1
fi

echo "[task $SLURM_ARRAY_TASK_ID] $CMD"
eval "$CMD"
