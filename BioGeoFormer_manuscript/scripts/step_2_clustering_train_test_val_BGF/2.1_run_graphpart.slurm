#!/bin/bash
#SBATCH --job-name=bgf_graphpart
#SBATCH --output=graphpart_%j.out
#SBATCH --error=graphpart_%j.err
#SBATCH --account=zoidberg
#SBATCH --partition=zoidberg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=750G
#SBATCH --time=48:00:00

# Example for the 30% threshold; run once per threshold (10-90%) with --threshold set accordingly.
# --test-ratio 0.4: 60% train, 40% test+val (split 50/50 in 2.2).

# Navigate to the cluster working directory
cd /home/micro/wynneja/jacob_david_lab/graph_30_twosets

# Manually source conda and activate environment
export CONDA_PATH=$(find $HOME -name "conda.sh" -path "*/etc/profile.d/conda.sh" | head -n 1)
if [ -z "$CONDA_PATH" ]; then
    echo "Could not find conda.sh. Please check your miniconda/anaconda installation path."
    exit 1
fi

source "$CONDA_PATH"
conda activate graphpart_new

# Define local filenames
FASTA_IN="bgf_reps_90_for_graphpart.fasta"
OUT_CSV="BioGeoFormer_stratified_splits.csv"

echo "Starting stratified homology-aware partitioning..."
echo "Method: mmseqs2 | Threshold: 0.3 | Test: 0.4"

# Removed --threads and --tmpdir as they were unrecognized in your version
# The tool will typically use available cores automatically or default to a safe temp path
graphpart mmseqs2 \
  --fasta-file $FASTA_IN \
  --threshold 0.3 \
  --out-file $OUT_CSV \
  --labels-name class \
  --test-ratio 0.4 \
  --prefilter \
  --initialization-mode fast-nn \
  --save-checkpoint-path checkpoint

echo "Partitioning complete. Results saved to $OUT_CSV"
