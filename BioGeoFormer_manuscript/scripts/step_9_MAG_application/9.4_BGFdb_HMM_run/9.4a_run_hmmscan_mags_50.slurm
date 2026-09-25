#!/bin/bash
#SBATCH --job-name=hmmscan_mags_50
#SBATCH --account=zoidberg
#SBATCH --partition=zoidberg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs5/PHARM/David_Lab/JACOB/hmm_runs_jul7/scan_results_mags_50/logs/hmmscan_%j.out
#SBATCH --error=/nfs5/PHARM/David_Lab/JACOB/hmm_runs_jul7/scan_results_mags_50/logs/hmmscan_%j.err

OUT_DIR=/nfs5/PHARM/David_Lab/JACOB/hmm_runs_jul7/scan_results_mags_50
mkdir -p "$OUT_DIR/logs"

HMM_DB=/nfs5/PHARM/David_Lab/JACOB/hmm_runs_jul7/hmm_dbs/bgf_train_50_combined.hmm
QUERY=/nfs5/MICRO/Thurber_Lab/jacob/hmm_runs/deepsea_mags_application/MAGs_combined_prodigal_cleaned.faa

hmmscan --cpu "$SLURM_CPUS_PER_TASK" -E 1e10 --domE 1e10 --noali \
  --tblout "$OUT_DIR/MAGs_vs_bgf_train_50.tblout" \
  --domtblout "$OUT_DIR/MAGs_vs_bgf_train_50.domtblout" \
  -o /dev/null \
  "$HMM_DB" \
  "$QUERY"
