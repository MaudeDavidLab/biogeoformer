#!/bin/bash

#SBATCH -J ml_bench
#SBATCH --time=60-00:00:00
#SBATCH -A david_gh
#SBATCH -p david_gh
#SBATCH -o jobs/bgf_ml_bench.out
#SBATCH -e jobs/bgf_ml_bench.err
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --mail-user=azbijarn@oregonstate.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=32

export HF_HOME=/nfs5/PHARM/David_Lab/NIMA/data
#export TORCH_USE_CUDA_DSA=1
#export OMP_NUM_THREADS=1


# ESM-2 checkpoints to benchmark, each paired with its [mid last] hidden-state
# layer indices (models have 6 / 30 / 33 transformer layers respectively).
#   8M   -> facebook/esm2_t6_8M_UR50D     layers 3 6
#   150M -> facebook/esm2_t30_150M_UR50D  layers 15 29
#   650M -> facebook/esm2_t33_650M_UR50D  layers 18 33
declare -A ESM_LAYERS=(
    ["facebook/esm2_t6_8M_UR50D"]="3 6"
    ["facebook/esm2_t12_35M_UR50D"]="6 12"
    ["facebook/esm2_t30_150M_UR50D"]="15 29"
)

IDS=(10 20 30 40 50 60 70 80 90)

for ckpt in "facebook/esm2_t6_8M_UR50D" \
            "facebook/esm2_t12_35M_UR50D" \
            "facebook/esm2_t30_150M_UR50D"; do
    layers=${ESM_LAYERS[$ckpt]}
    for id in "${IDS[@]}"; do
        python ml_bench.py \
            --id "$id" \
            --esm-ckpt "$ckpt" \
            --layers $layers \
            --embedding-col embedding_mid \
            --eval-split test \
            --models rf logreg knn
    done
done
