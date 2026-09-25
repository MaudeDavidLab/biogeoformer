#!/bin/bash

#SBATCH -J bgf_150M_1ep
#SBATCH --time=24:00:00
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH -o jobs/bgf_150M_1ep.out
#SBATCH -e jobs/bgf_150M_1ep.err
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --constraint=vram80g
#SBATCH --mail-user=azbijarn@oregonstate.edu
#SBATCH --mail-type=ALL

set -euo pipefail

ESM_CKPT="facebook/esm2_t30_150M_UR50D"
TAG="150M"                 # keep in sync with ESM_CKPT

# gLM2 instead of ESM2 (same 640-dim / 30-layer size, see backbones.py):
#ESM_CKPT="tattabio/gLM2_150M"
#TAG="glm2_150M"
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

# Every run appends one row here; the file is created on first use and is
# never rewritten, so point all experiments at the same path.
EXP_LOG="../results/experiments/experiments_train_runs_full_epoch.csv"

for ID in $(seq 10 10 90); do
    TR_DATA="../data/train/bgf_train_${ID}.csv"
    EVAL_DATA="../data/val/bgf_val_${ID}.csv"
    MODEL_DIR="../models/bgf_${TAG}_${ID}_1ep"

    # --- pre-flight: fail before burning GPU time, not after ---
    for f in "$TR_DATA" "$EVAL_DATA"; do
        if [[ ! -f "$f" ]]; then
            echo "[id=${ID}] MISSING: $f -- skipping" >&2
            continue 2
        fi
    done

    if [[ -d "$MODEL_DIR" ]]; then
        echo "[id=${ID}] $MODEL_DIR exists -- skipping (rm -rf to force)" >&2
        continue
    fi

    echo "=== [id=${ID}] $(date +%H:%M:%S) -> ${MODEL_DIR}"

    python train.py \
        --esm_ckpt "$ESM_CKPT" \
        --id "$ID" \
        --tr_data "$TR_DATA" \
        --eval_data "$EVAL_DATA" \
        --data_seed 42 \
        --seq_per_label -1 \
        --model_dir "$MODEL_DIR" \
        --total_steps 2000 \
        --warmup_ratio 0.1 \
        --beta_1 0.9 \
        --beta_2 0.99 \
        --weight_decay 0.1 \
        --epochs 1 \
        --max_grad_norm 0.5 \
        --bs 2 \
        --grad_accum_steps 64 \
        --label_smoothing 0.1 \
        --exp_log "$EXP_LOG" \
        --notes "${TAG} full epoch lora" \
		--lora \
		--lora_rank 16 \
		--backbone_lr 1e-3 \
		--head_lr 1e-3 \
        
        2>&1 | tee "${LOG_DIR}/bgf_${TAG}_${ID}_final.log"

    echo "=== [id=${ID}] done $(date +%H:%M:%S)"
done

echo "All runs finished."


# --lora \
#         --lora_rank 16 \
#         --backbone_lr 1e-3 \
#         --head_lr 1e-3 \
