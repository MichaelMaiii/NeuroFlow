#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python train_neurovae.py \
    --seed=1024 \
    --subject=1 \
    --batch_size=64 \
    --test_batch_size=300 \
    --num_epochs=30 \
    --model_name="neurovae-nsd-s1-bs64-d1664-zscore-v10-cycle-proj" \
    --hidden_dim=1664 \
    --linear_dim=1024 \
    --embed_dim=1664 \
    --clip_weight=1000 \
    --cycle_weight=1000 \
    --kl_weight=0.001 \
    --base_lr=1e-4 \
    --ckpt_interval=1 \
    --data_path="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/data" \
    --save_path="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow" \
    --zscore

# Optional flags (uncomment and append to the command above if needed):
#   --resume --resume_id="wandb_run_id"
#   --finetune --ckpt_name="previous_run_name"
# wandb_log / plot_recon default to True in train_neurovae.py

# nohup bash run_neurovae.sh > logs/neurovae_s1.log 2>&1 &
