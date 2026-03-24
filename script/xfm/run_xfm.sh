python train_xfm.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --prediction="v" \
    --path-type="cosine" \
    --weighting="uniform" \
    --data-path="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/data" \
    --output-dir="/mnt/shared-storage-user/ai4sdata2-share/maiweijian/BrainVL/NeuroFlow/train_logs" \
    --wandb-log \
    --batch-size=24 \
    --sampling-steps=10000 \
    --checkpointing-steps=2500 \
    --max-train-steps=50001 \
    --subject=1 \
    --encoder="vae" \
    --vae-path="neurovae-nsd-s1-vs1-bs64-d1664-zscore-v10-cycle-proj" \
    --hidden-dim=1664 \
    --linear-dim=1024 \
    --model-depth=12 \
    --model-head=13 \
    --num-steps=20 \
    --exp-name="fm-s1-d12-h13-bs24-v-cos-uni-d1664-zscore-v10-cycle-reverse-proj" \
    # --exp-name="tryfm" \
    # --resume \
    # --resume-id="44bopw0r" \

    # nohup bash run_xfm.sh > logs/fm_s1_cos_uni_v10_reverse_proj_d8.log 2>&1 &
    # nohup bash run_sit_multistep.sh > logs/fm_s1_ms_v1_cos_uni_stoch01.log 2>&1 &
    # nohup bash run_sit_multistep.sh > logs/fm_s1_ms_v1_cos_uni_end_mse_heun.log 2>&1 &
    # nohup bash run_sit_multistep.sh > logs/fm_s1_ms_v1_step.log 2>&1 &