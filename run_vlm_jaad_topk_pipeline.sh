#!/bin/bash
# run_vlm_jaad_topk_pipeline.sh
# ================================
# Waits for GPU headroom, then runs the JAAD VLM LoRA top-K
# checkpoint-robustness retraining + cross-dataset eval, entirely
# unattended. Meant to run inside tmux so it survives disconnects.
#
# Usage: tmux new-session -d -s vlm_jaad_topk 'bash run_vlm_jaad_topk_pipeline.sh'

set -uo pipefail
cd /usr1/home/mehon/emma_pedestrian-intent-multimodal
source /usr1/home/mehon/venvs/sfgru-train/bin/activate
mkdir -p logs

REQUIRED_MB=20000
POLL_SECS=300

echo "[$(date '+%H:%M:%S')] Waiting for >=${REQUIRED_MB}MiB free on a GPU..."
GPU=""
while true; do
    while IFS=, read -r idx free _; do
        idx=$(echo "$idx" | tr -d ' ')
        free=$(echo "$free" | tr -d ' MiB')
        if [ "$free" -ge "$REQUIRED_MB" ]; then
            GPU="$idx"
            break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)

    if [ -n "$GPU" ]; then
        echo "[$(date '+%H:%M:%S')] GPU $GPU has enough free memory -- proceeding."
        break
    fi
    echo "[$(date '+%H:%M:%S')] Not enough free GPU memory yet, sleeping ${POLL_SECS}s..."
    sleep "$POLL_SECS"
done

echo "[$(date '+%H:%M:%S')] === Stage 1: JAAD VLM LoRA top-K training ==="
CUDA_VISIBLE_DEVICES="$GPU" python3 vlm_lora_finetune_behonly.py \
    --dataset jaad --epochs 3 --top_k 3 \
    > logs/vlm_jaad_topk_retrain.log 2>&1
TRAIN_EXIT=$?
echo "[$(date '+%H:%M:%S')] Training exited with code $TRAIN_EXIT"
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[$(date '+%H:%M:%S')] Training FAILED -- aborting pipeline, see logs/vlm_jaad_topk_retrain.log"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] === Stage 2: cross-dataset eval, all top-K checkpoints ==="
CUDA_VISIBLE_DEVICES="$GPU" python3 vlm_lora_cross_dataset_eval_topk_behonly.py \
    --direction jaad_to_pie \
    > logs/vlm_jaad_topk_crossdataset.log 2>&1
EVAL_EXIT=$?
echo "[$(date '+%H:%M:%S')] Cross-dataset eval exited with code $EVAL_EXIT"

echo "[$(date '+%H:%M:%S')] === Pipeline complete (train_exit=$TRAIN_EXIT, eval_exit=$EVAL_EXIT) ==="
