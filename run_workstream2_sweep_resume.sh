#!/bin/bash
# run_workstream2_sweep_resume.sh
# ===================================
# Resumes run_workstream2_sweep.sh from where it was interrupted.
# pose_box (attention) and cross_attn (fusion) are already done --
# both were rerun with the get_path monkeypatch fix (see
# cross_dataset_eval_attention.py / cross_dataset_fusion_eval.py) after
# the original sweep driver died (set -e killed it when its
# pose_box audit child was killed to stop a run using corrupted,
# 100%-pose-missing JAAD->PIE data).
#
# Remaining: pose_pose, modality_fusion, cross_modal, other_modal
# (attention family) + gated_fusion, uncertainty (fusion family).

set -e
cd /usr1/home/mehon/emma_pedestrian-intent-multimodal
source /usr1/home/mehon/venvs/sfgru-train/bin/activate
mkdir -p logs results

ATTN_ARCHS="modality_fusion cross_modal other_modal"
FUSION_ARCHS="gated_fusion uncertainty"

run_pie_attn() {
  local arch=$1
  echo "[$(date +%H:%M:%S)] PIE attention/$arch starting"
  CUDA_VISIBLE_DEVICES=0 python3 train_full_pie_attention_nospeed.py \
    --architecture "$arch" --seeds 3 --backend rtmpose \
    > "logs/pie_attn_${arch}.log" 2>&1
  echo "[$(date +%H:%M:%S)] PIE attention/$arch done"
}

run_jaad_attn() {
  local arch=$1
  echo "[$(date +%H:%M:%S)] JAAD attention/$arch starting"
  CUDA_VISIBLE_DEVICES=1 python3 train_full_jaad_attention_nospeed.py \
    --architecture "$arch" --seeds 3 --backend rtmpose \
    > "logs/jaad_attn_${arch}.log" 2>&1
  echo "[$(date +%H:%M:%S)] JAAD attention/$arch done"
}

run_pie_fusion() {
  local arch=$1
  echo "[$(date +%H:%M:%S)] PIE fusion/$arch starting"
  CUDA_VISIBLE_DEVICES=0 python3 train_full_pie_fusion_nospeed.py \
    --architecture "$arch" --seeds 3 --backend rtmpose \
    > "logs/pie_fusion_${arch}.log" 2>&1
  echo "[$(date +%H:%M:%S)] PIE fusion/$arch done"
}

run_jaad_fusion() {
  local arch=$1
  echo "[$(date +%H:%M:%S)] JAAD fusion/$arch starting"
  CUDA_VISIBLE_DEVICES=1 python3 train_full_jaad_fusion_nospeed.py \
    --architecture "$arch" --seeds 3 --backend rtmpose \
    > "logs/jaad_fusion_${arch}.log" 2>&1
  echo "[$(date +%H:%M:%S)] JAAD fusion/$arch done"
}

for arch in $ATTN_ARCHS; do
  run_pie_attn "$arch" &
  PIE_PID=$!
  run_jaad_attn "$arch" &
  JAAD_PID=$!
  wait $PIE_PID $JAAD_PID
  python3 cross_dataset_eval_attention.py --architecture "$arch" > "logs/crossaudit_attn_${arch}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $arch cross-dataset audit done"
done

for arch in $FUSION_ARCHS; do
  run_pie_fusion "$arch" &
  PIE_PID=$!
  run_jaad_fusion "$arch" &
  JAAD_PID=$!
  wait $PIE_PID $JAAD_PID
  python3 cross_dataset_fusion_eval.py --architecture "$arch" > "logs/crossaudit_fusion_${arch}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $arch (fusion) cross-dataset audit done"
done

echo "[$(date +%H:%M:%S)] Workstream 2 sweep complete: all 7 architectures trained + cross-dataset audited."
