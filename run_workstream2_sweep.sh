#!/bin/bash
# run_workstream2_sweep.sh
# ===========================
# Drives the full "test all 7 architectures cross-dataset" sweep
# (workstream 2). Runs training sequentially per GPU (GPU0 = PIE runs,
# GPU1 = JAAD runs) so feature-cache writes for a given dataset+arch
# don't race, then launches the cross-dataset eval for each architecture
# once both directions are trained.
#
# Architectures:
#   Attention family (cross_dataset_eval_attention.py):
#     pose_box, pose_pose, modality_fusion, cross_modal, other_modal
#   Fusion family (cross_dataset_fusion_eval.py), remaining untested:
#     gated_fusion, uncertainty
#   (cross_attn already done -- see results/cross_dataset_fusion_audit_cross_attn.pkl)
#
# Usage: bash run_workstream2_sweep.sh 2>&1 | tee logs/workstream2_sweep.log

set -e
cd /usr1/home/mehon/emma_pedestrian-intent-multimodal
source /usr1/home/mehon/venvs/sfgru-train/bin/activate
mkdir -p logs results

ATTN_ARCHS="pose_pose modality_fusion cross_modal other_modal"
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

# --- Attention family: pose_box already launched separately (PIE side in flight). ---
echo "[$(date +%H:%M:%S)] Waiting for pose_box PIE run (launched earlier) to finish..."
while pgrep -f "train_full_pie_attention_nospeed.py --architecture pose_box" > /dev/null; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] pose_box PIE run finished."

run_jaad_attn pose_box
python3 cross_dataset_eval_attention.py --architecture pose_box > logs/crossaudit_attn_pose_box.log 2>&1
echo "[$(date +%H:%M:%S)] pose_box cross-dataset audit done"

for arch in $ATTN_ARCHS; do
  run_pie_attn "$arch" &
  PIE_PID=$!
  run_jaad_attn "$arch" &
  JAAD_PID=$!
  wait $PIE_PID $JAAD_PID
  python3 cross_dataset_eval_attention.py --architecture "$arch" > "logs/crossaudit_attn_${arch}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $arch cross-dataset audit done"
done

# --- Fusion family: gated_fusion, uncertainty (cross_attn already done) ---
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
