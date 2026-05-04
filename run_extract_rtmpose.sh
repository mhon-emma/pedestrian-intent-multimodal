#!/bin/bash
# Launcher for extract_rtmpose.py — sets up CUDA lib paths for onnxruntime-gpu
# Usage:  bash run_extract_rtmpose.sh [--sets set01 set03 ...] [--update]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMML_ENV="mmml"

SITE_PKG="/home/teamj/miniconda3/envs/${MMML_ENV}/lib/python3.10/site-packages"
NVIDIA_BASE="${SITE_PKG}/nvidia"

# Collect all nvidia sub-package lib dirs
NVIDIA_LIBS=""
for d in "${NVIDIA_BASE}"/*/lib; do
    [ -d "$d" ] && NVIDIA_LIBS="${d}:${NVIDIA_LIBS}"
done

export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

exec conda run --no-capture-output -n "${MMML_ENV}" python3 -u "${SCRIPT_DIR}/extract_rtmpose.py" "$@"
