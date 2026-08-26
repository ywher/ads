#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/road/wait_gpu_then_train.sh GPU CONFIG [FREE_MIB]" >&2
  exit 2
fi

GPU="$1"
CONFIG="$2"
FREE_MIB="${3:-20000}"
POLL_SECONDS="${ADS_GPU_POLL_SECONDS:-60}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Waiting for GPU ${GPU} to have at least ${FREE_MIB} MiB free..."
while true; do
  free_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= FREE_MIB )); then
    sleep "${POLL_SECONDS}"
    free_check="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_check}" =~ ^[0-9]+$ ]] && (( free_check >= FREE_MIB )); then
      break
    fi
  fi
  sleep "${POLL_SECONDS}"
done

source "${CONDA_SH:-/home/ywh/anaconda3/etc/profile.d/conda.sh}"
conda activate reinpy10
cd "${REPO_ROOT}"
echo "GPU ${GPU} is free; starting ${CONFIG}."
CUDA_VISIBLE_DEVICES="${GPU}" python -u train_road.py --config "${CONFIG}"
