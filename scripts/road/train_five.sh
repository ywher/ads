#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-vfm}"
PROTOCOL="${2:-ssda}"
GPU_LIST="${3:-0,1,2,3,4}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
TASKS=(gta2cityscapes synthia2cityscapes cityscapes2mapillary cityscapes2acdc cityscapes2muses)

if [[ ${#GPUS[@]} -lt ${#TASKS[@]} ]]; then
  echo "Five GPU ids are required, e.g. 0,1,2,3,4" >&2
  exit 2
fi

for index in "${!TASKS[@]}"; do
  task="${TASKS[$index]}"
  gpu="${GPUS[$index]}"
  config="configs/road/${MODEL}/${PROTOCOL}/${task}.yaml"
  CUDA_VISIBLE_DEVICES="${gpu}" bash scripts/road/train.sh \
    "${config}" "ads_${MODEL}_${PROTOCOL}_${task}"
done
