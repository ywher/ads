#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSDA_ROOT="${1:-${SSDA_ROOT:-}}"

if [[ -z "${SSDA_ROOT}" ]]; then
  echo "Usage: bash scripts/road/setup_assets.sh /path/to/SSDA" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/data/road" "${REPO_ROOT}/pretrained"
for dataset in cityscapes gta synthia acdc muses mapillary; do
  source_path="${SSDA_ROOT}/data/${dataset}"
  if [[ ! -e "${source_path}" ]]; then
    echo "Missing dataset: ${source_path}" >&2
    exit 1
  fi
  ln -sfn "${source_path}" "${REPO_ROOT}/data/road/${dataset}"
done
ln -sfn "${SSDA_ROOT}/splits" "${REPO_ROOT}/data/road/splits"

DINO_PATH="${SSDA_ROOT}/pretrained/dinov3/dinov3_vitb16.pth"
if [[ -f "${DINO_PATH}" ]]; then
  ln -sfn "${DINO_PATH}" "${REPO_ROOT}/pretrained/dinov3_vitb16.pth"
else
  echo "Warning: DINOv3-B checkpoint is missing: ${DINO_PATH}" >&2
fi

echo "Road datasets and split files are linked under data/road/."
echo "Place the official ADS COCO DeepLab checkpoint at:"
echo "  ${REPO_ROOT}/pretrained/MS_DeepLab_resnet_pretrained_COCO_init.pth"
