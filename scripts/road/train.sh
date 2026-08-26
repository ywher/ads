#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/road/train.sh CONFIG [SCREEN_NAME]" >&2
  exit 2
fi

CONFIG="$1"
SCREEN_NAME="${2:-ads_$(basename "${CONFIG}" .yaml)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -t 1 && "${ADS_USE_SCREEN:-1}" == "1" ]]; then
  screen -dmS "${SCREEN_NAME}" bash -lc \
    "cd '${REPO_ROOT}' && conda activate reinpy10 && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}' python -u train_road.py --config '${CONFIG}' 2>&1 | tee '${SCREEN_NAME}.log'"
  echo "Started screen session: ${SCREEN_NAME}"
  echo "Attach with: screen -r ${SCREEN_NAME}"
else
  python -u train_road.py --config "${CONFIG}"
fi
