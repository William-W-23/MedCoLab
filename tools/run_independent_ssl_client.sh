#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEDCOLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

: "${DATA_ROOT:?DATA_ROOT is required}"
: "${TASK_PREFIX:?TASK_PREFIX is required}"
: "${CLIENT_INDEX:?CLIENT_INDEX is required}"

GPU_INDEX="${GPU_INDEX:-0}"
ROUNDS="${ROUNDS:-100}"
TRAIN_MAX_BATCHES="${TRAIN_MAX_BATCHES:-0}"
MASTER_SEED="${MASTER_SEED:-42}"
MANIFEST_SHA256="${MANIFEST_SHA256:-}"
MODEL_VARIANT="${MODEL_VARIANT:-RTDETR_L}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-0.0001}"
IMAGE_SIZE="${IMAGE_SIZE:-320}"
SSL_MASK_ADAPTIVE="${SSL_MASK_ADAPTIVE:-false}"
SSL_MEAN_SAMPLES="${SSL_MEAN_SAMPLES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
DATASET_NAME="${TASK_PREFIX}_moco_ssl_client${CLIENT_INDEX}_seed${MASTER_SEED}"
VIEW_ROOT="$DATA_ROOT/independent_ssl_views/model_client${CLIENT_INDEX}"
OUTPUT_DIR="$OUTPUT_ROOT/ssl_moco_${MODEL_VARIANT}_Independent_${TASK_PREFIX}_client${CLIENT_INDEX}_seed${MASTER_SEED}"

if [[ ! -d "$VIEW_ROOT/client0/ssl_unlabeled/images/train" ]]; then
  echo "Missing independent SSL view: $VIEW_ROOT" >&2
  exit 2
fi
if [[ -f "$OUTPUT_DIR/server_result.json" ]]; then
  echo "Completed output already exists: $OUTPUT_DIR"
  python tools/finalize_independent_ssl.py "$OUTPUT_DIR"
  exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export MASTER_SEED PYTHONHASHSEED="$MASTER_SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export SOURCE_CLIENT="client${CLIENT_INDEX}"
export DATA_MANIFEST_SHA256="$MANIFEST_SHA256"
export CODE_VERSION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
export EXPERIMENT_STARTED_AT="$(date -Iseconds)"
CMD=(python tools/train_independent_ssl.py
  --data-root "$VIEW_ROOT"
  --output "$OUTPUT_DIR"
  --dataset-name "$DATASET_NAME"
  --source-client "client${CLIENT_INDEX}"
  --manifest-sha256 "$MANIFEST_SHA256"
  --rounds "$ROUNDS"
  --local-epochs "$LOCAL_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --image-size "$IMAGE_SIZE"
  --max-batches "$TRAIN_MAX_BATCHES"
  --seed "$MASTER_SEED"
  --model-variant "$MODEL_VARIANT"
)
if [[ "$SSL_MASK_ADAPTIVE" == true ]]; then
  CMD+=(--mask-adaptive --mean-samples "$SSL_MEAN_SAMPLES")
fi

printf 'Running independent SSL client%s on GPU %s:\n' "$CLIENT_INDEX" "$GPU_INDEX"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
python tools/finalize_independent_ssl.py "$OUTPUT_DIR"
