#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${EXP_ID:?set EXP_ID}"
GPU_ID="${GPU_ID:?set GPU_ID}"
ROUNDS="${ROUNDS:?set ROUNDS}"
CLIENT_LR="${CLIENT_LR:-0.00005}"
SERVER_LR="${SERVER_LR:-0.002}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
ROOT="${MEDCOLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${MEDCOLAB_ARTIFACT_ROOT:-$ROOT/outputs}"
RUN_ROOT="$ARTIFACT_ROOT/fedbn_fedyogi_runs/detection"
OUT="$RUN_ROOT/$EXP_ID"
BN="$ARTIFACT_ROOT/fedbn_fedyogi_state/detection/$EXP_ID/client_bn"
WORK="$ARTIFACT_ROOT/fedbn_flower_workdirs/detection/$EXP_ID"
LOG="$ARTIFACT_ROOT/fedbn_fedyogi_logs/detection/${EXP_ID}.log"
PREFLIGHT="$ARTIFACT_ROOT/fedbn_fedyogi_logs/detection/${EXP_ID}_preflight.json"
# Ray's Unix-domain socket path must remain below 108 bytes. GPU-scoped
# directories are independent for the five concurrent simulations and short.
RAY_TMP="${MEDCOLAB_RAY_TMP:-$ARTIFACT_ROOT/ray_detection_${GPU_ID}}"
MANIFEST="${MANIFEST:?set MANIFEST}"
MANIFEST_SHA="${MANIFEST_SHA:?set MANIFEST_SHA}"
SSL="${SSL_BACKBONE:-$ROOT/outputs/ssl_weighted_round0_medical5_detection_clientfirst_seed42/weighted_ssl_backbone.pt}"
MAX_BATCHES=0
PREFLIGHT_SMOKE=()
if [[ "$SMOKE" == 1 ]]; then MAX_BATCHES=2; PREFLIGHT_SMOKE=(--smoke); fi
if [[ "$SMOKE" == 1 ]]; then
  RUN_ROOT="$ARTIFACT_ROOT/fedbn_smoke/detection"
  OUT="$RUN_ROOT/$EXP_ID"
  BN="$ARTIFACT_ROOT/fedbn_smoke_state/detection/$EXP_ID/client_bn"
fi
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")" "$(dirname "$BN")" "$(dirname "$WORK")" "$RAY_TMP"
python "$ROOT/tools/preflight_fedbn.py" --task detection --experiment-id "$EXP_ID" \
  --manifest "$MANIFEST" --expected-manifest-sha "$MANIFEST_SHA" --output-dir "$OUT" \
  --bn-dir "$BN" --workdir "$WORK" --rounds "$ROUNDS" --local-epochs 1 --num-clients 5 \
  --participation 1.0 --client-lr "$CLIENT_LR" --server-lr "$SERVER_LR" --seed "$SEED" \
  --expected-sources "${EXPECTED_SOURCES:-5}" "${PREFLIGHT_SMOKE[@]}" >"$PREFLIGHT"
python "$ROOT/tools/create_fedbn_workdir.py" --source-root "$ROOT" --workdir "$WORK" \
  --serverapp fl.detection_server_app --clientapp fl.detection_client_app
export PYTHONPATH="$ROOT" PYTHONHASHSEED="$SEED" MASTER_SEED="$SEED"
export FL_CURRENT_DATASET="${FL_CURRENT_DATASET:?set FL_CURRENT_DATASET}"
export FL_MODEL_VARIANT=RTDETR_L_WithASEM
export FL_PRETRAINED_MODEL_PATH="${FL_PRETRAINED_MODEL_PATH:-$ROOT/weights/rtdetr-l.pt}"
export FL_SSL_BACKBONE_CKPT="$SSL"
export CODE_VERSION="$(git -C "$ROOT" rev-parse HEAD)"
export EXPERIMENT_STARTED_AT="$(date --iso-8601=seconds)"
export TMPDIR="$RAY_TMP" RAY_TMPDIR="$RAY_TMP"
CUDA_VISIBLE_DEVICES="$GPU_ID" flwr run "$WORK" local-simulation --stream \
  --federation-config "options.num-supernodes=5 options.backend.client-resources.num-cpus=2 options.backend.client-resources.num-gpus=1.0" \
  --run-config "num-server-rounds=$ROUNDS fraction-train=1.0 local-epochs=1 lr=$CLIENT_LR use_fedbn=true server_optimizer=\"FedYogi\" num_clients=5 master_seed=$SEED server_eta=$SERVER_LR fedyogi_beta1=0.8 fedyogi_beta2=0.95 fedyogi_tau=0.001 output_dir=\"$OUT\" fedbn_state_dir=\"$BN\" train_max_batches=$MAX_BATCHES eval_max_batches=$MAX_BATCHES" \
  2>&1 | tee "$LOG"
