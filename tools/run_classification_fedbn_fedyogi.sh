#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${EXP_ID:?set EXP_ID}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-}}"
if [[ -z "$GPU_IDS" ]]; then echo "set GPU_IDS or GPU_ID" >&2; exit 1; fi
GPU_COUNT="$(awk -F, '{print NF}' <<<"$GPU_IDS")"
ROUNDS="${ROUNDS:?set ROUNDS}"
CLIENT_LR="${CLIENT_LR:-0.00005}"
BACKBONE_LR="${BACKBONE_LR:-0.00001}"
MOE_LR="${MOE_LR:-0.00005}"
SERVER_LR="${SERVER_LR:-0.002}"
EARLY_STOP_MACRO_F1_THRESHOLD="${EARLY_STOP_MACRO_F1_THRESHOLD:-0}"
EARLY_STOP_MIN_ROUNDS="${EARLY_STOP_MIN_ROUNDS:-5}"
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"
ROOT="${MEDCOLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${MEDCOLAB_ARTIFACT_ROOT:-$ROOT/outputs}"
OUT="$ARTIFACT_ROOT/fedbn_fedyogi_runs/classification/$EXP_ID"
BN="$ARTIFACT_ROOT/fedbn_fedyogi_state/classification/$EXP_ID/client_bn"
WORK="$ARTIFACT_ROOT/fedbn_flower_workdirs/classification/$EXP_ID"
LOG="$ARTIFACT_ROOT/fedbn_fedyogi_logs/classification/${EXP_ID}.log"
PREFLIGHT="$ARTIFACT_ROOT/fedbn_fedyogi_logs/classification/${EXP_ID}_preflight.json"
# Keep Ray Unix-domain socket paths short while isolating the five GPUs.
RAY_TAG="${GPU_IDS//,/}"
RAY_TMP="${MEDCOLAB_RAY_TMP:-$ARTIFACT_ROOT/ray_classification_${RAY_TAG}}"
DATA="${DATA:?set DATA}"
MANIFEST="${MANIFEST:?set MANIFEST}"
MANIFEST_SHA="${MANIFEST_SHA:?set MANIFEST_SHA}"
SOURCE_SHA="${SOURCE_SHA:?set SOURCE_SHA}"
ROUND0="${ROUND0:?set ROUND0}"
DATASET_PROFILE="${DATASET_PROFILE:-medical5}"
PCAM_GROUP_BALANCED="${PCAM_GROUP_BALANCED:-true}"
EXPECTED_SOURCES="${EXPECTED_SOURCES:-5}"
LABEL_FIELD="${LABEL_FIELD:-label}"
REQUIRED_FREE_GIB="${REQUIRED_FREE_GIB:-8}"
GROUP_AUDIT_UNAVAILABLE="${GROUP_AUDIT_UNAVAILABLE:-false}"
MAX_SAMPLES=0; MAX_BATCHES=0; PREFLIGHT_SMOKE=(); PREFLIGHT_GROUP_AUDIT=()
if [[ "$SMOKE" == 1 ]]; then MAX_SAMPLES=32; MAX_BATCHES=1; PREFLIGHT_SMOKE=(--smoke); fi
if [[ "$GROUP_AUDIT_UNAVAILABLE" == true ]]; then PREFLIGHT_GROUP_AUDIT=(--group-audit-unavailable); fi
if [[ "$SMOKE" == 1 ]]; then
  OUT="$ARTIFACT_ROOT/fedbn_smoke/classification/$EXP_ID"
  BN="$ARTIFACT_ROOT/fedbn_smoke_state/classification/$EXP_ID/client_bn"
fi
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")" "$(dirname "$BN")" "$(dirname "$WORK")" "$RAY_TMP"
python "$ROOT/tools/preflight_fedbn.py" --task classification --experiment-id "$EXP_ID" \
  --manifest "$MANIFEST" --expected-manifest-sha "$MANIFEST_SHA" --output-dir "$OUT" \
  --bn-dir "$BN" --workdir "$WORK" --rounds "$ROUNDS" --local-epochs 1 --num-clients 5 \
  --participation 1.0 --client-lr "$CLIENT_LR" --server-lr "$SERVER_LR" --seed "$SEED" \
  --expected-sources "$EXPECTED_SOURCES" --label-field "$LABEL_FIELD" --required-free-gib "$REQUIRED_FREE_GIB" \
  "${PREFLIGHT_GROUP_AUDIT[@]}" "${PREFLIGHT_SMOKE[@]}" >"$PREFLIGHT"
python "$ROOT/tools/create_fedbn_workdir.py" --source-root "$ROOT" --workdir "$WORK" \
  --serverapp fl.classification_server_app --clientapp fl.classification_client_app
export PYTHONPATH="$ROOT" PYTHONHASHSEED="$SEED" MASTER_SEED="$SEED"
export CLASSIFICATION_DATASET_PROFILE="$DATASET_PROFILE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
export CODE_VERSION="$(git -C "$ROOT" rev-parse HEAD)"
export EXPERIMENT_STARTED_AT="$(date --iso-8601=seconds)"
export TMPDIR="$RAY_TMP" RAY_TMPDIR="$RAY_TMP"
CUDA_VISIBLE_DEVICES="$GPU_IDS" flwr run "$WORK" local-simulation --stream \
  --federation-config "options.num-supernodes=5 options.backend.init-args.num-cpus=$((GPU_COUNT * 10)) options.backend.init-args.num-gpus=$GPU_COUNT options.backend.init-args.include-dashboard=false options.backend.client-resources.num-cpus=2 options.backend.client-resources.num-gpus=1.0" \
  --run-config "num-server-rounds=$ROUNDS classification_num_clients=5 classification_master_seed=$SEED classification_data_root=\"$DATA\" classification_manifest_sha256=\"$SOURCE_SHA\" classification_stratified_manifest=\"$MANIFEST\" classification_stratified_manifest_sha256=\"$MANIFEST_SHA\" classification_group_audit_unavailable=$GROUP_AUDIT_UNAVAILABLE classification_early_stop_macro_f1_threshold=$EARLY_STOP_MACRO_F1_THRESHOLD classification_early_stop_min_rounds=$EARLY_STOP_MIN_ROUNDS classification_round0_path=\"$ROUND0\" classification_output_dir=\"$OUT\" classification_local_epochs=1 classification_lr=$CLIENT_LR classification_backbone_lr=$BACKBONE_LR classification_head_lr=$CLIENT_LR classification_moe_lr=$MOE_LR classification_moe_enabled=true classification_moe_num_experts=4 classification_moe_top_k=2 classification_moe_bottleneck=256 classification_moe_gamma_init=0.001 classification_moe_balance_loss_weight=0.01 classification_weight_decay=0.0001 classification_label_smoothing=0.02 classification_class_weight_power=0.25 classification_pcam_group_balanced=$PCAM_GROUP_BALANCED classification_batch_size=16 classification_eval_batch_size=32 classification_image_size=320 classification_num_workers=2 classification_train_max_batches=$MAX_BATCHES classification_eval_max_batches=$MAX_BATCHES classification_train_max_samples=$MAX_SAMPLES classification_eval_max_samples=$MAX_SAMPLES use_fedbn=true server_optimizer=\"FedYogi\" server_eta=$SERVER_LR fedyogi_beta1=0.8 fedyogi_beta2=0.95 fedyogi_tau=0.001 fedbn_state_dir=\"$BN\"" \
  2>&1 | tee "$LOG"
