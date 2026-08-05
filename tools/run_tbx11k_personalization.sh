#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEDCOLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${MEDCOLAB_ARTIFACT_ROOT:-$ROOT/outputs}"
BASE="${TBX11K_WORK_ROOT:-$ARTIFACT_ROOT/tbx11k}"
EXP_ID="${1:?usage: run_tbx11k_personalization.sh EXP_ID}"
FED_ROOT="$ARTIFACT_ROOT/fedbn_fedyogi_runs/detection/$EXP_ID"
BUNDLE="$FED_ROOT/best_by_val_map50.pt"
PERSONAL_ROOT="$BASE/personalization/$EXP_ID"
INIT="$PERSONAL_ROOT/fedbn_personalized_init"
RUN="$PERSONAL_ROOT/cfg_b_classaware3"
AUDIT="$PERSONAL_ROOT/protocol_audit"
DATASET=tbx11k_clientfirst_seed42

export PYTHONPATH="$ROOT"
export PYTHONHASHSEED=42
export MASTER_SEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FL_CURRENT_DATASET="$DATASET"

test -s "$BUNDLE"
python - "$BASE" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
if free < 15 * 1024**3:
    raise RuntimeError(f'data disk free space below 15 GiB before personalization: {free/1024**3:.2f}')
print(f'personalization_free_space_ok={free/1024**3:.2f}GiB')
PY

mkdir -p "$PERSONAL_ROOT"
if [[ ! -s "$INIT/materialization_meta.json" ]]; then
  if [[ -e "$INIT" ]]; then echo "incomplete initialization exists: $INIT" >&2; exit 1; fi
  python "$ROOT/tools/materialize_detection_fedbn_personalized.py" --bundle "$BUNDLE" --out-dir "$INIT" --expected-clients 5
fi

for client in 0 1 2 3 4; do
  client_root="$RUN/client$client"
  metrics="$(find "$client_root" -mindepth 2 -maxdepth 2 -name validation_metrics.json -print -quit 2>/dev/null || true)"
  checkpoint="$(find "$client_root" -mindepth 2 -maxdepth 2 -name best_model_by_map50.pt -print -quit 2>/dev/null || true)"
  if [[ -s "$metrics" && -s "$checkpoint" ]]; then echo "client${client} already complete; skipping"; continue; fi
  mkdir -p "$client_root"
  python "$ROOT/tools/local_finetune_detection_clients.py" \
    --model "$INIT/client0_personalized_round0.pt" \
    --model-template "$INIT/client{client_id}_personalized_round0.pt" \
    --out-dir "$client_root" --model-variant RTDETR_L_WithASEM \
    --dataset "$DATASET" --clients "$client" --epochs 40 --lr 5e-5 \
    --conservative-personalization --fedbn-aware-progressive --validation-only \
    --save-validation-checkpoint --late-backbone-lr 1e-7 --neck-lr 5e-7 \
    --decoder-lr 2e-6 --head-lr 8e-6 --head-only-decoder-lr 3e-6 \
    --head-only-head-lr 1e-5 --moe-lr 1e-7 --head-only-epochs 5 \
    --l2sp-lambda 0.05 --freeze-fedbn-bn --train-augmentation \
    --class-aware-sampling --class-aware-max-weight 3 --rare-class-protection \
    --rare-class-freeze-threshold 10 --rare-class-full-threshold 30 \
    --warmup-epochs 2 --early-stop-patience 8 --early-stop-min-delta 0.002 \
    --top-k-checkpoints 5 --soup-alphas 0,0.25,0.5,0.75,1 \
    --weight-decay 1e-4 --grad-clip 0.1 --batch-size 16 --eval-batch-size 8 \
    --seed 42 --disable-moe-domain-supervision --strict-load
done

mkdir -p "$AUDIT"
for client in 0 1 2 3 4; do
  if [[ ! -s "$AUDIT/client$client/audit.json" ]]; then
    python "$ROOT/tools/evaluate_detection.py" \
      --run-dir "$RUN" --out-dir "$AUDIT" --client-id "$client" \
      --dataset-key "$DATASET" --batch-size 8 --device cuda
  fi
done
python "$ROOT/tools/evaluate_detection.py" \
  --out-dir "$AUDIT" --aggregate --dataset-key "$DATASET" --batch-size 8 --device cuda
date --iso-8601=seconds > "$PERSONAL_ROOT/AUTO_COMPLETE"
echo "personalization_complete audit=$AUDIT/protocol_audit_summary.json"
