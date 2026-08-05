#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEDCOLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${MEDCOLAB_ARTIFACT_ROOT:-$ROOT/outputs}"
BASE="${TBX11K_WORK_ROOT:-$ARTIFACT_ROOT/tbx11k}"
ARCHIVE="$BASE/raw/TBX11K.zip"
ARCHIVE_HASH="$BASE/raw/TBX11K.zip.sha256"
SOURCE="$BASE/source"
STAGING_SOURCE="$BASE/.source.staging"
DATA="$BASE/prepared/tbx11k_clientfirst_seed42"
SSL_OUTPUT_ROOT="$BASE/ssl_outputs"
STATE="$BASE/pipeline_state.json"
LOG="$BASE/logs/server_pipeline.log"
EXP_ID_FILE="$BASE/federated_exp_id.txt"
EXPECTED_ARCHIVE_BYTES=3306757175

mkdir -p "$BASE/raw" "$BASE/logs"
status(){ python - "$STATE" "$1" "${2:-}" <<'PY'
import datetime,json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':sys.argv[2],'detail':sys.argv[3],'updated_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
PY
}
on_error(){
  local rc="$?" line="$1" current="unknown" retry="${TBX_PREP_RETRY:-0}"
  trap - ERR
  current="$(python - "$STATE" <<'PY' 2>/dev/null || true
import json,sys
try: print(json.load(open(sys.argv[1])).get('status','unknown'))
except Exception: print('unknown')
PY
)"
  case "$current" in
    validating_archive|extracting|preparing_patient_splits|validating_prepared_data|aam_smoke_test)
      if (( retry < 5 )); then
        status retrying_preparation "from=$current line=$line exit=$rc attempt=$((retry+1))/5"
        sleep $((30 * (retry + 1)))
        exec env TBX_PREP_RETRY="$((retry + 1))" bash "$0"
      fi
      ;;
  esac
  status failed "stage=$current line=$line exit=$rc"
  exit "$rc"
}
trap 'on_error "$LINENO"' ERR
exec >>"$LOG" 2>&1
echo "pipeline_start $(date -Iseconds) commit=$(git -C "$ROOT" rev-parse HEAD)"

while [[ ! -s "$ARCHIVE" || ! -s "$ARCHIVE_HASH" ]]; do
  partial_bytes=0
  [[ -e "$ARCHIVE.part" ]] && partial_bytes="$(stat -c %s "$ARCHIVE.part")"
  status awaiting_verified_archive "expected_bytes=$EXPECTED_ARCHIVE_BYTES partial_bytes=$partial_bytes"
  echo "awaiting_verified_archive $(date -Iseconds) partial_bytes=$partial_bytes"
  sleep 60
done

status validating_archive
[[ "$(stat -c %s "$ARCHIVE")" == "$EXPECTED_ARCHIVE_BYTES" ]]
(cd "$BASE/raw" && sha256sum -c "$(basename "$ARCHIVE_HASH")")
unzip -t "$ARCHIVE" >/dev/null
python - "$ARCHIVE" <<'PY'
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    names=set(archive.namelist())
    required={
        'TBX11K/README.md',
        'TBX11K/annotations/json/all_trainval.json',
        'TBX11K/lists/all_trainval.txt',
    }
    missing=required-names
    if missing: raise RuntimeError(f'missing archive members: {sorted(missing)}')
    images=sum(name.lower().endswith(('.png','.jpg','.jpeg')) and name.startswith('TBX11K/imgs/') for name in names)
    if images != 12278: raise RuntimeError(f'unexpected archive image inventory: {images}')
    print(f'archive_inventory_ok images={images}')
PY

source_ready(){
  [[ -f "$SOURCE/TBX11K/annotations/json/all_trainval.json" ]] || return 1
  [[ "$(find "$SOURCE/TBX11K/imgs" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)" == 12278 ]]
}
if ! source_ready; then
  status extracting
  rm -rf "$STAGING_SOURCE"
  mkdir -p "$STAGING_SOURCE"
  unzip -oq "$ARCHIVE" -d "$STAGING_SOURCE"
  [[ "$(find "$STAGING_SOURCE/TBX11K/imgs" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l)" == 12278 ]]
  if [[ -e "$SOURCE" ]]; then
    mv "$SOURCE" "$BASE/source.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  mv "$STAGING_SOURCE" "$SOURCE"
fi
source_ready || { echo "source inventory incomplete after extraction" >&2; exit 2; }

python - "$BASE" <<'PY'
import os,shutil,sys
free=shutil.disk_usage(sys.argv[1]).free
root_free=shutil.disk_usage(os.environ.get('MEDCOLAB_ROOT', '.')).free
if free < 30*1024**3: raise RuntimeError(f'data disk free space below 30 GiB: {free/1024**3:.2f}')
if root_free < 4*1024**3: raise RuntimeError(f'system disk free space below 4 GiB: {root_free/1024**3:.2f}')
print(f'free_space_ok data={free/1024**3:.2f}GiB system={root_free/1024**3:.2f}GiB')
PY

status preparing_patient_splits
python "$ROOT/tools/prepare_tbx11k_clientfirst.py" --source "$SOURCE/TBX11K" --output "$DATA" --seed 42
MANIFEST="$DATA/manifest.csv"
MANIFEST_SHA="$(tr -d '[:space:]' < "$DATA/manifest.sha256")"

status validating_prepared_data
if [[ ! -s "$BASE/prepared_preflight.json" ]]; then
  python "$ROOT/tools/preflight_fedbn.py" --task detection --experiment-id tbx11k_manifest_only \
    --manifest "$MANIFEST" --expected-manifest-sha "$MANIFEST_SHA" \
    --output-dir "$BASE/preflight_sentinel/output" --bn-dir "$BASE/preflight_sentinel/bn" \
    --workdir "$BASE/preflight_sentinel/work" --rounds 1 --local-epochs 1 --num-clients 5 \
    --participation 1.0 --client-lr 0.00005 --server-lr 0.002 --seed 43 \
    --expected-sources 5 --smoke > "$BASE/prepared_preflight.json"
fi
python - "$BASE/prepared_preflight.json" "$MANIFEST_SHA" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
assert payload['expected_manifest_sha'] == sys.argv[2]
assert payload['use_fedbn'] is True and payload['server_optimizer'] == 'fedyogi'
assert len(payload['data_audit']['clients']) == 5
print('prepared_preflight_reuse_ok')
PY

status aam_smoke_test
python - <<'PY'
from fl.ssl_task import SSL_DEFAULTS,build_mask_config
c=dict(SSL_DEFAULTS); c['ssl_mask_adaptive']=True
x=build_mask_config(c,round_num=1,total_rounds=100,num_samples=120,mean_samples=100)
assert x['enable'] and abs(x['ratio']-.36)<1e-9, x
print(x)
PY
COUNTS=($(python - "$DATA/manifest.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]))); print(*[sum(r['client']==str(i) and r['split']=='ssl' for r in rows) for i in range(5)])
PY
))
MEAN=$(python - "${COUNTS[@]}" <<'PY'
import sys
x=list(map(int,sys.argv[1:])); print(sum(x)/len(x))
PY
)

status ssl_training
for client in 0 1 2 3 4; do
  DATA_ROOT="$DATA" OUTPUT_ROOT="$SSL_OUTPUT_ROOT" TASK_PREFIX=tbx11k_clientfirst \
  MANIFEST_SHA256="$MANIFEST_SHA" CLIENT_INDEX="$client" GPU_INDEX=0 MASTER_SEED=42 \
  ROUNDS=100 LOCAL_EPOCHS=1 BATCH_SIZE=8 LR=0.0001 IMAGE_SIZE=320 TRAIN_MAX_BATCHES=0 \
  SSL_MASK_ADAPTIVE=true SSL_MEAN_SAMPLES="$MEAN" \
  bash "$ROOT/tools/run_independent_ssl_client.sh"
done

status aggregating_round0
INPUTS=()
for client in 0 1 2 3 4; do
  INPUTS+=("$SSL_OUTPUT_ROOT/ssl_moco_RTDETR_L_Independent_tbx11k_clientfirst_client${client}_seed42/ssl_best_by_loss_backbone.pt")
done
ROUND0="$BASE/round0/ssl_weighted_round0_tbx11k_clientfirst_seed42"
if [[ ! -s "$ROUND0/weighted_ssl_backbone.pt" ]]; then
  python "$ROOT/tools/aggregate_ssl_backbones.py" --inputs "${INPUTS[@]}" --counts "${COUNTS[@]}" --output "$ROUND0"
fi

if [[ -s "$EXP_ID_FILE" ]]; then
  EXP_ID="$(tr -d '[:space:]' < "$EXP_ID_FILE")"
else
  EXP_ID="tbx11k_fedbn_fedyogi_r200_lr5e-5_s43_$(date +%Y%m%d_%H%M%S)"
  printf '%s\n' "$EXP_ID" > "$EXP_ID_FILE"
fi
RESULT="$ARTIFACT_ROOT/fedbn_fedyogi_runs/detection/$EXP_ID/server_result.json"
if [[ ! -s "$RESULT" ]]; then
  FED_DIR="$ARTIFACT_ROOT/fedbn_fedyogi_runs/detection/$EXP_ID"
  if [[ -d "$FED_DIR" && -n "$(find "$FED_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    status failed_manual_federated_resume "partial_federated_dir=$FED_DIR; Flower run is not checkpoint-resumable"
    exit 3
  fi
  status federated_training "$EXP_ID"
  EXP_ID="$EXP_ID" GPU_ID=0 ROUNDS=200 CLIENT_LR=0.00005 SERVER_LR=0.002 SEED=43 \
  MANIFEST="$MANIFEST" MANIFEST_SHA="$MANIFEST_SHA" FL_CURRENT_DATASET=tbx11k_clientfirst_seed42 \
  SSL_BACKBONE="$ROUND0/weighted_ssl_backbone.pt" EXPECTED_SOURCES=5 \
  bash "$ROOT/tools/run_detection_fedbn_fedyogi_rebalanced.sh"
fi
test -s "$RESULT"
cp "$RESULT" "$BASE/federated_result.json"

status local_finetuning "$EXP_ID"
bash "$ROOT/tools/run_tbx11k_personalization.sh" "$EXP_ID"
LOCAL_RESULT="$BASE/personalization/$EXP_ID/protocol_audit/protocol_audit_summary.json"
test -s "$LOCAL_RESULT"
cp "$LOCAL_RESULT" "$BASE/local_finetune_result.json"
status complete "$LOCAL_RESULT"
echo "pipeline_complete $(date -Iseconds) federated_result=$RESULT local_result=$LOCAL_RESULT"
