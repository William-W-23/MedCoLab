#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEDCOLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_ROOT="${MEDCOLAB_ARTIFACT_ROOT:-$ROOT/outputs}"
BASE="${NCT_WORK_ROOT:-$ARTIFACT_ROOT/nct_crc_he100k_classification}"
DOWNLOADS="$BASE/downloads"
MAIN_ARCHIVE="$DOWNLOADS/NCT-CRC-HE-100K.zip"
EXTERNAL_ARCHIVE="$DOWNLOADS/CRC-VAL-HE-7K.zip"
RAW="$BASE/raw"
DATA="$BASE/prepared_seed42"
META="$DATA/metadata"
SSL_OUT="$BASE/ssl_outputs"
ROUND0="$BASE/round0_seed42"
EXP_ID=nct_crc_he100k_fedbn_fedyogi_r200_seed42
FED="$ARTIFACT_ROOT/fedbn_fedyogi_runs/classification/$EXP_ID"
BN="$ARTIFACT_ROOT/fedbn_fedyogi_state/classification/$EXP_ID/client_bn"
FED_WORK="$ARTIFACT_ROOT/fedbn_flower_workdirs/classification/$EXP_ID"
PERSONAL="$ARTIFACT_ROOT/fedbn_personalized/classification/${EXP_ID}_guarded_soup"
STATUS="$BASE/status.json"
FAILURES="$BASE/failed_attempts"
ACTION="${1:-run}"
FEDERATION_GPU_IDS="${FEDERATION_GPU_IDS:-0,1,2,3,4}"
LOCAL_PARALLEL_CLIENTS="${LOCAL_PARALLEL_CLIENTS:-1}"
TEST_GPU_INDEX="${TEST_GPU_INDEX:-0}"

mkdir -p "$BASE/logs" "$FAILURES"
cd "$ROOT"

write_status() {
  local phase="$1" state="$2" detail="${3:-}"
  python - "$STATUS" "$phase" "$state" "$detail" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
path, phase, state, detail = sys.argv[1:]
payload = {"phase": phase, "status": state, "detail": detail,
           "updated_at": datetime.now(timezone.utc).isoformat()}
fd, temporary = tempfile.mkstemp(prefix="status.", suffix=".json", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
os.replace(temporary, path)
PY
}

archive_partial() {
  local path="$1" label="$2"
  if [[ -e "$path" ]]; then
    local destination="$FAILURES/${label}_$(date +%Y%m%dT%H%M%S)"
    mv "$path" "$destination"
    echo "Preserved partial state at $destination"
  fi
}

check_invariants() {
  [[ "$(git branch --show-current)" == "codex/nct-crc-he100k-classification" ]]
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >/dev/null
  local free_gib existing_gib
  free_gib="$(df -Pk "$ARTIFACT_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
  existing_gib="$(du -sBG "$DOWNLOADS" 2>/dev/null | awk '{gsub(/G/,"",$1); print $1}' || echo 0)"
  if [[ ! -f "$DATA/PREPARATION_COMPLETE" ]] && (( free_gib + existing_gib < 30 )); then
    echo "Need at least 30 GiB free plus resumable archive bytes before preparation" >&2
    return 4
  fi
}

prepare_data() {
  if [[ -f "$DATA/PREPARATION_COMPLETE" ]]; then
    echo "verified prepared dataset already exists"
    return
  fi
  write_status download running "official Zenodo archives; curl resume; byte-size plus MD5 verification"
  bash tools/download_nct_crc_he100k.sh "$DOWNLOADS"
  write_status preprocessing running "100K for SSL/train/val; external 7K locked for test; patch-level synthetic clients"
  python tools/prepare_nct_crc_he100k_classification.py \
    --downloads "$DOWNLOADS" --raw-root "$RAW" --output "$DATA" --seed 42 --verify-images
  python - "$META/audit.json" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("duplicate_source_paths", "cross_split_duplicate_sha256", "cross_collection_duplicate_sha256",
            "cross_client_duplicate_sha256", "broken_symlinks", "source_boundary_violations"):
    if audit[key] != 0:
        raise SystemExit(f"audit failed: {key}={audit[key]}")
if audit["missing_class_coverage_cells"]:
    raise SystemExit(f"class coverage failed: {audit['missing_class_coverage_cells']}")
if audit["patient_id_available"] or audit["source_slide_id_available"]:
    raise SystemExit("unexpected grouping claim")
PY
  python - "$DOWNLOADS/ARCHIVES_REMOVED_AFTER_EXTRACTION.json" "$MAIN_ARCHIVE" "$EXTERNAL_ARCHIVE" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
rows = {}
for value in sys.argv[2:]:
    path = os.path.abspath(value)
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    rows[os.path.basename(path)] = {"bytes": os.path.getsize(path), "md5": h.hexdigest()}
payload = {"reason": "space budget after verified extraction and audit", "removed_at": datetime.now(timezone.utc).isoformat(), "archives": rows}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
PY
  rm -f -- "$MAIN_ARCHIVE" "$EXTERNAL_ARCHIVE"
}

normalize_ssl_layout() {
  python - "$DATA" "$META/split_summary.json" <<'PY'
import json
import sys
from pathlib import Path

data = Path(sys.argv[1])
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["by_client_split_images"]
for client_index in range(5):
    client = f"client{client_index}"
    flat = data / client / "ssl_unlabeled/images/train"
    nested = flat / "nct_crc_he100k"
    if nested.is_dir():
        for source in sorted(nested.iterdir()):
            if not source.is_symlink():
                raise SystemExit(f"unexpected non-symlink in legacy SSL layout: {source}")
            destination = flat / source.name
            if destination.exists() or destination.is_symlink():
                raise SystemExit(f"refusing SSL link collision: {destination}")
            source.rename(destination)
        nested.rmdir()
    expected = int(summary[client]["ssl"])
    actual = sum(path.is_symlink() for path in flat.iterdir())
    if actual != expected:
        raise SystemExit(f"SSL layout count mismatch for {client}: expected={expected} actual={actual}")
    view = data / f"independent_ssl_views/model_client{client_index}/client0/ssl_unlabeled/images/train"
    if view.resolve(strict=True) != flat.resolve(strict=True):
        raise SystemExit(f"independent SSL view mismatch for {client}: {view}")
print("verified flat fixed-client SSL layout for all five clients")
PY
}

run_pipeline() {
  check_invariants
  prepare_data
  normalize_ssl_layout
  [[ "$(df -Pk "$ARTIFACT_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')" -ge 8 ]]
  if [[ "${STOP_AFTER_PREPARATION:-0}" == 1 ]]; then
    write_status awaiting_gpu_upgrade paused \
      "download, integrity verification, extraction, image audit, and five-client preparation complete; SSL not started"
    return 89
  fi

  local source_sha manifest manifest_sha ssl_counts ssl_mean
  source_sha="$(cut -d' ' -f1 "$META/split_manifest.sha256")"
  manifest="$META/federated_manifest.json"
  manifest_sha="$(sha256sum "$manifest" | cut -d' ' -f1)"
  ssl_counts="$(python - "$META/split_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))["by_client_split_images"]
print(",".join(str(summary[f"client{i}"]["ssl"]) for i in range(5)))
PY
)"
  ssl_mean="$(python - "$ssl_counts" <<'PY'
import sys
values = [int(value) for value in sys.argv[1].split(",")]
print(sum(values) / len(values))
PY
)"

  run_ssl_client() {
    local client_index="$1" gpu_index="$2"
    DATA_ROOT="$DATA" TASK_PREFIX=nct_crc_he100k_classification CLIENT_INDEX="$client_index" GPU_INDEX="$gpu_index" \
      OUTPUTS_ROOT="$SSL_OUT" ROUNDS=100 LOCAL_EPOCHS=1 BATCH_SIZE=8 LR=0.0001 \
      IMAGE_SIZE=320 TRAIN_MAX_BATCHES=0 MASTER_SEED=42 MANIFEST_SHA256="$source_sha" \
      MODEL_VARIANT=RTDETR_L SSL_MASK_CURRICULUM=1 SSL_MASK_ADAPTIVE=1 \
      SSL_MEAN_SAMPLES="$ssl_mean" bash tools/run_independent_ssl_client.sh
  }

  if [[ "${SSL_PARALLEL_CLIENTS:-0}" == 1 ]]; then
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
    if (( gpu_count < 5 )); then
      echo "SSL_PARALLEL_CLIENTS=1 requires five visible GPUs; found $gpu_count" >&2
      return 5
    fi
    write_status ssl running "five independent clients in parallel on GPUs 0-4; per-round checkpoint resume enabled"
    local pids=() client_indices=() client_index pid failures=0
    for client_index in 0 1 2 3 4; do
      run_ssl_client "$client_index" "$client_index" >"$BASE/logs/ssl_client${client_index}.log" 2>&1 &
      pid=$!
      pids+=("$pid")
      client_indices+=("$client_index")
      echo "Started SSL client${client_index} on GPU ${client_index} as PID ${pid}"
    done
    for client_index in 0 1 2 3 4; do
      pid="${pids[$client_index]}"
      if ! wait "$pid"; then
        echo "SSL client${client_indices[$client_index]} failed; see $BASE/logs/ssl_client${client_indices[$client_index]}.log" >&2
        failures=$((failures + 1))
      fi
    done
    if (( failures > 0 )); then
      return 6
    fi
  else
    write_status ssl running "five independent clients sequentially on GPU 0; per-round checkpoint resume enabled"
    for client_index in 0 1 2 3 4; do
      run_ssl_client "$client_index" 0
    done
  fi
  if [[ "${STOP_AFTER_SSL:-0}" == 1 ]]; then
    write_status awaiting_gpu_downsize paused \
      "all five independent SSL clients complete; Round-0 and downstream federated stages not started"
    return 90
  fi

  write_status round0 running "sample-count weighted independent SSL aggregation"
  if [[ ! -f "$ROUND0/weighted_ssl_round0_backbone.pt" ]]; then
    if [[ -e "$ROUND0" ]]; then archive_partial "$ROUND0" round0; fi
    export CLASSIFICATION_DATASET_PROFILE=nct_crc_he100k
    python tools/prepare_classification_round0.py \
      --outputs-root "$SSL_OUT" --output-dir "$ROUND0" \
      --task-prefix nct_crc_he100k_classification --manifest-sha256 "$source_sha" --ssl-counts "$ssl_counts"
  fi

  write_status federation running \
    "five clients in parallel on GPUs ${FEDERATION_GPU_IDS}; FedBN plus FedYogi; 200 rounds; fixed hyperparameters"
  if [[ ! -f "$FED/server_result.json" ]]; then
    if [[ -e "$FED" ]]; then archive_partial "$FED" federation_output; fi
    if [[ -e "$BN" ]]; then archive_partial "$(dirname "$BN")" federation_bn; fi
    if [[ -e "$FED_WORK" ]]; then archive_partial "$FED_WORK" federation_workdir; fi
    if EXP_ID="$EXP_ID" GPU_IDS="$FEDERATION_GPU_IDS" ROUNDS=200 SEED=42 \
      DATA="$DATA" MANIFEST="$manifest" MANIFEST_SHA="$manifest_sha" SOURCE_SHA="$source_sha" \
      ROUND0="$ROUND0/weighted_ssl_round0_backbone.pt" DATASET_PROFILE=nct_crc_he100k \
      PCAM_GROUP_BALANCED=false EXPECTED_SOURCES=1 LABEL_FIELD=class REQUIRED_FREE_GIB=8 \
      GROUP_AUDIT_UNAVAILABLE=true \
      bash tools/run_classification_fedbn_fedyogi.sh; then
      :
    else
      local federation_code=$?
      return "$federation_code"
    fi
    if [[ ! -f "$FED/server_result.json" ]]; then
      echo "Federation command returned without server_result.json" >&2
      return 10
    fi
  fi
  if [[ "${STOP_AFTER_FEDERATION:-0}" == 1 ]]; then
    write_status awaiting_personalization_gpu_downsize paused \
      "Round-0 and 200-round FedBN/FedYogi complete; personalized materialization, local fine-tuning, and one-time test not started"
    return 91
  fi

  write_status materialize running "validation-selected shared bundle plus client-local FedBN state"
  local init="$PERSONAL/initial_states"
  if [[ ! -f "$init/materialization_meta.json" ]]; then
    if [[ -e "$init" ]]; then archive_partial "$init" materialization; fi
    mkdir -p "$PERSONAL"
    python tools/materialize_classification_fedbn_personalized.py \
      --bundle "$FED/best_by_val_weighted_f1.pt" --out-dir "$init" --expected-clients 5
  fi

  export CLASSIFICATION_DATASET_PROFILE=nct_crc_he100k
  export MASTER_SEED=42 PYTHONHASHSEED=42
  export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
  local common=(
    --model "$init/client0_personalized_round0.pt" --client-model-root "$init"
    --data-root "$DATA" --patience 7 --min-delta 0.001
    --moe-enabled --moe-num-experts 4 --moe-top-k 2 --moe-bottleneck 256
    --moe-gamma-init 0.001 --moe-balance-loss-weight 0.01
    --class-weight-power 0.0 --dataset-equal-weight 0.25 --max-dataset-f1-drop 0.015
    --batch-size 16 --eval-batch-size 32 --image-size 320 --num-workers 2 --seed 42
    --manifest-sha256 "$source_sha" --stratified-manifest "$manifest"
    --stratified-manifest-sha256 "$manifest_sha" --skip-test --compact-sweep
  )
  run_local_client() {
    local client_index="$1" gpu_index="$2" client_out
    client_out="$PERSONAL/client_runs/client${client_index}"
    CUDA_VISIBLE_DEVICES="$gpu_index" python tools/local_finetune_classification.py \
      "${common[@]}" --out-dir "$client_out" --clients "$client_index" \
      --max-epochs 25 --freeze-backbone-epochs 5 \
      --backbone-lr 0.0000002 --head-lr 0.000003 --moe-lr 0.000003 \
      --l2sp-mu 0.01 --soup-alphas 0,0.25,0.5
  }

  local checkpoints=() pending_clients=() client_index
  for client_index in 0 1 2 3 4; do
    local client_out="$PERSONAL/client_runs/client${client_index}"
    local checkpoint="$client_out/client${client_index}/best_by_val_weighted_f1.pt"
    if [[ ! -f "$checkpoint" || ! -f "$client_out/local_finetune_summary.json" ]]; then
      if [[ -e "$client_out" ]]; then archive_partial "$client_out" "local_client${client_index}"; fi
      pending_clients+=("$client_index")
    fi
    checkpoints+=("$checkpoint")
  done

  if (( ${#pending_clients[@]} > 0 )); then
    if [[ "$LOCAL_PARALLEL_CLIENTS" == 1 ]]; then
      local gpu_count
      gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
      if (( gpu_count < 5 )); then
        echo "LOCAL_PARALLEL_CLIENTS=1 requires five visible GPUs; found $gpu_count" >&2
        return 7
      fi
      write_status local_finetune running \
        "validation-only personalized fine-tuning in parallel on GPUs 0-4; external test locked"
      local pids=() pending_index pid failures=0
      for client_index in "${pending_clients[@]}"; do
        {
          echo "[$(date -Iseconds)] starting client${client_index} on GPU ${client_index}"
          run_local_client "$client_index" "$client_index"
        } >>"$BASE/logs/local_client${client_index}.log" 2>&1 &
        pid=$!
        pids+=("$pid")
        echo "Started personalized client${client_index} on GPU ${client_index} as PID ${pid}"
      done
      for pending_index in "${!pending_clients[@]}"; do
        client_index="${pending_clients[$pending_index]}"
        pid="${pids[$pending_index]}"
        if ! wait "$pid"; then
          echo "Personalized client${client_index} failed; see $BASE/logs/local_client${client_index}.log" >&2
          failures=$((failures + 1))
        fi
      done
      if (( failures > 0 )); then
        return 8
      fi
    else
      write_status local_finetune running \
        "validation-only personalized fine-tuning sequentially on GPU 0; external test locked"
      for client_index in "${pending_clients[@]}"; do
        run_local_client "$client_index" 0 >>"$BASE/logs/local_client${client_index}.log" 2>&1
      done
    fi
  fi

  for client_index in 0 1 2 3 4; do
    if [[ ! -f "${checkpoints[$client_index]}" || \
          ! -f "$PERSONAL/client_runs/client${client_index}/local_finetune_summary.json" ]]; then
      echo "Missing completed personalized output for client${client_index}" >&2
      return 9
    fi
  done

  local test_out="$PERSONAL/one_time_test"
  if [[ ! -f "$test_out/one_time_test_evaluation_summary.json" ]]; then
    if [[ -e "$test_out" ]]; then
      write_status one_time_test blocked "partial test output exists; automatic re-evaluation prohibited"
      return 88
    fi
    write_status one_time_test running "all validation selections frozen; external test evaluated exactly once"
    CUDA_VISIBLE_DEVICES="$TEST_GPU_INDEX" python tools/evaluate_classification.py \
      --checkpoints "${checkpoints[@]}" --data-root "$DATA" --manifest "$manifest" \
      --manifest-sha256 "$manifest_sha" --source-manifest-sha256 "$source_sha" \
      --output-dir "$test_out" --batch-size 32 --image-size 320 --num-workers 2 --seed 42 \
      --moe-enabled --moe-num-experts 4 --moe-top-k 2 --moe-bottleneck 256 --moe-gamma-init 0.001 \
      --selection-description "Per-client validation-only guarded soup selection; CRC-VAL-HE-7K evaluated once after all five selections"
  fi
  write_status completed completed "download, preparation, SSL, federation, local fine-tuning, and external one-time test complete"
}

supervise() {
  local failures=0 code
  while true; do
    if bash "$0" run; then
      echo "Pipeline completed successfully."
      return 0
    else
      code=$?
    fi
    if [[ "$code" == 88 ]]; then
      echo "Stopped to preserve the one-time test invariant."
      return 88
    fi
    if [[ "$code" == 89 ]]; then
      echo "Stopped after verified preparation; waiting for the GPU upgrade before SSL."
      return 0
    fi
    if [[ "$code" == 90 ]]; then
      echo "Stopped after all independent SSL clients; waiting for the GPU downsize before federation."
      return 0
    fi
    if [[ "$code" == 91 ]]; then
      echo "Stopped after federation; waiting for the GPU downsize before personalized fine-tuning."
      return 0
    fi
    failures=$((failures + 1))
    write_status supervisor retrying "attempt=$failures exit_code=$code; fixed parameters retained"
    if [[ "$failures" -ge 5 ]]; then
      write_status supervisor blocked "five consecutive failures; manual diagnosis required"
      return "$code"
    fi
    echo "Attempt $failures failed with code $code; retrying unchanged workflow in 60 seconds."
    sleep 60
  done
}

case "$ACTION" in
  run) run_pipeline ;;
  supervise) supervise ;;
  status) python -m json.tool "$STATUS" ;;
  *) echo "Usage: $0 [run|supervise|status]" >&2; exit 2 ;;
esac
