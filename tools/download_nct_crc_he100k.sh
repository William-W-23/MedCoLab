#!/usr/bin/env bash
set -euo pipefail

ROOT="${MEDCOLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEST="${1:-$ROOT/downloads/nct_crc_he100k}"
BASE_URL="https://zenodo.org/api/records/1214456/files"
mkdir -p "$DEST"

download_one() {
  local name="$1" expected_bytes="$2" expected_md5="$3"
  local target="$DEST/$name"
  if [[ -f "$target" ]] && [[ "$(stat -c %s "$target")" == "$expected_bytes" ]] && \
     [[ "$(md5sum "$target" | cut -d' ' -f1)" == "$expected_md5" ]]; then
    echo "verified existing $name"
    return
  fi
  if [[ -f "$target" ]]; then
    mv "$target" "$target.unverified.$(date +%Y%m%dT%H%M%S)"
  fi
  local segments="${DOWNLOAD_SEGMENTS:-8}" segment_dir="$target.segments"
  local active_segments="${DOWNLOAD_ACTIVE_SEGMENTS:-$segments}"
  if (( active_segments < 1 || active_segments > segments )); then
    echo "DOWNLOAD_ACTIVE_SEGMENTS must be between 1 and $segments" >&2
    return 2
  fi
  mkdir -p "$segment_dir"
  local failed=0 running=0 segment start end
  for ((segment=0; segment<segments; segment++)); do
    start=$((expected_bytes * segment / segments))
    end=$((expected_bytes * (segment + 1) / segments - 1))
    download_segment_group "$BASE_URL/$name/content" "$segment_dir/part$(printf '%02d' "$segment")" "$start" "$end" &
    running=$((running + 1))
    if (( running >= active_segments )); then
      if ! wait -n; then failed=1; fi
      running=$((running - 1))
    fi
  done
  while (( running > 0 )); do
    if ! wait -n; then failed=1; fi
    running=$((running - 1))
  done
  if [[ "$failed" != 0 ]]; then
    echo "One or more resumable ranges failed for $name" >&2
    return 4
  fi
  local assembling="$target.assembling"
  : >"$assembling"
  for ((segment=0; segment<segments; segment++)); do
    cat "$segment_dir/part$(printf '%02d' "$segment")" >>"$assembling"
  done
  mv "$assembling" "$target"
  rm -rf -- "$segment_dir"
  local actual_bytes actual_md5
  actual_bytes="$(stat -c %s "$target")"
  actual_md5="$(md5sum "$target" | cut -d' ' -f1)"
  if [[ "$actual_bytes" != "$expected_bytes" || "$actual_md5" != "$expected_md5" ]]; then
    mv "$target" "$target.bad.$(date +%Y%m%dT%H%M%S)"
    echo "Integrity failure for $name: bytes=$actual_bytes md5=$actual_md5" >&2
    return 3
  fi
}

download_segment_group() {
  local url="$1" part="$2" start="$3" end="$4"
  local expected=$((end - start + 1)) size received next
  local subsegments="${DOWNLOAD_SUBSEGMENTS:-1}"
  next="$part.next"

  # Recover bytes written by the previous single-range downloader before
  # defining stable subranges for the remaining interval.
  if [[ -f "$next" ]]; then
    size=0
    [[ -f "$part" ]] && size="$(stat -c %s "$part")"
    received="$(stat -c %s "$next")"
    if (( size + received > expected )); then
      mv "$next" "$next.oversize.$(date +%Y%m%dT%H%M%S)"
      return 6
    fi
    if (( received > 0 )); then cat "$next" >>"$part"; fi
    rm -f -- "$next"
  fi

  size=0
  [[ -f "$part" ]] && size="$(stat -c %s "$part")"
  if (( size == expected )); then return 0; fi
  if (( size > expected )); then
    mv "$part" "$part.oversize.$(date +%Y%m%dT%H%M%S)"
    return 5
  fi
  if (( subsegments <= 1 )); then
    download_segment "$url" "$part" "$start" "$end"
    return
  fi

  local subdir="$part.subsegments" manifest manifest_start manifest_end manifest_count
  local remaining_start remaining_bytes
  remaining_start=$((start + size))
  remaining_bytes=$((end - remaining_start + 1))
  touch "$part"
  mkdir -p "$subdir"
  manifest="$subdir/manifest"
  if [[ -f "$manifest" ]]; then
    read -r manifest_start manifest_end manifest_count <"$manifest"
    if [[ "$manifest_start" != "$remaining_start" || "$manifest_end" != "$end" || \
          "$manifest_count" != "$subsegments" ]]; then
      echo "Subrange manifest does not match current prefix for $part" >&2
      return 7
    fi
  else
    printf '%s %s %s\n' "$remaining_start" "$end" "$subsegments" >"$manifest.tmp"
    mv "$manifest.tmp" "$manifest"
  fi

  local pids=() failed=0 sub sub_start sub_end subpart
  for ((sub=0; sub<subsegments; sub++)); do
    sub_start=$((remaining_start + remaining_bytes * sub / subsegments))
    sub_end=$((remaining_start + remaining_bytes * (sub + 1) / subsegments - 1))
    subpart="$subdir/part$(printf '%02d' "$sub")"
    download_segment "$url" "$subpart" "$sub_start" "$sub_end" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" != 0 ]]; then return 8; fi

  local expected_before expected_after current_size child_size
  for ((sub=0; sub<subsegments; sub++)); do
    sub_start=$((remaining_start + remaining_bytes * sub / subsegments))
    sub_end=$((remaining_start + remaining_bytes * (sub + 1) / subsegments - 1))
    subpart="$subdir/part$(printf '%02d' "$sub")"
    child_size=$((sub_end - sub_start + 1))
    expected_before=$((sub_start - start))
    expected_after=$((expected_before + child_size))
    current_size="$(stat -c %s "$part")"
    if (( current_size > expected_before && current_size < expected_after )); then
      truncate -s "$expected_before" "$part"
      current_size="$expected_before"
    fi
    if (( current_size == expected_before )); then
      cat "$subpart" >>"$part"
      current_size="$(stat -c %s "$part")"
    fi
    if (( current_size != expected_after )); then
      echo "Cannot safely append completed subrange $sub for $part" >&2
      return 9
    fi
    rm -f -- "$subpart" "$subpart.next"
  done
  rm -rf -- "$subdir"
}

download_segment() {
  local url="$1" part="$2" start="$3" end="$4"
  local expected=$((end - start + 1)) size offset next received
  local proxy_url="${DOWNLOAD_PROXY_URL:-}" retry_count retry_delay transport
  local proxy_args=()
  while true; do
    next="$part.next"
    if [[ -f "$next" ]]; then
      size=0
      [[ -f "$part" ]] && size="$(stat -c %s "$part")"
      received="$(stat -c %s "$next")"
      if (( size + received > expected )); then
        mv "$next" "$next.oversize.$(date +%Y%m%dT%H%M%S)"
        return 6
      fi
      if (( received > 0 )); then cat "$next" >>"$part"; fi
      rm -f -- "$next"
    fi
    size=0
    [[ -f "$part" ]] && size="$(stat -c %s "$part")"
    if (( size == expected )); then return 0; fi
    if (( size > expected )); then
      mv "$part" "$part.oversize.$(date +%Y%m%dT%H%M%S)"
      return 5
    fi
    offset=$((start + size))
    rm -f -- "$next"
    proxy_args=()
    retry_count=10
    retry_delay=15
    transport=direct
    if [[ -n "$proxy_url" ]] && curl --proxy "$proxy_url" --fail --head \
      --silent --max-time 5 "$url" >/dev/null 2>&1; then
      proxy_args=(--proxy "$proxy_url")
      retry_count=2
      retry_delay=3
      transport=proxy
    elif [[ -n "$proxy_url" ]]; then
      echo "Proxy unavailable; using direct fallback for range $offset-$end" >&2
    fi
    if ! curl "${proxy_args[@]}" --fail --location --retry "$retry_count" \
      --retry-delay "$retry_delay" --retry-all-errors \
      --connect-timeout 30 --speed-time 180 --speed-limit 1024 \
      --range "$offset-$end" --output "$next" "$url"; then
      echo "$transport range $offset-$end interrupted; preserving received bytes" >&2
    fi
    received=0
    [[ -f "$next" ]] && received="$(stat -c %s "$next")"
    if (( received > end - offset + 1 )); then
      mv "$next" "$next.oversize.$(date +%Y%m%dT%H%M%S)"
      return 6
    fi
    if (( received > 0 )); then
      cat "$next" >>"$part"
      rm -f -- "$next"
    else
      sleep 30
    fi
  done
}

download_one NCT-CRC-HE-100K.zip 11690284003 6fd702d11df6292bc054397ae038a464
download_one CRC-VAL-HE-7K.zip 800276929 2fd1651b4f94ebd818ebf90ad2b6ce06

python - "$DEST/provenance.json.tmp" <<'PY'
import json, sys
payload = {
    "dataset": "NCT-CRC-HE-100K with CRC-VAL-HE-7K external test",
    "paper": "Predicting survival from colorectal cancer histology slides using deep learning: a retrospective multicenter study",
    "paper_doi": "10.1371/journal.pmed.1002730",
    "repository": "Zenodo record 1214456",
    "repository_doi": "10.5281/zenodo.1214456",
    "license": "CC BY 4.0",
    "archives": {
        "NCT-CRC-HE-100K.zip": {"bytes": 11690284003, "md5": "6fd702d11df6292bc054397ae038a464"},  # pragma: allowlist secret
        "CRC-VAL-HE-7K.zip": {"bytes": 800276929, "md5": "2fd1651b4f94ebd818ebf90ad2b6ce06"},  # pragma: allowlist secret
    },
    "protocol_note": "The 100K archive is restricted to SSL/train/validation; CRC-VAL-HE-7K is locked for one-time testing.",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
mv "$DEST/provenance.json.tmp" "$DEST/provenance.json"
touch "$DEST/DOWNLOAD_COMPLETE"
echo "Both official Zenodo archives passed byte-size and MD5 verification."
