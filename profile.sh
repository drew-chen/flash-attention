#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$repo_root/.venv/bin/python"
ncu_bin="${FLASH_ATTN_NCU_BIN:-/usr/local/cuda/bin/ncu}"

usage() {
  echo "usage: $0 <implementation|all>" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

if [[ ! -x "$python_bin" ]]; then
  echo "error: Python environment not found at $python_bin" >&2
  exit 1
fi

if [[ ! -x "$ncu_bin" ]]; then
  echo "error: Nsight Compute not found at $ncu_bin" >&2
  exit 1
fi

cd "$repo_root"

mapfile -t implementations < <(
  "$python_bin" -c 'from benchmark import IMPLEMENTATIONS; print(*IMPLEMENTATIONS, sep="\n")'
)
mapfile -t fused_implementations < <(
  "$python_bin" -c 'from profile_cuda import FUSED_IMPLEMENTATIONS; print(*FUSED_IMPLEMENTATIONS, sep="\n")'
)

contains() {
  local requested="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ "$candidate" == "$requested" ]]; then
      return 0
    fi
  done
  return 1
}

profile_implementation() {
  local implementation="$1"
  local report="/tmp/flash_${implementation}_s2048_profile"

  echo "Profiling $implementation..."
  "$ncu_bin" \
    --set roofline \
    --section Occupancy \
    --section LaunchStats \
    --replay-mode kernel \
    --nvtx \
    --nvtx-include "flash_attention.${implementation}/" \
    --export "$report" \
    --force-overwrite \
    "$python_bin" profile_cuda.py "$implementation"
  echo "Report: ${report}.ncu-rep"
}

requested="$1"
if [[ "$requested" == "all" ]]; then
  for implementation in "${implementations[@]}"; do
    if contains "$implementation" "${fused_implementations[@]}"; then
      profile_implementation "$implementation"
    else
      echo "Skipping $implementation: profiling is restricted to project fused implementations."
    fi
  done
  exit 0
fi

if ! contains "$requested" "${implementations[@]}"; then
  echo "error: unknown implementation '$requested'" >&2
  usage
  exit 2
fi

if ! contains "$requested" "${fused_implementations[@]}"; then
  echo "error: $requested is not a project fused implementation" >&2
  exit 2
fi

profile_implementation "$requested"
