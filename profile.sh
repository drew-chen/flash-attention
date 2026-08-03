#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$repo_root/.venv/bin/python"
ncu_bin="${FLASH_ATTN_NCU_BIN:-/usr/local/cuda/bin/ncu}"
fused_implementations=(v1 v2 v3 v4 v4-fp16)

usage() {
  cat <<EOF
Usage: $0 <implementation|all>

Profile a fused FlashAttention kernel with NVIDIA Nsight Compute.

Arguments:
  v1, v2, v3, v4, v4-fp16,  Profile one implementation.
  all             Profile every fused implementation.

Examples:
  $0 v4
  $0 all

Reports are written to /tmp/flash_<implementation>_s2048_profile.ncu-rep.
Set FLASH_ATTN_NCU_BIN to use an ncu executable outside /usr/local/cuda/bin.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

cd "$repo_root"

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

case "$1" in
  all)
    for implementation in "${fused_implementations[@]}"; do
      profile_implementation "$implementation"
    done
    ;;
  v1|v2|v3|v4|v4-fp16)
    profile_implementation "$1"
    ;;
  *)
    echo "error: unknown implementation '$1'" >&2
    usage >&2
    exit 2
    ;;
esac
