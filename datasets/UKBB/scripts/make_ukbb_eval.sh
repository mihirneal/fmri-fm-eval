#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UKBB_DIR="$(dirname "${SCRIPT_DIR}")"

spaces=(
    flat
    schaefer400
    schaefer400_tians3
    a424
    # mni
    # mni_cortex
    # schaefer400_tians3_buckner7 not possible with CIFTI only files
)

OUT_ROOT="${UKBB_DIR}/data/processed"

log_path="${UKBB_DIR}/logs/make_ukbb_eval.log"

mkdir -p "$(dirname "${log_path}")"

for space in "${spaces[@]}"; do
    uv run python "${SCRIPT_DIR}/make_ukbb_arrow.py" \
        --space "${space}" \
        --out-root "${OUT_ROOT}" \
        --num_proc 8 \
        2>&1 | tee -a "${log_path}"
done
