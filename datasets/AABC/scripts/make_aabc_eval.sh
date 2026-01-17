#!/bin/bash

# all target spaces required by different models
spaces=(
    schaefer400
    schaefer400_tians3
    flat
    a424
    mni
    mni_cortex
    schaefer400_tians3_buckner7
)

log_path="logs/make_aabc_eval.log"

mkdir -p "$(dirname "${log_path}")"

for space in "${spaces[@]}"; do
    uv run python datasets/AABC/scripts/make_aabc_eval.py \
        --space "${space}" \
        --num_proc 8 \
        2>&1 | tee -a "${log_path}"
done
