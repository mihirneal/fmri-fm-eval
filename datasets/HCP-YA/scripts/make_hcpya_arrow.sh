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

# nb, volume data not currently stored locally
# but remote is fine since the script is not blocked waiting for download
roots=(
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    data/sourcedata/HCP_1200
    s3://hcp-openaccess/HCP_1200
    s3://hcp-openaccess/HCP_1200
    s3://hcp-openaccess/HCP_1200
)

OUT_ROOT="s3://medarc/fmri-datasets/eval"

# SPACEIDS="0 1 2 3 4 5 6"
SPACEIDS="3 4 5 6"

log_path="logs/make_hcpya_arrow.log"

datasets="rest1lr task21 clips"

for ii in $SPACEIDS; do
    space=${spaces[ii]}
    root=${roots[ii]}
    for dataset in $datasets; do
        uv run python scripts/make_hcpya_${dataset}_arrow.py \
            --space "${space}" \
            --root "${root}" \
            --out-root "${OUT_ROOT}" \
            2>&1 | tee -a "${log_path}"
    done
done
