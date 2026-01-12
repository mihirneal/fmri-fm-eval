#!/bin/bash
# Generate ADNI evaluation datasets for all supported spaces

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Default number of processes
NUM_PROC=${NUM_PROC:-32}

# Spaces to generate
SPACES=(
    "schaefer400"
    "schaefer400_tians3"
    "flat"
    "a424"
    "mni"
    "mni_cortex"
)

echo "Generating ADNI evaluation datasets..."
echo "Number of processes: $NUM_PROC"
echo ""

for space in "${SPACES[@]}"; do
    echo "=========================================="
    echo "Generating: $space"
    echo "=========================================="
    uv run python scripts/make_adni_eval.py --space "$space" --num_proc "$NUM_PROC"
    echo ""
done

echo "Done! All datasets generated."
