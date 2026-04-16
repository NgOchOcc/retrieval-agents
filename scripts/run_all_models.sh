#!/bin/bash
# Run benchmark for all supported models

set -e

MODELS=(
    "BAAI/bge-base-en-v1.5"
    "BAAI/bge-large-en-v1.5"
    "intfloat/e5-base-v2"
    "intfloat/e5-large-v2"
)

DATASET_TYPE=${1:-"distractor"}  # Default to distractor for faster runs
SPLIT=${2:-"validation"}
MAX_SAMPLES=${3:-""}

echo "========================================="
echo "Running benchmark for all models"
echo "Dataset: ${DATASET_TYPE}"
echo "Split: ${SPLIT}"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: ${MAX_SAMPLES}"
fi
echo "========================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Benchmarking: ${model}"
    echo "========================================="

    if [ -n "$MAX_SAMPLES" ]; then
        python main.py \
            --model_name "${model}" \
            --dataset_type "${DATASET_TYPE}" \
            --split "${SPLIT}" \
            --max_samples "${MAX_SAMPLES}"
    else
        python main.py \
            --model_name "${model}" \
            --dataset_type "${DATASET_TYPE}" \
            --split "${SPLIT}"
    fi

    echo ""
done

echo "========================================="
echo "All benchmarks completed!"
echo "========================================="
