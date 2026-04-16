#!/bin/bash
# Compare sampling strategy across different models

N=100
P=0.5
K=20
MAX_SAMPLES=100

MODELS=(
    "BAAI/bge-base-en-v1.5"
    "BAAI/bge-large-en-v1.5"
    "intfloat/e5-base-v2"
    "intfloat/e5-large-v2"
)

echo "========================================="
echo "Sampling Strategy - Model Comparison"
echo "n=${N}, p=${P}, k=${K}"
echo "========================================="

for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================="
    echo "Model: ${model}"
    echo "========================================="

    python benchmark_sampling.py \
        --model_name "${model}" \
        --dataset_type "distractor" \
        --split "validation" \
        --max_samples ${MAX_SAMPLES} \
        --n ${N} \
        --p ${P} \
        --k ${K} \
        --k_values 1 5 10 20

    echo ""
done

echo "========================================="
echo "All models completed!"
echo "========================================="
