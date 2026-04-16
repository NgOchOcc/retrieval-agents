#!/bin/bash
# Run sampling experiment with different p values

MODEL="BAAI/bge-base-en-v1.5"
N=100
K=20
MAX_SAMPLES=100

echo "========================================="
echo "Sampling Experiment"
echo "Model: ${MODEL}"
echo "Strategy: Retrieve top-${N}, sample with p, select top-${K}"
echo "========================================="

# Test different sampling probabilities
P_VALUES=(0.3 0.5 0.7 0.9)

for p in "${P_VALUES[@]}"; do
    echo ""
    echo "========================================="
    echo "Running with p=${p}"
    echo "========================================="

    python benchmark_sampling.py \
        --model_name "${MODEL}" \
        --dataset_type "distractor" \
        --split "validation" \
        --max_samples ${MAX_SAMPLES} \
        --n ${N} \
        --p ${p} \
        --k ${K} \
        --k_values 1 5 10 20

    echo ""
done

echo "========================================="
echo "Experiment completed!"
echo "Results saved in results/sampling/"
echo "========================================="
