#!/bin/bash
# Run single sampling experiment

echo "========================================="
echo "Sampling Retrieval Experiment"
echo "========================================="

python benchmark_sampling.py \
    --model_name "BAAI/bge-base-en-v1.5" \
    --dataset_type "distractor" \
    --split "validation" \
    --max_samples 100 \
    --n 100 \
    --p 0.5 \
    --k 20 \
    --k_values 1 5 10 20

# Parameters explanation:
# --n 100: Retrieve top-100 documents initially
# --p 0.5: Sample each document with probability 0.5 (expect ~50 docs)
# --k 20: Select top-20 from sampled documents
# --k_values: Evaluate at k=1,5,10,20
