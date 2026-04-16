#!/bin/bash
# Benchmark BGE-large model on HotpotQA

echo "========================================="
echo "Benchmarking: BAAI/bge-large-en-v1.5"
echo "========================================="

python main.py \
    --model_name "BAAI/bge-large-en-v1.5" \
    --dataset_type "distractor" \
    --split "validation" \
    --max_samples 100

# For full benchmark, remove --max_samples:
# python main.py \
#     --model_name "BAAI/bge-large-en-v1.5" \
#     --dataset_type "fullwiki" \
#     --split "validation"
