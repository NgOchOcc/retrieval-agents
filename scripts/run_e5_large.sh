#!/bin/bash
# Benchmark E5-large model on HotpotQA

echo "========================================="
echo "Benchmarking: intfloat/e5-large-v2"
echo "========================================="

python main.py \
    --model_name "intfloat/e5-large-v2" \
    --dataset_type "distractor" \
    --split "validation" \
    --max_samples 100

# For full benchmark, remove --max_samples:
# python main.py \
#     --model_name "intfloat/e5-large-v2" \
#     --dataset_type "fullwiki" \
#     --split "validation"
