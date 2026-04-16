#!/bin/bash

# ================================================================
# Script chạy benchmark retrieval với nhiều settings
# Bao gồm: Normal retrieval và 2 loại sampling strategies
# ================================================================

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Configuration - CÓ THỂ THAY ĐỔI Ở ĐÂY
MODEL="bge-base"              # bge-base, bge-large, e5-base, e5-large, gte-base, gte-large
BATCH_SIZE=32
DEVICE="cuda"                 # cuda hoặc cpu
CACHE_DIR="./cache"
DATASET_SPLIT="test"
DATASET_CONFIG="fullwiki"     # fullwiki hoặc distractor
MAX_SAMPLES=""                # Để trống cho full dataset, hoặc set "--max_samples 100" để test

# Sampling parameters
EXPANSION_FACTOR=4
RANDOM_RATIO=0.3
SAMPLING_SEED=42

# Create directories
mkdir -p logs
mkdir -p results_comparison

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "================================================================"
echo "RETRIEVAL BENCHMARK - All Settings Comparison"
echo "================================================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Timestamp: $TIMESTAMP"
echo "================================================================"
echo ""

# ================================================================
# Setting 1: Normal Retrieval (top_k)
# ================================================================
echo ""
echo "▶ [1/3] Running Setting 1: Normal Retrieval (Top-K)"
echo "================================================================"

python benchmark.py \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR" \
    --dataset_split "$DATASET_SPLIT" \
    --dataset_config "$DATASET_CONFIG" \
    --sampling_strategy "top_k" \
    $MAX_SAMPLES \
    2>&1 | tee "logs/01_normal_${TIMESTAMP}.log"

if [ $? -ne 0 ]; then
    echo "✗ Setting 1 failed!"
    exit 1
fi
echo "✓ Setting 1 completed"

# ================================================================
# Setting 2: Random Sampling
# ================================================================
echo ""
echo "▶ [2/3] Running Setting 2: Random Sampling"
echo "================================================================"
echo "Parameters: expansion_factor=$EXPANSION_FACTOR, random_ratio=$RANDOM_RATIO"

python benchmark.py \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR" \
    --dataset_split "$DATASET_SPLIT" \
    --dataset_config "$DATASET_CONFIG" \
    --sampling_strategy "random_sample" \
    --expansion_factor $EXPANSION_FACTOR \
    --random_ratio $RANDOM_RATIO \
    --sampling_seed $SAMPLING_SEED \
    $MAX_SAMPLES \
    2>&1 | tee "logs/02_random_sampling_${TIMESTAMP}.log"

if [ $? -ne 0 ]; then
    echo "✗ Setting 2 failed!"
    exit 1
fi
echo "✓ Setting 2 completed"

# ================================================================
# Setting 3: Diverse Sampling (Optional - stratified sampling)
# ================================================================
echo ""
echo "▶ [3/3] Running Setting 3: Diverse Sampling (Stratified)"
echo "================================================================"

python benchmark.py \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR" \
    --dataset_split "$DATASET_SPLIT" \
    --dataset_config "$DATASET_CONFIG" \
    --sampling_strategy "diverse_sample" \
    --expansion_factor $EXPANSION_FACTOR \
    --random_ratio $RANDOM_RATIO \
    --sampling_seed $SAMPLING_SEED \
    $MAX_SAMPLES \
    2>&1 | tee "logs/03_diverse_sampling_${TIMESTAMP}.log"

if [ $? -ne 0 ]; then
    echo "✗ Setting 3 failed!"
    exit 1
fi
echo "✓ Setting 3 completed"

# ================================================================
# Summary
# ================================================================
echo ""
echo "================================================================"
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY"
echo "================================================================"
echo ""
echo "Results location: $CACHE_DIR/results/"
echo ""
echo "Logs:"
echo "  1. Normal Retrieval:   logs/01_normal_${TIMESTAMP}.log"
echo "  2. Random Sampling:    logs/02_random_sampling_${TIMESTAMP}.log"
echo "  3. Diverse Sampling:   logs/03_diverse_sampling_${TIMESTAMP}.log"
echo ""
echo "================================================================"
echo ""

# Create a summary comparison file
echo "Creating summary comparison..."
cat > "results_comparison/summary_${TIMESTAMP}.txt" << EOF
================================================================
RETRIEVAL BENCHMARK SUMMARY - ${TIMESTAMP}
================================================================

Model: $MODEL
Dataset: $DATASET_CONFIG ($DATASET_SPLIT)
Device: $DEVICE

Settings Tested:
1. Normal Retrieval (Top-K)
   - Strategy: top_k
   - No sampling applied

2. Random Sampling
   - Strategy: random_sample
   - Expansion Factor: $EXPANSION_FACTOR
   - Random Ratio: $RANDOM_RATIO

3. Diverse Sampling
   - Strategy: diverse_sample
   - Expansion Factor: $EXPANSION_FACTOR
   - Random Ratio: $RANDOM_RATIO (stratified)

================================================================
Check the detailed JSON results in: $CACHE_DIR/results/
================================================================
EOF

echo "Summary saved to: results_comparison/summary_${TIMESTAMP}.txt"
echo ""
echo "Done!"
