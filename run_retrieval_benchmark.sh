 #!/bin/bash

# ================================================================
# Script chạy benchmark retrieval với 2 settings:
# 1. Normal Retrieval (top_k)
# 2. Retrieval with Sampling (random_sample)
# ================================================================

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Chỉ định GPU (thay đổi nếu cần)

# Configuration
MODEL="bge-base"              # Model: bge-base, bge-large, e5-base, e5-large, gte-base, gte-large
BATCH_SIZE=32                 # Batch size
DEVICE="cuda"                 # cuda hoặc cpu
CACHE_DIR="./cache"           # Cache directory
DATASET_SPLIT="test"          # Dataset split
DATASET_CONFIG="fullwiki"     # fullwiki hoặc distractor
MAX_SAMPLES=""                # Để trống để chạy full dataset, hoặc set số như "--max_samples 100" để test

# Sampling parameters
EXPANSION_FACTOR=4            # Retrieve top (expansion_factor * k) documents
RANDOM_RATIO=0.3              # Ratio of random sampling (0 < x < 1)
SAMPLING_SEED=42              # Random seed

# Create results directory
mkdir -p logs

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "================================================================"
echo "RETRIEVAL BENCHMARK - Running 2 Settings"
echo "================================================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Dataset: $DATASET_CONFIG"
echo "================================================================"
echo ""

# ================================================================
# Setting 1: Normal Retrieval (top_k - No Sampling)
# ================================================================
echo "================================================================"
echo "SETTING 1: Normal Retrieval (Top-K)"
echo "================================================================"
echo "Running benchmark with standard top-k retrieval..."
echo ""

python benchmark.py \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR" \
    --dataset_split "$DATASET_SPLIT" \
    --dataset_config "$DATASET_CONFIG" \
    --sampling_strategy "top_k" \
    $MAX_SAMPLES \
    2>&1 | tee "logs/${MODEL}_normal_retrieval_${TIMESTAMP}.log"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Setting 1 completed successfully!"
    echo ""
else
    echo ""
    echo "✗ Setting 1 failed! Check logs for errors."
    echo ""
    exit 1
fi

# ================================================================
# Setting 2: Retrieval with Sampling (random_sample)
# ================================================================
echo "================================================================"
echo "SETTING 2: Retrieval with Sampling (Random Sample)"
echo "================================================================"
echo "Running benchmark with random sampling strategy..."
echo "  Expansion factor: $EXPANSION_FACTOR"
echo "  Random ratio: $RANDOM_RATIO"
echo ""

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
    2>&1 | tee "logs/${MODEL}_sampling_retrieval_${TIMESTAMP}.log"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Setting 2 completed successfully!"
    echo ""
else
    echo ""
    echo "✗ Setting 2 failed! Check logs for errors."
    echo ""
    exit 1
fi

# ================================================================
# Summary
# ================================================================
echo "================================================================"
echo "BENCHMARK COMPLETED"
echo "================================================================"
echo ""
echo "Results saved to: $CACHE_DIR/results/"
echo ""
echo "Logs:"
echo "  - Normal Retrieval: logs/${MODEL}_normal_retrieval_${TIMESTAMP}.log"
echo "  - Sampling Retrieval: logs/${MODEL}_sampling_retrieval_${TIMESTAMP}.log"
echo ""
echo "================================================================"
echo ""

# Optional: Compare results
echo "To view results, check files in: $CACHE_DIR/results/"
echo ""
