#!/bin/bash
# Comprehensive sampling sweep to find optimal p value
# Assumption: Model is acceptable (not perfect), sampling might help

MODEL="BAAI/bge-base-en-v1.5"
MAX_SAMPLES=500  # More samples for reliable statistics

echo "========================================="
echo "SAMPLING STRATEGY SWEEP"
echo "Hypothesis: Random sampling can improve"
echo "retrieval for acceptable (not perfect) models"
echo "========================================="

# Test 1: Fixed n=100, k=20, sweep p
echo ""
echo "========================================="
echo "Experiment 1: Sweep p values"
echo "n=100, k=20, varying p"
echo "========================================="

P_VALUES=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for p in "${P_VALUES[@]}"; do
    echo ""
    echo "Testing p=${p}..."
    python benchmark_sampling.py \
        --model_name "${MODEL}" \
        --dataset_type "distractor" \
        --max_samples ${MAX_SAMPLES} \
        --n 100 \
        --p ${p} \
        --k 20 \
        --k_values 1 5 10 20
done

# Test 2: Fixed p=0.5, sweep n values
echo ""
echo "========================================="
echo "Experiment 2: Sweep n values"
echo "p=0.5, k=20, varying n"
echo "========================================="

N_VALUES=(50 100 200 300)

for n in "${N_VALUES[@]}"; do
    echo ""
    echo "Testing n=${n}..."
    python benchmark_sampling.py \
        --model_name "${MODEL}" \
        --dataset_type "distractor" \
        --max_samples ${MAX_SAMPLES} \
        --n ${n} \
        --p 0.5 \
        --k 20 \
        --k_values 1 5 10 20
done

# Test 3: Different (n, p, k) combinations
echo ""
echo "========================================="
echo "Experiment 3: Strategic combinations"
echo "========================================="

# Strategy A: Large pool, low p
echo ""
echo "Strategy A: n=200, p=0.3, k=20 (large pool, aggressive sampling)"
python benchmark_sampling.py \
    --model_name "${MODEL}" \
    --dataset_type "distractor" \
    --max_samples ${MAX_SAMPLES} \
    --n 200 \
    --p 0.3 \
    --k 20 \
    --k_values 1 5 10 20

# Strategy B: Medium pool, medium p
echo ""
echo "Strategy B: n=100, p=0.5, k=20 (balanced)"
python benchmark_sampling.py \
    --model_name "${MODEL}" \
    --dataset_type "distractor" \
    --max_samples ${MAX_SAMPLES} \
    --n 100 \
    --p 0.5 \
    --k 20 \
    --k_values 1 5 10 20

# Strategy C: Small pool, high p
echo ""
echo "Strategy C: n=50, p=0.7, k=20 (conservative)"
python benchmark_sampling.py \
    --model_name "${MODEL}" \
    --dataset_type "distractor" \
    --max_samples ${MAX_SAMPLES} \
    --n 50 \
    --p 0.7 \
    --k 20 \
    --k_values 1 5 10 20

echo ""
echo "========================================="
echo "SWEEP COMPLETED"
echo "Results in: results/sampling/"
echo "========================================="
echo ""
echo "Key questions to analyze:"
echo "1. Does any p value beat baseline?"
echo "2. What's the optimal (n, p, k) combination?"
echo "3. Is the improvement consistent across k values?"
echo "4. Trade-off between pass@k and hit@k?"
