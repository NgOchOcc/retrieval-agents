# Sampling Strategies Guide

## Overview

Thay vì chỉ chọn top-k documents, bạn có thể sử dụng sampling strategies để:
- Test robustness của retrieval system
- Add randomness để tránh overfitting vào top results
- Explore diverse results

## Sampling Strategies

### 1. Standard Top-K (Mặc định)
Chọn top-k documents có score cao nhất.

```bash
python benchmark.py \
    --model bge-base \
    --sampling_strategy top_k
```

### 2. Random Sampling (Của bạn)
**Process:**
1. Retrieve top (4*k) documents
2. Random sample với ratio x từ pool này
3. Chọn top-k từ sampled pool

```bash
python benchmark.py \
    --model bge-base \
    --sampling_strategy random_sample \
    --expansion_factor 4 \
    --random_ratio 0.3
```

**Parameters:**
- `expansion_factor=4`: Retrieve top-80 nếu k=20
- `random_ratio=0.3`: Random sample 30% của pool (24 docs từ 80)
- Sau đó chọn top-20 từ 24 docs

### 3. Diverse Sampling
Stratified sampling để ensure diversity.

```bash
python benchmark.py \
    --model bge-base \
    --sampling_strategy diverse_sample \
    --expansion_factor 4 \
    --random_ratio 0.5
```

## Examples

### Baseline (Standard Top-K)
```bash
python benchmark.py \
    --model bge-base \
    --dataset_split validation \
    --max_samples 1000 \
    --sampling_strategy top_k
```

**Output:**
```
STEP 4: Retrieving Documents
Searching index...
Retrieving top-20
```

### Random Sampling (Your Request)
```bash
python benchmark.py \
    --model bge-base \
    --dataset_split validation \
    --max_samples 1000 \
    --sampling_strategy random_sample \
    --expansion_factor 4 \
    --random_ratio 0.3 \
    --sampling_seed 42
```

**Output:**
```
Setup complete!
Using sampling strategy: random_sample
  Expansion factor: 4
  Random ratio: 0.3

STEP 4: Retrieving Documents
Searching index...
Retrieving top-80 (expansion_factor=4)
Applying sampling strategy: random_sample
  Random ratio: 0.3
```

**Process for k=20:**
1. Retrieve top-80 documents (4 * 20)
2. Random sample 24 documents (30% of 80)
3. Sort by score and select top-20

### Compare Strategies
```bash
# Baseline
python benchmark.py --model bge-base --max_samples 1000 --sampling_strategy top_k

# Random 30%
python benchmark.py --model bge-base --max_samples 1000 \
    --sampling_strategy random_sample --random_ratio 0.3

# Random 50%
python benchmark.py --model bge-base --max_samples 1000 \
    --sampling_strategy random_sample --random_ratio 0.5

# Diverse
python benchmark.py --model bge-base --max_samples 1000 \
    --sampling_strategy diverse_sample --random_ratio 0.5
```

## Parameters Explained

### `--expansion_factor` (default: 4)
- Retrieve top `(expansion_factor * k)` documents
- Larger = more pool để sample từ đó
- **Recommended**: 4-10

```bash
# Retrieve top-200 for k=20
--expansion_factor 10
```

### `--random_ratio` (default: 0.3)
- Tỉ lệ random sampling (0 < x < 1)
- 0.3 = sample 30% của expanded pool
- Larger = more randomness, more diversity

```bash
# Sample 50% of pool
--random_ratio 0.5
```

### `--sampling_seed` (default: 42)
- Random seed for reproducibility
- Same seed = same results

```bash
# Different seed for different random samples
--sampling_seed 123
```

## Results Format

Results file sẽ include sampling config:

```json
{
  "model_name": "BAAI/bge-base-en-v1.5",
  "sampling_strategy": "random_sample",
  "expansion_factor": 4,
  "random_ratio": 0.3,
  "metrics": {
    "pass@1": 0.65,
    "pass@5": 0.82,
    ...
  }
}
```

## Use Cases

### 1. Test Robustness
So sánh top-k vs random sampling:

```bash
# Baseline
python benchmark.py --model bge-base --sampling_strategy top_k

# With noise
python benchmark.py --model bge-base --sampling_strategy random_sample --random_ratio 0.3
```

Nếu metrics giảm nhiều → model không robust.

### 2. Avoid Overfitting to Top Results
Random sampling giúp tránh overfitting vào top-ranked docs.

### 3. Diversity Testing
Test xem model có retrieve được diverse results không:

```bash
python benchmark.py --model bge-base --sampling_strategy diverse_sample --random_ratio 0.5
```

## Demo Script

Test sampling strategies:

```bash
python sampling_strategies.py
```

Output:
```
============================================================
Sampling Strategy Demonstration
============================================================

Input: 100 documents, k=5
Top 20 scores: [1.0, 0.99, 0.98, ...]

------------------------------------------------------------
Strategy 1: Standard Top-K
------------------------------------------------------------
Selected docs: ['doc_0', 'doc_1', 'doc_2', 'doc_3', 'doc_4']
Selected scores: [1.0, 0.99, 0.98, 0.97, 0.96]

------------------------------------------------------------
Strategy 2: Random Sampling (expansion_factor=4, random_ratio=0.3)
------------------------------------------------------------
Selected docs: ['doc_1', 'doc_7', 'doc_3', 'doc_12', 'doc_5']
Selected scores: [0.99, 0.93, 0.97, 0.88, 0.95]
Process: top-20 → random sample 6 docs → top-5

------------------------------------------------------------
Strategy 3: Diverse Sampling (stratified)
------------------------------------------------------------
Selected docs: ['doc_0', 'doc_6', 'doc_12', 'doc_14', 'doc_19']
Selected scores: [1.0, 0.94, 0.88, 0.86, 0.81]
```

## Best Practices

1. **Start with baseline** (top_k) để có reference
2. **Test multiple ratios** (0.2, 0.3, 0.5) để xem impact
3. **Use same seed** cho reproducibility
4. **Compare metrics** để understand trade-offs

## Example Workflow

```bash
# 1. Baseline
python benchmark.py \
    --model bge-base \
    --dataset_split validation \
    --max_samples 1000 \
    --sampling_strategy top_k

# 2. Light randomness
python benchmark.py \
    --model bge-base \
    --dataset_split validation \
    --max_samples 1000 \
    --sampling_strategy random_sample \
    --random_ratio 0.2

# 3. Moderate randomness
python benchmark.py \
    --model bge-base \
    --dataset_split validation \
    --max_samples 1000 \
    --sampling_strategy random_sample \
    --random_ratio 0.3

# 4. High randomness
python benchmark.py \
    --model bge-base \
    --dataset_split validation \
    --max_samples 1000 \
    --sampling_strategy random_sample \
    --random_ratio 0.5
```

## Summary

✅ **3 sampling strategies**: top_k, random_sample, diverse_sample
✅ **Flexible parameters**: expansion_factor, random_ratio, seed
✅ **Easy to use**: Just add `--sampling_strategy` flag
✅ **Results tracked**: Config saved in results JSON

**Your requested method:**
```bash
python benchmark.py \
    --model bge-base \
    --sampling_strategy random_sample \
    --expansion_factor 4 \
    --random_ratio 0.3
```

This retrieves top-4k, randomly samples with 30% ratio, then selects top-k! 🎯
