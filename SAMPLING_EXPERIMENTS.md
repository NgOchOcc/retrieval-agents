# Sampling Retrieval Experiments

## Overview

This document describes the **random sampling retrieval strategy** and how to run experiments comparing it against baseline retrieval.

## Strategy

### Baseline Retrieval
1. Retrieve top-k documents directly
2. Return top-k sorted by score (descending)

### Sampling Retrieval
1. **Retrieve top-n documents** (n > k)
2. **Randomly sample** each document with probability p (0 < p < 1)
3. **Select top-k** from sampled documents by score (descending)

### Example
- n = 100 (retrieve top-100)
- p = 0.5 (sample each with 50% probability → expect ~50 docs)
- k = 20 (final top-20 from sampled docs)

## Quick Start

### Single Experiment

```bash
bash scripts/run_sampling_single.sh
```

This runs:
- Model: BGE-base
- n=100, p=0.5, k=20
- 100 samples from distractor dataset

### Test Different p Values

```bash
bash scripts/run_sampling_experiment.sh
```

This tests p ∈ {0.3, 0.5, 0.7, 0.9} with same model and n, k.

### Compare Across Models

```bash
bash scripts/run_sampling_models.sh
```

This runs the same sampling strategy (n=100, p=0.5, k=20) across all 4 models.

## Custom Experiments

### Example 1: Low Sampling Probability

```bash
python benchmark_sampling.py \
    --model_name "BAAI/bge-base-en-v1.5" \
    --n 200 \
    --p 0.2 \
    --k 20 \
    --max_samples 100
```

Expected: ~40 docs sampled from top-200, select top-20.

### Example 2: High Sampling Probability

```bash
python benchmark_sampling.py \
    --model_name "intfloat/e5-large-v2" \
    --n 50 \
    --p 0.9 \
    --k 10 \
    --max_samples 500
```

Expected: ~45 docs sampled from top-50, select top-10.

### Example 3: Full Dataset Comparison

```bash
python benchmark_sampling.py \
    --model_name "BAAI/bge-large-en-v1.5" \
    --dataset_type "fullwiki" \
    --split "validation" \
    --n 100 \
    --p 0.5 \
    --k 20
```

## Output Format

The script outputs a comparison table:

```
================================================================================
BASELINE vs SAMPLING COMPARISON
================================================================================

Sampling Strategy:
  - Top-n retrieved: 100
  - Sampling probability (p): 0.50
  - Final top-k: 20
  - Expected samples per query: 50.0

Actual Sampling Statistics:
  - Avg sampled docs: 49.87
  - Min sampled docs: 32
  - Max sampled docs: 68
  - Queries with < k docs: 0/100

================================================================================
Metric          Baseline        Sampling        Difference
================================================================================

STRICT (pass@k) - Both docs in top-k:
--------------------------------------------------------------------------------
pass@1          0.1200          0.0800          -0.0400
pass@5          0.3500          0.3100          -0.0400
pass@10         0.4800          0.4300          -0.0500
pass@20         0.6200          0.5500          -0.0700

RELAXED (hit@k) - At least one doc in top-k:
--------------------------------------------------------------------------------
hit@1           0.4500          0.3900          -0.0600
hit@5           0.7800          0.7200          -0.0600
hit@10          0.8700          0.8100          -0.0600
hit@20          0.9400          0.8800          -0.0600
================================================================================
```

## Results Analysis

### Expected Behavior

1. **p → 1**: Sampling → Baseline (almost all docs sampled)
2. **p → 0**: More randomness, likely worse performance
3. **Trade-off**: Lower p = more diversity but potentially miss high-scoring docs

### Key Questions

- How does performance degrade as p decreases?
- Is there a sweet spot where sampling maintains decent performance?
- Does the impact differ between BGE and E5 models?
- Does it matter more for strict (pass@k) vs relaxed (hit@k)?

## Parameters Reference

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n` | Initial top-n to retrieve | 50-200 |
| `p` | Sampling probability | 0.1-0.9 |
| `k` | Final top-k after sampling | 10-50 |
| `k_values` | Evaluation k values | [1, 5, 10, 20] |

### Constraints

- 0 < p < 1
- k ≤ n
- On average, expect ~(n × p) sampled documents
- If sampled docs < k, returns all sampled docs

## File Locations

### Scripts
- `scripts/run_sampling_single.sh` - Single experiment
- `scripts/run_sampling_experiment.sh` - Test multiple p values
- `scripts/run_sampling_models.sh` - Test multiple models

### Code
- `retrieval/sampling_retriever.py` - SamplingRetriever class
- `benchmark_sampling.py` - Main comparison script

### Results
- Saved to: `results/sampling/sampling_comparison_*.json`
- Contains: baseline metrics, sampling metrics, statistics

## Advanced Usage

### Seed Control

```bash
python benchmark_sampling.py \
    --seed 123 \
    --n 100 --p 0.5 --k 20
```

Different seeds will produce different random samples.

### Batch Size Tuning

```bash
python benchmark_sampling.py \
    --passage_batch_size 256 \
    --query_batch_size 64 \
    --n 100 --p 0.5 --k 20
```

Larger batches = faster but more memory.

## Implementation Details

The `SamplingRetriever` class extends `FAISSRetriever`:

```python
# 1. Retrieve top-n with scores
results = self.retrieve(queries, k=n)

# 2. Sample each doc with probability p
sampled = [doc for doc in results if random.random() < p]

# 3. Sort sampled docs by score
sampled.sort(key=score, reverse=True)

# 4. Return top-k
return sampled[:k]
```

Sampling is done **per query**, so each query gets different sampled documents even with same seed (different random draws).
