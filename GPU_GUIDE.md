# GPU Acceleration Guide

This guide explains how to use GPU acceleration for both model encoding and FAISS indexing.

## Overview

The benchmark supports GPU acceleration in two places:
1. **Model Encoding** (embedding generation) - Uses PyTorch
2. **FAISS Indexing** (similarity search) - Uses FAISS GPU support

## Setup

### 1. Install FAISS-GPU

First, uninstall the CPU version and install GPU version:

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

Or with conda:
```bash
conda install -c pytorch faiss-gpu
```

### 2. Verify GPU Availability

Run the test script:
```bash
python test_setup.py
```

This will show:
- ✓ CUDA available (GPU: Tesla V100)
- Or ⚠ CUDA not available (will use CPU)

## Usage

### Default Behavior (GPU Enabled)

By default, both model encoding and indexing use GPU if available:

```bash
# GPU for encoding + GPU for indexing
python benchmark.py --model bge-base --batch_size 32
```

### Command Line Options

#### Use GPU for both encoding and indexing (default)
```bash
python benchmark.py --model bge-base --device cuda
```

#### Use GPU for encoding, CPU for indexing
```bash
python benchmark.py --model bge-base --device cuda --no_gpu_index
```

#### Use CPU for both
```bash
python benchmark.py --model bge-base --device cpu
```

#### Specify GPU device ID (for multi-GPU systems)
```bash
# Use GPU 0 for encoding, GPU 1 for indexing
python benchmark.py --model bge-base --device cuda --gpu_id 1
```

## FAISS Index Types

Different index types have different GPU performance characteristics:

### Flat Index (Exact Search)
```bash
python benchmark.py --model bge-base --index_type Flat
```
- **GPU Benefit**: High - search is much faster on GPU
- **Memory**: Stores all vectors in GPU memory
- **Accuracy**: 100% (exact search)
- **Best for**: Small to medium corpora (<1M documents)

### IVF Index (Approximate Search)
```bash
python benchmark.py --model bge-base --index_type IVF
```
- **GPU Benefit**: Very High - both training and search accelerated
- **Memory**: Moderate - uses clustering
- **Accuracy**: ~99% (configurable)
- **Best for**: Large corpora (>1M documents)

### IVFPQ Index (Compressed)
```bash
python benchmark.py --model bge-base --index_type IVFPQ
```
- **GPU Benefit**: High - good for very large corpora
- **Memory**: Low - uses product quantization
- **Accuracy**: ~95-98% (configurable)
- **Best for**: Very large corpora (>10M documents) with limited GPU memory

## Performance Comparison

### Model Encoding (100K documents, BGE-base)
- **CPU**: ~15-20 minutes
- **GPU (V100)**: ~2-3 minutes
- **Speedup**: ~6-8x

### FAISS Indexing (1M documents)

**Index Construction:**
- Flat CPU: ~5 seconds
- Flat GPU: ~2 seconds

**Search (1000 queries, k=10):**
- Flat CPU: ~10 seconds
- Flat GPU: ~0.5 seconds
- **Speedup**: ~20x

**IVF Training + Search:**
- IVF CPU: ~60 seconds
- IVF GPU: ~5 seconds
- **Speedup**: ~12x

## Memory Management

### GPU Memory Requirements

For a corpus with:
- 1M documents
- 768-dim embeddings (BGE-base)
- Flat index

**Memory needed**: ~3GB GPU memory

Formula: `num_docs × embedding_dim × 4 bytes`

### If You Run Out of GPU Memory

**Option 1: Use IVF or IVFPQ index**
```bash
# IVF uses ~50% less memory
python benchmark.py --model bge-base --index_type IVF

# IVFPQ uses ~75% less memory
python benchmark.py --model bge-base --index_type IVFPQ
```

**Option 2: Use CPU for indexing**
```bash
python benchmark.py --model bge-base --device cuda --no_gpu_index
```

**Option 3: Reduce batch size**
```bash
python benchmark.py --model bge-base --batch_size 16
```

**Option 4: Use smaller model**
```bash
python benchmark.py --model bge-small --batch_size 32
```

## Multi-GPU Setup

If you have multiple GPUs:

```bash
# Use GPU 0 for model, GPU 1 for index
CUDA_VISIBLE_DEVICES=0 python benchmark.py --model bge-base --gpu_id 1
```

Or specify in Python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

config = BenchmarkConfig(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda",  # Model uses GPU 0
    use_gpu_index=True,
    gpu_id=1,  # Index uses GPU 1
)
```

## Troubleshooting

### Error: "CUDA out of memory"

**Solution 1**: Reduce batch size
```bash
python benchmark.py --model bge-base --batch_size 8
```

**Solution 2**: Use CPU for indexing
```bash
python benchmark.py --model bge-base --no_gpu_index
```

**Solution 3**: Use compressed index
```bash
python benchmark.py --model bge-base --index_type IVFPQ
```

### Error: "FAISS was not compiled with GPU support"

**Solution**: Reinstall faiss-gpu
```bash
pip uninstall faiss-cpu faiss-gpu
pip install faiss-gpu
```

### Error: "GPU requested but no GPU available"

The code will automatically fall back to CPU:
```
Warning: GPU requested but no GPU available. Using CPU instead.
```

No action needed - will use CPU automatically.

### Slow GPU Performance

**Check 1**: Verify GPU is actually being used
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

**Check 2**: Check GPU utilization
```bash
# In another terminal
nvidia-smi -l 1
```

Should show high GPU utilization during encoding and search.

## Benchmarking GPU vs CPU

To compare GPU vs CPU performance:

```bash
# GPU benchmark
time python benchmark.py --model bge-base --device cuda --max_samples 1000

# CPU benchmark
time python benchmark.py --model bge-base --device cpu --max_samples 1000
```

## Best Practices

1. **Always use GPU for model encoding** if available (6-8x speedup)
2. **Use GPU for indexing** with Flat or IVF for best performance
3. **For corpora >1M docs**, use IVF index on GPU
4. **For corpora >10M docs**, use IVFPQ index or split across multiple GPUs
5. **Monitor GPU memory** with `nvidia-smi` during runs
6. **Start with small samples** (`--max_samples 100`) to test setup

## Example Commands

### Quick Test with GPU
```bash
# Test with 100 samples
python benchmark.py --model bge-base --max_samples 100 --device cuda
```

### Full Benchmark with GPU Optimization
```bash
# Full test set, optimized for GPU
python benchmark.py --model bge-base \
    --device cuda \
    --batch_size 64 \
    --index_type IVF \
    --gpu_id 0
```

### Maximum Performance (Multi-GPU)
```bash
# Use 2 GPUs - GPU 0 for model, GPU 1 for index
CUDA_VISIBLE_DEVICES=0,1 python benchmark.py \
    --model bge-large \
    --device cuda \
    --batch_size 128 \
    --index_type IVF \
    --gpu_id 1
```

### Memory-Efficient GPU Setup
```bash
# Use IVFPQ for low memory usage
python benchmark.py --model bge-small \
    --device cuda \
    --batch_size 32 \
    --index_type IVFPQ
```

## Expected Speedups

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Encode 100K docs (BGE-base) | 15 min | 2 min | 7.5x |
| Build Flat index (1M docs) | 5 sec | 2 sec | 2.5x |
| Search Flat (1K queries) | 10 sec | 0.5 sec | 20x |
| Build IVF index (1M docs) | 45 sec | 4 sec | 11x |
| Search IVF (1K queries) | 2 sec | 0.2 sec | 10x |

## Summary

- ✅ **GPU encoding**: 6-8x faster for embedding generation
- ✅ **GPU indexing**: 10-20x faster for similarity search
- ✅ **Automatic fallback**: Uses CPU if GPU unavailable
- ✅ **Flexible configuration**: Control GPU usage per component
- ✅ **Multi-GPU support**: Can use different GPUs for encoding and indexing

For best performance: **Use GPU for both encoding and indexing!**
