# Command Reference - Quick Cheatsheet

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# GPU acceleration (recommended)
pip uninstall faiss-cpu && pip install faiss-gpu
```

## Setup

```bash
# Test your setup
python test_setup.py

# Download Wikipedia corpus (small subset for testing)
python download_wikipedia.py --max_passages 100000

# Download full Wikipedia corpus
python download_wikipedia.py
```

## Basic Usage

```bash
# Quick test (100 samples, GPU)
python benchmark.py --model bge-base --max_samples 100

# Full benchmark (default: GPU encoding + GPU indexing)
python benchmark.py --model bge-base

# CPU only
python benchmark.py --model bge-base --device cpu
```

## Model Selection

```bash
# BGE models (BAAI)
python benchmark.py --model bge-small   # Fast, 768-dim
python benchmark.py --model bge-base    # Balanced, 768-dim (recommended)
python benchmark.py --model bge-large   # Best quality, 1024-dim

# E5 models (Microsoft)
python benchmark.py --model e5-small    # Fast, 384-dim
python benchmark.py --model e5-base     # Balanced, 768-dim
python benchmark.py --model e5-large    # Best quality, 1024-dim

# GTE models (Alibaba)
python benchmark.py --model gte-base    # 768-dim
python benchmark.py --model gte-large   # 1024-dim

# Custom model
python benchmark.py --model "sentence-transformers/all-MiniLM-L6-v2"
```

## GPU Configuration

```bash
# GPU for both encoding and indexing (default)
python benchmark.py --model bge-base --device cuda

# GPU encoding, CPU indexing
python benchmark.py --model bge-base --device cuda --no_gpu_index

# Use specific GPU (for multi-GPU systems)
python benchmark.py --model bge-base --gpu_id 1

# CPU only
python benchmark.py --model bge-base --device cpu
```

## Index Types

```bash
# Flat index - Exact search (default)
python benchmark.py --model bge-base --index_type Flat
# Best for: <1M docs, 100% accuracy, fastest on GPU

# IVF index - Approximate search
python benchmark.py --model bge-base --index_type IVF
# Best for: 1M-10M docs, ~99% accuracy, good GPU speedup

# IVFPQ index - Compressed approximate search
python benchmark.py --model bge-base --index_type IVFPQ
# Best for: >10M docs, ~95-98% accuracy, memory efficient
```

## Batch Size Tuning

```bash
# Small GPU memory
python benchmark.py --model bge-base --batch_size 16

# Medium GPU memory (default)
python benchmark.py --model bge-base --batch_size 32

# Large GPU memory
python benchmark.py --model bge-base --batch_size 64

# Very large GPU memory
python benchmark.py --model bge-large --batch_size 128
```

## Dataset Configuration

```bash
# Full Wikipedia test set (default)
python benchmark.py --model bge-base --dataset_config fullwiki

# With distractor paragraphs
python benchmark.py --model bge-base --dataset_config distractor

# Limit number of test samples
python benchmark.py --model bge-base --max_samples 500
```

## Performance Optimization

```bash
# Maximum performance (GPU, IVF, large batch)
python benchmark.py --model bge-base \
    --device cuda \
    --index_type IVF \
    --batch_size 64

# Memory efficient (small model, IVFPQ, small batch)
python benchmark.py --model bge-small \
    --device cuda \
    --index_type IVFPQ \
    --batch_size 16

# Multi-GPU setup (GPU 0 for model, GPU 1 for index)
CUDA_VISIBLE_DEVICES=0,1 python benchmark.py \
    --model bge-base \
    --device cuda \
    --gpu_id 1
```

## Comparison Workflows

```bash
# Compare BGE models
for model in bge-small bge-base bge-large; do
    python benchmark.py --model $model --max_samples 1000
done

# Compare E5 models
for model in e5-small e5-base e5-large; do
    python benchmark.py --model $model --max_samples 1000
done

# Compare index types
for index in Flat IVF IVFPQ; do
    python benchmark.py --model bge-base --index_type $index --max_samples 1000
done

# Compare GPU vs CPU
time python benchmark.py --model bge-base --device cuda --max_samples 1000
time python benchmark.py --model bge-base --device cpu --max_samples 1000
```

## Troubleshooting Commands

```bash
# Check setup
python test_setup.py

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS GPUs: {faiss.get_num_gpus()}')"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Clear cache and rebuild
rm -rf cache/index/
python benchmark.py --model bge-base --max_samples 100
```

## Output Locations

```bash
# Wikipedia corpus cache
cache/wikipedia_paragraphs.json

# FAISS indices
cache/index/<model_name>/index.faiss
cache/index/<model_name>/doc_ids.pkl
cache/index/<model_name>/config.pkl

# Results
cache/results/<model_name>_<config>_<timestamp>.json
```

## Common Scenarios

### Testing New Model Quickly
```bash
python download_wikipedia.py --max_passages 10000
python benchmark.py --model bge-base --max_samples 100
```

### Full Benchmark with GPU
```bash
python download_wikipedia.py  # Full corpus
python benchmark.py --model bge-base --device cuda --batch_size 64
```

### Memory-Constrained Setup
```bash
python download_wikipedia.py --max_passages 50000
python benchmark.py --model bge-small --batch_size 16 --index_type IVFPQ
```

### Production Evaluation
```bash
# Download full corpus once
python download_wikipedia.py

# Run multiple models
python benchmark.py --model bge-base --index_type IVF
python benchmark.py --model bge-large --index_type IVF
python benchmark.py --model e5-base --index_type IVF
python benchmark.py --model e5-large --index_type IVF
```

## Tips

- 💡 Always run `test_setup.py` first
- 💡 Use `--max_samples 100` for quick testing
- 💡 Use `--index_type IVF` for corpora >1M docs
- 💡 Use `--batch_size 64` or higher with GPU
- 💡 Monitor GPU with `nvidia-smi` during runs
- 💡 Results are cached - delete `cache/index/` to rebuild

## Help

```bash
# Show all options
python benchmark.py --help

# Read documentation
cat README.md
cat GPU_GUIDE.md
cat QUICKSTART.md
```
