# Quick Start Guide

## Setup

```bash
# Install dependencies (with GPU support)
pip install -r requirements.txt
```

## Usage

### 1. Basic Retrieval Benchmark

```bash
# Run with default settings (uses GPU and cache automatically)
python main.py --max_samples 100

# First run: builds index and caches it
# Second run: loads from cache (much faster!)

# Disable cache if needed
python main.py --no_cache
```

### 2. Sampling Retrieval Benchmark

```bash
# Compare baseline vs sampling retrieval
python benchmark_sampling.py --max_samples 100 --n 100 --p 0.5 --k 20

# Parameters:
#   --n: Number of top documents to retrieve
#   --p: Sampling probability (0-1)
#   --k: Final top-k after sampling
```

### 3. Analyze Sampling Results

```bash
# Analyze all experiments in results/sampling
python analysis_sampling.py --results_dir results/sampling
```

## GPU Support

- **Encoder**: Automatically uses GPU if available
- **FAISS Index**: Automatically uses GPU if available
- Falls back to CPU if no GPU detected

## Cache

- Cache is enabled by default
- Cached files stored in `cache/` folder
- Format: `{model_name}_index.faiss` and `{model_name}_passages.pkl`
- Use `--no_cache` to disable

## Examples

```bash
# Full benchmark on distractor dataset
python main.py --dataset_type distractor --max_samples 1000

# Sampling experiment
python benchmark_sampling.py --n 200 --p 0.3 --k 20

# Different model
python main.py --model_name BAAI/bge-large-en-v1.5 --max_samples 100
```

## Output

Results saved to:
- `results/`: Main benchmark results
- `results/sampling/`: Sampling experiment results
