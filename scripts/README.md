# Benchmark Scripts

Individual scripts to run benchmarks for each model.

## Quick Start (Debug Mode - 100 samples)

```bash
# BGE models
bash scripts/run_bge_base.sh
bash scripts/run_bge_large.sh

# E5 models
bash scripts/run_e5_base.sh
bash scripts/run_e5_large.sh
```

## Full Benchmark

Edit each script and uncomment the full benchmark command, or run directly:

```bash
# BGE-base full benchmark
python main.py \
    --model_name "BAAI/bge-base-en-v1.5" \
    --dataset_type "fullwiki" \
    --split "validation"

# BGE-large full benchmark
python main.py \
    --model_name "BAAI/bge-large-en-v1.5" \
    --dataset_type "fullwiki" \
    --split "validation"

# E5-base full benchmark
python main.py \
    --model_name "intfloat/e5-base-v2" \
    --dataset_type "fullwiki" \
    --split "validation"

# E5-large full benchmark
python main.py \
    --model_name "intfloat/e5-large-v2" \
    --dataset_type "fullwiki" \
    --split "validation"
```

## Customization

Each script can be modified to change:
- `--dataset_type`: `distractor` (faster) or `fullwiki` (complete)
- `--split`: `validation` or `train`
- `--max_samples`: Number of samples (remove for all)
- `--k_values`: List of k values to evaluate
- `--device`: `cuda` or `cpu`

## Example: Custom Parameters

```bash
python main.py \
    --model_name "BAAI/bge-base-en-v1.5" \
    --dataset_type "fullwiki" \
    --split "validation" \
    --max_samples 1000 \
    --k_values 1 5 10 20 50 \
    --device cuda
```
