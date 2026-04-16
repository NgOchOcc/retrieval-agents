# HotpotQA Dense Retrieval Benchmark

Clean, modular Python project for benchmarking dense retrieval models on multi-hop question answering using the HotpotQA dataset.

## Overview

This project evaluates **BGE** and **E5** embedding models on their ability to retrieve gold supporting documents for multi-hop questions from HotpotQA.

**Key Features:**
- Proper instruction formatting for BGE and E5 models
- FAISS-based dense retrieval with cosine similarity
- Strict (pass@k) and relaxed (hit@k) multi-hop metrics
- Reproducible experiments with seed control
- Modular, type-hinted, documented code

## Supported Models

- `BAAI/bge-base-en-v1.5`
- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-base-v2`
- `intfloat/e5-large-v2`

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### GPU Setup (Recommended for Speed)

For **significant speedup** (10-50x faster), use GPU:

1. **Install PyTorch with CUDA:**
```bash
# Check CUDA version first: nvidia-smi
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

2. **Install FAISS with GPU support:**
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

3. **Test GPU:**
```bash
python scripts/test_gpu.py
```

You should see:
```
GPU Available: True
Device Name: NVIDIA GeForce RTX 3090
Loaded model on cuda:0
```

**Note:** Batch sizes are optimized for GPU (passage=256, query=64). Reduce if you encounter OOM errors.

## Quick Start

### Basic Usage

```bash
python main.py --model_name BAAI/bge-base-en-v1.5
```

### Fast Debugging (Subsample 100 examples)

```bash
python main.py \
    --model_name BAAI/bge-base-en-v1.5 \
    --max_samples 100 \
    --dataset_type distractor
```

### Full Benchmark

```bash
python main.py \
    --model_name BAAI/bge-large-en-v1.5 \
    --dataset_type fullwiki \
    --split validation
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `BAAI/bge-base-en-v1.5` | HuggingFace model identifier |
| `--device` | `auto` | Device (`cuda` or `cpu`) |
| `--split` | `validation` | Dataset split (`train` or `validation`) |
| `--dataset_type` | `fullwiki` | HotpotQA type (`fullwiki` or `distractor`) |
| `--max_samples` | `None` | Limit samples for debugging |
| `--max_k` | `20` | Maximum k for retrieval |
| `--k_values` | `[1, 5, 10, 20]` | K values to evaluate |
| `--passage_batch_size` | `128` | Batch size for encoding passages |
| `--query_batch_size` | `32` | Batch size for encoding queries |
| `--seed` | `42` | Random seed |
| `--output_dir` | `results` | Output directory |
| `--no_save` | `False` | Don't save results |

## Example Output

```
============================================================
HOTPOTQA RETRIEVAL BENCHMARK
============================================================
Model: BAAI/bge-base-en-v1.5
Dataset: fullwiki (validation)
Device: cuda
Seed: 42
============================================================

[1/5] Loading HotpotQA dataset...
  → Loaded 7405 examples
  → Built corpus with 5233329 passages

[2/5] Loading encoder model...
Loaded BAAI/bge-base-en-v1.5 on cuda

[3/5] Building FAISS index...
Building index for 5233329 passages...
Index built with 5233329 vectors

[4/5] Retrieving passages for queries...

[5/5] Evaluating retrieval results...

========================================
Retrieval Evaluation Results
========================================

STRICT (pass@k) - Both docs in top-k:
----------------------------------------
  pass@1      : 0.1234
  pass@5      : 0.3456
  pass@10     : 0.4567
  pass@20     : 0.5678

RELAXED (hit@k) - At least one doc in top-k:
----------------------------------------
  hit@1       : 0.4321
  hit@5       : 0.7654
  hit@10      : 0.8432
  hit@20      : 0.9012
========================================

Results saved to: results/BAAI_bge-base-en-v1.5_fullwiki_20260416_120000.json

============================================================
BENCHMARK COMPLETED
============================================================
```

## Project Structure

```
retrieval_agents/
├── data/
│   ├── __init__.py
│   └── hotpotqa_loader.py      # Dataset loading and corpus building
├── models/
│   ├── __init__.py
│   └── encoder.py              # Model wrappers with instruction formatting
├── retrieval/
│   ├── __init__.py
│   └── faiss_retriever.py      # FAISS-based dense retrieval
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # pass@k and hit@k metrics
├── scripts/                    # Optional utility scripts
├── config.py                   # Configuration management
├── main.py                     # Main evaluation pipeline
├── requirements.txt
└── README.md
```

## Metrics

### pass@k (STRICT)
Returns 1 if **BOTH** supporting documents are in top-k, 0 otherwise.
This is the primary metric for multi-hop retrieval.

### hit@k (RELAXED)
Returns 1 if **AT LEAST ONE** supporting document is in top-k, 0 otherwise.
This measures first-hop retrieval performance.

## Extending the Project

### Add New Models

1. Implement instruction formatting in `models/encoder.py`
2. Update `get_encoder()` factory function
3. Add to `SUPPORTED_MODELS` list

### Add New Metrics

1. Implement metric function in `evaluation/metrics.py`
2. Update `RetrievalEvaluator.evaluate()` method
3. Update `format_results()` for display

### Two-Step Retrieval (Optional)

To implement iterative retrieval:
1. Create `retrieval/iterative_retriever.py`
2. Retrieve first hop documents
3. Use their content to refine query
4. Retrieve second hop documents

## Notes

- **Instruction Formatting**: BGE and E5 require different query/passage prefixes. This is handled automatically.
- **Normalization**: All embeddings are L2-normalized for cosine similarity via inner product.
- **Memory**: For large corpora, embeddings are not stored after indexing. Set `save_embeddings=True` in `FAISSRetriever.build_index()` if needed.
- **GPU**: Automatically uses CUDA if available. Force CPU with `--device cpu`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William and Salakhutdinov, Ruslan and Manning, Christopher D},
  booktitle={EMNLP},
  year={2018}
}
```

## License

MIT
