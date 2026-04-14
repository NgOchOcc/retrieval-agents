# HotpotQA Retrieval Benchmark

A comprehensive pipeline for benchmarking dense retrieval models (BGE, E5, GTE, etc.) on the HotpotQA dataset with paragraph-level retrieval evaluation.

## Features

- **Multiple Model Support**: BGE, E5, GTE, and any HuggingFace sentence-transformer compatible model
- **Full Wikipedia Retrieval**: Evaluate on the complete Wikipedia corpus (fullwiki mode)
- **Efficient Indexing**: FAISS-based indexing for fast similarity search
- **Comprehensive Metrics**: Pass@k, Recall@k, Precision@k, NDCG@k, and MRR
- **Caching**: Smart caching of embeddings and indices for faster re-runs

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd retrieval-agents

# Install dependencies
pip install -r requirements.txt

# Optional: Install FAISS with GPU support
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Quick Start

### Basic Usage

```bash
# Benchmark BGE-base model on HotpotQA test set
python benchmark.py --model bge-base --batch_size 32

# Use E5-large model
python benchmark.py --model e5-large --batch_size 64

# Test on a smaller subset (useful for debugging)
python benchmark.py --model bge-base --max_samples 100
```

### Command Line Arguments

```bash
python benchmark.py --help

Options:
  --model MODEL              Model shorthand or HuggingFace model name
                            Supported: bge-base, bge-large, bge-small,
                                      e5-base, e5-large, e5-small,
                                      gte-base, gte-large
  --batch_size SIZE          Batch size for encoding (default: 32)
  --max_samples N            Limit number of samples for testing
  --device DEVICE            Device: cuda or cpu (default: cuda)
  --cache_dir DIR            Cache directory (default: ./cache)
  --dataset_split SPLIT      Dataset split (default: test)
  --dataset_config CONFIG    Dataset config (default: fullwiki)
  --no_faiss                 Use simple numpy retriever instead of FAISS
```

### Example Commands

```bash
# Benchmark multiple models
python benchmark.py --model bge-base --batch_size 32
python benchmark.py --model bge-large --batch_size 64
python benchmark.py --model e5-base --batch_size 32
python benchmark.py --model e5-large --batch_size 64

# Use custom HuggingFace model
python benchmark.py --model "sentence-transformers/all-MiniLM-L6-v2" --batch_size 128

# Run on CPU
python benchmark.py --model bge-base --device cpu --batch_size 16

# Quick test with 500 samples
python benchmark.py --model bge-base --max_samples 500
```

## Pipeline Overview

The benchmark pipeline consists of 5 main steps:

1. **Load Dataset**: Downloads and processes HotpotQA from HuggingFace
2. **Build Corpus**: Extracts paragraphs from Wikipedia or example contexts
3. **Build Index**: Encodes corpus and creates FAISS index
4. **Retrieve**: Encodes queries and retrieves top-k documents
5. **Evaluate**: Computes metrics (Pass@k, NDCG@k, etc.)

## Project Structure

```
retrieval-agents/
├── benchmark.py          # Main benchmark script
├── config.py            # Configuration and supported models
├── data_loader.py       # HotpotQA data loading and preprocessing
├── retrieval_model.py   # Dense retrieval model wrapper
├── indexer.py           # FAISS indexing and search
├── metrics.py           # Evaluation metrics
├── requirements.txt     # Python dependencies
└── cache/              # Cached data (created automatically)
    ├── index/          # FAISS indices and embeddings
    ├── results/        # Evaluation results (JSON)
    └── wikipedia_paragraphs.json  # Cached Wikipedia corpus
```

## Metrics Explained

### Pass@k (Success@k)
Binary metric: 1 if at least one relevant document is in top-k, 0 otherwise.
- **Use case**: Measuring if the system can find any relevant document
- **Range**: [0, 1]

### Recall@k
Fraction of relevant documents found in top-k.
- **Formula**: (# relevant docs in top-k) / (# total relevant docs)
- **Use case**: Measuring coverage of relevant documents
- **Range**: [0, 1]

### Precision@k
Fraction of retrieved documents that are relevant.
- **Formula**: (# relevant docs in top-k) / k
- **Use case**: Measuring retrieval accuracy
- **Range**: [0, 1]

### NDCG@k
Normalized Discounted Cumulative Gain - accounts for ranking quality.
- **Use case**: Measuring both relevance and ranking position
- **Range**: [0, 1]

### MRR (Mean Reciprocal Rank)
Reciprocal of the rank of the first relevant document.
- **Use case**: Measuring how early the first relevant document appears
- **Range**: [0, 1]

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class BenchmarkConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 32
    max_length: int = 512
    k_values: List[int] = [1, 3, 5, 10, 20]
    use_faiss: bool = True
    normalize_embeddings: bool = True
    # ... more options
```

## Adding New Models

To add a new model, update `config.py`:

```python
SUPPORTED_MODELS = {
    "your-model": "huggingface/model-name",
    # ... existing models
}
```

Then run:
```bash
python benchmark.py --model your-model
```

## Results

Results are automatically saved to `cache/results/` as JSON files:

```json
{
  "model_name": "BAAI/bge-base-en-v1.5",
  "dataset_config": "fullwiki",
  "dataset_split": "test",
  "metrics": {
    "pass@1": 0.6543,
    "pass@5": 0.8234,
    "ndcg@10": 0.7456,
    "mrr": 0.7123
  },
  "timestamp": "20240414_143022"
}
```

## Performance Tips

1. **Use GPU**: Encoding is much faster on GPU
   ```bash
   python benchmark.py --model bge-base --device cuda
   ```

2. **Adjust Batch Size**: Increase for faster encoding (if you have enough memory)
   ```bash
   python benchmark.py --model bge-base --batch_size 128
   ```

3. **Cache Embeddings**: First run creates cache, subsequent runs are much faster

4. **Use FAISS**: For large corpora, FAISS is essential
   - For small tests, use `--no_faiss` for simpler setup

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Use smaller model (e.g., `bge-small` instead of `bge-large`)
- Use CPU: `--device cpu`

### Slow Wikipedia Download
- First run downloads ~20GB Wikipedia dump
- Subsequent runs use cached version at `cache/wikipedia_paragraphs.json`

### FAISS GPU Error
- Install `faiss-gpu` or use `--no_faiss` for CPU-only mode

## Citation

If you use this benchmark, please cite HotpotQA:

```bibtex
@inproceedings{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W and Salakhutdinov, Ruslan and Manning, Christopher D},
  booktitle={EMNLP},
  year={2018}
}
```

## License

MIT License
