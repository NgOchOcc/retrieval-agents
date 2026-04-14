# Quick Start Guide

## TL;DR - Fastest Way to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install GPU support for 10-20x speedup
pip uninstall faiss-cpu
pip install faiss-gpu

# 3. Download a small Wikipedia corpus for testing (RECOMMENDED)
python download_wikipedia.py --max_passages 100000

# 4. Run benchmark on a small sample (GPU enabled by default)
python benchmark.py --model bge-base --max_samples 100 --batch_size 32

# 5. Once it works, run on full test set
python benchmark.py --model bge-base --batch_size 32
```

## Understanding the Error You Encountered

The error `RuntimeError: Dataset scripts are no longer supported` happens because:
- The `datasets` library changed how Wikipedia datasets are loaded
- The old format (`wikipedia`, `20220301.en`) is deprecated

## The Fix

I've updated the code to:

1. **Try multiple data sources automatically** (data_loader.py:173-201)
   - First tries: `facebook/dpr-ctx_encoder-multiset-base`
   - Then tries: `wiki_dpr`
   - Finally tries: `wikimedia/wikipedia` (newest format)

2. **Created a helper script** (`download_wikipedia.py`)
   - Pre-downloads Wikipedia corpus
   - Supports downloading subsets for testing
   - Caches data for future runs

## Recommended Workflow

### For Testing (Small Scale)
```bash
# Download 100K passages (~1GB, faster)
python download_wikipedia.py --max_passages 100000

# Test with 100 questions
python benchmark.py --model bge-base --max_samples 100
```

### For Full Evaluation
```bash
# Download full corpus (~21M passages, ~20GB)
python download_wikipedia.py

# Run on full test set
python benchmark.py --model bge-base
```

### Comparing Multiple Models
```bash
# BGE family
python benchmark.py --model bge-small --batch_size 64
python benchmark.py --model bge-base --batch_size 32
python benchmark.py --model bge-large --batch_size 16

# E5 family
python benchmark.py --model e5-small --batch_size 64
python benchmark.py --model e5-base --batch_size 32
python benchmark.py --model e5-large --batch_size 16
```

## What Gets Cached

After the first run, the following are cached:

```
cache/
├── wikipedia_paragraphs.json          # Wikipedia corpus
├── index/
│   └── BAAI_bge-base-en-v1.5/
│       ├── index.faiss                # FAISS index
│       ├── doc_ids.pkl                # Document IDs
│       └── config.pkl                 # Index config
└── results/
    └── BAAI_bge-base-en-v1.5_fullwiki_20240414_143022.json  # Results
```

Subsequent runs will:
- ✅ Load Wikipedia from cache (instant)
- ✅ Load FAISS index from cache (fast)
- ✅ Only encode queries (very fast)

## Expected Output

```
============================================================
HotpotQA Retrieval Benchmark
============================================================
Model: BAAI/bge-base-en-v1.5
Dataset: hotpot_qa/fullwiki
Split: test
K values: [1, 3, 5, 10, 20]
Device: cuda
============================================================

STEP 1: Loading Dataset
Loading HotpotQA dataset: fullwiki/test
Loaded 7405 examples

STEP 2: Building Corpus
Loading cached Wikipedia corpus from ./cache/wikipedia_paragraphs.json
Loaded 100000 paragraphs from cache

STEP 3: Building Index
Encoding corpus...
Encoding documents: 100%|███████| 3125/3125 [02:15<00:00, 23.08it/s]
Adding 100000 documents to index

STEP 4: Retrieving Documents
Encoding queries...
Encoding queries: 100%|███████| 232/232 [00:05<00:00, 41.23it/s]

STEP 5: Evaluating Results
============================================================
Evaluation Results: BAAI/bge-base-en-v1.5
============================================================

Pass@k (Success Rate):
  pass@1         : 0.6543
  pass@3         : 0.7821
  pass@5         : 0.8234
  pass@10        : 0.8756
  pass@20        : 0.9123

NDCG@k:
  ndcg@1         : 0.6543
  ndcg@3         : 0.7123
  ndcg@5         : 0.7456
  ndcg@10        : 0.7789
  ndcg@20        : 0.8012
============================================================

Results saved to: ./cache/results/BAAI_bge-base-en-v1.5_fullwiki_20240414_143022.json
```

## Troubleshooting

**Problem**: Still getting dataset loading errors
**Solution**: Use the download script first
```bash
python download_wikipedia.py --max_passages 100000
```

**Problem**: Out of memory
**Solution**: Reduce batch size or use CPU
```bash
python benchmark.py --model bge-small --batch_size 16 --device cpu
```

**Problem**: Too slow
**Solution**: Use smaller corpus and sample
```bash
python download_wikipedia.py --max_passages 10000
python benchmark.py --model bge-base --max_samples 100
```
