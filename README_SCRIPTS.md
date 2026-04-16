# 📚 Retrieval Benchmark - Scripts Documentation

## 📋 Tổng Quan

Tôi đã phân tích kỹ codebase của bạn và tạo một bộ scripts hoàn chỉnh để chạy benchmark retrieval với 2 settings chính:

1. **Normal Retrieval** - Retrieval bình thường (top-k)
2. **Retrieval with Sampling** - Retrieval có sampling (random/diverse)

---

## 📦 Files Đã Tạo

### 1. **Shell Scripts** (Executable)

#### `run_retrieval_benchmark.sh` ⭐ RECOMMEND
- **Mục đích**: Chạy 2 settings cơ bản
- **Settings**:
  - Normal Retrieval (top_k)
  - Random Sampling
- **Chạy**: `./run_retrieval_benchmark.sh`

#### `run_all_settings.sh`
- **Mục đích**: Chạy tất cả 3 settings + tạo summary
- **Settings**:
  - Normal Retrieval (top_k)
  - Random Sampling
  - Diverse Sampling (stratified)
- **Chạy**: `./run_all_settings.sh`

### 2. **Python Scripts**

#### `compare_results.py`
- **Mục đích**: So sánh kết quả giữa các settings
- **Features**:
  - Compare metrics across strategies
  - Show best performing strategy
  - Detailed breakdown for all k values
- **Chạy**: `python compare_results.py --detailed`

### 3. **Documentation**

#### `QUICK_START.md`
- Quick reference guide
- Cách chạy nhanh
- Troubleshooting tips

#### `RUN_SCRIPTS_GUIDE.md`
- Hướng dẫn chi tiết
- Giải thích sampling strategies
- Configuration options
- Best practices

#### `README_SCRIPTS.md` (file này)
- Tổng quan tất cả scripts
- Architecture overview

---

## 🚀 Quick Start

### Bước 1: Test Setup
```bash
python test_setup.py
```

### Bước 2: Chạy Benchmark (Test Nhỏ Trước)

**Option A: Chạy 2 settings cơ bản (RECOMMEND)**
```bash
# Sửa trong script để test nhanh:
# MAX_SAMPLES="--max_samples 100"

./run_retrieval_benchmark.sh
```

**Option B: Chạy đầy đủ 3 settings**
```bash
./run_all_settings.sh
```

### Bước 3: So Sánh Kết Quả
```bash
python compare_results.py --detailed
```

---

## 🎯 Hiểu Về Sampling Strategies

### Strategy 1: Normal Retrieval (Baseline)
```
┌─────────┐     ┌────────┐     ┌───────────┐     ┌────────┐
│  Query  │ --> │ Encode │ --> │ Search    │ --> │ Top-K  │
└─────────┘     └────────┘     │ Index     │     └────────┘
                                └───────────┘
```
**Process**: Lấy trực tiếp K documents có score cao nhất

**Use Case**: Baseline để so sánh

---

### Strategy 2: Random Sampling
```
┌─────────┐     ┌────────┐     ┌───────────┐     ┌──────────┐
│  Query  │ --> │ Encode │ --> │ Search    │ --> │ Top-(K×4)│
└─────────┘     └────────┘     │ Index     │     └──────────┘
                                └───────────┘           │
                                                        ▼
                              ┌────────┐     ┌──────────────────┐
                              │ Top-K  │ <-- │ Random Sample 30%│
                              └────────┘     │ + Re-rank        │
                                             └──────────────────┘
```

**Process**:
1. Retrieve top (K × expansion_factor) documents → top (K × 4)
2. Random sample (30% × expanded_docs)
3. Re-rank sampled pool by scores
4. Select final top-K

**Parameters**:
- `expansion_factor=4`: Lấy top (K×4) documents ban đầu
- `random_ratio=0.3`: Random chọn 30% từ expanded pool

**Use Case**:
- Exploration vs exploitation
- Tăng diversity trong results
- Test robustness of ranking

---

### Strategy 3: Diverse Sampling
```
┌─────────┐     ┌────────┐     ┌───────────┐     ┌──────────┐
│  Query  │ --> │ Encode │ --> │ Search    │ --> │ Top-(K×4)│
└─────────┘     └────────┘     │ Index     │     └──────────┘
                                └───────────┘           │
                                                        ▼
                              ┌────────┐     ┌──────────────────┐
                              │ Top-K  │ <-- │ Stratified Sample│
                              └────────┘     │ (4 bins)         │
                                             │ + Re-rank        │
                                             └──────────────────┘
```

**Process**:
1. Retrieve top (K × expansion_factor) documents
2. Divide into 4 bins by score ranges:
   - Bin 1: High scores (top 25%)
   - Bin 2: Mid-high scores
   - Bin 3: Mid-low scores
   - Bin 4: Lower scores
3. Sample from each bin proportionally
4. Re-rank and select top-K

**Use Case**:
- Ensure coverage across score ranges
- Balance between precision and recall
- Useful for multi-hop reasoning

---

## 🔧 Configuration Guide

### Thay Đổi Model

Edit trong `.sh` files:
```bash
MODEL="bge-base"  # Thay đổi ở đây
```

**Available Models**:
| Short Name | Full Model Name | Size | Best For |
|-----------|----------------|------|----------|
| `bge-base` | BAAI/bge-base-en-v1.5 | ~400MB | General purpose |
| `bge-large` | BAAI/bge-large-en-v1.5 | ~1.3GB | High accuracy |
| `bge-small` | BAAI/bge-small-en-v1.5 | ~130MB | Fast inference |
| `e5-base` | intfloat/e5-base-v2 | ~400MB | Good baseline |
| `e5-large` | intfloat/e5-large-v2 | ~1.3GB | High performance |
| `gte-base` | thenlper/gte-base | ~400MB | Alternative |
| `gte-large` | thenlper/gte-large | ~1.3GB | Alternative |

### Sampling Parameters

```bash
EXPANSION_FACTOR=4      # Retrieve top (k × expansion_factor)
RANDOM_RATIO=0.3        # Ratio of random sampling
SAMPLING_SEED=42        # For reproducibility
```

**Tuning Guide**:
- `EXPANSION_FACTOR`:
  - Higher (8, 10): More diversity, but slower
  - Lower (2, 3): Faster, but less exploration
  - Default (4): Good balance

- `RANDOM_RATIO`:
  - Higher (0.5, 0.7): More randomness, more diversity
  - Lower (0.1, 0.2): More conservative, closer to top-k
  - Default (0.3): Balanced exploration

### GPU/CPU Settings

```bash
DEVICE="cuda"           # GPU
# DEVICE="cpu"          # CPU

BATCH_SIZE=32           # Adjust based on GPU memory
```

**Batch Size Guide**:
| GPU VRAM | Recommended Batch Size |
|----------|----------------------|
| 24GB+ | 128-256 |
| 16GB | 64-128 |
| 12GB | 32-64 |
| 8GB | 16-32 |
| CPU | 8-16 |

---

## 📊 Output Structure

### Directory Layout
```
retrieval-agents_v1/
├── logs/                           # Execution logs
│   ├── 01_normal_20260417_143022.log
│   ├── 02_random_sampling_20260417_143022.log
│   └── 03_diverse_sampling_20260417_143022.log
│
├── cache/
│   └── results/                    # JSON results
│       ├── BAAI_bge-base-en-v1.5_fullwiki_20260417_143022.json
│       └── ...
│
└── results_comparison/             # Summary files
    └── summary_20260417_143022.txt
```

### Result JSON Format
```json
{
  "model_name": "BAAI/bge-base-en-v1.5",
  "dataset_config": "fullwiki",
  "dataset_split": "test",
  "k_values": [1, 3, 5, 10, 20],
  "sampling_strategy": "random_sample",
  "expansion_factor": 4,
  "random_ratio": 0.3,
  "metrics": {
    "recall@1": 0.4234,
    "recall@5": 0.6789,
    "recall@10": 0.7891,
    "precision@5": 0.1358,
    "mrr": 0.5432,
    "f1@5": 0.2234
  },
  "timestamp": "20260417_143022"
}
```

---

## 📈 Analyzing Results

### View Comparison
```bash
# Basic comparison
python compare_results.py

# Detailed comparison (all k values)
python compare_results.py --detailed

# Compare specific pattern
python compare_results.py --pattern "bge-base*.json"
```

### Extract Specific Metrics
```bash
# Find best Recall@5
grep -h "recall@5" cache/results/*.json | sort -t: -k2 -nr | head -1

# Compare MRR across runs
grep -h "mrr" cache/results/*.json
```

### Manual Analysis
```bash
# Pretty print JSON
cat cache/results/<filename>.json | jq .

# Extract only metrics
cat cache/results/<filename>.json | jq .metrics

# Compare two specific files
diff <(cat file1.json | jq .metrics) <(cat file2.json | jq .metrics)
```

---

## 🎓 Best Practices

### 1. Testing Workflow
```bash
# Step 1: Always test with small sample first
MAX_SAMPLES="--max_samples 100"
./run_retrieval_benchmark.sh

# Step 2: Review results
python compare_results.py

# Step 3: If OK, run full dataset
MAX_SAMPLES=""
./run_all_settings.sh
```

### 2. Experiment Tracking
- Each run creates timestamped logs and results
- Use meaningful model names
- Keep notes about parameter changes
- Save successful configurations

### 3. Performance Optimization
- Use GPU when possible (`DEVICE="cuda"`)
- Tune batch size for your GPU
- Use `--max_samples` for quick iterations
- Monitor GPU memory with `nvidia-smi`

### 4. Reproducibility
- Set consistent `SAMPLING_SEED`
- Document parameter changes
- Save configuration snapshots
- Keep logs for debugging

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
BATCH_SIZE=8

# Solution 2: Use CPU
DEVICE="cpu"

# Solution 3: Use smaller model
MODEL="bge-small"
```

#### 2. Permission Denied
```bash
chmod +x *.sh
```

#### 3. Module Not Found
```bash
pip install -r requirements.txt
```

#### 4. Wikipedia Corpus Missing
```bash
python download_wikipedia.py --max_passages 100000
```

#### 5. Slow Execution
```bash
# Test with subset
MAX_SAMPLES="--max_samples 100"

# Use smaller model
MODEL="bge-small"

# Increase batch size (if GPU allows)
BATCH_SIZE=64
```

---

## 📚 Understanding the Codebase

### Architecture Overview

```
benchmark.py                    # Main benchmark pipeline
├── config.py                   # Configuration classes
├── data_loader.py              # HotpotQA data loading
├── retrieval_model.py          # Dense retriever (basic)
├── retrieval_model_optimized.py # Dense retriever (optimized)
├── indexer.py                  # FAISS indexing
├── sampling_strategies.py      # Sampling strategies
└── metrics.py                  # Evaluation metrics

Scripts Created:
├── run_retrieval_benchmark.sh  # Run 2 settings
├── run_all_settings.sh         # Run 3 settings + summary
└── compare_results.py          # Compare results

Documentation:
├── QUICK_START.md              # Quick reference
├── RUN_SCRIPTS_GUIDE.md        # Detailed guide
└── README_SCRIPTS.md           # This file
```

### Key Components

**RetrievalBenchmark** (benchmark.py:29)
- Main pipeline orchestrator
- Handles: setup → load data → build corpus → build index → retrieve → evaluate

**RetrievalSampler** (sampling_strategies.py:18)
- Implements 3 sampling strategies
- Applied after initial retrieval
- Re-ranks and selects final top-K

**DenseRetrieverOptimized** (retrieval_model_optimized.py:53)
- Optimized encoder with GPU acceleration
- Multi-processing support
- Batch encoding for efficiency

---

## 🎯 Example Workflows

### Workflow 1: Quick Experiment
```bash
# 1. Test with 100 samples
./run_retrieval_benchmark.sh  # Already configured for small test

# 2. Compare
python compare_results.py

# 3. If good, run full
# Edit script: MAX_SAMPLES=""
./run_retrieval_benchmark.sh
```

### Workflow 2: Model Comparison
```bash
# Create a loop script
for model in "bge-base" "e5-base" "gte-base"; do
    sed -i '' "s/MODEL=\".*\"/MODEL=\"$model\"/" run_retrieval_benchmark.sh
    ./run_retrieval_benchmark.sh
done

# Compare all
python compare_results.py --detailed
```

### Workflow 3: Hyperparameter Tuning
```bash
# Test different expansion factors
for exp in 2 4 6 8; do
    sed -i '' "s/EXPANSION_FACTOR=.*/EXPANSION_FACTOR=$exp/" run_all_settings.sh
    ./run_all_settings.sh
done

# Analyze
python compare_results.py
```

---

## 📞 Support

### Debug Steps
1. ✅ Verify setup: `python test_setup.py`
2. ✅ Check GPU: `nvidia-smi`
3. ✅ Review logs: `tail -f logs/*.log`
4. ✅ Test small sample: `MAX_SAMPLES="--max_samples 10"`

### Files to Check
- **Logs**: `logs/` directory
- **Results**: `cache/results/` directory
- **Config**: Scripts parameters at top of `.sh` files

---

## 📖 Additional Resources

### Related Files in Codebase
- `benchmark.py` - Main benchmark script (line 347: argparse configs)
- `sampling_strategies.py` - Sampling implementation (line 18: RetrievalSampler)
- `config.py` - BenchmarkConfig dataclass (line 8)

### Key Configuration Points
- Sampling strategy: `benchmark.py:371-379` (argparse)
- Model loading: `benchmark.py:59-68` (setup method)
- Sampling application: `benchmark.py:228-248` (retrieve method)

---

**Tạo bởi**: Automated Script Generation
**Ngày**: 2026-04-17
**Version**: 1.0

Chúc bạn chạy benchmark thành công! 🚀
