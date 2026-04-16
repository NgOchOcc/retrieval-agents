# Hướng Dẫn Chạy Retrieval Benchmark

## Tổng Quan

Tôi đã tạo 2 script bash để chạy benchmark retrieval:

1. **`run_retrieval_benchmark.sh`** - Chạy 2 settings cơ bản:
   - Normal Retrieval (top_k)
   - Retrieval with Sampling (random_sample)

2. **`run_all_settings.sh`** - Chạy tất cả 3 settings:
   - Normal Retrieval (top_k)
   - Random Sampling
   - Diverse Sampling (stratified)

## Cách Sử Dụng

### Script 1: Chạy 2 Settings Cơ Bản

```bash
# Chạy trực tiếp
./run_retrieval_benchmark.sh

# Hoặc
bash run_retrieval_benchmark.sh
```

**Chức năng:**
- ✅ Chạy normal retrieval (không sampling)
- ✅ Chạy retrieval với random sampling
- ✅ Lưu kết quả và logs riêng biệt
- ✅ Tự động tạo timestamp cho mỗi lần chạy

### Script 2: Chạy Tất Cả Settings (Đầy Đủ)

```bash
# Chạy trực tiếp
./run_all_settings.sh

# Hoặc
bash run_all_settings.sh
```

**Chức năng:**
- ✅ Chạy normal retrieval
- ✅ Chạy random sampling
- ✅ Chạy diverse sampling (stratified)
- ✅ Tạo file summary so sánh
- ✅ Lưu logs và kết quả chi tiết

## Cấu Hình Scripts

### Thay Đổi Model

Mở file `.sh` và sửa dòng:
```bash
MODEL="bge-base"  # Thay đổi model ở đây
```

**Models hỗ trợ:**
- `bge-base` - BAAI/bge-base-en-v1.5
- `bge-large` - BAAI/bge-large-en-v1.5
- `bge-small` - BAAI/bge-small-en-v1.5
- `e5-base` - intfloat/e5-base-v2
- `e5-large` - intfloat/e5-large-v2
- `gte-base` - thenlper/gte-base
- `gte-large` - thenlper/gte-large

### Thay Đổi Batch Size

```bash
BATCH_SIZE=32  # Tăng hoặc giảm tùy GPU memory
```

### Thay Đổi Device

```bash
DEVICE="cuda"  # Hoặc "cpu" nếu không có GPU
```

### Test Với Subset Nhỏ

```bash
MAX_SAMPLES="--max_samples 100"  # Chỉ chạy 100 samples để test
```

### Thay Đổi Sampling Parameters

```bash
EXPANSION_FACTOR=4      # Retrieve top (k * 4) documents
RANDOM_RATIO=0.3        # 30% random sampling
SAMPLING_SEED=42        # Random seed
```

## Cấu Trúc Output

### Logs
```
logs/
├── 01_normal_20260417_143022.log
├── 02_random_sampling_20260417_143022.log
└── 03_diverse_sampling_20260417_143022.log
```

### Results
```
cache/results/
├── BAAI_bge-base-en-v1.5_fullwiki_20260417_143022.json
├── BAAI_bge-base-en-v1.5_fullwiki_20260417_145630.json
└── ...
```

### Summary (từ run_all_settings.sh)
```
results_comparison/
└── summary_20260417_143022.txt
```

## Hiểu Về Sampling Strategies

### 1. Normal Retrieval (top_k)
```
Query → Encode → Search Index → Top-K Documents
```
- Lấy trực tiếp K documents có score cao nhất
- Baseline standard cho so sánh

### 2. Random Sampling (random_sample)
```
Query → Encode → Search Index → Top-(K×4) → Random Sample 30% → Re-rank → Top-K
```
- Expand: Lấy top (K × expansion_factor) documents
- Sample: Random chọn (30% × expanded_docs)
- Re-rank: Sắp xếp lại theo score và lấy top-K

**Parameters:**
- `expansion_factor=4`: Lấy top (K×4) trước
- `random_ratio=0.3`: Random chọn 30% từ expanded pool

### 3. Diverse Sampling (diverse_sample)
```
Query → Encode → Search Index → Top-(K×4) → Stratified Sample → Re-rank → Top-K
```
- Expand: Lấy top (K × expansion_factor) documents
- Stratify: Chia thành 4 bins theo score ranges
- Sample: Lấy từng bin để đảm bảo diversity
- Re-rank: Sắp xếp và lấy top-K

## Ví Dụ Chạy

### Ví Dụ 1: Test Nhanh Với 100 Samples

```bash
# Sửa trong script:
MAX_SAMPLES="--max_samples 100"
BATCH_SIZE=64

# Chạy:
./run_retrieval_benchmark.sh
```

### Ví Dụ 2: Chạy Full Dataset Với BGE-Large

```bash
# Sửa trong script:
MODEL="bge-large"
BATCH_SIZE=16  # Giảm batch size cho model lớn
MAX_SAMPLES=""  # Full dataset

# Chạy:
./run_all_settings.sh
```

### Ví Dụ 3: So Sánh Nhiều Models

Tạo script mới `compare_models.sh`:
```bash
#!/bin/bash

for model in "bge-base" "e5-base" "gte-base"; do
    echo "Running model: $model"
    sed -i "s/MODEL=\".*\"/MODEL=\"$model\"/" run_retrieval_benchmark.sh
    ./run_retrieval_benchmark.sh
done
```

## Xem Kết Quả

### 1. Xem Logs Real-time
```bash
tail -f logs/01_normal_*.log
```

### 2. Xem Kết Quả JSON
```bash
# List results
ls -lh cache/results/

# View a result
cat cache/results/BAAI_bge-base-en-v1.5_fullwiki_*.json | jq .
```

### 3. Extract Metrics
```bash
# Extract recall@5 from all results
grep -r "recall@5" cache/results/*.json
```

## Troubleshooting

### Lỗi: CUDA Out of Memory
```bash
# Giảm batch size
BATCH_SIZE=8

# Hoặc dùng CPU
DEVICE="cpu"
```

### Lỗi: Permission Denied
```bash
chmod +x run_retrieval_benchmark.sh
chmod +x run_all_settings.sh
```

### Lỗi: Module Not Found
```bash
pip install -r requirements.txt
```

### Lỗi: Wikipedia Corpus Not Cached
```bash
python download_wikipedia.py --max_passages 100000
```

## Metrics Được Đánh Giá

Mỗi run sẽ output các metrics:
- **Recall@K** (K=1,3,5,10,20): Tỷ lệ tìm thấy relevant documents
- **Precision@K**: Độ chính xác
- **F1@K**: Harmonic mean của precision và recall
- **MRR (Mean Reciprocal Rank)**: Vị trí trung bình của relevant doc đầu tiên

## Best Practices

1. **Luôn test với small dataset trước** (`--max_samples 100`)
2. **Monitor GPU memory** khi chạy models lớn
3. **Lưu logs** để debug khi có lỗi
4. **So sánh kết quả** giữa các strategies
5. **Use timestamp** để track experiments

## Support

Nếu có lỗi:
1. Check logs trong folder `logs/`
2. Verify setup: `python test_setup.py`
3. Check GPU memory: `nvidia-smi`
4. Review configuration trong scripts

---

**Created:** 2026-04-17
**Author:** Automated Setup for Retrieval Benchmark
