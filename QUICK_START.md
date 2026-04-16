# Quick Start Guide - Retrieval Benchmark

## 🚀 Chạy Ngay (Quick Run)

### Option 1: Chạy 2 Settings Cơ Bản
```bash
./run_retrieval_benchmark.sh
```
**Chạy:**
- ✅ Normal Retrieval (no sampling)
- ✅ Random Sampling

### Option 2: Chạy Đầy Đủ 3 Settings
```bash
./run_all_settings.sh
```
**Chạy:**
- ✅ Normal Retrieval
- ✅ Random Sampling
- ✅ Diverse Sampling

---

## 📝 Cấu Hình Nhanh

### Thay Đổi Model
Mở file `.sh` và sửa:
```bash
MODEL="bge-base"  # bge-base, bge-large, e5-base, e5-large, gte-base
```

### Test Với Subset Nhỏ (Recommend!)
```bash
MAX_SAMPLES="--max_samples 100"
```

### Chạy CPU (Không GPU)
```bash
DEVICE="cpu"
```

---

## 📊 Kết Quả

### Logs
```
logs/
├── 01_normal_<timestamp>.log
├── 02_random_sampling_<timestamp>.log
└── 03_diverse_sampling_<timestamp>.log
```

### Results (JSON)
```
cache/results/
└── <model_name>_fullwiki_<timestamp>.json
```

### Xem Kết Quả
```bash
# View JSON
cat cache/results/*.json | jq .metrics

# Extract Recall@5
grep "recall@5" cache/results/*.json
```

---

## 🔧 Troubleshooting

| Vấn Đề | Giải Pháp |
|--------|-----------|
| CUDA OOM | Giảm `BATCH_SIZE=8` |
| Permission Denied | `chmod +x *.sh` |
| Module Not Found | `pip install -r requirements.txt` |
| Slow | Dùng `--max_samples 100` để test |

---

## 💡 So Sánh Settings

| Setting | Cách Hoạt Động | Use Case |
|---------|----------------|----------|
| **Normal** | Top-K trực tiếp | Baseline |
| **Random** | Expand × Random × Re-rank | Exploration |
| **Diverse** | Stratified sampling | Coverage |

---

## ⚙️ Parameters Chính

```bash
# Sampling Parameters
EXPANSION_FACTOR=4    # Retrieve top (k×4) first
RANDOM_RATIO=0.3      # 30% random selection
SAMPLING_SEED=42      # Reproducibility

# Model Parameters
BATCH_SIZE=32         # GPU memory dependent
DEVICE="cuda"         # cuda or cpu
```

---

## 📈 Metrics

Mỗi run outputs:
- **Recall@K**: % relevant docs found
- **Precision@K**: % retrieved docs relevant
- **F1@K**: Harmonic mean
- **MRR**: Mean Reciprocal Rank

---

**Need Help?** Check `RUN_SCRIPTS_GUIDE.md` for detailed documentation.
