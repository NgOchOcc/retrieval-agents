# GPU Setup Guide

## Why GPU Matters

**Speed comparison for encoding 10,000 passages:**
- CPU (Intel i9): ~15-30 minutes
- GPU (RTX 3090): ~30-60 seconds

**That's 15-30x faster!**

---

## Quick GPU Check

```bash
# Test if GPU is available
python scripts/test_gpu.py
```

**Expected output if GPU working:**
```
============================================================
GPU AVAILABILITY TEST
============================================================

CUDA Available: True
CUDA Version: 11.8
Device Count: 1
Current Device: 0
Device Name: NVIDIA GeForce RTX 3090
Total Memory: 24.00 GB

============================================================
LOADING MODEL
============================================================

Loading BAAI/bge-base-en-v1.5...
Loaded BAAI/bge-base-en-v1.5 on cuda:0
GPU Available: NVIDIA GeForce RTX 3090
GPU Memory: 24.00 GB

============================================================
ENCODING SPEED TEST
============================================================

Encoding 100 texts...
Time: 0.45 seconds
Throughput: 222.22 texts/second
```

---

## Installation Steps

### 1. Check CUDA Version

```bash
nvidia-smi
```

Look for CUDA Version in top right corner (e.g., 11.8 or 12.1)

### 2. Install PyTorch with CUDA

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```python
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### 3. Install FAISS-GPU

```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

**Note:** If you have issues with faiss-gpu, you can use faiss-cpu (slower but works).

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Problem: "CUDA Available: False"

**Solution 1: Check PyTorch installation**
```python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

If cuda is None, reinstall PyTorch with CUDA support.

**Solution 2: Check NVIDIA driver**
```bash
nvidia-smi
```

If this fails, install/update NVIDIA drivers.

**Solution 3: CUDA toolkit mismatch**
Make sure your PyTorch CUDA version matches your driver's CUDA version.

### Problem: "CUDA out of memory"

**Solution: Reduce batch size**
```bash
python main.py \
    --passage_batch_size 64 \
    --query_batch_size 16
```

### Problem: "RuntimeError: CUDA error: no kernel image is available"

**Solution:** Your GPU architecture is not supported. Try:
1. Update PyTorch to latest version
2. Check GPU compute capability (must be ≥3.5)

---

## Optimal Batch Sizes

The system **auto-detects** your GPU memory and suggests batch sizes.

### Manual override:

**For 24GB GPU (RTX 3090, 4090):**
```bash
python main.py \
    --passage_batch_size 512 \
    --query_batch_size 128
```

**For 16GB GPU (RTX 4080):**
```bash
python main.py \
    --passage_batch_size 384 \
    --query_batch_size 96
```

**For 12GB GPU (RTX 3080, 4070):**
```bash
python main.py \
    --passage_batch_size 256 \
    --query_batch_size 64
```

**For 8GB GPU (RTX 3070, 4060):**
```bash
python main.py \
    --passage_batch_size 128 \
    --query_batch_size 32
```

---

## Performance Tips

### 1. Use Mixed Precision (for newer GPUs)

The models automatically use float32. For A100/V100, you can enable fp16 in sentence-transformers for more speedup.

### 2. Monitor GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- GPU utilization: 80-100%
- Memory usage: Increasing during encoding

### 3. Large Models Need More Memory

- **base models** (768 dim): Can use larger batches
- **large models** (1024 dim): Need smaller batches

The system automatically adjusts, but you can override.

---

## CPU Fallback

If you don't have GPU, the code still works (just slower):

```bash
python main.py \
    --device cpu \
    --passage_batch_size 32 \
    --query_batch_size 16 \
    --max_samples 100  # Limit samples for testing
```

**Recommendation:** Start with small `--max_samples` to test before running full dataset.

---

## Verification Checklist

✅ `nvidia-smi` works
✅ `torch.cuda.is_available()` returns True
✅ `python scripts/test_gpu.py` shows GPU info
✅ Model loads on "cuda:0"
✅ Encoding speed > 100 texts/second

If all checked, you're good to go! 🚀
