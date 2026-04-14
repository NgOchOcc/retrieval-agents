# Debug: Empty Metrics Issue

## Vấn Đề

Results file chỉ có `"metrics": {}` - không có điểm số nào.

## Nguyên Nhân Có Thể

1. **Question IDs không khớp** giữa retrieval results và ground truth
2. **Không tìm thấy supporting facts** trong corpus
3. **Corpus sai** - dùng fullwiki nhưng chỉ có distractor paragraphs
4. **Ground truth rỗng** - không có relevant docs

## Debug Steps

### Step 1: Run Debug Script

```bash
python debug_evaluation.py
```

**Check output:**
- ✓ Examples loaded
- ✓ Paragraphs extracted
- ✓ Ground truth created
- ✓ Supporting facts mapped

### Step 2: Run Benchmark với Debug Logging

Code đã được update với debug logging. Chạy lại:

```bash
python benchmark.py \
    --model bge-base \
    --batch_size 128 \
    --num_workers 4 \
    --max_samples 100
```

**Look for:**
```
STEP 5: Evaluating Results
Building ground truth labels...
Number of questions: 100
Number of ground truth entries: 100

Sample ground truth for question <id>:
  Relevant docs: X
  Retrieved docs: 20
  First 5 retrieved: [...]

Evaluation Statistics:
  Total questions: 100
  Evaluated: ???  ← Should be > 0
  Skipped (no ground truth): ???
  Skipped (no relevant docs): ???
```

## Possible Issues & Fixes

### Issue 1: No Supporting Facts in Corpus

**Symptom:**
```
Skipped (no relevant docs): 100
Evaluated: 0
```

**Cause:** Dùng fullwiki mode nhưng corpus không có supporting facts

**Fix:**
```bash
# Option 1: Use distractor mode (có sẵn supporting facts)
python benchmark.py \
    --model bge-base \
    --dataset_config distractor \
    --max_samples 100

# Option 2: Ensure Wikipedia corpus is loaded properly
python download_wikipedia.py --max_passages 100000
python benchmark.py --model bge-base --max_samples 100
```

### Issue 2: Question ID Mismatch

**Symptom:**
```
Skipped (no ground truth): 100
Evaluated: 0
```

**Cause:** Question IDs từ retrieval không khớp với ground truth

**Debug:**
```python
# In benchmark.py evaluate() function
print(f"Sample retrieval question ID: {list(retrieval_results.keys())[0]}")
print(f"Sample ground truth question ID: {list(ground_truth.keys())[0]}")
```

**Fix:** Ensure question IDs are consistent

### Issue 3: Empty Ground Truth

**Symptom:**
```
Number of ground truth entries: 0
```

**Cause:** get_ground_truth_labels() không tìm thấy supporting facts

**Fix:** Check data_loader.py:get_ground_truth_labels()

## Recommended: Test with Distractor Mode First

Fullwiki mode phức tạp hơn. Test với distractor mode trước:

```bash
python benchmark.py \
    --model bge-base \
    --dataset_config distractor \
    --batch_size 128 \
    --num_workers 4 \
    --max_samples 100
```

**Why distractor is easier:**
- ✓ Corpus nhỏ hơn (chỉ context paragraphs)
- ✓ Supporting facts luôn có trong corpus
- ✓ Faster để test

## Expected Output (Working)

```
STEP 5: Evaluating Results
Building ground truth labels...
Number of questions: 100
Number of ground truth entries: 100

Sample ground truth for question 5a8b57f25542995d1e6f1371:
  Relevant docs: 2
  Retrieved docs: 20
  First 5 retrieved: ['para_123', 'para_456', ...]

Evaluation Statistics:
  Total questions: 100
  Evaluated: 100  ← Should match total
  Skipped (no ground truth): 0
  Skipped (no relevant docs): 0

============================================================
Evaluation Results: BAAI/bge-base-en-v1.5
============================================================

Pass@k (Success Rate):
  pass@1         : 0.6543
  pass@3         : 0.7821
  ...
```

## Quick Fix Commands

### Test Small with Distractor
```bash
python benchmark.py \
    --model bge-base \
    --dataset_config distractor \
    --max_samples 100 \
    --batch_size 128 \
    --num_workers 4
```

### Debug Full Pipeline
```bash
# 1. Debug data loading
python debug_evaluation.py

# 2. Run with debug logging
python benchmark.py \
    --model bge-base \
    --max_samples 10 \
    --batch_size 32 \
    --num_workers 0
```

### Force Rebuild Ground Truth
```bash
# Clear cache and rebuild
rm -rf cache/index/
python benchmark.py \
    --model bge-base \
    --dataset_config distractor \
    --max_samples 100
```

## Summary

1. ✅ Code updated with debug logging
2. ✅ Run `debug_evaluation.py` to check data
3. ✅ Run benchmark with `--dataset_config distractor` first
4. ✅ Check debug output for "Evaluated: X" - should be > 0
5. ✅ If still empty, share debug output

The issue is likely corpus vs ground truth mismatch in fullwiki mode.
