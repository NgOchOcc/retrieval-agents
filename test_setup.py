"""Test script to verify setup is working correctly."""

import sys
import os

print("Testing HotpotQA Retrieval Benchmark Setup")
print("=" * 60)

# Test 1: Check Python version
print("\n1. Checking Python version...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 8:
    print(f"   ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ✗ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
    sys.exit(1)

# Test 2: Check required packages
print("\n2. Checking required packages...")
required_packages = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "datasets": "Datasets",
    "numpy": "NumPy",
    "faiss": "FAISS",
    "tqdm": "tqdm",
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} (missing)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   Missing packages: {', '.join(missing_packages)}")
    print("   Install with: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check CUDA availability
print("\n3. Checking CUDA availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print("   ⚠ CUDA not available (will use CPU)")
except Exception as e:
    print(f"   ⚠ Could not check CUDA: {e}")

# Test 4: Check cache directory
print("\n4. Checking cache directory...")
cache_dir = "./cache"
if os.path.exists(cache_dir):
    print(f"   ✓ Cache directory exists: {cache_dir}")

    # Check for Wikipedia corpus
    wiki_file = os.path.join(cache_dir, "wikipedia_paragraphs.json")
    if os.path.exists(wiki_file):
        import json
        with open(wiki_file, "r") as f:
            data = json.load(f)
        print(f"   ✓ Wikipedia corpus cached ({len(data):,} paragraphs)")
    else:
        print("   ⚠ Wikipedia corpus not cached yet")
        print("      Run: python download_wikipedia.py --max_passages 100000")
else:
    print(f"   ⚠ Cache directory will be created on first run")

# Test 5: Test model loading (optional, can be slow)
print("\n5. Testing model loading (this may take a moment)...")
try:
    from transformers import AutoTokenizer, AutoModel
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Small model for testing
    print(f"   Loading test model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("   ✓ Model loading works")
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    sys.exit(1)

# Test 6: Test imports of custom modules
print("\n6. Testing custom module imports...")
custom_modules = [
    "config",
    "data_loader",
    "retrieval_model",
    "indexer",
    "metrics",
]

for module in custom_modules:
    try:
        __import__(module)
        print(f"   ✓ {module}.py")
    except Exception as e:
        print(f"   ✗ {module}.py: {e}")
        sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("Setup Test Complete!")
print("=" * 60)

if os.path.exists(os.path.join(cache_dir, "wikipedia_paragraphs.json")):
    print("\n✓ Your setup is ready!")
    print("\nNext step: Run benchmark")
    print("  python benchmark.py --model bge-base --max_samples 100")
else:
    print("\n⚠ Setup is mostly ready, but you need Wikipedia corpus")
    print("\nNext step: Download Wikipedia corpus")
    print("  python download_wikipedia.py --max_passages 100000")
    print("\nThen run benchmark:")
    print("  python benchmark.py --model bge-base --max_samples 100")

print("\n" + "=" * 60)
