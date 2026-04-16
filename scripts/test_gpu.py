#!/usr/bin/env python3
"""
Test GPU usage and benchmark encoding speed.
"""

import torch
import time
from models import get_encoder


def test_gpu():
    """Test GPU availability and model loading."""
    print("=" * 60)
    print("GPU AVAILABILITY TEST")
    print("=" * 60)

    # Check CUDA
    print(f"\nCUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

        # Memory info
        props = torch.cuda.get_device_properties(0)
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    else:
        print("⚠️  CUDA not available. Will use CPU.")

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    # Load model
    model_name = "BAAI/bge-base-en-v1.5"
    print(f"\nLoading {model_name}...")

    encoder = get_encoder(model_name)

    # Check model device
    print("\n" + "=" * 60)
    print("ENCODING SPEED TEST")
    print("=" * 60)

    # Test encoding speed
    test_texts = ["This is a test sentence."] * 100

    print(f"\nEncoding {len(test_texts)} texts...")

    start_time = time.time()
    embeddings = encoder.encode(test_texts, batch_size=32, show_progress=True)
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = len(test_texts) / elapsed

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Throughput: {throughput:.2f} texts/second")
    print(f"  Embedding shape: {embeddings.shape}")

    if torch.cuda.is_available():
        print(f"\nGPU Memory After Encoding:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")

    print("\n" + "=" * 60)

    # Recommendations
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_memory_gb >= 16:
            recommended_batch = 256
        elif total_memory_gb >= 8:
            recommended_batch = 128
        else:
            recommended_batch = 64

        print("RECOMMENDATIONS")
        print("=" * 60)
        print(f"\nBased on {total_memory_gb:.1f} GB GPU memory:")
        print(f"  Recommended passage batch size: {recommended_batch}")
        print(f"  Recommended query batch size: {recommended_batch // 2}")
    else:
        print("RECOMMENDATIONS")
        print("=" * 60)
        print("\n⚠️  No GPU detected. Performance will be slow.")
        print("To use GPU:")
        print("  1. Install PyTorch with CUDA support")
        print("  2. Install faiss-gpu instead of faiss-cpu")


if __name__ == "__main__":
    test_gpu()
