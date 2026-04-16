"""
Utility functions for GPU detection and batch size optimization.
"""

import torch


def get_optimal_batch_sizes(model_size: str = "base") -> tuple:
    """
    Get optimal batch sizes based on available GPU memory.

    Args:
        model_size: 'base' or 'large'

    Returns:
        Tuple of (passage_batch_size, query_batch_size)
    """
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected. Using CPU with small batch sizes.")
        return (32, 16)  # Small batches for CPU

    # Get GPU memory in GB
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
    print(f"Model size: {model_size}")

    # Batch size recommendations based on GPU memory and model size
    if model_size == "large":
        if gpu_memory_gb >= 40:  # A100, A6000
            passage_batch = 512
            query_batch = 128
        elif gpu_memory_gb >= 24:  # RTX 3090, 4090
            passage_batch = 256
            query_batch = 64
        elif gpu_memory_gb >= 16:  # RTX 4080
            passage_batch = 128
            query_batch = 32
        elif gpu_memory_gb >= 12:  # RTX 3080, 4070
            passage_batch = 96
            query_batch = 24
        elif gpu_memory_gb >= 8:  # RTX 3070, 4060
            passage_batch = 64
            query_batch = 16
        else:
            passage_batch = 32
            query_batch = 8
    else:  # base model
        if gpu_memory_gb >= 40:
            passage_batch = 1024
            query_batch = 256
        elif gpu_memory_gb >= 24:
            passage_batch = 512
            query_batch = 128
        elif gpu_memory_gb >= 16:
            passage_batch = 384
            query_batch = 96
        elif gpu_memory_gb >= 12:
            passage_batch = 256
            query_batch = 64
        elif gpu_memory_gb >= 8:
            passage_batch = 128
            query_batch = 32
        else:
            passage_batch = 64
            query_batch = 16

    print(f"Recommended batch sizes:")
    print(f"  Passage: {passage_batch}")
    print(f"  Query: {query_batch}")

    return (passage_batch, query_batch)


def print_gpu_info():
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        print("=" * 60)
        print("⚠️  CUDA NOT AVAILABLE")
        print("=" * 60)
        print("\nTo enable GPU acceleration:")
        print("1. Install PyTorch with CUDA:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("2. Install faiss-gpu:")
        print("   pip uninstall faiss-cpu && pip install faiss-gpu")
        print("\nRunning on CPU will be 10-50x slower!")
        print("=" * 60)
        return

    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    print(f"CUDA Available: ✅")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print("=" * 60)


def check_gpu_memory():
    """Check current GPU memory usage."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU Memory Usage:")
    print(f"  Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"  Reserved: {reserved:.2f} GB")
