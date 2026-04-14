"""Test script to compare original vs optimized performance."""

import time
import torch
from retrieval_model_optimized import DenseRetrieverOptimized


def test_encoding_speed(model_name="BAAI/bge-base-en-v1.5", num_docs=10000):
    """Test encoding speed with different configurations."""

    print("=" * 80)
    print("Performance Test - Encoding Speed Comparison")
    print("=" * 80)

    # Create dummy documents
    print(f"\nCreating {num_docs:,} dummy documents...")
    dummy_text = "This is a test document for benchmarking retrieval model encoding performance."
    documents = [dummy_text] * num_docs

    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU: Not available (using CPU)")

    print("\n" + "=" * 80)

    # Test configurations
    configs = [
        {"name": "Small Batch (32), No Workers", "batch_size": 32, "num_workers": 0},
        {"name": "Medium Batch (64), 2 Workers", "batch_size": 64, "num_workers": 2},
        {"name": "Large Batch (128), 4 Workers", "batch_size": 128, "num_workers": 4},
        {"name": "XL Batch (256), 8 Workers", "batch_size": 256, "num_workers": 8},
    ]

    results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  Batch size: {config['batch_size']}, Workers: {config['num_workers']}")

        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Initialize model
            retriever = DenseRetrieverOptimized(
                model_name=model_name,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Warm up
            _ = retriever.encode(documents[:100], show_progress=False)

            # Benchmark
            start_time = time.time()
            embeddings = retriever.encode(documents, show_progress=False)
            elapsed = time.time() - start_time

            docs_per_sec = num_docs / elapsed
            speedup = docs_per_sec / results[0]["docs_per_sec"] if results else 1.0

            result = {
                "config": config["name"],
                "batch_size": config["batch_size"],
                "workers": config["num_workers"],
                "time": elapsed,
                "docs_per_sec": docs_per_sec,
                "speedup": speedup,
            }
            results.append(result)

            print(f"  ✓ Time: {elapsed:.2f}s")
            print(f"  ✓ Speed: {docs_per_sec:.1f} docs/sec")
            print(f"  ✓ Speedup: {speedup:.1f}x")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ Out of Memory - Skip")
            else:
                raise e

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Config':<35} {'Time':>10} {'Speed':>15} {'Speedup':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['config']:<35} {r['time']:>9.2f}s {r['docs_per_sec']:>12.1f} d/s {r['speedup']:>9.1f}x")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if results:
        best = max(results, key=lambda x: x["docs_per_sec"])
        print(f"\nBest configuration: {best['config']}")
        print(f"  → Use: --batch_size {best['batch_size']} --num_workers {best['workers']}")
        print(f"  → Speed: {best['docs_per_sec']:.1f} docs/sec")
        print(f"  → Speedup: {best['speedup']:.1f}x over baseline")

        # Estimate time for 1M docs
        time_1m = 1_000_000 / best['docs_per_sec']
        print(f"\nEstimated time for 1M docs: {time_1m/60:.1f} minutes")

    print("\n" + "=" * 80)

    return results


def test_auto_batch_size(model_name="BAAI/bge-base-en-v1.5"):
    """Test auto batch size detection."""

    print("\n" + "=" * 80)
    print("Testing Auto Batch Size Detection")
    print("=" * 80)

    retriever = DenseRetrieverOptimized(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    optimal_bs = retriever.auto_batch_size()

    print(f"\nOptimal batch size: {optimal_bs}")
    print(f"Recommendation: --batch_size {optimal_bs} --num_workers 8")

    return optimal_bs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test encoding performance")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5", help="Model name")
    parser.add_argument("--num_docs", type=int, default=10000, help="Number of documents to test")
    parser.add_argument("--auto_batch_size", action="store_true", help="Test auto batch size")

    args = parser.parse_args()

    if args.auto_batch_size:
        test_auto_batch_size(args.model)
    else:
        test_encoding_speed(args.model, args.num_docs)
