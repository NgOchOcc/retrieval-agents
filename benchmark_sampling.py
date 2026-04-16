#!/usr/bin/env python3
"""
Benchmark script comparing baseline retrieval vs random sampling strategy.

Strategy:
1. Retrieve top-n documents
2. Randomly sample each with probability p
3. Select top-k from sampled documents
4. Compare against baseline top-k retrieval
"""

import os
import json
import random
import numpy as np
import torch
import argparse
from datetime import datetime
from pathlib import Path

from config import set_seed
from data import HotpotQALoader
from models import get_encoder
from retrieval import SamplingRetriever
from evaluation import RetrievalEvaluator
from utils import print_gpu_info, get_optimal_batch_sizes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval with random sampling strategy"
    )

    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )

    # Dataset settings
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Dataset split"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="distractor",
        choices=["fullwiki", "distractor"],
        help="HotpotQA dataset type"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Limit number of samples"
    )

    # Sampling parameters
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of top documents to retrieve initially"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Sampling probability (0 < p < 1)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Final number of documents after sampling"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values to evaluate"
    )

    # Other settings
    parser.add_argument(
        "--passage_batch_size",
        type=int,
        default=128,
        help="Batch size for passages"
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=32,
        help="Batch size for queries"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sampling",
        help="Output directory"
    )

    return parser.parse_args()


def save_results(results: dict, args, output_path: Path):
    """Save comparison results."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "dataset_type": args.dataset_type,
        "dataset_split": args.split,
        "num_samples": args.max_samples,
        "sampling_params": {
            "n": args.n,
            "p": args.p,
            "k": args.k,
        },
        "results": results,
        "config": {
            "k_values": args.k_values,
            "seed": args.seed,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def format_comparison_table(baseline_results: dict, sampled_results: dict, stats: dict):
    """Format comparison results as table."""
    lines = ["=" * 80]
    lines.append("BASELINE vs SAMPLING COMPARISON")
    lines.append("=" * 80)
    lines.append(f"\nSampling Strategy:")
    lines.append(f"  - Top-n retrieved: {stats['n']}")
    lines.append(f"  - Sampling probability (p): {stats['p']:.2f}")
    lines.append(f"  - Final top-k: {stats['k']}")
    lines.append(f"  - Expected samples per query: {stats['expected_sample_size']:.1f}")
    lines.append(f"\nActual Sampling Statistics:")
    lines.append(f"  - Avg sampled docs: {stats['avg_sampled_docs']:.2f}")
    lines.append(f"  - Min sampled docs: {stats['min_sampled_docs']}")
    lines.append(f"  - Max sampled docs: {stats['max_sampled_docs']}")
    lines.append(f"  - Queries with < k docs: {stats['queries_with_less_than_k']}/{stats['num_queries']}")

    lines.append("\n" + "=" * 80)
    lines.append(f"{'Metric':<15} {'Baseline':<15} {'Sampling':<15} {'Difference':<15}")
    lines.append("=" * 80)

    # Extract k_values from metrics
    k_values = sorted(set(int(k.split('@')[1]) for k in baseline_results.keys()))

    # pass@k metrics
    lines.append("\nSTRICT (pass@k) - Both docs in top-k:")
    lines.append("-" * 80)
    for k in k_values:
        metric = f'pass@{k}'
        if metric in baseline_results and metric in sampled_results:
            baseline_val = baseline_results[metric]
            sampled_val = sampled_results[metric]
            diff = sampled_val - baseline_val
            diff_str = f"{diff:+.4f}"
            lines.append(f"{metric:<15} {baseline_val:<15.4f} {sampled_val:<15.4f} {diff_str:<15}")

    # hit@k metrics
    lines.append("\nRELAXED (hit@k) - At least one doc in top-k:")
    lines.append("-" * 80)
    for k in k_values:
        metric = f'hit@{k}'
        if metric in baseline_results and metric in sampled_results:
            baseline_val = baseline_results[metric]
            sampled_val = sampled_results[metric]
            diff = sampled_val - baseline_val
            diff_str = f"{diff:+.4f}"
            lines.append(f"{metric:<15} {baseline_val:<15.4f} {sampled_val:<15.4f} {diff_str:<15}")

    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Main comparison pipeline."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    print("=" * 80)
    print("SAMPLING RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_type} ({args.split})")
    print(f"Samples: {args.max_samples}")
    print(f"Strategy: Retrieve top-{args.n}, sample with p={args.p}, select top-{args.k}")
    print("=" * 80)

    # Print GPU info
    print_gpu_info()

    # Auto-adjust batch sizes based on GPU
    model_size = "large" if "large" in args.model_name.lower() else "base"
    suggested_passage_batch, suggested_query_batch = get_optimal_batch_sizes(model_size)

    if args.passage_batch_size == 128:  # default value
        args.passage_batch_size = suggested_passage_batch
    if args.query_batch_size == 32:  # default value
        args.query_batch_size = suggested_query_batch

    print(f"\nBatch sizes:")
    print(f"  Passage: {args.passage_batch_size}")
    print(f"  Query: {args.query_batch_size}")
    print("")

    # Validate parameters
    if not (0 < args.p < 1):
        raise ValueError(f"p must be between 0 and 1, got {args.p}")
    if args.k > args.n:
        raise ValueError(f"k ({args.k}) cannot be greater than n ({args.n})")

    # Load data
    print("\n[1/5] Loading HotpotQA dataset...")
    loader = HotpotQALoader(
        split=args.split,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        seed=args.seed
    )

    examples = loader.load_examples()
    passages = loader.build_corpus()

    # Load encoder
    print("\n[2/5] Loading encoder model...")
    encoder = get_encoder(args.model_name, device=args.device)

    # Build index with sampling retriever
    print("\n[3/5] Building FAISS index...")
    retriever = SamplingRetriever(encoder, seed=args.seed)
    retriever.build_index(
        passages=passages,
        batch_size=args.passage_batch_size,
        save_embeddings=False
    )

    # Retrieve with both strategies
    print("\n[4/5] Retrieving with baseline and sampling strategies...")
    queries = [ex.question for ex in examples]
    gold_titles_list = [ex.gold_titles for ex in examples]

    baseline_retrieved, sampled_retrieved, stats = retriever.retrieve_batch_with_sampling(
        queries=queries,
        n=args.n,
        p=args.p,
        k=args.k,
        batch_size=args.query_batch_size
    )

    # Evaluate both strategies
    print("\n[5/5] Evaluating results...")
    evaluator = RetrievalEvaluator(k_values=args.k_values)

    print("\n  Evaluating BASELINE retrieval...")
    baseline_results = evaluator.evaluate(gold_titles_list, baseline_retrieved)

    print("  Evaluating SAMPLING retrieval...")
    sampled_results = evaluator.evaluate(gold_titles_list, sampled_retrieved)

    # Print comparison
    print("\n" + format_comparison_table(baseline_results, sampled_results, stats))

    # Save results
    output_filename = f"sampling_comparison_{args.model_name.replace('/', '_')}_n{args.n}_p{args.p}_k{args.k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(args.output_dir) / output_filename

    save_results(
        {
            'baseline': baseline_results,
            'sampling': sampled_results,
            'statistics': stats,
        },
        args,
        output_path
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
