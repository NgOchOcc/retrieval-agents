#!/usr/bin/env python3
"""
Main script for benchmarking dense retrieval models on HotpotQA.
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from config import Config, parse_args
from data import HotpotQALoader
from models import get_encoder, SUPPORTED_MODELS
from retrieval import FAISSRetriever
from evaluation import RetrievalEvaluator
from utils import print_gpu_info, get_optimal_batch_sizes


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(results: dict, config: Config, output_path: Path):
    """Save evaluation results to JSON file."""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": config.model_name,
        "dataset_type": config.dataset_type,
        "dataset_split": config.dataset_split,
        "num_samples": config.max_samples,
        "metrics": results,
        "config": {
            "k_values": config.k_values,
            "seed": config.seed,
            "passage_batch_size": config.passage_batch_size,
            "query_batch_size": config.query_batch_size,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main evaluation pipeline."""
    # Parse arguments and create config
    args = parse_args()
    config = Config.from_args(args)

    # Set seed for reproducibility
    set_seed(config.seed)

    print("=" * 60)
    print("HOTPOTQA RETRIEVAL BENCHMARK")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_type} ({config.dataset_split})")
    if config.max_samples:
        print(f"Max samples: {config.max_samples}")
    print(f"Seed: {config.seed}")
    print("=" * 60)

    # Print GPU info
    print_gpu_info()

    # Auto-adjust batch sizes based on GPU and model size
    model_size = "large" if "large" in config.model_name.lower() else "base"
    suggested_passage_batch, suggested_query_batch = get_optimal_batch_sizes(model_size)

    # Use suggested batch sizes if not explicitly set by user
    if config.passage_batch_size == 256:  # default value
        config.passage_batch_size = suggested_passage_batch
    if config.query_batch_size == 64:  # default value
        config.query_batch_size = suggested_query_batch

    print(f"\nBatch sizes:")
    print(f"  Passage: {config.passage_batch_size}")
    print(f"  Query: {config.query_batch_size}")

    # Validate model
    if config.model_name not in SUPPORTED_MODELS:
        print(f"\nWarning: {config.model_name} not in supported models list.")
        print(f"Supported models: {SUPPORTED_MODELS}")
        print("Attempting to load anyway...\n")

    # Step 1: Load data
    print("\n[1/5] Loading HotpotQA dataset...")
    loader = HotpotQALoader(
        split=config.dataset_split,
        dataset_type=config.dataset_type,
        max_samples=config.max_samples,
        seed=config.seed
    )

    examples = loader.load_examples()
    passages = loader.build_corpus()

    print(f"  → Loaded {len(examples)} examples")
    print(f"  → Built corpus with {len(passages)} passages")

    # Step 2: Initialize encoder
    print("\n[2/5] Loading encoder model...")
    encoder = get_encoder(config.model_name, device=config.device)

    # Step 3: Build retrieval index
    print("\n[3/5] Building FAISS index...")
    retriever = FAISSRetriever(encoder)
    retriever.build_index(
        passages=passages,
        batch_size=config.passage_batch_size,
        save_embeddings=False  # Don't store embeddings to save memory
    )

    # Step 4: Retrieve for all queries
    print("\n[4/5] Retrieving passages for queries...")
    queries = [ex.question for ex in examples]
    gold_titles_list = [ex.gold_titles for ex in examples]

    retrieved_titles_list = retriever.retrieve_batch(
        queries=queries,
        k=config.max_k,
        batch_size=config.query_batch_size
    )

    # Step 5: Evaluate
    print("\n[5/5] Evaluating retrieval results...")
    evaluator = RetrievalEvaluator(k_values=config.k_values)
    results = evaluator.evaluate(gold_titles_list, retrieved_titles_list)

    # Print results
    print("\n" + evaluator.format_results(results))

    # Save results
    if config.save_results:
        model_name_safe = config.model_name.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{model_name_safe}_{config.dataset_type}_{timestamp}.json"
        output_path = Path(config.output_dir) / output_filename

        save_results(results, config, output_path)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
