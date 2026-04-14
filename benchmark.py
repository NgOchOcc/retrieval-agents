"""Main benchmark script for evaluating retrieval models on HotpotQA."""

import os
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from config import BenchmarkConfig, SUPPORTED_MODELS
from data_loader import HotpotQADataLoader, Paragraph
from retrieval_model import DenseRetriever
from indexer import FAISSIndexer, SimpleRetriever
from metrics import BenchmarkEvaluator


class RetrievalBenchmark:
    """Main benchmark pipeline."""

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.data_loader = None
        self.retriever = None
        self.indexer = None
        self.evaluator = None

        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)

    def setup(self):
        """Setup all components."""
        print("Setting up benchmark components...")

        # Initialize data loader
        self.data_loader = HotpotQADataLoader(
            split=self.config.dataset_split,
            config=self.config.dataset_config,
            cache_dir=self.config.cache_dir,
        )

        # Initialize retrieval model
        self.retriever = DenseRetriever(
            model_name=self.config.model_name,
            device=self.config.device,
            normalize_embeddings=self.config.normalize_embeddings,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
        )

        # Initialize evaluator
        self.evaluator = BenchmarkEvaluator(k_values=self.config.k_values)

        print("Setup complete!")

    def load_data(self):
        """Load and prepare dataset."""
        print("\n" + "=" * 60)
        print("STEP 1: Loading Dataset")
        print("=" * 60)

        # Load HotpotQA examples
        examples = self.data_loader.load_dataset(max_samples=self.config.max_samples)

        return examples

    def build_corpus(self, examples):
        """Build or load corpus."""
        print("\n" + "=" * 60)
        print("STEP 2: Building Corpus")
        print("=" * 60)

        if self.config.dataset_config == "fullwiki":
            # Load full Wikipedia corpus
            paragraphs = self.data_loader.load_wikipedia_corpus()
        else:
            # Build corpus from examples (distractor mode)
            paragraphs, _ = self.data_loader.build_corpus_from_examples(examples)

        return paragraphs

    def build_index(self, paragraphs: List[Paragraph]):
        """Build or load index."""
        print("\n" + "=" * 60)
        print("STEP 3: Building Index")
        print("=" * 60)

        index_dir = os.path.join(self.config.cache_dir, "index", self.config.model_name.replace("/", "_"))

        # Check if index exists
        if os.path.exists(os.path.join(index_dir, "index.faiss")) and not self.config.save_embeddings:
            print(f"Loading existing index from {index_dir}")
            embedding_dim = self.retriever.get_embedding_dim()

            if self.config.use_faiss:
                self.indexer = FAISSIndexer(
                    embedding_dim=embedding_dim,
                    index_type=self.config.faiss_index_type,
                    use_gpu="cuda" in self.config.device,
                    normalize=self.config.normalize_embeddings,
                )
                self.indexer.load(index_dir)
            else:
                # Load embeddings for simple retriever
                embeddings_path = os.path.join(index_dir, "embeddings.npy")
                embeddings = np.load(embeddings_path)
                doc_ids = [p.para_id for p in paragraphs]
                self.indexer = SimpleRetriever(normalize=self.config.normalize_embeddings)
                self.indexer.add_documents(embeddings, doc_ids)

            return

        # Encode corpus
        print("Encoding corpus...")
        corpus_texts = [p.text for p in paragraphs]
        corpus_embeddings = self.retriever.encode_corpus(corpus_texts, show_progress=True)

        # Create index
        embedding_dim = corpus_embeddings.shape[1]
        doc_ids = [p.para_id for p in paragraphs]

        if self.config.use_faiss:
            self.indexer = FAISSIndexer(
                embedding_dim=embedding_dim,
                index_type=self.config.faiss_index_type,
                use_gpu="cuda" in self.config.device,
                normalize=self.config.normalize_embeddings,
            )
            self.indexer.add_documents(corpus_embeddings, doc_ids)

            # Save index
            if self.config.save_embeddings:
                self.indexer.save(index_dir)
        else:
            self.indexer = SimpleRetriever(normalize=self.config.normalize_embeddings)
            self.indexer.add_documents(corpus_embeddings, doc_ids)

            # Save embeddings
            if self.config.save_embeddings:
                os.makedirs(index_dir, exist_ok=True)
                np.save(os.path.join(index_dir, "embeddings.npy"), corpus_embeddings)

        print("Index built successfully!")

    def retrieve(self, examples):
        """Perform retrieval for all queries."""
        print("\n" + "=" * 60)
        print("STEP 4: Retrieving Documents")
        print("=" * 60)

        # Extract queries
        queries = [ex.question for ex in examples]
        question_ids = [ex.question_id for ex in examples]

        # Encode queries
        print("Encoding queries...")
        query_embeddings = self.retriever.encode_queries(queries, show_progress=True)

        # Retrieve documents
        print("Searching index...")
        max_k = max(self.config.k_values)
        results = self.indexer.search(query_embeddings, k=max_k)

        # Convert to dict
        retrieval_results = {}
        for qid, result in zip(question_ids, results):
            retrieval_results[qid] = result.doc_ids

        return retrieval_results

    def evaluate(self, examples, paragraphs, retrieval_results):
        """Evaluate retrieval results."""
        print("\n" + "=" * 60)
        print("STEP 5: Evaluating Results")
        print("=" * 60)

        # Build ground truth
        ground_truth = self.data_loader.get_ground_truth_labels(examples, paragraphs)

        # Evaluate
        metrics = self.evaluator.evaluate(retrieval_results, ground_truth)

        return metrics

    def run(self):
        """Run complete benchmark pipeline."""
        print("\n" + "=" * 60)
        print("HotpotQA Retrieval Benchmark")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Dataset: {self.config.dataset_name}/{self.config.dataset_config}")
        print(f"Split: {self.config.dataset_split}")
        print(f"K values: {self.config.k_values}")
        print(f"Device: {self.config.device}")
        print("=" * 60)

        # Setup
        self.setup()

        # Load data
        examples = self.load_data()

        # Build corpus
        paragraphs = self.build_corpus(examples)

        # Build index
        self.build_index(paragraphs)

        # Retrieve
        retrieval_results = self.retrieve(examples)

        # Evaluate
        metrics = self.evaluate(examples, paragraphs, retrieval_results)

        # Print results
        self.evaluator.print_results(metrics, model_name=self.config.model_name)

        # Save results
        self.save_results(metrics)

        return metrics

    def save_results(self, metrics: Dict[str, float]):
        """Save evaluation results."""
        results_dir = os.path.join(self.config.cache_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model_name.replace("/", "_")
        filename = f"{model_name}_{self.config.dataset_config}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)

        results = {
            "model_name": self.config.model_name,
            "dataset_config": self.config.dataset_config,
            "dataset_split": self.config.dataset_split,
            "k_values": self.config.k_values,
            "metrics": metrics,
            "timestamp": timestamp,
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark retrieval models on HotpotQA")
    parser.add_argument(
        "--model",
        type=str,
        default="bge-base",
        help=f"Model name or shorthand ({', '.join(SUPPORTED_MODELS.keys())})",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples (for testing)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split")
    parser.add_argument("--dataset_config", type=str, default="fullwiki", help="Dataset config")
    parser.add_argument("--no_faiss", action="store_true", help="Use simple numpy retriever instead of FAISS")

    args = parser.parse_args()

    # Resolve model name
    model_name = SUPPORTED_MODELS.get(args.model, args.model)

    # Create config
    config = BenchmarkConfig(
        model_name=model_name,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        cache_dir=args.cache_dir,
        dataset_split=args.dataset_split,
        dataset_config=args.dataset_config,
        use_faiss=not args.no_faiss,
        k_values=[1, 3, 5, 10, 20],
    )

    # Run benchmark
    benchmark = RetrievalBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()
