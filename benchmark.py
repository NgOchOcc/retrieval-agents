"""Main benchmark script for evaluating retrieval models on HotpotQA."""

import os
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime

import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

# Set multiprocessing start method for CUDA compatibility
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

from config import BenchmarkConfig, SUPPORTED_MODELS
from data_loader import HotpotQADataLoader, Paragraph
from retrieval_model_optimized import DenseRetrieverOptimized
from indexer import FAISSIndexer, SimpleRetriever
from metrics import BenchmarkEvaluator
from sampling_strategies import RetrievalSampler, SamplingConfig


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

        # Initialize optimized retrieval model
        self.retriever = DenseRetrieverOptimized(
            model_name=self.config.model_name,
            device=self.config.device,
            normalize_embeddings=self.config.normalize_embeddings,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        # Auto-detect optimal batch size if requested
        if self.config.auto_batch_size and self.config.device == "cuda":
            optimal_bs = self.retriever.auto_batch_size()
            self.config.batch_size = optimal_bs
            self.retriever.batch_size = optimal_bs

        # Initialize evaluator
        self.evaluator = BenchmarkEvaluator(k_values=self.config.k_values)

        # Initialize sampler
        sampling_config = SamplingConfig(
            strategy=self.config.sampling_strategy,
            expansion_factor=self.config.expansion_factor,
            random_ratio=self.config.random_ratio,
            seed=self.config.sampling_seed,
        )
        self.sampler = RetrievalSampler(sampling_config)

        print("Setup complete!")
        if self.config.sampling_strategy != "top_k":
            print(f"Using sampling strategy: {self.config.sampling_strategy}")
            print(f"  Expansion factor: {self.config.expansion_factor}")
            print(f"  Random ratio: {self.config.random_ratio}")

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
            print("WARNING: Fullwiki mode requires full Wikipedia corpus with exact title/sentence matching")
            print("For evaluation, using context paragraphs from questions instead")
            print("(This ensures supporting facts are in the corpus)")

            # Build corpus from question contexts (ensures supporting facts are present)
            paragraphs, _ = self.data_loader.build_corpus_from_examples(examples)

            # Optionally: Could add more Wikipedia paragraphs as distractors here
            # wiki_paragraphs = self.data_loader.load_wikipedia_corpus()
            # paragraphs.extend(wiki_paragraphs[:10000])  # Add 10K distractors
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
                # Determine if GPU should be used for index
                use_gpu = self.config.use_gpu_index and "cuda" in self.config.device

                self.indexer = FAISSIndexer(
                    embedding_dim=embedding_dim,
                    index_type=self.config.faiss_index_type,
                    use_gpu=use_gpu,
                    normalize=self.config.normalize_embeddings,
                    gpu_id=self.config.gpu_id,
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
            # Determine if GPU should be used for index
            use_gpu = self.config.use_gpu_index and "cuda" in self.config.device

            self.indexer = FAISSIndexer(
                embedding_dim=embedding_dim,
                index_type=self.config.faiss_index_type,
                use_gpu=use_gpu,
                normalize=self.config.normalize_embeddings,
                gpu_id=self.config.gpu_id,
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

        # Retrieve documents (with expansion if using sampling)
        print("Searching index...")
        max_k = max(self.config.k_values)

        # If using sampling strategy, retrieve more documents
        if self.config.sampling_strategy != "top_k":
            retrieval_k = max_k * self.config.expansion_factor
            print(f"Retrieving top-{retrieval_k} (expansion_factor={self.config.expansion_factor})")
        else:
            retrieval_k = max_k

        # Use CPU for search to avoid GPU memory issues with large corpus
        # (GPU is still used for encoding, which is the main bottleneck)
        search_batch_size = 256
        use_cpu_for_search = True  # Always use CPU for search to avoid memory issues
        results = self.indexer.search(
            query_embeddings,
            k=retrieval_k,
            search_batch_size=search_batch_size,
            use_cpu_for_search=use_cpu_for_search
        )

        # Apply sampling strategy
        if self.config.sampling_strategy != "top_k":
            print(f"Applying sampling strategy: {self.config.sampling_strategy}")
            print(f"  Random ratio: {self.config.random_ratio}")

            # Extract doc_ids and scores
            batch_doc_ids = [result.doc_ids for result in results]
            batch_scores = [result.scores for result in results]

            # Apply sampling to get final top-k for each query
            sampled_doc_ids, sampled_scores = self.sampler.apply_to_batch(
                batch_doc_ids, batch_scores, max_k
            )

            # Convert to dict
            retrieval_results = {}
            for qid, doc_ids in zip(question_ids, sampled_doc_ids):
                retrieval_results[qid] = doc_ids
        else:
            # Standard top-k (no sampling)
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
        print("Building ground truth labels...")
        ground_truth = self.data_loader.get_ground_truth_labels(examples, paragraphs)

        # Debug info
        print(f"Number of questions: {len(retrieval_results)}")
        print(f"Number of ground truth entries: {len(ground_truth)}")

        # Sample check
        if ground_truth:
            sample_qid = list(ground_truth.keys())[0]
            print(f"Sample ground truth for question {sample_qid}:")
            print(f"  Relevant docs: {len(ground_truth[sample_qid])}")
            if sample_qid in retrieval_results:
                print(f"  Retrieved docs: {len(retrieval_results[sample_qid])}")
                print(f"  First 5 retrieved: {retrieval_results[sample_qid][:5]}")

        # Evaluate
        print("\nCalculating metrics...")
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
            "sampling_strategy": self.config.sampling_strategy,
            "expansion_factor": self.config.expansion_factor if self.config.sampling_strategy != "top_k" else None,
            "random_ratio": self.config.random_ratio if self.config.sampling_strategy != "top_k" else None,
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
    parser.add_argument("--index_type", type=str, default="Flat", help="FAISS index type (Flat, IVF, IVFPQ)")
    parser.add_argument("--no_gpu_index", action="store_true", help="Disable GPU for FAISS index (use CPU only)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID for indexing")
    parser.add_argument("--auto_batch_size", action="store_true", help="Auto-detect optimal batch size for GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading (0 for main process only)")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pinned memory (use if having memory issues)")

    # Sampling strategy arguments
    parser.add_argument("--sampling_strategy", type=str, default="top_k",
                       choices=["top_k", "random_sample", "diverse_sample"],
                       help="Sampling strategy for retrieval results")
    parser.add_argument("--expansion_factor", type=int, default=4,
                       help="Expansion factor for sampling (retrieve top expansion_factor*k)")
    parser.add_argument("--random_ratio", type=float, default=0.3,
                       help="Random sampling ratio (0 < x < 1)")
    parser.add_argument("--sampling_seed", type=int, default=42,
                       help="Random seed for sampling")

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
        faiss_index_type=args.index_type,
        use_gpu_index=not args.no_gpu_index,
        gpu_id=args.gpu_id,
        auto_batch_size=args.auto_batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        sampling_strategy=args.sampling_strategy,
        expansion_factor=args.expansion_factor,
        random_ratio=args.random_ratio,
        sampling_seed=args.sampling_seed,
        k_values=[1, 3, 5, 10, 20],
    )

    # Run benchmark
    benchmark = RetrievalBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()
