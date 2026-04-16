"""
Configuration management for retrieval benchmark.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import argparse


@dataclass
class Config:
    """Configuration for retrieval benchmark."""

    # Model settings
    model_name: str = "BAAI/bge-base-en-v1.5"
    device: Optional[str] = None

    # Dataset settings
    dataset_split: str = "validation"
    dataset_type: str = "fullwiki"  # 'fullwiki' or 'distractor'
    max_samples: Optional[int] = None  # Subsample for debugging

    # Retrieval settings
    max_k: int = 20  # Maximum k for retrieval
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Batch sizes
    passage_batch_size: int = 128
    query_batch_size: int = 32

    # Reproducibility
    seed: int = 42

    # Output settings
    save_results: bool = True
    output_dir: str = "results"
    verbose: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create config from argparse args."""
        return cls(
            model_name=args.model_name,
            device=args.device,
            dataset_split=args.split,
            dataset_type=args.dataset_type,
            max_samples=args.max_samples,
            max_k=args.max_k,
            k_values=args.k_values,
            passage_batch_size=args.passage_batch_size,
            query_batch_size=args.query_batch_size,
            seed=args.seed,
            save_results=args.save_results,
            output_dir=args.output_dir,
            verbose=args.verbose
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark dense retrieval models on HotpotQA"
    )

    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model name (BGE or E5)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu, default: auto)"
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
        default="fullwiki",
        choices=["fullwiki", "distractor"],
        help="HotpotQA dataset type"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)"
    )

    # Retrieval settings
    parser.add_argument(
        "--max_k",
        type=int,
        default=20,
        help="Maximum k for retrieval"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values to evaluate"
    )

    # Batch sizes
    parser.add_argument(
        "--passage_batch_size",
        type=int,
        default=128,
        help="Batch size for encoding passages"
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=32,
        help="Batch size for encoding queries"
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Output settings
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output"
    )

    args = parser.parse_args()
    args.save_results = not args.no_save

    return args
