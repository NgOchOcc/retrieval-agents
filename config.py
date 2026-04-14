"""Configuration for HotpotQA retrieval benchmark."""

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkConfig:
    """Configuration for retrieval benchmark."""

    # Model configuration
    model_name: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 32
    max_length: int = 512
    auto_batch_size: bool = False  # Auto-detect optimal batch size
    num_workers: int = 4  # Number of workers for data loading (0 = main process only)
    pin_memory: bool = True  # Use pinned memory for faster GPU transfer

    # Dataset configuration
    dataset_name: str = "hotpot_qa"
    dataset_split: str = "test"
    dataset_config: str = "fullwiki"
    max_samples: int = None  # None for all samples

    # Index configuration
    use_faiss: bool = True
    faiss_index_type: str = "Flat"  # 'Flat', 'IVF', or 'IVFPQ'
    normalize_embeddings: bool = True
    use_gpu_index: bool = True  # Use GPU for FAISS index (if available)
    gpu_id: int = 0  # GPU device ID for indexing

    # Evaluation configuration
    k_values: List[int] = None

    # Sampling configuration
    sampling_strategy: str = "top_k"  # 'top_k', 'random_sample', 'diverse_sample'
    expansion_factor: int = 4  # For sampling: retrieve top (expansion_factor * k) first
    random_ratio: float = 0.3  # For random sampling: ratio of random selection (0 < x < 1)
    sampling_seed: int = 42  # Random seed for sampling

    # Cache configuration
    cache_dir: str = "./cache"
    save_embeddings: bool = True

    # Device configuration
    device: str = "cuda"  # 'cuda' or 'cpu'

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10, 20]


# Supported models
SUPPORTED_MODELS = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
    "e5-large": "intfloat/e5-large-v2",
    "e5-small": "intfloat/e5-small-v2",
    "gte-base": "thenlper/gte-base",
    "gte-large": "thenlper/gte-large",
}
