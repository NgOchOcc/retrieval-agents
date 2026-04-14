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
