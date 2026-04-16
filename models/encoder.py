"""
Embedding model wrapper with proper instruction formatting for BGE and E5 models.
"""

from typing import List, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """Base class for embedding models."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and explicitly move to device
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)

        # Verify device
        actual_device = next(self.model.parameters()).device
        print(f"Loaded {model_name} on {actual_device}")

        if torch.cuda.is_available():
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    @abstractmethod
    def format_query(self, query: str) -> str:
        """Format query with model-specific instruction."""
        pass

    @abstractmethod
    def format_passage(self, passage: str) -> str:
        """Format passage with model-specific instruction."""
        pass

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar

        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        # Ensure model is on correct device
        self.model = self.model.to(self.device)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device  # Explicitly pass device
        )
        return embeddings


class BGEEncoder(BaseEncoder):
    """BGE model encoder with query instruction."""

    def format_query(self, query: str) -> str:
        """Add BGE query instruction."""
        return f"Represent this sentence for searching relevant passages: {query}"

    def format_passage(self, passage: str) -> str:
        """BGE passages don't need instruction prefix."""
        return passage


class E5Encoder(BaseEncoder):
    """E5 model encoder with query/passage instructions."""

    def format_query(self, query: str) -> str:
        """Add E5 query instruction."""
        return f"query: {query}"

    def format_passage(self, passage: str) -> str:
        """Add E5 passage instruction."""
        return f"passage: {passage}"


def get_encoder(model_name: str, device: Optional[str] = None) -> BaseEncoder:
    """
    Factory function to get appropriate encoder based on model name.

    Args:
        model_name: HuggingFace model identifier
        device: Device to run model on

    Returns:
        Appropriate encoder instance
    """
    if 'bge' in model_name.lower():
        return BGEEncoder(model_name, device)
    elif 'e5' in model_name.lower():
        return E5Encoder(model_name, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use BGE or E5 models.")


# Supported models
SUPPORTED_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
]
